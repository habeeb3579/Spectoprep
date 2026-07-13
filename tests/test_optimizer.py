"""Tests for the PipelineOptimizer."""

import inspect
import json
import types

import numpy as np
import pytest

from spectoprep.pipeline.optimizer import PipelineOptimizer


def _make(regression_data, **kwargs):
    X, y, groups = regression_data
    defaults = {
        "X_train": X,
        "y_train": y,
        "preprocessing_steps": ["scaler", "snv", "savgol"],
        "groups": groups,
        "n_splits": 2,
        "max_pipeline_length": 1,
        "allowed_preprocess_combinations": [1],
        "log_level": "ERROR",
    }
    defaults.update(kwargs)
    return PipelineOptimizer(**defaults)


class TestValidation:
    def test_requires_numpy_arrays(self):
        with pytest.raises(ValueError, match="numpy arrays"):
            PipelineOptimizer(X_train=[[1, 2]], y_train=[1], preprocessing_steps=["scaler"])

    def test_sample_count_mismatch(self, rng):
        with pytest.raises(ValueError, match="same number of samples"):
            PipelineOptimizer(
                X_train=rng.normal(size=(10, 4)),
                y_train=rng.normal(size=8),
                preprocessing_steps=["scaler"],
            )

    def test_test_set_xor(self, rng):
        X = rng.normal(size=(10, 4))
        y = rng.normal(size=10)
        with pytest.raises(ValueError, match="provided together"):
            PipelineOptimizer(X_train=X, y_train=y, X_test=X, preprocessing_steps=["scaler"])

    def test_invalid_step(self, regression_data):
        X, y, groups = regression_data
        with pytest.raises(ValueError, match="Invalid preprocessing steps"):
            PipelineOptimizer(X_train=X, y_train=y, preprocessing_steps=["nope"], groups=groups)

    def test_invalid_cv_method(self, regression_data):
        X, y, groups = regression_data
        with pytest.raises(ValueError, match="cv_method"):
            PipelineOptimizer(
                X_train=X,
                y_train=y,
                preprocessing_steps=["scaler"],
                groups=groups,
                cv_method="bogus",
            )

    def test_invalid_log_level(self, regression_data):
        X, y, groups = regression_data
        with pytest.raises(ValueError, match="Invalid log level"):
            PipelineOptimizer(
                X_train=X,
                y_train=y,
                preprocessing_steps=["scaler"],
                groups=groups,
                log_level="LOUD",
            )


class TestNoTestSetLeakage:
    def test_objective_source_does_not_reference_test_set(self):
        src = inspect.getsource(PipelineOptimizer.bayes_objective)
        assert "X_test" not in src
        assert "y_test" not in src

    def test_objective_identical_with_or_without_test_set(self, regression_data):
        X, y, groups = regression_data
        opt_no_test = _make(regression_data)
        opt_with_test = _make(regression_data, X_test=X, y_test=y)
        params = {k: (lo + hi) / 2 for k, (lo, hi) in opt_no_test.param_bounds.items()}
        params["ridge_alpha"] = 1.0
        # Same CV objective regardless of whether a test set was supplied.
        assert opt_no_test.bayes_objective(**params) == pytest.approx(
            opt_with_test.bayes_objective(**params)
        )


class TestObjective:
    def test_returns_float(self, regression_data):
        opt = _make(regression_data)
        params = {k: (lo + hi) / 2 for k, (lo, hi) in opt.param_bounds.items()}
        params["ridge_alpha"] = 1.0
        score = opt.bayes_objective(**params)
        assert isinstance(score, float)
        assert score <= 0.0  # negative RMSE

    def test_penalty_on_bad_config(self, regression_data):
        # savgol requires transform parameters that are absent here, so the
        # objective hits its except branch and returns the penalty score.
        opt = _make(regression_data, preprocessing_steps=["savgol"])
        assert opt.bayes_objective(pipeline_config=0.0) == -1e6

    def test_ridge_alpha_not_in_search_space(self, regression_data):
        # The downstream RidgeCV selects its own penalty, so ridge_alpha is
        # not a Bayesian search dimension.
        opt = _make(regression_data)
        assert "ridge_alpha" not in opt.param_bounds


class TestStubbedSummaries:
    """Exercise reporting methods with a stubbed optimizer object."""

    def _attach_stub(self, opt):
        res = [
            {"target": -1.0, "params": {"pipeline_config": 0.0, "ridge_alpha": 1.0}},
            {"target": -0.5, "params": {"pipeline_config": 0.0, "ridge_alpha": 2.0}},
        ]
        opt.optimizer = types.SimpleNamespace(res=res, max=res[1])
        return opt

    def test_summarize(self, regression_data):
        opt = self._attach_stub(_make(regression_data))
        summary = opt.summarize_optimization()
        assert summary["n_trials"] == 2
        assert summary["best_score"] == -0.5
        assert summary["best_rmse"] == pytest.approx(0.5)

    def test_get_all_tested_r2_is_nan(self, regression_data):
        opt = self._attach_stub(_make(regression_data))
        results = opt.get_all_tested_pipelines()
        assert len(results) == 2
        assert np.isnan(results[0]["r2"])

    def test_export(self, regression_data, tmp_path):
        opt = self._attach_stub(_make(regression_data))
        path = tmp_path / "best.json"
        opt.export_best_pipeline(str(path))
        data = json.loads(path.read_text())
        assert "best_score" in data and "pipeline_config" in data

    def test_summarize_without_optimizer_raises(self, regression_data):
        with pytest.raises(AttributeError, match="No optimizer"):
            _make(regression_data).summarize_optimization()


@pytest.mark.parametrize("cv_method", ["group_shuffle_split", "group_kfold"])
def test_build_cv_supports_group_kfold(regression_data, cv_method):
    from sklearn.model_selection import GroupKFold, GroupShuffleSplit

    opt = _make(regression_data, cv_method=cv_method)
    cv = opt._build_cv()
    expected = GroupKFold if cv_method == "group_kfold" else GroupShuffleSplit
    assert isinstance(cv, expected)


@pytest.mark.slow
def test_end_to_end_optimize(regression_data):
    from spectoprep.modelling.ridge import OptimizedRidgeCV

    opt = _make(regression_data, cv_method="group_kfold")
    best_params, pipeline = opt.bayesian_optimize(init_points=2, n_iter=1)
    assert "pipeline_config" in best_params
    # The downstream estimator is a cross-validated Ridge.
    assert isinstance(pipeline.steps[-1][1], OptimizedRidgeCV)
    preds, rmse, r2 = opt.get_best_pipeline_predictions(pipeline)
    assert len(preds) == len(regression_data[1])
    assert rmse >= 0.0
    # Out-of-fold preds must stay aligned with y_train sample order.
    from sklearn.metrics import r2_score

    assert r2_score(regression_data[1], preds) == pytest.approx(r2, rel=1e-6)
