"""Tests for build_preprocessor_from_bayes transformer factory."""

import pytest
from sklearn.base import BaseEstimator

from spectoprep.pipeline.builder import build_preprocessor_from_bayes
from spectoprep.pipeline.config import AVAILABLE_STEPS, DEFAULT_PARAM_BOUNDS

# A parameter dict that supplies the midpoint of every bound, enough to build
# any transformer branch.
PARAMS = {k: (lo + hi) / 2 for k, (lo, hi) in DEFAULT_PARAM_BOUNDS.items()}
PARAMS["ridge_alpha"] = 1.0

X_SHAPE = (30, 20)


@pytest.mark.parametrize("name", sorted(AVAILABLE_STEPS))
def test_every_step_builds_an_estimator(name):
    transformer = build_preprocessor_from_bayes(name, PARAMS, X_SHAPE, random_state=0, n_jobs=1)
    assert isinstance(transformer, BaseEstimator)
    assert hasattr(transformer, "fit")


def test_unknown_name_raises():
    with pytest.raises(ValueError, match="Unknown preprocessor"):
        build_preprocessor_from_bayes("does_not_exist", PARAMS, X_SHAPE, 0, 1)


def test_savgol_window_snapped_to_allowed():
    t = build_preprocessor_from_bayes("savgol", {**PARAMS, "savgol_filter_win": 8.2}, X_SHAPE, 0, 1)
    assert t.filter_win in (7, 9, 11, 13)


def test_pca_components_capped_by_features():
    t = build_preprocessor_from_bayes("pca", {**PARAMS, "pca_n_components": 999}, X_SHAPE, 0, 1)
    assert t.n_components <= X_SHAPE[1]
