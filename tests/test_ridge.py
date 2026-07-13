"""Tests for the OptimizedRidgeCV wrapper (fixed for modern scikit-learn)."""

import numpy as np
import pytest
from sklearn.exceptions import NotFittedError

from spectoprep.modelling.ridge import OptimizedRidgeCV


@pytest.fixture
def linear_data(rng):
    X = rng.normal(size=(40, 6))
    coef = rng.normal(size=6)
    y = X @ coef + rng.normal(scale=0.05, size=40)
    return X, y


class TestOptimizedRidgeCV:
    def test_fit_predict_score(self, linear_data):
        X, y = linear_data
        model = OptimizedRidgeCV(cv=3).fit(X, y)
        preds = model.predict(X)
        assert preds.shape == (40,)
        assert model.score(X, y) > 0.9
        assert model.alpha_ > 0

    def test_grouped_cv(self, linear_data):
        X, y = linear_data
        groups = np.repeat(np.arange(10), 4)
        model = OptimizedRidgeCV(cv=5, groups=groups).fit(X, y)
        assert model.score(X, y) > 0.9

    def test_grouped_length_mismatch_raises(self, linear_data):
        X, y = linear_data
        with pytest.raises(ValueError, match="same length"):
            OptimizedRidgeCV(cv=3, groups=np.arange(5)).fit(X, y)

    def test_gcv_path_stores_results(self, linear_data):
        X, y = linear_data
        model = OptimizedRidgeCV(cv=None, store_cv_results=True).fit(X, y)
        assert hasattr(model, "cv_results_")
        assert model.get_cv_results()["cv_results"] is not None

    def test_predict_before_fit_raises(self, linear_data):
        X, _ = linear_data
        with pytest.raises(NotFittedError):
            OptimizedRidgeCV().predict(X)

    def test_get_params_roundtrip(self):
        # __init__ must store params unchanged (sklearn contract).
        model = OptimizedRidgeCV(cv=4, fit_intercept=False)
        params = model.get_params()
        assert params["cv"] == 4
        assert params["fit_intercept"] is False
        assert params["alphas"] is None
