"""Tests for baseline-correction, detrend, and smoothing transformers."""

import numpy as np
import pytest
from sklearn.exceptions import NotFittedError

from spectoprep.preprocessing.baseline import ALSBaselineCorrection, DetrendTransformer
from spectoprep.preprocessing.smoothing import SavitzkyGolay


class TestALSBaselineCorrection:
    def test_shape_preserved(self, spectra):
        als = ALSBaselineCorrection(lam=1e3, p=0.01, niter=5).fit(spectra)
        out = als.transform(spectra)
        assert out.shape == spectra.shape
        assert np.isfinite(out).all()

    def test_not_fitted_raises(self, spectra):
        with pytest.raises(NotFittedError):
            ALSBaselineCorrection().transform(spectra)

    def test_feature_mismatch_raises(self, spectra):
        als = ALSBaselineCorrection(niter=3).fit(spectra)
        with pytest.raises(ValueError, match="different from what was seen"):
            als.transform(spectra[:, :-1])


class TestDetrend:
    @pytest.mark.parametrize("method", ["simple", "polynomial"])
    def test_linear_trend_removed(self, method):
        x = np.linspace(0, 10, 30)
        X = np.vstack([3 * x + 2, 5 * x - 1])
        out = DetrendTransformer(method=method, order=1).transform(X)
        assert np.allclose(out, 0.0, atol=1e-6)

    def test_1d_input_reshaped(self):
        out = DetrendTransformer(method="simple").transform(np.arange(10.0))
        assert out.shape == (1, 10)

    def test_unknown_method_raises(self, spectra):
        with pytest.raises(ValueError, match="Unknown method"):
            DetrendTransformer(method="bogus").transform(spectra)


class TestSavitzkyGolay:
    def test_shape_preserved(self, spectra):
        out = SavitzkyGolay(filter_win=5, poly_order=2).fit_transform(spectra)
        assert out.shape == spectra.shape

    def test_even_window_raises(self, spectra):
        with pytest.raises(ValueError, match="odd"):
            SavitzkyGolay(filter_win=6).fit(spectra)

    def test_poly_order_too_high_raises(self, spectra):
        with pytest.raises(ValueError, match="less than"):
            SavitzkyGolay(filter_win=5, poly_order=5).fit(spectra)

    def test_first_derivative_of_ramp_is_constant(self):
        x = np.linspace(0, 1, 40)
        X = (2 * x).reshape(1, -1)
        out = SavitzkyGolay(filter_win=7, poly_order=2, deriv_order=1).fit_transform(X)
        # Interior derivative should be roughly constant (ignore edges).
        interior = out[0, 5:-5]
        assert np.std(interior) < 1e-6
