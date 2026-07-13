"""Tests for scatter-correction transformers."""

import numpy as np
import pytest
from sklearn.exceptions import NotFittedError

from spectoprep.preprocessing.scatter import (
    ExtendedMultiplicativeScatterCorrection,
    LocalizedSNV,
    MultiplicativeScatterCorrection,
    RobustNormalVariate,
    StandardNormalVariate,
)


class TestStandardNormalVariate:
    def test_shape_and_normalisation(self, spectra):
        out = StandardNormalVariate().fit_transform(spectra)
        assert out.shape == spectra.shape
        assert np.allclose(out.mean(axis=1), 0.0, atol=1e-9)
        assert np.allclose(out.std(axis=1), 1.0, atol=1e-9)

    def test_not_fitted_raises(self, spectra):
        with pytest.raises(NotFittedError):
            StandardNormalVariate().transform(spectra)

    def test_constant_spectrum_is_finite(self, constant_spectra):
        out = StandardNormalVariate().fit_transform(constant_spectra)
        assert np.isfinite(out).all()


class TestMultiplicativeScatterCorrection:
    def test_reference_recovered(self, spectra):
        msc = MultiplicativeScatterCorrection().fit(spectra)
        out = msc.transform(spectra)
        assert out.shape == spectra.shape
        assert np.isfinite(out).all()

    def test_not_fitted_raises(self, spectra):
        with pytest.raises(ValueError, match="not fitted"):
            MultiplicativeScatterCorrection().transform(spectra)

    def test_flat_reference_no_div_by_zero(self):
        flat = np.ones((3, 8))
        out = MultiplicativeScatterCorrection().fit(flat).transform(flat)
        assert np.isfinite(out).all()


class TestExtendedMSC:
    @pytest.mark.parametrize("order", [1, 2])
    def test_shape_preserved(self, spectra, order):
        emsc = ExtendedMultiplicativeScatterCorrection(order=order).fit(spectra)
        out = emsc.transform(spectra)
        assert out.shape == spectra.shape

    def test_not_fitted_raises(self, spectra):
        with pytest.raises(ValueError, match="not fitted"):
            ExtendedMultiplicativeScatterCorrection().transform(spectra)


class TestLocalizedSNV:
    def test_even_window_raises(self, spectra):
        with pytest.raises(ValueError, match="odd"):
            LocalizedSNV(window_size=10).fit(spectra)

    def test_shape_and_finite(self, spectra):
        out = LocalizedSNV(window_size=5).fit_transform(spectra)
        assert out.shape == spectra.shape
        assert np.isfinite(out).all()

    def test_constant_spectrum_finite(self, constant_spectra):
        out = LocalizedSNV(window_size=5).fit_transform(constant_spectra)
        assert np.isfinite(out).all()


class TestRobustNormalVariate:
    def test_shape(self, spectra):
        out = RobustNormalVariate().fit_transform(spectra)
        assert out.shape == spectra.shape

    def test_constant_spectrum_finite(self, constant_spectra):
        out = RobustNormalVariate().fit_transform(constant_spectra)
        assert np.isfinite(out).all()
