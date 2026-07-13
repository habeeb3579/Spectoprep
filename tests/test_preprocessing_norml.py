"""Tests for normalization/scaling transformers."""

import numpy as np
import pytest
from sklearn.exceptions import NotFittedError

from spectoprep.preprocessing.norml import (
    Autoscaling,
    ColumnStandardizer,
    GlobalScaler,
    MeanCentering,
    Normalization,
    RowStandardizer,
)


class TestNormalization:
    def test_minmax_range(self, spectra):
        out = Normalization(method="minmax", feature_range=(0, 1)).fit_transform(spectra)
        assert np.allclose(out.min(axis=1), 0.0, atol=1e-9)
        assert np.allclose(out.max(axis=1), 1.0, atol=1e-9)

    def test_zscore(self, spectra):
        out = Normalization(method="zscore").fit_transform(spectra)
        assert np.allclose(out.mean(axis=1), 0.0, atol=1e-9)
        assert np.allclose(out.std(axis=1), 1.0, atol=1e-9)

    def test_unknown_method_raises(self, spectra):
        with pytest.raises(ValueError, match="Unsupported"):
            Normalization(method="bogus").fit(spectra)

    def test_not_fitted_raises(self, spectra):
        with pytest.raises(NotFittedError):
            Normalization().transform(spectra)

    @pytest.mark.parametrize("method", ["minmax", "zscore"])
    def test_constant_spectrum_finite(self, constant_spectra, method):
        out = Normalization(method=method).fit_transform(constant_spectra)
        assert np.isfinite(out).all()


class TestAutoscaling:
    def test_columns_standardised(self, spectra):
        out = Autoscaling().fit_transform(spectra)
        assert np.allclose(out.mean(axis=0), 0.0, atol=1e-9)
        assert np.allclose(out.std(axis=0), 1.0, atol=1e-9)

    def test_not_fitted_raises(self, spectra):
        with pytest.raises(ValueError, match="not fitted"):
            Autoscaling().transform(spectra)

    def test_constant_column_finite(self):
        X = np.ones((5, 4))
        out = Autoscaling().fit_transform(X)
        assert np.isfinite(out).all()


class TestMeanCentering:
    def test_columns_centred(self, spectra):
        out = MeanCentering().fit_transform(spectra)
        assert np.allclose(out.mean(axis=0), 0.0, atol=1e-9)


class TestGlobalScaler:
    def test_factor(self, spectra):
        out = GlobalScaler(factor=2.0).fit_transform(spectra.astype(float))
        assert np.allclose(out, spectra * 2.0)

    def test_mean_flag(self, spectra):
        out = GlobalScaler(factor=1.0, mean=True).fit_transform(spectra.astype(float))
        assert np.allclose(out.mean(axis=0), 0.0, atol=1e-9)


class TestRowAndColumnStandardizers:
    def test_row_standardizer(self, spectra):
        out = RowStandardizer().fit_transform(spectra)
        assert np.allclose(out.mean(axis=1), 0.0, atol=1e-9)

    def test_row_standardizer_constant_finite(self, constant_spectra):
        out = RowStandardizer().fit_transform(constant_spectra)
        assert np.isfinite(out).all()

    def test_column_standardizer_matches_sklearn(self, spectra):
        from sklearn.preprocessing import StandardScaler

        expected = StandardScaler().fit_transform(spectra)
        out = ColumnStandardizer().fit(spectra).transform(spectra)
        assert np.allclose(out, expected)
