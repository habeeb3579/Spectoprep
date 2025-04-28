"""
Scatter correction methods for spectroscopic data preprocessing.
"""

import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

class StandardNormalVariate(BaseEstimator, TransformerMixin):
    """
    Standard Normal Variate (SNV) transformation.
    
    SNV is a row-wise transformation that centers and scales each spectrum 
    individually. It's commonly used to remove scatter effects in 
    spectroscopic data.
    
    Attributes
    ----------
    mean_ : ndarray of shape (n_samples, 1)
        Mean of each sample (row) computed during fit.
    std_ : ndarray of shape (n_samples, 1)
        Standard deviation of each sample computed during fit.
    is_fitted_ : bool
        Flag indicating if the transformer has been fitted.
    """
    
    def __init__(self):
        self.mean_ = None
        self.std_ = None
        self.is_fitted_ = False

    def fit(self, X, y=None):
        """
        Compute mean and standard deviation of each sample.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input spectra.
        y : None
            Ignored.
            
        Returns
        -------
        self : object
            Returns self.
        """
        self.mean_ = np.mean(X, axis=1, keepdims=True)
        self.std_ = np.std(X, axis=1, keepdims=True)
        self.is_fitted_ = True
        return self

    def transform(self, X):
        """
        Apply the StandardNormalVariate transformation.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input spectra.
            
        Returns
        -------
        X_transformed : ndarray of shape (n_samples, n_features)
            Transformed spectra.
        """
        check_is_fitted(self, 'is_fitted_')
        mean = np.mean(X, axis=1, keepdims=True)
        std = np.std(X, axis=1, keepdims=True)
        return (X - mean) / std


class MultiplicativeScatterCorrection(BaseEstimator, TransformerMixin):
    """
    Multiplicative Scatter Correction (MSC) for spectroscopic data.
    
    MSC performs a linear regression of each spectrum on a reference spectrum
    (usually the mean spectrum) and corrects using the estimated coefficients.
    
    Attributes
    ----------
    mean_reference : ndarray of shape (n_features,)
        Reference spectrum (mean of all spectra by default).
    """
    
    def __init__(self):
        self.mean_reference = None

    def fit(self, X, y=None):
        """
        Compute the reference spectrum.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input spectra.
        y : None
            Ignored.
            
        Returns
        -------
        self : object
            Returns self.
        """
        self.mean_reference = np.mean(X, axis=0)
        return self

    def transform(self, X):
        """
        Apply the MSC transformation.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input spectra.
            
        Returns
        -------
        X_transformed : ndarray of shape (n_samples, n_features)
            Transformed spectra.
        """
        if self.mean_reference is None:
            raise ValueError("MSC instance is not fitted yet.")
        corrected_spectra = []
        for spectrum in X:
            coef = np.polyfit(self.mean_reference, spectrum, 1)
            corrected = (spectrum - coef[1]) / coef[0]
            corrected_spectra.append(corrected)
        return np.array(corrected_spectra)


class ExtendedMultiplicativeScatterCorrection(BaseEstimator, TransformerMixin):
    """
    Extended Multiplicative Scatter Correction (EMSC) for spectroscopic data.
    
    EMSC extends the MSC method by incorporating polynomial terms to account
    for more complex spectral variations.
    
    Parameters
    ----------
    order : int, default=2
        Order of the polynomial used in the correction.
        
    Attributes
    ----------
    reference_spectrum : ndarray of shape (n_features,)
        Reference spectrum (mean of training spectra).
    """
    
    def __init__(self, order=2):
        self.order = order
        self.reference_spectrum = None

    def fit(self, X, y=None):
        """
        Compute the reference spectrum.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input spectra.
        y : None
            Ignored.
            
        Returns
        -------
        self : object
            Returns self.
        """
        self.reference_spectrum = np.mean(X, axis=0)
        return self

    def transform(self, X):
        """
        Apply the EMSC transformation.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input spectra.
            
        Returns
        -------
        X_transformed : ndarray of shape (n_samples, n_features)
            Transformed spectra.
        """
        if self.reference_spectrum is None:
            raise ValueError("EMSC instance is not fitted yet.")
        corrected_spectra = []
        for spectrum in X:
            Xt = np.vstack([self.reference_spectrum**i for i in range(self.order + 1)]).T
            coef = np.linalg.lstsq(Xt, spectrum, rcond=None)[0]
            corrected = spectrum - Xt[:, 1:] @ coef[1:]
            corrected_spectra.append(corrected)
        return np.array(corrected_spectra)


class LocalizedSNV(BaseEstimator, TransformerMixin):
    """
    Localized Standard Normal Variate (LSNV) using a sliding window.
    
    LSNV applies the SNV transformation using a local window around each
    wavelength point rather than the entire spectrum.
    
    Parameters
    ----------
    window_size : int, default=11
        Size of the sliding window. Must be odd.
    """
    
    def __init__(self, window_size=11):
        self.window_size = window_size

    def fit(self, X, y=None):
        """
        No-op.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input spectra.
        y : None
            Ignored.
            
        Returns
        -------
        self : object
            Returns self.
        """
        if self.window_size % 2 == 0:
            raise ValueError("window_size must be odd")
        return self

    def transform(self, X):
        """
        Apply LSNV transformation.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input spectra.
            
        Returns
        -------
        X_transformed : ndarray of shape (n_samples, n_features)
            Transformed spectra.
        """
        def localized_snv(spectrum):
            half_win = self.window_size // 2
            corrected = np.zeros_like(spectrum)
            for i in range(len(spectrum)):
                start = max(0, i - half_win)
                end = min(len(spectrum), i + half_win + 1)
                window = spectrum[start:end]
                corrected[i] = (spectrum[i] - np.mean(window)) / np.std(window)
            return corrected

        return np.apply_along_axis(localized_snv, axis=1, arr=X)


class RobustNormalVariate(BaseEstimator, TransformerMixin):
    """
    Robust Normal Variate (RNV) Preprocessing.
    
    RNV is a robust version of SNV that uses percentiles instead of mean and
    standard deviation to reduce the influence of outliers.
    
    Parameters
    ----------
    lower_percentile : float, default=25
        The lower percentile for robust scaling.
    upper_percentile : float, default=75
        The upper percentile for robust scaling.
    """
    
    def __init__(self, lower_percentile=25, upper_percentile=75):
        self.lower_percentile = lower_percentile
        self.upper_percentile = upper_percentile

    def fit(self, X, y=None):
        """
        No-op.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input spectra.
        y : None
            Ignored.
            
        Returns
        -------
        self : object
            Returns self.
        """
        return self

    def transform(self, X):
        """
        Apply RNV transformation.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input spectra.
            
        Returns
        -------
        X_transformed : ndarray of shape (n_samples, n_features)
            Transformed spectra.
        """
        lower_bound = np.percentile(X, self.lower_percentile, axis=1, keepdims=True)
        upper_bound = np.percentile(X, self.upper_percentile, axis=1, keepdims=True)
        scale = upper_bound - lower_bound
        median = np.median(X, axis=1, keepdims=True)
        return (X - median) / scale