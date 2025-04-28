"""
Baseline correction methods for spectroscopic data.
"""

import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted
from scipy.interpolate import LSQUnivariateSpline

class ALSBaselineCorrection(BaseEstimator, TransformerMixin):
    """
    Asymmetric Least Squares Baseline Correction.
    
    This method estimates the baseline of spectra by fitting a smooth curve
    that preferentially lies beneath the data points.
    
    Parameters
    ----------
    lam : float, default=1e4
        Smoothness parameter. Higher values make the baseline smoother.
    p : float, default=0.001
        Asymmetry parameter. Small values (<<1) force the baseline to lie
        below the data points.
    niter : int, default=10
        Number of iterations for the ALS algorithm.
        
    Attributes
    ----------
    baseline_ : ndarray of shape (n_samples, n_features)
        Estimated baseline for each spectrum.
    is_fitted_ : bool
        Flag indicating if the transformer has been fitted.
    """
    
    def __init__(self, lam=1e4, p=0.001, niter=10):
        self.lam = lam
        self.p = p
        self.niter = niter
        self.baseline_ = None
        self.is_fitted_ = False

    def fit(self, X, y=None):
        """
        Fit the baseline for the input data X.
        
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
        n_samples, n_features = X.shape
        self.baseline_ = np.zeros_like(X)
        diag = diags([1, -2, 1], [0, -1, -2], shape=(n_features, n_features - 2))
        penalty_matrix = self.lam * diag.dot(diag.T)

        for i in range(n_samples):
            yi = X[i]
            w = np.ones(len(yi))
            for _ in range(self.niter):
                W = diags(w, 0)
                Z = W + penalty_matrix
                self.baseline_[i] = spsolve(Z, w * yi)
                w = self.p * (yi > self.baseline_[i]) + (1 - self.p) * (yi < self.baseline_[i])
        
        self.is_fitted_ = True
        return self

    def transform(self, X):
        """
        Transform the input data X by subtracting the baseline.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input spectra.
            
        Returns
        -------
        X_transformed : ndarray of shape (n_samples, n_features)
            Baseline-corrected spectra.
        """
        check_is_fitted(self, 'is_fitted_')
        
        if X.shape[1] != self.baseline_.shape[1]:
            raise ValueError(f"Shape of input is different from what was seen in `fit`")
        
        # Recompute the baseline for the new dataset
        n_samples, n_features = X.shape
        baseline = np.zeros_like(X)
        diag = diags([1, -2, 1], [0, -1, -2], shape=(n_features, n_features - 2))
        penalty_matrix = self.lam * diag.dot(diag.T)

        for i in range(n_samples):
            yi = X[i]
            w = np.ones(len(yi))
            for _ in range(self.niter):
                W = diags(w, 0)
                Z = W + penalty_matrix
                baseline[i] = spsolve(Z, w * yi)
                w = self.p * (yi > baseline[i]) + (1 - self.p) * (yi < baseline[i])

        return X - baseline


class DetrendTransformer(BaseEstimator, TransformerMixin):
    """
    A transformer for detrending time series or spectral data using various methods.
    
    Parameters
    ----------
    method : str, default='polynomial'
        The detrending method to use:
        - 'simple': Linear detrend between first and last points
        - 'polynomial': Polynomial detrend of specified order
        - 'spline': Spline detrend with specified order and spacing
    
    order : int, default=2
        The order of the polynomial or spline fit.
        Ignored if method='simple'.
    
    dspline : int, default=100
        The spacing between spline knots.
        Only used if method='spline'.
    """
    
    def __init__(self, method='polynomial', order=2, dspline=100):
        self.method = method
        self.order = order
        self.dspline = dspline
    
    def _simple(self, data):
        """Simple linear detrend."""
        if not np.issubdtype(data.dtype, np.floating):
            data = np.require(data, dtype=np.float64)
        ndat = len(data)
        x1, x2 = data[0], data[-1]
        data -= x1 + np.arange(ndat) * (x2 - x1) / float(ndat - 1)
        return data
    
    def _polynomial(self, data):
        """Polynomial detrend."""
        if not np.issubdtype(data.dtype, np.floating):
            data = np.require(data, dtype=np.float64)
        x = np.arange(len(data))
        fit = np.polyval(np.polyfit(x, data, deg=self.order), x)
        data -= fit
        return data
    
    def _spline(self, data):
        """Spline detrend."""
        if not np.issubdtype(data.dtype, np.floating):
            data = np.require(data, dtype=np.float64)
        x = np.arange(len(data))
        splknots = np.arange(self.dspline / 2.0, 
                           len(data) - self.dspline / 2.0 + 2,
                           self.dspline)
        spl = LSQUnivariateSpline(x=x, y=data, t=splknots, k=self.order)
        fit = spl(x)
        data -= fit
        return data
    
    def fit(self, X, y=None):
        """
        Fit the transformer (no-op).
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data.
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
        Apply detrending to the input data.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data to detrend.
            
        Returns
        -------
        X_detrended : array-like of shape (n_samples, n_features)
            Detrended data.
        """
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(1, -1)
            
        X_detrended = np.zeros_like(X, dtype=np.float64)
        
        for i in range(X.shape[0]):
            if self.method == 'simple':
                X_detrended[i] = self._simple(X[i].copy())
            elif self.method == 'polynomial':
                X_detrended[i] = self._polynomial(X[i].copy())
            elif self.method == 'spline':
                X_detrended[i] = self._spline(X[i].copy())
            else:
                raise ValueError(f"Unknown method: {self.method}")
                
        return X_detrended