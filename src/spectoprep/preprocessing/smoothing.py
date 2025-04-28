"""
Smoothing methods for spectroscopic data.
"""

import numpy as np
from scipy.signal import savgol_filter
from sklearn.base import BaseEstimator, TransformerMixin

class SavitzkyGolay(BaseEstimator, TransformerMixin):
    """
    Savitzky-Golay filter for smoothing and differentiation of data.
    
    Parameters
    ----------
    filter_win : int, default=11
        Length of the filter window (must be positive odd integer).
    poly_order : int, default=2
        Order of the polynomial used to fit the samples.
        Must be less than filter_win.
    deriv_order : int, default=0
        Order of the derivative to compute. 0 means smoothing only.
        
    Notes
    -----
    The Savitzky-Golay filter is a digital smoothing polynomial filter
    that can preserve the high-frequency components of the signal
    better than standard averaging techniques.
    """
    
    def __init__(self, filter_win=11, poly_order=2, deriv_order=0):
        self.filter_win = filter_win
        self.poly_order = poly_order
        self.deriv_order = deriv_order

    def fit(self, X, y=None):
        """
        Validate parameters.
        
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
        if self.filter_win % 2 == 0 or self.filter_win < 1:
            raise ValueError("filter_win must be a positive odd number.")
        if self.poly_order >= self.filter_win:
            raise ValueError("poly_order must be less than filter_win")
        return self

    def transform(self, X):
        """
        Apply Savitzky-Golay filter to the data.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input spectra.
            
        Returns
        -------
        X_transformed : ndarray of shape (n_samples, n_features)
            Filtered spectra.
        """
        if self.filter_win % 2 == 0 or self.filter_win < 1:
            raise ValueError("filter_win must be a positive odd number.")
        return savgol_filter(X, self.filter_win, self.poly_order, deriv=self.deriv_order)
        # return np.array([
        #     savgol_filter(
        #         row, 
        #         self.filter_win, 
        #         self.poly_order, 
        #         deriv=self.deriv_order
        #     ) for row in X
        # ])
    
