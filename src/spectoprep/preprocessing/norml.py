"""
Normalization methods for spectroscopic data.
"""

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted
from sklearn.preprocessing import StandardScaler


class Normalization(BaseEstimator, TransformerMixin):
    """
    General normalization transformer that supports different methods.
    
    Parameters
    ----------
    method : str, default='minmax'
        Normalization method:
        - 'minmax': Scales each spectrum to a range specified by feature_range
        - 'zscore': Standardizes each spectrum (mean=0, std=1)
    feature_range : tuple (min, max), default=(0, 1)
        Range to scale the spectra to when method='minmax'.
        
    Attributes
    ----------
    min_ : ndarray of shape (n_samples, 1)
        Minimum values of each sample (for minmax scaling).
    max_ : ndarray of shape (n_samples, 1)
        Maximum values of each sample (for minmax scaling).
    mean_ : ndarray of shape (n_samples, 1)
        Mean values of each sample (for zscore scaling).
    std_ : ndarray of shape (n_samples, 1)
        Standard deviation of each sample (for zscore scaling).
    is_fitted_ : bool
        Flag indicating if the transformer has been fitted.
    """
    
    def __init__(self, method='minmax', feature_range=(0, 1)):
        self.method = method
        self.feature_range = feature_range
        self.min_ = None
        self.max_ = None
        self.mean_ = None
        self.std_ = None
        self.is_fitted_ = False

    def fit(self, X, y=None):
        """
        Fit the normalization parameters.
        
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
        if self.method == 'minmax':
            self.min_ = np.min(X, axis=1, keepdims=True)
            self.max_ = np.max(X, axis=1, keepdims=True)
        elif self.method == 'zscore':
            self.mean_ = np.mean(X, axis=1, keepdims=True)
            self.std_ = np.std(X, axis=1, keepdims=True)
        else:
            raise ValueError("Unsupported normalization method. Choose 'minmax' or 'zscore'.")
        self.is_fitted_ = True
        return self

    def transform(self, X):
        """
        Apply the normalization transformation.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input spectra.
            
        Returns
        -------
        X_transformed : ndarray of shape (n_samples, n_features)
            Normalized spectra.
        """
        check_is_fitted(self, 'is_fitted_')
        
        if self.method == 'minmax':
            range_min, range_max = self.feature_range
            min_vals = np.min(X, axis=1, keepdims=True)
            max_vals = np.max(X, axis=1, keepdims=True)
            return ((X - min_vals) / (max_vals - min_vals)) * (range_max - range_min) + range_min
        elif self.method == 'zscore':
            mean_vals = np.mean(X, axis=1, keepdims=True)
            std_vals = np.std(X, axis=1, keepdims=True)
            return (X - mean_vals) / std_vals


class Autoscaling(BaseEstimator, TransformerMixin):
    """
    Autoscaling (column-wise standardization).
    
    Centers and scales data to unit variance along columns (features).
    
    Attributes
    ----------
    mean_ : ndarray of shape (n_features,)
        Mean value for each feature.
    std_ : ndarray of shape (n_features,)
        Standard deviation for each feature.
    """
    
    def __init__(self):
        self.mean_ = None
        self.std_ = None

    def fit(self, X, y=None):
        """
        Compute the mean and standard deviation of each feature.
        
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
        self.mean_ = np.mean(X, axis=0)
        self.std_ = np.std(X, axis=0)
        return self

    def transform(self, X):
        """
        Center and scale the data.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input spectra.
            
        Returns
        -------
        X_transformed : ndarray of shape (n_samples, n_features)
            Transformed array.
        """
        if self.mean_ is None or self.std_ is None:
            raise ValueError("Autoscaling instance is not fitted yet.")
        return (X - self.mean_) / self.std_


class MeanCentering(BaseEstimator, TransformerMixin):
    """
    Mean Centering transformation.
    
    Centers data by subtracting the column means, without scaling.
    
    Attributes
    ----------
    mean_ : ndarray of shape (n_features,)
        Mean value for each feature.
    """
    
    def __init__(self):
        self.mean_ = None

    def fit(self, X, y=None):
        """
        Compute the mean of each feature.
        
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
        self.mean_ = np.mean(X, axis=0)
        return self

    def transform(self, X):
        """
        Center the data.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input spectra.
            
        Returns
        -------
        X_transformed : ndarray of shape (n_samples, n_features)
            Centered array.
        """
        return X - self.mean_


class GlobalScaler(BaseEstimator, TransformerMixin):
    """
    Applies global scaling to spectra by a constant factor with optional mean centering and standardization.
    
    Parameters
    ----------
    factor : float, default=1.0
        Scaling factor to multiply spectra.
    mean : bool, default=False
        Whether to subtract the mean of each feature.
    std : bool, default=False
        Whether to divide by the standard deviation of each feature.
    """
    def __init__(self, factor=1.0, mean=False, std=False):
        self.factor = factor
        self.mean = mean
        self.std = std
        self.mean_value = None
        self.std_value = None

    def fit(self, X, y=None):
        """
        Compute mean and standard deviation if needed.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The training input samples.
        y : None
            Ignored.
            
        Returns
        -------
        self : object
            Returns self.
        """
        if self.mean:
            self.mean_value = np.mean(X, axis=0)
        if self.std:
            self.std_value = np.std(X, axis=0)
        return self

    def transform(self, X):
        """
        Apply global scaling transformation.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.
            
        Returns
        -------
        X_scaled : array-like of shape (n_samples, n_features)
            Transformed samples.
        """
        scaled_spectra = X * self.factor
        if self.mean and self.mean_value is not None:
            scaled_spectra -= self.mean_value
        if self.std and self.std_value is not None:
            scaled_spectra /= self.std_value
        return scaled_spectra
    


class RowStandardizer(BaseEstimator, TransformerMixin):
    """Standardizes each row independently (i.e. across columns)."""
    def fit(self, X, y=None):
        # No fitting needed since each row is standardized independently
        return self
    
    def transform(self, X, y=None):
        # Apply row-wise standardization
        return np.apply_along_axis(lambda row: (row - np.mean(row)) / np.std(row), axis=1, arr=X)

class ColumnStandardizer(BaseEstimator, TransformerMixin):
    """Standardizes columns using StandardScaler fitted on the training set."""
    def __init__(self):
        self.scaler = StandardScaler()
        
    def fit(self, X, y=None):
        self.scaler.fit(X)
        return self
    
    def transform(self, X, y=None):
        return self.scaler.transform(X)
    

#autoscaling is same as standardscaler or column-wise standardization
#SNV is same as row standardization