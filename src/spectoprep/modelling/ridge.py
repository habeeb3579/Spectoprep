# spectroprep/modeling/ridge.py

import numpy as np
from sklearn.linear_model import RidgeCV as SklearnRidgeCV
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.model_selection import GroupKFold, KFold
from sklearn.metrics import mean_squared_error, r2_score

class OptimizedRidgeCV(BaseEstimator, RegressorMixin):
    """
    Ridge regression with built-in cross-validation and optimization capabilities.
    
    Parameters
    ----------
    alphas : array-like, default=np.logspace(-3, 3, 10)
        Array of alpha values to try. A large array of values will slow
        down the computation.
    cv : int, cross-validation generator or an iterable, default=5
        Determines the cross-validation splitting strategy.
    scoring : str, callable, default='neg_mean_squared_error'
        A string or a scorer callable object / function with signature
        ``scorer(estimator, X, y)``.
    fit_intercept : bool, default=True
        Whether to calculate the intercept for this model.
    normalize : bool, default=False
        This parameter is ignored when fit_intercept is set to False.
        If True, the regressors X will be normalized before regression
        by subtracting the mean and dividing by the l2-norm.
    gcv_mode : {None, 'auto', 'svd', 'eigen'}, default=None
        Flag indicating which strategy to use when performing
        Generalized Cross-Validation.
    store_cv_values : bool, default=False
        Flag indicating if the cross-validation values corresponding to
        each alpha should be stored in the cv_values_ attribute.
    groups : array-like, default=None
        Group labels for the samples. Only used if cv is a group-based
        cross-validation splitter.
    """
    
    def __init__(self, alphas=None, cv=5, scoring='neg_mean_squared_error', 
                 fit_intercept=True, normalize=False, gcv_mode=None, 
                 store_cv_values=False, groups=None):
        self.alphas = alphas if alphas is not None else np.logspace(-3, 3, 10)
        self.cv = cv
        self.scoring = scoring
        self.fit_intercept = fit_intercept
        self.normalize = normalize
        self.gcv_mode = gcv_mode
        self.store_cv_values = store_cv_values
        self.groups = groups
        
    def fit(self, X, y, sample_weight=None):
        """
        Fit Ridge regression model with cross-validation.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,) or (n_samples, n_targets)
            Target values.
        sample_weight : float or array-like of shape (n_samples,), default=None
            Individual weights for each sample.
            
        Returns
        -------
        self : object
            Returns self.
        """
        X, y = check_X_y(X, y, y_numeric=True, multi_output=True)
        
        # Create cross-validation object
        if self.groups is not None:
            if len(self.groups) != X.shape[0]:
                raise ValueError("groups must have the same length as X")
            cv = GroupKFold(n_splits=self.cv) if isinstance(self.cv, int) else self.cv
        else:
            cv = KFold(n_splits=self.cv, shuffle=True, random_state=42) if isinstance(self.cv, int) else self.cv
        
        # Create RidgeCV estimator from scikit-learn
        self.ridge_cv_ = SklearnRidgeCV(
            alphas=self.alphas,
            fit_intercept=self.fit_intercept,
            normalize=self.normalize,
            scoring=self.scoring,
            cv=cv,
            gcv_mode=self.gcv_mode,
            store_cv_values=self.store_cv_values
        )
        
        # Fit the model
        if self.groups is not None:
            # Custom CV with groups
            self.ridge_cv_.fit(X, y, sample_weight=sample_weight)
        else:
            self.ridge_cv_.fit(X, y, sample_weight=sample_weight)
        
        # Store fitted attributes
        self.alpha_ = self.ridge_cv_.alpha_
        self.coef_ = self.ridge_cv_.coef_
        self.intercept_ = self.ridge_cv_.intercept_
        if hasattr(self.ridge_cv_, 'cv_values_'):
            self.cv_values_ = self.ridge_cv_.cv_values_
        
        return self
    
    def predict(self, X):
        """
        Predict using the Ridge model.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples.
            
        Returns
        -------
        y_pred : array-like of shape (n_samples,) or (n_samples, n_targets)
            Returns predicted values.
        """
        check_is_fitted(self, ["ridge_cv_", "alpha_", "coef_", "intercept_"])
        X = check_array(X)
        return self.ridge_cv_.predict(X)
    
    def score(self, X, y, sample_weight=None):
        """
        Return the coefficient of determination R^2 of the prediction.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples.
        y : array-like of shape (n_samples,) or (n_samples, n_targets)
            True values for X.
        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights.
            
        Returns
        -------
        score : float
            R^2 of self.predict(X) wrt. y.
        """
        check_is_fitted(self, ["ridge_cv_", "alpha_", "coef_", "intercept_"])
        return self.ridge_cv_.score(X, y, sample_weight=sample_weight)
    
    def get_cv_results(self):
        """
        Return cross-validation results.
        
        Returns
        -------
        cv_results : dict
            Results from cross-validation.
        """
        check_is_fitted(self, ["ridge_cv_"])
        
        return {
            'alpha': self.alpha_,
            'alphas_tested': self.alphas,
            'cv_values': getattr(self, 'cv_values_', None),
            'coef': self.coef_,
            'intercept': self.intercept_
        }