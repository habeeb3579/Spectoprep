# spectoprep/modelling/ridge.py
"""Ridge regression with built-in cross-validation for spectral modelling."""

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import RidgeCV as SklearnRidgeCV
from sklearn.model_selection import GroupKFold, KFold
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y


class OptimizedRidgeCV(BaseEstimator, RegressorMixin):
    """Ridge regression with alpha selected by cross-validation.

    Thin, scikit-learn-compatible wrapper around
    :class:`sklearn.linear_model.RidgeCV` that adds optional group-aware
    cross-validation.

    Parameters
    ----------
    alphas : array-like, default=None
        Alpha values to try. When ``None``, ``np.logspace(-3, 3, 10)`` is used.
        Resolved in :meth:`fit` so the estimator honours the scikit-learn
        contract that ``__init__`` only stores its arguments unchanged.
    cv : int, cross-validation generator, iterable or None, default=5
        Cross-validation splitting strategy. When ``None``, ``RidgeCV`` uses
        its efficient leave-one-out generalized cross-validation (GCV), and
        ``gcv_mode``/``store_cv_results`` become available.
    scoring : str or callable, default='neg_mean_squared_error'
        Scoring used to select ``alpha``.
    fit_intercept : bool, default=True
        Whether to fit an intercept.
    gcv_mode : {None, 'auto', 'svd', 'eigen'}, default=None
        GCV strategy. Only used when ``cv is None``.
    store_cv_results : bool, default=False
        Store per-alpha CV results in ``cv_results_``. Only valid when
        ``cv is None`` (RidgeCV restriction).
    groups : array-like, default=None
        Group labels. When provided (and ``cv`` is not ``None``), grouped
        K-fold splits are materialised so no group is split across folds.

    Notes
    -----
    ``normalize`` was removed from scikit-learn's ``RidgeCV`` in 1.2 and
    ``store_cv_values`` renamed to ``store_cv_results`` in 1.5; this wrapper
    tracks the current API. Standardise inputs with a scaler in a
    :class:`~sklearn.pipeline.Pipeline` instead of relying on ``normalize``.
    """

    def __init__(
        self,
        alphas=None,
        cv=5,
        scoring="neg_mean_squared_error",
        fit_intercept=True,
        gcv_mode=None,
        store_cv_results=False,
        groups=None,
    ):
        self.alphas = alphas
        self.cv = cv
        self.scoring = scoring
        self.fit_intercept = fit_intercept
        self.gcv_mode = gcv_mode
        self.store_cv_results = store_cv_results
        self.groups = groups

    def _resolve_cv(self, X, y):
        """Build the cv argument passed to scikit-learn's RidgeCV."""
        if self.cv is None:
            return None
        if self.groups is not None:
            groups = np.asarray(self.groups)
            if len(groups) != X.shape[0]:
                raise ValueError("groups must have the same length as X")
            n_splits = self.cv if isinstance(self.cv, int) else 5
            # Materialise grouped splits into an explicit (train, test) iterable,
            # which RidgeCV accepts and which keeps groups intact across folds.
            return list(GroupKFold(n_splits=n_splits).split(X, y, groups))
        if isinstance(self.cv, int):
            return KFold(n_splits=self.cv, shuffle=True, random_state=42)
        return self.cv

    def fit(self, X, y, sample_weight=None):
        """Fit the Ridge model, selecting ``alpha`` by cross-validation."""
        X, y = check_X_y(X, y, y_numeric=True, multi_output=True)

        alphas = self.alphas if self.alphas is not None else np.logspace(-3, 3, 10)
        cv = self._resolve_cv(X, y)

        ridge_kwargs = {
            "alphas": alphas,
            "fit_intercept": self.fit_intercept,
            "scoring": self.scoring,
            "cv": cv,
        }
        # gcv_mode / store_cv_results are only accepted by RidgeCV for the
        # built-in GCV path (cv is None); passing them alongside an explicit
        # cv raises in scikit-learn.
        if cv is None:
            ridge_kwargs["gcv_mode"] = self.gcv_mode
            ridge_kwargs["store_cv_results"] = self.store_cv_results

        self.ridge_cv_ = SklearnRidgeCV(**ridge_kwargs)
        self.ridge_cv_.fit(X, y, sample_weight=sample_weight)

        self.alpha_ = self.ridge_cv_.alpha_
        self.coef_ = self.ridge_cv_.coef_
        self.intercept_ = self.ridge_cv_.intercept_
        self.n_features_in_ = self.ridge_cv_.n_features_in_
        if hasattr(self.ridge_cv_, "cv_results_"):
            self.cv_results_ = self.ridge_cv_.cv_results_
        return self

    def predict(self, X):
        """Predict target values for ``X``."""
        check_is_fitted(self, ["ridge_cv_", "alpha_", "coef_", "intercept_"])
        X = check_array(X)
        return self.ridge_cv_.predict(X)

    def score(self, X, y, sample_weight=None):
        """Return the :math:`R^2` score of the prediction."""
        check_is_fitted(self, ["ridge_cv_", "alpha_", "coef_", "intercept_"])
        return self.ridge_cv_.score(X, y, sample_weight=sample_weight)

    def get_cv_results(self):
        """Return a summary of the cross-validation results."""
        check_is_fitted(self, ["ridge_cv_"])
        return {
            "alpha": self.alpha_,
            "alphas_tested": self.alphas if self.alphas is not None else np.logspace(-3, 3, 10),
            "cv_results": getattr(self, "cv_results_", None),
            "coef": self.coef_,
            "intercept": self.intercept_,
        }
