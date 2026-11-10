"""Dimensionality-reduction wrapper that caps its sizes by the data it is fitted on."""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin, clone


class CappedComponents(TransformerMixin, BaseEstimator):
    """Fit ``estimator`` with ``n_components`` (and ``n_neighbors``) capped by the fit data.

    Search procedures propose component counts from fixed bounds, but the cross-validation
    folds a transform is fitted on can be smaller than any bound chosen in advance. The
    wrapper clones ``estimator`` at fit time and lowers ``n_components`` to what the
    training fold supports (at most ``n_samples - 1`` and ``n_features``; for estimators
    with ``n_neighbors``, at most ``n_neighbors - 1`` and ``n_neighbors <= n_samples - 1``),
    so that a proposal that is too large still yields a valid transform instead of a failure.

    Parameters
    ----------
    estimator : estimator with an ``n_components`` parameter
        For example ``PCA``, ``FastICA``, ``KernelPCA`` or ``LocallyLinearEmbedding``.
    """

    def __init__(self, estimator):
        self.estimator = estimator

    def fit(self, X, y=None):
        X = np.asarray(X)
        n_samples, n_features = X.shape
        est = clone(self.estimator)
        params = est.get_params()
        cap = min(n_features, n_samples - 1)
        if "n_neighbors" in params:
            n_neighbors = int(max(2, min(params["n_neighbors"], n_samples - 1)))
            est.set_params(n_neighbors=n_neighbors)
            cap = min(cap, n_neighbors - 1)
        est.set_params(n_components=int(max(1, min(params["n_components"], cap))))
        self.estimator_ = est.fit(X, y)
        return self

    def transform(self, X):
        return self.estimator_.transform(np.asarray(X))
