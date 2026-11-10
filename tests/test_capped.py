"""Tests for the fit-time component cap used by the dimensionality-reduction transforms."""

import numpy as np
import pytest
from sklearn.decomposition import PCA, FastICA, KernelPCA
from sklearn.manifold import LocallyLinearEmbedding

from spectoprep.pipeline.builder import build_preprocessor_from_bayes
from spectoprep.pipeline.config import DEFAULT_PARAM_BOUNDS
from spectoprep.preprocessing.capped import CappedComponents


@pytest.mark.parametrize(
    "est",
    [
        PCA(n_components=100),
        FastICA(n_components=100, max_iter=500),
        KernelPCA(n_components=100, kernel="rbf"),
        LocallyLinearEmbedding(n_components=30, n_neighbors=50),
    ],
)
def test_oversized_request_is_capped_by_the_fit_data(est):
    X = np.random.default_rng(0).normal(size=(25, 40))
    out = CappedComponents(est).fit_transform(X)
    assert out.shape[0] == 25 and 1 <= out.shape[1] <= 24
    assert np.isfinite(out).all()


def test_small_request_is_left_alone():
    X = np.random.default_rng(0).normal(size=(40, 30))
    out = CappedComponents(PCA(n_components=5)).fit_transform(X)
    assert out.shape == (40, 5)


def test_builder_pca_survives_a_fold_smaller_than_the_proposed_components():
    params = {k: (lo + hi) / 2 for k, (lo, hi) in DEFAULT_PARAM_BOUNDS.items()}
    params["pca_n_components"] = 90
    tr = build_preprocessor_from_bayes("pca", params, (80, 700), 0, 1)
    Xfold = np.random.default_rng(0).normal(size=(43, 700))  # inner-fold training size
    assert tr.fit_transform(Xfold).shape == (43, 42)
