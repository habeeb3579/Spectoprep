"""Shared pytest fixtures for the SpectoPrep test suite."""

import matplotlib
import numpy as np
import pytest

# Use a non-interactive backend so plotting tests never open a window.
matplotlib.use("Agg")


@pytest.fixture
def rng():
    """Deterministic random generator."""
    return np.random.default_rng(0)


@pytest.fixture
def spectra(rng):
    """A small, well-conditioned spectral matrix (6 samples x 20 channels)."""
    base = np.linspace(0.0, 1.0, 20)
    return np.array([base + rng.normal(scale=0.05, size=20) + i * 0.1 for i in range(6)])


@pytest.fixture
def constant_spectra():
    """Spectra whose rows are constant, to exercise divide-by-zero guards."""
    return np.ones((4, 12), dtype=float)


@pytest.fixture
def regression_data(rng):
    """A grouped regression dataset for the optimizer.

    30 samples, 10 features, 10 groups of 3 replicates each, with a linear
    signal plus small noise.
    """
    X = rng.normal(size=(30, 10))
    coef = rng.normal(size=10)
    y = X @ coef + rng.normal(scale=0.1, size=30)
    groups = np.repeat(np.arange(10), 3)
    return X, y, groups
