"""Smoke tests for SpectroPrepPlotter (Agg backend, no display)."""

import types

import numpy as np
import pytest
from matplotlib.figure import Figure

from spectoprep.visualization.plots import SpectroPrepPlotter


@pytest.fixture
def wavenumbers():
    return np.linspace(900, 1800, 20)


def test_plot_spectra_returns_fig_ax(wavenumbers, spectra):
    fig, ax = SpectroPrepPlotter.plot_spectra(wavenumbers, spectra)
    assert isinstance(fig, Figure)
    assert len(ax.lines) == spectra.shape[0]


def test_plot_spectra_saves(tmp_path, wavenumbers, spectra):
    path = tmp_path / "spectra.png"
    SpectroPrepPlotter.plot_spectra(wavenumbers, spectra, save_path=str(path))
    assert path.exists()


def test_plot_prediction_scatter(rng):
    y_true = rng.normal(size=25)
    y_pred = y_true + rng.normal(scale=0.1, size=25)
    fig, ax = SpectroPrepPlotter.plot_prediction_scatter(y_true, y_pred)
    assert isinstance(fig, Figure)


def test_plot_optimization_results_handles_nan_r2():
    """Regression test: r2 is NaN per trial and must not raise."""
    results = [
        {"pipeline_config": ("snv",), "rmse": 1.0, "r2": float("nan")},
        {"pipeline_config": ("scaler",), "rmse": 2.0, "r2": float("nan")},
    ]
    fake = types.SimpleNamespace(
        optimizer=object(),
        get_all_tested_pipelines=lambda: results,
    )
    fig = SpectroPrepPlotter.plot_optimization_results(fake, top_n=2)
    assert isinstance(fig, Figure)


def test_plot_optimization_results_requires_run():
    fake = types.SimpleNamespace(optimizer=None)
    with pytest.raises(ValueError, match="Run bayesian_optimize"):
        SpectroPrepPlotter.plot_optimization_results(fake)


def test_set_style_changes_rcparams():
    SpectroPrepPlotter.set_style(font_scale=1.5)
    import matplotlib.pyplot as plt

    assert plt.rcParams["figure.dpi"] == 100


def test_plot_spectra_invert_xaxis(wavenumbers, spectra):
    fig, ax = SpectroPrepPlotter.plot_spectra(wavenumbers, spectra, invert_xaxis=True)
    assert ax.xaxis_inverted()


def test_plot_preprocessing_comparison_all_samples_invert(wavenumbers, spectra):
    processed = {"SNV": spectra * 0.5}
    fig = SpectroPrepPlotter.plot_preprocessing_comparison(
        wavenumbers,
        spectra,
        processed,
        sample_indices=None,
        invert_xaxis=True,
        xlabel="Wavelength (nm)",
    )
    assert isinstance(fig, Figure)


def test_plot_optimization_progress_smoke():
    results = [
        {"trial": 0, "pipeline_config": ("snv",), "rmse": 2.0, "r2": float("nan")},
        {"trial": 1, "pipeline_config": ("scaler",), "rmse": 1.0, "r2": float("nan")},
    ]
    fake = types.SimpleNamespace(
        optimizer=object(),
        get_all_tested_pipelines=lambda: results,
    )
    fig, ax = SpectroPrepPlotter.plot_optimization_progress(fake)
    assert isinstance(fig, Figure)


def test_plot_feature_importance(wavenumbers, rng):
    coefs = rng.normal(size=20)
    fig, ax = SpectroPrepPlotter.plot_feature_importance(
        wavenumbers, coefs, highlight_threshold=0.5
    )
    assert isinstance(fig, Figure)
