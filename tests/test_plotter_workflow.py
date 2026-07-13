"""End-to-end smoke: short optimize then SpectroPrepPlotter outputs."""

from __future__ import annotations

import numpy as np
import pytest
from matplotlib.figure import Figure
from sklearn.pipeline import Pipeline

from spectoprep import PipelineOptimizer, SpectroPrepPlotter


@pytest.mark.slow
def test_plotter_after_short_optimize(tmp_path):
    rng = np.random.default_rng(0)
    n, p = 30, 40
    wavelengths = np.linspace(1100, 1800, p)
    X = rng.normal(size=(n, p))
    y = X[:, :3].sum(axis=1) + rng.normal(scale=0.1, size=n)
    groups = np.arange(n)

    optimizer = PipelineOptimizer(
        X_train=X,
        y_train=y,
        groups=groups,
        preprocessing_steps=["snv", "scaler", "savgol"],
        cv_method="group_kfold",
        n_splits=3,
        max_pipeline_length=1,
        allowed_preprocess_combinations=[1],
        random_state=0,
        log_level="ERROR",
    )
    _, best_pipeline = optimizer.bayesian_optimize(init_points=2, n_iter=3)
    preds, _, _ = optimizer.get_best_pipeline_predictions(best_pipeline)

    SpectroPrepPlotter.set_style()
    fig1, _ = SpectroPrepPlotter.plot_spectra(
        wavelengths, X[:5], save_path=str(tmp_path / "spectra.png")
    )
    assert isinstance(fig1, Figure)
    assert (tmp_path / "spectra.png").exists()

    prep = Pipeline(best_pipeline.steps[:-1])
    fig2 = SpectroPrepPlotter.plot_preprocessing_comparison(
        wavelengths,
        X,
        {"prep": np.asarray(prep.transform(X))},
        sample_indices=[0, 1],
        save_path=str(tmp_path / "prep.png"),
    )
    assert isinstance(fig2, Figure)

    fig3, _ = SpectroPrepPlotter.plot_prediction_scatter(
        y, preds, save_path=str(tmp_path / "scatter.png")
    )
    assert isinstance(fig3, Figure)

    fig4, _ = SpectroPrepPlotter.plot_optimization_progress(optimizer)
    assert isinstance(fig4, Figure)
    fig5 = SpectroPrepPlotter.plot_optimization_results(optimizer, top_n=3)
    assert isinstance(fig5, Figure)
