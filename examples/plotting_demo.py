"""Short SpectroPrepPlotter demo used by docs and the SoftwareX figure.

Uses a held-out test split so predicted-versus-reference is a real test-set
scatter (not training-set or misaligned fold order).
"""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import numpy as np
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline

from spectoprep import PipelineOptimizer, SpectroPrepPlotter


def make_synthetic(n_samples: int = 80, n_wavelengths: int = 120, seed: int = 21):
    rng = np.random.default_rng(seed)
    wavelengths = np.linspace(1100, 2500, n_wavelengths)
    baseline = 0.2 + 0.001 * (wavelengths - wavelengths.mean())
    peaks = (
        0.8 * np.exp(-0.5 * ((wavelengths - 1450) / 40) ** 2)
        + 0.5 * np.exp(-0.5 * ((wavelengths - 1900) / 55) ** 2)
    )
    concentrations = rng.uniform(0.5, 1.5, size=n_samples)
    scatter = rng.normal(1.0, 0.05, size=(n_samples, 1))
    X = scatter * (baseline + np.outer(concentrations, peaks))
    X += rng.normal(0.0, 0.01, size=X.shape)
    y = concentrations + rng.normal(0.0, 0.03, size=n_samples)
    groups = np.arange(n_samples)
    return wavelengths, X, y, groups


def main(figdir: Path | None = None) -> Path:
    figdir = Path(figdir or os.environ.get("SPECTOPREP_FIGDIR", "examples/figures"))
    figdir.mkdir(parents=True, exist_ok=True)

    wavelengths, X, y, groups = make_synthetic()
    gss = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=21)
    train_idx, test_idx = next(gss.split(X, y, groups))
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    groups_train = groups[train_idx]

    SpectroPrepPlotter.set_style(context="paper", font_scale=1.1)
    SpectroPrepPlotter.plot_spectra(
        wavelengths,
        X_train[:12],
        title="Synthetic NIR-like spectra (subset)",
        xlabel="Wavelength (nm)",
        ylabel="Absorbance (a.u.)",
        save_path=str(figdir / "spectra_raw.png"),
    )

    optimizer = PipelineOptimizer(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        groups=groups_train,
        preprocessing_steps=["msc", "savgol", "snv", "scaler", "detrend"],
        cv_method="group_kfold",
        n_splits=4,
        max_pipeline_length=2,
        allowed_preprocess_combinations=[1, 2],
        random_state=21,
        log_level="WARNING",
    )
    _, best_pipeline = optimizer.bayesian_optimize(init_points=8, n_iter=25)

    prep = Pipeline(best_pipeline.steps[:-1])
    X_prep = prep.fit_transform(X_train)
    SpectroPrepPlotter.plot_preprocessing_comparison(
        wavelengths,
        X_train,
        {"Best pipeline": np.asarray(X_prep)},
        sample_indices=[0, 1, 2],
        title="Raw vs. optimised preprocessing",
        save_path=str(figdir / "spectra_preprocessed.png"),
    )

    # Held-out predictions: y_test and preds are the same length and order.
    preds, rmse, r2 = optimizer.get_best_pipeline_predictions(best_pipeline)
    assert preds.shape == y_test.shape
    SpectroPrepPlotter.plot_prediction_scatter(
        y_test,
        preds,
        title="Predicted vs. reference (held-out test set)",
        xlabel="Reference",
        ylabel="Predicted",
        save_path=str(figdir / "prediction_scatter.png"),
    )

    SpectroPrepPlotter.plot_optimization_progress(
        optimizer,
        title="Bayesian search progress",
        save_path=str(figdir / "optimization_progress.png"),
    )
    SpectroPrepPlotter.plot_optimization_results(
        optimizer,
        top_n=5,
        title="Top pipelines by CV RMSE",
        save_path=str(figdir / "optimization_results.png"),
    )

    summary = optimizer.summarize_optimization()
    print(f"Wrote figures to {figdir.resolve()}")
    print(f"Best pipeline: {summary['best_pipeline']}")
    print(f"Held-out RMSE={rmse:.4f}, R²={r2:.4f}")
    return figdir


if __name__ == "__main__":
    main()
