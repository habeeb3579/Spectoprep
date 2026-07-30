"""Full Cargill corn NIR (m5) nested-CV benchmark + figures.

For each constituent (Moisture, Oil, Protein, Starch):
  * nested 5-fold SpectoPrep search (init_points=10, n_iter=25)
  * pooled OOF metrics and predicted-vs-reference figure
  * per-fold and consensus optimal preprocessing chains
  * full-spectra raw vs consensus-preprocessing plots

Outputs land in ``examples/corn_results/`` (JSON + PNG) and key paper figures
are copied into ``vitamin_publication/figs/``.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import numpy as np
import scipy.io as sio
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline

from spectoprep import OptimizedRidgeCV, PipelineOptimizer, SpectroPrepPlotter, configure_logging
from spectoprep.pipeline.builder import build_preprocessor_from_bayes
from spectoprep.pipeline.config import DEFAULT_PARAM_BOUNDS
from spectoprep.preprocessing.scatter import MultiplicativeScatterCorrection, StandardNormalVariate
from spectoprep.preprocessing.smoothing import SavitzkyGolay

COMPONENTS = ("Moisture", "Oil", "Protein", "Starch")
ROOT = Path(__file__).resolve().parents[1]


def _default_corn_path() -> Path:
    env = os.environ.get("SPECTOPREP_CORN_MAT")
    candidates = []
    if env:
        candidates.append(Path(env))
    candidates.extend(
        [
            ROOT / "examples" / "data" / "corn.mat",
            Path("/tmp/spectoprep-data/corn.mat"),
        ]
    )
    for path in candidates:
        if path.exists():
            return path
    return candidates[0]


DEFAULT_CORN = _default_corn_path()
STEPS = ["snv", "msc", "savgol", "detrend", "scaler", "robust_scaler", "meancn"]
ALPHAS = np.logspace(-3, 3, 25)
OUTER_SEED = 42
INIT_POINTS = 10
N_ITER = 25
FIXED_BASELINES = {
    "snv": [StandardNormalVariate()],
    "msc": [MultiplicativeScatterCorrection()],
    "snv_sg1": [StandardNormalVariate(), SavitzkyGolay(filter_win=11, poly_order=2, deriv_order=1)],
    "msc_sg1": [
        MultiplicativeScatterCorrection(),
        SavitzkyGolay(filter_win=11, poly_order=2, deriv_order=1),
    ],
}
BASELINE_LABELS = {
    "snv": "SNV",
    "msc": "MSC",
    "snv_sg1": "SNV+SG$_1$",
    "msc_sg1": "MSC+SG$_1$",
}
SPECTOPREP_PAPER_PIPELINES = {
    "SavGol + scaler (SpectoPrep)": ["savgol", "scaler"],
    "Scaler (SpectoPrep)": ["scaler"],
}


def load_corn_m5(mat_path: Path):
    data = sio.loadmat(mat_path, struct_as_record=False, squeeze_me=True)
    X = np.asarray(data["m5spec"].data, dtype=float)
    Y = np.asarray(data["propvals"].data, dtype=float)
    # Eigenvector axisscale: wavelength axis is typically [1]
    try:
        wavelengths = np.asarray(data["m5spec"].axisscale[1], dtype=float).ravel()
        if wavelengths.size != X.shape[1]:
            wavelengths = np.linspace(1100, 2498, X.shape[1])
    except Exception:
        wavelengths = np.linspace(1100, 2498, X.shape[1])
    return wavelengths, X, Y


def _metrics(y_true, y_pred):
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = float(r2_score(y_true, y_pred))
    rpd = float(np.std(y_true, ddof=1) / rmse)
    return {"r2": r2, "rmse": rmse, "rpd": rpd}


def _fit_fixed_baseline(train_X: np.ndarray, train_y: np.ndarray, transforms: list[object]) -> Pipeline:
    steps = [(f"prep_{i}", transform) for i, transform in enumerate(transforms, start=1)]
    steps.append(("ridge", OptimizedRidgeCV(alphas=ALPHAS)))
    pipe = Pipeline(steps)
    pipe.fit(train_X, train_y)
    return pipe


def _fixed_baseline_oof(
    X: np.ndarray, y: np.ndarray, transforms: list[object], outer: KFold
) -> np.ndarray:
    oof = np.zeros(len(y), dtype=float)
    for tr, te in outer.split(X):
        pipe = _fit_fixed_baseline(X[tr], y[tr], transforms)
        oof[te] = np.ravel(pipe.predict(X[te]))
    return oof


def nested_cv_constituent(X: np.ndarray, y: np.ndarray) -> dict:
    """Run nested CV; return OOF preds, metrics, and selected chains."""
    configure_logging(level="ERROR")
    outer = KFold(n_splits=5, shuffle=True, random_state=OUTER_SEED)
    oof = np.zeros(len(y), dtype=float)
    base_oof = np.zeros(len(y), dtype=float)
    fixed_oof = {name: np.zeros(len(y), dtype=float) for name in FIXED_BASELINES}
    fold_chains: list[str] = []

    for fold, (tr, te) in enumerate(outer.split(X), start=1):
        print(f"  outer fold {fold}/5")
        base = OptimizedRidgeCV(alphas=ALPHAS).fit(X[tr], y[tr])
        base_oof[te] = base.predict(X[te])
        for baseline_name, transforms in FIXED_BASELINES.items():
            fixed_pipe = _fit_fixed_baseline(X[tr], y[tr], transforms)
            fixed_oof[baseline_name][te] = np.ravel(fixed_pipe.predict(X[te]))

        opt = PipelineOptimizer(
            X_train=X[tr],
            y_train=y[tr],
            X_test=X[te],
            y_test=y[te],
            preprocessing_steps=STEPS,
            cv_method="group_kfold",
            n_splits=3,
            random_state=OUTER_SEED,
            groups=np.arange(len(tr)),
            max_pipeline_length=2,
            allowed_preprocess_combinations=[1, 2],
            log_level="ERROR",
        )
        _, pipe = opt.bayesian_optimize(init_points=INIT_POINTS, n_iter=N_ITER)
        preds, _, _ = opt.get_best_pipeline_predictions(pipe)
        oof[te] = np.ravel(preds)
        chain = list(opt.summarize_optimization()["best_pipeline"])
        fold_chains.append(chain)
        print(f"    selected: {' + '.join(chain)}")

    consensus_tuple, consensus_count = Counter(tuple(c) for c in fold_chains).most_common(1)[0]
    return {
        "oof_preds": oof,
        "base_oof": base_oof,
        "fixed_oof": fixed_oof,
        "fold_pipelines": fold_chains,
        "consensus_pipeline": list(consensus_tuple),
        "consensus_count": consensus_count,
        "spectoprep": _metrics(y, oof),
        "raw": _metrics(y, base_oof),
        "fixed_baselines": {name: _metrics(y, preds) for name, preds in fixed_oof.items()},
    }


def _default_bayes_params() -> dict[str, float]:
    params: dict[str, float] = {}
    for key, (lo, hi) in DEFAULT_PARAM_BOUNDS.items():
        if key.endswith("_method") or key.startswith("rs_with_") or key == "power_standardize":
            params[key] = 1.0 if hi == 1 else float(lo)
        elif key in {"kpca_kernel", "fa_linkage", "quantile_dist", "power_method"}:
            params[key] = float(lo)
        else:
            params[key] = (lo + hi) / 2.0
    return params


def _consensus_preprocess_pipeline(steps: list[str], n_samples: int, n_features: int) -> Pipeline:
    """Build a consensus preprocessing pipeline for visualization (no BO re-fit)."""
    params = _default_bayes_params()
    transformers = [
        (
            step,
            build_preprocessor_from_bayes(
                step, params, (n_samples, n_features), OUTER_SEED, n_jobs=1
            ),
        )
        for step in steps
    ]
    return Pipeline(transformers)


def _transform_fixed_spectra(X: np.ndarray, transforms: list[object]) -> np.ndarray:
    prep = Pipeline([(f"prep_{i}", transform) for i, transform in enumerate(transforms, start=1)])
    return np.asarray(prep.fit_transform(X))


def _transform_spectoprep_pipeline(X: np.ndarray, steps: list[str]) -> np.ndarray:
    prep = _consensus_preprocess_pipeline(steps, X.shape[0], X.shape[1])
    return np.asarray(prep.fit_transform(X))


def _plot_raw_spectra(wavelengths: np.ndarray, X: np.ndarray, fig_dir: Path, paper_figs: Path) -> None:
    for path, title in (
        (fig_dir / "spectra_raw_full.png", "Corn NIR (m5) — raw spectra (n=80)"),
        (paper_figs / "spectoprep_spectra_raw.png", "Corn NIR (m5) — raw spectra (n=80)"),
    ):
        SpectroPrepPlotter.plot_spectra(
            wavelengths,
            X,
            title=title,
            xlabel="Wavelength (nm)",
            ylabel="Absorbance",
            alpha=0.35,
            save_path=str(path),
        )


def _plot_paper_preprocessing_grid(
    wavelengths: np.ndarray, X: np.ndarray, fig_dir: Path, paper_figs: Path
) -> None:
    panels: list[tuple[str, np.ndarray]] = [("Raw", X)]
    for name, transforms in FIXED_BASELINES.items():
        panels.append((BASELINE_LABELS[name], _transform_fixed_spectra(X, transforms)))
    for label, steps in SPECTOPREP_PAPER_PIPELINES.items():
        panels.append((label, _transform_spectoprep_pipeline(X, steps)))

    kwargs = dict(
        ncols=2,
        figsize=(7, 9),
        suptitle="Corn NIR (m5) — preprocessing comparison",
        xlabel="Wavelength (nm)",
        alpha=0.3,
        xticks=[1100, 1500, 1900, 2300, 2500],
    )
    SpectroPrepPlotter.plot_preprocessing_grid(
        wavelengths,
        panels,
        save_path=str(fig_dir / "spectra_prep_paper_grid.png"),
        **kwargs,
    )
    SpectroPrepPlotter.plot_preprocessing_grid(
        wavelengths,
        panels,
        save_path=str(paper_figs / "spectoprep_spectra_preprocessing_grid.png"),
        **kwargs,
    )


def _plot_fixed_baseline_spectra(wavelengths: np.ndarray, X: np.ndarray, fig_dir: Path) -> None:
    processed = {
        BASELINE_LABELS[name]: _transform_fixed_spectra(X, transforms)
        for name, transforms in FIXED_BASELINES.items()
    }
    SpectroPrepPlotter.plot_preprocessing_comparison(
        wavelengths,
        X,
        processed,
        sample_indices=None,
        title="Corn NIR (m5) — raw vs fixed preprocessing baselines",
        xlabel="Wavelength (nm)",
        alpha=0.3,
        figsize=(12, 14),
        save_path=str(fig_dir / "spectra_prep_fixed_baselines.png"),
    )
    for name, transforms in FIXED_BASELINES.items():
        label = BASELINE_LABELS[name]
        SpectroPrepPlotter.plot_preprocessing_comparison(
            wavelengths,
            X,
            {label: _transform_fixed_spectra(X, transforms)},
            sample_indices=None,
            title=f"Corn NIR (m5) — raw vs {label}",
            xlabel="Wavelength (nm)",
            alpha=0.3,
            figsize=(12, 8),
            save_path=str(fig_dir / f"spectra_prep_fixed_{name}.png"),
        )


def _plot_constituent_figures(
    name: str,
    y: np.ndarray,
    oof: np.ndarray,
    fixed_oof: dict[str, np.ndarray],
    consensus: list[str],
    X: np.ndarray,
    wavelengths: np.ndarray,
    fig_dir: Path,
    paper_figs: Path,
    *,
    refit_consensus: bool,
) -> None:
    pred_path = fig_dir / f"prediction_{name.lower()}.png"
    SpectroPrepPlotter.plot_prediction_scatter(
        y,
        oof,
        title=f"Corn NIR (m5) — {name} (nested 5-fold OOF)",
        xlabel=f"Reference {name}",
        ylabel=f"Predicted {name}",
        save_path=str(pred_path),
    )
    if name == "Protein":
        SpectroPrepPlotter.plot_prediction_scatter(
            y,
            oof,
            title=f"Corn NIR (m5) — {name} (nested 5-fold OOF)",
            xlabel=f"Reference {name}",
            ylabel=f"Predicted {name}",
            save_path=str(paper_figs / "spectoprep_prediction_scatter.png"),
        )
    SpectroPrepPlotter.plot_prediction_scatter(
        y,
        oof,
        title=f"Corn NIR (m5) — {name} (nested 5-fold OOF)",
        xlabel=f"Reference {name}",
        ylabel=f"Predicted {name}",
        save_path=str(paper_figs / f"spectoprep_prediction_{name.lower()}.png"),
    )

    for baseline_name, preds in fixed_oof.items():
        label = BASELINE_LABELS[baseline_name]
        SpectroPrepPlotter.plot_prediction_scatter(
            y,
            preds,
            title=f"Corn NIR (m5) — {name}: {label} (nested 5-fold OOF)",
            xlabel=f"Reference {name}",
            ylabel=f"Predicted {name}",
            save_path=str(fig_dir / f"prediction_{name.lower()}_{baseline_name}.png"),
        )

    if refit_consensus:
        pipe = fit_prep_pipeline(X, y, consensus)
        prep = Pipeline(pipe.steps[:-1])
        X_prep = np.asarray(prep.transform(X))
    else:
        prep = _consensus_preprocess_pipeline(consensus, X.shape[0], X.shape[1])
        X_prep = np.asarray(prep.fit_transform(X))
    chain_label = " → ".join(consensus)
    SpectroPrepPlotter.plot_preprocessing_comparison(
        wavelengths,
        X,
        {f"Optimal preprocessing ({chain_label})": X_prep},
        sample_indices=None,
        title=f"Corn NIR (m5) — {name}: raw vs optimal preprocessing",
        xlabel="Wavelength (nm)",
        alpha=0.3,
        figsize=(12, 8),
        save_path=str(fig_dir / f"spectra_prep_{name.lower()}.png"),
    )
    if name == "Protein":
        SpectroPrepPlotter.plot_preprocessing_comparison(
            wavelengths,
            X,
            {f"Optimal preprocessing ({chain_label})": X_prep},
            sample_indices=None,
            title=f"Corn NIR (m5) — {name}: raw vs optimal preprocessing",
            xlabel="Wavelength (nm)",
            alpha=0.3,
            figsize=(12, 8),
            save_path=str(paper_figs / "spectoprep_spectra_preprocessed.png"),
        )


def fit_prep_pipeline(X: np.ndarray, y: np.ndarray, steps: list[str]) -> Pipeline:
    """Fit a pipeline whose structure matches ``steps`` (hyperparams re-tuned)."""
    length = len(steps)
    opt = PipelineOptimizer(
        X_train=X,
        y_train=y,
        groups=np.arange(len(y)),
        preprocessing_steps=list(steps),
        cv_method="group_kfold",
        n_splits=5,
        random_state=OUTER_SEED,
        max_pipeline_length=length,
        allowed_preprocess_combinations=[length],
        log_level="ERROR",
    )
    _, pipe = opt.bayesian_optimize(init_points=INIT_POINTS, n_iter=N_ITER)
    return pipe


def figures_only(corn_path: Path | None = None, out_dir: Path | None = None) -> Path:
    """Regenerate PNG figures from ``full_results.json`` without re-running nested CV."""
    corn_path = Path(corn_path or os.environ.get("SPECTOPREP_CORN_MAT", "") or DEFAULT_CORN)
    out_dir = Path(out_dir or ROOT / "examples" / "corn_results")
    fig_dir = out_dir / "figures"
    paper_figs = ROOT / "vitamin_publication" / "figs"
    results_path = out_dir / "full_results.json"
    if not results_path.exists():
        raise FileNotFoundError(f"Missing results file: {results_path}")

    results = json.loads(results_path.read_text())
    wavelengths, X, Y = load_corn_m5(corn_path)
    SpectroPrepPlotter.set_style(context="paper", font_scale=1.05)
    _plot_raw_spectra(wavelengths, X, fig_dir, paper_figs)
    _plot_fixed_baseline_spectra(wavelengths, X, fig_dir)
    _plot_paper_preprocessing_grid(wavelengths, X, fig_dir, paper_figs)

    for j, name in enumerate(COMPONENTS):
        data = results["constituents"][name]
        y = Y[:, j]
        fixed_oof = {
            baseline: np.asarray(data["fixed_oof_predictions"][baseline], dtype=float)
            for baseline in FIXED_BASELINES
        }
        _plot_constituent_figures(
            name,
            y,
            np.asarray(data["oof_predictions"], dtype=float),
            fixed_oof,
            data["consensus_pipeline"],
            X,
            wavelengths,
            fig_dir,
            paper_figs,
            refit_consensus=False,
        )

    print(f"\nRegenerated figures in {fig_dir}")
    return out_dir


def main(
    corn_path: Path | None = None,
    out_dir: Path | None = None,
    *,
    figures_only_mode: bool = False,
) -> Path:
    if figures_only_mode:
        return figures_only(corn_path=corn_path, out_dir=out_dir)
    corn_path = Path(corn_path or os.environ.get("SPECTOPREP_CORN_MAT", "") or DEFAULT_CORN)
    out_dir = Path(out_dir or ROOT / "examples" / "corn_results")
    fig_dir = out_dir / "figures"
    paper_figs = ROOT / "vitamin_publication" / "figs"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    paper_figs.mkdir(parents=True, exist_ok=True)

    wavelengths, X, Y = load_corn_m5(corn_path)
    SpectroPrepPlotter.set_style(context="paper", font_scale=1.05)

    _plot_raw_spectra(wavelengths, X, fig_dir, paper_figs)
    _plot_fixed_baseline_spectra(wavelengths, X, fig_dir)
    _plot_paper_preprocessing_grid(wavelengths, X, fig_dir, paper_figs)

    results: dict = {
        "protocol": {
            "instrument": "m5",
            "outer_cv": "KFold(n_splits=5, shuffle=True, random_state=42)",
            "inner_cv": "group_kfold n_splits=3",
            "init_points": INIT_POINTS,
            "n_iter": N_ITER,
            "evaluations_per_outer_fold": INIT_POINTS + N_ITER,
            "preprocessing_steps": STEPS,
        },
        "constituents": {},
    }

    for j, name in enumerate(COMPONENTS):
        print(f"\n=== {name} ===")
        y = Y[:, j]
        run = nested_cv_constituent(X, y)
        oof = run["oof_preds"]
        consensus = run["consensus_pipeline"]
        spx = run["spectoprep"]
        raw = run["raw"]
        fixed_baselines = run["fixed_baselines"]
        d_rmse = 100.0 * (raw["rmse"] - spx["rmse"]) / raw["rmse"]

        print(
            f"  raw R²={raw['r2']:.3f} RMSE={raw['rmse']:.4f} | "
            f"SpectoPrep R²={spx['r2']:.3f} RMSE={spx['rmse']:.4f} | "
            f"consensus={'+'.join(consensus)}"
        )
        for baseline_name, metrics in fixed_baselines.items():
            print(
                f"    {baseline_name:8s} R²={metrics['r2']:.3f} "
                f"RMSE={metrics['rmse']:.4f} RPD={metrics['rpd']:.3f}"
            )

        _plot_constituent_figures(
            name,
            y,
            oof,
            run["fixed_oof"],
            consensus,
            X,
            wavelengths,
            fig_dir,
            paper_figs,
            refit_consensus=True,
        )

        results["constituents"][name] = {
            "raw": raw,
            "fixed_baselines": fixed_baselines,
            "spectoprep": spx,
            "delta_rmse_pct": d_rmse,
            "fold_pipelines": run["fold_pipelines"],
            "consensus_pipeline": consensus,
            "consensus_count": run["consensus_count"],
            "consensus_note": (
                f"selected in {run['consensus_count']}/5 outer folds"
                if run["consensus_count"] > 1
                else "no majority across folds; see fold_pipelines"
            ),
            "fixed_oof_predictions": {k: v.tolist() for k, v in run["fixed_oof"].items()},
            "oof_predictions": oof.tolist(),
            "references": y.tolist(),
        }

    # Persist tracking of optimal preprocessing + metrics
    # Drop bulky arrays from a slim summary too
    summary = {
        "protocol": results["protocol"],
        "constituents": {
            name: {
                "raw": data["raw"],
                "fixed_baselines": data["fixed_baselines"],
                "spectoprep": data["spectoprep"],
                "delta_rmse_pct": data["delta_rmse_pct"],
                "fold_pipelines": data["fold_pipelines"],
                "consensus_pipeline": data["consensus_pipeline"],
                "consensus_count": data["consensus_count"],
                "consensus_note": data["consensus_note"],
            }
            for name, data in results["constituents"].items()
        },
    }
    (out_dir / "optimal_pipelines.json").write_text(json.dumps(summary, indent=2))
    (out_dir / "full_results.json").write_text(json.dumps(results, indent=2))

    print("\n=== Consensus optimal preprocessing ===")
    for name, data in summary["constituents"].items():
        print(f"{name:9s}: {' → '.join(data['consensus_pipeline'])}")
        print(f"           folds: {data['fold_pipelines']}")

    print(f"\nWrote results to {out_dir}")
    return out_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--figures-only",
        action="store_true",
        help="Regenerate PNG figures from existing full_results.json (skip nested CV).",
    )
    parser.add_argument("--corn-path", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args()
    main(corn_path=args.corn_path, out_dir=args.out_dir, figures_only_mode=args.figures_only)
