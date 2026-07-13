"""Full Cargill corn NIR (m5) nested-CV benchmark + figures.

For each constituent (Moisture, Oil, Protein, Starch):
  * nested 5-fold SpectoPrep search (init_points=10, n_iter=25)
  * pooled OOF metrics and predicted-vs-reference figure
  * per-fold and consensus optimal preprocessing chains
  * full-spectra raw vs consensus-preprocessing plot (inverted x-axis)

Outputs land in ``examples/corn_results/`` (JSON + PNG) and key paper figures
are copied into ``vitamin_publication/figs/``.
"""

from __future__ import annotations

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


def nested_cv_constituent(X: np.ndarray, y: np.ndarray) -> dict:
    """Run nested CV; return OOF preds, metrics, and selected chains."""
    configure_logging(level="ERROR")
    outer = KFold(n_splits=5, shuffle=True, random_state=OUTER_SEED)
    oof = np.zeros(len(y), dtype=float)
    base_oof = np.zeros(len(y), dtype=float)
    fold_chains: list[str] = []

    for fold, (tr, te) in enumerate(outer.split(X), start=1):
        print(f"  outer fold {fold}/5")
        base = OptimizedRidgeCV(alphas=ALPHAS).fit(X[tr], y[tr])
        base_oof[te] = base.predict(X[te])

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
        "fold_pipelines": fold_chains,
        "consensus_pipeline": list(consensus_tuple),
        "consensus_count": consensus_count,
        "spectoprep": _metrics(y, oof),
        "raw": _metrics(y, base_oof),
    }


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


def main(corn_path: Path | None = None, out_dir: Path | None = None) -> Path:
    corn_path = Path(corn_path or os.environ.get("SPECTOPREP_CORN_MAT", "") or DEFAULT_CORN)
    out_dir = Path(out_dir or ROOT / "examples" / "corn_results")
    fig_dir = out_dir / "figures"
    paper_figs = ROOT / "vitamin_publication" / "figs"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    paper_figs.mkdir(parents=True, exist_ok=True)

    wavelengths, X, Y = load_corn_m5(corn_path)
    SpectroPrepPlotter.set_style(context="paper", font_scale=1.05)

    # Raw full spectra once
    SpectroPrepPlotter.plot_spectra(
        wavelengths,
        X,
        title="Corn NIR (m5) — raw spectra (n=80)",
        xlabel="Wavelength (nm)",
        ylabel="Absorbance",
        alpha=0.35,
        invert_xaxis=True,
        save_path=str(fig_dir / "spectra_raw_full.png"),
    )
    # Paper copy
    SpectroPrepPlotter.plot_spectra(
        wavelengths,
        X,
        title="Corn NIR (m5) — raw spectra (n=80)",
        xlabel="Wavelength (nm)",
        ylabel="Absorbance",
        alpha=0.35,
        invert_xaxis=True,
        save_path=str(paper_figs / "spectoprep_spectra_raw.png"),
    )

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
        d_rmse = 100.0 * (raw["rmse"] - spx["rmse"]) / raw["rmse"]

        print(
            f"  raw R²={raw['r2']:.3f} RMSE={raw['rmse']:.4f} | "
            f"SpectoPrep R²={spx['r2']:.3f} RMSE={spx['rmse']:.4f} | "
            f"consensus={'+'.join(consensus)}"
        )

        # Prediction scatter (all 80 OOF points)
        pred_path = fig_dir / f"prediction_{name.lower()}.png"
        SpectroPrepPlotter.plot_prediction_scatter(
            y,
            oof,
            title=f"Corn NIR (m5) — {name} (nested 5-fold OOF)",
            xlabel=f"Reference {name}",
            ylabel=f"Predicted {name}",
            save_path=str(pred_path),
        )
        # Keep protein as the paper's primary scatter
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

        # Full-spectra raw vs consensus preprocessing
        pipe = fit_prep_pipeline(X, y, consensus)
        prep = Pipeline(pipe.steps[:-1])
        X_prep = np.asarray(prep.transform(X))
        chain_label = " → ".join(consensus)
        SpectroPrepPlotter.plot_preprocessing_comparison(
            wavelengths,
            X,
            {f"Optimal preprocessing ({chain_label})": X_prep},
            sample_indices=None,  # all spectra
            title=f"Corn NIR (m5) — {name}: raw vs optimal preprocessing",
            xlabel="Wavelength (nm)",
            alpha=0.3,
            invert_xaxis=True,
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
                invert_xaxis=True,
                figsize=(12, 8),
                save_path=str(paper_figs / "spectoprep_spectra_preprocessed.png"),
            )

        results["constituents"][name] = {
            "raw": raw,
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
    main()
