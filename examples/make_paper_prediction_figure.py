"""Generate SoftwareX predicted-vs-reference figure from nested 5-fold OOF preds.

Matches the paper table protocol (run_corn_nested.py): outer 5-fold CV, SpectoPrep
search inside each outer training fold, pooled predictions over all 80 samples.
Default constituent is Protein.
"""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import numpy as np
import scipy.io as sio
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold

from spectoprep import PipelineOptimizer, SpectroPrepPlotter, configure_logging

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

# Same search space / budget as the nested-CV paper experiment.
STEPS = ["snv", "msc", "savgol", "detrend", "scaler", "robust_scaler", "meancn"]
OUTER_SEED = 42
INIT_POINTS = 10
N_ITER = 25


def load_corn_m5(mat_path: Path):
    data = sio.loadmat(mat_path, struct_as_record=False, squeeze_me=True)
    X = np.asarray(data["m5spec"].data, dtype=float)
    Y = np.asarray(data["propvals"].data, dtype=float)
    labels = list(COMPONENTS)
    return X, Y, labels


def nested_oof_predictions(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Return SpectoPrep nested-CV out-of-fold predictions (length = n_samples)."""
    configure_logging(level="ERROR")
    outer = KFold(n_splits=5, shuffle=True, random_state=OUTER_SEED)
    oof = np.zeros(len(y), dtype=float)

    for fold, (tr, te) in enumerate(outer.split(X), start=1):
        print(f"Outer fold {fold}/5  (train={len(tr)}, test={len(te)})")
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
        chain = " + ".join(opt.summarize_optimization()["best_pipeline"])
        print(f"  selected: {chain}")

    return oof


def main(
    component: str | None = None,
    corn_path: Path | None = None,
    out_path: Path | None = None,
) -> Path:
    component = (component or os.environ.get("SPECTOPREP_COMPONENT", "Protein")).strip()
    if component not in COMPONENTS:
        raise ValueError(f"component must be one of {COMPONENTS}, got {component!r}")

    corn_path = Path(corn_path or os.environ.get("SPECTOPREP_CORN_MAT", "") or DEFAULT_CORN)
    if not corn_path.exists():
        raise FileNotFoundError(f"Corn .mat not found at {corn_path}")

    X, Y, labels = load_corn_m5(corn_path)
    # Fall back to positional index if label parsing differs
    idx = labels.index(component) if component in labels else COMPONENTS.index(component)
    y = Y[:, idx]

    oof = nested_oof_predictions(X, y)
    rmse = float(np.sqrt(mean_squared_error(y, oof)))
    r2 = float(r2_score(y, oof))
    print(f"\nPooled nested-CV {component}: RMSE={rmse:.4f}, R²={r2:.4f}")

    out_path = Path(
        out_path
        or ROOT / "vitamin_publication" / "figs" / "spectoprep_prediction_scatter.png"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    examples_copy = ROOT / "examples" / "figures" / "prediction_scatter.png"
    examples_copy.parent.mkdir(parents=True, exist_ok=True)

    title = f"Corn NIR (m5) — {component} (nested 5-fold OOF)"
    xlabel = f"Reference {component}"
    ylabel = f"Predicted {component}"

    SpectroPrepPlotter.set_style(context="paper", font_scale=1.1)
    for dest in (out_path, examples_copy):
        SpectroPrepPlotter.plot_prediction_scatter(
            y,
            oof,
            title=title,
            xlabel=xlabel,
            ylabel=ylabel,
            save_path=str(dest),
        )

    print(f"Wrote {out_path}")
    return out_path


if __name__ == "__main__":
    main()
