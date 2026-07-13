"""Recompute nested-CV corn metrics (raw + SpectoPrep) including absolute RMSE."""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import scipy.io as sio
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold

from spectoprep import OptimizedRidgeCV, PipelineOptimizer, configure_logging

configure_logging(level="ERROR")

_ROOT = Path(__file__).resolve().parents[1]
CORN = Path(os.environ["SPECTOPREP_CORN_MAT"]) if os.environ.get("SPECTOPREP_CORN_MAT") else (
    _ROOT / "examples" / "data" / "corn.mat"
)
if not CORN.exists():
    CORN = Path("/tmp/spectoprep-data/corn.mat")
ALPHAS = np.logspace(-3, 3, 25)
STEPS = ["snv", "msc", "savgol", "detrend", "scaler", "robust_scaler", "meancn"]
PROPS = ["Moisture", "Oil", "Protein", "Starch"]


def metrics(y_true, y_pred):
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = float(r2_score(y_true, y_pred))
    rpd = float(np.std(y_true, ddof=1) / rmse)
    return r2, rmse, rpd


def main():
    d = sio.loadmat(CORN, struct_as_record=False, squeeze_me=True)
    X = np.asarray(d["m5spec"].data, dtype=float)
    Y = np.asarray(d["propvals"].data, dtype=float)
    outer = KFold(n_splits=5, shuffle=True, random_state=42)

    print(
        "Constituent | raw R2 RMSE RPD | SpectoPrep R2 RMSE RPD | dRMSE%"
    )
    rows = []
    for j, name in enumerate(PROPS):
        y = Y[:, j]
        base_oof = np.zeros(len(y))
        spx_oof = np.zeros(len(y))
        for tr, te in outer.split(X):
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
                random_state=42,
                groups=np.arange(len(tr)),
                max_pipeline_length=2,
                allowed_preprocess_combinations=[1, 2],
                log_level="ERROR",
            )
            _, pipe = opt.bayesian_optimize(init_points=10, n_iter=25)
            p, _, _ = opt.get_best_pipeline_predictions(pipe)
            spx_oof[te] = np.ravel(p)

        b_r2, b_rmse, b_rpd = metrics(y, base_oof)
        s_r2, s_rmse, s_rpd = metrics(y, spx_oof)
        d_rmse = 100.0 * (b_rmse - s_rmse) / b_rmse
        print(
            f"{name:9s} | {b_r2:.3f} {b_rmse:.4f} {b_rpd:.1f} | "
            f"{s_r2:.3f} {s_rmse:.4f} {s_rpd:.1f} | {d_rmse:.0f}%"
        )
        rows.append((name, b_r2, b_rmse, b_rpd, s_r2, s_rmse, s_rpd, d_rmse))

    print("\nLaTeX rows:")
    for name, b_r2, b_rmse, b_rpd, s_r2, s_rmse, s_rpd, d_rmse in rows:
        print(
            f"{name} & {b_r2:.3f} & {b_rmse:.3f} & {b_rpd:.1f} "
            f"& {s_r2:.3f} & {s_rmse:.3f} & {s_rpd:.1f} "
            f"& $-{d_rmse:.0f}\\%$ \\\\"
        )


if __name__ == "__main__":
    main()
