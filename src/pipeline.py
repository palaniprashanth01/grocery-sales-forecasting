"""End-to-end runner: data -> features -> validation -> model -> submission.

Usage from the project root:

    python -m src.pipeline

Outputs go to ./outputs/.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import config, evaluate
from .data_loader import build_panel
from .features import CATEGORICAL_COLS, build_features, feature_columns
from .models import (
    LGBMConfig,
    SeasonalNaive,
    predict_lightgbm,
    train_lightgbm,
)
from .validation import (
    Fold,
    final_holdout_split,
    kaggle_test_mask,
    rolling_window_folds,
)


def _select(df: pd.DataFrame, idx: np.ndarray, feat_cols: list[str]):
    sub = df.iloc[idx]
    return sub[feat_cols], sub["sales"].to_numpy(), sub


def run_baseline(df: pd.DataFrame, fold: Fold) -> float:
    """Seasonal-naive baseline RMSLE on the given fold. Establishes a floor."""
    train = df.iloc[fold.train_idx]
    valid = df.iloc[fold.valid_idx]
    model = SeasonalNaive(season_days=7).fit(train)
    preds = model.predict(valid)
    score = evaluate.rmsle(valid["sales"].to_numpy(), preds)
    print(f"[baseline] seasonal-naive RMSLE on {fold.name}: {score:.5f}")
    return score


def run_cv(
    df: pd.DataFrame,
    feat_cols: list[str],
    cat_cols: list[str],
    folds: list[Fold],
    lgbm_cfg: LGBMConfig,
) -> dict:
    """Backtest LightGBM across multiple time folds. Reports per-fold + mean."""
    cv_scores: list[dict] = []
    for f in folds:
        X_tr, y_tr, _ = _select(df, f.train_idx, feat_cols)
        X_va, y_va, _ = _select(df, f.valid_idx, feat_cols)
        result = train_lightgbm(X_tr, y_tr, X_va, y_va, cat_cols, lgbm_cfg)
        print(
            f"[cv] {f.name}  best_iter={result.best_iteration}  "
            f"valid RMSLE={result.val_rmsle:.5f}"
        )
        cv_scores.append(
            {
                "fold": f.name,
                "best_iter": result.best_iteration,
                "rmsle": result.val_rmsle,
                "train_end": str(f.train_end.date()),
                "valid_start": str(f.valid_start.date()),
                "valid_end": str(f.valid_end.date()),
            }
        )
    mean_rmsle = float(np.mean([s["rmsle"] for s in cv_scores]))
    std_rmsle = float(np.std([s["rmsle"] for s in cv_scores]))
    print(f"[cv] mean RMSLE = {mean_rmsle:.5f}  +/- {std_rmsle:.5f}")
    return {"folds": cv_scores, "mean": mean_rmsle, "std": std_rmsle}


def fit_final(
    df: pd.DataFrame,
    feat_cols: list[str],
    cat_cols: list[str],
    holdout: Fold,
    lgbm_cfg: LGBMConfig,
):
    """Final fit + holdout error analysis.

    We train on data up to (train_end - HORIZON), validate on the last 16
    days, and use the best_iteration from there to size the final-fit
    boosting rounds.
    """
    X_tr, y_tr, _ = _select(df, holdout.train_idx, feat_cols)
    X_ho, y_ho, valid_df = _select(df, holdout.valid_idx, feat_cols)
    result = train_lightgbm(X_tr, y_tr, X_ho, y_ho, cat_cols, lgbm_cfg)

    yp_ho = predict_lightgbm(result, X_ho)
    print(
        f"[final-holdout] RMSLE={result.val_rmsle:.5f}  "
        f"MAE={evaluate.mae(y_ho, yp_ho):.3f}  "
        f"RevMAE={evaluate.revenue_weighted_error(y_ho, yp_ho):.3f}"
    )

    err = evaluate.error_table(valid_df, y_ho, yp_ho)
    print("\n[error] holiday vs normal:")
    print(evaluate.holiday_vs_normal(err))
    print("\n[error] promo vs normal:")
    print(evaluate.promo_vs_normal(err))
    print("\n[error] top-10 worst families (by RMSLE):")
    print(evaluate.worst_segments(err, ["family"]))
    print("\n[error] top-10 worst stores (by RMSLE):")
    print(evaluate.worst_segments(err, ["store_nbr"]))

    fi_path = evaluate.plot_feature_importance(result.feature_importance)
    print(f"[plot] feature importance -> {fi_path}")

    return result, err


def predict_kaggle(
    df: pd.DataFrame,
    feat_cols: list[str],
    cat_cols: list[str],
    final_n_estimators: int,
    lgbm_cfg: LGBMConfig,
) -> pd.DataFrame:
    """Refit on ALL training data, predict the Kaggle test window, write CSV."""
    train_mask = df["is_test"] == 0
    test_mask = df["is_test"] == 1
    X_tr = df.loc[train_mask, feat_cols]
    y_tr = df.loc[train_mask, "sales"].to_numpy()
    X_te = df.loc[test_mask, feat_cols]

    final_cfg = LGBMConfig(**{**lgbm_cfg.__dict__, "num_boost_round": final_n_estimators,
                              "early_stopping_rounds": 0})
    result = train_lightgbm(X_tr, y_tr, None, None, cat_cols, final_cfg)
    preds = predict_lightgbm(result, X_te)

    submission = df.loc[test_mask, ["id"]].copy()
    submission["sales"] = preds
    submission = submission.sort_values("id").reset_index(drop=True)
    config.OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    submission.to_csv(config.SUBMISSION_CSV, index=False)
    print(f"[submission] wrote {len(submission)} rows -> {config.SUBMISSION_CSV}")
    return submission


def main(n_folds: int = 4, run_kaggle: bool = True) -> None:
    config.set_seeds()

    print("[load] reading + merging tables ...")
    panel = build_panel()
    print(f"[load] panel shape: {panel.shape}")

    print("[features] building features ...")
    df = build_features(panel)
    feat_cols = feature_columns(df)
    print(f"[features] {len(feat_cols)} features")

    # 1. Baseline RMSLE on the final holdout -- floor we must beat.
    holdout = final_holdout_split(df)
    baseline = run_baseline(df, holdout)

    # 2. Time-based CV to estimate variance and right-size the model.
    folds = rolling_window_folds(df, n_folds=n_folds)
    lgbm_cfg = LGBMConfig()
    cv = run_cv(df, feat_cols, CATEGORICAL_COLS, folds, lgbm_cfg)

    # 3. Final fit + holdout error analysis, with feature importance.
    result, _ = fit_final(df, feat_cols, CATEGORICAL_COLS, holdout, lgbm_cfg)

    # 4. Refit on all train data, predict the Kaggle test window.
    if run_kaggle:
        predict_kaggle(
            df, feat_cols, CATEGORICAL_COLS,
            final_n_estimators=result.best_iteration,
            lgbm_cfg=lgbm_cfg,
        )

    summary = {
        "baseline_seasonal_naive_rmsle": baseline,
        "cv": cv,
        "final_holdout_rmsle": result.val_rmsle,
        "final_holdout_best_iter": result.best_iteration,
    }
    summary_path = config.OUTPUTS_DIR / "run_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"[summary] {summary_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--folds", type=int, default=4)
    p.add_argument("--no-kaggle", action="store_true",
                   help="skip Kaggle submission refit/predict")
    args = p.parse_args()
    main(n_folds=args.folds, run_kaggle=not args.no_kaggle)
