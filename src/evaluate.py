"""Metrics, error slicing, and feature importance reporting.

Why not just RMSLE?
-------------------
RMSLE is what Kaggle scores, but it answers "how good is the model on the
log scale" -- it doesn't tell the business which families it's losing
money on or whether it's blowing the holiday weeks. The functions here
give us those views.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import config


# ---------------------------------------------------------------------------
# Core metric
# ---------------------------------------------------------------------------
def rmsle(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.clip(np.asarray(y_true, dtype=np.float64), 0, None)
    y_pred = np.clip(np.asarray(y_pred, dtype=np.float64), 0, None)
    return float(np.sqrt(np.mean((np.log1p(y_pred) - np.log1p(y_true)) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))


def revenue_weighted_error(
    y_true: np.ndarray, y_pred: np.ndarray, price_proxy: np.ndarray | None = None
) -> float:
    """MAE weighted by realised revenue.

    Without per-SKU prices we use realised sales as the weight: a unit of
    error on a high-volume series matters more than the same on a long-tail
    one. If you have a price feed, pass it as `price_proxy`.
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    w = np.asarray(price_proxy, dtype=np.float64) if price_proxy is not None else y_true
    w = np.clip(w, 0, None)
    if w.sum() == 0:
        return float("nan")
    return float(np.sum(w * np.abs(y_true - y_pred)) / w.sum())


# ---------------------------------------------------------------------------
# Error slicing
# ---------------------------------------------------------------------------
def error_table(
    df_valid: pd.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> pd.DataFrame:
    """Return a long table with one row per prediction + slicing dims."""
    out = df_valid[
        [
            "store_nbr", "family", "date",
            "nat_holiday", "reg_holiday", "loc_holiday",
            "onpromotion",
        ]
    ].copy()
    out["y_true"] = np.asarray(y_true, dtype=np.float64)
    out["y_pred"] = np.asarray(y_pred, dtype=np.float64)
    out["abs_err"] = (out["y_pred"] - out["y_true"]).abs()
    out["sq_log_err"] = (np.log1p(np.clip(out["y_pred"], 0, None))
                        - np.log1p(np.clip(out["y_true"], 0, None))) ** 2
    out["is_any_holiday"] = (
        (out["nat_holiday"] + out["reg_holiday"] + out["loc_holiday"]) > 0
    ).astype(int)
    out["is_on_promo"] = (out["onpromotion"] > 0).astype(int)
    return out


def worst_segments(err: pd.DataFrame, by: list[str], top: int = 10) -> pd.DataFrame:
    grp = err.groupby(by).agg(
        rmsle=("sq_log_err", lambda s: float(np.sqrt(s.mean()))),
        mae=("abs_err", "mean"),
        n=("abs_err", "size"),
        mean_sales=("y_true", "mean"),
    )
    return grp.sort_values("rmsle", ascending=False).head(top)


def holiday_vs_normal(err: pd.DataFrame) -> pd.DataFrame:
    """Compare RMSLE/MAE during holidays vs. non-holidays."""
    return err.groupby("is_any_holiday").agg(
        rmsle=("sq_log_err", lambda s: float(np.sqrt(s.mean()))),
        mae=("abs_err", "mean"),
        n=("abs_err", "size"),
        mean_sales=("y_true", "mean"),
    )


def promo_vs_normal(err: pd.DataFrame) -> pd.DataFrame:
    return err.groupby("is_on_promo").agg(
        rmsle=("sq_log_err", lambda s: float(np.sqrt(s.mean()))),
        mae=("abs_err", "mean"),
        n=("abs_err", "size"),
        mean_sales=("y_true", "mean"),
    )


def per_family_mae(err: pd.DataFrame) -> pd.DataFrame:
    return err.groupby("family").agg(
        mae=("abs_err", "mean"),
        rmsle=("sq_log_err", lambda s: float(np.sqrt(s.mean()))),
        n=("abs_err", "size"),
        mean_sales=("y_true", "mean"),
    ).sort_values("mae", ascending=False)


# ---------------------------------------------------------------------------
# Feature importance
# ---------------------------------------------------------------------------
def plot_feature_importance(
    fi: pd.DataFrame, out_path: Path | None = None, top: int = 30
) -> Path:
    """Bar chart of top-N features by gain. Returns the path written."""
    import matplotlib
    matplotlib.use("Agg")  # safe in headless / CI environments.
    import matplotlib.pyplot as plt

    top_fi = fi.head(top).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, max(4, 0.25 * len(top_fi))))
    ax.barh(top_fi["feature"], top_fi["gain"])
    ax.set_title(f"Top {top} features by gain")
    ax.set_xlabel("gain")
    fig.tight_layout()

    out_path = out_path or config.FEATURE_IMPORTANCE_PNG
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path
