"""Time-based validation for the Favorita forecasting task.

Why this looks the way it does
------------------------------
We are forecasting 16 future days. A random k-fold split would let the
model peek at the future inside the same week it's trying to predict --
that is the single most common way teams overstate their score on this
problem. So we do **only** time-based splits:

1. `final_holdout_split`  - the last 16 days of training become a held-out
   validation set, mirroring the Kaggle test horizon exactly.

2. `rolling_window_folds` - a sequence of N consecutive 16-day folds
   stepping backwards from the end of training. Each fold's train set
   contains only data strictly before its validation start. This gives
   us multiple RMSLE samples so we can read the variance, not just one
   lucky split.

Both functions return *index arrays into the input dataframe*, so no rows
are copied and no future information can possibly leak through them.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import config


@dataclass(frozen=True)
class Fold:
    """One time-based train/validation split."""
    name: str
    train_idx: np.ndarray
    valid_idx: np.ndarray
    train_end: pd.Timestamp
    valid_start: pd.Timestamp
    valid_end: pd.Timestamp


def _train_only(df: pd.DataFrame) -> pd.DataFrame:
    """Restrict to the rows that have a real target (train half of the panel)."""
    return df[df["is_test"] == 0]


def final_holdout_split(df: pd.DataFrame, horizon: int = config.HORIZON_DAYS) -> Fold:
    """Last `horizon` days of train -> validation. Everything else -> training.

    This is the most honest single-number estimate of test performance:
    the validation window has the same length and recency as the actual
    Kaggle test window.
    """
    tr = _train_only(df)
    last_train_day = tr["date"].max()
    valid_start = last_train_day - pd.Timedelta(days=horizon - 1)

    train_mask = (df["is_test"] == 0) & (df["date"] < valid_start)
    valid_mask = (df["is_test"] == 0) & (df["date"] >= valid_start)

    # Sanity assertion: no overlap, no future leakage.
    assert df.loc[train_mask, "date"].max() < df.loc[valid_mask, "date"].min(), (
        "Train period must end strictly before validation begins."
    )

    return Fold(
        name="final_holdout",
        train_idx=np.where(train_mask)[0],
        valid_idx=np.where(valid_mask)[0],
        train_end=df.loc[train_mask, "date"].max(),
        valid_start=df.loc[valid_mask, "date"].min(),
        valid_end=df.loc[valid_mask, "date"].max(),
    )


def rolling_window_folds(
    df: pd.DataFrame,
    n_folds: int = 4,
    horizon: int = config.HORIZON_DAYS,
    step: int | None = None,
    expanding: bool = True,
) -> list[Fold]:
    """Generate N consecutive backward-stepping 16-day folds.

    expanding=True (default): training window grows over folds (uses ALL
    history up to the fold boundary). This matches how we'd retrain in
    production.
    expanding=False: rolling window of fixed length -- useful if you
    suspect old data is misleading.
    """
    step = step or horizon
    tr = _train_only(df)
    last_day = tr["date"].max()

    folds: list[Fold] = []
    for i in range(n_folds):
        valid_end = last_day - pd.Timedelta(days=i * step)
        valid_start = valid_end - pd.Timedelta(days=horizon - 1)
        train_end_exclusive = valid_start  # strict <

        train_mask = (df["is_test"] == 0) & (df["date"] < train_end_exclusive)
        if not expanding:
            # Keep the same number of days as the full training history at
            # fold 0 minus the cumulative validation budget.
            min_train_start = train_end_exclusive - pd.Timedelta(days=365 * 2)
            train_mask &= df["date"] >= min_train_start

        valid_mask = (
            (df["is_test"] == 0)
            & (df["date"] >= valid_start)
            & (df["date"] <= valid_end)
        )

        if train_mask.sum() == 0 or valid_mask.sum() == 0:
            continue

        # Verified-no-leakage assertion: every train row's date is strictly
        # less than every valid row's date.
        assert df.loc[train_mask, "date"].max() < df.loc[valid_mask, "date"].min()

        folds.append(
            Fold(
                name=f"fold_{i}_{valid_start.date()}_{valid_end.date()}",
                train_idx=np.where(train_mask)[0],
                valid_idx=np.where(valid_mask)[0],
                train_end=df.loc[train_mask, "date"].max(),
                valid_start=df.loc[valid_mask, "date"].min(),
                valid_end=df.loc[valid_mask, "date"].max(),
            )
        )

    # Return oldest-first; nicer for plotting CV scores over time.
    return list(reversed(folds))


def kaggle_test_mask(df: pd.DataFrame) -> np.ndarray:
    """Index array selecting the rows we must score for the Kaggle submission."""
    return np.where(df["is_test"] == 1)[0]
