"""Feature engineering for the Favorita panel.

LEAKAGE NOTE
============
The Kaggle test set is the 16 days following the train end. So at prediction
time we know every column EXCEPT the target (`sales`) for the last 16 days.

Any feature that depends on `sales` must therefore use a lag of at least 16
days — otherwise we'd need to recursively predict in order to fill it in.
Every sales-derived feature in this module is shifted by `MIN_LAG` (=16).
Calendar/static features (holidays, oil, store metadata, payday flags) are
known in advance and need no shift.

The same builder runs over train + test in one pass: features for a test
date use only data strictly before that date.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config


# ---------------------------------------------------------------------------
# Calendar features
# ---------------------------------------------------------------------------
def _add_calendar(df: pd.DataFrame) -> pd.DataFrame:
    d = df["date"].dt
    df["dow"] = d.dayofweek.astype("int8")
    df["day"] = d.day.astype("int8")
    df["month"] = d.month.astype("int8")
    df["year"] = d.year.astype("int16")
    df["weekofyear"] = d.isocalendar().week.astype("int8")
    df["quarter"] = d.quarter.astype("int8")
    df["is_weekend"] = (d.dayofweek >= 5).astype("int8")

    # Ecuadorian paydays: public sector is paid on the 15th and the last day
    # of the month — both trigger visible grocery spend spikes.
    last_day = d.days_in_month
    df["is_payday"] = ((d.day == 15) | (d.day == last_day)).astype("int8")
    # Day-after-payday also bumps; let the model decide if it cares.
    df["is_post_payday"] = ((d.day == 16) | (d.day == 1)).astype("int8")
    return df


def _add_earthquake_flag(df: pd.DataFrame) -> pd.DataFrame:
    """Flag the window after the Apr-16 2016 earthquake.

    Grocery/water/first-aid sales spiked sharply for ~30 days as relief
    supplies were purchased. Without this flag the model treats it as a
    permanent regime shift and over-forecasts the same week in 2017.
    """
    eq = pd.Timestamp(config.EARTHQUAKE_DATE)
    in_window = (df["date"] >= eq) & (
        df["date"] <= eq + pd.Timedelta(days=config.EARTHQUAKE_WINDOW_DAYS)
    )
    df["earthquake_window"] = in_window.astype("int8")
    return df


# ---------------------------------------------------------------------------
# Target-derived features — must lag by >= MIN_LAG
# ---------------------------------------------------------------------------
def _add_lags(df: pd.DataFrame, group_cols=("store_nbr", "family")) -> pd.DataFrame:
    """Sales lags. Each is sales SHIFTed by k days within (store, family)."""
    g = df.groupby(list(group_cols), observed=True)["sales"]
    for k in config.LAGS:
        # By construction k >= MIN_LAG, so this is leakage-safe.
        df[f"sales_lag_{k}"] = g.shift(k).astype("float32")
    return df


def _add_rolling(df: pd.DataFrame, group_cols=("store_nbr", "family")) -> pd.DataFrame:
    """Rolling stats over `window` days, ending at t - MIN_LAG.

    Implementation: shift the series by MIN_LAG first, THEN roll. That
    guarantees the window at time t uses only data from before t-MIN_LAG.
    """
    grouped = df.groupby(list(group_cols), observed=True)["sales"]
    shifted = grouped.shift(config.MIN_LAG)
    for w in config.ROLLING_WINDOWS:
        df[f"sales_roll_mean_{w}"] = (
            shifted.groupby([df["store_nbr"], df["family"]], observed=True)
            .rolling(w, min_periods=1)
            .mean()
            .reset_index(level=[0, 1], drop=True)
            .astype("float32")
        )
        df[f"sales_roll_std_{w}"] = (
            shifted.groupby([df["store_nbr"], df["family"]], observed=True)
            .rolling(w, min_periods=2)
            .std()
            .reset_index(level=[0, 1], drop=True)
            .astype("float32")
        )
    return df


def _add_promo_features(df: pd.DataFrame) -> pd.DataFrame:
    """onpromotion is known for the test window, so we can use it directly.

    We still add a few aggregations: store-day promo intensity (how many
    families on promo today) and a lagged promo count.
    """
    df["onpromotion"] = df["onpromotion"].fillna(0).astype("int16")

    store_day_promo = (
        df.groupby(["store_nbr", "date"], observed=True)["onpromotion"]
        .transform("sum")
        .astype("int32")
    )
    df["store_day_promo_count"] = store_day_promo

    # Rolling promo intensity (no lag needed — onpromotion is known forward).
    df["promo_roll_7"] = (
        df.groupby(["store_nbr", "family"], observed=True)["onpromotion"]
        .rolling(7, min_periods=1)
        .mean()
        .reset_index(level=[0, 1], drop=True)
        .astype("float32")
    )
    return df


def _add_transactions_lag(df: pd.DataFrame) -> pd.DataFrame:
    """Transactions are NOT given for test dates -> must be lagged."""
    if "store_transactions" not in df.columns:
        return df
    g = df.groupby("store_nbr", observed=True)["store_transactions"]
    df["store_tx_lag_16"] = g.shift(config.MIN_LAG).astype("float32")
    df["store_tx_roll_mean_28"] = (
        g.shift(config.MIN_LAG)
        .groupby(df["store_nbr"], observed=True)
        .rolling(28, min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
        .astype("float32")
    )
    return df


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
CATEGORICAL_COLS = ["store_nbr", "family", "city", "state", "type", "cluster"]


def build_features(panel: pd.DataFrame) -> pd.DataFrame:
    """Add all engineered features. Returns a new dataframe."""
    df = panel.copy()

    df = _add_calendar(df)
    df = _add_earthquake_flag(df)
    df = _add_lags(df)
    df = _add_rolling(df)
    df = _add_promo_features(df)
    df = _add_transactions_lag(df)

    for col in CATEGORICAL_COLS:
        if col in df.columns:
            df[col] = df[col].astype("category")

    return df


def feature_columns(df: pd.DataFrame) -> list[str]:
    """Columns to feed the model. Excludes raw identifiers/target/leakage."""
    drop = {
        "id", "date", "sales", "is_test",
        # store_transactions is only known historically — keep its lagged
        # versions, drop the raw column.
        "store_transactions",
    }
    return [c for c in df.columns if c not in drop]
