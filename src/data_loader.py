"""Load and merge the six Favorita tables into one tidy panel.

Output is a long-format dataframe with one row per (store_nbr, family, date)
covering both train and test ranges. `is_test` distinguishes the two halves
so downstream code can treat them uniformly when building features.
"""
from __future__ import annotations

import pandas as pd

from . import config


def _read_csv(path, **kwargs) -> pd.DataFrame:
    return pd.read_csv(path, **kwargs)


def load_raw() -> dict[str, pd.DataFrame]:
    """Read all six CSVs with appropriate dtypes."""
    train = _read_csv(config.TRAIN_CSV, parse_dates=["date"])
    test = _read_csv(config.TEST_CSV, parse_dates=["date"])
    stores = _read_csv(config.STORES_CSV)
    oil = _read_csv(config.OIL_CSV, parse_dates=["date"])
    holidays = _read_csv(config.HOLIDAYS_CSV, parse_dates=["date"])
    transactions = _read_csv(config.TRANSACTIONS_CSV, parse_dates=["date"])
    return {
        "train": train,
        "test": test,
        "stores": stores,
        "oil": oil,
        "holidays": holidays,
        "transactions": transactions,
    }


def _prepare_oil(oil: pd.DataFrame, full_date_range: pd.DatetimeIndex) -> pd.DataFrame:
    """Reindex oil to every calendar day, then forward-fill gaps.

    The Kaggle oil file has missing days (weekends, market holidays). Models
    care about today's price, not whether the market was open, so we
    forward-fill. We back-fill only the very first rows where we have no
    history at all.
    """
    oil = oil.set_index("date").sort_index()
    oil = oil.reindex(full_date_range)
    oil.index.name = "date"
    oil["dcoilwtico"] = oil["dcoilwtico"].ffill().bfill()
    return oil.reset_index()


def _prepare_holidays(holidays: pd.DataFrame) -> pd.DataFrame:
    """Normalise the holidays_events table for join-time use.

    Favorita's holiday calendar is tricky:
      * `transferred = True` means the holiday was officially moved — so the
        ORIGINAL date is effectively a normal workday and should NOT count.
      * Rows with `type == "Transfer"` are the dates the holiday landed on
        AFTER the transfer — those ARE the effective holiday.
      * `type == "Bridge"` and `Additional` are extra days off; treat as
        holiday.
      * `type == "Work Day"` is a make-up workday after a bridge; explicitly
        NOT a holiday.
      * `locale` controls scope: National / Regional (state) / Local (city).

    We collapse into a per-(date, scope, scope_name) flag table so the merge
    in `build_panel` can hit national/state/city flags separately.
    """
    h = holidays.copy()

    # Effective holidays: not transferred-away, and not work-day make-ups.
    effective = h[(h["transferred"] == False) & (h["type"] != "Work Day")].copy()

    # Locale-specific scope columns we'll merge against the panel.
    effective["scope"] = effective["locale"].map(
        {"National": "national", "Regional": "regional", "Local": "local"}
    )
    effective["is_holiday"] = 1
    effective["is_bridge"] = (effective["type"] == "Bridge").astype(int)
    effective["is_event"] = (effective["type"] == "Event").astype(int)

    return effective[
        ["date", "scope", "locale_name", "is_holiday", "is_bridge", "is_event"]
    ].drop_duplicates(subset=["date", "scope", "locale_name"])


def build_panel(raw: dict[str, pd.DataFrame] | None = None) -> pd.DataFrame:
    """Merge all six tables into a single panel.

    Join keys:
      train/test  -> stores         on store_nbr
                  -> oil            on date  (forward-filled)
                  -> transactions   on (store_nbr, date)
                  -> holidays       on date / scope / scope_name
    """
    raw = raw or load_raw()
    train = raw["train"].copy()
    test = raw["test"].copy()

    # Mark which rows came from the test set so feature code knows where to
    # stop expecting a real target.
    train["is_test"] = 0
    test["is_test"] = 1
    test["sales"] = float("nan")  # target unknown — predicted later.

    if "transactions" not in test.columns:
        test["transactions"] = float("nan")  # not provided for the test split.

    panel = pd.concat([train, test], ignore_index=True, sort=False)
    panel["date"] = pd.to_datetime(panel["date"])

    # 1) Store metadata.
    panel = panel.merge(raw["stores"], on="store_nbr", how="left")

    # 2) Oil. Reindex to full date range before merging so test dates that
    #    fall on weekends still get a valid price.
    full_range = pd.date_range(panel["date"].min(), panel["date"].max(), freq="D")
    oil_full = _prepare_oil(raw["oil"], full_range)
    panel = panel.merge(oil_full, on="date", how="left")

    # 3) Transactions (historical only; null for test rows by design).
    tx = raw["transactions"].rename(columns={"transactions": "store_transactions"})
    panel = panel.merge(tx, on=["store_nbr", "date"], how="left")
    if "transactions" in panel.columns:
        panel = panel.drop(columns=["transactions"])

    # 4) Holidays — split scopes so we can flag whether the day is a holiday
    #    for THIS specific store's city / state / nationally.
    hol = _prepare_holidays(raw["holidays"])

    nat = (
        hol[hol["scope"] == "national"]
        .groupby("date", as_index=False)[["is_holiday", "is_bridge", "is_event"]]
        .max()
        .rename(
            columns={
                "is_holiday": "nat_holiday",
                "is_bridge": "nat_bridge",
                "is_event": "nat_event",
            }
        )
    )
    reg = hol[hol["scope"] == "regional"].rename(
        columns={
            "locale_name": "state",
            "is_holiday": "reg_holiday",
            "is_bridge": "reg_bridge",
            "is_event": "reg_event",
        }
    )[["date", "state", "reg_holiday", "reg_bridge", "reg_event"]]
    loc = hol[hol["scope"] == "local"].rename(
        columns={
            "locale_name": "city",
            "is_holiday": "loc_holiday",
            "is_bridge": "loc_bridge",
            "is_event": "loc_event",
        }
    )[["date", "city", "loc_holiday", "loc_bridge", "loc_event"]]

    panel = panel.merge(nat, on="date", how="left")
    panel = panel.merge(reg, on=["date", "state"], how="left")
    panel = panel.merge(loc, on=["date", "city"], how="left")

    holiday_cols = [
        "nat_holiday", "nat_bridge", "nat_event",
        "reg_holiday", "reg_bridge", "reg_event",
        "loc_holiday", "loc_bridge", "loc_event",
    ]
    panel[holiday_cols] = panel[holiday_cols].fillna(0).astype("int8")

    # Sort canonical order so lag/rolling features are correct.
    panel = panel.sort_values(["store_nbr", "family", "date"]).reset_index(drop=True)
    return panel
