"""Project-wide configuration: paths, seeds, constants.

All other modules import from here so the pipeline has a single source of truth.
"""
from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
# Resolve project root from this file's location so paths work regardless of
# the current working directory (CLI, notebook, pytest, etc).
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

# Kaggle competition file names (drop the raw CSVs into DATA_DIR).
TRAIN_CSV = DATA_DIR / "train.csv"
TEST_CSV = DATA_DIR / "test.csv"
STORES_CSV = DATA_DIR / "stores.csv"
OIL_CSV = DATA_DIR / "oil.csv"
HOLIDAYS_CSV = DATA_DIR / "holidays_events.csv"
TRANSACTIONS_CSV = DATA_DIR / "transactions.csv"

SUBMISSION_CSV = OUTPUTS_DIR / "submission.csv"
FEATURE_IMPORTANCE_PNG = OUTPUTS_DIR / "feature_importance.png"

# ---------------------------------------------------------------------------
# Forecasting constants
# ---------------------------------------------------------------------------
HORIZON_DAYS = 16  # Kaggle test set is the 16 days following train end.

# Minimum lag we are allowed to use as a feature. At test time we only know
# the target up to (train_end). For any day in the horizon we therefore can
# only safely use lags >= 16 (otherwise we'd need recursive predictions, which
# leak / accumulate error).
MIN_LAG = 16

LAGS = [16, 17, 21, 28, 35, 42, 56]
ROLLING_WINDOWS = [7, 14, 28, 56]

# Earthquake hit Ecuador on 2016-04-16. Mark a window after as anomalous —
# donation drives caused real sales spikes in groceries / first aid families.
EARTHQUAKE_DATE = "2016-04-16"
EARTHQUAKE_WINDOW_DAYS = 30

SEED = 42


def set_seeds(seed: int = SEED) -> None:
    """Set every RNG we touch so runs are reproducible."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
