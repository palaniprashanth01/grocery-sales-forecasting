"""Models: seasonal-naive baseline, global LightGBM, optional CatBoost.

Design choices
--------------
* GLOBAL model. One LightGBM trained on every (store, family) at once,
  with store_nbr/family/city/state/type/cluster as native categorical
  features. This is the strongest default for this competition because
  it lets the model pool information across the long tail of low-volume
  series, where per-series models overfit.

* log1p target. Sales are non-negative, zero-inflated, and right-skewed.
  The competition is judged on RMSLE, which is exactly RMSE on log1p
  values. Training on log1p with RMSE loss gives us the metric we care
  about as a direct gradient.

* Predictions are exponentiated (expm1) and CLIPPED to >= 0. Negative
  sales are impossible and would blow up RMSLE.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from . import config


# ---------------------------------------------------------------------------
# Baseline
# ---------------------------------------------------------------------------
class SeasonalNaive:
    """Predict sales[t] = sales[t - 7*k], where k is the smallest integer so
    that t - 7*k <= last training day.

    Equivalent to: copy the same day-of-week from the most recent fully-
    observed week. This is the right baseline for retail because demand is
    overwhelmingly weekly-seasonal. Beating it is the bar.
    """

    def __init__(self, season_days: int = 7):
        self.season = season_days
        self.history_: pd.DataFrame | None = None

    def fit(self, df_train: pd.DataFrame) -> "SeasonalNaive":
        self.history_ = (
            df_train[["store_nbr", "family", "date", "sales"]]
            .sort_values(["store_nbr", "family", "date"])
            .copy()
        )
        return self

    def predict(self, df_future: pd.DataFrame) -> np.ndarray:
        assert self.history_ is not None, "fit() first"
        # Cache the last day available per series.
        last_day = self.history_.groupby(["store_nbr", "family"], observed=True)[
            "date"
        ].max()

        # Build a lookup of (store, family, date) -> sales for fast joins.
        hist_lookup = self.history_.set_index(["store_nbr", "family", "date"])["sales"]

        preds = np.zeros(len(df_future), dtype=np.float64)
        for i, row in enumerate(
            df_future[["store_nbr", "family", "date"]].itertuples(index=False)
        ):
            store, family, date = row
            ref_last = last_day.get((store, family), None)
            if ref_last is None:
                preds[i] = 0.0
                continue
            # Step backwards in 7-day jumps until we land on/before last_day.
            k = max(1, int(np.ceil((date - ref_last).days / self.season)))
            ref = date - pd.Timedelta(days=self.season * k)
            try:
                preds[i] = float(hist_lookup.loc[(store, family, ref)])
            except KeyError:
                preds[i] = 0.0
        return np.clip(preds, 0.0, None)


# ---------------------------------------------------------------------------
# LightGBM global model
# ---------------------------------------------------------------------------
@dataclass
class LGBMConfig:
    """Hyperparameters. Deliberately conservative -- tuned for stability
    over the 16-day horizon, not leaderboard squeezing."""
    objective: str = "regression"           # RMSE on log1p == RMSLE on raw.
    metric: str = "rmse"
    learning_rate: float = 0.05
    num_leaves: int = 127
    min_data_in_leaf: int = 200
    feature_fraction: float = 0.85
    bagging_fraction: float = 0.85
    bagging_freq: int = 1
    lambda_l2: float = 1.0
    num_boost_round: int = 3000
    early_stopping_rounds: int = 100
    verbose_eval: int = 200
    seed: int = config.SEED

    def to_lgb_params(self) -> dict[str, Any]:
        return {
            "objective": self.objective,
            "metric": self.metric,
            "learning_rate": self.learning_rate,
            "num_leaves": self.num_leaves,
            "min_data_in_leaf": self.min_data_in_leaf,
            "feature_fraction": self.feature_fraction,
            "bagging_fraction": self.bagging_fraction,
            "bagging_freq": self.bagging_freq,
            "lambda_l2": self.lambda_l2,
            "seed": self.seed,
            "verbose": -1,
        }


@dataclass
class LGBMResult:
    model: Any
    feature_names: list[str]
    categorical_features: list[str]
    best_iteration: int
    val_rmse_log: float | None = None
    val_rmsle: float | None = None
    feature_importance: pd.DataFrame = field(default_factory=pd.DataFrame)


def _to_log_target(y: np.ndarray) -> np.ndarray:
    # Defensive clip; spec says sales can be 0 but never negative.
    return np.log1p(np.clip(y, 0.0, None))


def _from_log_pred(yp: np.ndarray) -> np.ndarray:
    return np.clip(np.expm1(yp), 0.0, None)


def train_lightgbm(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_valid: pd.DataFrame | None,
    y_valid: np.ndarray | None,
    categorical_features: list[str],
    cfg: LGBMConfig | None = None,
) -> LGBMResult:
    """Train one LightGBM model with log1p target and early stopping."""
    import lightgbm as lgb  # imported lazily so the module is importable
    # without lightgbm installed (useful for skeleton inspection).

    cfg = cfg or LGBMConfig()
    feature_names = list(X_train.columns)
    cat_features = [c for c in categorical_features if c in feature_names]

    dtrain = lgb.Dataset(
        X_train, label=_to_log_target(y_train),
        categorical_feature=cat_features, free_raw_data=False,
    )
    valid_sets = [dtrain]
    valid_names = ["train"]
    if X_valid is not None and y_valid is not None:
        dvalid = lgb.Dataset(
            X_valid, label=_to_log_target(y_valid),
            categorical_feature=cat_features, reference=dtrain, free_raw_data=False,
        )
        valid_sets.append(dvalid)
        valid_names.append("valid")

    callbacks = [lgb.log_evaluation(period=cfg.verbose_eval)]
    if X_valid is not None:
        callbacks.append(lgb.early_stopping(cfg.early_stopping_rounds))

    model = lgb.train(
        cfg.to_lgb_params(),
        dtrain,
        num_boost_round=cfg.num_boost_round,
        valid_sets=valid_sets,
        valid_names=valid_names,
        callbacks=callbacks,
    )

    val_rmse_log = None
    val_rmsle = None
    if X_valid is not None and y_valid is not None:
        yp_log = model.predict(X_valid, num_iteration=model.best_iteration)
        val_rmse_log = float(np.sqrt(np.mean((yp_log - _to_log_target(y_valid)) ** 2)))
        # On log1p targets, RMSE-in-log-space IS RMSLE on raw -- we re-compute
        # against the raw target for clarity.
        yp_raw = _from_log_pred(yp_log)
        val_rmsle = float(
            np.sqrt(np.mean((np.log1p(yp_raw) - np.log1p(y_valid)) ** 2))
        )

    fi = pd.DataFrame(
        {
            "feature": feature_names,
            "gain": model.feature_importance(importance_type="gain"),
            "split": model.feature_importance(importance_type="split"),
        }
    ).sort_values("gain", ascending=False)

    return LGBMResult(
        model=model,
        feature_names=feature_names,
        categorical_features=cat_features,
        best_iteration=int(model.best_iteration or cfg.num_boost_round),
        val_rmse_log=val_rmse_log,
        val_rmsle=val_rmsle,
        feature_importance=fi,
    )


def predict_lightgbm(result: LGBMResult, X: pd.DataFrame) -> np.ndarray:
    yp_log = result.model.predict(X, num_iteration=result.best_iteration)
    return _from_log_pred(yp_log)


# ---------------------------------------------------------------------------
# Optional CatBoost (for ensembling)
# ---------------------------------------------------------------------------
def train_catboost(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_valid: pd.DataFrame | None,
    y_valid: np.ndarray | None,
    categorical_features: list[str],
    iterations: int = 2000,
    learning_rate: float = 0.05,
):
    """Trains a CatBoost regressor on log1p(sales). Returns a fitted model."""
    from catboost import CatBoostRegressor, Pool

    feature_names = list(X_train.columns)
    cat_features = [c for c in categorical_features if c in feature_names]

    # CatBoost wants categoricals as strings (or codes) without NaN.
    def _prep(df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        for c in cat_features:
            out[c] = out[c].astype(str).fillna("__nan__")
        return out

    train_pool = Pool(_prep(X_train), label=_to_log_target(y_train), cat_features=cat_features)
    valid_pool = None
    if X_valid is not None and y_valid is not None:
        valid_pool = Pool(_prep(X_valid), label=_to_log_target(y_valid), cat_features=cat_features)

    model = CatBoostRegressor(
        iterations=iterations,
        learning_rate=learning_rate,
        loss_function="RMSE",
        random_seed=config.SEED,
        verbose=200,
        early_stopping_rounds=100,
    )
    model.fit(train_pool, eval_set=valid_pool, use_best_model=valid_pool is not None)
    return model


def predict_catboost(model, X: pd.DataFrame, categorical_features: list[str]) -> np.ndarray:
    Xp = X.copy()
    for c in categorical_features:
        if c in Xp.columns:
            Xp[c] = Xp[c].astype(str).fillna("__nan__")
    yp_log = model.predict(Xp)
    return _from_log_pred(yp_log)
