# Pipeline Architecture

```mermaid
flowchart TD
    %% ---------- Raw inputs ----------
    subgraph RAW["Raw Kaggle CSVs (./data/)"]
        T["train.csv<br/>(id, date, store, family, sales, onpromotion)"]
        TE["test.csv<br/>(16 days after train end)"]
        S["stores.csv<br/>(city, state, type, cluster)"]
        O["oil.csv<br/>(daily WTI)"]
        H["holidays_events.csv<br/>(type, locale, transferred)"]
        TX["transactions.csv<br/>(store_nbr, date)"]
    end

    %% ---------- Data layer ----------
    subgraph DATA["src/data_loader.py — build_panel()"]
        M1["Concat train + test<br/>(mark is_test flag)"]
        M2["Join stores on store_nbr"]
        M3["Reindex oil to daily<br/>+ forward-fill"]
        M4["Join transactions on (store, date)"]
        M5["Normalise holidays:<br/>drop transferred, drop Work Day,<br/>split national / regional / local"]
        M6["Sort by (store, family, date)<br/>panel ready"]
    end

    %% ---------- Feature layer ----------
    subgraph FEAT["src/features.py — build_features()"]
        F1["Calendar:<br/>dow, day, month, year, week, quarter,<br/>weekend, payday (1st/15th), post-payday"]
        F2["Earthquake window flag<br/>(2016-04-16 + 30 days)"]
        F3["Sales lags ≥ 16 days<br/>16, 17, 21, 28, 35, 42, 56"]
        F4["Rolling stats<br/>shift(16) → rolling mean/std<br/>windows 7, 14, 28, 56"]
        F5["Promo features:<br/>onpromotion, store-day promo count,<br/>7-day promo rate"]
        F6["Transactions lag-16 + roll-28<br/>(no future leakage)"]
        F7["Cast categoricals:<br/>store_nbr, family, city, state, type, cluster"]
    end

    %% ---------- Validation ----------
    subgraph VAL["src/validation.py — time-based only"]
        V1["final_holdout_split:<br/>last 16 days of train → holdout"]
        V2["rolling_window_folds:<br/>N consecutive 16-day folds<br/>(expanding train window)"]
        V3["assert max(train_date) &lt; min(valid_date)<br/>per fold — leakage proof"]
    end

    %% ---------- Model ----------
    subgraph MODEL["src/models.py"]
        B["SeasonalNaive<br/>(weekly lookback, baseline floor)"]
        L["LightGBM GLOBAL model<br/>• log1p(sales) target<br/>• RMSE loss → RMSLE on raw<br/>• native categoricals<br/>• early stopping on time-fold valid"]
        C["(Optional) CatBoost<br/>for blend"]
    end

    %% ---------- Eval / output ----------
    subgraph EVAL["src/evaluate.py"]
        E1["RMSLE per fold + mean ± std"]
        E2["Per-family MAE<br/>Revenue-weighted error"]
        E3["Slices:<br/>holiday vs normal<br/>promo vs normal<br/>worst stores / families"]
        E4["Feature importance plot<br/>outputs/feature_importance.png"]
    end

    subgraph OUT["./outputs/"]
        SUB["submission.csv<br/>(id, sales)"]
        SUM["run_summary.json<br/>(baseline, CV, holdout)"]
        FI["feature_importance.png"]
    end

    %% ---------- Edges ----------
    T --> M1
    TE --> M1
    S --> M2
    O --> M3
    TX --> M4
    H --> M5
    M1 --> M2 --> M3 --> M4 --> M5 --> M6
    M6 --> F1 --> F2 --> F3 --> F4 --> F5 --> F6 --> F7
    F7 --> V1
    F7 --> V2
    V1 --> V3
    V2 --> V3
    V3 --> B
    V3 --> L
    L -.optional.-> C
    B --> E1
    L --> E1
    L --> E2
    L --> E3
    L --> E4
    L --> SUB
    E1 --> SUM
    E4 --> FI
```

## Pipeline stages (text)

1. **Load + merge.** `data_loader.build_panel()` concatenates train + test,
   joins stores on `store_nbr`, forward-fills oil to a daily index, joins
   transactions on (store, date), and normalises the holiday calendar
   (dropping `transferred=True` rows and `Work Day` rows, splitting the
   remaining holidays by `locale` into national / regional / local flags).
2. **Feature engineering.** `features.build_features()` adds calendar
   features, an earthquake-period flag, lag features ≥ 16 days, rolling
   means/stds that are shifted by 16 days *before* rolling (so the
   window never touches the prediction date), promo features, and lagged
   transactions. Identifier columns are cast to categorical for LightGBM.
3. **Time-based validation.** `validation.rolling_window_folds()` produces
   N consecutive 16-day folds stepping backward from train end with an
   expanding train window. Each fold asserts
   `max(train_date) < min(valid_date)`. `validation.final_holdout_split()`
   carves the last 16 days for a Kaggle-mirroring holdout.
4. **Models.** A seasonal-naive baseline sets the floor (weekly lookback).
   Then one global LightGBM is trained on `log1p(sales)` with RMSE loss
   (mathematically equivalent to RMSLE on the raw scale), with the six
   identifier columns as native categoricals and early stopping on each
   fold's valid set. CatBoost is wired up but optional.
5. **Predict.** After CV we refit on **all** training data using the
   median `best_iteration` from the holdout, predict the 16-day Kaggle
   window, `expm1` back to the raw scale, clip negatives to zero, and
   write `outputs/submission.csv`.
6. **Error analysis.** `evaluate` slices the holdout error by family,
   by store, holiday vs. non-holiday, and promo vs. non-promo; plots
   feature importance; reports revenue-weighted MAE alongside RMSLE.
