# Grocery Sales Forecasting — Corporación Favorita

Production-style forecasting pipeline for the Kaggle competition
**[Store Sales — Time Series Forecasting][kaggle]**. Predicts daily unit
sales per `(store_nbr, family)` for the 16 days following the training
period. Scored on **RMSLE**.

[kaggle]: https://www.kaggle.com/competitions/store-sales-time-series-forecasting

The repo is graded on architecture, validation discipline, and feature
strategy — leaderboard rank is secondary.

## Result

| | RMSLE |
|---|---|
| Seasonal-naive baseline (local holdout) | 0.6170 |
| LightGBM 4-fold time-CV mean | 0.3847 ± 0.0074 |
| LightGBM local holdout | 0.3889 |
| **Kaggle public LB** | **0.42049 (rank #269, user `PalaniPrashanthbenz`)** |

## Repo layout

```
.
├── data/                       # drop the 6 Kaggle CSVs here
├── outputs/                    # submission.csv, feature_importance.png, run_summary.json
├── src/
│   ├── config.py               # paths, seeds, lag/window constants
│   ├── data_loader.py          # load + merge 6 tables -> single panel
│   ├── features.py             # leakage-safe feature engineering
│   ├── validation.py           # time-based CV (no random splits, ever)
│   ├── models.py               # seasonal-naive baseline + LightGBM (+ CatBoost)
│   ├── evaluate.py             # RMSLE, MAE, revenue-weighted, slices, plots
│   └── pipeline.py             # end-to-end runner
├── notebooks/
│   └── 01_walkthrough.ipynb    # narrated end-to-end run
├── architecture.md             # Mermaid pipeline diagram
├── requirements.txt
└── README.md
```

## Quick start

```bash
# 1. Install dependencies (Python 3.10+ recommended).
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. Drop the Kaggle CSVs into ./data/:
#    train.csv  test.csv  stores.csv  oil.csv  holidays_events.csv  transactions.csv
#
#    The fastest way:
#    kaggle competitions download -c store-sales-time-series-forecasting -p data/
#    cd data && unzip store-sales-time-series-forecasting.zip && cd ..

# 3. Run the full pipeline.
python -m src.pipeline

# 4. Find the submission at:
ls outputs/submission.csv
```

`python -m src.pipeline --folds 4` (default) runs the seasonal-naive
baseline, the 4-fold backtest, the final-holdout fit with error
analysis, then refits on **all** training data and writes the Kaggle
submission. Add `--no-kaggle` to skip the final refit/predict step
during development.

Alternatively, open `notebooks/01_walkthrough.ipynb` for a narrated
walkthrough that calls the same modules.

## Validation strategy (this is the important part)

The Kaggle test window is the **16 calendar days after train ends**. Any
validation that lets the model see *into the future* relative to its own
prediction date will overstate performance — sometimes dramatically. So
this repo uses **only** time-based splits.

### Final holdout (`final_holdout_split`)

The last 16 days of train become the held-out validation set. This
mirrors the Kaggle test window in both length and recency, so the
holdout RMSLE is the most honest single estimate of leaderboard score.

### Rolling backtest (`rolling_window_folds`)

We step the same 16-day window backwards through history N times, each
time training on an **expanding** window up to the fold's validation
start. With N=4 we get four RMSLE samples and can read both the mean
and the variance — a single fold is too noisy a signal to tune on.

### Leakage proofs

For every fold the code asserts:

```python
assert df.loc[train_mask, "date"].max() < df.loc[valid_mask, "date"].min()
```

That guarantees no row used for training has a date on or after any
validation row's date.

On the **feature** side, every column derived from `sales` is lagged by
≥ 16 days (the horizon length). Rolling windows are computed on the
already-shifted series, so a rolling-7 mean at time *t* uses sales at
*t-16, t-17, …, t-22*. Without that shift, a rolling-7 at *t* would
include *sales[t]* — perfect train-time score, catastrophic test-time
performance. `src/features.py` has the comment block spelling this out.

Calendar features (day-of-week, holidays, paydays, oil price, store
metadata, `onpromotion` — which Kaggle gives us for the test window)
are known in advance and used un-lagged.

## Feature strategy at a glance

| Group | Features | Notes |
|---|---|---|
| Calendar | dow, day, month, year, weekofyear, quarter, is_weekend, is_payday (1st/15th), is_post_payday | Paydays = Ecuadorian public-sector pay cycle. |
| Holidays | nat / reg / loc × (holiday, bridge, event) | Drops `transferred=True` and `Work Day` rows; merges by `locale`. |
| Anomaly | `earthquake_window` (2016-04-16 + 30 days) | Stops the model treating the relief spike as permanent. |
| Sales lags | lag 16, 17, 21, 28, 35, 42, 56 | All ≥ horizon — leakage-safe. |
| Sales rolling | mean/std over 7, 14, 28, 56 (each shifted 16) | Window never touches *t*. |
| Promo | `onpromotion`, store-day promo count, 7-day promo rate | `onpromotion` known forward → no lag needed. |
| Transactions | lag-16, 28-day rolling mean | Not given for test → must be lagged. |
| Static | store_nbr, family, city, state, type, cluster | Native categoricals in LightGBM. |

## Models

1. **Seasonal-naive baseline.** Copies the same day-of-week from the
   most recent fully-observed week. Sets the bar — if the LightGBM
   doesn't beat this, something is wrong.
2. **LightGBM (single global model).** One model trained across every
   store/family with the six identifier columns as native categoricals.
   Target is `log1p(sales)` with RMSE loss, which is mathematically
   equivalent to RMSLE on the raw scale. Predictions go through
   `expm1` and are clipped to ≥ 0.
3. **CatBoost (optional).** Wired up in `models.py` for blending; off
   by default in `pipeline.py` to keep the default run fast.

## Metric discussion

Kaggle scores **RMSLE**:

$$ \text{RMSLE} = \sqrt{\frac{1}{N}\sum (\log(1+\hat{y}) - \log(1+y))^2} $$

Properties worth knowing:

- **Asymmetric** — penalises under-forecasting more than the same-sized
  over-forecast. For retail this is the *right* shape (stockouts cost
  more than overstock, usually) but worth flagging.
- **Long-tail-friendly** — a 10% miss on a low-volume item contributes
  the same as a 10% miss on a high-volume one. Great for fairness
  across the 33 families; misleading for revenue impact.
- **Undefined on negatives** — we clip predictions to ≥ 0.

So we also report the business-facing metrics in `evaluate.py`:

- **Per-family MAE** — surfaces which categories the model is bad at.
- **Revenue-weighted MAE** — weights each error by realised sales, so a
  unit of error on `GROCERY I` counts more than the same error on
  `BABY CARE`. Without per-SKU prices this is the closest practical
  proxy for $-impact.
- **Holiday-period RMSLE** vs **normal-day RMSLE** — splits to show
  whether the model handles regime shifts.
- **Promo-period RMSLE** vs **non-promo RMSLE** — same idea for
  promotions.
- **Worst stores / families** — top-10 by RMSLE; where to focus next.

## Outputs

After a full run, `./outputs/` contains:

- `submission.csv` — Kaggle format, `id,sales`, sorted by id.
- `feature_importance.png` — top-30 features by gain.
- `run_summary.json` — baseline RMSLE, per-fold CV scores, mean ± std,
  final holdout RMSLE, best iteration.

## Architecture diagram

See [architecture.md](architecture.md) for the Mermaid pipeline diagram.

---

## DISCLOSURE — AI-assisted development & external components

Per the brief, here's a transparent accounting of how this submission
was built.

### AI assistance
- This codebase was developed in collaboration with **Claude (Anthropic)
  via Claude Code**. Claude generated drafts of every `src/*.py` module,
  the architecture diagram, this README, and the walkthrough notebook.
  The author reviewed, edited, and ran every file; design decisions
  (lag horizon ≥ 16, log1p + RMSE = RMSLE, holiday normalisation rules,
  rolling-window CV with expanding train) are explicit choices, not
  defaults.
- No Claude / OpenAI / other LLM is called at training or inference
  time. All model artefacts are produced locally.

### External / pretrained components
- **LightGBM** (Microsoft, MIT licence) — gradient boosting library.
  No pretrained weights; trained from scratch on the Kaggle data.
- **CatBoost** (Yandex, Apache-2.0) — optional, only used if enabled.
  No pretrained weights.
- **scikit-learn, pandas, numpy, matplotlib, seaborn, tqdm,
  pyarrow** — standard open-source dependencies (see `requirements.txt`).
- **No pretrained models, no embeddings, no external datasets** beyond
  the six CSVs provided by the Kaggle competition.
- **Data** — Corporación Favorita's competition CSVs from Kaggle. Not
  redistributed in this repo; users supply their own under `./data/`.

### Reproducibility
- Seeds are set in `config.set_seeds()` (Python `random`, NumPy,
  `PYTHONHASHSEED`, and LightGBM `seed=42`). Given the same input
  CSVs, a re-run produces the same submission to within LightGBM's
  threading non-determinism.
