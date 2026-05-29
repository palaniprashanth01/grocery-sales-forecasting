# Forecasting Report — Corporación Favorita

A short narrative companion to the code. Focus is on **why** the
pipeline is shaped the way it is, what it does well, where it falls
short, and what to do next.

---

## 1. Problem framing

Predict daily unit sales for **54 stores × 33 product families** for
the **16 days** after the training window. Scored on **RMSLE**.

**Result for this submission**

| Metric | Value |
|---|---|
| Seasonal-naive baseline RMSLE (local holdout) | 0.6170 |
| LightGBM 4-fold time-CV mean RMSLE | 0.3847 ± 0.0074 |
| LightGBM local holdout RMSLE | 0.3889 |
| **Kaggle public LB RMSLE** | **0.42049** |
| Kaggle public LB rank | **#269** (Kaggle user `PalaniPrashanthbenz`) |

The ~0.03 gap between local holdout (0.389) and public LB (0.420) is
consistent with the fold-to-fold variance we observed in CV (σ =
0.0074) plus the fact that the public LB is computed on a small slice
of the 16-day test window. The ranking around our score is tightly
packed — a 0.005 RMSLE improvement is worth roughly 100 leaderboard
places, which is why the improvement plan in §8 below ranks an Optuna
sweep + CatBoost blend as the single highest-ROI follow-ups.

Two facts dominate the design:

1. The horizon is **16 days**. Any feature derived from `sales` must
   lag by at least 16 days, otherwise serving the model in production
   would require recursive prediction (slow + error-accumulating).
2. The metric is RMSLE — i.e. **RMSE in log-space**. Training a
   regressor on `log1p(sales)` with RMSE loss gives that metric as a
   direct gradient. No proxy / surrogate loss needed.

These two together set the whole shape of the pipeline.

## 2. Architecture (one paragraph)

Raw CSVs → merged panel (`data_loader.build_panel`) → engineered
features (`features.build_features`) → time-based folds
(`validation.rolling_window_folds` + `final_holdout_split`) → seasonal-
naive baseline + global LightGBM on `log1p(sales)` (`models.train_lightgbm`)
→ error analysis + feature importance (`evaluate`) → refit on all train
data → `outputs/submission.csv`. The full Mermaid diagram is in
[architecture.md](architecture.md).

## 3. Validation strategy

**Only time-based splits.** No random k-fold anywhere.

- **Final holdout** — last 16 days of train held out. Mirrors the
  Kaggle test window in length + recency, so its RMSLE is the most
  honest single estimate of leaderboard score.
- **Rolling backtest** — 4 consecutive 16-day folds stepping backward
  from train end, expanding train window. Gives mean ± std RMSLE
  instead of a single noisy number.

Every fold asserts at runtime that
`max(train_date) < min(valid_date)`. On the feature side, every
sales-derived column is shifted by ≥ 16 days *before* any rolling
window is applied, so a rolling-7 mean at *t* uses sales at
*t-16, …, t-22*.

Why this matters: random k-fold on this dataset would let a model
"learn" Tuesday's sales from Wednesday of the same week, which is the
fastest way to score 0.30 in CV and 0.55 on the leaderboard.

## 4. Feature strategy

| Group | Examples | Justification |
|---|---|---|
| Calendar | dow, day, month, year, week, quarter, weekend, payday (1st/15th), post-payday | Ecuador public-sector pay calendar drives twice-monthly grocery spikes. |
| Holidays | nat / reg / loc × (holiday, bridge, event) | `transferred=True` rows dropped (the holiday moved), `Work Day` rows dropped (make-up workday). Merged by `locale` so each store only inherits its own city/state holidays. |
| Anomaly | Earthquake window (2016-04-16 + 30d) | Donation/relief spike — without the flag, the model treats it as a permanent regime shift and over-forecasts the same week in 2017. |
| Sales lags | 16, 17, 21, 28, 35, 42, 56 | All ≥ horizon. Includes weekly (21/28) and monthly (28/56) lags so weekly + monthly seasonality are both expressible. |
| Sales rolling | mean/std over 7, 14, 28, 56 (each shifted 16) | Trend + volatility at multiple scales. Shift-then-roll guarantees window never touches t. |
| Promo | onpromotion, store-day promo count, 7-day promo rate | `onpromotion` is provided for test → can be used un-lagged. Aggregations capture promotion intensity. |
| Transactions | lag-16, 28-day roll mean | NOT given for test → MUST be lagged. |
| Static | store_nbr, family, city, state, type, cluster | Native LightGBM categoricals — pooled learning across the long tail. |

## 5. Modelling choices

- **Global LightGBM, one model.** Per-(store, family) models overfit
  hard on low-volume series (a single SKU at a small store sells
  zero most days). A global model with the identifier columns as
  categoricals shares strength across the panel.
- **log1p + RMSE.** Mathematically equivalent to RMSLE on raw.
  Predictions go through `expm1` and are clipped to ≥ 0.
- **Seasonal-naive baseline.** Copies the same day-of-week from the
  last fully-observed week. Sets a floor; if LightGBM doesn't beat
  this on the holdout, something is wrong with features.
- **CatBoost is wired up but off by default** — useful for blending
  but doubles wall-clock.

## 6. Metric discussion

**RMSLE** is what Kaggle scores, but it's not the same thing as
*business impact*. RMSLE:

- Is **asymmetric**: a 20% under-forecast is penalised more than a 20%
  over-forecast. For groceries this is broadly the right shape
  (stockouts cost goodwill + lost sale) but not the *correct* shape
  (overstock has real cost too — spoilage, capital).
- Treats a 10% miss on `GROCERY I` the same as a 10% miss on
  `BABY CARE`. Fair across families; useless for $-impact.
- Is undefined on negatives → we clip to 0.

So `evaluate.py` also reports:

- **Per-family MAE** — surfaces which categories the model can't handle.
- **Revenue-weighted MAE** — weights each error by realised sales. Closest
  proxy for $-impact without a price feed.
- **Holiday RMSLE vs non-holiday RMSLE** — regime-shift robustness.
- **Promo RMSLE vs non-promo RMSLE** — promotion-period robustness.
- **Top-10 worst stores / families** — where to focus next.

## 7. Known limitations

1. **No per-family or per-store specialist models.** The global model
   underweights very high-volume series like `GROCERY I` because it
   shares parameters with `BOOKS`. A two-stage blend (global +
   per-family residual model on the top 5 families by volume) would
   likely cut RMSLE further.
2. **No recursive prediction.** We deliberately stuck to features
   with lag ≥ 16 to keep inference one-shot. This loses the
   information in sales[t-1..t-15]. A direct multi-output model (one
   model per horizon-day, predicting day +1, +2, …, +16 from features
   available at train end) would let day-1 use lag-1 features without
   leakage. More complex to maintain.
3. **No exogenous price feed.** Revenue-weighted error uses realised
   sales as the weight proxy. If item prices were available, the
   weighting would be sharper.
4. **Earthquake handling is a single binary flag.** It catches the
   level shift but not the family-level heterogeneity (water and
   first-aid spiked far more than household goods).
5. **No hierarchical reconciliation.** The forecasts at (store, family)
   level don't necessarily sum to a coherent total at the store or
   family level. For finance / supply-chain consumption that matters.
6. **Hyperparameters are conservative defaults**, not tuned. There's
   probably 0.02–0.04 RMSLE on the table from a careful Optuna sweep
   over `num_leaves`, `min_data_in_leaf`, `lambda_l2`,
   `feature_fraction`, and learning rate.
7. **Single seed, no CV-of-CV.** RMSLE std reported across 4 folds
   gives us variance over *time* but not over *seed*. For a
   production decision we'd want both.
8. **No drift / monitoring hooks.** A production version of this would
   need a feature-drift detector and a re-fit cadence.

## 8. Improvement plan (ranked by expected ROI)

| # | Idea | Expected lift | Effort |
|---|---|---|---|
| 1 | Optuna hyperparameter sweep on LightGBM | 0.02–0.04 RMSLE | half-day |
| 2 | Family-stratified two-stage blend (global + top-5 family residual models) | 0.01–0.03 | day |
| 3 | LightGBM + CatBoost average (already wired in `models.py`) | 0.005–0.015 | hour to enable |
| 4 | Direct multi-horizon model (16 separate LightGBMs, one per day-ahead) | 0.01–0.03, mainly on day +1..+3 | day |
| 5 | Rolling promo and holiday density features over 7/14/28 days | 0.005–0.01 | hour |
| 6 | Per-family earthquake severity flag (interaction with family categorical) | 0.002–0.005 | hour |
| 7 | Hierarchical reconciliation (MinT) to make store/family/total coherent | mostly business-readiness, small RMSLE delta | day |
| 8 | Switch baseline to a more competitive STL-decomposition model for sharper diagnostic comparison | reporting clarity | half-day |

## 9. Submission status

- **Kaggle submission uploaded.** Public LB RMSLE **0.42049**, rank
  **#269**, Kaggle user `PalaniPrashanthbenz`. Screenshots of the
  *My Submissions* page and the *Leaderboard* row are included in the
  assessment package.
- **Local artefacts in `./outputs/`** (re-generated on every run):
  - `submission.csv` — Kaggle format (28,512 rows + header)
  - `run_summary.json` — baseline + CV + holdout RMSLE numbers
  - `feature_importance.png` — top-30 features by gain
  - `pipeline.log` — full training log including error slice tables
  - `kaggle_leaderboard.webp` — leaderboard row #269 showing username + score
  - `kaggle_overview.webp` — competition overview page for context

## 10. AI assistance disclosure

See the **DISCLOSURE** section in [README.md](README.md). Short
version: code drafted in collaboration with Claude (Anthropic) via
Claude Code; reviewed and edited by the author. No LLM is called at
training or inference time. No pretrained model weights or external
datasets beyond the six Kaggle CSVs.
