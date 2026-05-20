# Development Plan: SPY ML Trading Baseline
### v3.0 — Narrowed First Milestone

---

## 1. Project Goal

Build a first-pass machine learning trading system for **SPY on the Daily (1D) timeframe only**.

The first milestone is intentionally narrow:

- **Instrument:** SPY
- **Timeframe:** 1D only
- **Strategy shape:** Long / Flat only
- **Features:** Price-only baseline
- **Validation:** Purged walk-forward cross-validation
- **Execution assumption:** Signal at bar close, trade at next bar open

This version does **not** attempt to solve:

- 4H trading
- short selling
- sentiment integration
- broker execution
- live deployment

Those are follow-on experiments after the baseline is stable.

---

## 2. First-Milestone Principles

- Keep the first system small enough to debug.
- Do not mix research questions in the same milestone.
- Preserve strict temporal integrity at every step.
- Use a backtest that can be falsified against buy-and-hold.
- Treat `pyproject.toml` and `uv.lock` as the only dependency source of truth.

---

## 3. Canonical Design Decisions

| Decision Area | Choice | Reason |
| :--- | :--- | :--- |
| Instrument | SPY | Liquid, simple, easy to benchmark |
| Timeframe | 1D only | Lower noise, simpler clock semantics |
| Positioning | Long / Flat | Removes false equivalence between "bad long" and "good short" |
| Labeling | Triple-Barrier, binary target | Keeps labels trade-oriented while matching long-only execution |
| Profit Target | `entry + 1.5 * ATR_14` | Encodes meaningful upside move |
| Stop Loss | `entry - 1.0 * ATR_14` | Defines adverse move clearly |
| Timeout | 10 bars | Limits holding horizon and regime drift |
| Timeout Label | Positive return -> Long, otherwise Flat | Avoids inventing a short class |
| Validation | Purged walk-forward CV with embargo | Reduces leakage from overlapping event horizons |
| Embargo | At least 10 bars | Must cover max holding period |
| Execution Timing | Predict at close, fill at next open | Prevents same-bar fill fantasy |
| Benchmark | Buy-and-hold SPY | Required baseline, not optional |
| Dependency Management | `uv` with `pyproject.toml` and `uv.lock` | One workflow, one lockfile |

If implementation code conflicts with this table, the table wins.

---

## 4. Pipeline

```
OHLCV Daily Data
      |
      v
Technical Indicators
      |
      v
Triple-Barrier Labels
      |
      v
Feature Matrix
      |
      v
Purged Walk-Forward Training
      |
      v
Probability-to-Position Rules
      |
      v
Friction-Aware Backtest
      |
      v
Compare Against Buy-and-Hold
```

---

## 5. Scope of the Baseline

### In Scope

- SPY daily OHLCV ingestion
- ATR, RSI, MACD, Bollinger feature engineering
- binary Long / Flat labels
- XGBoost baseline classifier
- purged walk-forward validation
- friction-aware backtest
- fold-by-fold diagnostics

### Out of Scope

- 4H data
- news ingestion
- FinBERT
- SHAP-driven pruning before the baseline exists
- paper trading
- live execution

SHAP can still be added later, but it should not block the first credible baseline.

---

## 6. Label Definition

### Triple-Barrier Parameters

| Barrier | Value |
| :--- | :--- |
| Profit Target | `entry + 1.5 * ATR_14` |
| Stop Loss | `entry - 1.0 * ATR_14` |
| Time-Out | 10 bars |

### Binary Label Assignment

| First Event | Label |
| :--- | :--- |
| Profit target hit first | `Long = 1` |
| Stop loss hit first | `Flat = 0` |
| Time-out hit first and realized return > `+0.1%` | `Long = 1` |
| Time-out hit first and realized return <= `+0.1%` | `Flat = 0` |

### Labeling Rules

- ATR at time `t` must be computed from information available at or before `t`.
- Labels are generated causally across the full time series.
- Leakage control comes from **purging and embargo during validation**, not from rebuilding ATR separately per fold.

---

## 7. Features

### Initial Feature Set

| Feature | Description |
| :--- | :--- |
| `return_1d` | 1-day return |
| `return_5d` | 5-day return |
| `volatility_20d` | Rolling realized volatility |
| `ATR_14` | Average True Range |
| `RSI_14` | Momentum oscillator |
| `MACD_line` | Trend signal |
| `MACD_signal` | Trend confirmation |
| `MACD_hist` | Trend acceleration |
| `BB_upper` | Upper Bollinger band |
| `BB_lower` | Lower Bollinger band |
| `BB_pct_b` | Price location inside bands |
| `volume_zscore_20d` | Relative volume |

### Feature Rules

- Every feature must be computable using data available at the prediction timestamp.
- No centered windows.
- No future returns embedded anywhere in the feature matrix.
- Warm up indicators before the first eligible sample; do not use a train/test "gap" as a substitute for causal feature construction.

---

## 8. Validation Protocol

Random splits are forbidden.

### Walk-Forward Structure

- **Training window:** fixed 2 years
- **Test window:** 1 year
- **Step size:** 1 year
- **Purging:** remove training samples whose label horizons overlap the test window
- **Embargo:** at least 10 bars after each test block boundary

### Why This Replaces the Old Gap Rule

The main leakage risk here is not MACD lookback. It is **overlapping event horizons from triple-barrier labels**. Purged walk-forward CV addresses that directly.

### Per-Fold Checklist

- generate train and test splits with purge + embargo
- fit the model only on the train fold
- tune only within the training fold
- store fold metrics separately
- compare fold performance stability, not just average performance

---

## 9. Model Plan

### Baseline Model

- **Primary model:** XGBoost classifier
- **Target:** binary probability of `Long`
- **Objective:** `binary:logistic`
- **Class imbalance:** use per-fold sample weights if needed, only after checking label distribution

### Initial Hyperparameter Guardrails

| Parameter | Initial Range |
| :--- | :--- |
| `max_depth` | 3-5 |
| `learning_rate` | 0.02-0.05 |
| `subsample` | 0.7-0.9 |
| `colsample_bytree` | 0.7-0.9 |
| `min_child_weight` | 1-5 |
| `n_estimators` | 200-800 with early stopping |

### Modeling Notes

- Do not optimize for raw accuracy.
- Primary outputs are probability, trade frequency, and risk-adjusted return.
- A trivial mostly-flat model is not acceptable just because its classification score looks good.

---

## 10. Backtest Rules

### Execution Semantics

- Features are computed at the **close** of day `t`.
- The model produces a signal after that close.
- Any position change happens at the **open** of day `t+1`.
- No same-bar entry using the close that generated the signal.

### Position Rules

- one position at a time
- long or cash only
- no leverage
- no shorting

### Signal Rule

Start simple:

- enter long when `P(Long) >= threshold`
- go flat when `P(Long) < threshold`

The threshold is a tunable research parameter, but it must be selected inside the walk-forward process, not on the full sample.

### Friction

At minimum include:

- commissions
- slippage
- next-open fill assumption

Backtests without friction are not decision-grade.

---

## 11. Evaluation Criteria

### Required Comparisons

- model strategy vs SPY buy-and-hold
- fold-by-fold metrics, not only aggregate metrics
- strategy returns after friction

### Required Metrics

| Metric | Why It Matters |
| :--- | :--- |
| CAGR | headline growth rate |
| Sharpe ratio | risk-adjusted return |
| Max drawdown | capital risk |
| Profit factor | trade efficiency |
| Win rate | only meaningful in context of payoff ratio |
| Trade count | detects under-trading |
| Exposure | detects always-in or never-in behavior |
| Turnover | cost sensitivity |

### Go / No-Go Criteria for the Baseline

- beats buy-and-hold on at least one risk-adjusted metric
- does not collapse in most folds
- produces a sensible number of trades
- remains viable after friction

If it fails those checks, do not add sentiment or 4H complexity yet.

---

## 12. Implementation Phases

### Phase 0: Repository and Environment

- make `pyproject.toml` the only dependency manifest
- use `uv` for install, lock, and run workflows
- commit `uv.lock`
- add dependency groups for `dev` and `research`
- align `.python-version` with the Python version declared in `pyproject.toml`

### Phase 1: Data and Features

- ingest SPY daily OHLCV
- normalize timestamps
- compute indicator warmup region
- build the first price-only feature matrix

### Phase 2: Labels and Validation

- implement binary triple-barrier labels
- implement purged walk-forward splitter with embargo
- add tests for label correctness and split leakage

### Phase 3: Modeling

- train XGBoost baseline
- tune threshold and hyperparameters within fold boundaries
- record per-fold diagnostics and artifact outputs

### Phase 4: Backtesting

- simulate next-open execution
- apply friction
- compare to buy-and-hold
- produce a compact research report

### Phase 5: Extensions Only If Baseline Holds Up

- add SHAP-based feature analysis
- add sentiment as a separate experiment
- test 4H as a separate experiment
- consider long/short only after long/flat has evidence

---

## 13. Project Structure

```
algotrader/
├── data/
│   ├── raw/
│   ├── interim/
│   └── features/
├── src/
│   └── algotrader/
│       ├── ingestion/
│       ├── features/
│       ├── labels/
│       ├── training/
│       ├── backtest/
│       └── utils/
├── tests/
├── notebooks/
├── reports/
├── pyproject.toml
├── uv.lock
├── .python-version
└── plan.md
```

### Structure Notes

- keep the package under `src/algotrader/`
- do not add `requirements.txt`
- generated data, models, and reports stay out of git
- notebooks are exploratory only, never production logic

---

## 14. Testing Priorities

The first tests should cover correctness invariants, not model quality.

- triple-barrier label edge cases
- purge + embargo split correctness
- next-open execution timing
- indicator calculations on warmup boundaries
- backtest friction application

---

## 15. Deferred Work

These remain valid ideas, but they are explicitly deferred:

- FinBERT sentiment
- fuzzy headline deduplication
- 4H trading
- local LLM-based news interpretation
- paper trading bridge
- live broker execution

They should be treated as independent experiments layered on top of a credible baseline, not as part of the first build.
