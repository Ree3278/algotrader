# Comprehensive Development Plan: Independent S&P 500 Trading Agent
### v2.0 — Revised & Hardened Edition

---

## 1. Project Overview & Strategy

This document is the revised architectural blueprint for an independent, automated ML trading agent targeting the S&P 500 on **4-Hour (4H) and Daily (1D)** timeframes. It supersedes the original draft, incorporating a hardened label framework, corrected class-imbalance handling, time-decay sentiment aggregation, SHAP-driven feature pruning, and a formally specified Walk-Forward validation protocol.

### Core Design Principles
- Mid-to-long-term structural signals only — no intraday noise.
- Strict temporal integrity enforced at every pipeline stage (no lookahead bias).
- Volatility-adaptive labeling via the Triple-Barrier Method (promoted from Future Work to Phase 0).
- All design decisions documented with explicit parameters and rationale.

### Core Tech Stack

| Component | Technology |
| :--- | :--- |
| Market & News Data | Alpha Vantage API (OHLCV + News Sentiment feed) |
| Technical Indicators | pandas-ta (RSI, MACD, ATR, Bollinger Bands) |
| Sentiment Engine | FinBERT — `ProsusAI/finbert` via Hugging Face Transformers |
| Deduplication | `rapidfuzz` (fuzzy headline matching, pre-sentiment step) |
| Core Model | XGBoost Classifier (primary) + Random Forest (sanity check) |
| Explainability | SHAP — global feature importance for post-training pruning |
| Backtesting | `Backtesting.py` or `Backtrader` (hardcoded friction parameters) |
| Data Pipeline | Pandas, NumPy, Scikit-Learn |

---

## 2. Architectural Decisions Log

Every non-trivial design choice in this system has been explicitly resolved. The table below is the canonical reference — if implementation code conflicts with this table, **the table wins**.

| Decision Area | Choice | Rationale |
| :--- | :--- | :--- |
| Label Method | Triple-Barrier | Volatility-adaptive; avoids brittle fixed-direction labels |
| R:R Ratio | 1.5 : 1.0 (PT/SL) | Requires only ~40% win rate to break even |
| Timeout Horizon | 15 bars (4H) / 10 bars (1D) | Captures structural moves; avoids regime drift on long holds |
| Timeout Label | Realized return at close | Retains training samples; avoids Hold class inflation |
| Class Imbalance | `sample_weight='balanced'` | Correct for multiclass XGBoost; `scale_pos_weight` is binary-only |
| Sentiment Aggregation | Exp. time-decay (λ = 0.007) | 1hr-old headline retains ~67% weight |
| Deduplication | Pre-decay fuzzy hash (`rapidfuzz`) | Prevents wire-service syndication from inflating a single event |
| Feature Pruning | SHAP global, < 1% threshold | Evidence-based; avoids arbitrary correlation cuts |
| Walk-Forward Window | Fixed 2-year window | Prevents stale regimes (e.g. 2008) contaminating recent folds |
| Train/Test Gap | 30 days (1D) / 120 bars (4H) | Covers max lookback (26-day MACD EMA) with buffer |

---

## 3. System Architecture & Pipeline

```
+-----------------------------------------------------------------------+
|                           Data Sourcing                               |
|   Alpha Vantage API (OHLCV Price Data & Financial News Stream)        |
+-----------------------------------------------------------------------+
                                    │
            ┌───────────────────────┴───────────────────────┐
            ▼                                               ▼
+-----------------------+                       +-----------------------+
|  Raw Sentiment Text   |                       |    OHLCV Price Data   |
+-----------------------+                       +-----------------------+
            │                                               │
            ▼                                               ▼
+-----------------------+                       +-----------------------+
|  Dedup (rapidfuzz)    |                       |  Triple-Barrier       |
|  + FinBERT            |                       |  Label Generator      |
|  + Time-Decay Agg.    |                       |  (ATR-scaled)         |
+-----------------------+                       +-----------------------+
            │                                               │
            ▼ (Net Sentiment, Abs. Emotion)                 ▼ (Labels)
            └───────────────────────┬───────────────────────┘
                                    ▼
                        +-----------------------+
                        |  Feature Matrix       |
                        |  (Aligned Time-Series)|
                        +-----------------------+
                                    │
                                    ▼
                        +-----------------------+
                        |  XGBoost Classifier   |
                        |  + SHAP Pruning       |
                        |  + Walk-Forward CV    |
                        +-----------------------+
                                    │
                                    ▼
                        +-----------------------+
                        |  Trading Signal       |
                        |  (Buy / Sell / Hold)  |
                        +-----------------------+
                                    │
                                    ▼
                        +-----------------------+
                        |  Backtester           |
                        |  (Friction-Adjusted)  |
                        +-----------------------+
```

---

## 4. Critical Problem Register & Mitigations

### Problem A — Noise-to-Signal
The S&P 500 is one of the most efficiently priced assets in the world.

- Filter headlines where FinBERT `P_neutral > 0.80` before aggregation.
- Time-decay aggregation (λ = 0.007) attenuates stale noise naturally.
- SHAP pruning eliminates indicators that contribute noise rather than signal.

### Problem B — Overfitting / Data Leakage
Random `train_test_split` on time-series data is fatal — future data leaks into training, creating spectacular backtests that collapse live.

- Fixed 2-year training window, Walk-Forward sliding 1 year per fold.
- Mandatory gap of 30 calendar days (1D) or 120 bars (4H) between train-end and test-start.
- Gap must be ≥ longest feature lookback (26-day MACD EMA = minimum bound).

### Problem C — Lookahead Bias / Clock Trap
News published after a bar closes cannot be incorporated into that bar's feature vector.

- Hard constraint at ingestion: `t_news < t_bar_close` (strict inequality).
- All timestamps in UTC throughout — no local time conversions.
- Deduplication runs **before** the timestamp filter to prevent stale duplicates surviving the cut.

### Problem D — Market Friction
A model with 1.5 Sharpe on paper can lose money live due to hidden frictional parameters.

- **Commissions:** Hardcode exact broker fees per share/contract.
- **Slippage:** Minimum 0.5–1.0 tick penalty on every market order fill.
- **Spread:** Conservative bid-ask spread assumption for 4H bar entries.

### Problem E — Class Imbalance *(Added)*
Markets trend ~30% of the time. A naive Buy/Sell/Hold classifier will predict Hold ~70% of the time, achieve 65%+ accuracy, and generate zero trades.

- Do **not** use `scale_pos_weight` — this is binary-only in XGBoost and silently fails in multiclass mode.
- Use `compute_sample_weight('balanced', y_train)` from `sklearn.utils.class_weight`, passed as `sample_weight=weights` to `model.fit()`.
- Recompute weights fresh on each Walk-Forward fold — never carry over.

### Problem F — Brittle Label Definition *(Added)*
Binary next-bar direction labels treat a +0.05% move identically to a +3% move. The model learns to predict noise, not tradeable moves.

- **Resolution:** Triple-Barrier Method promoted from Future Work to Phase 0 (see Section 5).

---

## 5. Implementation Phases

```
[Phase 0: Labels] ──► [Phase 1: Ingestion] ──► [Phase 2: NLP] ──► [Phase 3: ML + CV] ──► [Phase 4: Backtest]
```

### Phase 0: Label Engineering — Triple-Barrier Method

> **Priority:** This must be completed before any model training. Label quality determines everything downstream.

#### Barrier Parameters

| Barrier | Value | Notes |
| :--- | :--- | :--- |
| B1: Profit Target | Entry + 1.5 × ATR_14 | Upper ceiling — exit on win |
| B2: Stop-Loss | Entry − 1.0 × ATR_14 | Lower floor — exit on loss |
| B3: Time-Out (4H) | 15 bars (~4 trading days) | Force-exit if neither B1 nor B2 hit |
| B3: Time-Out (1D) | 10 bars (~2 weeks) | Force-exit if neither B1 nor B2 hit |
| Risk:Reward Ratio | 1.5 : 1.0 | Breakeven win rate = 40%; target > 35% |

#### Label Assignment Logic

| First Event | Label | Notes |
| :--- | :--- | :--- |
| B1 hit (Profit Target) | Buy (1) | Positive outcome confirmed |
| B2 hit (Stop-Loss) | Sell (−1) | Negative outcome confirmed |
| B3 hit (Time-Out) | Realized return at close | ret > +0.1%: Buy; ret < −0.1%: Sell; else: Hold |

> **Design Note:** Dropping timeout samples preserves label purity but can eliminate 40–60% of the dataset in low-volatility regimes. Labeling by realized return retains samples at the cost of slightly noisier labels — a trade-off that favors fold stability.

---

### Phase 1: Data Ingestion & Feature Engineering

1. Set up Alpha Vantage API keys for Historical Market Data and News Sentiment.
2. Extract historical OHLCV data for SPY ETF (more liquid, tighter spreads than S&P futures). Minimum 5 years of history.
3. Compute ATR_14 **first** — required for Triple-Barrier label generation before any other feature work.
4. Compute remaining technical indicators via `pandas-ta`:
   - **RSI(14):** Momentum oscillator — overbought/oversold signals
   - **MACD(12,26,9):** Trend direction and acceleration. Longest lookback = 26 bars; sets minimum Walk-Forward gap.
   - **Bollinger Bands(20,2):** Volatility context and mean-reversion signals

#### Clock Integrity Rules
- All timestamps in UTC throughout the pipeline.
- Ingestion constraint: `t_news < t_bar_close` (strict inequality — not `<=`).
- Deduplication runs before the timestamp filter.

---

### Phase 2: NLP & Sentiment Pipeline

#### Step 1 — Deduplication (Pre-Decay)
Wire services syndicate identical headlines to dozens of outlets within minutes. Dedup runs **before** any aggregation.

- **Tool:** `rapidfuzz.fuzz.ratio` — fuzzy string similarity on headline text
- **Threshold:** Similarity > 0.90 → keep highest-confidence article (by FinBERT score), discard duplicates
- **Scope:** Within each time block only — cross-block dedup would suppress legitimate repeated coverage

#### Step 2 — FinBERT Processing
- **Model:** `ProsusAI/finbert` — outputs `[P_positive, P_negative, P_neutral]` per headline
- **Pre-filter:** Discard headlines where `P_neutral > 0.80` before aggregation
- **Input:** Headline + summary concatenated (up to 512-token FinBERT limit)

#### Step 3 — Time-Decay Aggregation

| Parameter | Value & Rationale |
| :--- | :--- |
| Decay function | `w(t) = exp(−0.007 × Δt)`, where Δt = minutes before bar close |
| Lambda (λ) | **0.007** — headline 60 min before close retains ~67% weight; 120 min retains ~43% |
| Net Sentiment | `Σ[w(t) × (P_pos − P_neg)] / Σw(t)` → range [−1.0, 1.0] |
| Absolute Emotion | `Σ[w(t) × (P_pos + P_neg)] / Σw(t)` → range [0.0, 1.0] (market hype proxy) |
| Empty block handling | No surviving headlines: Net Sentiment = 0.0, Absolute Emotion = 0.0, `is_empty_block = 1` |

---

### Phase 3: Model Training & Walk-Forward Validation

#### Feature Matrix

| Feature | Type | Description |
| :--- | :--- | :--- |
| `RSI_14` | Continuous | Momentum oscillator [0–100] |
| `MACD_line` | Continuous | MACD line value |
| `MACD_signal` | Continuous | Signal line value |
| `MACD_hist` | Continuous | Histogram (line − signal) |
| `ATR_14` | Continuous | Average True Range — also used in Triple-Barrier scaling |
| `BB_upper` | Continuous | Bollinger upper band |
| `BB_lower` | Continuous | Bollinger lower band |
| `BB_pct_b` | Continuous | Price position within bands [0–1] |
| `net_sentiment` | Continuous | Decay-weighted Net Sentiment [−1, 1] |
| `abs_emotion` | Continuous | Decay-weighted Absolute Emotion [0, 1] |
| `is_empty_block` | Binary | 1 if no headlines survived the P_neutral > 0.80 filter |

#### Class Imbalance Correction
- **Method:** `compute_sample_weight('balanced', y_train)` from `sklearn.utils.class_weight`
- **Usage:** `model.fit(X_train, y_train, sample_weight=weights)` — not `scale_pos_weight` (binary-only, silently wrong on multiclass)
- Weights recomputed fresh on each Walk-Forward fold — never carried over.

#### XGBoost Hyperparameters

| Parameter | Value | Rationale |
| :--- | :--- | :--- |
| `max_depth` | 3–5 | Limits tree complexity; primary overfitting guard |
| `learning_rate` | 0.01 | Low eta; requires more trees but generalizes better |
| `subsample` | 0.80 | Row subsampling per tree — reduces variance |
| `colsample_bytree` | 0.80 | Feature subsampling — reduces inter-tree correlation |
| `n_estimators` | 500–1000 | Tuned via early stopping on validation fold |
| `objective` | `multi:softprob` | Outputs class probabilities for all three classes |
| `eval_metric` | `mlogloss` | Multiclass log-loss for early stopping |

#### Walk-Forward Validation Protocol

- **Window type:** Fixed 2-year training window, slides forward 1 year per fold
- **Gap:** 30 calendar days (1D) / 120 bars (4H) — clears 26-bar MACD lookback with buffer
- **Rationale for fixed vs expanding:** Prevents 2008-era regime data from distorting modern folds

| Fold | Train Start | Train End | Test Start | Test End |
| :---: | :---: | :---: | :---: | :---: |
| 1 | T₀ | T₀ + 2yr | T₀ + 2yr + gap | T₀ + 3yr |
| 2 | T₀ + 1yr | T₀ + 3yr | T₀ + 3yr + gap | T₀ + 4yr |
| 3 | T₀ + 2yr | T₀ + 4yr | T₀ + 4yr + gap | T₀ + 5yr |
| N | slides +1yr | slides +1yr | slides +1yr | Continue until data exhausted |

**Per-fold checklist:**
- Recompute `sample_weight` on each fold's `y_train` independently
- Recompute Triple-Barrier labels using only that fold's ATR (no future ATR leakage)
- Record Sharpe, Profit Factor, Win Rate, Max Drawdown per fold
- Flag folds with significant performance degradation — may signal regime change

#### SHAP Feature Pruning
1. Train full feature set across all Walk-Forward folds.
2. Run `SHAP TreeExplainer` on each fold's trained model.
3. Compute mean |SHAP value| per feature, averaged globally across all folds.
4. Drop features where global mean |SHAP| < 1% of the top feature's SHAP value.
5. Retrain all folds from scratch with the pruned feature set and re-evaluate metrics.

> **Note:** SHAP importance can vary across folds. Track per-fold SHAP rankings as a secondary diagnostic — large fold-to-fold variance in a feature's importance signals regime-dependence.

---

### Phase 4: Backtesting & Metric Verification

#### Execution Rules
- **Buy signal:** `P(Buy class) > 0.65`
- **Sell / Short signal:** `P(Sell class) > 0.65`
- **Hold / No action:** All class probabilities below threshold
- **Position sizing:** Fixed fractional (2% of equity per trade) — no Kelly sizing until Sharpe is proven stable across ≥ 3 folds

#### Hardcoded Friction Parameters
- **Commissions:** Exact broker fee per share / per contract
- **Slippage:** 0.5–1.0 tick deviation penalty on every market order
- **Spread:** Conservative bid-ask spread assumption for 4H bar entries

#### Target Performance Metrics

| Metric | Calculation | Target Threshold |
| :--- | :--- | :--- |
| **Sharpe Ratio** | (Rp − Rf) / Volatility | **> 1.5** in stable environments |
| **Profit Factor** | Gross Profits / Gross Losses | **> 1.35** |
| **Max Drawdown** | Peak-to-trough maximum account drop | **< 15%** |
| **Win Rate** | Profitable trades / Total trades | **> 35%** (breakeven at 1.5:1 R:R is 40%) |

> **Note:** Win Rate target revised from > 53% to > 35% — the original target was inconsistent with the 1.5:1 R:R ratio baked into Triple-Barrier labels.

---

## 6. Folder Structure

```
sp500-trading-agent/
│
├── data/                          # All data — never committed to version control
│   ├── raw/                       # Unprocessed API responses, exactly as received
│   │   ├── ohlcv/                 # Alpha Vantage OHLCV responses (SPY, 4H + 1D)
│   │   └── news/                  # Alpha Vantage news JSON feeds, by date
│   │
│   ├── processed/                 # Cleaned, aligned, timestamp-validated data
│   │   ├── ohlcv_4h.parquet
│   │   ├── ohlcv_1d.parquet
│   │   └── news_deduplicated/     # Post-dedup, pre-FinBERT news, by date
│   │
│   ├── features/                  # Final feature matrices, ready for model input
│   │   ├── features_4h.parquet    # OHLCV indicators + sentiment, aligned on bar index
│   │   └── features_1d.parquet
│   │
│   └── labels/                    # Triple-Barrier labels, kept separate from features
│       ├── labels_4h.parquet      # Buy/Sell/Hold per bar, with barrier metadata
│       └── labels_1d.parquet
│
├── src/                           # All source code — one module per concern
│   │
│   ├── ingestion/                 # Phase 1: data collection and clock integrity
│   │   ├── alpha_vantage.py       # API client — OHLCV and news endpoints
│   │   ├── clock_guard.py         # UTC enforcement, t_news < t_bar_close filter
│   │   └── storage.py             # Save raw responses to data/raw/
│   │
│   ├── labels/                    # Phase 0: Triple-Barrier label generation
│   │   ├── triple_barrier.py      # Core labeling logic (B1, B2, B3 + timeout handling)
│   │   └── label_stats.py         # Class distribution diagnostics per fold
│   │
│   ├── nlp/                       # Phase 2: sentiment pipeline
│   │   ├── deduplicator.py        # rapidfuzz fuzzy dedup within time blocks
│   │   ├── finbert_runner.py      # FinBERT inference, P_neutral filter
│   │   └── aggregator.py          # Time-decay aggregation (lambda=0.007)
│   │
│   ├── features/                  # Phase 1/2 output: feature engineering
│   │   ├── technicals.py          # pandas-ta indicators (RSI, MACD, ATR, BB)
│   │   ├── sentiment_features.py  # Merge sentiment columns into feature matrix
│   │   └── builder.py             # Orchestrates full feature matrix construction
│   │
│   ├── training/                  # Phase 3: model training and validation
│   │   ├── walk_forward.py        # Fixed-window fold generator with gap enforcement
│   │   ├── class_weights.py       # compute_sample_weight wrapper, per-fold
│   │   ├── xgboost_model.py       # Model definition, hyperparameters, fit/predict
│   │   ├── shap_pruning.py        # Global SHAP aggregation and feature elimination
│   │   └── fold_metrics.py        # Per-fold Sharpe, PF, win rate, drawdown logging
│   │
│   ├── backtest/                  # Phase 4: simulation and metric verification
│   │   ├── strategy.py            # Signal thresholds, position sizing rules
│   │   ├── friction.py            # Commissions, slippage, spread parameters
│   │   └── reporter.py            # Aggregate metrics across folds, final report
│   │
│   └── utils/                     # Shared helpers, no business logic
│       ├── config.py              # Central config (API keys via env, hyperparams)
│       ├── logger.py              # Structured logging, consistent across modules
│       └── time_utils.py          # UTC conversion, bar alignment helpers
│
├── notebooks/                     # Exploratory analysis and diagnostics only
│   ├── 01_data_exploration.ipynb  # Raw data quality checks, coverage gaps
│   ├── 02_label_diagnostics.ipynb # Triple-Barrier class distribution, barrier hit rates
│   ├── 03_sentiment_audit.ipynb   # Dedup rates, decay weight distributions
│   ├── 04_feature_correlations.ipynb  # Pre-SHAP correlation matrix
│   └── 05_backtest_analysis.ipynb # Fold-by-fold performance breakdown
│
├── models/                        # Serialized model artifacts
│   ├── fold_1/
│   │   ├── xgb_model.json         # XGBoost model (fold 1)
│   │   └── shap_values.pkl        # SHAP output for this fold
│   ├── fold_2/
│   └── ...
│
├── reports/                       # Generated outputs — not source code
│   ├── walk_forward_summary.csv   # Per-fold metrics table
│   ├── shap_global_ranking.csv    # Global feature importance after pruning
│   └── backtest_report.html       # Final backtester output
│
├── tests/                         # Unit and integration tests
│   ├── test_clock_guard.py        # Verifies t_news < t_bar_close is enforced
│   ├── test_triple_barrier.py     # Label assignment correctness, edge cases
│   ├── test_deduplicator.py       # Dedup threshold behavior
│   ├── test_aggregator.py         # Decay weight calculations
│   └── test_walk_forward.py       # Fold gap correctness, no leakage
│
├── .env.example                   # Template for API keys — never commit .env
├── .gitignore                     # Excludes data/, models/, .env, __pycache__
├── requirements.txt               # Pinned dependencies
└── plan.md                        # This document
```

### Structure Rationale

**`data/` is split into four layers** (raw → processed → features → labels) so that any stage can be rerun independently without re-fetching the API. Raw data is the most expensive to regenerate and is never modified in place.

**Labels live separately from features** because Triple-Barrier labels must be computed using ATR from the same fold's training window only. Keeping them decoupled prevents accidental future-ATR leakage if the feature matrix is rebuilt.

**`src/` mirrors the phase order** — ingestion → labels → nlp → features → training → backtest. A developer can trace any data artifact back to the module that produced it without reading the full codebase.

**Notebooks are diagnostics only** — no production logic lives there. Each notebook corresponds to a validation checkpoint in the pipeline (data quality, label health, sentiment audit, feature correlations, backtest breakdown).

**`tests/` covers the highest-risk correctness invariants** — clock guard, barrier assignment, dedup behavior, decay math, and walk-forward gap enforcement. These are the failure modes most likely to be silent (passing tests, wrong behavior).

---

## 7. Model Evaluation Metrics

| Metric | Calculation | Target Threshold |
| :--- | :--- | :--- |
| **Sharpe Ratio** | (Rp − Rf) / Volatility | **> 1.5** in stable environments |
| **Profit Factor** | Gross Profits / Gross Losses | **> 1.35** |
| **Max Drawdown** | Peak-to-trough maximum account drop | **< 15%** |
| **Win Rate** | Profitable trades / Total trades | **> 35%** (1.5:1 R:R; breakeven = 40%) |

---

## 8. Future Work & Scaling Expansion

### I. Upgrade to Local LLMs
Transition from FinBERT to localized small language models (e.g., **Llama-3.2-3B** or **Mistral-7B** via `Ollama` / `vLLM`). LLMs capture advanced context — economic sarcasm, geopolitical escalation, nuanced policy pivots — that FinBERT's three-class architecture structurally cannot. Output: structured JSON with multi-dimensional sentiment fields.

### II. Macroeconomic Context Layer
Integrate FRED (Federal Reserve Economic Data) macro features into the training matrix:
- Yield Curve Inversion: 10-Year vs 2-Year Treasury spread
- Consumer Price Index (CPI) monthly delta
- Non-Farm Payroll change

Resample and forward-fill to bar frequency. Apply SHAP pruning to confirm predictive contribution before committing to the integration overhead.

### III. Safe Execution Bridge
Once backtesting demonstrates consistent alpha across ≥ 4 Walk-Forward folds and ≥ 3 years of out-of-sample data:
- Connect to paper-trading broker (Alpaca API or Interactive Brokers TWS) via isolated execution bridge.
- Model engine remains air-gapped on local server — only signals cross the bridge, never raw data or model weights.
- All orders submitted as limit orders to cap slippage exposure.
- Hard kill-switch: if live drawdown exceeds 10% of paper capital, halt execution and re-evaluate.

---

## Appendix: Key Formulas Reference

| Formula | Definition |
| :--- | :--- |
| Time-Decay Weight | `w(t) = exp(−0.007 × Δt_minutes)` |
| Net Sentiment | `Σ[w(t) × (P_pos − P_neg)] / Σw(t)` |
| Absolute Emotion | `Σ[w(t) × (P_pos + P_neg)] / Σw(t)` |
| Profit Target (B1) | `Entry Price + (1.5 × ATR_14)` |
| Stop-Loss (B2) | `Entry Price − (1.0 × ATR_14)` |
| Breakeven Win Rate | `SL / (PT + SL) = 1.0 / 2.5 = 40%` |
| Sharpe Ratio | `(R_portfolio − R_risk_free) / σ_portfolio` |
| Profit Factor | `Gross Profits / Gross Losses` |
