# algotrader

SPY daily ML trading baseline built around a narrow first milestone:

- SPY only
- daily bars only
- long / flat only
- price-only features first
- purged walk-forward validation

## Environment

This project uses `uv` as the only environment and dependency workflow.

```bash
uv sync --extra dev --extra research --extra nlp
uv run pytest
```

## Current baseline modules

- technical price feature engineering
- causal 200-day SMA regime flag
- binary triple-barrier labeling aligned to next-open execution
- purged walk-forward split generation
- walk-forward model training with fold-level backtest metrics
- separate `train` and `test` pipeline stages
- yfinance fetch script for daily OHLCV data
- standalone metrics/debug script for saved runs
- optional VIX regime feature via local CSV or automatic `^VIX` fetch
- optional FinBERT-based sentiment feature layer
- centralized defaults in `src/algotrader/settings.py`

## Settings

Project defaults now live in `src/algotrader/settings.py`.

That file is the source of truth for:

- symbol and data-path defaults
- triple-barrier label settings
- walk-forward split sizes
- model defaults
- backtest costs and threshold defaults
- experiment calibration settings
- default model profile
- default threshold policy

Current default label geometry:

- `profit_target_atr = 1.25`
- `stop_loss_atr = 1.25`
- `max_holding_bars = 10`
- `timeout_return_threshold = 0.0`

Current default model profile:

- `price_plus_regime_plus_trend_state`

Current default threshold policy:

- `global`

## Model Profiles

Feature improvements are now composed as reusable blocks in `src/algotrader/profiles.py`.

Available blocks:

- `price_only`
- `regime`
- `trend_state`
- `vol_state`
- `sentiment`

Preset profiles:

- `price_only`
- `price_plus_regime`
- `price_plus_regime_plus_trend_state`
- `price_plus_regime_plus_trend_state_plus_vol_state`
- `price_plus_regime_plus_sentiment`

The CLI uses `--profile` to select one of these presets. If you do not pass `--profile`, the default is `price_plus_regime_plus_trend_state`.

## Threshold Policies

Threshold selection is also composable. Policies live in `src/algotrader/thresholds.py`.

Available policies:

- `global`
- `trend_regime`

`trend_regime` calibrates one threshold when both `price_above_sma_200` and `sma_50_above_sma_200` are true, and another threshold otherwise.

## Fetch Data

Download normalized daily OHLCV with `yfinance`:

```bash
uv run algotrader-fetch-yf
```

This also writes normalized VIX data by default to `data/interim/vix_daily.csv`.

## Build Sentiment Features

Collect raw market news from Alpha Vantage:

```bash
export ALPHA_VANTAGE_API_KEY=...
uv run algotrader-fetch-news \
  --tickers SPY \
  --topics financial_markets,economy_macro \
  --time-from 20240101T0000 \
  --output-csv data/raw/news/news.csv \
  --output-json data/raw/news/news_raw.json
```

Turn raw news into daily sentiment features:

```bash
uv run --extra nlp algotrader-build-sentiment \
  --news-csv data/raw/news/news.csv \
  --price-csv data/interim/spy_daily.csv \
  --output-csv data/interim/sentiment_daily.csv \
  --scored-news-csv data/interim/news_scored.csv
```

Raw news CSV should include at least:

- `timestamp`
- `headline`

Optional raw news columns:

- `summary`
- `source`
- `url`

## Train

Train fold models from local CSV:

```bash
uv run --extra research algotrader-train
```

By default this uses:

- `data/interim/spy_daily.csv`
- companion `data/interim/vix_daily.csv` if the selected profile needs it
- companion `data/interim/sentiment_daily.csv` if the selected profile needs it

To train a different preset profile:

```bash
uv run --extra research algotrader-train --profile price_plus_regime
```

To try regime-conditional thresholding on the current baseline:

```bash
uv run --extra research algotrader-train --threshold-policy trend_regime
uv run --extra research algotrader-test
```

You can still override paths explicitly if needed:

```bash
uv run --extra research algotrader-train \
  --input-csv data/interim/spy_daily.csv \
  --vix-csv data/interim/vix_daily.csv \
  --sentiment-features-csv data/interim/sentiment_daily.csv
```

Train directly from `yfinance` without a pre-existing CSV:

```bash
uv run --extra research algotrader-train --fetch-yfinance --symbol SPY
```

Training artifacts are written under `models/latest/` by default:

- `manifest.json`
- `fold_manifest.csv`
- `fold_XXX.pkl`

## Test

Evaluate saved fold models and write reports:

```bash
uv run --extra research algotrader-test
```

The terminal output now prints a compact summary with:

- mean total return
- mean Sharpe
- mean trade count
- mean max drawdown
- label distribution
- hit-reason distribution

The test pipeline now reloads the saved profile and feature list from the training manifest, so in the common case you do not need to repeat those details manually.

## Inspect Metrics

Compute the debugging metrics for a saved run:

```bash
uv run --extra research algotrader-metrics \
  --input-csv data/interim/spy_daily.csv \
  --vix-csv data/interim/vix_daily.csv \
  --sentiment-features-csv data/interim/sentiment_daily.csv \
  --model-dir models/latest \
  --reports-dir reports/latest
```

This prints:

- label distribution
- B1 / B2 / timeout resolution mix
- chronological per-fold Sharpe values
- averaged feature importances when the backend supports them

## Label Sweep

Run a controlled sweep over the triple-barrier label geometry:

```bash
uv run --extra research algotrader-label-sweep
```

Default sweep grid:

- `profit_target_atr`: `1.0`, `1.25`, `1.5`
- `stop_loss_atr`: `1.0`, `1.25`, `1.5`
- `max_holding_bars`: `5`, `10`, `15`
- `timeout_return_threshold`: `0.0`, `0.001`

The sweep writes:

- aggregate results CSV: `label_sweep_results.csv`
- aggregate results JSON: `label_sweep_results.json`
- top-config summary: `label_sweep_summary.json`
- per-run model/report artifacts under `runs/<config_slug>/`

To run a smaller sweep:

```bash
uv run --extra research algotrader-label-sweep \
  --profile price_only \
  --profit-target-atrs 1.0 1.25 \
  --stop-loss-atrs 1.0 1.25 \
  --max-holding-bars 5 10 \
  --timeout-thresholds 0.0 0.001
```

## Feature Ablation

Compare the preset feature profiles under the same label configuration:

```bash
uv run --extra research algotrader-ablation
```

The ablation now also includes a threshold-policy comparison on the `price_plus_regime_plus_trend_state` baseline:

- `price_plus_regime_plus_trend_state`
- `price_plus_regime_plus_trend_state_plus_regime_thresholding`

This runs:

- `price_only`
- `price_plus_regime`
- `price_plus_regime_plus_trend_state`
- `price_plus_regime_plus_trend_state_plus_vol_state`
- `price_plus_regime_plus_sentiment`

and writes:

- `ablation_results.csv`
- `ablation_results.json`
- `ablation_summary.json`

The terminal output is now a fixed-width comparison table so you can scan metrics across profiles quickly.

The trend-state variant adds richer SMA context on top of the regime baseline:

- `price_to_sma_200`
- `sma_200_slope_20d`
- `sma_50_above_sma_200`

The volatility-state variant adds:

- `atr_percentile_252d`
- `bb_bandwidth_percentile_252d`
- `volatility_20d_zscore_252d`

## Combined Run

Local default CSV:

```bash
uv run --extra research algotrader-run
```

Alpha Vantage:

```bash
export ALPHA_VANTAGE_API_KEY=...
uv run --extra research algotrader-run --fetch-alpha-vantage --symbol SPY
```

Artifacts are written under `reports/latest/` by default:

- `fold_summaries.csv`
- `test_predictions.csv`
- `summary.json`
