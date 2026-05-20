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
uv sync --extra dev --extra research
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

## Fetch Data

Download normalized daily OHLCV with `yfinance`:

```bash
uv run algotrader-fetch-yf --symbol SPY --output-csv data/interim/spy_daily.csv
```

This also writes normalized VIX data by default to `data/interim/vix_daily.csv`.

## Train

Train fold models from local CSV:

```bash
uv run --extra research algotrader-train --input-csv data/interim/spy_daily.csv
```

If `data/interim/vix_daily.csv` exists beside the SPY CSV, it is picked up automatically. You can also pass it explicitly:

```bash
uv run --extra research algotrader-train --input-csv data/interim/spy_daily.csv --vix-csv data/interim/vix_daily.csv
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
uv run --extra research algotrader-test --input-csv data/interim/spy_daily.csv --model-dir models/latest
```

If the saved model was trained with VIX features, provide the same VIX CSV or keep `vix_daily.csv` beside the SPY file.

## Inspect Metrics

Compute the debugging metrics for a saved run:

```bash
uv run --extra research algotrader-metrics --input-csv data/interim/spy_daily.csv --model-dir models/latest --reports-dir reports/latest
```

If the run used VIX features:

```bash
uv run --extra research algotrader-metrics --input-csv data/interim/spy_daily.csv --vix-csv data/interim/vix_daily.csv --model-dir models/latest --reports-dir reports/latest
```

This prints:

- label distribution
- B1 / B2 / timeout resolution mix
- chronological per-fold Sharpe values
- averaged feature importances when the backend supports them

## Combined Run

Local CSV:

```bash
uv run --extra research algotrader-run --input-csv data/interim/spy_daily.csv
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
