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
- binary triple-barrier labeling aligned to next-open execution
- purged walk-forward split generation
- walk-forward model training with fold-level backtest metrics
- runnable pipeline CLI that writes reports

## Run The Pipeline

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
