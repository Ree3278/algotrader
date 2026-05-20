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

## Next implementation areas

- OHLCV ingestion
- backtest engine with friction
- model training and fold reporting
