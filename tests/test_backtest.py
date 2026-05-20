from __future__ import annotations

import pandas as pd

from algotrader.backtest import BacktestConfig, run_long_flat_backtest, summarize_backtest


def test_backtest_uses_previous_close_signal_for_next_open_position() -> None:
    index = pd.date_range("2024-01-01", periods=5, freq="D", tz="UTC")
    price_frame = pd.DataFrame({"open": [100.0, 102.0, 101.0, 104.0, 103.0]}, index=index)
    probabilities = pd.Series([0.9, 0.2, 0.8, 0.1, 0.1], index=index)

    results = run_long_flat_backtest(
        price_frame,
        probabilities,
        config=BacktestConfig(probability_threshold=0.5, commission_bps=0.0, slippage_bps=0.0),
    )

    assert results.iloc[0]["target_position"] == 1.0
    assert round(results.iloc[0]["gross_return"], 6) == round((101.0 / 102.0) - 1, 6)
    assert results.iloc[1]["target_position"] == 0.0
    assert results.iloc[1]["gross_return"] == 0.0
    assert results.iloc[2]["target_position"] == 1.0
    assert round(results.iloc[2]["gross_return"], 6) == round((103.0 / 104.0) - 1, 6)


def test_backtest_applies_transaction_costs_on_position_changes() -> None:
    index = pd.date_range("2024-01-01", periods=4, freq="D", tz="UTC")
    price_frame = pd.DataFrame({"open": [100.0, 100.0, 101.0, 101.0]}, index=index)
    probabilities = pd.Series([0.9, 0.1, 0.1, 0.1], index=index)
    config = BacktestConfig(probability_threshold=0.5, commission_bps=1.0, slippage_bps=2.0)

    results = run_long_flat_backtest(price_frame, probabilities, config=config)
    summary = summarize_backtest(results)

    assert results.iloc[0]["transaction_cost"] == 0.0003
    assert results.iloc[1]["transaction_cost"] == 0.0003
    assert summary["trade_count"] == 2.0
    assert summary["turnover"] == 2.0
