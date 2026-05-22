from __future__ import annotations

import pandas as pd

from algotrader.backtest import BacktestConfig, run_long_flat_backtest, summarize_backtest


def test_backtest_holds_until_label_exit_and_skips_overlapping_signals() -> None:
    index = pd.date_range("2024-01-01", periods=6, freq="D", tz="UTC")
    price_frame = pd.DataFrame(
        {
            "open": [100.0, 102.0, 104.0, 106.0, 108.0, 109.0],
            "close": [101.0, 103.0, 105.0, 107.0, 109.0, 110.0],
        },
        index=index,
    )
    signal_frame = pd.DataFrame(
        {
            "entry_index": [index[1], index[2], index[4]],
            "exit_index": [index[3], index[3], index[4]],
            "hit_reason": ["profit_target", "profit_target", "timeout"],
            "entry_price": [102.0, 104.0, 108.0],
            "exit_price": [109.0, 110.0, 109.0],
            "realized_return": [(109.0 / 102.0) - 1, (110.0 / 104.0) - 1, (109.0 / 108.0) - 1],
        },
        index=index[:3],
    )
    probabilities = pd.Series([0.9, 0.95, 0.8], index=signal_frame.index)

    results = run_long_flat_backtest(
        price_frame,
        signal_frame,
        probabilities,
        config=BacktestConfig(probability_threshold=0.5, commission_bps=0.0, slippage_bps=0.0),
    )
    summary = summarize_backtest(results)

    assert results.loc[index[1], "position"] == 1.0
    assert results.loc[index[2], "position"] == 1.0
    assert results.loc[index[3], "position"] == 1.0
    assert results.loc[index[4], "position"] == 1.0
    assert summary["trade_count"] == 2.0
    assert summary["turnover"] == 4.0
    assert summary["win_rate"] == 1.0
    assert round(float(results.loc[index[3], "trade_net_return"]), 6) == round((109.0 / 102.0) - 1, 6)


def test_backtest_applies_round_trip_costs_to_trade_returns() -> None:
    index = pd.date_range("2024-01-01", periods=3, freq="D", tz="UTC")
    price_frame = pd.DataFrame(
        {
            "open": [100.0, 100.0, 101.0],
            "close": [100.0, 100.0, 101.0],
        },
        index=index,
    )
    signal_frame = pd.DataFrame(
        {
            "entry_index": [index[1]],
            "exit_index": [index[1]],
            "hit_reason": ["timeout"],
            "entry_price": [100.0],
            "exit_price": [100.0],
            "realized_return": [0.0],
        },
        index=index[:1],
    )
    config = BacktestConfig(probability_threshold=0.5, commission_bps=1.0, slippage_bps=2.0)
    probabilities = pd.Series([0.9], index=signal_frame.index)

    results = run_long_flat_backtest(price_frame, signal_frame, probabilities, config=config)
    summary = summarize_backtest(results)

    assert round(float(results.loc[index[1], "transaction_cost"]), 6) == 0.0006
    assert round(float(results.loc[index[1], "trade_net_return"]), 6) == -0.0006
    assert summary["trade_count"] == 1.0
    assert summary["turnover"] == 2.0
