from __future__ import annotations

import pandas as pd

from algotrader.labels import TripleBarrierConfig, generate_long_flat_labels


def _frame(rows: list[dict[str, float]]) -> pd.DataFrame:
    index = pd.date_range("2024-01-01", periods=len(rows), freq="D", tz="UTC")
    return pd.DataFrame(rows, index=index)


def test_generates_profit_target_label_from_next_open_entry() -> None:
    frame = _frame(
        [
            {"open": 100, "high": 101, "low": 99, "close": 100, "ATR_14": 2},
            {"open": 101, "high": 104.5, "low": 100.5, "close": 104, "ATR_14": 2},
            {"open": 104, "high": 105, "low": 103, "close": 104.5, "ATR_14": 2},
        ]
    )
    config = TripleBarrierConfig(max_holding_bars=2)

    labels = generate_long_flat_labels(frame, config=config)

    first_row = labels.iloc[0]
    assert first_row["label"] == 1
    assert first_row["hit_reason"] == "profit_target"
    assert first_row["entry_price"] == 101
    assert first_row["exit_price"] == 104
    assert first_row["entry_index"] == frame.index[1]


def test_generates_flat_label_when_stop_loss_hits_first() -> None:
    frame = _frame(
        [
            {"open": 100, "high": 100.5, "low": 99.5, "close": 100, "ATR_14": 2},
            {"open": 100, "high": 101, "low": 97.5, "close": 98, "ATR_14": 2},
            {"open": 98, "high": 99, "low": 97, "close": 98.5, "ATR_14": 2},
        ]
    )
    config = TripleBarrierConfig(max_holding_bars=2)

    labels = generate_long_flat_labels(frame, config=config)

    first_row = labels.iloc[0]
    assert first_row["label"] == 0
    assert first_row["hit_reason"] == "stop_loss"
    assert first_row["exit_price"] == 98


def test_timeout_label_stays_flat_below_threshold() -> None:
    frame = _frame(
        [
            {"open": 100, "high": 100.5, "low": 99.5, "close": 100, "ATR_14": 10},
            {"open": 100, "high": 100.4, "low": 99.7, "close": 100.0, "ATR_14": 10},
            {"open": 100.0, "high": 100.3, "low": 99.9, "close": 100.05, "ATR_14": 10},
            {"open": 100.05, "high": 100.2, "low": 100.0, "close": 100.08, "ATR_14": 10},
        ]
    )
    config = TripleBarrierConfig(max_holding_bars=3, timeout_return_threshold=0.001)

    labels = generate_long_flat_labels(frame, config=config)

    first_row = labels.iloc[0]
    assert first_row["label"] == 0
    assert first_row["hit_reason"] == "timeout"


def test_intrabar_tie_break_defaults_to_stop_for_conservative_label() -> None:
    frame = _frame(
        [
            {"open": 100, "high": 100.2, "low": 99.8, "close": 100, "ATR_14": 1},
            {"open": 100, "high": 101.6, "low": 98.9, "close": 100.5, "ATR_14": 1},
            {"open": 100.5, "high": 100.8, "low": 100.1, "close": 100.4, "ATR_14": 1},
        ]
    )
    config = TripleBarrierConfig(max_holding_bars=2)

    labels = generate_long_flat_labels(frame, config=config)

    first_row = labels.iloc[0]
    assert first_row["label"] == 0
    assert first_row["hit_reason"] == "stop_loss"
