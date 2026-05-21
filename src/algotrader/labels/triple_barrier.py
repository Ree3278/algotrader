"""Binary triple-barrier labels aligned to next-open execution."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class TripleBarrierConfig:
    profit_target_atr: float = 1.5
    stop_loss_atr: float = 2.0
    max_holding_bars: int = 10
    timeout_return_threshold: float = 0.001
    intrabar_tie_break: str = "stop"


def _validate_frame(frame: pd.DataFrame) -> None:
    required = {"open", "high", "low", "close", "ATR_14"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    if not frame.index.is_monotonic_increasing:
        raise ValueError("Price frame index must be sorted ascending")
    if frame.index.has_duplicates:
        raise ValueError("Price frame index must not contain duplicates")


def _resolve_intrabar_hit(
    profit_hit: bool,
    stop_hit: bool,
    config: TripleBarrierConfig,
) -> str | None:
    if profit_hit and stop_hit:
        if config.intrabar_tie_break == "stop":
            return "stop_loss"
        if config.intrabar_tie_break == "profit":
            return "profit_target"
        raise ValueError("intrabar_tie_break must be 'stop' or 'profit'")
    if profit_hit:
        return "profit_target"
    if stop_hit:
        return "stop_loss"
    return None


def generate_long_flat_labels(
    frame: pd.DataFrame,
    config: TripleBarrierConfig | None = None,
) -> pd.DataFrame:
    """Generate binary long/flat labels for signal timestamps.

    Label rows are indexed by the signal timestamp. Entry happens at the next
    bar open, which keeps labels aligned to the planned execution semantics.
    """

    config = config or TripleBarrierConfig()
    _validate_frame(frame)

    result = pd.DataFrame(
        index=frame.index,
        data={"label": np.nan, "hit_reason": None, "entry_price": np.nan, "exit_price": np.nan, "realized_return": np.nan},
    )
    datetime_dtype = frame.index.dtype
    result["entry_index"] = pd.Series(pd.NaT, index=frame.index, dtype=datetime_dtype)
    result["exit_index"] = pd.Series(pd.NaT, index=frame.index, dtype=datetime_dtype)
    result["event_end_index"] = pd.Series(pd.NaT, index=frame.index, dtype=datetime_dtype)

    final_signal_idx = len(frame) - config.max_holding_bars - 1
    if final_signal_idx < 0:
        return result

    for signal_pos in range(final_signal_idx + 1):
        entry_pos = signal_pos + 1
        entry_price = frame["open"].iloc[entry_pos]
        atr_value = frame["ATR_14"].iloc[signal_pos]

        if pd.isna(entry_price) or pd.isna(atr_value):
            continue

        profit_level = entry_price + (config.profit_target_atr * atr_value)
        stop_level = entry_price - (config.stop_loss_atr * atr_value)
        last_eval_pos = min(entry_pos + config.max_holding_bars - 1, len(frame) - 1)

        exit_price = np.nan
        exit_pos = last_eval_pos
        hit_reason = "timeout"
        label = 0

        for bar_pos in range(entry_pos, last_eval_pos + 1):
            bar_high = frame["high"].iloc[bar_pos]
            bar_low = frame["low"].iloc[bar_pos]

            resolved_hit = _resolve_intrabar_hit(
                profit_hit=bar_high >= profit_level,
                stop_hit=bar_low <= stop_level,
                config=config,
            )
            if resolved_hit is None:
                continue

            exit_pos = bar_pos
            hit_reason = resolved_hit
            if resolved_hit == "profit_target":
                exit_price = profit_level
                label = 1
            else:
                exit_price = stop_level
                label = 0
            break
        else:
            exit_price = frame["close"].iloc[last_eval_pos]
            realized_timeout_return = (exit_price / entry_price) - 1
            label = int(realized_timeout_return > config.timeout_return_threshold)

        realized_return = (exit_price / entry_price) - 1
        signal_index = frame.index[signal_pos]
        result.at[signal_index, "label"] = int(label)
        result.at[signal_index, "entry_index"] = frame.index[entry_pos]
        result.at[signal_index, "exit_index"] = frame.index[exit_pos]
        result.at[signal_index, "event_end_index"] = frame.index[last_eval_pos]
        result.at[signal_index, "hit_reason"] = hit_reason
        result.at[signal_index, "entry_price"] = float(entry_price)
        result.at[signal_index, "exit_price"] = float(exit_price)
        result.at[signal_index, "realized_return"] = float(realized_return)

    return result
