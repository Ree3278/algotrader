"""Event-driven long/flat backtest aligned to triple-barrier labels."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class BacktestConfig:
    probability_threshold: float = 0.55
    commission_bps: float = 1.0
    slippage_bps: float = 2.0

    @property
    def total_cost_bps(self) -> float:
        return self.commission_bps + self.slippage_bps

    @property
    def cost_per_side(self) -> float:
        return self.total_cost_bps / 10_000


REQUIRED_SIGNAL_COLUMNS = {
    "entry_index",
    "exit_index",
    "hit_reason",
    "entry_price",
    "exit_price",
    "realized_return",
}


def _build_benchmark_returns(window: pd.DataFrame) -> pd.Series:
    benchmark = pd.Series(0.0, index=window.index, dtype=float)
    if window.empty:
        return benchmark

    closes = window["close"]
    benchmark.iloc[0] = float((closes.iloc[0] / window["open"].iloc[0]) - 1)
    if len(window) > 1:
        benchmark.iloc[1:] = closes.iloc[1:].to_numpy() / closes.iloc[:-1].to_numpy() - 1
    return benchmark


def _build_trade_daily_returns(
    price_frame: pd.DataFrame,
    *,
    entry_pos: int,
    exit_pos: int,
    entry_price: float,
    exit_price: float,
) -> pd.Series:
    trade_index = price_frame.index[entry_pos : exit_pos + 1]
    trade_returns = pd.Series(0.0, index=trade_index, dtype=float)
    closes = price_frame["close"]

    if entry_pos == exit_pos:
        trade_returns.iloc[0] = float((exit_price / entry_price) - 1)
        return trade_returns

    trade_returns.iloc[0] = float((closes.iloc[entry_pos] / entry_price) - 1)
    for price_pos in range(entry_pos + 1, exit_pos):
        trade_returns.loc[price_frame.index[price_pos]] = float(
            (closes.iloc[price_pos] / closes.iloc[price_pos - 1]) - 1
        )
    trade_returns.iloc[-1] = float((exit_price / closes.iloc[exit_pos - 1]) - 1)
    return trade_returns


def run_long_flat_backtest(
    price_frame: pd.DataFrame,
    signal_frame: pd.DataFrame,
    long_probabilities: pd.Series,
    config: BacktestConfig | None = None,
    threshold_series: pd.Series | None = None,
) -> pd.DataFrame:
    """Simulate one-position event-driven trades from label-aligned signal metadata.

    A signal is generated at close of bar `t`, entered at open of `t+1`, and
    then held until the precomputed label exit. Signals that fire while a trade
    is already open are ignored because the current system is long/flat, not
    long/stacked.
    """

    config = config or BacktestConfig()
    required_price_columns = {"open", "close"}
    missing_price_columns = sorted(required_price_columns.difference(price_frame.columns))
    if missing_price_columns:
        raise ValueError(f"price_frame is missing required columns: {missing_price_columns}")
    missing_signal_columns = sorted(REQUIRED_SIGNAL_COLUMNS.difference(signal_frame.columns))
    if missing_signal_columns:
        raise ValueError(f"signal_frame is missing required columns: {missing_signal_columns}")
    if not price_frame.index.is_monotonic_increasing:
        raise ValueError("price_frame index must be sorted ascending")
    if not signal_frame.index.is_monotonic_increasing:
        raise ValueError("signal_frame index must be sorted ascending")
    if not long_probabilities.index.isin(signal_frame.index).all():
        raise ValueError("All probability timestamps must exist in the signal frame index")
    if threshold_series is not None and not threshold_series.index.isin(signal_frame.index).all():
        raise ValueError("All threshold timestamps must exist in the signal frame index")

    valid_signals = signal_frame.dropna(subset=["entry_index", "exit_index", "entry_price", "exit_price"]).copy()
    if valid_signals.empty:
        return pd.DataFrame()

    evaluation_start = pd.Timestamp(valid_signals["entry_index"].min())
    evaluation_end = pd.Timestamp(valid_signals["exit_index"].max())
    evaluation_window = price_frame.loc[evaluation_start:evaluation_end].copy()
    if evaluation_window.empty:
        return pd.DataFrame()

    results = pd.DataFrame(
        index=evaluation_window.index,
        data={
            "position": 0.0,
            "turnover": 0.0,
            "gross_return": 0.0,
            "transaction_cost": 0.0,
            "net_return": 0.0,
            "benchmark_return": _build_benchmark_returns(evaluation_window),
            "is_trade_entry": False,
            "is_trade_exit": False,
            "trade_net_return": np.nan,
            "trade_id": pd.Series(pd.NA, index=evaluation_window.index, dtype="Int64"),
            "entry_signal_index": pd.Series(pd.NaT, index=evaluation_window.index, dtype=evaluation_window.index.dtype),
            "exit_reason": pd.Series(pd.NA, index=evaluation_window.index, dtype="object"),
        },
    )

    signal_probabilities = long_probabilities.reindex(valid_signals.index)
    signal_thresholds = (
        threshold_series.reindex(valid_signals.index).astype(float)
        if threshold_series is not None
        else pd.Series(config.probability_threshold, index=valid_signals.index, dtype=float)
    )
    active_exit_pos = -1
    trade_id = 0

    for signal_index, signal_row in valid_signals.iterrows():
        probability = signal_probabilities.get(signal_index)
        threshold = signal_thresholds.get(signal_index, config.probability_threshold)
        if pd.isna(probability) or pd.isna(threshold) or float(probability) < float(threshold):
            continue

        entry_index = pd.Timestamp(signal_row["entry_index"])
        exit_index = pd.Timestamp(signal_row["exit_index"])
        if entry_index not in price_frame.index or exit_index not in price_frame.index:
            continue

        entry_pos = int(price_frame.index.get_loc(entry_index))
        exit_pos = int(price_frame.index.get_loc(exit_index))
        if entry_pos <= active_exit_pos:
            continue

        trade_id += 1
        trade_returns = _build_trade_daily_returns(
            price_frame,
            entry_pos=entry_pos,
            exit_pos=exit_pos,
            entry_price=float(signal_row["entry_price"]),
            exit_price=float(signal_row["exit_price"]),
        )

        results.loc[trade_returns.index, "position"] = 1.0
        results.loc[trade_returns.index, "gross_return"] += trade_returns.to_numpy()
        results.loc[trade_returns.index, "trade_id"] = trade_id
        results.loc[trade_returns.index, "entry_signal_index"] = signal_index

        results.at[entry_index, "turnover"] += 1.0
        results.at[exit_index, "turnover"] += 1.0
        results.at[entry_index, "transaction_cost"] += config.cost_per_side
        results.at[exit_index, "transaction_cost"] += config.cost_per_side
        results.at[entry_index, "is_trade_entry"] = True
        results.at[exit_index, "is_trade_exit"] = True
        results.at[exit_index, "exit_reason"] = signal_row["hit_reason"]

        trade_daily_net = trade_returns.copy()
        trade_daily_net.loc[entry_index] -= config.cost_per_side
        trade_daily_net.loc[exit_index] -= config.cost_per_side
        results.at[exit_index, "trade_net_return"] = float((1 + trade_daily_net).prod() - 1)

        active_exit_pos = exit_pos

    results["net_return"] = results["gross_return"] - results["transaction_cost"]
    results["equity_curve"] = (1 + results["net_return"]).cumprod()
    results["benchmark_equity_curve"] = (1 + results["benchmark_return"]).cumprod()
    return results


def summarize_backtest(results: pd.DataFrame) -> dict[str, float]:
    """Compute daily and trade-level diagnostics from event-driven backtest output."""

    if results.empty:
        return {
            "total_return": 0.0,
            "benchmark_total_return": 0.0,
            "cagr": 0.0,
            "sharpe": 0.0,
            "max_drawdown": 0.0,
            "profit_factor": 0.0,
            "win_rate": 0.0,
            "trade_count": 0.0,
            "exposure": 0.0,
            "turnover": 0.0,
        }

    net_returns = results["net_return"]
    total_return = float(results["equity_curve"].iloc[-1] - 1)
    benchmark_total_return = float(results["benchmark_equity_curve"].iloc[-1] - 1)
    periods = len(results)
    cagr = float(results["equity_curve"].iloc[-1] ** (252 / periods) - 1) if periods > 0 else 0.0
    volatility = float(net_returns.std(ddof=0))
    sharpe = float(np.sqrt(252) * net_returns.mean() / volatility) if volatility > 0 else 0.0
    running_peak = results["equity_curve"].cummax()
    drawdown = (results["equity_curve"] / running_peak) - 1
    max_drawdown = float(drawdown.min())

    trade_returns = results.loc[results["is_trade_exit"], "trade_net_return"].dropna()
    positive_trade_returns = trade_returns[trade_returns > 0].sum()
    negative_trade_returns = trade_returns[trade_returns < 0].sum()
    if negative_trade_returns < 0:
        profit_factor = float(positive_trade_returns / abs(negative_trade_returns))
    elif positive_trade_returns > 0:
        profit_factor = float("inf")
    else:
        profit_factor = 0.0

    win_rate = float((trade_returns > 0).mean()) if not trade_returns.empty else 0.0
    trade_count = float(len(trade_returns))
    exposure = float(results["position"].mean())
    turnover = float(results["turnover"].sum())

    return {
        "total_return": total_return,
        "benchmark_total_return": benchmark_total_return,
        "cagr": cagr,
        "sharpe": sharpe,
        "max_drawdown": max_drawdown,
        "profit_factor": profit_factor,
        "win_rate": win_rate,
        "trade_count": trade_count,
        "exposure": exposure,
        "turnover": turnover,
    }
