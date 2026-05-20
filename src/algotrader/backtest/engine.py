"""Minimal long/flat backtest engine using next-open execution."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class BacktestConfig:
    probability_threshold: float = 0.55
    commission_bps: float = 1.0
    slippage_bps: float = 2.0

    @property
    def total_cost_bps(self) -> float:
        return self.commission_bps + self.slippage_bps


def run_long_flat_backtest(
    price_frame: pd.DataFrame,
    long_probabilities: pd.Series,
    config: BacktestConfig | None = None,
) -> pd.DataFrame:
    """Simulate a long/flat strategy on open-to-open returns.

    A signal produced at close of bar `t` becomes a position at open of `t+1`.
    The resulting position is held until the next open.
    """

    config = config or BacktestConfig()
    if "open" not in price_frame.columns:
        raise ValueError("price_frame must contain an 'open' column")
    if not price_frame.index.is_monotonic_increasing:
        raise ValueError("price_frame index must be sorted ascending")
    if not long_probabilities.index.isin(price_frame.index).all():
        raise ValueError("All probability timestamps must exist in the price frame index")

    opens = price_frame["open"]
    signals = long_probabilities.reindex(price_frame.index)
    target_positions = (signals >= config.probability_threshold).astype(float).fillna(0.0)

    records: list[dict[str, float | pd.Timestamp]] = []
    previous_position = 0.0

    for bar_pos in range(1, len(price_frame) - 1):
        signal_index = price_frame.index[bar_pos - 1]
        entry_index = price_frame.index[bar_pos]
        exit_index = price_frame.index[bar_pos + 1]

        target_position = float(target_positions.iloc[bar_pos - 1])
        turnover = abs(target_position - previous_position)
        open_to_open_return = (opens.iloc[bar_pos + 1] / opens.iloc[bar_pos]) - 1
        gross_return = target_position * open_to_open_return
        transaction_cost = turnover * (config.total_cost_bps / 10_000)
        net_return = gross_return - transaction_cost

        records.append(
            {
                "signal_index": signal_index,
                "entry_index": entry_index,
                "exit_index": exit_index,
                "long_probability": float(signals.iloc[bar_pos - 1]),
                "target_position": target_position,
                "turnover": turnover,
                "gross_return": gross_return,
                "transaction_cost": transaction_cost,
                "net_return": net_return,
            }
        )
        previous_position = target_position

    results = pd.DataFrame(records).set_index("signal_index")
    if results.empty:
        return results

    results["equity_curve"] = (1 + results["net_return"]).cumprod()
    results["benchmark_return"] = (opens.shift(-1) / opens - 1).reindex(results["entry_index"]).to_numpy()
    return results


def summarize_backtest(results: pd.DataFrame) -> dict[str, float]:
    """Compute compact diagnostics from a backtest result frame."""

    if results.empty:
        return {
            "total_return": 0.0,
            "trade_count": 0.0,
            "exposure": 0.0,
            "turnover": 0.0,
        }

    total_return = float(results["equity_curve"].iloc[-1] - 1)
    trade_count = float((results["turnover"] > 0).sum())
    exposure = float(results["target_position"].mean())
    turnover = float(results["turnover"].sum())
    return {
        "total_return": total_return,
        "trade_count": trade_count,
        "exposure": exposure,
        "turnover": turnover,
    }
