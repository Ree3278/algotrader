"""Label-parameter sweep utilities and CLI."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, replace
from itertools import product
from pathlib import Path
from typing import Any

import pandas as pd

from algotrader.ingestion import save_ohlcv_csv
from algotrader.pipeline import (
    TestPipelineConfig,
    _add_shared_data_args,
    _add_shared_model_args,
    _load_or_fetch_frames,
    _settings_from_args,
    run_pipeline,
)
from algotrader.reporting import to_json_safe


@dataclass(frozen=True)
class LabelSweepGrid:
    profit_target_atrs: tuple[float, ...] = (1.0, 1.25, 1.5)
    stop_loss_atrs: tuple[float, ...] = (1.0, 1.25, 1.5)
    max_holding_bars: tuple[int, ...] = (5, 10, 15)
    timeout_return_thresholds: tuple[float, ...] = (0.0, 0.001)

    @property
    def run_count(self) -> int:
        return (
            len(self.profit_target_atrs)
            * len(self.stop_loss_atrs)
            * len(self.max_holding_bars)
            * len(self.timeout_return_thresholds)
        )


def _save_timeseries_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=True)


def _materialize_local_inputs(
    base_config: TestPipelineConfig,
    output_dir: Path,
) -> tuple[TestPipelineConfig, dict[str, Path | None]]:
    price_frame, vix_frame, sentiment_frame = _load_or_fetch_frames(base_config)
    input_root = output_dir / "inputs"
    price_csv = input_root / f"{base_config.symbol.lower()}_daily.csv"
    vix_csv = input_root / "vix_daily.csv" if vix_frame is not None else None
    sentiment_csv = input_root / "sentiment_daily.csv" if sentiment_frame is not None else None

    save_ohlcv_csv(price_frame, price_csv)
    if vix_frame is not None and vix_csv is not None:
        save_ohlcv_csv(vix_frame, vix_csv)
    if sentiment_frame is not None and sentiment_csv is not None:
        _save_timeseries_csv(sentiment_frame, sentiment_csv)

    local_config = replace(
        base_config,
        input_csv=price_csv,
        vix_input_csv=vix_csv,
        sentiment_features_csv=sentiment_csv,
        auto_discover_companion_inputs=False,
        fetch_yfinance=False,
        fetch_alpha_vantage=False,
    )
    return local_config, {"input_csv": price_csv, "vix_csv": vix_csv, "sentiment_csv": sentiment_csv}


def _format_slug_value(value: float | int) -> str:
    if isinstance(value, int):
        return str(value)
    return f"{value:.3f}".rstrip("0").rstrip(".").replace(".", "p")


def _run_slug(
    *,
    profit_target_atr: float,
    stop_loss_atr: float,
    max_holding_bars: int,
    timeout_return_threshold: float,
) -> str:
    return (
        f"pt_{_format_slug_value(profit_target_atr)}"
        f"_sl_{_format_slug_value(stop_loss_atr)}"
        f"_hold_{max_holding_bars}"
        f"_to_{_format_slug_value(timeout_return_threshold)}"
    )


def _distribution_pct(series: pd.Series, key: Any) -> float:
    return round(100 * float(series.get(key, 0.0)), 2)


def _top_result_lines(results: pd.DataFrame, limit: int = 10) -> list[str]:
    if results.empty:
        return ["No runs completed."]

    ordered = results.sort_values(
        by=["mean_sharpe", "mean_total_return", "mean_max_drawdown"],
        ascending=[False, False, False],
    ).head(limit)
    lines = ["Top configs by mean Sharpe:"]
    for row in ordered.itertuples(index=False):
        lines.append(
            "  "
            f"{row.run_slug}: "
            f"sharpe={row.mean_sharpe:.3f}, "
            f"return={100 * row.mean_total_return:.2f}%, "
            f"dd={100 * row.mean_max_drawdown:.2f}%, "
            f"trades={row.mean_trade_count:.2f}"
        )
    return lines


def run_label_sweep(
    base_config: TestPipelineConfig,
    *,
    output_dir: str | Path,
    grid: LabelSweepGrid = LabelSweepGrid(),
) -> tuple[pd.DataFrame, dict[str, Path]]:
    """Run a full train/test sweep across multiple label-parameter combinations."""

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    local_base_config, materialized_inputs = _materialize_local_inputs(base_config, destination)

    rows: list[dict[str, Any]] = []
    for profit_target_atr, stop_loss_atr, max_holding_bars, timeout_return_threshold in product(
        grid.profit_target_atrs,
        grid.stop_loss_atrs,
        grid.max_holding_bars,
        grid.timeout_return_thresholds,
    ):
        run_slug = _run_slug(
            profit_target_atr=profit_target_atr,
            stop_loss_atr=stop_loss_atr,
            max_holding_bars=max_holding_bars,
            timeout_return_threshold=timeout_return_threshold,
        )
        run_root = destination / "runs" / run_slug
        label_settings = replace(
            local_base_config.settings.labels,
            profit_target_atr=profit_target_atr,
            stop_loss_atr=stop_loss_atr,
            max_holding_bars=max_holding_bars,
            timeout_return_threshold=timeout_return_threshold,
        )
        run_settings = replace(local_base_config.settings, labels=label_settings)
        run_config = replace(
            local_base_config,
            settings=run_settings,
            auto_discover_companion_inputs=False,
            model_dir=run_root / "models",
            output_dir=run_root / "reports",
        )

        result = run_pipeline(run_config)
        label_distribution = result.dataset["label"].value_counts(normalize=True).sort_index()
        hit_reason_distribution = result.dataset["hit_reason"].value_counts(normalize=True)

        rows.append(
            {
                "run_slug": run_slug,
                "profit_target_atr": profit_target_atr,
                "stop_loss_atr": stop_loss_atr,
                "max_holding_bars": max_holding_bars,
                "timeout_return_threshold": timeout_return_threshold,
                "dataset_rows": int(len(result.dataset)),
                "fold_count": int(result.summary["fold_count"]),
                "mean_total_return": float(result.summary.get("mean_total_return", 0.0)),
                "mean_sharpe": float(result.summary.get("mean_sharpe", 0.0)),
                "mean_trade_count": float(result.summary.get("mean_trade_count", 0.0)),
                "mean_max_drawdown": float(result.summary.get("mean_max_drawdown", 0.0)),
                "label_flat_pct": _distribution_pct(label_distribution, 0),
                "label_long_pct": _distribution_pct(label_distribution, 1),
                "hit_profit_target_pct": _distribution_pct(hit_reason_distribution, "profit_target"),
                "hit_stop_loss_pct": _distribution_pct(hit_reason_distribution, "stop_loss"),
                "hit_timeout_pct": _distribution_pct(hit_reason_distribution, "timeout"),
                "model_dir": str(run_config.model_dir),
                "reports_dir": str(run_config.output_dir),
            }
        )

    results = pd.DataFrame(rows)
    results = results.sort_values(
        by=["mean_sharpe", "mean_total_return", "mean_max_drawdown"],
        ascending=[False, False, False],
    ).reset_index(drop=True)

    csv_path = destination / "label_sweep_results.csv"
    json_path = destination / "label_sweep_results.json"
    summary_path = destination / "label_sweep_summary.json"
    results.to_csv(csv_path, index=False)
    records = to_json_safe(results.to_dict(orient="records"))
    json_path.write_text(json.dumps(records, indent=2), encoding="utf-8")

    summary_payload = {
        "run_count": int(len(results)),
        "materialized_inputs": {key: None if value is None else str(value) for key, value in materialized_inputs.items()},
        "best_by_mean_sharpe": None if results.empty else results.iloc[0].to_dict(),
        "top_configs": [] if results.empty else to_json_safe(results.head(10).to_dict(orient="records")),
    }
    summary_path.write_text(json.dumps(to_json_safe(summary_payload), indent=2), encoding="utf-8")
    return results, {"csv": csv_path, "json": json_path, "summary": summary_path}


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Sweep triple-barrier label parameters across train/test runs.")
    _add_shared_data_args(parser)
    _add_shared_model_args(parser)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports/label_sweep/latest"),
        help="Directory for aggregate sweep outputs and per-run artifacts",
    )
    parser.add_argument(
        "--profit-target-atrs",
        type=float,
        nargs="+",
        default=list(LabelSweepGrid().profit_target_atrs),
        help="Profit target ATR multipliers to sweep",
    )
    parser.add_argument(
        "--stop-loss-atrs",
        type=float,
        nargs="+",
        default=list(LabelSweepGrid().stop_loss_atrs),
        help="Stop loss ATR multipliers to sweep",
    )
    parser.add_argument(
        "--max-holding-bars",
        type=int,
        nargs="+",
        default=list(LabelSweepGrid().max_holding_bars),
        help="Max holding periods in bars to sweep",
    )
    parser.add_argument(
        "--timeout-thresholds",
        type=float,
        nargs="+",
        default=list(LabelSweepGrid().timeout_return_thresholds),
        help="Timeout return thresholds to sweep",
    )
    return parser


def _config_from_args(args: argparse.Namespace) -> TestPipelineConfig:
    return TestPipelineConfig(
        symbol=args.symbol,
        input_csv=args.input_csv,
        vix_input_csv=args.vix_csv,
        sentiment_features_csv=args.sentiment_features_csv,
        fetch_yfinance=args.fetch_yfinance,
        yfinance_period=args.yf_period,
        yfinance_start=args.yf_start,
        yfinance_end=args.yf_end,
        fetch_alpha_vantage=args.fetch_alpha_vantage,
        alpha_vantage_key=args.alpha_vantage_key,
        settings=_settings_from_args(args),
    )


def main() -> None:
    args = build_arg_parser().parse_args()
    grid = LabelSweepGrid(
        profit_target_atrs=tuple(args.profit_target_atrs),
        stop_loss_atrs=tuple(args.stop_loss_atrs),
        max_holding_bars=tuple(args.max_holding_bars),
        timeout_return_thresholds=tuple(args.timeout_thresholds),
    )
    results, paths = run_label_sweep(_config_from_args(args), output_dir=args.output_dir, grid=grid)
    print(f"Completed {grid.run_count} label-sweep runs.")
    for line in _top_result_lines(results):
        print(line)
    print(f"results_csv={paths['csv']}")
    print(f"results_json={paths['json']}")
    print(f"summary_json={paths['summary']}")
