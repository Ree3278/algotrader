"""Runnable pipeline for the SPY daily walk-forward baseline."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from algotrader.backtest import BacktestConfig
from algotrader.ingestion import (
    fetch_daily_adjusted,
    load_ohlcv_csv,
    normalize_daily_adjusted,
    save_json,
    save_ohlcv_csv,
)
from algotrader.reporting import build_experiment_summary, write_experiment_reports
from algotrader.training.dataset import build_training_dataset
from algotrader.training.experiment import WalkForwardExperimentConfig, run_walk_forward_experiment
from algotrader.training.walk_forward import PurgedWalkForwardConfig
from algotrader.training.xgboost_model import XGBoostConfig


DEFAULT_SPLIT_CONFIG = PurgedWalkForwardConfig(
    train_size=126,
    test_size=63,
    step_size=63,
    embargo_size=10,
    max_label_horizon=10,
)


@dataclass(frozen=True)
class PipelineConfig:
    symbol: str = "SPY"
    output_dir: Path = Path("reports/latest")
    input_csv: Path | None = None
    fetch_alpha_vantage: bool = False
    alpha_vantage_key: str | None = None
    alpha_vantage_outputsize: str = "full"
    raw_data_dir: Path = Path("data/raw/ohlcv")
    normalized_data_dir: Path = Path("data/interim")
    experiment_config: WalkForwardExperimentConfig = field(
        default_factory=lambda: WalkForwardExperimentConfig(split_config=DEFAULT_SPLIT_CONFIG)
    )


@dataclass(frozen=True)
class PipelineRunResult:
    dataset: pd.DataFrame
    fold_summaries: pd.DataFrame
    test_predictions: pd.DataFrame
    summary: dict[str, Any]
    report_paths: dict[str, Path]
    price_frame: pd.DataFrame


def _load_or_fetch_price_frame(config: PipelineConfig) -> pd.DataFrame:
    if config.input_csv is not None:
        return load_ohlcv_csv(config.input_csv)

    if not config.fetch_alpha_vantage:
        raise ValueError("Provide --input-csv or enable --fetch-alpha-vantage")

    api_key = config.alpha_vantage_key or os.getenv("ALPHA_VANTAGE_API_KEY")
    if not api_key:
        raise ValueError("Alpha Vantage API key not found. Set ALPHA_VANTAGE_API_KEY or pass --alpha-vantage-key.")

    payload = fetch_daily_adjusted(
        config.symbol,
        api_key,
        outputsize=config.alpha_vantage_outputsize,
    )
    price_frame = normalize_daily_adjusted(payload, symbol=config.symbol)

    raw_path = config.raw_data_dir / f"{config.symbol.lower()}_daily_adjusted.json"
    normalized_path = config.normalized_data_dir / f"{config.symbol.lower()}_daily.csv"
    save_json(payload, raw_path)
    save_ohlcv_csv(price_frame, normalized_path)
    return price_frame


def run_pipeline(config: PipelineConfig) -> PipelineRunResult:
    """Execute the end-to-end local research pipeline."""

    price_frame = _load_or_fetch_price_frame(config)
    dataset = build_training_dataset(price_frame)
    result = run_walk_forward_experiment(
        dataset,
        price_frame,
        config=config.experiment_config,
    )
    model_backend = None
    if not result.fold_summaries.empty and "model_backend" in result.fold_summaries.columns:
        model_backend = str(result.fold_summaries["model_backend"].mode(dropna=True).iloc[0])

    summary = build_experiment_summary(
        result,
        symbol=config.symbol,
        dataset_rows=len(dataset.data),
        feature_count=len(dataset.feature_columns),
        model_backend=model_backend,
    )
    report_paths = write_experiment_reports(result, config.output_dir, summary=summary)

    return PipelineRunResult(
        dataset=dataset.data,
        fold_summaries=result.fold_summaries,
        test_predictions=result.test_predictions,
        summary=summary,
        report_paths=report_paths,
        price_frame=price_frame,
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the algotrader walk-forward baseline pipeline.")
    parser.add_argument("--symbol", default="SPY", help="Ticker symbol to analyze")
    parser.add_argument("--input-csv", type=Path, help="Path to normalized OHLCV CSV")
    parser.add_argument(
        "--fetch-alpha-vantage",
        action="store_true",
        help="Fetch daily adjusted OHLCV from Alpha Vantage instead of reading a local CSV",
    )
    parser.add_argument("--alpha-vantage-key", help="Alpha Vantage API key. Defaults to ALPHA_VANTAGE_API_KEY.")
    parser.add_argument("--output-dir", type=Path, default=Path("reports/latest"), help="Output directory for reports")
    parser.add_argument("--backend", default="auto", choices=["auto", "xgboost", "hist_gradient_boosting"])
    parser.add_argument("--threshold", type=float, default=0.55, help="Default probability threshold")
    parser.add_argument("--commission-bps", type=float, default=1.0, help="Commission in basis points")
    parser.add_argument("--slippage-bps", type=float, default=2.0, help="Slippage in basis points")
    return parser


def _config_from_args(args: argparse.Namespace) -> PipelineConfig:
    experiment_config = WalkForwardExperimentConfig(
        split_config=DEFAULT_SPLIT_CONFIG,
        model_config=XGBoostConfig(backend=args.backend),
        backtest_config=BacktestConfig(
            probability_threshold=args.threshold,
            commission_bps=args.commission_bps,
            slippage_bps=args.slippage_bps,
        ),
    )
    return PipelineConfig(
        symbol=args.symbol,
        output_dir=args.output_dir,
        input_csv=args.input_csv,
        fetch_alpha_vantage=args.fetch_alpha_vantage,
        alpha_vantage_key=args.alpha_vantage_key,
        experiment_config=experiment_config,
    )


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    config = _config_from_args(args)
    result = run_pipeline(config)
    print(json.dumps({key: str(value) if isinstance(value, Path) else value for key, value in result.summary.items()}, indent=2))
    print(f"reports_written={config.output_dir}")


if __name__ == "__main__":
    main()
