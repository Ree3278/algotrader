"""Runnable train/test pipelines for the SPY daily baseline."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from algotrader.backtest import BacktestConfig
from algotrader.ingestion import (
    fetch_daily_adjusted,
    fetch_yfinance_daily,
    load_ohlcv_csv,
    normalize_daily_adjusted,
    save_json,
    save_ohlcv_csv,
)
from algotrader.reporting import build_experiment_summary, write_experiment_reports
from algotrader.training.artifacts import (
    load_model_artifact,
    load_training_manifest,
    save_model_artifact,
    save_training_manifest,
)
from algotrader.training.dataset import TrainingDataset, build_training_dataset
from algotrader.training.experiment import (
    WalkForwardExperimentConfig,
    WalkForwardExperimentResult,
    _select_threshold,
    _slice_price_for_signal_window,
)
from algotrader.training.walk_forward import PurgedWalkForwardConfig, generate_splits
from algotrader.training.xgboost_model import XGBoostConfig, train_xgboost_classifier


DEFAULT_SPLIT_CONFIG = PurgedWalkForwardConfig(
    train_size=504,
    test_size=252,
    step_size=252,
    embargo_size=10,
    max_label_horizon=10,
)


@dataclass(frozen=True)
class PipelineConfig:
    symbol: str = "SPY"
    input_csv: Path | None = None
    fetch_yfinance: bool = False
    yfinance_period: str = "max"
    yfinance_start: str | None = None
    yfinance_end: str | None = None
    fetch_alpha_vantage: bool = False
    alpha_vantage_key: str | None = None
    alpha_vantage_outputsize: str = "full"
    raw_data_dir: Path = Path("data/raw/ohlcv")
    normalized_data_dir: Path = Path("data/interim")
    experiment_config: WalkForwardExperimentConfig = field(
        default_factory=lambda: WalkForwardExperimentConfig(split_config=DEFAULT_SPLIT_CONFIG)
    )


@dataclass(frozen=True)
class TrainPipelineConfig(PipelineConfig):
    model_dir: Path = Path("models/latest")


@dataclass(frozen=True)
class TestPipelineConfig(PipelineConfig):
    model_dir: Path = Path("models/latest")
    output_dir: Path = Path("reports/latest")


TrainPipelineConfig.__test__ = False
TestPipelineConfig.__test__ = False


@dataclass(frozen=True)
class TrainingRunResult:
    dataset: pd.DataFrame
    price_frame: pd.DataFrame
    manifest: dict[str, Any]
    fold_manifest: pd.DataFrame
    artifact_paths: dict[str, Path]


@dataclass(frozen=True)
class TestRunResult:
    dataset: pd.DataFrame
    price_frame: pd.DataFrame
    fold_summaries: pd.DataFrame
    test_predictions: pd.DataFrame
    summary: dict[str, Any]
    report_paths: dict[str, Path]
    manifest: dict[str, Any]


def _iso_or_none(value: pd.Timestamp | None) -> str | None:
    return value.isoformat() if value is not None and not pd.isna(value) else None


def _load_or_fetch_price_frame(config: PipelineConfig) -> pd.DataFrame:
    if config.input_csv is not None:
        return load_ohlcv_csv(config.input_csv)

    if config.fetch_yfinance:
        price_frame = fetch_yfinance_daily(
            config.symbol,
            period=config.yfinance_period,
            start=config.yfinance_start,
            end=config.yfinance_end,
        )
        normalized_path = config.normalized_data_dir / f"{config.symbol.lower()}_daily.csv"
        save_ohlcv_csv(price_frame, normalized_path)
        return price_frame

    if config.fetch_alpha_vantage:
        api_key = config.alpha_vantage_key or os.getenv("ALPHA_VANTAGE_API_KEY")
        if not api_key:
            raise ValueError(
                "Alpha Vantage API key not found. Set ALPHA_VANTAGE_API_KEY or pass --alpha-vantage-key."
            )

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

    raise ValueError("Provide --input-csv or enable --fetch-yfinance / --fetch-alpha-vantage")


def _build_dataset(price_frame: pd.DataFrame) -> TrainingDataset:
    dataset = build_training_dataset(price_frame)
    if dataset.data.empty:
        raise ValueError("Training dataset is empty after feature warmup and label construction")
    return dataset


def run_training_pipeline(config: TrainPipelineConfig) -> TrainingRunResult:
    """Train fold models and persist artifact manifests."""

    price_frame = _load_or_fetch_price_frame(config)
    dataset = _build_dataset(price_frame)

    config.model_dir.mkdir(parents=True, exist_ok=True)
    fold_records: list[dict[str, Any]] = []

    for split in generate_splits(dataset.data.index, config.experiment_config.split_config):
        train_data = dataset.data.iloc[split.train_indices]
        test_data = dataset.data.iloc[split.test_indices]

        if len(train_data) < config.experiment_config.min_training_size or test_data.empty:
            continue

        calibration_size = max(
            int(len(train_data) * config.experiment_config.calibration_fraction),
            config.experiment_config.min_calibration_size,
        )
        calibration_data = train_data.iloc[0:0]
        selected_threshold = config.experiment_config.backtest_config.probability_threshold

        if len(train_data) >= config.experiment_config.min_training_size + calibration_size:
            train_core = train_data.iloc[:-calibration_size]
            calibration_data = train_data.iloc[-calibration_size:]

            calibration_model = train_xgboost_classifier(
                train_core[dataset.feature_columns],
                train_core[dataset.target_column],
                config=config.experiment_config.model_config,
            )
            calibration_probabilities = pd.Series(
                calibration_model.predict_proba(calibration_data[dataset.feature_columns])[:, 1],
                index=calibration_data.index,
            )
            selected_threshold = _select_threshold(
                price_frame,
                calibration_data.index,
                calibration_probabilities,
                config.experiment_config.backtest_config,
                config.experiment_config.threshold_grid,
            )

        final_model = train_xgboost_classifier(
            train_data[dataset.feature_columns],
            train_data[dataset.target_column],
            config=config.experiment_config.model_config,
        )
        model_backend = getattr(final_model, "_algotrader_backend", "unknown")
        model_filename = f"fold_{split.fold:03d}.pkl"
        save_model_artifact(final_model, config.model_dir / model_filename)

        fold_records.append(
            {
                "fold": split.fold,
                "model_file": model_filename,
                "model_backend": model_backend,
                "selected_threshold": float(selected_threshold),
                "train_size": int(len(train_data)),
                "calibration_size": int(len(calibration_data)),
                "test_size": int(len(test_data)),
                "train_start": _iso_or_none(train_data.index.min()),
                "train_end": _iso_or_none(train_data.index.max()),
                "calibration_start": _iso_or_none(calibration_data.index.min() if not calibration_data.empty else None),
                "calibration_end": _iso_or_none(calibration_data.index.max() if not calibration_data.empty else None),
                "test_start": _iso_or_none(test_data.index.min()),
                "test_end": _iso_or_none(test_data.index.max()),
            }
        )

    fold_manifest = pd.DataFrame(fold_records)
    if fold_manifest.empty:
        raise ValueError("No trainable folds were produced by the current split configuration")

    manifest = {
        "symbol": config.symbol,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset_rows": int(len(dataset.data)),
        "feature_columns": dataset.feature_columns,
        "target_column": dataset.target_column,
        "input_csv": str(config.input_csv) if config.input_csv is not None else None,
        "experiment_config": json.loads(json.dumps(asdict(config.experiment_config), default=str)),
    }
    artifact_paths = save_training_manifest(
        config.model_dir,
        manifest=manifest,
        fold_manifest=fold_manifest,
    )

    return TrainingRunResult(
        dataset=dataset.data,
        price_frame=price_frame,
        manifest=manifest,
        fold_manifest=fold_manifest,
        artifact_paths=artifact_paths,
    )


def run_test_pipeline(config: TestPipelineConfig) -> TestRunResult:
    """Load trained fold models, score test windows, and write reports."""

    price_frame = _load_or_fetch_price_frame(config)
    dataset = _build_dataset(price_frame)
    manifest, fold_manifest = load_training_manifest(config.model_dir)

    prediction_frames: list[pd.DataFrame] = []
    fold_summaries: list[dict[str, Any]] = []

    for fold_row in fold_manifest.itertuples(index=False):
        test_data = dataset.data.loc[fold_row.test_start : fold_row.test_end]
        if test_data.empty:
            continue

        model = load_model_artifact(config.model_dir / fold_row.model_file)
        test_probabilities = pd.Series(
            model.predict_proba(test_data[manifest["feature_columns"]])[:, 1],
            index=test_data.index,
        )

        backtest_config = BacktestConfig(
            probability_threshold=float(fold_row.selected_threshold),
            commission_bps=config.experiment_config.backtest_config.commission_bps,
            slippage_bps=config.experiment_config.backtest_config.slippage_bps,
        )
        test_price_slice = _slice_price_for_signal_window(price_frame, test_data.index)
        from algotrader.backtest import run_long_flat_backtest, summarize_backtest

        backtest_results = run_long_flat_backtest(test_price_slice, test_probabilities, config=backtest_config)
        metrics = summarize_backtest(backtest_results)

        prediction_frames.append(
            pd.DataFrame(
                {
                    "fold": fold_row.fold,
                    "probability_long": test_probabilities,
                    "label": test_data[manifest["target_column"]],
                    "selected_threshold": float(fold_row.selected_threshold),
                }
            )
        )

        fold_summaries.append(
            {
                "fold": int(fold_row.fold),
                "train_size": int(fold_row.train_size),
                "calibration_size": int(fold_row.calibration_size),
                "test_size": int(fold_row.test_size),
                "model_backend": str(fold_row.model_backend),
                "selected_threshold": float(fold_row.selected_threshold),
                **metrics,
            }
        )

    result = WalkForwardExperimentResult(
        fold_summaries=pd.DataFrame(fold_summaries),
        test_predictions=pd.concat(prediction_frames).sort_index() if prediction_frames else pd.DataFrame(),
    )
    model_backend = None
    if not result.fold_summaries.empty and "model_backend" in result.fold_summaries.columns:
        model_backend = str(result.fold_summaries["model_backend"].mode(dropna=True).iloc[0])
    summary = build_experiment_summary(
        result,
        symbol=manifest["symbol"],
        dataset_rows=len(dataset.data),
        feature_count=len(manifest["feature_columns"]),
        model_backend=model_backend,
    )
    report_paths = write_experiment_reports(result, config.output_dir, summary=summary)

    return TestRunResult(
        dataset=dataset.data,
        price_frame=price_frame,
        fold_summaries=result.fold_summaries,
        test_predictions=result.test_predictions,
        summary=summary,
        report_paths=report_paths,
        manifest=manifest,
    )


def run_pipeline(config: TestPipelineConfig) -> TestRunResult:
    """Convenience wrapper that trains then immediately tests."""

    train_config = TrainPipelineConfig(
        symbol=config.symbol,
        input_csv=config.input_csv,
        fetch_yfinance=config.fetch_yfinance,
        yfinance_period=config.yfinance_period,
        yfinance_start=config.yfinance_start,
        yfinance_end=config.yfinance_end,
        fetch_alpha_vantage=config.fetch_alpha_vantage,
        alpha_vantage_key=config.alpha_vantage_key,
        alpha_vantage_outputsize=config.alpha_vantage_outputsize,
        raw_data_dir=config.raw_data_dir,
        normalized_data_dir=config.normalized_data_dir,
        experiment_config=config.experiment_config,
        model_dir=config.model_dir,
    )
    run_training_pipeline(train_config)
    return run_test_pipeline(config)


def run_fetch_yfinance(symbol: str, output_csv: Path, *, period: str = "max", start: str | None = None, end: str | None = None) -> Path:
    """Download daily OHLCV from yfinance and save it as normalized CSV."""

    price_frame = fetch_yfinance_daily(symbol, period=period, start=start, end=end)
    save_ohlcv_csv(price_frame, output_csv)
    return output_csv


def _add_shared_data_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--symbol", default="SPY", help="Ticker symbol to analyze")
    parser.add_argument("--input-csv", type=Path, help="Path to normalized OHLCV CSV")
    parser.add_argument("--fetch-yfinance", action="store_true", help="Fetch data from yfinance before running")
    parser.add_argument("--yf-period", default="max", help="yfinance period, e.g. max, 10y, 5y")
    parser.add_argument("--yf-start", help="Optional yfinance start date, YYYY-MM-DD")
    parser.add_argument("--yf-end", help="Optional yfinance end date, YYYY-MM-DD")
    parser.add_argument("--fetch-alpha-vantage", action="store_true", help="Fetch daily adjusted OHLCV from Alpha Vantage")
    parser.add_argument("--alpha-vantage-key", help="Alpha Vantage API key. Defaults to ALPHA_VANTAGE_API_KEY.")


def _add_shared_model_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--backend", default="auto", choices=["auto", "xgboost", "hist_gradient_boosting"])
    parser.add_argument("--threshold", type=float, default=0.55, help="Default probability threshold")
    parser.add_argument("--commission-bps", type=float, default=1.0, help="Commission in basis points")
    parser.add_argument("--slippage-bps", type=float, default=2.0, help="Slippage in basis points")


def _experiment_config_from_args(args: argparse.Namespace) -> WalkForwardExperimentConfig:
    return WalkForwardExperimentConfig(
        split_config=DEFAULT_SPLIT_CONFIG,
        model_config=XGBoostConfig(backend=args.backend),
        backtest_config=BacktestConfig(
            probability_threshold=args.threshold,
            commission_bps=args.commission_bps,
            slippage_bps=args.slippage_bps,
        ),
    )


def build_train_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train walk-forward fold models.")
    _add_shared_data_args(parser)
    _add_shared_model_args(parser)
    parser.add_argument("--model-dir", type=Path, default=Path("models/latest"), help="Directory for saved fold models")
    return parser


def build_test_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate saved fold models on test windows.")
    _add_shared_data_args(parser)
    _add_shared_model_args(parser)
    parser.add_argument("--model-dir", type=Path, default=Path("models/latest"), help="Directory containing saved fold models")
    parser.add_argument("--output-dir", type=Path, default=Path("reports/latest"), help="Directory for generated reports")
    return parser


def build_fetch_yfinance_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fetch and normalize daily OHLCV data from yfinance.")
    parser.add_argument("--symbol", default="SPY", help="Ticker symbol to download")
    parser.add_argument("--output-csv", type=Path, default=Path("data/interim/spy_daily.csv"), help="Output CSV path")
    parser.add_argument("--period", default="max", help="yfinance period, e.g. max, 10y, 5y")
    parser.add_argument("--start", help="Optional yfinance start date, YYYY-MM-DD")
    parser.add_argument("--end", help="Optional yfinance end date, YYYY-MM-DD")
    return parser


def _train_config_from_args(args: argparse.Namespace) -> TrainPipelineConfig:
    return TrainPipelineConfig(
        symbol=args.symbol,
        input_csv=args.input_csv,
        fetch_yfinance=args.fetch_yfinance,
        yfinance_period=args.yf_period,
        yfinance_start=args.yf_start,
        yfinance_end=args.yf_end,
        fetch_alpha_vantage=args.fetch_alpha_vantage,
        alpha_vantage_key=args.alpha_vantage_key,
        experiment_config=_experiment_config_from_args(args),
        model_dir=args.model_dir,
    )


def _test_config_from_args(args: argparse.Namespace) -> TestPipelineConfig:
    return TestPipelineConfig(
        symbol=args.symbol,
        input_csv=args.input_csv,
        fetch_yfinance=args.fetch_yfinance,
        yfinance_period=args.yf_period,
        yfinance_start=args.yf_start,
        yfinance_end=args.yf_end,
        fetch_alpha_vantage=args.fetch_alpha_vantage,
        alpha_vantage_key=args.alpha_vantage_key,
        experiment_config=_experiment_config_from_args(args),
        model_dir=args.model_dir,
        output_dir=args.output_dir,
    )


def train_main() -> None:
    parser = build_train_arg_parser()
    args = parser.parse_args()
    result = run_training_pipeline(_train_config_from_args(args))
    print(json.dumps({"model_dir": str(args.model_dir), "fold_count": len(result.fold_manifest), "dataset_rows": len(result.dataset)}, indent=2))


def test_main() -> None:
    parser = build_test_arg_parser()
    args = parser.parse_args()
    result = run_test_pipeline(_test_config_from_args(args))
    print(json.dumps({key: str(value) if isinstance(value, Path) else value for key, value in result.summary.items()}, indent=2))
    print(f"reports_written={args.output_dir}")


def fetch_yfinance_main() -> None:
    parser = build_fetch_yfinance_arg_parser()
    args = parser.parse_args()
    output = run_fetch_yfinance(args.symbol, args.output_csv, period=args.period, start=args.start, end=args.end)
    print(f"saved_csv={output}")


def main() -> None:
    parser = build_test_arg_parser()
    args = parser.parse_args()
    result = run_pipeline(_test_config_from_args(args))
    print(json.dumps({key: str(value) if isinstance(value, Path) else value for key, value in result.summary.items()}, indent=2))
    print(f"reports_written={args.output_dir}")


if __name__ == "__main__":
    main()
