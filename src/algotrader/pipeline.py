"""Runnable train/test pipelines for the SPY daily baseline."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from algotrader.backtest import BacktestConfig
from algotrader.ingestion import (
    fetch_daily_adjusted,
    fetch_yfinance_daily,
    load_ohlcv_csv,
    load_timeseries_csv,
    normalize_daily_adjusted,
    save_json,
    save_ohlcv_csv,
)
from algotrader.labels import TripleBarrierConfig
from algotrader.profiles import build_model_profile, list_profile_names
from algotrader.reporting import build_experiment_summary, format_test_terminal_summary, write_experiment_reports
from algotrader.settings import DEFAULT_SETTINGS, ProjectSettings
from algotrader.thresholds import list_threshold_policy_names
from algotrader.training.calibration import (
    ProbabilityCalibrator,
    apply_probability_calibration,
    fit_probability_calibrator,
)
from algotrader.training.artifacts import (
    load_model_artifact,
    load_training_manifest,
    save_model_artifact,
    save_training_manifest,
)
from algotrader.training.dataset import (
    REGIME_FEATURE_COLUMNS,
    SENTIMENT_FEATURE_COLUMNS,
    TrainingDataset,
    build_training_dataset,
)
from algotrader.training.experiment import (
    WalkForwardExperimentConfig,
    WalkForwardExperimentResult,
    build_threshold_application,
    default_threshold_selection,
    select_thresholds,
)
from algotrader.training.walk_forward import generate_splits
from algotrader.training.xgboost_model import train_xgboost_classifier


@dataclass(frozen=True)
class PipelineConfig:
    symbol: str = DEFAULT_SETTINGS.data.symbol
    input_csv: Path | None = DEFAULT_SETTINGS.paths.default_price_csv(DEFAULT_SETTINGS.data.symbol)
    vix_input_csv: Path | None = None
    sentiment_features_csv: Path | None = None
    feature_columns: list[str] | None = None
    profile_name: str = DEFAULT_SETTINGS.profiles.default_profile_name
    threshold_policy_name: str = DEFAULT_SETTINGS.thresholds.default_policy_name
    probability_calibration_method: str = DEFAULT_SETTINGS.experiment.probability_calibration_method
    max_calibration_exposure: float | None = DEFAULT_SETTINGS.experiment.max_calibration_exposure
    auto_discover_companion_inputs: bool = True
    fetch_yfinance: bool = False
    yfinance_period: str = "max"
    yfinance_start: str | None = None
    yfinance_end: str | None = None
    fetch_alpha_vantage: bool = False
    alpha_vantage_key: str | None = None
    alpha_vantage_outputsize: str = "full"
    raw_data_dir: Path = DEFAULT_SETTINGS.paths.raw_data_dir
    normalized_data_dir: Path = DEFAULT_SETTINGS.paths.normalized_data_dir
    settings: ProjectSettings = field(default_factory=lambda: DEFAULT_SETTINGS)
    experiment_config: WalkForwardExperimentConfig | None = None


@dataclass(frozen=True)
class TrainPipelineConfig(PipelineConfig):
    model_dir: Path = DEFAULT_SETTINGS.paths.model_dir


@dataclass(frozen=True)
class TestPipelineConfig(PipelineConfig):
    model_dir: Path = DEFAULT_SETTINGS.paths.model_dir
    output_dir: Path = DEFAULT_SETTINGS.paths.report_dir


TrainPipelineConfig.__test__ = False
TestPipelineConfig.__test__ = False


@dataclass(frozen=True)
class TrainingRunResult:
    dataset: pd.DataFrame
    price_frame: pd.DataFrame
    vix_frame: pd.DataFrame | None
    sentiment_frame: pd.DataFrame | None
    manifest: dict[str, Any]
    fold_manifest: pd.DataFrame
    artifact_paths: dict[str, Path]


@dataclass(frozen=True)
class TestRunResult:
    dataset: pd.DataFrame
    price_frame: pd.DataFrame
    vix_frame: pd.DataFrame | None
    sentiment_frame: pd.DataFrame | None
    fold_summaries: pd.DataFrame
    test_predictions: pd.DataFrame
    summary: dict[str, Any]
    report_paths: dict[str, Path]
    manifest: dict[str, Any]


def _iso_or_none(value: pd.Timestamp | None) -> str | None:
    return value.isoformat() if value is not None and not pd.isna(value) else None


def _default_vix_filename() -> str:
    return "vix_daily.csv"


def _default_sentiment_filename() -> str:
    return "sentiment_daily.csv"


def _default_input_csv(symbol: str) -> Path:
    return DEFAULT_SETTINGS.paths.default_price_csv(symbol)


def _default_vix_output_csv() -> Path:
    return DEFAULT_SETTINGS.paths.default_vix_csv


def _resolved_experiment_config(config: PipelineConfig) -> WalkForwardExperimentConfig:
    base_experiment_config = config.experiment_config or config.settings.build_experiment_config()
    return replace(
        base_experiment_config,
        threshold_policy_name=config.threshold_policy_name,
        probability_calibration_method=config.probability_calibration_method,
        max_calibration_exposure=config.max_calibration_exposure,
    )


def _resolved_label_config(config: PipelineConfig):
    return config.settings.build_label_config()


def _resolved_profile(config: PipelineConfig):
    return build_model_profile(name=config.profile_name)


def _resolve_vix_input_csv(config: PipelineConfig) -> Path | None:
    if config.vix_input_csv is not None:
        return config.vix_input_csv
    if not config.auto_discover_companion_inputs:
        return None
    if config.input_csv is not None:
        sibling_vix = config.input_csv.parent / _default_vix_filename()
        if sibling_vix.exists():
            return sibling_vix
    return None


def _resolve_sentiment_features_csv(config: PipelineConfig) -> Path | None:
    if config.sentiment_features_csv is not None:
        return config.sentiment_features_csv
    if not config.auto_discover_companion_inputs:
        return None
    if config.input_csv is not None:
        sibling_sentiment = config.input_csv.parent / _default_sentiment_filename()
        if sibling_sentiment.exists():
            return sibling_sentiment
    return None


def _load_or_fetch_frames(config: PipelineConfig) -> tuple[pd.DataFrame, pd.DataFrame | None, pd.DataFrame | None]:
    if config.input_csv is not None:
        price_frame = load_ohlcv_csv(config.input_csv)
        resolved_vix_csv = _resolve_vix_input_csv(config)
        vix_frame = load_ohlcv_csv(resolved_vix_csv) if resolved_vix_csv is not None else None
        resolved_sentiment_csv = _resolve_sentiment_features_csv(config)
        sentiment_frame = load_timeseries_csv(resolved_sentiment_csv) if resolved_sentiment_csv is not None else None
        return price_frame, vix_frame, sentiment_frame

    if config.fetch_yfinance:
        price_frame = fetch_yfinance_daily(
            config.symbol,
            period=config.yfinance_period,
            start=config.yfinance_start,
            end=config.yfinance_end,
        )
        vix_frame = fetch_yfinance_daily(
            config.settings.data.vix_symbol,
            period=config.yfinance_period,
            start=config.yfinance_start,
            end=config.yfinance_end,
        )
        normalized_path = config.normalized_data_dir / f"{config.symbol.lower()}_daily.csv"
        vix_path = config.normalized_data_dir / _default_vix_filename()
        save_ohlcv_csv(price_frame, normalized_path)
        save_ohlcv_csv(vix_frame, vix_path)
        resolved_sentiment_csv = _resolve_sentiment_features_csv(config)
        sentiment_frame = load_timeseries_csv(resolved_sentiment_csv) if resolved_sentiment_csv is not None else None
        return price_frame, vix_frame, sentiment_frame

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
        resolved_vix_csv = _resolve_vix_input_csv(config)
        vix_frame = load_ohlcv_csv(resolved_vix_csv) if resolved_vix_csv is not None else None
        resolved_sentiment_csv = _resolve_sentiment_features_csv(config)
        sentiment_frame = load_timeseries_csv(resolved_sentiment_csv) if resolved_sentiment_csv is not None else None
        return price_frame, vix_frame, sentiment_frame

    raise ValueError("Provide --input-csv or enable --fetch-yfinance / --fetch-alpha-vantage")


def _build_dataset(
    config: PipelineConfig,
    price_frame: pd.DataFrame,
    vix_frame: pd.DataFrame | None = None,
    sentiment_frame: pd.DataFrame | None = None,
    label_config: TripleBarrierConfig | None = None,
) -> TrainingDataset:
    profile = _resolved_profile(config)
    selected_feature_columns = config.feature_columns or profile.feature_columns
    requires_vix = any(column in REGIME_FEATURE_COLUMNS for column in selected_feature_columns)
    requires_sentiment = any(column in SENTIMENT_FEATURE_COLUMNS for column in selected_feature_columns)
    if requires_vix and vix_frame is None:
        raise ValueError(f"Profile '{profile.name}' requires VIX input, but no VIX data was available")
    if requires_sentiment and sentiment_frame is None:
        raise ValueError(f"Profile '{profile.name}' requires sentiment input, but no sentiment data was available")
    dataset = build_training_dataset(
        price_frame,
        vix_frame=vix_frame,
        sentiment_frame=sentiment_frame,
        label_config=label_config or _resolved_label_config(config),
        feature_columns=selected_feature_columns,
    )
    if dataset.data.empty:
        raise ValueError("Training dataset is empty after feature warmup and label construction")
    return dataset


def run_training_pipeline(config: TrainPipelineConfig) -> TrainingRunResult:
    """Train fold models and persist artifact manifests."""

    price_frame, vix_frame, sentiment_frame = _load_or_fetch_frames(config)
    dataset = _build_dataset(config, price_frame, vix_frame=vix_frame, sentiment_frame=sentiment_frame)
    experiment_config = _resolved_experiment_config(config)

    config.model_dir.mkdir(parents=True, exist_ok=True)
    fold_records: list[dict[str, Any]] = []

    for split in generate_splits(dataset.data.index, experiment_config.split_config):
        train_data = dataset.data.iloc[split.train_indices]
        test_data = dataset.data.iloc[split.test_indices]

        if len(train_data) < experiment_config.min_training_size or test_data.empty:
            continue

        calibration_size = max(
            int(len(train_data) * experiment_config.calibration_fraction),
            experiment_config.min_calibration_size,
        )
        calibration_data = train_data.iloc[0:0]
        threshold_selection = default_threshold_selection(
            threshold_policy_name=experiment_config.threshold_policy_name,
            default_threshold=experiment_config.backtest_config.probability_threshold,
        )

        if len(train_data) >= experiment_config.min_training_size + calibration_size:
            train_core = train_data.iloc[:-calibration_size]
            calibration_data = train_data.iloc[-calibration_size:]

            calibration_model = train_xgboost_classifier(
                train_core[dataset.feature_columns],
                train_core[dataset.target_column],
                config=experiment_config.model_config,
            )
            calibration_probabilities = pd.Series(
                calibration_model.predict_proba(calibration_data[dataset.feature_columns])[:, 1],
                index=calibration_data.index,
            )
            calibrator = fit_probability_calibrator(
                calibration_probabilities,
                calibration_data[dataset.target_column],
                method=experiment_config.probability_calibration_method,
            )
            calibration_probabilities = apply_probability_calibration(calibrator, calibration_probabilities)
            threshold_selection = select_thresholds(
                price_frame,
                calibration_data,
                calibration_probabilities,
                experiment_config.backtest_config,
                experiment_config.threshold_grid,
                experiment_config.threshold_policy_name,
                experiment_config.max_calibration_exposure,
            )
            if calibrator.method == "none":
                final_model = train_xgboost_classifier(
                    train_data[dataset.feature_columns],
                    train_data[dataset.target_column],
                    config=experiment_config.model_config,
                )
            else:
                final_model = calibration_model
        else:
            calibrator = ProbabilityCalibrator(method="none")
            final_model = train_xgboost_classifier(
                train_data[dataset.feature_columns],
                train_data[dataset.target_column],
                config=experiment_config.model_config,
            )
        model_backend = getattr(final_model, "_algotrader_backend", "unknown")
        model_filename = f"fold_{split.fold:03d}.pkl"
        calibrator_filename = f"fold_{split.fold:03d}_calibrator.pkl"
        save_model_artifact(final_model, config.model_dir / model_filename)
        save_model_artifact(calibrator, config.model_dir / calibrator_filename)

        fold_records.append(
            {
                "fold": split.fold,
                "model_file": model_filename,
                "calibrator_file": calibrator_filename,
                "model_backend": model_backend,
                "threshold_policy_name": threshold_selection.policy_name,
                "probability_calibration_method": calibrator.method,
                "selected_threshold": float(threshold_selection.representative_threshold),
                "selected_threshold_map": json.dumps(threshold_selection.thresholds_by_regime, sort_keys=True),
                "threshold_selection_mode": threshold_selection.selection_mode,
                "calibration_exposure": threshold_selection.calibration_exposure,
                "feasible_threshold_count": int(threshold_selection.feasible_candidate_count),
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
        "profile_name": config.profile_name,
        "profile_blocks": _resolved_profile(config).block_names,
        "threshold_policy_name": experiment_config.threshold_policy_name,
        "probability_calibration_method": experiment_config.probability_calibration_method,
        "max_calibration_exposure": experiment_config.max_calibration_exposure,
        "target_column": dataset.target_column,
        "input_csv": str(config.input_csv) if config.input_csv is not None else None,
        "vix_input_csv": str(config.vix_input_csv) if config.vix_input_csv is not None else None,
        "sentiment_features_csv": str(config.sentiment_features_csv) if config.sentiment_features_csv is not None else None,
        "label_config": json.loads(json.dumps(asdict(_resolved_label_config(config)), default=str)),
        "experiment_config": json.loads(json.dumps(asdict(experiment_config), default=str)),
    }
    artifact_paths = save_training_manifest(
        config.model_dir,
        manifest=manifest,
        fold_manifest=fold_manifest,
    )

    return TrainingRunResult(
        dataset=dataset.data,
        price_frame=price_frame,
        vix_frame=vix_frame,
        sentiment_frame=sentiment_frame,
        manifest=manifest,
        fold_manifest=fold_manifest,
        artifact_paths=artifact_paths,
    )


def run_test_pipeline(config: TestPipelineConfig) -> TestRunResult:
    """Load trained fold models, score test windows, and write reports."""

    price_frame, vix_frame, sentiment_frame = _load_or_fetch_frames(config)
    manifest, fold_manifest = load_training_manifest(config.model_dir)
    dataset_config = replace(
        config,
        profile_name=manifest.get("profile_name", config.profile_name),
        feature_columns=manifest.get("feature_columns"),
        threshold_policy_name=manifest.get("threshold_policy_name", config.threshold_policy_name),
        probability_calibration_method=manifest.get(
            "probability_calibration_method",
            config.probability_calibration_method,
        ),
        max_calibration_exposure=manifest.get(
            "max_calibration_exposure",
            config.max_calibration_exposure,
        ),
    )
    label_config = None
    if manifest.get("label_config") is not None:
        label_config = TripleBarrierConfig(**manifest["label_config"])
    dataset = _build_dataset(
        dataset_config,
        price_frame,
        vix_frame=vix_frame,
        sentiment_frame=sentiment_frame,
        label_config=label_config,
    )
    missing_features = [feature for feature in manifest["feature_columns"] if feature not in dataset.data.columns]
    if missing_features:
        raise ValueError(
            "Dataset is missing required features from the trained manifest: "
            f"{missing_features}. Supply the matching VIX input if the model was trained with VIX features."
        )

    prediction_frames: list[pd.DataFrame] = []
    fold_summaries: list[dict[str, Any]] = []
    experiment_config = _resolved_experiment_config(config)

    for fold_row in fold_manifest.itertuples(index=False):
        test_data = dataset.data.loc[fold_row.test_start : fold_row.test_end]
        if test_data.empty:
            continue

        model = load_model_artifact(config.model_dir / fold_row.model_file)
        test_probabilities = pd.Series(
            model.predict_proba(test_data[manifest["feature_columns"]])[:, 1],
            index=test_data.index,
        )
        calibrator = ProbabilityCalibrator(method="none")
        calibrator_file = getattr(fold_row, "calibrator_file", None)
        if isinstance(calibrator_file, str) and calibrator_file:
            calibrator = load_model_artifact(config.model_dir / calibrator_file)
        test_probabilities = apply_probability_calibration(calibrator, test_probabilities)

        threshold_policy_name = getattr(
            fold_row,
            "threshold_policy_name",
            manifest.get("threshold_policy_name", experiment_config.threshold_policy_name),
        )
        threshold_map = {"all": float(fold_row.selected_threshold)}
        if hasattr(fold_row, "selected_threshold_map") and isinstance(fold_row.selected_threshold_map, str):
            threshold_map = {key: float(value) for key, value in json.loads(fold_row.selected_threshold_map).items()}
        threshold_series, threshold_regimes = build_threshold_application(
            test_data,
            threshold_policy_name=threshold_policy_name,
            thresholds_by_regime=threshold_map,
        )
        backtest_config = BacktestConfig(
            probability_threshold=float(fold_row.selected_threshold),
            commission_bps=experiment_config.backtest_config.commission_bps,
            slippage_bps=experiment_config.backtest_config.slippage_bps,
        )
        from algotrader.backtest import run_long_flat_backtest, summarize_backtest

        backtest_results = run_long_flat_backtest(
            price_frame,
            test_data,
            test_probabilities,
            config=backtest_config,
            threshold_series=threshold_series,
        )
        metrics = summarize_backtest(backtest_results)

        prediction_frames.append(
            pd.DataFrame(
                {
                    "fold": fold_row.fold,
                    "probability_long": test_probabilities,
                    "label": test_data[manifest["target_column"]],
                    "probability_calibration_method": calibrator.method,
                    "selected_threshold": threshold_series,
                    "threshold_regime": threshold_regimes,
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
                "threshold_policy_name": threshold_policy_name,
                "probability_calibration_method": calibrator.method,
                "selected_threshold": float(fold_row.selected_threshold),
                "threshold_selection_mode": getattr(fold_row, "threshold_selection_mode", "unknown"),
                "calibration_exposure": getattr(fold_row, "calibration_exposure", None),
                "feasible_threshold_count": getattr(fold_row, "feasible_threshold_count", None),
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
        vix_frame=vix_frame,
        sentiment_frame=sentiment_frame,
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
        vix_input_csv=config.vix_input_csv,
        sentiment_features_csv=config.sentiment_features_csv,
        feature_columns=config.feature_columns,
        profile_name=config.profile_name,
        threshold_policy_name=config.threshold_policy_name,
        probability_calibration_method=config.probability_calibration_method,
        max_calibration_exposure=config.max_calibration_exposure,
        auto_discover_companion_inputs=config.auto_discover_companion_inputs,
        fetch_yfinance=config.fetch_yfinance,
        yfinance_period=config.yfinance_period,
        yfinance_start=config.yfinance_start,
        yfinance_end=config.yfinance_end,
        fetch_alpha_vantage=config.fetch_alpha_vantage,
        alpha_vantage_key=config.alpha_vantage_key,
        alpha_vantage_outputsize=config.alpha_vantage_outputsize,
        raw_data_dir=config.raw_data_dir,
        normalized_data_dir=config.normalized_data_dir,
        settings=config.settings,
        experiment_config=config.experiment_config,
        model_dir=config.model_dir,
    )
    run_training_pipeline(train_config)
    return run_test_pipeline(config)


def run_fetch_yfinance(
    symbol: str,
    output_csv: Path,
    *,
    vix_output_csv: Path | None = None,
    period: str = "max",
    start: str | None = None,
    end: str | None = None,
) -> tuple[Path, Path | None]:
    """Download daily OHLCV from yfinance and save it as normalized CSV."""

    price_frame = fetch_yfinance_daily(symbol, period=period, start=start, end=end)
    save_ohlcv_csv(price_frame, output_csv)
    saved_vix_path = None
    if vix_output_csv is not None:
        vix_frame = fetch_yfinance_daily(DEFAULT_SETTINGS.data.vix_symbol, period=period, start=start, end=end)
        save_ohlcv_csv(vix_frame, vix_output_csv)
        saved_vix_path = vix_output_csv
    return output_csv, saved_vix_path


def _add_shared_data_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--symbol", default=DEFAULT_SETTINGS.data.symbol, help="Ticker symbol to analyze")
    parser.add_argument("--input-csv", type=Path, help="Path to normalized OHLCV CSV")
    parser.add_argument("--vix-csv", type=Path, help="Optional path to normalized VIX CSV")
    parser.add_argument("--sentiment-features-csv", type=Path, help="Optional path to daily sentiment features CSV")
    parser.add_argument("--fetch-yfinance", action="store_true", help="Fetch data from yfinance before running")
    parser.add_argument("--yf-period", default="max", help="yfinance period, e.g. max, 10y, 5y")
    parser.add_argument("--yf-start", help="Optional yfinance start date, YYYY-MM-DD")
    parser.add_argument("--yf-end", help="Optional yfinance end date, YYYY-MM-DD")
    parser.add_argument("--fetch-alpha-vantage", action="store_true", help="Fetch daily adjusted OHLCV from Alpha Vantage")
    parser.add_argument("--alpha-vantage-key", help="Alpha Vantage API key. Defaults to ALPHA_VANTAGE_API_KEY.")


def _add_shared_model_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--profile", default=DEFAULT_SETTINGS.profiles.default_profile_name, choices=list_profile_names())
    parser.add_argument(
        "--threshold-policy",
        default=DEFAULT_SETTINGS.thresholds.default_policy_name,
        choices=list_threshold_policy_names(),
    )
    parser.add_argument(
        "--probability-calibration",
        default=DEFAULT_SETTINGS.experiment.probability_calibration_method,
        choices=["none", "platt"],
    )
    parser.add_argument(
        "--max-calibration-exposure",
        type=float,
        default=DEFAULT_SETTINGS.experiment.max_calibration_exposure,
        help="Optional cap on calibration-period exposure during threshold selection",
    )
    parser.add_argument("--backend", default=DEFAULT_SETTINGS.model.backend, choices=["auto", "xgboost", "hist_gradient_boosting"])
    parser.add_argument("--threshold", type=float, default=DEFAULT_SETTINGS.backtest.probability_threshold, help="Default probability threshold")
    parser.add_argument("--commission-bps", type=float, default=DEFAULT_SETTINGS.backtest.commission_bps, help="Commission in basis points")
    parser.add_argument("--slippage-bps", type=float, default=DEFAULT_SETTINGS.backtest.slippage_bps, help="Slippage in basis points")


def _settings_from_args(args: argparse.Namespace) -> ProjectSettings:
    data_settings = replace(DEFAULT_SETTINGS.data, symbol=args.symbol)
    model_settings = replace(DEFAULT_SETTINGS.model, backend=args.backend)
    backtest_settings = replace(
        DEFAULT_SETTINGS.backtest,
        probability_threshold=args.threshold,
        commission_bps=args.commission_bps,
        slippage_bps=args.slippage_bps,
    )
    threshold_settings = replace(DEFAULT_SETTINGS.thresholds, default_policy_name=args.threshold_policy)
    experiment_settings = replace(
        DEFAULT_SETTINGS.experiment,
        probability_calibration_method=args.probability_calibration,
        max_calibration_exposure=args.max_calibration_exposure,
    )
    return replace(
        DEFAULT_SETTINGS,
        data=data_settings,
        model=model_settings,
        backtest=backtest_settings,
        experiment=experiment_settings,
        thresholds=threshold_settings,
    )


def build_train_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train walk-forward fold models.")
    _add_shared_data_args(parser)
    _add_shared_model_args(parser)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_SETTINGS.paths.model_dir, help="Directory for saved fold models")
    return parser


def build_test_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate saved fold models on test windows.")
    _add_shared_data_args(parser)
    _add_shared_model_args(parser)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_SETTINGS.paths.model_dir, help="Directory containing saved fold models")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_SETTINGS.paths.report_dir, help="Directory for generated reports")
    return parser


def build_fetch_yfinance_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fetch and normalize daily OHLCV data from yfinance.")
    parser.add_argument("--symbol", default=DEFAULT_SETTINGS.data.symbol, help="Ticker symbol to download")
    parser.add_argument("--output-csv", type=Path, help="Output CSV path")
    parser.add_argument("--vix-output-csv", type=Path, help="Optional output CSV path for normalized VIX data")
    parser.add_argument("--period", default="max", help="yfinance period, e.g. max, 10y, 5y")
    parser.add_argument("--start", help="Optional yfinance start date, YYYY-MM-DD")
    parser.add_argument("--end", help="Optional yfinance end date, YYYY-MM-DD")
    return parser


def _train_config_from_args(args: argparse.Namespace) -> TrainPipelineConfig:
    return TrainPipelineConfig(
        symbol=args.symbol,
        input_csv=args.input_csv or _default_input_csv(args.symbol),
        vix_input_csv=args.vix_csv,
        sentiment_features_csv=args.sentiment_features_csv,
        profile_name=args.profile,
        threshold_policy_name=args.threshold_policy,
        probability_calibration_method=args.probability_calibration,
        max_calibration_exposure=args.max_calibration_exposure,
        auto_discover_companion_inputs=True,
        fetch_yfinance=args.fetch_yfinance,
        yfinance_period=args.yf_period,
        yfinance_start=args.yf_start,
        yfinance_end=args.yf_end,
        fetch_alpha_vantage=args.fetch_alpha_vantage,
        alpha_vantage_key=args.alpha_vantage_key,
        settings=_settings_from_args(args),
        model_dir=args.model_dir,
    )


def _test_config_from_args(args: argparse.Namespace) -> TestPipelineConfig:
    return TestPipelineConfig(
        symbol=args.symbol,
        input_csv=args.input_csv or _default_input_csv(args.symbol),
        vix_input_csv=args.vix_csv,
        sentiment_features_csv=args.sentiment_features_csv,
        profile_name=args.profile,
        threshold_policy_name=args.threshold_policy,
        probability_calibration_method=args.probability_calibration,
        max_calibration_exposure=args.max_calibration_exposure,
        auto_discover_companion_inputs=True,
        fetch_yfinance=args.fetch_yfinance,
        yfinance_period=args.yf_period,
        yfinance_start=args.yf_start,
        yfinance_end=args.yf_end,
        fetch_alpha_vantage=args.fetch_alpha_vantage,
        alpha_vantage_key=args.alpha_vantage_key,
        settings=_settings_from_args(args),
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
    print(format_test_terminal_summary(result.summary, result.dataset))


def fetch_yfinance_main() -> None:
    parser = build_fetch_yfinance_arg_parser()
    args = parser.parse_args()
    output, vix_output = run_fetch_yfinance(
        args.symbol,
        args.output_csv or _default_input_csv(args.symbol),
        vix_output_csv=args.vix_output_csv or _default_vix_output_csv(),
        period=args.period,
        start=args.start,
        end=args.end,
    )
    print(f"saved_csv={output}")
    if vix_output is not None:
        print(f"saved_vix_csv={vix_output}")


def main() -> None:
    parser = build_test_arg_parser()
    args = parser.parse_args()
    result = run_pipeline(_test_config_from_args(args))
    print(format_test_terminal_summary(result.summary, result.dataset))


if __name__ == "__main__":
    main()
