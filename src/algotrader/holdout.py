"""Frozen-baseline holdout evaluation on a final untouched time slice."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from algotrader.backtest import BacktestConfig, run_long_flat_backtest, summarize_backtest
from algotrader.pipeline import (
    PipelineConfig,
    _add_shared_data_args,
    _add_shared_model_args,
    _build_dataset,
    _load_or_fetch_frames,
    _resolved_experiment_config,
)
from algotrader.reporting import build_experiment_summary, format_test_terminal_summary, write_experiment_reports
from algotrader.settings import DEFAULT_SETTINGS
from algotrader.training.calibration import ProbabilityCalibrator, apply_probability_calibration, fit_probability_calibrator
from algotrader.training.experiment import (
    WalkForwardExperimentResult,
    build_threshold_application,
    default_threshold_selection,
    select_thresholds,
)
from algotrader.training.xgboost_model import train_xgboost_classifier


@dataclass(frozen=True)
class HoldoutPipelineConfig(PipelineConfig):
    output_dir: Path = DEFAULT_SETTINGS.paths.holdout_report_dir
    holdout_size: int = DEFAULT_SETTINGS.holdout.size


@dataclass(frozen=True)
class HoldoutRunResult:
    dataset: pd.DataFrame
    holdout_dataset: pd.DataFrame
    price_frame: pd.DataFrame
    summary: dict[str, Any]
    report_paths: dict[str, Path]


def run_holdout_pipeline(config: HoldoutPipelineConfig) -> HoldoutRunResult:
    price_frame, vix_frame, sentiment_frame = _load_or_fetch_frames(config)
    dataset = _build_dataset(config, price_frame, vix_frame=vix_frame, sentiment_frame=sentiment_frame)
    experiment_config = _resolved_experiment_config(config)

    if len(dataset.data) <= config.holdout_size:
        raise ValueError("Holdout size is too large for the available dataset")

    train_data = dataset.data.iloc[:-config.holdout_size]
    holdout_data = dataset.data.iloc[-config.holdout_size:]
    if len(train_data) < experiment_config.min_training_size:
        raise ValueError("Training data before the holdout slice is too small")

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

    holdout_probabilities = pd.Series(
        final_model.predict_proba(holdout_data[dataset.feature_columns])[:, 1],
        index=holdout_data.index,
    )
    holdout_probabilities = apply_probability_calibration(calibrator, holdout_probabilities)
    threshold_series, threshold_regimes = build_threshold_application(
        holdout_data,
        threshold_policy_name=threshold_selection.policy_name,
        thresholds_by_regime=threshold_selection.thresholds_by_regime,
    )
    backtest_config = BacktestConfig(
        probability_threshold=threshold_selection.representative_threshold,
        commission_bps=experiment_config.backtest_config.commission_bps,
        slippage_bps=experiment_config.backtest_config.slippage_bps,
    )
    backtest_results = run_long_flat_backtest(
        price_frame,
        holdout_data,
        holdout_probabilities,
        config=backtest_config,
        threshold_series=threshold_series,
    )
    metrics = summarize_backtest(backtest_results)

    result = WalkForwardExperimentResult(
        fold_summaries=pd.DataFrame(
            [
                {
                    "fold": 1,
                    "train_size": int(len(train_data)),
                    "calibration_size": int(len(calibration_data)),
                    "test_size": int(len(holdout_data)),
                    "model_backend": getattr(final_model, "_algotrader_backend", "unknown"),
                    "threshold_policy_name": threshold_selection.policy_name,
                    "probability_calibration_method": calibrator.method,
                    "selected_threshold": float(threshold_selection.representative_threshold),
                    **metrics,
                }
            ]
        ),
        test_predictions=pd.DataFrame(
            {
                "fold": 1,
                "probability_long": holdout_probabilities,
                "label": holdout_data[dataset.target_column],
                "probability_calibration_method": calibrator.method,
                "selected_threshold": threshold_series,
                "threshold_regime": threshold_regimes,
            }
        ),
    )
    summary = build_experiment_summary(
        result,
        symbol=config.symbol,
        dataset_rows=len(holdout_data),
        feature_count=len(dataset.feature_columns),
        model_backend=getattr(final_model, "_algotrader_backend", "unknown"),
    )
    summary.update(
        {
            "evaluation_mode": "frozen_holdout",
            "profile_name": config.profile_name,
            "threshold_policy_name": threshold_selection.policy_name,
            "probability_calibration_method": calibrator.method,
            "max_calibration_exposure": experiment_config.max_calibration_exposure,
            "holdout_size": int(config.holdout_size),
            "holdout_start": holdout_data.index.min().isoformat(),
            "holdout_end": holdout_data.index.max().isoformat(),
            "pre_holdout_train_rows": int(len(train_data)),
        }
    )
    report_paths = write_experiment_reports(result, config.output_dir, summary=summary)
    return HoldoutRunResult(
        dataset=dataset.data,
        holdout_dataset=holdout_data,
        price_frame=price_frame,
        summary=summary,
        report_paths=report_paths,
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate the frozen baseline on a final untouched holdout slice.")
    _add_shared_data_args(parser)
    _add_shared_model_args(parser)
    parser.add_argument(
        "--holdout-size",
        type=int,
        default=DEFAULT_SETTINGS.holdout.size,
        help="Number of final labeled rows reserved for untouched holdout evaluation",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_SETTINGS.paths.holdout_report_dir,
        help="Directory for holdout reports",
    )
    return parser


def _config_from_args(args: argparse.Namespace) -> HoldoutPipelineConfig:
    from algotrader.pipeline import _default_input_csv, _settings_from_args

    return HoldoutPipelineConfig(
        symbol=args.symbol,
        input_csv=args.input_csv or _default_input_csv(args.symbol),
        vix_input_csv=args.vix_csv,
        sentiment_features_csv=args.sentiment_features_csv,
        profile_name=args.profile,
        threshold_policy_name=args.threshold_policy,
        probability_calibration_method=args.probability_calibration,
        max_calibration_exposure=args.max_calibration_exposure,
        fetch_yfinance=args.fetch_yfinance,
        yfinance_period=args.yf_period,
        yfinance_start=args.yf_start,
        yfinance_end=args.yf_end,
        fetch_alpha_vantage=args.fetch_alpha_vantage,
        alpha_vantage_key=args.alpha_vantage_key,
        settings=_settings_from_args(args),
        output_dir=args.output_dir,
        holdout_size=args.holdout_size,
    )


def main() -> None:
    args = build_arg_parser().parse_args()
    result = run_holdout_pipeline(_config_from_args(args))
    print(format_test_terminal_summary(result.summary, result.holdout_dataset))
    print(f"summary_json={result.report_paths['summary']}")
