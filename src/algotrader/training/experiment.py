"""Walk-forward training and evaluation pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from algotrader.backtest import BacktestConfig, run_long_flat_backtest, summarize_backtest
from algotrader.training.dataset import TrainingDataset
from algotrader.training.walk_forward import PurgedWalkForwardConfig, generate_splits
from algotrader.training.xgboost_model import XGBoostConfig, train_xgboost_classifier


@dataclass(frozen=True)
class WalkForwardExperimentConfig:
    split_config: PurgedWalkForwardConfig
    model_config: XGBoostConfig = field(default_factory=XGBoostConfig)
    backtest_config: BacktestConfig = field(default_factory=BacktestConfig)
    threshold_grid: tuple[float, ...] = (0.5, 0.55, 0.6, 0.65)
    calibration_fraction: float = 0.2
    min_calibration_size: int = 20
    min_training_size: int = 30


@dataclass(frozen=True)
class WalkForwardExperimentResult:
    fold_summaries: pd.DataFrame
    test_predictions: pd.DataFrame


def _select_threshold(
    price_frame: pd.DataFrame,
    calibration_data: pd.DataFrame,
    calibration_probabilities: pd.Series,
    base_config: BacktestConfig,
    threshold_grid: tuple[float, ...],
) -> float:
    best_threshold = base_config.probability_threshold
    best_score = None

    for threshold in threshold_grid:
        config = BacktestConfig(
            probability_threshold=threshold,
            commission_bps=base_config.commission_bps,
            slippage_bps=base_config.slippage_bps,
        )
        results = run_long_flat_backtest(price_frame, calibration_data, calibration_probabilities, config=config)
        metrics = summarize_backtest(results)
        score = (
            metrics["sharpe"],
            metrics["total_return"],
            -metrics["turnover"],
        )
        if best_score is None or score > best_score:
            best_score = score
            best_threshold = threshold

    return best_threshold


def run_walk_forward_experiment(
    dataset: TrainingDataset,
    price_frame: pd.DataFrame,
    *,
    config: WalkForwardExperimentConfig,
) -> WalkForwardExperimentResult:
    """Run walk-forward training, threshold tuning, and test-time backtests."""

    fold_summaries: list[dict[str, float | int]] = []
    prediction_frames: list[pd.DataFrame] = []

    for split in generate_splits(dataset.data.index, config.split_config):
        train_data = dataset.data.iloc[split.train_indices]
        test_data = dataset.data.iloc[split.test_indices]

        if len(train_data) < config.min_training_size or test_data.empty:
            continue

        calibration_size = max(int(len(train_data) * config.calibration_fraction), config.min_calibration_size)
        if len(train_data) >= config.min_training_size + calibration_size:
            train_core = train_data.iloc[:-calibration_size]
            calibration_data = train_data.iloc[-calibration_size:]

            calibration_model = train_xgboost_classifier(
                train_core[dataset.feature_columns],
                train_core[dataset.target_column],
                config=config.model_config,
            )
            calibration_probabilities = pd.Series(
                calibration_model.predict_proba(calibration_data[dataset.feature_columns])[:, 1],
                index=calibration_data.index,
            )
            selected_threshold = _select_threshold(
                price_frame,
                calibration_data,
                calibration_probabilities,
                config.backtest_config,
                config.threshold_grid,
            )
        else:
            calibration_data = train_data.iloc[0:0]
            selected_threshold = config.backtest_config.probability_threshold

        final_model = train_xgboost_classifier(
            train_data[dataset.feature_columns],
            train_data[dataset.target_column],
            config=config.model_config,
        )
        test_probabilities = pd.Series(
            final_model.predict_proba(test_data[dataset.feature_columns])[:, 1],
            index=test_data.index,
        )

        fold_backtest_config = BacktestConfig(
            probability_threshold=selected_threshold,
            commission_bps=config.backtest_config.commission_bps,
            slippage_bps=config.backtest_config.slippage_bps,
        )
        backtest_results = run_long_flat_backtest(price_frame, test_data, test_probabilities, config=fold_backtest_config)
        metrics = summarize_backtest(backtest_results)

        prediction_frame = pd.DataFrame(
            {
                "fold": split.fold,
                "probability_long": test_probabilities,
                "label": test_data[dataset.target_column],
                "selected_threshold": selected_threshold,
            }
        )
        prediction_frames.append(prediction_frame)

        fold_summaries.append(
            {
                "fold": split.fold,
                "train_size": int(len(train_data)),
                "calibration_size": int(len(calibration_data)),
                "test_size": int(len(test_data)),
                "model_backend": getattr(final_model, "_algotrader_backend", "unknown"),
                "selected_threshold": float(selected_threshold),
                **metrics,
            }
        )

    fold_summary_frame = pd.DataFrame(fold_summaries)
    prediction_output = pd.concat(prediction_frames).sort_index() if prediction_frames else pd.DataFrame()
    return WalkForwardExperimentResult(
        fold_summaries=fold_summary_frame,
        test_predictions=prediction_output,
    )
