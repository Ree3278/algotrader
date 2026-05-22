"""Walk-forward training and evaluation pipeline."""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field

import pandas as pd

from algotrader.backtest import BacktestConfig, run_long_flat_backtest, summarize_backtest
from algotrader.thresholds import build_threshold_policy
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
    threshold_policy_name: str = "global"


@dataclass(frozen=True)
class WalkForwardExperimentResult:
    fold_summaries: pd.DataFrame
    test_predictions: pd.DataFrame


@dataclass(frozen=True)
class ThresholdSelection:
    policy_name: str
    thresholds_by_regime: dict[str, float]

    @property
    def representative_threshold(self) -> float:
        values = list(self.thresholds_by_regime.values())
        return float(sum(values) / len(values)) if values else 0.0


def default_threshold_selection(
    *,
    threshold_policy_name: str,
    default_threshold: float,
) -> ThresholdSelection:
    policy = build_threshold_policy(threshold_policy_name)
    return ThresholdSelection(
        policy_name=policy.name,
        thresholds_by_regime={regime: float(default_threshold) for regime in policy.regime_names},
    )


def build_threshold_application(
    signal_frame: pd.DataFrame,
    *,
    threshold_policy_name: str,
    thresholds_by_regime: dict[str, float],
) -> tuple[pd.Series, pd.Series]:
    policy = build_threshold_policy(threshold_policy_name)
    return policy.build_threshold_series(signal_frame, thresholds_by_regime)


def select_thresholds(
    price_frame: pd.DataFrame,
    calibration_data: pd.DataFrame,
    calibration_probabilities: pd.Series,
    base_config: BacktestConfig,
    threshold_grid: tuple[float, ...],
    threshold_policy_name: str,
) -> ThresholdSelection:
    policy = build_threshold_policy(threshold_policy_name)
    best_thresholds = {regime: base_config.probability_threshold for regime in policy.regime_names}
    best_score = None

    for threshold_values in itertools.product(threshold_grid, repeat=len(policy.regime_names)):
        threshold_map = {
            regime: float(threshold)
            for regime, threshold in zip(policy.regime_names, threshold_values, strict=True)
        }
        if not policy.allows_threshold_map(threshold_map):
            continue
        threshold_series, _ = policy.build_threshold_series(calibration_data, threshold_map)
        results = run_long_flat_backtest(
            price_frame,
            calibration_data,
            calibration_probabilities,
            config=BacktestConfig(
                probability_threshold=base_config.probability_threshold,
                commission_bps=base_config.commission_bps,
                slippage_bps=base_config.slippage_bps,
            ),
            threshold_series=threshold_series,
        )
        metrics = summarize_backtest(results)
        score = (
            metrics["sharpe"],
            metrics["total_return"],
            -metrics["turnover"],
        )
        if best_score is None or score > best_score:
            best_score = score
            best_thresholds = threshold_map

    if best_score is None:
        raise ValueError(
            f"Threshold policy '{threshold_policy_name}' rejected all candidate threshold combinations"
        )

    return ThresholdSelection(
        policy_name=policy.name,
        thresholds_by_regime=best_thresholds,
    )


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
        calibration_data = train_data.iloc[0:0]
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
            threshold_selection = select_thresholds(
                price_frame,
                calibration_data,
                calibration_probabilities,
                config.backtest_config,
                config.threshold_grid,
                config.threshold_policy_name,
            )
        else:
            threshold_selection = default_threshold_selection(
                threshold_policy_name=config.threshold_policy_name,
                default_threshold=config.backtest_config.probability_threshold,
            )

        final_model = train_xgboost_classifier(
            train_data[dataset.feature_columns],
            train_data[dataset.target_column],
            config=config.model_config,
        )
        test_probabilities = pd.Series(
            final_model.predict_proba(test_data[dataset.feature_columns])[:, 1],
            index=test_data.index,
        )

        threshold_series, threshold_regimes = build_threshold_application(
            test_data,
            threshold_policy_name=threshold_selection.policy_name,
            thresholds_by_regime=threshold_selection.thresholds_by_regime,
        )
        fold_backtest_config = BacktestConfig(
            probability_threshold=threshold_selection.representative_threshold,
            commission_bps=config.backtest_config.commission_bps,
            slippage_bps=config.backtest_config.slippage_bps,
        )
        backtest_results = run_long_flat_backtest(
            price_frame,
            test_data,
            test_probabilities,
            config=fold_backtest_config,
            threshold_series=threshold_series,
        )
        metrics = summarize_backtest(backtest_results)

        prediction_frame = pd.DataFrame(
            {
                "fold": split.fold,
                "probability_long": test_probabilities,
                "label": test_data[dataset.target_column],
                "selected_threshold": threshold_series,
                "threshold_regime": threshold_regimes,
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
                "threshold_policy_name": threshold_selection.policy_name,
                "selected_threshold": float(threshold_selection.representative_threshold),
                **metrics,
            }
        )

    fold_summary_frame = pd.DataFrame(fold_summaries)
    prediction_output = pd.concat(prediction_frames).sort_index() if prediction_frames else pd.DataFrame()
    return WalkForwardExperimentResult(
        fold_summaries=fold_summary_frame,
        test_predictions=prediction_output,
    )
