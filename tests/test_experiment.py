from __future__ import annotations

import numpy as np
import pandas as pd

from algotrader.backtest import BacktestConfig
from algotrader.training.dataset import build_training_dataset
from algotrader.training import experiment as experiment_module
from algotrader.training.experiment import (
    WalkForwardExperimentConfig,
    run_walk_forward_experiment,
    select_thresholds,
)
from algotrader.training.walk_forward import PurgedWalkForwardConfig
from algotrader.training.xgboost_model import XGBoostConfig


def _synthetic_price_frame(periods: int = 520) -> pd.DataFrame:
    index = pd.date_range("2023-01-01", periods=periods, freq="D", tz="UTC")
    base = 100 + np.linspace(0, 25, periods)
    wave = 5 * np.sin(np.arange(periods) / 8)
    close = base + wave
    open_ = close * (1 + 0.002 * np.cos(np.arange(periods) / 5))
    high = np.maximum(open_, close) + 1.0
    low = np.minimum(open_, close) - 1.0
    volume = 1_000_000 + (50_000 * np.sin(np.arange(periods) / 4))
    return pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        },
        index=index,
    )


def test_run_walk_forward_experiment_produces_fold_summaries_and_predictions() -> None:
    price_frame = _synthetic_price_frame()
    dataset = build_training_dataset(price_frame)
    config = WalkForwardExperimentConfig(
        split_config=PurgedWalkForwardConfig(
            train_size=90,
            test_size=30,
            step_size=30,
            embargo_size=10,
            max_label_horizon=10,
        ),
        model_config=XGBoostConfig(
            n_estimators=25,
            max_depth=2,
            learning_rate=0.1,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=7,
        ),
        backtest_config=BacktestConfig(
            probability_threshold=0.55,
            commission_bps=0.0,
            slippage_bps=0.0,
        ),
        threshold_grid=(0.45, 0.55, 0.65),
        calibration_fraction=0.2,
        min_calibration_size=15,
        min_training_size=50,
    )

    result = run_walk_forward_experiment(dataset, price_frame, config=config)

    assert not result.fold_summaries.empty
    assert not result.test_predictions.empty
    assert {"fold", "model_backend", "selected_threshold", "total_return", "sharpe", "max_drawdown"}.issubset(
        result.fold_summaries.columns
    )
    assert result.fold_summaries["model_backend"].isin(["xgboost", "hist_gradient_boosting"]).all()
    assert result.test_predictions["selected_threshold"].isin([0.45, 0.55, 0.65]).all()
    assert set(result.test_predictions["fold"].unique()) == set(result.fold_summaries["fold"].unique())


def test_run_walk_forward_experiment_supports_platt_calibration() -> None:
    price_frame = _synthetic_price_frame()
    dataset = build_training_dataset(price_frame)
    config = WalkForwardExperimentConfig(
        split_config=PurgedWalkForwardConfig(
            train_size=90,
            test_size=30,
            step_size=30,
            embargo_size=10,
            max_label_horizon=10,
        ),
        model_config=XGBoostConfig(
            n_estimators=25,
            max_depth=2,
            learning_rate=0.1,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=7,
            backend="hist_gradient_boosting",
        ),
        backtest_config=BacktestConfig(
            probability_threshold=0.55,
            commission_bps=0.0,
            slippage_bps=0.0,
        ),
        threshold_grid=(0.45, 0.55, 0.65),
        calibration_fraction=0.2,
        min_calibration_size=15,
        min_training_size=50,
        probability_calibration_method="platt",
    )

    result = run_walk_forward_experiment(dataset, price_frame, config=config)

    assert not result.fold_summaries.empty


def test_select_thresholds_falls_back_to_min_exposure_when_cap_is_infeasible() -> None:
    price_frame = _synthetic_price_frame(periods=120)
    dataset = build_training_dataset(price_frame)
    calibration_data = dataset.data.iloc[:30]
    calibration_probabilities = pd.Series(0.99, index=calibration_data.index)

    selection = select_thresholds(
        price_frame,
        calibration_data,
        calibration_probabilities,
        BacktestConfig(
            probability_threshold=0.55,
            commission_bps=0.0,
            slippage_bps=0.0,
        ),
        threshold_grid=(0.45, 0.55, 0.65),
        threshold_policy_name="global",
        max_calibration_exposure=-0.1,
    )

    assert selection.policy_name == "global"
    assert selection.selection_mode == "fallback_min_exposure"
    assert selection.feasible_candidate_count == 0
    assert selection.calibration_exposure is not None
    assert selection.calibration_exposure >= 0.0


def test_soft_risk_adjusted_objective_can_prefer_lower_exposure_candidate(monkeypatch) -> None:
    price_frame = _synthetic_price_frame(periods=520)
    dataset = build_training_dataset(price_frame)
    calibration_data = dataset.data.iloc[:30]
    calibration_probabilities = pd.Series(0.60, index=calibration_data.index)

    def fake_backtest(_price_frame, _calibration_data, _calibration_probabilities, *, config, threshold_series):
        threshold = float(threshold_series.iloc[0])
        return pd.DataFrame({"candidate_threshold": [threshold]})

    def fake_summary(results):
        threshold = float(results["candidate_threshold"].iloc[0])
        if threshold == 0.45:
            return {
                "sharpe": 0.55,
                "total_return": 0.020,
                "turnover": 30.0,
                "exposure": 0.95,
                "max_drawdown": -0.10,
            }
        return {
            "sharpe": 0.50,
            "total_return": 0.020,
            "turnover": 20.0,
            "exposure": 0.40,
            "max_drawdown": -0.08,
        }

    monkeypatch.setattr(experiment_module, "run_long_flat_backtest", fake_backtest)
    monkeypatch.setattr(experiment_module, "summarize_backtest", fake_summary)

    legacy = select_thresholds(
        price_frame,
        calibration_data,
        calibration_probabilities,
        BacktestConfig(probability_threshold=0.55, commission_bps=0.0, slippage_bps=0.0),
        threshold_grid=(0.45, 0.55),
        threshold_policy_name="global",
    )
    soft = select_thresholds(
        price_frame,
        calibration_data,
        calibration_probabilities,
        BacktestConfig(probability_threshold=0.55, commission_bps=0.0, slippage_bps=0.0),
        threshold_grid=(0.45, 0.55),
        threshold_policy_name="global",
        threshold_selection_objective_name="soft_risk_adjusted",
        calibration_return_weight=5.0,
        calibration_exposure_target=0.70,
        calibration_exposure_penalty=1.0,
        calibration_turnover_penalty=0.0025,
        calibration_drawdown_target=0.12,
        calibration_drawdown_penalty=2.0,
    )

    assert legacy.thresholds_by_regime == {"all": 0.45}
    assert soft.thresholds_by_regime == {"all": 0.55}
