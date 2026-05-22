from __future__ import annotations

import numpy as np
import pandas as pd

from algotrader.backtest import BacktestConfig
from algotrader.training.dataset import build_training_dataset
from algotrader.training.experiment import WalkForwardExperimentConfig, run_walk_forward_experiment
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
