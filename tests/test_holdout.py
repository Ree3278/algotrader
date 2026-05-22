from __future__ import annotations

import numpy as np
import pandas as pd

from algotrader.holdout import HoldoutPipelineConfig, run_holdout_pipeline
from algotrader.ingestion.storage import save_ohlcv_csv
from algotrader.training.experiment import WalkForwardExperimentConfig
from algotrader.training.walk_forward import PurgedWalkForwardConfig
from algotrader.training.xgboost_model import XGBoostConfig


def _synthetic_price_frame(periods: int = 520) -> pd.DataFrame:
    index = pd.date_range("2023-01-01", periods=periods, freq="D", tz="UTC")
    base = 100 + np.linspace(0, 25, periods)
    wave = 4 * np.sin(np.arange(periods) / 9)
    close = base + wave
    open_ = close * (1 + 0.002 * np.cos(np.arange(periods) / 6))
    high = np.maximum(open_, close) + 1.0
    low = np.minimum(open_, close) - 1.0
    volume = 1_000_000 + (60_000 * np.sin(np.arange(periods) / 5))
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


def _synthetic_vix_frame(periods: int = 520) -> pd.DataFrame:
    index = pd.date_range("2023-01-01", periods=periods, freq="D", tz="UTC")
    close = 20 + 4 * np.sin(np.arange(periods) / 11) + 0.5 * np.cos(np.arange(periods) / 3)
    return pd.DataFrame(
        {
            "symbol": "^VIX",
            "open": close,
            "high": close + 0.5,
            "low": close - 0.5,
            "close": close,
            "adjusted_close": close,
            "volume": np.zeros(periods),
            "dividend_amount": np.zeros(periods),
            "split_coefficient": np.ones(periods),
        },
        index=index,
    )


def _experiment_config() -> WalkForwardExperimentConfig:
    return WalkForwardExperimentConfig(
        split_config=PurgedWalkForwardConfig(
            train_size=90,
            test_size=30,
            step_size=30,
            embargo_size=10,
            max_label_horizon=10,
        ),
        model_config=XGBoostConfig(
            n_estimators=20,
            max_depth=2,
            learning_rate=0.1,
            backend="hist_gradient_boosting",
            random_state=11,
        ),
        threshold_grid=(0.45, 0.55),
        min_training_size=50,
        min_calibration_size=15,
    )


def test_run_holdout_pipeline_writes_frozen_holdout_reports(tmp_path) -> None:
    price_frame = _synthetic_price_frame()
    vix_frame = _synthetic_vix_frame()
    input_csv = tmp_path / "spy_daily.csv"
    vix_csv = tmp_path / "vix_daily.csv"
    save_ohlcv_csv(price_frame, input_csv)
    save_ohlcv_csv(vix_frame, vix_csv)

    result = run_holdout_pipeline(
        HoldoutPipelineConfig(
            symbol="SPY",
            input_csv=input_csv,
            vix_input_csv=vix_csv,
            output_dir=tmp_path / "holdout_reports",
            holdout_size=60,
            experiment_config=_experiment_config(),
        )
    )

    assert result.summary["evaluation_mode"] == "frozen_holdout"
    assert result.summary["threshold_policy_name"] == "trend_regime"
    assert result.summary["holdout_size"] == 60
    assert result.report_paths["summary"].exists()
    assert len(result.holdout_dataset) == 60
