from __future__ import annotations

import json

import numpy as np
import pandas as pd

from algotrader.backtest import BacktestConfig
from algotrader.ingestion.storage import save_ohlcv_csv
from algotrader.pipeline import PipelineConfig, run_pipeline
from algotrader.training.experiment import WalkForwardExperimentConfig
from algotrader.training.walk_forward import PurgedWalkForwardConfig
from algotrader.training.xgboost_model import XGBoostConfig


def _synthetic_price_frame(periods: int = 320) -> pd.DataFrame:
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


def test_run_pipeline_from_local_csv_writes_reports(tmp_path) -> None:
    price_frame = _synthetic_price_frame()
    input_csv = tmp_path / "spy_daily.csv"
    output_dir = tmp_path / "reports"
    save_ohlcv_csv(price_frame, input_csv)

    config = PipelineConfig(
        symbol="SPY",
        input_csv=input_csv,
        output_dir=output_dir,
        experiment_config=WalkForwardExperimentConfig(
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
            backtest_config=BacktestConfig(
                probability_threshold=0.55,
                commission_bps=0.0,
                slippage_bps=0.0,
            ),
            threshold_grid=(0.45, 0.55),
            min_training_size=50,
            min_calibration_size=15,
        ),
    )

    result = run_pipeline(config)

    assert not result.fold_summaries.empty
    assert not result.test_predictions.empty
    assert result.summary["symbol"] == "SPY"
    assert result.summary["fold_count"] >= 1
    assert result.report_paths["fold_summaries"].exists()
    assert result.report_paths["test_predictions"].exists()
    assert result.report_paths["summary"].exists()

    saved_summary = json.loads(result.report_paths["summary"].read_text(encoding="utf-8"))
    assert saved_summary["symbol"] == "SPY"
    assert saved_summary["model_backend"] == "hist_gradient_boosting"
