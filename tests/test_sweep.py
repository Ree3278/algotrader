from __future__ import annotations

import json

import numpy as np
import pandas as pd

from algotrader.ingestion.storage import save_ohlcv_csv
from algotrader.pipeline import TestPipelineConfig
from algotrader.sweep import LabelSweepGrid, run_label_sweep
from algotrader.training.experiment import WalkForwardExperimentConfig
from algotrader.training.walk_forward import PurgedWalkForwardConfig
from algotrader.training.xgboost_model import XGBoostConfig


def _synthetic_price_frame(periods: int = 360) -> pd.DataFrame:
    index = pd.date_range("2023-01-01", periods=periods, freq="D", tz="UTC")
    base = 100 + np.linspace(0, 18, periods)
    wave = 3 * np.sin(np.arange(periods) / 8)
    close = base + wave
    open_ = close * (1 + 0.002 * np.cos(np.arange(periods) / 7))
    high = np.maximum(open_, close) + 1.0
    low = np.minimum(open_, close) - 1.0
    volume = 1_000_000 + 50_000 * np.sin(np.arange(periods) / 6)
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


def test_run_label_sweep_writes_aggregate_outputs(tmp_path) -> None:
    price_frame = _synthetic_price_frame()
    input_csv = tmp_path / "spy_daily.csv"
    save_ohlcv_csv(price_frame, input_csv)

    results, paths = run_label_sweep(
        TestPipelineConfig(
            symbol="SPY",
            input_csv=input_csv,
            profile_name="price_only",
            experiment_config=_experiment_config(),
        ),
        output_dir=tmp_path / "label_sweep",
        grid=LabelSweepGrid(
            profit_target_atrs=(1.0, 1.5),
            stop_loss_atrs=(1.0,),
            max_holding_bars=(5,),
            timeout_return_thresholds=(0.0,),
        ),
    )

    assert len(results) == 2
    assert paths["csv"].exists()
    assert paths["json"].exists()
    assert paths["summary"].exists()
    assert {"run_slug", "mean_sharpe", "label_flat_pct", "hit_stop_loss_pct"}.issubset(results.columns)

    saved_summary = json.loads(paths["summary"].read_text(encoding="utf-8"))
    assert saved_summary["run_count"] == 2
    assert saved_summary["best_by_mean_sharpe"] is not None
