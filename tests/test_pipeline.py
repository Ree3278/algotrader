from __future__ import annotations

import json

import numpy as np
import pandas as pd

from algotrader.backtest import BacktestConfig
from algotrader.ingestion.storage import save_ohlcv_csv
from algotrader.metrics import compute_debug_metrics
from algotrader.pipeline import TestPipelineConfig, TrainPipelineConfig, run_pipeline, run_test_pipeline, run_training_pipeline
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
        backtest_config=BacktestConfig(
            probability_threshold=0.55,
            commission_bps=0.0,
            slippage_bps=0.0,
        ),
        threshold_grid=(0.45, 0.55),
        min_training_size=50,
        min_calibration_size=15,
    )


def test_train_then_test_pipeline_from_local_csv_writes_artifacts(tmp_path) -> None:
    price_frame = _synthetic_price_frame()
    input_csv = tmp_path / "spy_daily.csv"
    model_dir = tmp_path / "models"
    output_dir = tmp_path / "reports"
    save_ohlcv_csv(price_frame, input_csv)

    train_config = TrainPipelineConfig(
        symbol="SPY",
        input_csv=input_csv,
        model_dir=model_dir,
        experiment_config=_experiment_config(),
    )
    test_config = TestPipelineConfig(
        symbol="SPY",
        input_csv=input_csv,
        model_dir=model_dir,
        output_dir=output_dir,
        experiment_config=_experiment_config(),
    )

    train_result = run_training_pipeline(train_config)
    test_result = run_test_pipeline(test_config)

    assert train_result.artifact_paths["manifest"].exists()
    assert train_result.artifact_paths["fold_manifest"].exists()
    assert not train_result.fold_manifest.empty

    assert not test_result.fold_summaries.empty
    assert not test_result.test_predictions.empty
    assert test_result.summary["symbol"] == "SPY"
    assert test_result.summary["fold_count"] >= 1
    assert test_result.report_paths["fold_summaries"].exists()
    assert test_result.report_paths["test_predictions"].exists()
    assert test_result.report_paths["summary"].exists()

    saved_summary = json.loads(test_result.report_paths["summary"].read_text(encoding="utf-8"))
    assert saved_summary["symbol"] == "SPY"
    assert saved_summary["model_backend"] == "hist_gradient_boosting"


def test_run_pipeline_wrapper_still_executes_train_plus_test(tmp_path) -> None:
    price_frame = _synthetic_price_frame()
    input_csv = tmp_path / "spy_daily.csv"
    save_ohlcv_csv(price_frame, input_csv)

    result = run_pipeline(
        TestPipelineConfig(
            symbol="SPY",
            input_csv=input_csv,
            model_dir=tmp_path / "models",
            output_dir=tmp_path / "reports",
            experiment_config=_experiment_config(),
        )
    )

    assert result.summary["fold_count"] >= 1


def test_compute_debug_metrics_reads_saved_artifacts(tmp_path) -> None:
    price_frame = _synthetic_price_frame()
    input_csv = tmp_path / "spy_daily.csv"
    model_dir = tmp_path / "models"
    output_dir = tmp_path / "reports"
    save_ohlcv_csv(price_frame, input_csv)

    run_pipeline(
        TestPipelineConfig(
            symbol="SPY",
            input_csv=input_csv,
            model_dir=model_dir,
            output_dir=output_dir,
            experiment_config=_experiment_config(),
        )
    )

    metrics = compute_debug_metrics(
        input_csv=input_csv,
        model_dir=model_dir,
        reports_dir=output_dir,
    )

    assert metrics["dataset_rows"] > 0
    assert metrics["fold_count"] >= 1
    assert "label_distribution_pct" in metrics
    assert "hit_reason_pct" in metrics
    assert len(metrics["fold_sharpes"]) == metrics["fold_count"]
