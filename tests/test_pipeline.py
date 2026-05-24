from __future__ import annotations

import json

import numpy as np
import pandas as pd

from algotrader.backtest import BacktestConfig
from algotrader.ingestion.storage import save_ohlcv_csv
from algotrader.metrics import compute_debug_metrics
from algotrader.pipeline import TestPipelineConfig, TrainPipelineConfig, run_pipeline, run_test_pipeline, run_training_pipeline
from algotrader.reporting import format_test_terminal_summary
from algotrader.settings import DEFAULT_SETTINGS
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


def _synthetic_sentiment_frame(periods: int = 520) -> pd.DataFrame:
    index = pd.date_range("2023-01-01", periods=periods, freq="D", tz="UTC")
    return pd.DataFrame(
        {
            "net_sentiment": np.sin(np.arange(periods) / 15),
            "abs_emotion": 0.5 + 0.1 * np.cos(np.arange(periods) / 7),
            "is_empty_block": np.zeros(periods),
            "headline_count": 3 + (np.arange(periods) % 4),
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
    vix_frame = _synthetic_vix_frame()
    input_csv = tmp_path / "spy_daily.csv"
    vix_csv = tmp_path / "vix_daily.csv"
    model_dir = tmp_path / "models"
    output_dir = tmp_path / "reports"
    save_ohlcv_csv(price_frame, input_csv)
    save_ohlcv_csv(vix_frame, vix_csv)

    train_config = TrainPipelineConfig(
        symbol="SPY",
        input_csv=input_csv,
        vix_input_csv=vix_csv,
        model_dir=model_dir,
        experiment_config=_experiment_config(),
    )
    test_config = TestPipelineConfig(
        symbol="SPY",
        input_csv=input_csv,
        vix_input_csv=vix_csv,
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
    assert train_result.manifest["threshold_policy_name"] == DEFAULT_SETTINGS.thresholds.default_policy_name
    assert train_result.manifest["probability_calibration_method"] == DEFAULT_SETTINGS.experiment.probability_calibration_method
    assert (
        train_result.manifest["threshold_selection_objective_name"]
        == DEFAULT_SETTINGS.experiment.threshold_selection_objective_name
    )
    assert train_result.manifest["experiment_spec"]["profile"]["block_names"] == ["price_only", "regime", "trend_state"]
    assert train_result.manifest["label_config"]["max_holding_bars"] == DEFAULT_SETTINGS.labels.max_holding_bars
    assert train_result.manifest["label_config"]["profit_target_atr"] == DEFAULT_SETTINGS.labels.profit_target_atr
    assert train_result.manifest["label_config"]["stop_loss_atr"] == DEFAULT_SETTINGS.labels.stop_loss_atr
    assert train_result.manifest["label_config"]["timeout_return_threshold"] == DEFAULT_SETTINGS.labels.timeout_return_threshold


def test_run_pipeline_wrapper_still_executes_train_plus_test(tmp_path) -> None:
    price_frame = _synthetic_price_frame()
    vix_frame = _synthetic_vix_frame()
    input_csv = tmp_path / "spy_daily.csv"
    vix_csv = tmp_path / "vix_daily.csv"
    save_ohlcv_csv(price_frame, input_csv)
    save_ohlcv_csv(vix_frame, vix_csv)

    result = run_pipeline(
        TestPipelineConfig(
            symbol="SPY",
            input_csv=input_csv,
            vix_input_csv=vix_csv,
            model_dir=tmp_path / "models",
            output_dir=tmp_path / "reports",
            experiment_config=_experiment_config(),
        )
    )

    assert result.summary["fold_count"] >= 1


def test_terminal_summary_includes_requested_headline_metrics(tmp_path) -> None:
    price_frame = _synthetic_price_frame()
    vix_frame = _synthetic_vix_frame()
    input_csv = tmp_path / "spy_daily.csv"
    vix_csv = tmp_path / "vix_daily.csv"
    save_ohlcv_csv(price_frame, input_csv)
    save_ohlcv_csv(vix_frame, vix_csv)

    result = run_pipeline(
        TestPipelineConfig(
            symbol="SPY",
            input_csv=input_csv,
            vix_input_csv=vix_csv,
            model_dir=tmp_path / "models",
            output_dir=tmp_path / "reports",
            experiment_config=_experiment_config(),
        )
    )

    terminal_summary = format_test_terminal_summary(result.summary, result.dataset)
    assert "Mean Total Return:" in terminal_summary
    assert "Mean Sharpe:" in terminal_summary
    assert "Mean Trade Count:" in terminal_summary
    assert "Mean Max Drawdown:" in terminal_summary
    assert "Label Distribution:" in terminal_summary
    assert "Hit-Reason Distribution:" in terminal_summary


def test_compute_debug_metrics_reads_saved_artifacts(tmp_path) -> None:
    price_frame = _synthetic_price_frame()
    vix_frame = _synthetic_vix_frame()
    input_csv = tmp_path / "spy_daily.csv"
    vix_csv = tmp_path / "vix_daily.csv"
    model_dir = tmp_path / "models"
    output_dir = tmp_path / "reports"
    save_ohlcv_csv(price_frame, input_csv)
    save_ohlcv_csv(vix_frame, vix_csv)

    run_pipeline(
        TestPipelineConfig(
            symbol="SPY",
            input_csv=input_csv,
            vix_input_csv=vix_csv,
            model_dir=model_dir,
            output_dir=output_dir,
            experiment_config=_experiment_config(),
        )
    )

    metrics = compute_debug_metrics(
        input_csv=input_csv,
        vix_csv=vix_csv,
        model_dir=model_dir,
        reports_dir=output_dir,
    )

    assert metrics["dataset_rows"] > 0
    assert metrics["fold_count"] >= 1
    assert "label_distribution_pct" in metrics
    assert "hit_reason_pct" in metrics
    assert len(metrics["fold_sharpes"]) == metrics["fold_count"]


def test_train_and_test_pipeline_supports_regime_conditional_thresholding(tmp_path) -> None:
    price_frame = _synthetic_price_frame()
    vix_frame = _synthetic_vix_frame()
    input_csv = tmp_path / "spy_daily.csv"
    vix_csv = tmp_path / "vix_daily.csv"
    model_dir = tmp_path / "models"
    output_dir = tmp_path / "reports"
    save_ohlcv_csv(price_frame, input_csv)
    save_ohlcv_csv(vix_frame, vix_csv)

    run_training_pipeline(
        TrainPipelineConfig(
            symbol="SPY",
            input_csv=input_csv,
            vix_input_csv=vix_csv,
            profile_name="price_plus_regime_plus_trend_state",
            threshold_policy_name="trend_regime",
            model_dir=model_dir,
            experiment_config=_experiment_config(),
        )
    )
    test_result = run_test_pipeline(
        TestPipelineConfig(
            symbol="SPY",
            input_csv=input_csv,
            vix_input_csv=vix_csv,
            profile_name="price_plus_regime_plus_trend_state",
            threshold_policy_name="trend_regime",
            model_dir=model_dir,
            output_dir=output_dir,
            experiment_config=_experiment_config(),
        )
    )

    assert "threshold_regime" in test_result.test_predictions.columns
    assert set(test_result.test_predictions["threshold_regime"].dropna().unique()).issubset({"bull_trend", "other"})


def test_train_and_test_pipeline_supports_platt_probability_calibration(tmp_path) -> None:
    price_frame = _synthetic_price_frame()
    vix_frame = _synthetic_vix_frame()
    input_csv = tmp_path / "spy_daily.csv"
    vix_csv = tmp_path / "vix_daily.csv"
    model_dir = tmp_path / "models"
    output_dir = tmp_path / "reports"
    save_ohlcv_csv(price_frame, input_csv)
    save_ohlcv_csv(vix_frame, vix_csv)

    train_result = run_training_pipeline(
        TrainPipelineConfig(
            symbol="SPY",
            input_csv=input_csv,
            vix_input_csv=vix_csv,
            profile_name="price_plus_regime_plus_trend_state",
            threshold_policy_name="trend_regime",
            probability_calibration_method="platt",
            model_dir=model_dir,
            experiment_config=_experiment_config(),
        )
    )
    test_result = run_test_pipeline(
        TestPipelineConfig(
            symbol="SPY",
            input_csv=input_csv,
            vix_input_csv=vix_csv,
            profile_name="price_plus_regime_plus_trend_state",
            threshold_policy_name="trend_regime",
            probability_calibration_method="platt",
            model_dir=model_dir,
            output_dir=output_dir,
            experiment_config=_experiment_config(),
        )
    )

    assert train_result.manifest["probability_calibration_method"] == "platt"
    assert "probability_calibration_method" in test_result.test_predictions.columns
    observed_methods = set(test_result.test_predictions["probability_calibration_method"].unique())
    assert observed_methods.issubset({"platt", "none"})
    assert "platt" in observed_methods


def test_train_pipeline_persists_soft_threshold_objective_settings(tmp_path) -> None:
    price_frame = _synthetic_price_frame()
    vix_frame = _synthetic_vix_frame()
    input_csv = tmp_path / "spy_daily.csv"
    vix_csv = tmp_path / "vix_daily.csv"
    model_dir = tmp_path / "models"
    save_ohlcv_csv(price_frame, input_csv)
    save_ohlcv_csv(vix_frame, vix_csv)

    train_result = run_training_pipeline(
        TrainPipelineConfig(
            symbol="SPY",
            input_csv=input_csv,
            vix_input_csv=vix_csv,
            profile_name="price_plus_regime_plus_trend_state",
            threshold_policy_name="trend_regime",
            threshold_selection_objective_name="soft_risk_adjusted",
            calibration_return_weight=5.0,
            calibration_exposure_target=0.70,
            calibration_exposure_penalty=1.0,
            calibration_turnover_penalty=0.0025,
            calibration_drawdown_target=0.12,
            calibration_drawdown_penalty=2.0,
            model_dir=model_dir,
            experiment_config=_experiment_config(),
        )
    )

    assert train_result.manifest["threshold_selection_objective_name"] == "soft_risk_adjusted"
    assert train_result.manifest["calibration_return_weight"] == 5.0
    assert train_result.manifest["calibration_exposure_target"] == 0.70


def test_pipeline_can_resolve_named_experiment_spec(tmp_path) -> None:
    price_frame = _synthetic_price_frame()
    vix_frame = _synthetic_vix_frame()
    input_csv = tmp_path / "spy_daily.csv"
    vix_csv = tmp_path / "vix_daily.csv"
    save_ohlcv_csv(price_frame, input_csv)
    save_ohlcv_csv(vix_frame, vix_csv)

    result = run_pipeline(
        TestPipelineConfig(
            symbol="SPY",
            input_csv=input_csv,
            vix_input_csv=vix_csv,
            experiment_name="price_plus_regime_plus_trend_state_plus_regime_thresholding",
            model_dir=tmp_path / "models",
            output_dir=tmp_path / "reports",
            experiment_config=_experiment_config(),
        )
    )

    assert result.manifest["experiment_name"] == "price_plus_regime_plus_trend_state_plus_regime_thresholding"
    assert result.manifest["threshold_policy_name"] == "trend_regime"


def test_train_and_test_pipeline_accepts_sentiment_features_csv(tmp_path) -> None:
    price_frame = _synthetic_price_frame()
    vix_frame = _synthetic_vix_frame()
    sentiment_frame = _synthetic_sentiment_frame()
    input_csv = tmp_path / "spy_daily.csv"
    vix_csv = tmp_path / "vix_daily.csv"
    sentiment_csv = tmp_path / "sentiment_daily.csv"
    model_dir = tmp_path / "models"
    output_dir = tmp_path / "reports"
    save_ohlcv_csv(price_frame, input_csv)
    save_ohlcv_csv(vix_frame, vix_csv)
    sentiment_frame.to_csv(sentiment_csv, index=True)

    train_result = run_training_pipeline(
        TrainPipelineConfig(
            symbol="SPY",
            input_csv=input_csv,
            vix_input_csv=vix_csv,
            sentiment_features_csv=sentiment_csv,
            profile_name="price_plus_regime_plus_sentiment",
            threshold_policy_name="global",
            model_dir=model_dir,
            experiment_config=_experiment_config(),
        )
    )
    test_result = run_test_pipeline(
        TestPipelineConfig(
            symbol="SPY",
            input_csv=input_csv,
            vix_input_csv=vix_csv,
            sentiment_features_csv=sentiment_csv,
            profile_name="price_plus_regime_plus_sentiment",
            threshold_policy_name="global",
            model_dir=model_dir,
            output_dir=output_dir,
            experiment_config=_experiment_config(),
        )
    )

    assert "net_sentiment" in train_result.manifest["feature_columns"]
    assert "abs_emotion" in train_result.manifest["feature_columns"]
    assert test_result.summary["fold_count"] >= 1
