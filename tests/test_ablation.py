from __future__ import annotations

import json

import numpy as np
import pandas as pd

from algotrader.ablation import run_feature_ablation
from algotrader.ingestion.storage import save_ohlcv_csv
from algotrader.pipeline import TestPipelineConfig
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


def _synthetic_vix_frame(periods: int = 360) -> pd.DataFrame:
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


def _synthetic_sentiment_frame(periods: int = 360) -> pd.DataFrame:
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
        threshold_grid=(0.45, 0.55),
        min_training_size=50,
        min_calibration_size=15,
    )


def test_run_feature_ablation_writes_outputs(tmp_path) -> None:
    price_frame = _synthetic_price_frame()
    vix_frame = _synthetic_vix_frame()
    sentiment_frame = _synthetic_sentiment_frame()
    input_csv = tmp_path / "spy_daily.csv"
    vix_csv = tmp_path / "vix_daily.csv"
    sentiment_csv = tmp_path / "sentiment_daily.csv"
    save_ohlcv_csv(price_frame, input_csv)
    save_ohlcv_csv(vix_frame, vix_csv)
    sentiment_frame.to_csv(sentiment_csv, index=True)

    results, paths = run_feature_ablation(
        TestPipelineConfig(
            symbol="SPY",
            input_csv=input_csv,
            vix_input_csv=vix_csv,
            sentiment_features_csv=sentiment_csv,
            experiment_config=_experiment_config(),
        ),
        output_dir=tmp_path / "ablation",
    )

    assert len(results) == 4
    assert list(results["variant"].sort_values()) == [
        "price_only",
        "price_plus_regime",
        "price_plus_regime_plus_sentiment",
        "price_plus_regime_plus_trend_state",
    ]
    feature_counts = dict(zip(results["variant"], results["feature_count"]))
    assert feature_counts["price_only"] == 13
    assert feature_counts["price_plus_regime"] == 14
    assert feature_counts["price_plus_regime_plus_trend_state"] == 17
    assert feature_counts["price_plus_regime_plus_sentiment"] == 18
    assert paths["csv"].exists()
    assert paths["json"].exists()
    assert paths["summary"].exists()

    summary = json.loads(paths["summary"].read_text(encoding="utf-8"))
    assert summary["best_by_mean_sharpe"] is not None
    assert len(summary["variants"]) == 4
