from __future__ import annotations

import numpy as np
import pandas as pd

from algotrader.training import build_training_dataset


def test_build_training_dataset_returns_aligned_trainable_rows() -> None:
    index = pd.date_range("2024-01-01", periods=260, freq="D", tz="UTC")
    price_frame = pd.DataFrame(
        {
            "open": np.linspace(100, 359, 260),
            "high": np.linspace(101, 360, 260),
            "low": np.linspace(99, 358, 260),
            "close": np.linspace(100, 359, 260),
            "volume": np.linspace(1_000_000, 3_590_000, 260),
        },
        index=index,
    )

    dataset = build_training_dataset(price_frame)

    assert not dataset.data.empty
    assert set(dataset.feature_columns).issubset(dataset.data.columns)
    assert dataset.y.isin([0, 1]).all()
    assert dataset.data.index.min() > price_frame.index.min()
    assert dataset.data["entry_index"].min() > dataset.data.index.min()
    assert "price_above_sma_200" in dataset.feature_columns


def test_build_training_dataset_drops_unlabeled_tail_rows() -> None:
    index = pd.date_range("2024-01-01", periods=30, freq="D", tz="UTC")
    price_frame = pd.DataFrame(
        {
            "open": np.linspace(100, 129, 30),
            "high": np.linspace(101, 130, 30),
            "low": np.linspace(99, 128, 30),
            "close": np.linspace(100, 129, 30),
            "volume": np.linspace(1_000_000, 1_290_000, 30),
        },
        index=index,
    )

    dataset = build_training_dataset(price_frame)

    assert dataset.data.empty


def test_build_training_dataset_includes_vix_feature_when_vix_frame_is_provided() -> None:
    index = pd.date_range("2024-01-01", periods=320, freq="D", tz="UTC")
    price_frame = pd.DataFrame(
        {
            "open": np.linspace(100, 419, 320),
            "high": np.linspace(101, 420, 320),
            "low": np.linspace(99, 418, 320),
            "close": np.linspace(100, 419, 320),
            "volume": np.linspace(1_000_000, 4_190_000, 320),
        },
        index=index,
    )
    vix_frame = pd.DataFrame({"close": 20 + 5 * np.sin(np.arange(320) / 10)}, index=index)

    dataset = build_training_dataset(price_frame, vix_frame=vix_frame)

    assert "vix_zscore_60d" in dataset.feature_columns
    assert "vix_zscore_60d" in dataset.data.columns
