from __future__ import annotations

import numpy as np
import pandas as pd

from algotrader.training import build_training_dataset


def test_build_training_dataset_returns_aligned_trainable_rows() -> None:
    index = pd.date_range("2024-01-01", periods=60, freq="D", tz="UTC")
    price_frame = pd.DataFrame(
        {
            "open": np.linspace(100, 159, 60),
            "high": np.linspace(101, 160, 60),
            "low": np.linspace(99, 158, 60),
            "close": np.linspace(100, 159, 60),
            "volume": np.linspace(1_000_000, 1_590_000, 60),
        },
        index=index,
    )

    dataset = build_training_dataset(price_frame)

    assert not dataset.data.empty
    assert set(dataset.feature_columns).issubset(dataset.data.columns)
    assert dataset.y.isin([0, 1]).all()
    assert dataset.data.index.min() > price_frame.index.min()
    assert dataset.data["entry_index"].min() > dataset.data.index.min()


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
