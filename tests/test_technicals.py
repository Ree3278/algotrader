from __future__ import annotations

import numpy as np
import pandas as pd

from algotrader.features import build_price_features


def test_build_price_features_adds_expected_columns_and_respects_warmup() -> None:
    index = pd.date_range("2024-01-01", periods=40, freq="D", tz="UTC")
    frame = pd.DataFrame(
        {
            "open": np.linspace(100, 139, 40),
            "high": np.linspace(101, 140, 40),
            "low": np.linspace(99, 138, 40),
            "close": np.linspace(100, 139, 40),
            "volume": np.linspace(1_000_000, 1_390_000, 40),
        },
        index=index,
    )

    features = build_price_features(frame)

    expected_columns = {
        "return_1d",
        "return_5d",
        "volatility_20d",
        "ATR_14",
        "RSI_14",
        "MACD_line",
        "MACD_signal",
        "MACD_hist",
        "BB_upper",
        "BB_lower",
        "BB_pct_b",
        "volume_zscore_20d",
    }

    assert expected_columns.issubset(features.columns)
    assert pd.isna(features.iloc[0]["return_1d"])
    assert pd.isna(features.iloc[12]["ATR_14"])
    assert not pd.isna(features.iloc[13]["ATR_14"])
    assert not pd.isna(features.iloc[-1]["MACD_line"])
