from __future__ import annotations

import numpy as np
import pandas as pd

from algotrader.features import build_price_features


def test_build_price_features_adds_expected_columns_and_respects_warmup() -> None:
    index = pd.date_range("2024-01-01", periods=260, freq="D", tz="UTC")
    frame = pd.DataFrame(
        {
            "open": np.linspace(100, 359, 260),
            "high": np.linspace(101, 360, 260),
            "low": np.linspace(99, 358, 260),
            "close": np.linspace(100, 359, 260),
            "volume": np.linspace(1_000_000, 3_590_000, 260),
            "vix_close": 20 + 5 * np.sin(np.arange(260) / 10),
        },
        index=index,
    )

    features = build_price_features(frame)

    expected_columns = {
        "return_1d",
        "return_5d",
        "volatility_20d",
        "price_above_sma_200",
        "price_to_sma_200",
        "sma_200_slope_20d",
        "sma_50_above_sma_200",
        "vix_zscore_60d",
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
    assert pd.isna(features.iloc[198]["price_above_sma_200"])
    assert features.iloc[199]["price_above_sma_200"] == 1.0
    assert pd.isna(features.iloc[198]["price_to_sma_200"])
    assert features.iloc[199]["price_to_sma_200"] > 0.0
    assert pd.isna(features.iloc[218]["sma_200_slope_20d"])
    assert features.iloc[219]["sma_200_slope_20d"] > 0.0
    assert features.iloc[199]["sma_50_above_sma_200"] == 1.0
    assert pd.isna(features.iloc[58]["vix_zscore_60d"])
    assert not pd.isna(features.iloc[59]["vix_zscore_60d"])
    assert not pd.isna(features.iloc[-1]["MACD_line"])
