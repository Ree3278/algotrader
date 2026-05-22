from __future__ import annotations

import pandas as pd

from algotrader.thresholds import build_threshold_policy, list_threshold_policy_names


def test_trend_regime_policy_assigns_bull_and_other_labels() -> None:
    frame = pd.DataFrame(
        {
            "price_above_sma_200": [1.0, 0.0, 1.0],
            "sma_50_above_sma_200": [1.0, 1.0, 0.0],
        },
        index=pd.date_range("2024-01-01", periods=3, freq="D", tz="UTC"),
    )

    policy = build_threshold_policy("trend_regime")
    thresholds, regimes = policy.build_threshold_series(
        frame,
        {"bull_trend": 0.6, "other": 0.5},
    )

    assert list(regimes) == ["bull_trend", "other", "other"]
    assert list(thresholds) == [0.6, 0.5, 0.5]


def test_list_threshold_policy_names_includes_global_and_trend_regime() -> None:
    names = list_threshold_policy_names()

    assert "global" in names
    assert "trend_regime" in names
