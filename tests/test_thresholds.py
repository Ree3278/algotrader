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


def test_trend_regime_constrained_policy_enforces_monotonic_thresholds() -> None:
    policy = build_threshold_policy("trend_regime_constrained")

    assert policy.allows_threshold_map({"bull_trend": 0.55, "other": 0.65}) is True
    assert policy.allows_threshold_map({"bull_trend": 0.65, "other": 0.55}) is False


def test_trend_vix_regime_policy_assigns_four_regime_buckets() -> None:
    frame = pd.DataFrame(
        {
            "price_above_sma_200": [1.0, 1.0, 0.0, 0.0],
            "sma_50_above_sma_200": [1.0, 1.0, 0.0, 0.0],
            "vix_zscore_60d": [-0.2, 0.3, -0.1, 0.5],
        },
        index=pd.date_range("2024-01-01", periods=4, freq="D", tz="UTC"),
    )

    policy = build_threshold_policy("trend_vix_regime")
    thresholds, regimes = policy.build_threshold_series(
        frame,
        {
            "bull_calm": 0.55,
            "bull_stressed": 0.65,
            "other_calm": 0.5,
            "other_stressed": 0.6,
        },
    )

    assert list(regimes) == ["bull_calm", "bull_stressed", "other_calm", "other_stressed"]
    assert list(thresholds) == [0.55, 0.65, 0.5, 0.6]


def test_list_threshold_policy_names_includes_global_and_regime_variants() -> None:
    names = list_threshold_policy_names()

    assert "global" in names
    assert "trend_regime" in names
    assert "trend_regime_constrained" in names
    assert "trend_vix_regime" in names
