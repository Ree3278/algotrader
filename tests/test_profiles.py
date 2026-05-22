from __future__ import annotations

from algotrader.profiles import build_model_profile, list_profile_names


def test_build_model_profile_from_preset_exposes_blocks_and_requirements() -> None:
    profile = build_model_profile(name="price_plus_regime_plus_trend_state")

    assert profile.block_names == ["price_only", "regime", "trend_state"]
    assert profile.requires_vix is True
    assert profile.requires_sentiment is False
    assert "vix_zscore_60d" in profile.feature_columns
    assert "sma_200_slope_20d" in profile.feature_columns


def test_list_profile_names_includes_expected_presets() -> None:
    profile_names = list_profile_names()

    assert "price_only" in profile_names
    assert "price_plus_regime_plus_trend_state" in profile_names
    assert "price_plus_regime_plus_trend_state_plus_atr_percentile" in profile_names
