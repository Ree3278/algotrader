from __future__ import annotations

from algotrader.experiment_registry import build_registered_experiment, list_experiment_names


def test_registered_experiments_include_core_baselines() -> None:
    names = list_experiment_names()

    assert "price_only" in names
    assert "price_plus_regime_plus_trend_state_plus_regime_thresholding" in names


def test_build_registered_experiment_returns_expected_spec() -> None:
    spec = build_registered_experiment("price_plus_regime_plus_trend_state_plus_regime_thresholding")

    assert spec.profile_name == "price_plus_regime_plus_trend_state"
    assert spec.profile.block_names == ["price_only", "regime", "trend_state"]
    assert spec.decision_policy.threshold_policy_name == "trend_regime"
