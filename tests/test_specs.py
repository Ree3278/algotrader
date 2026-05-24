from __future__ import annotations

from algotrader.settings import DEFAULT_SETTINGS
from algotrader.specs import build_experiment_spec, build_experiment_spec_from_dict


def test_build_experiment_spec_from_profile_preset_is_round_trippable() -> None:
    spec = build_experiment_spec(
        settings=DEFAULT_SETTINGS,
        name="baseline_spec",
        profile_name="price_plus_regime_plus_trend_state",
        threshold_policy_name="trend_regime",
    )

    payload = spec.to_dict()
    restored = build_experiment_spec_from_dict(payload)

    assert restored.name == "baseline_spec"
    assert restored.profile.block_names == ["price_only", "regime", "trend_state"]
    assert restored.decision_policy.threshold_policy_name == "trend_regime"
    assert restored.feature_columns == spec.feature_columns


def test_build_experiment_spec_accepts_explicit_feature_blocks() -> None:
    spec = build_experiment_spec(
        settings=DEFAULT_SETTINGS,
        name="custom_spec",
        feature_block_names=("price_only", "regime", "sentiment"),
        threshold_policy_name="global",
    )

    assert spec.profile.block_names == ["price_only", "regime", "sentiment"]
    assert spec.requires_vix is True
    assert spec.requires_sentiment is True
