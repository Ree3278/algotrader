"""Named experiment-spec registry for reusable research definitions."""

from __future__ import annotations

from algotrader.settings import DEFAULT_SETTINGS
from algotrader.specs import ExperimentSpec, build_experiment_spec


EXPERIMENT_SPECS: dict[str, ExperimentSpec] = {
    "price_only": build_experiment_spec(
        settings=DEFAULT_SETTINGS,
        name="price_only",
        profile_name="price_only",
        threshold_policy_name="global",
    ),
    "price_plus_regime": build_experiment_spec(
        settings=DEFAULT_SETTINGS,
        name="price_plus_regime",
        profile_name="price_plus_regime",
        threshold_policy_name="global",
    ),
    "price_plus_regime_plus_trend_state": build_experiment_spec(
        settings=DEFAULT_SETTINGS,
        name="price_plus_regime_plus_trend_state",
        profile_name="price_plus_regime_plus_trend_state",
        threshold_policy_name="trend_regime",
    ),
    "price_plus_regime_plus_sentiment": build_experiment_spec(
        settings=DEFAULT_SETTINGS,
        name="price_plus_regime_plus_sentiment",
        profile_name="price_plus_regime_plus_sentiment",
        threshold_policy_name="global",
    ),
    "price_plus_regime_plus_trend_state_plus_regime_thresholding": build_experiment_spec(
        settings=DEFAULT_SETTINGS,
        name="price_plus_regime_plus_trend_state_plus_regime_thresholding",
        profile_name="price_plus_regime_plus_trend_state",
        threshold_policy_name="trend_regime",
    ),
    "price_plus_regime_plus_trend_state_plus_regime_thresholding_plus_soft_objective": build_experiment_spec(
        settings=DEFAULT_SETTINGS,
        name="price_plus_regime_plus_trend_state_plus_regime_thresholding_plus_soft_objective",
        profile_name="price_plus_regime_plus_trend_state",
        threshold_policy_name="trend_regime",
        threshold_selection_objective_name="soft_risk_adjusted",
        calibration_return_weight=1.0,
        calibration_exposure_target=0.0,
        calibration_exposure_penalty=100.0,
        calibration_turnover_penalty=0.025,
        calibration_drawdown_target=0.0,
        calibration_drawdown_penalty=4.0,
    ),
}


def build_registered_experiment(name: str) -> ExperimentSpec:
    if name not in EXPERIMENT_SPECS:
        raise ValueError(f"Unknown experiment: {name}")
    return EXPERIMENT_SPECS[name]


def list_experiment_names() -> list[str]:
    return list(EXPERIMENT_SPECS)
