"""Reusable experiment-spec composition for research workflows."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any

from algotrader.backtest import BacktestConfig
from algotrader.labels import TripleBarrierConfig
from algotrader.profiles import ModelProfile, build_model_profile
from algotrader.training.experiment import WalkForwardExperimentConfig
from algotrader.training.walk_forward import PurgedWalkForwardConfig
from algotrader.training.xgboost_model import XGBoostConfig

if TYPE_CHECKING:
    from algotrader.settings import ProjectSettings


@dataclass(frozen=True)
class LabelerSpec:
    name: str
    config: TripleBarrierConfig


@dataclass(frozen=True)
class ModelSpec:
    name: str
    config: XGBoostConfig


@dataclass(frozen=True)
class DecisionPolicySpec:
    threshold_policy_name: str
    probability_calibration_method: str = "none"
    max_calibration_exposure: float | None = None
    threshold_selection_objective_name: str = "legacy"
    calibration_return_weight: float = 0.0
    calibration_exposure_target: float | None = None
    calibration_exposure_penalty: float = 0.0
    calibration_turnover_penalty: float = 0.0
    calibration_drawdown_target: float | None = None
    calibration_drawdown_penalty: float = 0.0


@dataclass(frozen=True)
class EvaluationSpec:
    name: str
    split_config: PurgedWalkForwardConfig
    backtest_config: BacktestConfig
    threshold_grid: tuple[float, ...]
    calibration_fraction: float
    min_calibration_size: int
    min_training_size: int
    holdout_size: int | None = None


@dataclass(frozen=True)
class ExperimentSpec:
    name: str
    profile: ModelProfile
    labeler: LabelerSpec
    model: ModelSpec
    decision_policy: DecisionPolicySpec
    evaluation: EvaluationSpec

    @property
    def feature_columns(self) -> list[str]:
        return self.profile.feature_columns

    @property
    def requires_vix(self) -> bool:
        return self.profile.requires_vix

    @property
    def requires_sentiment(self) -> bool:
        return self.profile.requires_sentiment

    @property
    def profile_name(self) -> str:
        return self.profile.name

    @property
    def profile_blocks(self) -> list[str]:
        return self.profile.block_names

    def build_walk_forward_config(self) -> WalkForwardExperimentConfig:
        return WalkForwardExperimentConfig(
            split_config=self.evaluation.split_config,
            model_config=self.model.config,
            backtest_config=self.evaluation.backtest_config,
            threshold_grid=self.evaluation.threshold_grid,
            calibration_fraction=self.evaluation.calibration_fraction,
            min_calibration_size=self.evaluation.min_calibration_size,
            min_training_size=self.evaluation.min_training_size,
            threshold_policy_name=self.decision_policy.threshold_policy_name,
            probability_calibration_method=self.decision_policy.probability_calibration_method,
            max_calibration_exposure=self.decision_policy.max_calibration_exposure,
            threshold_selection_objective_name=self.decision_policy.threshold_selection_objective_name,
            calibration_return_weight=self.decision_policy.calibration_return_weight,
            calibration_exposure_target=self.decision_policy.calibration_exposure_target,
            calibration_exposure_penalty=self.decision_policy.calibration_exposure_penalty,
            calibration_turnover_penalty=self.decision_policy.calibration_turnover_penalty,
            calibration_drawdown_target=self.decision_policy.calibration_drawdown_target,
            calibration_drawdown_penalty=self.decision_policy.calibration_drawdown_penalty,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "profile": {
                "name": self.profile.name,
                "block_names": self.profile.block_names,
                "requires_vix": self.profile.requires_vix,
                "requires_sentiment": self.profile.requires_sentiment,
                "feature_columns": self.profile.feature_columns,
            },
            "labeler": {
                "name": self.labeler.name,
                "config": asdict(self.labeler.config),
            },
            "model": {
                "name": self.model.name,
                "config": asdict(self.model.config),
            },
            "decision_policy": asdict(self.decision_policy),
            "evaluation": {
                "name": self.evaluation.name,
                "split_config": asdict(self.evaluation.split_config),
                "backtest_config": asdict(self.evaluation.backtest_config),
                "threshold_grid": list(self.evaluation.threshold_grid),
                "calibration_fraction": self.evaluation.calibration_fraction,
                "min_calibration_size": self.evaluation.min_calibration_size,
                "min_training_size": self.evaluation.min_training_size,
                "holdout_size": self.evaluation.holdout_size,
            },
        }


def build_experiment_spec(
    *,
    settings: ProjectSettings,
    name: str | None = None,
    profile_name: str | None = None,
    feature_block_names: list[str] | tuple[str, ...] | None = None,
    threshold_policy_name: str | None = None,
    probability_calibration_method: str | None = None,
    max_calibration_exposure: float | None = None,
    threshold_selection_objective_name: str | None = None,
    calibration_return_weight: float | None = None,
    calibration_exposure_target: float | None = None,
    calibration_exposure_penalty: float | None = None,
    calibration_turnover_penalty: float | None = None,
    calibration_drawdown_target: float | None = None,
    calibration_drawdown_penalty: float | None = None,
) -> ExperimentSpec:
    profile = (
        build_model_profile(name=profile_name)
        if profile_name is not None
        else build_model_profile(block_names=tuple(feature_block_names or ()))
    )
    experiment_config = settings.build_experiment_config()
    decision_policy = DecisionPolicySpec(
        threshold_policy_name=threshold_policy_name or settings.thresholds.default_policy_name,
        probability_calibration_method=(
            probability_calibration_method or settings.experiment.probability_calibration_method
        ),
        max_calibration_exposure=max_calibration_exposure,
        threshold_selection_objective_name=(
            threshold_selection_objective_name or settings.experiment.threshold_selection_objective_name
        ),
        calibration_return_weight=(
            settings.experiment.calibration_return_weight
            if calibration_return_weight is None
            else calibration_return_weight
        ),
        calibration_exposure_target=(
            settings.experiment.calibration_exposure_target
            if calibration_exposure_target is None
            else calibration_exposure_target
        ),
        calibration_exposure_penalty=(
            settings.experiment.calibration_exposure_penalty
            if calibration_exposure_penalty is None
            else calibration_exposure_penalty
        ),
        calibration_turnover_penalty=(
            settings.experiment.calibration_turnover_penalty
            if calibration_turnover_penalty is None
            else calibration_turnover_penalty
        ),
        calibration_drawdown_target=(
            settings.experiment.calibration_drawdown_target
            if calibration_drawdown_target is None
            else calibration_drawdown_target
        ),
        calibration_drawdown_penalty=(
            settings.experiment.calibration_drawdown_penalty
            if calibration_drawdown_penalty is None
            else calibration_drawdown_penalty
        ),
    )
    evaluation = EvaluationSpec(
        name="walk_forward",
        split_config=experiment_config.split_config,
        backtest_config=experiment_config.backtest_config,
        threshold_grid=experiment_config.threshold_grid,
        calibration_fraction=experiment_config.calibration_fraction,
        min_calibration_size=experiment_config.min_calibration_size,
        min_training_size=experiment_config.min_training_size,
        holdout_size=settings.holdout.size,
    )
    return ExperimentSpec(
        name=name or profile.name,
        profile=profile,
        labeler=LabelerSpec(name="triple_barrier", config=settings.build_label_config()),
        model=ModelSpec(name="xgboost_classifier", config=experiment_config.model_config),
        decision_policy=decision_policy,
        evaluation=evaluation,
    )


def build_experiment_spec_from_dict(payload: dict[str, Any]) -> ExperimentSpec:
    profile_payload = payload["profile"]
    return ExperimentSpec(
        name=payload["name"],
        profile=build_model_profile(block_names=tuple(profile_payload["block_names"])),
        labeler=LabelerSpec(
            name=payload["labeler"]["name"],
            config=TripleBarrierConfig(**payload["labeler"]["config"]),
        ),
        model=ModelSpec(
            name=payload["model"]["name"],
            config=XGBoostConfig(**payload["model"]["config"]),
        ),
        decision_policy=DecisionPolicySpec(**payload["decision_policy"]),
        evaluation=EvaluationSpec(
            name=payload["evaluation"]["name"],
            split_config=PurgedWalkForwardConfig(**payload["evaluation"]["split_config"]),
            backtest_config=BacktestConfig(**payload["evaluation"]["backtest_config"]),
            threshold_grid=tuple(payload["evaluation"]["threshold_grid"]),
            calibration_fraction=payload["evaluation"]["calibration_fraction"],
            min_calibration_size=payload["evaluation"]["min_calibration_size"],
            min_training_size=payload["evaluation"]["min_training_size"],
            holdout_size=payload["evaluation"].get("holdout_size"),
        ),
    )
