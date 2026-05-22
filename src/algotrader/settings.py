"""Centralized project settings and config builders."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from algotrader.backtest import BacktestConfig
from algotrader.labels import TripleBarrierConfig
from algotrader.training.experiment import WalkForwardExperimentConfig
from algotrader.training.walk_forward import PurgedWalkForwardConfig
from algotrader.training.xgboost_model import XGBoostConfig


@dataclass(frozen=True)
class DataSettings:
    symbol: str = "SPY"
    vix_symbol: str = "^VIX"


@dataclass(frozen=True)
class PathSettings:
    raw_data_dir: Path = Path("data/raw/ohlcv")
    normalized_data_dir: Path = Path("data/interim")
    model_dir: Path = Path("models/latest")
    report_dir: Path = Path("reports/latest")
    news_dir: Path = Path("data/raw/news")

    def default_price_csv(self, symbol: str) -> Path:
        return self.normalized_data_dir / f"{symbol.lower()}_daily.csv"

    @property
    def default_vix_csv(self) -> Path:
        return self.normalized_data_dir / "vix_daily.csv"

    @property
    def default_sentiment_csv(self) -> Path:
        return self.normalized_data_dir / "sentiment_daily.csv"


@dataclass(frozen=True)
class LabelSettings:
    profit_target_atr: float = 1.25
    stop_loss_atr: float = 1.25
    max_holding_bars: int = 10
    timeout_return_threshold: float = 0.0
    intrabar_tie_break: str = "stop"

    def build(self) -> TripleBarrierConfig:
        return TripleBarrierConfig(
            profit_target_atr=self.profit_target_atr,
            stop_loss_atr=self.stop_loss_atr,
            max_holding_bars=self.max_holding_bars,
            timeout_return_threshold=self.timeout_return_threshold,
            intrabar_tie_break=self.intrabar_tie_break,
        )


@dataclass(frozen=True)
class SplitSettings:
    train_size: int = 504
    test_size: int = 252
    step_size: int = 252
    embargo_size: int = 10
    max_label_horizon: int = 10

    def build(self) -> PurgedWalkForwardConfig:
        return PurgedWalkForwardConfig(
            train_size=self.train_size,
            test_size=self.test_size,
            step_size=self.step_size,
            embargo_size=self.embargo_size,
            max_label_horizon=self.max_label_horizon,
        )


@dataclass(frozen=True)
class ModelSettings:
    n_estimators: int = 200
    max_depth: int = 4
    learning_rate: float = 0.05
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    min_child_weight: float = 1.0
    random_state: int = 42
    use_balanced_sample_weights: bool = True
    backend: str = "auto"

    def build(self) -> XGBoostConfig:
        return XGBoostConfig(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            min_child_weight=self.min_child_weight,
            random_state=self.random_state,
            use_balanced_sample_weights=self.use_balanced_sample_weights,
            backend=self.backend,
        )


@dataclass(frozen=True)
class BacktestSettings:
    probability_threshold: float = 0.55
    commission_bps: float = 1.0
    slippage_bps: float = 2.0

    def build(self) -> BacktestConfig:
        return BacktestConfig(
            probability_threshold=self.probability_threshold,
            commission_bps=self.commission_bps,
            slippage_bps=self.slippage_bps,
        )


@dataclass(frozen=True)
class ExperimentSettings:
    threshold_grid: tuple[float, ...] = (0.50, 0.55, 0.6, 0.65)
    calibration_fraction: float = 0.2
    min_calibration_size: int = 20
    min_training_size: int = 30
    probability_calibration_method: str = "none"


@dataclass(frozen=True)
class ProfileSettings:
    default_profile_name: str = "price_plus_regime_plus_trend_state"


@dataclass(frozen=True)
class ThresholdPolicySettings:
    default_policy_name: str = "global"


@dataclass(frozen=True)
class ProjectSettings:
    data: DataSettings = DataSettings()
    paths: PathSettings = PathSettings()
    labels: LabelSettings = LabelSettings()
    split: SplitSettings = SplitSettings()
    model: ModelSettings = ModelSettings()
    backtest: BacktestSettings = BacktestSettings()
    experiment: ExperimentSettings = ExperimentSettings()
    profiles: ProfileSettings = ProfileSettings()
    thresholds: ThresholdPolicySettings = ThresholdPolicySettings()

    def build_label_config(self) -> TripleBarrierConfig:
        return self.labels.build()

    def build_experiment_config(self) -> WalkForwardExperimentConfig:
        return WalkForwardExperimentConfig(
            split_config=self.split.build(),
            model_config=self.model.build(),
            backtest_config=self.backtest.build(),
            threshold_grid=self.experiment.threshold_grid,
            calibration_fraction=self.experiment.calibration_fraction,
            min_calibration_size=self.experiment.min_calibration_size,
            min_training_size=self.experiment.min_training_size,
            threshold_policy_name=self.thresholds.default_policy_name,
            probability_calibration_method=self.experiment.probability_calibration_method,
        )


DEFAULT_SETTINGS = ProjectSettings()
