"""Training and validation utilities."""

from .dataset import TrainingDataset, build_training_dataset
from .walk_forward import PurgedWalkForwardConfig, PurgedWalkForwardSplit, generate_splits

__all__ = [
    "PurgedWalkForwardConfig",
    "PurgedWalkForwardSplit",
    "TrainingDataset",
    "build_training_dataset",
    "generate_splits",
]

try:
    from .experiment import WalkForwardExperimentConfig, WalkForwardExperimentResult, run_walk_forward_experiment
    from .xgboost_model import XGBoostConfig, train_xgboost_classifier
except ImportError:
    pass
else:
    __all__ += [
        "WalkForwardExperimentConfig",
        "WalkForwardExperimentResult",
        "XGBoostConfig",
        "run_walk_forward_experiment",
        "train_xgboost_classifier",
    ]
