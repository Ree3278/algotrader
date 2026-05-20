"""XGBoost baseline model helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.utils.class_weight import compute_sample_weight


@dataclass(frozen=True)
class XGBoostConfig:
    n_estimators: int = 200
    max_depth: int = 4
    learning_rate: float = 0.05
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    min_child_weight: float = 1.0
    random_state: int = 42
    use_balanced_sample_weights: bool = True
    backend: str = "auto"


def _compute_sample_weights(y: pd.Series, enabled: bool) -> pd.Series | None:
    if not enabled or y.nunique() < 2:
        return None
    return pd.Series(compute_sample_weight(class_weight="balanced", y=y), index=y.index)


def _build_estimator(config: XGBoostConfig) -> tuple[Any, str]:
    if config.backend not in {"auto", "xgboost", "hist_gradient_boosting"}:
        raise ValueError("backend must be 'auto', 'xgboost', or 'hist_gradient_boosting'")

    if config.backend in {"auto", "xgboost"}:
        try:
            from xgboost import XGBClassifier
        except Exception as exc:
            if config.backend == "xgboost":
                raise RuntimeError(
                    "XGBoost backend requested but unavailable. On macOS this usually means `libomp` is missing."
                ) from exc
        else:
            estimator = XGBClassifier(
                objective="binary:logistic",
                eval_metric="logloss",
                n_estimators=config.n_estimators,
                max_depth=config.max_depth,
                learning_rate=config.learning_rate,
                subsample=config.subsample,
                colsample_bytree=config.colsample_bytree,
                min_child_weight=config.min_child_weight,
                random_state=config.random_state,
                n_jobs=1,
            )
            return estimator, "xgboost"

    estimator = HistGradientBoostingClassifier(
        learning_rate=config.learning_rate,
        max_depth=config.max_depth,
        max_iter=config.n_estimators,
        min_samples_leaf=max(int(config.min_child_weight), 1),
        random_state=config.random_state,
    )
    return estimator, "hist_gradient_boosting"


def train_xgboost_classifier(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    *,
    config: XGBoostConfig | None = None,
) -> Any:
    """Fit the baseline binary tree model.

    The preferred backend is XGBoost. When that native dependency is not
    available, `backend='auto'` falls back to sklearn's histogram gradient
    boosting so the research pipeline stays runnable.
    """

    config = config or XGBoostConfig()
    sample_weight = _compute_sample_weights(y_train, enabled=config.use_balanced_sample_weights)
    model, backend_name = _build_estimator(config)
    model.fit(X_train, y_train, sample_weight=sample_weight)
    setattr(model, "_algotrader_backend", backend_name)
    return model
