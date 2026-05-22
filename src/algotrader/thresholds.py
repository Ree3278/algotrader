"""Composable threshold-policy factory for walk-forward calibration."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd


RegimeAssigner = Callable[[pd.DataFrame], pd.Series]


@dataclass(frozen=True)
class ThresholdPolicy:
    name: str
    regime_names: tuple[str, ...]
    required_columns: tuple[str, ...]
    assigner: RegimeAssigner

    def assign_regimes(self, frame: pd.DataFrame) -> pd.Series:
        missing = [column for column in self.required_columns if column not in frame.columns]
        if missing:
            raise ValueError(f"Threshold policy '{self.name}' requires columns that are missing: {missing}")
        regimes = self.assigner(frame)
        return pd.Series(regimes, index=frame.index, dtype="object")

    def build_threshold_series(
        self,
        frame: pd.DataFrame,
        threshold_map: dict[str, float],
    ) -> tuple[pd.Series, pd.Series]:
        regimes = self.assign_regimes(frame)
        missing_regimes = sorted(set(regimes.dropna().unique()).difference(threshold_map))
        if missing_regimes:
            raise ValueError(
                f"Threshold map for policy '{self.name}' is missing regimes: {missing_regimes}"
            )
        thresholds = regimes.map(threshold_map).astype(float)
        return thresholds, regimes


def _assign_global_regime(frame: pd.DataFrame) -> pd.Series:
    return pd.Series("all", index=frame.index, dtype="object")


def _assign_trend_regime(frame: pd.DataFrame) -> pd.Series:
    bullish = (frame["price_above_sma_200"].fillna(0.0) >= 0.5) & (
        frame["sma_50_above_sma_200"].fillna(0.0) >= 0.5
    )
    labels = np.where(bullish, "bull_trend", "other")
    return pd.Series(labels, index=frame.index, dtype="object")


THRESHOLD_POLICIES = {
    "global": ThresholdPolicy(
        name="global",
        regime_names=("all",),
        required_columns=(),
        assigner=_assign_global_regime,
    ),
    "trend_regime": ThresholdPolicy(
        name="trend_regime",
        regime_names=("bull_trend", "other"),
        required_columns=("price_above_sma_200", "sma_50_above_sma_200"),
        assigner=_assign_trend_regime,
    ),
}


def build_threshold_policy(name: str) -> ThresholdPolicy:
    if name not in THRESHOLD_POLICIES:
        raise ValueError(f"Unknown threshold policy: {name}")
    return THRESHOLD_POLICIES[name]


def list_threshold_policy_names() -> list[str]:
    return list(THRESHOLD_POLICIES)
