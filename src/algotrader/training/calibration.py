"""Probability calibration helpers for walk-forward training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class ProbabilityCalibrator:
    method: str
    estimator: Any | None = None

    def transform(self, probabilities: pd.Series) -> pd.Series:
        if self.method == "none" or self.estimator is None:
            return probabilities.astype(float)
        calibrated = self.estimator.predict_proba(probabilities.to_numpy().reshape(-1, 1))[:, 1]
        return pd.Series(calibrated, index=probabilities.index, dtype=float)


def fit_probability_calibrator(
    probabilities: pd.Series,
    labels: pd.Series,
    *,
    method: str,
) -> ProbabilityCalibrator:
    if method not in {"none", "platt"}:
        raise ValueError("Probability calibration method must be 'none' or 'platt'")
    if method == "none":
        return ProbabilityCalibrator(method="none")

    aligned = pd.concat([probabilities.rename("probability"), labels.rename("label")], axis=1).dropna()
    if aligned.empty or aligned["label"].nunique() < 2:
        return ProbabilityCalibrator(method="none")

    from sklearn.linear_model import LogisticRegression

    estimator = LogisticRegression(random_state=42, solver="lbfgs")
    estimator.fit(aligned["probability"].to_numpy().reshape(-1, 1), aligned["label"].to_numpy())
    return ProbabilityCalibrator(method="platt", estimator=estimator)


def apply_probability_calibration(
    calibrator: ProbabilityCalibrator,
    probabilities: pd.Series,
) -> pd.Series:
    return calibrator.transform(probabilities)
