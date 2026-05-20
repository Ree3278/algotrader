"""Purged walk-forward split generation for time-series labels."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PurgedWalkForwardConfig:
    train_size: int
    test_size: int
    step_size: int
    embargo_size: int
    max_label_horizon: int


@dataclass(frozen=True)
class PurgedWalkForwardSplit:
    fold: int
    train_indices: np.ndarray
    test_indices: np.ndarray


def _validate_config(config: PurgedWalkForwardConfig) -> None:
    for field_name in ("train_size", "test_size", "step_size", "embargo_size", "max_label_horizon"):
        if getattr(config, field_name) <= 0:
            raise ValueError(f"{field_name} must be positive")


def generate_splits(
    index: pd.Index,
    config: PurgedWalkForwardConfig,
) -> Iterator[PurgedWalkForwardSplit]:
    """Yield purged walk-forward splits over positional indices.

    Each split uses a fixed-size training window ending before the embargo
    region. The purge step then removes any remaining training samples whose
    label horizon could overlap the first test sample.
    """

    _validate_config(config)
    if len(index) == 0:
        return

    total_needed = config.train_size + config.embargo_size + config.test_size
    if len(index) < total_needed:
        return

    fold = 1
    test_start = config.train_size + config.embargo_size
    while test_start + config.test_size <= len(index):
        test_end = test_start + config.test_size
        pre_purge_train_end = test_start - config.embargo_size
        train_start = pre_purge_train_end - config.train_size

        train_indices = np.arange(train_start, pre_purge_train_end)
        overlap_cutoff = test_start - config.max_label_horizon
        train_indices = train_indices[train_indices < overlap_cutoff]

        if len(train_indices) > 0:
            yield PurgedWalkForwardSplit(
                fold=fold,
                train_indices=train_indices,
                test_indices=np.arange(test_start, test_end),
            )

        fold += 1
        test_start += config.step_size
