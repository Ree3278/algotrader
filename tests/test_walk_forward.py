from __future__ import annotations

import pandas as pd

from algotrader.training import PurgedWalkForwardConfig, generate_splits


def test_generates_fixed_window_splits() -> None:
    index = pd.RangeIndex(40)
    config = PurgedWalkForwardConfig(
        train_size=12,
        test_size=6,
        step_size=6,
        embargo_size=2,
        max_label_horizon=2,
    )

    splits = list(generate_splits(index, config))

    assert len(splits) == 4
    assert splits[0].test_indices.tolist() == [14, 15, 16, 17, 18, 19]
    assert splits[1].test_indices.tolist() == [20, 21, 22, 23, 24, 25]


def test_purge_and_embargo_remove_overlap_with_test_horizon() -> None:
    index = pd.RangeIndex(40)
    config = PurgedWalkForwardConfig(
        train_size=12,
        test_size=6,
        step_size=6,
        embargo_size=2,
        max_label_horizon=3,
    )

    split = list(generate_splits(index, config))[0]
    test_start = split.test_indices[0]

    assert split.train_indices.max() < test_start - config.max_label_horizon
    assert split.train_indices.max() < test_start - config.embargo_size

    for train_idx in split.train_indices:
        event_end = train_idx + config.max_label_horizon
        assert event_end < test_start
