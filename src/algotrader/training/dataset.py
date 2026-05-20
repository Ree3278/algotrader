"""Dataset assembly for the price-only baseline."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from algotrader.features import build_price_features
from algotrader.labels import TripleBarrierConfig, generate_long_flat_labels

DEFAULT_FEATURE_COLUMNS = [
    "return_1d",
    "return_5d",
    "volatility_20d",
    "ATR_14",
    "RSI_14",
    "MACD_line",
    "MACD_signal",
    "MACD_hist",
    "BB_upper",
    "BB_lower",
    "BB_pct_b",
    "volume_zscore_20d",
]


@dataclass(frozen=True)
class TrainingDataset:
    data: pd.DataFrame
    feature_columns: list[str]
    target_column: str = "label"

    @property
    def X(self) -> pd.DataFrame:
        return self.data[self.feature_columns]

    @property
    def y(self) -> pd.Series:
        return self.data[self.target_column]


def build_training_dataset(
    price_frame: pd.DataFrame,
    *,
    label_config: TripleBarrierConfig | None = None,
    feature_columns: list[str] | None = None,
) -> TrainingDataset:
    """Build the baseline trainable dataset from raw daily bars."""

    features = build_price_features(price_frame)
    labels = generate_long_flat_labels(features, config=label_config)
    selected_feature_columns = feature_columns or DEFAULT_FEATURE_COLUMNS

    dataset = pd.concat(
        [
            features[selected_feature_columns],
            labels[
                [
                    "label",
                    "entry_index",
                    "exit_index",
                    "event_end_index",
                    "hit_reason",
                    "entry_price",
                    "exit_price",
                    "realized_return",
                ]
            ],
        ],
        axis=1,
    )

    dataset = dataset.dropna(subset=selected_feature_columns + ["label"])
    dataset["label"] = dataset["label"].astype(int)
    return TrainingDataset(data=dataset, feature_columns=selected_feature_columns)
