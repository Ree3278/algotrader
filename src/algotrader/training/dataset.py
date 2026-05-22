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
    "price_above_sma_200",
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

TREND_STATE_FEATURE_COLUMNS = [
    "price_to_sma_200",
    "sma_200_slope_20d",
    "sma_50_above_sma_200",
]

REGIME_FEATURE_COLUMNS = [
    "vix_zscore_60d",
]

SENTIMENT_FEATURE_COLUMNS = [
    "net_sentiment",
    "abs_emotion",
    "is_empty_block",
    "headline_count",
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
    vix_frame: pd.DataFrame | None = None,
    sentiment_frame: pd.DataFrame | None = None,
    label_config: TripleBarrierConfig | None = None,
    feature_columns: list[str] | None = None,
) -> TrainingDataset:
    """Build the baseline trainable dataset from raw daily bars."""

    base_frame = price_frame.copy()
    if vix_frame is not None:
        if "close" not in vix_frame.columns:
            raise ValueError("vix_frame must contain a 'close' column")
        aligned_vix = vix_frame["close"].reindex(base_frame.index).ffill()
        base_frame["vix_close"] = aligned_vix
    if sentiment_frame is not None:
        sentiment_required = {"net_sentiment", "abs_emotion", "is_empty_block"}
        missing = sorted(sentiment_required.difference(sentiment_frame.columns))
        if missing:
            raise ValueError(f"sentiment_frame is missing required columns: {missing}")
        aligned_sentiment = sentiment_frame.reindex(base_frame.index)
        for column in aligned_sentiment.columns:
            base_frame[column] = aligned_sentiment[column]

    features = build_price_features(base_frame)
    labels = generate_long_flat_labels(features, config=label_config)
    if feature_columns is not None:
        selected_feature_columns = feature_columns
    else:
        selected_feature_columns = list(DEFAULT_FEATURE_COLUMNS)
        selected_feature_columns.extend(column for column in REGIME_FEATURE_COLUMNS if column in features.columns)
        selected_feature_columns.extend(column for column in SENTIMENT_FEATURE_COLUMNS if column in features.columns)

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
