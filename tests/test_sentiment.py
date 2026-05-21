from __future__ import annotations

import numpy as np
import pandas as pd

from algotrader.sentiment.aggregate import aggregate_daily_sentiment
from algotrader.sentiment.dedup import deduplicate_news


def test_deduplicate_news_drops_near_duplicate_headlines_within_day() -> None:
    news = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                [
                    "2024-01-02T14:00:00Z",
                    "2024-01-02T14:05:00Z",
                    "2024-01-03T14:00:00Z",
                ],
                utc=True,
            ),
            "headline": [
                "Apple jumps after earnings beat estimates",
                "Apple jumps after earnings beat estimate",
                "Fed signals rates may stay higher longer",
            ],
            "summary": ["", "", ""],
        }
    )

    deduplicated = deduplicate_news(news, similarity_threshold=0.90)

    assert len(deduplicated) == 2


def test_aggregate_daily_sentiment_builds_bar_aligned_features() -> None:
    price_index = pd.date_range("2024-01-02", periods=3, freq="D", tz="UTC")
    scored_news = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                [
                    "2024-01-02T15:00:00Z",
                    "2024-01-02T18:00:00Z",
                    "2024-01-03T16:00:00Z",
                ],
                utc=True,
            ),
            "p_positive": [0.70, 0.10, 0.20],
            "p_negative": [0.10, 0.70, 0.10],
            "p_neutral": [0.20, 0.20, 0.70],
        }
    )

    features = aggregate_daily_sentiment(scored_news, price_index, close_hour_utc=21)

    assert {"net_sentiment", "abs_emotion", "is_empty_block", "headline_count"}.issubset(features.columns)
    assert features.loc[price_index[0], "headline_count"] == 2.0
    assert features.loc[price_index[1], "headline_count"] == 1.0
    assert features.loc[price_index[2], "is_empty_block"] == 1.0
