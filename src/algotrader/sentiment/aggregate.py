"""Aggregate scored news into daily sentiment features."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd


def aggregate_daily_sentiment(
    scored_news: pd.DataFrame,
    price_index: pd.Index,
    *,
    close_hour_utc: int = 21,
    close_minute_utc: int = 0,
    neutral_threshold: float = 0.80,
    decay_lambda: float = 0.007,
) -> pd.DataFrame:
    """Aggregate scored news into daily bar-aligned sentiment features."""

    required = {"timestamp", "p_positive", "p_negative", "p_neutral"}
    missing = sorted(required.difference(scored_news.columns))
    if missing:
        raise ValueError(f"Missing required scored news columns: {missing}")

    news = scored_news.copy()
    news["timestamp"] = pd.to_datetime(news["timestamp"], utc=True)
    news = news.sort_values("timestamp").reset_index(drop=True)

    index = pd.DatetimeIndex(price_index)
    if index.tz is None:
        index = index.tz_localize("UTC")
    else:
        index = index.tz_convert("UTC")

    bar_closes = index.normalize() + pd.Timedelta(hours=close_hour_utc, minutes=close_minute_utc)
    records: list[dict[str, float | int]] = []

    for i, (signal_index, bar_close) in enumerate(zip(index, bar_closes)):
        previous_close = bar_close - pd.Timedelta(days=1) if i == 0 else bar_closes[i - 1]
        mask = (news["timestamp"] > previous_close) & (news["timestamp"] < bar_close)
        block = news.loc[mask].copy()
        raw_count = int(len(block))

        if raw_count == 0:
            records.append(
                {
                    "timestamp": signal_index,
                    "net_sentiment": 0.0,
                    "abs_emotion": 0.0,
                    "is_empty_block": 1.0,
                    "headline_count": 0.0,
                }
            )
            continue

        block = block.loc[block["p_neutral"] <= neutral_threshold].copy()
        if block.empty:
            records.append(
                {
                    "timestamp": signal_index,
                    "net_sentiment": 0.0,
                    "abs_emotion": 0.0,
                    "is_empty_block": 1.0,
                    "headline_count": 0.0,
                }
            )
            continue

        delta_minutes = (bar_close - block["timestamp"]).dt.total_seconds() / 60.0
        weights = np.exp(-decay_lambda * delta_minutes.to_numpy())
        net_components = (block["p_positive"] - block["p_negative"]).to_numpy()
        abs_components = (block["p_positive"] + block["p_negative"]).to_numpy()
        denom = float(weights.sum())
        net_sentiment = float(np.dot(weights, net_components) / denom)
        abs_emotion = float(np.dot(weights, abs_components) / denom)

        records.append(
            {
                "timestamp": signal_index,
                "net_sentiment": net_sentiment,
                "abs_emotion": abs_emotion,
                "is_empty_block": 0.0,
                "headline_count": float(len(block)),
            }
        )

    aggregated = pd.DataFrame(records).set_index("timestamp")
    aggregated.index = pd.to_datetime(aggregated.index, utc=True)
    return aggregated
