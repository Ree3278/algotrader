"""News deduplication utilities."""

from __future__ import annotations

from typing import Iterable

import pandas as pd


def _similarity(a: str, b: str) -> float:
    try:
        from rapidfuzz import fuzz
    except ImportError as exc:
        raise RuntimeError("rapidfuzz is required for sentiment deduplication. Install the `nlp` extra.") from exc

    return float(fuzz.ratio(a, b)) / 100.0


def _combined_text(row: pd.Series) -> str:
    headline = str(row.get("headline", "")).strip()
    summary = str(row.get("summary", "")).strip()
    return f"{headline} {summary}".strip().lower()


def deduplicate_news(
    frame: pd.DataFrame,
    *,
    similarity_threshold: float = 0.90,
) -> pd.DataFrame:
    """Deduplicate highly similar headlines within the same UTC day."""

    if frame.empty:
        return frame.copy()

    required = {"timestamp", "headline"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Missing required news columns: {missing}")

    normalized = frame.copy()
    normalized["timestamp"] = pd.to_datetime(normalized["timestamp"], utc=True)
    normalized = normalized.sort_values("timestamp").reset_index(drop=True)
    normalized["_dedup_text"] = normalized.apply(_combined_text, axis=1)
    normalized["_day_bucket"] = normalized["timestamp"].dt.floor("D")

    kept_indices: list[int] = []
    for _, day_group in normalized.groupby("_day_bucket", sort=True):
        group_kept: list[int] = []
        group_texts: list[str] = []

        # Keep the richest text representation within each near-duplicate cluster.
        sorted_group = day_group.sort_values("_dedup_text", key=lambda s: s.str.len(), ascending=False)
        for row_index, row in sorted_group.iterrows():
            text = str(row["_dedup_text"])
            if not group_texts:
                group_kept.append(int(row_index))
                group_texts.append(text)
                continue

            is_duplicate = any(_similarity(text, kept_text) >= similarity_threshold for kept_text in group_texts)
            if not is_duplicate:
                group_kept.append(int(row_index))
                group_texts.append(text)

        kept_indices.extend(group_kept)

    deduplicated = normalized.loc[sorted(kept_indices)].drop(columns=["_dedup_text", "_day_bucket"])
    return deduplicated.reset_index(drop=True)
