"""Raw news loading and validation."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_news_csv(path: str | Path) -> pd.DataFrame:
    """Load raw news rows from CSV.

    Required columns:
    - `timestamp`
    - `headline`

    Optional columns:
    - `summary`
    - `source`
    - `url`
    """

    frame = pd.read_csv(path)
    required = {"timestamp", "headline"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Missing required news columns: {missing}")

    normalized = frame.copy()
    normalized["timestamp"] = pd.to_datetime(normalized["timestamp"], utc=True)
    normalized["headline"] = normalized["headline"].fillna("").astype(str)
    if "summary" in normalized.columns:
        normalized["summary"] = normalized["summary"].fillna("").astype(str)
    else:
        normalized["summary"] = ""
    normalized = normalized[normalized["headline"].str.strip() != ""]
    normalized = normalized.sort_values("timestamp").reset_index(drop=True)
    return normalized
