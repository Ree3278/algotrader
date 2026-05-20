"""Local storage helpers for raw payloads and normalized OHLCV data."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


def save_json(payload: dict[str, Any], path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def save_ohlcv_csv(frame: pd.DataFrame, path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(destination, index=True)


def load_ohlcv_csv(path: str | Path) -> pd.DataFrame:
    source = Path(path)
    frame = pd.read_csv(source, index_col=0, parse_dates=True)
    frame.index = pd.to_datetime(frame.index, utc=True)
    return frame.sort_index()
