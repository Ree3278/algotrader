"""Alpha Vantage ingestion for daily adjusted OHLCV data."""

from __future__ import annotations

import json
from typing import Any
from urllib.parse import urlencode
from urllib.request import urlopen

import pandas as pd

ALPHA_VANTAGE_BASE_URL = "https://www.alphavantage.co/query"


def fetch_daily_adjusted(
    symbol: str,
    api_key: str,
    *,
    outputsize: str = "full",
    timeout: int = 30,
) -> dict[str, Any]:
    """Fetch daily adjusted OHLCV data from Alpha Vantage."""

    query = urlencode(
        {
            "function": "TIME_SERIES_DAILY_ADJUSTED",
            "symbol": symbol,
            "outputsize": outputsize,
            "apikey": api_key,
        }
    )
    url = f"{ALPHA_VANTAGE_BASE_URL}?{query}"
    with urlopen(url, timeout=timeout) as response:
        payload = json.loads(response.read().decode("utf-8"))
    return payload


def normalize_daily_adjusted(
    payload: dict[str, Any],
    *,
    symbol: str | None = None,
) -> pd.DataFrame:
    """Normalize Alpha Vantage daily adjusted payloads into a sorted DataFrame."""

    if "Error Message" in payload:
        raise ValueError(payload["Error Message"])
    if "Note" in payload:
        raise ValueError(payload["Note"])
    if "Information" in payload and "Time Series (Daily)" not in payload:
        raise ValueError(payload["Information"])

    series = payload.get("Time Series (Daily)")
    if not series:
        raise ValueError("Payload does not contain 'Time Series (Daily)' data")

    resolved_symbol = symbol or payload.get("Meta Data", {}).get("2. Symbol")
    rows = []
    for timestamp, values in series.items():
        rows.append(
            {
                "timestamp": pd.Timestamp(timestamp, tz="UTC"),
                "symbol": resolved_symbol,
                "open": float(values["1. open"]),
                "high": float(values["2. high"]),
                "low": float(values["3. low"]),
                "close": float(values["4. close"]),
                "adjusted_close": float(values["5. adjusted close"]),
                "volume": float(values["6. volume"]),
                "dividend_amount": float(values["7. dividend amount"]),
                "split_coefficient": float(values["8. split coefficient"]),
            }
        )

    frame = pd.DataFrame(rows).set_index("timestamp").sort_index()
    if frame.index.has_duplicates:
        raise ValueError("Normalized payload contains duplicate timestamps")
    return frame
