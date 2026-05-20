"""yfinance ingestion for daily OHLCV data."""

from __future__ import annotations

from typing import Any

import pandas as pd


def _flatten_yfinance_columns(frame: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(frame.columns, pd.MultiIndex):
        return frame

    if frame.columns.nlevels != 2:
        raise ValueError("Unexpected yfinance column shape")

    flattened = frame.copy()
    flattened.columns = [str(column[0]) for column in frame.columns]
    return flattened


def normalize_yfinance_ohlcv(
    frame: pd.DataFrame,
    *,
    symbol: str,
) -> pd.DataFrame:
    """Normalize yfinance output into the project's OHLCV schema."""

    if frame.empty:
        raise ValueError("yfinance returned no rows")

    normalized = _flatten_yfinance_columns(frame).copy()
    normalized.index = pd.to_datetime(normalized.index, utc=True)
    required = {"Open", "High", "Low", "Close", "Volume"}
    missing = required.difference(normalized.columns)
    if missing:
        raise ValueError(f"Missing expected yfinance columns: {sorted(missing)}")

    result = pd.DataFrame(index=normalized.index)
    result["symbol"] = symbol
    result["open"] = normalized["Open"].astype(float).to_numpy()
    result["high"] = normalized["High"].astype(float).to_numpy()
    result["low"] = normalized["Low"].astype(float).to_numpy()
    result["close"] = normalized["Close"].astype(float).to_numpy()
    if "Adj Close" in normalized.columns:
        result["adjusted_close"] = normalized["Adj Close"].astype(float).to_numpy()
    else:
        result["adjusted_close"] = result["close"]
    result["volume"] = normalized["Volume"].astype(float).to_numpy()
    if "Dividends" in normalized.columns:
        result["dividend_amount"] = normalized["Dividends"].astype(float).to_numpy()
    else:
        result["dividend_amount"] = 0.0
    if "Stock Splits" in normalized.columns:
        result["split_coefficient"] = normalized["Stock Splits"].astype(float).replace(0.0, 1.0).to_numpy()
    else:
        result["split_coefficient"] = 1.0

    result = result.sort_index()
    if result.index.has_duplicates:
        raise ValueError("Normalized yfinance data contains duplicate timestamps")
    return result


def fetch_yfinance_daily(
    symbol: str,
    *,
    period: str = "max",
    start: str | None = None,
    end: str | None = None,
    auto_adjust: bool = False,
) -> pd.DataFrame:
    """Fetch and normalize daily OHLCV data from yfinance."""

    try:
        import yfinance as yf
    except ImportError as exc:
        raise RuntimeError("yfinance is not installed in the current environment") from exc

    raw = yf.download(
        tickers=symbol,
        period=period,
        start=start,
        end=end,
        interval="1d",
        auto_adjust=auto_adjust,
        progress=False,
        actions=True,
        group_by="column",
        threads=False,
    )
    return normalize_yfinance_ohlcv(raw, symbol=symbol)
