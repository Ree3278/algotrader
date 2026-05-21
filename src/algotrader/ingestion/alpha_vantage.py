"""Alpha Vantage ingestion for daily adjusted OHLCV and news data."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any
from urllib.parse import urlencode
from urllib.request import urlopen

import pandas as pd

from .storage import save_json, save_timeseries_csv

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


def fetch_news_sentiment(
    api_key: str,
    *,
    tickers: str | None = None,
    topics: str | None = None,
    time_from: str | None = None,
    time_to: str | None = None,
    sort: str = "LATEST",
    limit: int = 1000,
    timeout: int = 30,
) -> dict[str, Any]:
    """Fetch market news and sentiment from Alpha Vantage.

    Official docs: `function=NEWS_SENTIMENT` with optional `tickers`, `topics`,
    `time_from`, `time_to`, `sort`, and `limit`.
    """

    params: dict[str, Any] = {
        "function": "NEWS_SENTIMENT",
        "sort": sort,
        "limit": limit,
        "apikey": api_key,
    }
    if tickers:
        params["tickers"] = tickers
    if topics:
        params["topics"] = topics
    if time_from:
        params["time_from"] = time_from
    if time_to:
        params["time_to"] = time_to

    url = f"{ALPHA_VANTAGE_BASE_URL}?{urlencode(params)}"
    with urlopen(url, timeout=timeout) as response:
        payload = json.loads(response.read().decode("utf-8"))
    return payload


def normalize_news_sentiment(
    payload: dict[str, Any],
    *,
    requested_tickers: str | None = None,
    requested_topics: str | None = None,
) -> pd.DataFrame:
    """Normalize Alpha Vantage news payloads into the raw-news CSV schema."""

    if "Error Message" in payload:
        raise ValueError(payload["Error Message"])
    if "Note" in payload:
        raise ValueError(payload["Note"])
    if "Information" in payload and "feed" not in payload:
        raise ValueError(payload["Information"])

    feed = payload.get("feed")
    if feed is None:
        raise ValueError("Payload does not contain 'feed' data")

    rows: list[dict[str, Any]] = []
    for item in feed:
        timestamp = pd.to_datetime(item.get("time_published"), utc=True)
        ticker_sentiment = item.get("ticker_sentiment", [])
        rows.append(
            {
                "timestamp": timestamp,
                "headline": str(item.get("title", "")),
                "summary": str(item.get("summary", "")),
                "source": item.get("source"),
                "category_within_source": item.get("category_within_source"),
                "source_domain": item.get("source_domain"),
                "url": item.get("url"),
                "authors": json.dumps(item.get("authors", [])),
                "banner_image": item.get("banner_image"),
                "topics": json.dumps(item.get("topics", [])),
                "overall_sentiment_score": item.get("overall_sentiment_score"),
                "overall_sentiment_label": item.get("overall_sentiment_label"),
                "ticker_sentiment": json.dumps(ticker_sentiment),
                "requested_tickers": requested_tickers,
                "requested_topics": requested_topics,
            }
        )

    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    frame = frame.sort_values("timestamp").reset_index(drop=True)
    return frame


def build_news_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fetch raw market news from Alpha Vantage and save it as CSV/JSON.")
    parser.add_argument("--tickers", help="Comma-separated tickers filter, e.g. SPY or AAPL,MSFT")
    parser.add_argument("--topics", help="Comma-separated topics filter")
    parser.add_argument("--time-from", help="Alpha Vantage time_from in YYYYMMDDTHHMM format")
    parser.add_argument("--time-to", help="Alpha Vantage time_to in YYYYMMDDTHHMM format")
    parser.add_argument("--sort", default="LATEST", choices=["LATEST", "EARLIEST", "RELEVANCE"])
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--alpha-vantage-key", help="Alpha Vantage API key. Defaults to ALPHA_VANTAGE_API_KEY.")
    parser.add_argument("--output-csv", type=Path, default=Path("data/raw/news/news.csv"))
    parser.add_argument("--output-json", type=Path, default=Path("data/raw/news/news_raw.json"))
    return parser


def news_cli_main() -> None:
    args = build_news_arg_parser().parse_args()
    api_key = args.alpha_vantage_key or os.getenv("ALPHA_VANTAGE_API_KEY")
    if not api_key:
        raise ValueError("Alpha Vantage API key not found. Set ALPHA_VANTAGE_API_KEY or pass --alpha-vantage-key.")

    payload = fetch_news_sentiment(
        api_key,
        tickers=args.tickers,
        topics=args.topics,
        time_from=args.time_from,
        time_to=args.time_to,
        sort=args.sort,
        limit=args.limit,
    )
    normalized = normalize_news_sentiment(
        payload,
        requested_tickers=args.tickers,
        requested_topics=args.topics,
    )
    save_json(payload, args.output_json)
    if normalized.empty:
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        normalized.to_csv(args.output_csv, index=False)
    else:
        save_timeseries_csv(normalized.set_index("timestamp"), args.output_csv)
    print(f"saved_news_csv={args.output_csv}")
    print(f"saved_news_json={args.output_json}")
