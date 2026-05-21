from __future__ import annotations

import pandas as pd

from algotrader.ingestion import normalize_daily_adjusted, normalize_news_sentiment, normalize_yfinance_ohlcv


def test_normalize_daily_adjusted_sorts_rows_and_parses_numeric_fields() -> None:
    payload = {
        "Meta Data": {"2. Symbol": "SPY"},
        "Time Series (Daily)": {
            "2024-01-03": {
                "1. open": "100.0",
                "2. high": "101.0",
                "3. low": "99.0",
                "4. close": "100.5",
                "5. adjusted close": "100.4",
                "6. volume": "1000000",
                "7. dividend amount": "0.0000",
                "8. split coefficient": "1.0",
            },
            "2024-01-02": {
                "1. open": "99.0",
                "2. high": "100.0",
                "3. low": "98.0",
                "4. close": "99.5",
                "5. adjusted close": "99.4",
                "6. volume": "900000",
                "7. dividend amount": "0.0000",
                "8. split coefficient": "1.0",
            },
        },
    }

    frame = normalize_daily_adjusted(payload)

    assert frame.index[0].isoformat() == "2024-01-02T00:00:00+00:00"
    assert frame.iloc[0]["symbol"] == "SPY"
    assert frame.iloc[1]["close"] == 100.5
    assert frame.iloc[0]["volume"] == 900000.0


def test_normalize_daily_adjusted_raises_on_rate_limit_payload() -> None:
    payload = {"Note": "API call frequency exceeded"}

    try:
        normalize_daily_adjusted(payload)
    except ValueError as exc:
        assert "frequency exceeded" in str(exc)
    else:
        raise AssertionError("Expected ValueError for rate-limited payload")


def test_normalize_yfinance_ohlcv_maps_standard_columns() -> None:
    frame = pd.DataFrame(
        {
            "Open": [100.0, 101.0],
            "High": [101.0, 102.0],
            "Low": [99.0, 100.0],
            "Close": [100.5, 101.5],
            "Adj Close": [100.4, 101.4],
            "Volume": [1_000_000, 1_100_000],
            "Dividends": [0.0, 0.0],
            "Stock Splits": [0.0, 0.0],
        },
        index=pd.to_datetime(["2024-01-02", "2024-01-03"]),
    )

    normalized = normalize_yfinance_ohlcv(frame, symbol="SPY")

    assert normalized.index[0].isoformat() == "2024-01-02T00:00:00+00:00"
    assert normalized.iloc[0]["symbol"] == "SPY"
    assert normalized.iloc[1]["adjusted_close"] == 101.4
    assert normalized.iloc[0]["split_coefficient"] == 1.0


def test_normalize_news_sentiment_maps_feed_to_raw_news_schema() -> None:
    payload = {
        "feed": [
            {
                "title": "SPY rallies as inflation cools",
                "summary": "Markets react positively to cooler inflation data.",
                "time_published": "20240520T153000",
                "source": "Reuters",
                "url": "https://example.com/story",
                "overall_sentiment_score": 0.25,
                "overall_sentiment_label": "Bullish",
                "ticker_sentiment": [{"ticker": "SPY", "ticker_sentiment_score": "0.4"}],
                "topics": [{"topic": "economy_macro"}],
                "authors": ["Jane Doe"],
            }
        ]
    }

    frame = normalize_news_sentiment(payload, requested_tickers="SPY", requested_topics="economy_macro")

    assert frame.iloc[0]["headline"] == "SPY rallies as inflation cools"
    assert frame.iloc[0]["source"] == "Reuters"
    assert frame.iloc[0]["requested_tickers"] == "SPY"
    assert frame.iloc[0]["requested_topics"] == "economy_macro"
    assert frame.iloc[0]["timestamp"].isoformat() == "2024-05-20T15:30:00+00:00"
