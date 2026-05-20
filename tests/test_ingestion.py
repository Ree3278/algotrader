from __future__ import annotations

import json

from algotrader.ingestion import normalize_daily_adjusted


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
