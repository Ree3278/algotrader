"""CLI for building daily sentiment features from raw news."""

from __future__ import annotations

import argparse
from pathlib import Path

from algotrader.ingestion import load_ohlcv_csv, save_timeseries_csv
from algotrader.sentiment.aggregate import aggregate_daily_sentiment
from algotrader.sentiment.dedup import deduplicate_news
from algotrader.sentiment.finbert import score_news_with_finbert
from algotrader.sentiment.news import load_news_csv


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build daily sentiment features from raw news using FinBERT.")
    parser.add_argument("--news-csv", type=Path, required=True, help="Raw news CSV with timestamp/headline columns")
    parser.add_argument("--price-csv", type=Path, required=True, help="Normalized OHLCV CSV used to align daily bars")
    parser.add_argument("--output-csv", type=Path, default=Path("data/interim/sentiment_daily.csv"), help="Output CSV for daily sentiment features")
    parser.add_argument("--scored-news-csv", type=Path, help="Optional output CSV for deduplicated scored news rows")
    parser.add_argument("--batch-size", type=int, default=16, help="FinBERT inference batch size")
    parser.add_argument("--similarity-threshold", type=float, default=0.90, help="Dedup similarity threshold")
    parser.add_argument("--neutral-threshold", type=float, default=0.80, help="Filter threshold for neutral headlines")
    parser.add_argument("--close-hour-utc", type=int, default=21, help="Daily bar close hour in UTC")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    news = load_news_csv(args.news_csv)
    price_frame = load_ohlcv_csv(args.price_csv)
    deduplicated = deduplicate_news(news, similarity_threshold=args.similarity_threshold)
    scored = score_news_with_finbert(deduplicated, batch_size=args.batch_size)
    features = aggregate_daily_sentiment(
        scored,
        price_frame.index,
        close_hour_utc=args.close_hour_utc,
        neutral_threshold=args.neutral_threshold,
    )
    save_timeseries_csv(features, args.output_csv)
    if args.scored_news_csv is not None:
        save_timeseries_csv(scored.set_index("timestamp"), args.scored_news_csv)
    print(f"saved_sentiment_csv={args.output_csv}")
    if args.scored_news_csv is not None:
        print(f"saved_scored_news_csv={args.scored_news_csv}")
