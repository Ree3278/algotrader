"""Technical indicators for the initial price-only baseline."""

from __future__ import annotations

import numpy as np
import pandas as pd

REQUIRED_OHLCV_COLUMNS = ("open", "high", "low", "close", "volume")


def _validate_ohlcv_frame(frame: pd.DataFrame) -> None:
    missing = [column for column in REQUIRED_OHLCV_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing required OHLCV columns: {missing}")
    if not frame.index.is_monotonic_increasing:
        raise ValueError("OHLCV frame index must be sorted ascending")
    if frame.index.has_duplicates:
        raise ValueError("OHLCV frame index must not contain duplicates")


def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False, min_periods=span).mean()


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gains = delta.clip(lower=0.0)
    losses = -delta.clip(upper=0.0)
    avg_gain = gains.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = losses.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.mask((avg_loss == 0) & (avg_gain > 0), 100.0)
    rsi = rsi.mask((avg_gain == 0) & (avg_loss > 0), 0.0)
    rsi = rsi.mask((avg_gain == 0) & (avg_loss == 0), 50.0)
    return rsi


def _atr(frame: pd.DataFrame, period: int = 14) -> pd.Series:
    prev_close = frame["close"].shift(1)
    tr = pd.concat(
        [
            frame["high"] - frame["low"],
            (frame["high"] - prev_close).abs(),
            (frame["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.rolling(window=period, min_periods=period).mean()


def build_price_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Return the initial causal price-only feature matrix.

    The returned frame preserves the input index. Warmup rows remain present
    with NaNs where an indicator is not yet available.
    """

    _validate_ohlcv_frame(frame)

    features = frame.copy()
    close = features["close"]

    features["return_1d"] = close.pct_change(1)
    features["return_5d"] = close.pct_change(5)
    features["volatility_20d"] = features["return_1d"].rolling(window=20, min_periods=20).std()
    sma_200 = close.rolling(window=200, min_periods=200).mean()
    features["price_above_sma_200"] = np.where(sma_200.notna(), (close > sma_200).astype(float), np.nan)
    if "vix_close" in features.columns:
        vix_mean_60 = features["vix_close"].rolling(window=60, min_periods=60).mean()
        vix_std_60 = features["vix_close"].rolling(window=60, min_periods=60).std().replace(0.0, np.nan)
        features["vix_zscore_60d"] = (features["vix_close"] - vix_mean_60) / vix_std_60
    features["ATR_14"] = _atr(features, period=14)
    features["RSI_14"] = _rsi(close, period=14)

    ema_fast = _ema(close, span=12)
    ema_slow = _ema(close, span=26)
    features["MACD_line"] = ema_fast - ema_slow
    features["MACD_signal"] = _ema(features["MACD_line"], span=9)
    features["MACD_hist"] = features["MACD_line"] - features["MACD_signal"]

    rolling_mean_20 = close.rolling(window=20, min_periods=20).mean()
    rolling_std_20 = close.rolling(window=20, min_periods=20).std()
    band_width = 2 * rolling_std_20
    features["BB_upper"] = rolling_mean_20 + band_width
    features["BB_lower"] = rolling_mean_20 - band_width
    denom = (features["BB_upper"] - features["BB_lower"]).replace(0.0, np.nan)
    features["BB_pct_b"] = (close - features["BB_lower"]) / denom

    volume_mean_20 = features["volume"].rolling(window=20, min_periods=20).mean()
    volume_std_20 = features["volume"].rolling(window=20, min_periods=20).std().replace(0.0, np.nan)
    features["volume_zscore_20d"] = (features["volume"] - volume_mean_20) / volume_std_20

    return features
