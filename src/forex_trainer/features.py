"""Feature registry for forex-trainer (ADR-0002).

Every function maps a per-pair OHLCV DataFrame (columns Open/High/Low/Close/
Volume) to a Series aligned to its index. Functions must be causal (use only
past rows) and finite from the first few rows so they survive the env's
warmup check; the test suite enforces both properties for every registry
entry automatically.
"""

from __future__ import annotations

import pandas as pd

from forex_env.features import FeatureFn

_EPSILON = 1e-12


def sma20_ratio(frame: pd.DataFrame) -> pd.Series:
    """Close relative to its 20-bar simple moving average, minus one.

    Args:
        frame: Per-pair OHLCV DataFrame.

    Returns:
        Stationary ratio series, defined from the first row
        (min_periods=1 makes the early SMA a partial-window mean).
    """
    close = frame["Close"]
    sma = close.rolling(window=20, min_periods=1).mean()
    return close / sma - 1.0


def rsi14(frame: pd.DataFrame) -> pd.Series:
    """Relative Strength Index over 14 bars (simple-mean variant).

    Args:
        frame: Per-pair OHLCV DataFrame.

    Returns:
        RSI series in [0, 100]; the first row is NaN (no price change yet),
        which falls inside the env warmup for any volatility_window >= 2.
    """
    delta = frame["Close"].diff()
    gain = delta.clip(lower=0.0).rolling(window=14, min_periods=1).mean()
    loss = (-delta.clip(upper=0.0)).rolling(window=14, min_periods=1).mean()
    return 100.0 - 100.0 / (1.0 + gain / (loss + _EPSILON))


def atr14_ratio(frame: pd.DataFrame) -> pd.Series:
    """Average True Range over 14 bars, as a fraction of Close.

    Args:
        frame: Per-pair OHLCV DataFrame.

    Returns:
        ATR/Close series, defined from the first row (true range degrades to
        High-Low where no previous close exists).
    """
    previous_close = frame["Close"].shift(1)
    ranges = pd.concat(
        [
            frame["High"] - frame["Low"],
            (frame["High"] - previous_close).abs(),
            (frame["Low"] - previous_close).abs(),
        ],
        axis=1,
    )
    true_range = ranges.max(axis=1)
    atr = true_range.rolling(window=14, min_periods=1).mean()
    return atr / frame["Close"]


def macd_ratio(frame: pd.DataFrame) -> pd.Series:
    """MACD line (EMA12 - EMA26) as a fraction of Close.

    Args:
        frame: Per-pair OHLCV DataFrame.

    Returns:
        Stationary MACD ratio series, defined from the first row (recursive
        EMAs with adjust=False are causal by construction).
    """
    close = frame["Close"]
    ema_fast = close.ewm(span=12, adjust=False).mean()
    ema_slow = close.ewm(span=26, adjust=False).mean()
    return (ema_fast - ema_slow) / close


FEATURE_REGISTRY: dict[str, FeatureFn] = {
    "sma20_ratio": sma20_ratio,
    "rsi14": rsi14,
    "atr14_ratio": atr14_ratio,
    "macd_ratio": macd_ratio,
}
