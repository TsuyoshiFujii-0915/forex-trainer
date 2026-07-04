"""Feature registry for forex-trainer (ADR-0002).

Every function maps a per-pair OHLCV DataFrame (columns Open/High/Low/Close/
Volume) to a Series aligned to its index. Functions must be causal (use only
past rows) and finite from the first few rows so they survive the env's
warmup check; the test suite enforces both properties for every registry
entry automatically.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from forex_env.features import CrossFeatureFn, FeatureFn

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


def _momentum(frame: pd.DataFrame, bars: int) -> pd.Series:
    """Log return of Close over the last `bars` bars.

    Args:
        frame: Per-pair OHLCV DataFrame.
        bars: Lookback horizon in bars.

    Returns:
        Momentum series, defined from the first row: before `bars` rows of
        history exist, the lookback is anchored to the first Close
        (partial-horizon momentum), which keeps the series causal and finite.
    """
    close = frame["Close"]
    baseline = close.shift(bars).fillna(close.iloc[0])
    return np.log(close / baseline)


def mom24(frame: pd.DataFrame) -> pd.Series:
    """24-bar momentum (log return over the last 24 bars).

    Args:
        frame: Per-pair OHLCV DataFrame.

    Returns:
        Momentum series (see `_momentum` for early-row semantics).
    """
    return _momentum(frame, 24)


def mom72(frame: pd.DataFrame) -> pd.Series:
    """72-bar momentum (log return over the last 72 bars).

    Args:
        frame: Per-pair OHLCV DataFrame.

    Returns:
        Momentum series (see `_momentum` for early-row semantics).
    """
    return _momentum(frame, 72)


def mom168(frame: pd.DataFrame) -> pd.Series:
    """168-bar momentum (log return over the last 168 bars).

    Args:
        frame: Per-pair OHLCV DataFrame.

    Returns:
        Momentum series (see `_momentum` for early-row semantics).
    """
    return _momentum(frame, 168)


def mom720(frame: pd.DataFrame) -> pd.Series:
    """720-bar momentum (~1 month of hourly bars).

    Args:
        frame: Per-pair OHLCV DataFrame.

    Returns:
        Momentum series (see `_momentum` for early-row semantics).
    """
    return _momentum(frame, 720)


FEATURE_REGISTRY: dict[str, FeatureFn] = {
    "sma20_ratio": sma20_ratio,
    "rsi14": rsi14,
    "atr14_ratio": atr14_ratio,
    "macd_ratio": macd_ratio,
    "mom24": mom24,
    "mom72": mom72,
    "mom168": mom168,
    "mom720": mom720,
}


def _cross_momentum(
    data: pd.DataFrame, symbols: tuple[str, ...], bars: int
) -> pd.DataFrame:
    """Per-symbol momentum matrix over the last `bars` bars.

    Args:
        data: Full MultiIndex OHLCV frame.
        symbols: Symbol order for the output columns.
        bars: Lookback horizon in bars.

    Returns:
        DataFrame indexed like data with one momentum column per symbol,
        anchored to the first close before a full horizon exists (causal).
    """
    closes = pd.DataFrame(
        {symbol: data[(symbol, "Close")].astype(float) for symbol in symbols}
    )
    baseline = closes.shift(bars).fillna(closes.iloc[0])
    return np.log(closes / baseline)


def xz_mom24(data: pd.DataFrame, symbols: tuple[str, ...]) -> pd.DataFrame:
    """Cross-sectional z-score of 24-bar momentum (ADR-0008 contract).

    Args:
        data: Full MultiIndex OHLCV frame.
        symbols: Symbol order.

    Returns:
        Per-symbol z-score of momentum across symbols at each bar.
    """
    momentum = _cross_momentum(data, symbols, 24)
    centered = momentum.sub(momentum.mean(axis=1), axis=0)
    return centered.div(momentum.std(axis=1) + _EPSILON, axis=0)


def xr_mom24(data: pd.DataFrame, symbols: tuple[str, ...]) -> pd.DataFrame:
    """Cross-sectional rank of 24-bar momentum, scaled to [-1, 1].

    Args:
        data: Full MultiIndex OHLCV frame.
        symbols: Symbol order.

    Returns:
        Per-symbol momentum rank across symbols at each bar; -1 is the
        weakest symbol, +1 the strongest.
    """
    momentum = _cross_momentum(data, symbols, 24)
    count = len(symbols)
    return (momentum.rank(axis=1) - (count + 1) / 2.0) / ((count - 1) / 2.0)


def xz_mom720(data: pd.DataFrame, symbols: tuple[str, ...]) -> pd.DataFrame:
    """Cross-sectional z-score of 720-bar momentum (~1 month of hourly bars).

    Args:
        data: Full MultiIndex OHLCV frame.
        symbols: Symbol order.

    Returns:
        Per-symbol z-score of momentum across symbols at each bar.
    """
    momentum = _cross_momentum(data, symbols, 720)
    centered = momentum.sub(momentum.mean(axis=1), axis=0)
    return centered.div(momentum.std(axis=1) + _EPSILON, axis=0)


def xr_mom720(data: pd.DataFrame, symbols: tuple[str, ...]) -> pd.DataFrame:
    """Cross-sectional rank of 720-bar momentum, scaled to [-1, 1].

    Args:
        data: Full MultiIndex OHLCV frame.
        symbols: Symbol order.

    Returns:
        Per-symbol momentum rank across symbols at each bar; -1 is the
        weakest symbol, +1 the strongest.
    """
    momentum = _cross_momentum(data, symbols, 720)
    count = len(symbols)
    return (momentum.rank(axis=1) - (count + 1) / 2.0) / ((count - 1) / 2.0)


CROSS_FEATURE_REGISTRY: dict[str, CrossFeatureFn] = {
    "xz_mom24": xz_mom24,
    "xr_mom24": xr_mom24,
    "xz_mom720": xz_mom720,
    "xr_mom720": xr_mom720,
}
