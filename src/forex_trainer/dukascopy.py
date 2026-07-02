"""Long-history hourly cache builder from the Dukascopy public datafeed (ADR-0006).

Downloads month-level BID hourly candle files (LZMA-compressed bi5, big-endian
records of (delta_seconds, open, close, low, high, volume:float32), prices
scaled by 1000 for JPY-quoted pairs), inverts them to the env's JPY-based
convention, and writes a parquet cache compatible with forex-env's file
provider. Non-trading bars (volume <= 0) are dropped.
"""

from __future__ import annotations

import argparse
import lzma
import struct
import sys
import time
import urllib.error
import urllib.request
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Callable

import pandas as pd
from forex_env.data.base import require_jpy_pair, validate_ohlcv
from forex_env.data.file_provider import save_ohlcv_parquet
from forex_env.data.yfinance_provider import invert_quote

_DATAFEED_URL = (
    "https://datafeed.dukascopy.com/datafeed/{instrument}/{year}/{month0:02d}/"
    "BID_candles_hour_1.bi5"
)
_RECORD_FORMAT = ">5if"
_RECORD_SIZE = struct.calcsize(_RECORD_FORMAT)
# Dukascopy quotes JPY-counter pairs with a 0.001 point value; env symbols are
# always JPY-based (require_jpy_pair), so every instrument is XXXJPY.
_JPY_PRICE_SCALE = 1e-3
_TIMEFRAME = "1h"
# Politeness settings for the public datafeed: it throttles aggressively
# (503s and connection drops observed on rapid sequential requests), so every
# request is spaced out and retries back off in tens of seconds.
_DOWNLOAD_ATTEMPTS = 5
_RETRY_BACKOFF_SECONDS = 15.0
_REQUEST_INTERVAL_SECONDS = 1.0
_USER_AGENT = "forex-trainer/0.1 (personal research)"

FetchMonthFn = Callable[[str, int, int], bytes]


class DukascopyError(Exception):
    """Raised when downloading or decoding Dukascopy data fails."""


def to_dukascopy_instrument(symbol: str) -> str:
    """Map a JPY-based env symbol to its Dukascopy instrument name.

    Args:
        symbol: JPY-based symbol like "JPY/USD".

    Returns:
        Dukascopy instrument, e.g. "USDJPY".
    """
    counter = require_jpy_pair(symbol)
    return f"{counter}JPY"


def decode_hour_candles(
    payload: bytes, month_start: datetime, price_scale: float
) -> pd.DataFrame:
    """Decode one month-level hourly candle file.

    Args:
        payload: LZMA-compressed bi5 file contents.
        month_start: UTC timestamp of the month's first hour.
        price_scale: Multiplier converting integer prices to quote prices.

    Returns:
        OHLCV DataFrame indexed by UTC timestamps, with non-trading bars
        (volume <= 0) removed.

    Raises:
        DukascopyError: If the payload cannot be decompressed or its size is
            not a multiple of the record size.
    """
    try:
        raw = lzma.decompress(payload)
    except lzma.LZMAError as exc:
        raise DukascopyError(f"Failed to decompress candle payload: {exc}") from exc
    if len(raw) % _RECORD_SIZE != 0:
        raise DukascopyError(
            f"Candle payload of {len(raw)} bytes is not a multiple of the "
            f"record size {_RECORD_SIZE}."
        )
    records = [
        struct.unpack(_RECORD_FORMAT, raw[offset : offset + _RECORD_SIZE])
        for offset in range(0, len(raw), _RECORD_SIZE)
    ]
    frame = pd.DataFrame(
        {
            "Open": [record[1] * price_scale for record in records],
            "High": [record[4] * price_scale for record in records],
            "Low": [record[3] * price_scale for record in records],
            "Close": [record[2] * price_scale for record in records],
            "Volume": [float(record[5]) for record in records],
        },
        index=pd.DatetimeIndex(
            [month_start + pd.Timedelta(seconds=record[0]) for record in records]
        ),
    )
    return frame[frame["Volume"] > 0.0]


def fetch_month_http(instrument: str, year: int, month: int) -> bytes:
    """Download one month-level hourly candle file.

    Args:
        instrument: Dukascopy instrument name, e.g. "USDJPY".
        year: Calendar year.
        month: Calendar month (1-12); the datafeed URL uses 0-based months.

    Returns:
        Raw bi5 file contents.

    Raises:
        DukascopyError: On HTTP errors (after bounded retries on transient
            failures) or an empty response.
    """
    url = _DATAFEED_URL.format(instrument=instrument, year=year, month0=month - 1)
    last_error: Exception | None = None
    for attempt in range(_DOWNLOAD_ATTEMPTS):
        time.sleep(
            _REQUEST_INTERVAL_SECONDS
            if attempt == 0
            else _RETRY_BACKOFF_SECONDS * 2 ** (attempt - 1)
        )
        request = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                payload = response.read()
        except urllib.error.HTTPError as exc:
            # 5xx responses are transient datafeed throttling; anything else
            # (404 etc.) is a real request error and must surface immediately.
            if exc.code < 500:
                raise DukascopyError(f"Failed to download {url}: {exc}") from exc
            last_error = exc
            continue
        except (urllib.error.URLError, OSError) as exc:
            last_error = exc
            continue
        if not payload:
            raise DukascopyError(f"Empty candle file at {url}.")
        return payload
    raise DukascopyError(
        f"Failed to download {url} after {_DOWNLOAD_ATTEMPTS} attempts: {last_error}"
    )


def _iterate_months(start: date, end: date) -> list[tuple[int, int]]:
    """List (year, month) pairs covering [start, end].

    Args:
        start: Inclusive range start.
        end: Inclusive range end.

    Returns:
        Chronological list of (year, month) tuples.
    """
    months: list[tuple[int, int]] = []
    year, month = start.year, start.month
    while (year, month) <= (end.year, end.month):
        months.append((year, month))
        year, month = (year + 1, 1) if month == 12 else (year, month + 1)
    return months


def build_cache(
    symbols: tuple[str, ...],
    start: str,
    end: str,
    output: Path,
    fetch_month: FetchMonthFn,
) -> pd.DataFrame:
    """Build a file-provider-compatible hourly parquet cache.

    Args:
        symbols: JPY-based currency pairs.
        start: Inclusive start date (YYYY-MM-DD).
        end: Inclusive end date (YYYY-MM-DD).
        output: Destination parquet path.
        fetch_month: Function returning the bi5 payload for
            (instrument, year, month); injectable so tests avoid the network.

    Returns:
        The DataFrame that was written (MultiIndex columns (symbol, field)).

    Raises:
        DukascopyError: If any month fails to download or decode.
        forex_env.errors.DataError: If the assembled data fails validation.
    """
    start_date = date.fromisoformat(start)
    end_date = date.fromisoformat(end)
    months = _iterate_months(start_date, end_date)
    frames: list[pd.DataFrame] = []
    for symbol in symbols:
        instrument = to_dukascopy_instrument(symbol)
        monthly: list[pd.DataFrame] = []
        for year, month in months:
            payload = fetch_month(instrument, year, month)
            month_start = datetime(year, month, 1, tzinfo=timezone.utc)
            monthly.append(decode_hour_candles(payload, month_start, _JPY_PRICE_SCALE))
        combined = pd.concat(monthly).loc[start:end]
        inverted = invert_quote(combined)
        inverted.columns = pd.MultiIndex.from_product([[symbol], inverted.columns])
        frames.append(inverted)

    # Inner join keeps only timestamps quoted for every symbol (same rule as
    # the yfinance provider): no prices are fabricated by filling.
    data = pd.concat(frames, axis=1, join="inner")
    validate_ohlcv(data, symbols)
    save_ohlcv_parquet(data, _TIMEFRAME, start, end, output)
    return data


def main(argv: list[str] | None = None) -> int:
    """CLI entry point.

    Args:
        argv: CLI arguments; None lets argparse read sys.argv (testability).

    Returns:
        Process exit code: 0 on success, 1 on download/validation errors.
    """
    parser = argparse.ArgumentParser(
        prog="forex-fetch-dukascopy",
        description="Build an hourly parquet cache from the Dukascopy datafeed.",
    )
    parser.add_argument(
        "--pairs",
        type=str,
        required=True,
        help='Comma-separated JPY-based pairs, e.g. "JPY/USD,JPY/EUR".',
    )
    parser.add_argument(
        "--start", type=str, required=True, help="Inclusive start date (YYYY-MM-DD)."
    )
    parser.add_argument(
        "--end", type=str, required=True, help="Inclusive end date (YYYY-MM-DD)."
    )
    parser.add_argument(
        "--output", type=str, required=True, help="Destination parquet path."
    )
    args = parser.parse_args(argv)
    symbols = tuple(pair.strip() for pair in args.pairs.split(","))
    try:
        data = build_cache(
            symbols, args.start, args.end, Path(args.output), fetch_month_http
        )
    except Exception as exc:  # surface every failure explicitly at the CLI
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(
        f"wrote {args.output}: {len(data.index)} rows, "
        f"{data.index[0]} .. {data.index[-1]}, symbols {list(symbols)}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
