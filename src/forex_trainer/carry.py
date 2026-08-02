"""FRED carry augmentation CLI: forex-add-carry (ADR-0010 / env ADR-0012).

Adds a per-symbol CarryAnnual field (annualized short-rate differential
counter currency minus the denomination/base currency, decimal) to an
existing parquet cache. Monthly FRED rates are forward-filled daily and
lagged for causality before being aligned to the cache timestamps.

Rates are keyed by 3-letter CURRENCY CODE, not by pair, so the same mapping
serves any denomination currency (env ADR-0010): for symbol "BASE/COUNTER",
the differential is rate[COUNTER] - rate[BASE].
"""

from __future__ import annotations

import argparse
import io
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Callable

import pandas as pd
from forex_env.data.base import split_pair
from forex_env.data.file_provider import load_ohlcv_parquet
from forex_env.errors import DataError

from .data_lineage import DataLineageError, read_data_lineage, save_augmented_cache

CURRENCY_SHORT_RATE_SERIES: dict[str, str] = {
    "USD": "FEDFUNDS",
    "JPY": "IR3TIB01JPM156N",
    "EUR": "ECBDFR",
    "GBP": "IR3TIB01GBM156N",
    "AUD": "IR3TIB01AUM156N",
    "CHF": "IR3TIB01CHM156N",
    "CAD": "IR3TIB01CAM156N",
    "NZD": "IR3TIB01NZM156N",
    "NOK": "IR3TIB01NOM156N",
    "SEK": "IR3TIB01SEM156N",
    "ZAR": "IR3TIB01ZAM156N",
}
_FRED_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
_PERCENT_TO_DECIMAL = 0.01
FetchSeriesFn = Callable[[str], pd.Series]


class CarryError(Exception):
    """Raised when carry augmentation fails."""


def load_current_cache(input_path: Path) -> tuple[pd.DataFrame, str, str, str]:
    """Load a cache only after validating the current forex-env contract.

    Both the ``absent`` and ``counter-minus-base-v1`` carry contracts are
    supported. Their declared contract must agree with the stored columns.
    This allows carry regeneration to replace an explicitly understood
    current contract without interpreting unversioned legacy data.

    Args:
        input_path: Cache parquet written by the current forex-env writer.

    Returns:
        Tuple of frame, timeframe, declared start date, and declared end date.

    Raises:
        CarryError: If the parquet cannot be read or its schema/carry contract
            is missing, unsupported, or inconsistent with its columns.
    """
    try:
        cache = load_ohlcv_parquet(input_path)
        read_data_lineage(input_path)
    except (DataError, DataLineageError) as exc:
        raise CarryError(str(exc)) from exc
    return cache.data, cache.timeframe, cache.start_date, cache.end_date


def _validate_lag_days(lag_days: int) -> None:
    """Reject lags that would move future observations into the past.

    Args:
        lag_days: Calendar-day publication lag.

    Raises:
        CarryError: If lag_days is negative.
    """
    if lag_days < 0:
        raise CarryError(f"lag_days must be non-negative, got {lag_days}.")


def fetch_fred_series(series_id: str) -> pd.Series:
    """Download one FRED series as a date-indexed value series (percent).

    Args:
        series_id: FRED series identifier.

    Returns:
        Series indexed by observation date with numeric values; missing
        observations are dropped.

    Raises:
        CarryError: On download or parse failures.
    """
    url = _FRED_URL.format(series_id=series_id)
    try:
        with urllib.request.urlopen(url, timeout=60) as response:
            text = response.read().decode("utf-8")
    except (urllib.error.URLError, OSError) as exc:
        raise CarryError(f"Failed to download {url}: {exc}") from exc
    frame = pd.read_csv(io.StringIO(text))
    if frame.shape[1] != 2:
        raise CarryError(f"Unexpected FRED response shape for {series_id}.")
    frame.columns = ["date", "value"]
    frame["date"] = pd.to_datetime(frame["date"])
    series = pd.to_numeric(frame.set_index("date")["value"], errors="coerce").dropna()
    if series.empty:
        raise CarryError(f"FRED series {series_id} contains no numeric values.")
    return series


def augment_cache_with_carry(
    input_path: Path,
    output_path: Path,
    lag_days: int,
    fetch_series: FetchSeriesFn,
) -> pd.DataFrame:
    """Add CarryAnnual columns to a parquet cache.

    Args:
        input_path: Existing cache written by save_ohlcv_parquet.
        output_path: Destination for the augmented cache.
        lag_days: Calendar-day lag applied to rates before alignment
            (publication-delay causality; the caller chooses the margin).
        fetch_series: Function returning a FRED series by id; injectable so
            tests avoid the network.

    Returns:
        The augmented DataFrame that was written.

    Raises:
        CarryError: If lag_days is negative, the cache contract is invalid,
            a symbol lacks a FRED mapping, or the resulting carry is not
            finite over the cache range.
    """
    _validate_lag_days(lag_days)
    frame, timeframe, start, end = load_current_cache(input_path)
    symbols = sorted({symbol for symbol, _ in frame.columns})
    currencies = {code for symbol in symbols for code in split_pair(symbol)}
    unmapped = sorted(currencies - set(CURRENCY_SHORT_RATE_SERIES))
    if unmapped:
        raise CarryError(
            f"No FRED series mapping for currency(ies) {unmapped}; "
            f"supported: {sorted(CURRENCY_SHORT_RATE_SERIES)}."
        )

    calendar = pd.date_range(
        pd.Timestamp(start) - pd.Timedelta(days=2 * lag_days + 400),
        pd.Timestamp(end),
        freq="D",
    )
    cache_days = pd.DatetimeIndex(frame.index).tz_localize(None).normalize()
    source_rates = {
        code: fetch_series(CURRENCY_SHORT_RATE_SERIES[code])
        for code in sorted(currencies)
    }
    rates: dict[str, pd.Series] = {
        code: source_rates[code].reindex(calendar, method="ffill").shift(lag_days)
        for code in sorted(currencies)
    }
    for symbol in symbols:
        base, counter = split_pair(symbol)
        differential = (rates[counter] - rates[base]) * _PERCENT_TO_DECIMAL
        aligned = differential.reindex(cache_days).to_numpy()
        if not pd.notna(aligned).all():
            raise CarryError(
                f"CarryAnnual for {symbol} is not finite over the cache range; "
                f"check series coverage and lag ({lag_days} days)."
            )
        frame[(symbol, "CarryAnnual")] = aligned

    try:
        save_augmented_cache(
            frame,
            timeframe,
            start,
            end,
            input_path,
            output_path,
            "carry",
            lag_days,
            {code: CURRENCY_SHORT_RATE_SERIES[code] for code in sorted(currencies)},
            source_rates,
        )
    except DataLineageError as exc:
        raise CarryError(str(exc)) from exc
    return frame


def main(argv: list[str] | None = None) -> int:
    """CLI entry point.

    Args:
        argv: CLI arguments; None lets argparse read sys.argv (testability).

    Returns:
        Process exit code: 0 on success, 1 on errors.
    """
    parser = argparse.ArgumentParser(
        prog="forex-add-carry",
        description="Add FRED-derived CarryAnnual columns to a parquet cache.",
    )
    parser.add_argument("--input", type=str, required=True, help="Input cache path.")
    parser.add_argument("--output", type=str, required=True, help="Output cache path.")
    parser.add_argument(
        "--lag-days",
        type=int,
        required=True,
        help="Calendar-day lag applied to rates (publication-delay causality).",
    )
    args = parser.parse_args(argv)
    try:
        frame = augment_cache_with_carry(
            Path(args.input), Path(args.output), args.lag_days, fetch_fred_series
        )
    except Exception as exc:  # surface every failure explicitly at the CLI
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"wrote {args.output}: {len(frame.index)} rows with CarryAnnual")
    return 0


if __name__ == "__main__":
    sys.exit(main())
