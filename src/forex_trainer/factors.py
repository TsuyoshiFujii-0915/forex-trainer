"""External-factor augmentation CLI: forex-add-factors.

Adds economically-motivated per-symbol auxiliary fields to an existing
parquet cache, reusing the FRED ingestion pattern from carry.py
(monthly/irregular series forward-filled daily and lagged for causality):

- TermCarryAnnual: 10-year government bond yield differential (counter minus
  JPY, decimal) — a slower, more structural analog of the short-rate
  CarryAnnual in carry.py.
- PppGap: relative-PPP misalignment. Anchored at the first row of the
  INPUT cache (an arbitrary but fixed and fully causal reference point —
  relative PPP has no natural absolute anchor), it is the log price change
  since the anchor minus the log relative-CPI change since the anchor. The
  sign convention is not asserted to mean "over/under-valued" without
  testing both directions empirically (see the project's rule-ceiling
  discipline in docs/research).
- Global series (e.g. VixLevel, OilLevel): broadcast the same value to every
  symbol's column, reusing the env file provider's per-symbol auxiliary
  field passthrough — no env changes are needed for genuinely global data.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Callable, Mapping

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from forex_env.data.file_provider import save_ohlcv_parquet

from .carry import fetch_fred_series

RATE10Y_SERIES: dict[str, str] = {
    "JPY/USD": "IRLTLT01USM156N",
    "JPY/EUR": "IRLTLT01DEM156N",  # German bund: standard euro-area long-rate proxy
    "JPY/GBP": "IRLTLT01GBM156N",
    "JPY/AUD": "IRLTLT01AUM156N",
    "JPY/CHF": "IRLTLT01CHM156N",
    "JPY/CAD": "IRLTLT01CAM156N",
    "JPY/NZD": "IRLTLT01NZM156N",
    "JPY/NOK": "IRLTLT01NOM156N",
    "JPY/SEK": "IRLTLT01SEM156N",
}
JPY_RATE10Y_SERIES = "IRLTLT01JPM156N"

CPI_SERIES: dict[str, str] = {
    "JPY/USD": "USACPIALLMINMEI",
    "JPY/EUR": "CP0000EZ19M086NEST",  # Euro area HICP level
    "JPY/GBP": "GBRCPIALLMINMEI",
    "JPY/AUD": "AUSCPIALLQINMEI",  # quarterly; ffill handles the frequency
    "JPY/CHF": "CHECPIALLMINMEI",
    "JPY/CAD": "CANCPIALLMINMEI",
    "JPY/NZD": "NZLCPIALLQINMEI",  # quarterly
    "JPY/NOK": "NORCPIALLMINMEI",
    "JPY/SEK": "SWECPIALLMINMEI",
}
JPY_CPI_SERIES = "JPNCPIALLMINMEI"

_PERCENT_TO_DECIMAL = 0.01

FetchSeriesFn = Callable[[str], pd.Series]


class FactorError(Exception):
    """Raised when factor augmentation fails."""


def _load_cache(input_path: Path) -> tuple[pd.DataFrame, str, str, str]:
    """Load a parquet cache into a MultiIndex-column frame with its metadata.

    Args:
        input_path: Cache path written by save_ohlcv_parquet.

    Returns:
        Tuple of (frame, timeframe, start_date, end_date).
    """
    table = pq.read_table(input_path)
    metadata = table.schema.metadata or {}
    timeframe = metadata[b"forex_env_timeframe"].decode("utf-8")
    start = metadata[b"forex_env_start_date"].decode("utf-8")
    end = metadata[b"forex_env_end_date"].decode("utf-8")
    frame = table.to_pandas()
    frame.columns = pd.MultiIndex.from_tuples(
        [tuple(column.rsplit("|", 1)) for column in frame.columns]
    )
    return frame, timeframe, start, end


def _daily_calendar(start: str, end: str, lag_days: int) -> pd.DatetimeIndex:
    """Build a daily calendar wide enough to cover the cache range after lagging.

    Args:
        start: Cache start date (YYYY-MM-DD).
        end: Cache end date (YYYY-MM-DD).
        lag_days: Lag that will be applied to series aligned on this calendar.

    Returns:
        Daily DatetimeIndex from well before `start` to `end`.
    """
    return pd.date_range(
        pd.Timestamp(start) - pd.Timedelta(days=2 * lag_days + 400),
        pd.Timestamp(end),
        freq="D",
    )


def add_term_carry(
    input_path: Path,
    output_path: Path,
    lag_days: int,
    fetch_series: FetchSeriesFn,
) -> pd.DataFrame:
    """Add a TermCarryAnnual (10-year rate differential) field per symbol.

    Args:
        input_path: Existing cache written by save_ohlcv_parquet.
        output_path: Destination for the augmented cache.
        lag_days: Calendar-day lag applied to rates before alignment.
        fetch_series: Function returning a FRED series by id; injectable so
            tests avoid the network.

    Returns:
        The augmented DataFrame that was written.

    Raises:
        FactorError: If the cache contains symbols without a mapping or the
            resulting field is not finite over the cache range.
    """
    frame, timeframe, start, end = _load_cache(input_path)
    symbols = sorted({symbol for symbol, _ in frame.columns})
    unmapped = [symbol for symbol in symbols if symbol not in RATE10Y_SERIES]
    if unmapped:
        raise FactorError(
            f"No 10-year rate series mapping for symbol(s) {unmapped}; "
            f"supported: {sorted(RATE10Y_SERIES)}."
        )
    calendar = _daily_calendar(start, end, lag_days)
    jpy_rate = (
        fetch_series(JPY_RATE10Y_SERIES)
        .reindex(calendar, method="ffill")
        .shift(lag_days)
    )
    cache_days = pd.DatetimeIndex(frame.index).tz_localize(None).normalize()
    for symbol in symbols:
        counter_rate = (
            fetch_series(RATE10Y_SERIES[symbol])
            .reindex(calendar, method="ffill")
            .shift(lag_days)
        )
        differential = (counter_rate - jpy_rate) * _PERCENT_TO_DECIMAL
        aligned = differential.reindex(cache_days).to_numpy()
        if not pd.notna(aligned).all():
            raise FactorError(
                f"TermCarryAnnual for {symbol} is not finite over the cache "
                f"range; check series coverage and lag ({lag_days} days)."
            )
        frame[(symbol, "TermCarryAnnual")] = aligned
    save_ohlcv_parquet(frame, timeframe, start, end, output_path)
    return frame


def add_ppp_misalignment(
    input_path: Path,
    output_path: Path,
    lag_days: int,
    fetch_series: FetchSeriesFn,
) -> pd.DataFrame:
    """Add a PppGap (relative-PPP misalignment) field per symbol.

    PppGap(t) = log(Close(t)/Close(t0)) - [log(CPI_counter(t)/CPI_counter(t0))
    - log(CPI_JPY(t)/CPI_JPY(t0))], anchored at the cache's first row (t0).
    The anchor is arbitrary (relative PPP has no natural absolute reference)
    but fixed and fully causal: only data up to each t is used.

    Args:
        input_path: Existing cache written by save_ohlcv_parquet.
        output_path: Destination for the augmented cache.
        lag_days: Calendar-day lag applied to CPI series before alignment
            (statistical-agency publication delay).
        fetch_series: Function returning a FRED series by id; injectable so
            tests avoid the network.

    Returns:
        The augmented DataFrame that was written.

    Raises:
        FactorError: If the cache contains symbols without a mapping or the
            resulting field is not finite over the cache range.
    """
    frame, timeframe, start, end = _load_cache(input_path)
    symbols = sorted({symbol for symbol, _ in frame.columns})
    unmapped = [symbol for symbol in symbols if symbol not in CPI_SERIES]
    if unmapped:
        raise FactorError(
            f"No CPI series mapping for symbol(s) {unmapped}; "
            f"supported: {sorted(CPI_SERIES)}."
        )
    calendar = _daily_calendar(start, end, lag_days)
    jpy_cpi = (
        fetch_series(JPY_CPI_SERIES).reindex(calendar, method="ffill").shift(lag_days)
    )
    cache_days = pd.DatetimeIndex(frame.index).tz_localize(None).normalize()
    for symbol in symbols:
        counter_cpi = (
            fetch_series(CPI_SERIES[symbol])
            .reindex(calendar, method="ffill")
            .shift(lag_days)
        )
        # Raw log CPI differential over the padded calendar, THEN reindexed
        # to the cache's own days and anchored at the cache's first row
        # (not the padding calendar's first row, which is all-NaN after the
        # lag shift).
        raw_log_cpi_diff = np.log(counter_cpi) - np.log(jpy_cpi)
        cpi_diff_aligned = raw_log_cpi_diff.reindex(cache_days)
        cpi_diff_aligned = cpi_diff_aligned - cpi_diff_aligned.iloc[0]
        close = frame[(symbol, "Close")].astype(float)
        log_price_change = np.log(close / close.iloc[0])
        gap = log_price_change.to_numpy() - cpi_diff_aligned.to_numpy()
        if not pd.notna(gap).all():
            raise FactorError(
                f"PppGap for {symbol} is not finite over the cache range; "
                f"check series coverage and lag ({lag_days} days)."
            )
        frame[(symbol, "PppGap")] = gap
    save_ohlcv_parquet(frame, timeframe, start, end, output_path)
    return frame


def add_global_series(
    input_path: Path,
    output_path: Path,
    lag_days: int,
    fetch_series: FetchSeriesFn,
    series_map: Mapping[str, str],
) -> pd.DataFrame:
    """Broadcast global (non-per-symbol) series equally to every symbol.

    Args:
        input_path: Existing cache written by save_ohlcv_parquet.
        output_path: Destination for the augmented cache.
        lag_days: Calendar-day lag applied to the series before alignment.
        fetch_series: Function returning a FRED series by id; injectable so
            tests avoid the network.
        series_map: Mapping from output field name to FRED series id, e.g.
            {"VixLevel": "VIXCLS", "OilLevel": "DCOILWTICO"}.

    Returns:
        The augmented DataFrame that was written.

    Raises:
        FactorError: If a resulting field is not finite over the cache range.
    """
    frame, timeframe, start, end = _load_cache(input_path)
    symbols = sorted({symbol for symbol, _ in frame.columns})
    calendar = _daily_calendar(start, end, lag_days)
    cache_days = pd.DatetimeIndex(frame.index).tz_localize(None).normalize()
    for field_name, series_id in series_map.items():
        series = (
            fetch_series(series_id).reindex(calendar, method="ffill").shift(lag_days)
        )
        aligned = series.reindex(cache_days).to_numpy()
        if not pd.notna(aligned).all():
            raise FactorError(
                f"Global series '{field_name}' ({series_id}) is not finite "
                f"over the cache range; check coverage and lag ({lag_days} days)."
            )
        for symbol in symbols:
            frame[(symbol, field_name)] = aligned
    save_ohlcv_parquet(frame, timeframe, start, end, output_path)
    return frame


def main(argv: list[str] | None = None) -> int:
    """CLI entry point.

    Args:
        argv: CLI arguments; None lets argparse read sys.argv (testability).

    Returns:
        Process exit code: 0 on success, 1 on errors.
    """
    parser = argparse.ArgumentParser(
        prog="forex-add-factors",
        description="Add external-factor fields (term carry, PPP, global series).",
    )
    parser.add_argument("--input", type=str, required=True, help="Input cache path.")
    parser.add_argument("--output", type=str, required=True, help="Output cache path.")
    parser.add_argument(
        "--which",
        type=str,
        required=True,
        choices=("term", "ppp", "global"),
        help="Which factor to add.",
    )
    parser.add_argument(
        "--lag-days",
        type=int,
        required=True,
        help="Calendar-day lag applied to source series (publication delay).",
    )
    parser.add_argument(
        "--global-field",
        type=str,
        default=None,
        help="For --which global: 'FieldName=FRED_SERIES_ID'.",
    )
    args = parser.parse_args(argv)
    try:
        if args.which == "term":
            frame = add_term_carry(
                Path(args.input), Path(args.output), args.lag_days, fetch_fred_series
            )
        elif args.which == "ppp":
            frame = add_ppp_misalignment(
                Path(args.input), Path(args.output), args.lag_days, fetch_fred_series
            )
        else:
            if not args.global_field or "=" not in args.global_field:
                raise FactorError(
                    "--which global requires --global-field 'FieldName=SERIES_ID'."
                )
            field_name, series_id = args.global_field.split("=", 1)
            frame = add_global_series(
                Path(args.input),
                Path(args.output),
                args.lag_days,
                fetch_fred_series,
                {field_name: series_id},
            )
    except Exception as exc:  # surface every failure explicitly at the CLI
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"wrote {args.output}: {len(frame.index)} rows with {args.which} factor(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
