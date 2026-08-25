"""Deterministic repair of isolated multiplicative FX price spikes."""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from forex_env.data.base import validate_ohlcv
from forex_env.data.file_provider import save_ohlcv_parquet

_PRICE_FIELDS: tuple[str, ...] = ("Open", "High", "Low", "Close")
_EXTREME_LEVEL_RATIO = 3.0
_LOCAL_RADIUS = 2


class DataQualityError(Exception):
    """Raised when market data cannot be audited or repaired safely."""


@dataclass(frozen=True)
class SpikeRepair:
    """One repaired timestamp and the affected OHLC fields."""

    symbol: str
    timestamp: str
    fields: tuple[str, ...]


def _read_cache(path: Path) -> tuple[pd.DataFrame, str, str, str]:
    """Read a forex-env parquet cache and its required lineage metadata.

    Args:
        path: Cache path.

    Returns:
        Frame, timeframe, declared start, and declared end.

    Raises:
        DataQualityError: If the file or required metadata is absent.
    """
    if not path.is_file():
        raise DataQualityError(f"Input cache does not exist: {path}")
    table = pq.read_table(path)
    metadata = table.schema.metadata or {}
    required = (
        b"forex_env_timeframe",
        b"forex_env_start_date",
        b"forex_env_end_date",
    )
    missing = [key.decode("utf-8") for key in required if key not in metadata]
    if missing:
        raise DataQualityError(f"Cache {path} lacks metadata keys: {missing}.")
    frame = table.to_pandas()
    frame.columns = pd.MultiIndex.from_tuples(
        [tuple(column.rsplit("|", 1)) for column in frame.columns]
    )
    return (
        frame,
        metadata[b"forex_env_timeframe"].decode("utf-8"),
        metadata[b"forex_env_start_date"].decode("utf-8"),
        metadata[b"forex_env_end_date"].decode("utf-8"),
    )


def _validate_thresholds(
    residual_threshold: float,
    reversal_tolerance: float,
    expected_repairs: int,
) -> None:
    """Validate explicit spike-detection parameters.

    Args:
        residual_threshold: Minimum absolute cross-pair residual log return.
        reversal_tolerance: Maximum two-day residual log-return remainder.
        expected_repairs: Exact number of timestamp rows expected to be repaired.

    Raises:
        DataQualityError: If a parameter is outside its meaningful domain.
    """
    if not math.isfinite(residual_threshold) or residual_threshold <= 0.0:
        raise DataQualityError(
            f"residual_threshold must be finite and positive, got {residual_threshold}."
        )
    if not math.isfinite(reversal_tolerance) or reversal_tolerance <= 0.0:
        raise DataQualityError(
            f"reversal_tolerance must be finite and positive, got {reversal_tolerance}."
        )
    if isinstance(expected_repairs, bool) or expected_repairs < 0:
        raise DataQualityError(
            f"expected_repairs must be a non-negative integer, got {expected_repairs}."
        )


def _local_close_median(close: pd.Series, position: int) -> float:
    """Return the local close median excluding the current observation.

    Args:
        close: Positive close-price series.
        position: Integer row position.

    Returns:
        Median of up to two preceding and two following closes.
    """
    neighbors = pd.concat(
        [
            close.iloc[max(0, position - _LOCAL_RADIUS) : position],
            close.iloc[position + 1 : position + _LOCAL_RADIUS + 1],
        ]
    )
    return float(neighbors.median())


def _detect_repairs(
    frame: pd.DataFrame,
    residual_threshold: float,
    reversal_tolerance: float,
) -> dict[tuple[str, pd.Timestamp], set[str]]:
    """Detect field-level magnitude errors and full-row reversing spikes.

    Args:
        frame: MultiIndex-column OHLCV frame.
        residual_threshold: Minimum absolute cross-pair residual log return.
        reversal_tolerance: Maximum two-day residual remainder.

    Returns:
        Mapping from symbol/timestamp to price fields requiring interpolation.

    Raises:
        DataQualityError: If the frame lacks sufficient pairs or valid prices.
    """
    symbols = tuple(sorted(set(frame.columns.get_level_values(0))))
    if len(symbols) < 3:
        raise DataQualityError(
            f"Cross-sectional spike detection requires at least 3 symbols, got {len(symbols)}."
        )
    missing = [
        (symbol, field)
        for symbol in symbols
        for field in _PRICE_FIELDS
        if (symbol, field) not in frame.columns
    ]
    if missing:
        raise DataQualityError(f"Cache lacks required OHLC fields: {missing}.")
    prices = frame.loc[:, [(symbol, field) for symbol in symbols for field in _PRICE_FIELDS]]
    values = prices.to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values <= 0.0).any():
        raise DataQualityError("OHLC prices must all be finite and positive.")

    repairs: dict[tuple[str, pd.Timestamp], set[str]] = {}
    for symbol in symbols:
        close = frame[(symbol, "Close")]
        for position in range(_LOCAL_RADIUS, len(frame.index) - _LOCAL_RADIUS):
            timestamp = frame.index[position]
            reference = _local_close_median(close, position)
            for field in _PRICE_FIELDS:
                ratio = float(frame.loc[timestamp, (symbol, field)]) / reference
                if ratio > _EXTREME_LEVEL_RATIO or ratio < 1.0 / _EXTREME_LEVEL_RATIO:
                    repairs.setdefault((symbol, timestamp), set()).add(field)

    close_frame = pd.DataFrame(
        {symbol: frame[(symbol, "Close")] for symbol in symbols}, index=frame.index
    )
    log_returns = np.log(close_frame).diff()
    residuals = log_returns.sub(log_returns.median(axis=1), axis=0)
    for symbol in symbols:
        for position in range(1, len(frame.index) - 1):
            current = float(residuals[symbol].iloc[position])
            following = float(residuals[symbol].iloc[position + 1])
            if (
                abs(current) > residual_threshold
                and current * following < 0.0
                and abs(current + following) < reversal_tolerance
            ):
                timestamp = frame.index[position]
                repairs.setdefault((symbol, timestamp), set()).update(_PRICE_FIELDS)
    return repairs


def _interpolated_field_value(
    frame: pd.DataFrame,
    symbol: str,
    field: str,
    position: int,
    repairs: dict[tuple[str, pd.Timestamp], set[str]],
) -> float:
    """Interpolate one bad field from the nearest clean values on both sides.

    Args:
        frame: Original market-data frame.
        symbol: Pair whose field is repaired.
        field: One OHLC field.
        position: Integer position of the bad row.
        repairs: Complete detected repair set.

    Returns:
        Geometric interpolation of the nearest clean neighbors.

    Raises:
        DataQualityError: If a clean neighbor is unavailable on either side.
    """
    previous: float | None = None
    following: float | None = None
    for candidate in range(position - 1, -1, -1):
        key = (symbol, frame.index[candidate])
        if field not in repairs.get(key, set()):
            previous = float(frame.iloc[candidate][(symbol, field)])
            break
    for candidate in range(position + 1, len(frame.index)):
        key = (symbol, frame.index[candidate])
        if field not in repairs.get(key, set()):
            following = float(frame.iloc[candidate][(symbol, field)])
            break
    if previous is None or following is None:
        raise DataQualityError(
            f"Cannot interpolate {symbol} {field} at {frame.index[position]}: "
            "a clean neighbor is missing."
        )
    return math.sqrt(previous * following)


def repair_reversing_spikes(
    input_path: Path,
    output_path: Path,
    residual_threshold: float,
    reversal_tolerance: float,
    expected_repairs: int,
) -> tuple[SpikeRepair, ...]:
    """Repair deterministic FX bad prints and write a lineage-preserving cache.

    Detection combines two signatures: an OHLC field more than 3x away from
    neighboring closes, or a pair-specific close return that reverses almost
    completely on the following bar. Requiring the exact repair count makes a
    provider revision fail instead of silently changing the research dataset.

    Args:
        input_path: Raw forex-env parquet cache.
        output_path: Destination cleaned cache.
        residual_threshold: Minimum pair-residual log return for reversal checks.
        reversal_tolerance: Maximum absolute two-bar residual remainder.
        expected_repairs: Exact timestamp-row count expected from this source.

    Returns:
        Ordered repaired row descriptions.

    Raises:
        DataQualityError: If detection, lineage, or expected-count checks fail.
    """
    _validate_thresholds(residual_threshold, reversal_tolerance, expected_repairs)
    frame, timeframe, start, end = _read_cache(input_path)
    repairs = _detect_repairs(frame, residual_threshold, reversal_tolerance)
    if len(repairs) != expected_repairs:
        raise DataQualityError(
            f"expected {expected_repairs} reversing spikes, found {len(repairs)} "
            f"in {input_path}."
        )

    cleaned = frame.copy()
    ordered: list[SpikeRepair] = []
    positions = {timestamp: position for position, timestamp in enumerate(frame.index)}
    for (symbol, timestamp), fields in sorted(
        repairs.items(), key=lambda item: (item[0][1], item[0][0])
    ):
        position = positions[timestamp]
        ordered_fields = tuple(field for field in _PRICE_FIELDS if field in fields)
        for field in ordered_fields:
            cleaned.loc[timestamp, (symbol, field)] = _interpolated_field_value(
                frame, symbol, field, position, repairs
            )
        ordered.append(
            SpikeRepair(
                symbol=symbol,
                timestamp=pd.Timestamp(timestamp).isoformat(),
                fields=ordered_fields,
            )
        )
    symbols = tuple(sorted(set(cleaned.columns.get_level_values(0))))
    validate_ohlcv(cleaned, symbols)
    save_ohlcv_parquet(cleaned, timeframe, start, end, output_path)
    return tuple(ordered)


def main(argv: list[str] | None = None) -> int:
    """Run the deterministic data-quality repair CLI.

    Args:
        argv: CLI arguments; None lets argparse read sys.argv.

    Returns:
        Process exit code: 0 on success, 1 on data-quality errors.
    """
    parser = argparse.ArgumentParser(
        prog="forex-clean-spikes",
        description="Repair isolated multiplicative FX bad prints.",
    )
    parser.add_argument("--input", type=str, required=True, help="Raw cache path.")
    parser.add_argument("--output", type=str, required=True, help="Clean cache path.")
    parser.add_argument(
        "--residual-threshold",
        type=float,
        required=True,
        help="Minimum absolute pair-residual log return.",
    )
    parser.add_argument(
        "--reversal-tolerance",
        type=float,
        required=True,
        help="Maximum absolute two-bar residual remainder.",
    )
    parser.add_argument(
        "--expected-repairs",
        type=int,
        required=True,
        help="Exact repaired timestamp-row count expected from this source.",
    )
    args = parser.parse_args(argv)
    try:
        repairs = repair_reversing_spikes(
            Path(args.input),
            Path(args.output),
            args.residual_threshold,
            args.reversal_tolerance,
            args.expected_repairs,
        )
    except DataQualityError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(json.dumps([asdict(repair) for repair in repairs], indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
