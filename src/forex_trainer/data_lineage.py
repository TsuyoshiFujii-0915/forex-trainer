"""Immutable lineage metadata for FRED-derived cache transformations."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from forex_env.data.file_provider import save_ohlcv_parquet

_LINEAGE_KEY = b"forex_trainer_data_lineage"
_LINEAGE_VERSION = 1


class DataLineageError(Exception):
    """Raised when cache transformation lineage is malformed."""


def _series_sha256(series: pd.Series, label: str) -> str:
    """Hash one source series using a stable timestamp/value encoding.

    Args:
        series: Raw FRED source series.
        label: Series label for error context.

    Returns:
        Lowercase SHA256 hexadecimal digest.

    Raises:
        DataLineageError: If the series index or values cannot be encoded.
    """
    if not isinstance(series, pd.Series) or not isinstance(
        series.index, pd.DatetimeIndex
    ):
        raise DataLineageError(
            f"Source series '{label}' must be a pandas Series with DatetimeIndex."
        )
    if not series.index.is_unique:
        raise DataLineageError(f"Source series '{label}' has duplicate timestamps.")
    try:
        numeric = pd.to_numeric(series, errors="raise").astype(float).sort_index()
    except (TypeError, ValueError) as exc:
        raise DataLineageError(
            f"Source series '{label}' contains non-numeric values: {exc}"
        ) from exc
    digest = hashlib.sha256()
    for timestamp, value in numeric.items():
        digest.update(pd.Timestamp(timestamp).isoformat().encode("utf-8"))
        digest.update(b"\t")
        digest.update(float(value).hex().encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def read_data_lineage(path: Path) -> list[dict[str, Any]]:
    """Read and validate the ordered transformation history from a cache.

    Absence means the cache has not received a trainer-side FRED transform;
    it does not mean an unknown transformation is accepted.

    Args:
        path: Current parquet cache path.

    Returns:
        Validated transformation records in application order.

    Raises:
        DataLineageError: If the metadata is malformed or unsupported.
    """
    try:
        metadata = dict(pq.read_schema(path).metadata or {})
    except (pa.ArrowException, OSError) as exc:
        raise DataLineageError(f"Failed to read data lineage from {path}: {exc}") from exc
    if _LINEAGE_KEY not in metadata:
        return []
    try:
        payload = json.loads(metadata[_LINEAGE_KEY].decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise DataLineageError(f"Cache {path} has malformed data lineage.") from exc
    if not isinstance(payload, dict) or set(payload) != {"version", "transforms"}:
        raise DataLineageError(f"Cache {path} has malformed data lineage structure.")
    if payload["version"] != _LINEAGE_VERSION:
        raise DataLineageError(
            f"Cache {path} has unsupported data lineage version "
            f"{payload['version']!r}."
        )
    transforms = payload["transforms"]
    if not isinstance(transforms, list) or not all(
        isinstance(item, dict) for item in transforms
    ):
        raise DataLineageError(f"Cache {path} has malformed data lineage transforms.")
    required = {"operation", "lag_days", "series", "source_sha256"}
    for item in transforms:
        if set(item) != required:
            raise DataLineageError(
                f"Cache {path} has malformed data lineage transform keys."
            )
        if not isinstance(item["operation"], str) or not item["operation"]:
            raise DataLineageError(
                f"Cache {path} has malformed data lineage operation."
            )
        if (
            isinstance(item["lag_days"], bool)
            or not isinstance(item["lag_days"], int)
            or item["lag_days"] < 0
        ):
            raise DataLineageError(f"Cache {path} has malformed data lineage lag.")
        if not isinstance(item["series"], dict) or not isinstance(
            item["source_sha256"], dict
        ):
            raise DataLineageError(
                f"Cache {path} has malformed data lineage series mappings."
            )
        if set(item["series"]) != set(item["source_sha256"]):
            raise DataLineageError(
                f"Cache {path} has inconsistent data lineage source hashes."
            )
        if not all(
            isinstance(key, str)
            and isinstance(value, str)
            and key
            and value
            for mapping in (item["series"], item["source_sha256"])
            for key, value in mapping.items()
        ):
            raise DataLineageError(
                f"Cache {path} has malformed data lineage mapping values."
            )
        if not all(
            len(value) == 64 and all(char in "0123456789abcdef" for char in value)
            for value in item["source_sha256"].values()
        ):
            raise DataLineageError(
                f"Cache {path} has malformed data lineage SHA256 digest."
            )
    return transforms


def save_augmented_cache(
    data: pd.DataFrame,
    timeframe: str,
    start_date: str,
    end_date: str,
    input_path: Path,
    output_path: Path,
    operation: str,
    lag_days: int,
    series_map: Mapping[str, str],
    source_series: Mapping[str, pd.Series],
) -> None:
    """Save an augmented cache and append its exact source lineage.

    Args:
        data: Augmented market data.
        timeframe: Cache timeframe.
        start_date: Declared cache start date.
        end_date: Declared cache end date.
        input_path: Source cache whose lineage is inherited.
        output_path: Destination cache.
        operation: Transformation identifier.
        lag_days: Non-negative source publication lag.
        series_map: Output/currency label to FRED series id.
        source_series: Raw fetched series keyed exactly like series_map.

    Raises:
        DataLineageError: If lineage or source series are malformed.
    """
    if not operation:
        raise DataLineageError("Data lineage operation must be non-empty.")
    if lag_days < 0:
        raise DataLineageError(f"Data lineage lag_days must be non-negative: {lag_days}.")
    if set(series_map) != set(source_series):
        raise DataLineageError(
            "Data lineage series_map and source_series keys must match exactly."
        )
    transforms = read_data_lineage(input_path)
    ordered_series = {key: str(series_map[key]) for key in sorted(series_map)}
    hashes = {
        key: _series_sha256(source_series[key], ordered_series[key])
        for key in ordered_series
    }
    transforms.append(
        {
            "operation": operation,
            "lag_days": lag_days,
            "series": ordered_series,
            "source_sha256": hashes,
        }
    )
    save_ohlcv_parquet(data, timeframe, start_date, end_date, output_path)
    table = pq.read_table(output_path)
    metadata = dict(table.schema.metadata or {})
    metadata[_LINEAGE_KEY] = json.dumps(
        {"version": _LINEAGE_VERSION, "transforms": transforms},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    pq.write_table(table.replace_schema_metadata(metadata), output_path)
