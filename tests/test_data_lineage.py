"""Behavior tests for immutable FRED transformation lineage."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pytest
from forex_env.data.file_provider import save_ohlcv_parquet
from forex_env.data.synthetic import SyntheticDataProvider

from forex_trainer.carry import CarryError, augment_cache_with_carry
from forex_trainer.factors import add_global_series

_LINEAGE_KEY = b"forex_trainer_data_lineage"
_SYMBOLS = ("JPY/USD", "JPY/EUR")


def _base_cache(tmp_path: Path) -> Path:
    """Write a current OHLCV-only cache for lineage tests."""
    data = SyntheticDataProvider(seed=7).get_data(
        _SYMBOLS, "2020-01-01", "2020-02-01", "1d"
    )
    data = data.drop(columns=[(symbol, "CarryAnnual") for symbol in _SYMBOLS])
    path = tmp_path / "base.parquet"
    save_ohlcv_parquet(data, "1d", "2020-01-01", "2020-02-01", path)
    return path


def _series(series_id: str) -> pd.Series:
    """Return a deterministic source series whose values depend on its id."""
    index = pd.date_range("2018-01-01", "2020-03-01", freq="D")
    offset = float(sum(series_id.encode("utf-8")) % 17)
    return pd.Series(np.linspace(offset, offset + 1.0, len(index)), index=index)


def test_augmentations_append_fred_series_lag_and_source_hashes(
    tmp_path: Path,
) -> None:
    """Every semantic transformation leaves an ordered immutable lineage."""
    carry_path = tmp_path / "carry.parquet"
    output_path = tmp_path / "global.parquet"
    augment_cache_with_carry(
        _base_cache(tmp_path), carry_path, lag_days=30, fetch_series=_series
    )
    add_global_series(
        carry_path,
        output_path,
        lag_days=1,
        fetch_series=_series,
        series_map={"VixLevel": "VIXCLS"},
    )

    metadata = dict(pq.read_schema(output_path).metadata or {})
    lineage = json.loads(metadata[_LINEAGE_KEY].decode("utf-8"))

    assert lineage["version"] == 1
    assert [item["operation"] for item in lineage["transforms"]] == [
        "carry",
        "global",
    ]
    assert lineage["transforms"][0]["lag_days"] == 30
    assert lineage["transforms"][0]["series"] == {
        "EUR": "ECBDFR",
        "JPY": "IR3TIB01JPM156N",
        "USD": "FEDFUNDS",
    }
    assert lineage["transforms"][1]["series"] == {"VixLevel": "VIXCLS"}
    for transform in lineage["transforms"]:
        assert set(transform["source_sha256"]) == set(transform["series"])
        assert all(
            len(digest) == 64 for digest in transform["source_sha256"].values()
        )


def test_malformed_existing_lineage_is_rejected_before_augmentation(
    tmp_path: Path,
) -> None:
    """A corrupted lineage cannot be silently replaced by a new history."""
    source = _base_cache(tmp_path)
    table = pq.read_table(source)
    metadata = dict(table.schema.metadata or {})
    metadata[_LINEAGE_KEY] = b"not-json"
    pq.write_table(table.replace_schema_metadata(metadata), source)

    with pytest.raises(CarryError, match="data lineage"):
        augment_cache_with_carry(
            source,
            tmp_path / "carry.parquet",
            lag_days=30,
            fetch_series=_series,
        )
