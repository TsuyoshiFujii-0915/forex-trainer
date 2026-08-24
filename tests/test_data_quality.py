"""Behavior tests for deterministic market-data spike repair."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pytest
from forex_env.data.file_provider import save_ohlcv_parquet

from forex_trainer.data_quality import DataQualityError, repair_reversing_spikes


_PAIRS = ("JPY/USD", "JPY/EUR", "JPY/GBP")


def _write_cache(path: Path, closes: dict[str, list[float]]) -> None:
    """Write a small multi-pair cache with internally valid OHLC values.

    Args:
        path: Destination parquet path.
        closes: Close-price values keyed by pair.
    """
    index = pd.date_range("2020-01-01", periods=6, freq="D", tz="UTC")
    columns: dict[tuple[str, str], np.ndarray] = {}
    for pair in _PAIRS:
        close = np.asarray(closes[pair], dtype=float)
        columns[(pair, "Open")] = close * 0.999
        columns[(pair, "High")] = close * 1.001
        columns[(pair, "Low")] = close * 0.998
        columns[(pair, "Close")] = close
        columns[(pair, "Volume")] = np.ones(len(index), dtype=float)
    frame = pd.DataFrame(columns, index=index)
    save_ohlcv_parquet(frame, "1d", "2020-01-01", "2020-01-06", path)


def _closes(path: Path) -> pd.DataFrame:
    """Read close columns from a forex-env parquet cache.

    Args:
        path: Cache path.

    Returns:
        Close-price frame keyed by pair.
    """
    frame = pq.read_table(path).to_pandas()
    frame.columns = pd.MultiIndex.from_tuples(
        [tuple(column.rsplit("|", 1)) for column in frame.columns]
    )
    return frame.xs("Close", axis=1, level=1)


def test_repair_reversing_spikes_interpolates_only_the_bad_bar(
    tmp_path: Path,
) -> None:
    """An isolated multiplicative spike is repaired without changing other bars."""
    source = tmp_path / "source.parquet"
    output = tmp_path / "clean.parquet"
    stable = [1.00, 1.01, 1.02, 1.03, 1.04, 1.05]
    _write_cache(
        source,
        {
            "JPY/USD": [1.00, 1.01, 10.2, 1.03, 1.04, 1.05],
            "JPY/EUR": stable,
            "JPY/GBP": stable,
        },
    )

    repairs = repair_reversing_spikes(
        source,
        output,
        residual_threshold=0.08,
        reversal_tolerance=0.04,
        expected_repairs=1,
    )

    assert [(repair.symbol, repair.timestamp[:10]) for repair in repairs] == [
        ("JPY/USD", "2020-01-03")
    ]
    before = _closes(source)
    after = _closes(output)
    expected = float(np.sqrt(before["JPY/USD"].iloc[1] * before["JPY/USD"].iloc[3]))
    assert after["JPY/USD"].iloc[2] == pytest.approx(expected)
    pd.testing.assert_series_equal(
        after["JPY/USD"].drop(after.index[2]),
        before["JPY/USD"].drop(before.index[2]),
    )
    assert pq.read_table(output).schema.metadata == pq.read_table(source).schema.metadata


def test_repair_preserves_large_move_that_does_not_reverse(tmp_path: Path) -> None:
    """A persistent large market move is not classified as a bad print."""
    source = tmp_path / "source.parquet"
    output = tmp_path / "clean.parquet"
    stable = [1.00, 1.01, 1.02, 1.03, 1.04, 1.05]
    _write_cache(
        source,
        {
            "JPY/USD": [1.00, 1.01, 1.80, 1.82, 1.84, 1.86],
            "JPY/EUR": stable,
            "JPY/GBP": stable,
        },
    )

    repairs = repair_reversing_spikes(
        source,
        output,
        residual_threshold=0.08,
        reversal_tolerance=0.04,
        expected_repairs=0,
    )

    assert repairs == ()
    pd.testing.assert_frame_equal(_closes(output), _closes(source))


def test_repair_rejects_changed_anomaly_count(tmp_path: Path) -> None:
    """A source-data revision cannot silently change the repair set."""
    source = tmp_path / "source.parquet"
    stable = [1.00, 1.01, 1.02, 1.03, 1.04, 1.05]
    _write_cache(
        source,
        {
            "JPY/USD": [1.00, 1.01, 10.2, 1.03, 1.04, 1.05],
            "JPY/EUR": stable,
            "JPY/GBP": stable,
        },
    )

    with pytest.raises(DataQualityError, match="expected 2 reversing spikes, found 1"):
        repair_reversing_spikes(
            source,
            tmp_path / "clean.parquet",
            residual_threshold=0.08,
            reversal_tolerance=0.04,
            expected_repairs=2,
        )
