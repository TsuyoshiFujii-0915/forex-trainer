"""Tests for the Dukascopy hourly cache builder (ADR-0006)."""

from __future__ import annotations

import lzma
import struct
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pytest
from forex_env.data.file_provider import FileDataProvider

from forex_trainer.dukascopy import (
    DukascopyError,
    build_cache,
    decode_hour_candles,
    to_dukascopy_instrument,
)

_PRICE_SCALE = 1e-3


def _encode_candles(records: list[tuple[int, int, int, int, int, float]]) -> bytes:
    """LZMA-encode candle records in the Dukascopy bi5 layout.

    Args:
        records: Tuples of (delta_seconds, open, close, low, high, volume).

    Returns:
        Compressed bytes as served by datafeed.dukascopy.com.
    """
    raw = b"".join(struct.pack(">5if", *record) for record in records)
    return lzma.compress(raw)


def test_decode_hour_candles_drops_inactive_bars() -> None:
    """Decoding maps fields correctly and drops volume<=0 (non-trading) bars."""
    month_start = datetime(2020, 1, 1, tzinfo=timezone.utc)
    payload = _encode_candles(
        [
            (0, 108631, 108631, 108631, 108631, 0.0),  # holiday bar
            (3600, 108700, 108650, 108600, 108750, 1200.5),
            (7200, 108650, 108800, 108640, 108820, 900.0),
        ]
    )
    frame = decode_hour_candles(payload, month_start, _PRICE_SCALE)
    assert list(frame.columns) == ["Open", "High", "Low", "Close", "Volume"]
    assert len(frame) == 2
    assert frame.index[0] == month_start + pd.Timedelta(hours=1)
    assert frame["Open"].iloc[0] == pytest.approx(108.700)
    assert frame["Close"].iloc[0] == pytest.approx(108.650)
    assert frame["Low"].iloc[0] == pytest.approx(108.600)
    assert frame["High"].iloc[0] == pytest.approx(108.750)


def test_decode_rejects_malformed_payload() -> None:
    """Payloads whose size is not a multiple of the record size must fail."""
    month_start = datetime(2020, 1, 1, tzinfo=timezone.utc)
    with pytest.raises(DukascopyError, match="record size"):
        decode_hour_candles(lzma.compress(b"\x00" * 10), month_start, _PRICE_SCALE)


def test_to_dukascopy_instrument() -> None:
    """JPY-based env symbols map to Dukascopy instrument names."""
    assert to_dukascopy_instrument("JPY/USD") == "USDJPY"
    assert to_dukascopy_instrument("JPY/EUR") == "EURJPY"


def test_build_cache_produces_file_provider_compatible_parquet(
    tmp_path: Path,
) -> None:
    """The built parquet round-trips through the env's file provider."""

    def fake_fetch(instrument: str, year: int, month: int) -> bytes:
        base = 108000 if instrument == "USDJPY" else 118000
        records = []
        for hour in range(24 * 28):
            price = base + hour
            records.append(
                (hour * 3600, price, price + 20, price - 30, price + 40, 10.0)
            )
        return _encode_candles(records)

    output = tmp_path / "cache.parquet"
    build_cache(
        symbols=("JPY/USD", "JPY/EUR"),
        start="2020-01-01",
        end="2020-02-28",
        output=output,
        fetch_month=fake_fetch,
    )
    provider = FileDataProvider(str(output))
    data = provider.get_data(("JPY/USD", "JPY/EUR"), "2020-01-01", "2020-02-28", "1h")
    assert set(symbol for symbol, _ in data.columns) == {"JPY/USD", "JPY/EUR"}
    # Inversion: JPY/USD Close = 1 / (USDJPY Close), High from original Low.
    first_close = data[("JPY/USD", "Close")].iloc[0]
    assert first_close == pytest.approx(1.0 / 108.020)
    first_high = data[("JPY/USD", "High")].iloc[0]
    assert first_high == pytest.approx(1.0 / 107.970)
    assert (data[("JPY/USD", "High")] >= data[("JPY/USD", "Low")]).all()
