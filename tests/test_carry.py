"""Tests for the FRED carry augmentation (ADR-0010 / env ADR-0012)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pytest
from forex_env import ForexEnv, parse_config
from forex_env.data.file_provider import FileDataProvider, save_ohlcv_parquet
from forex_env.data.synthetic import SyntheticDataProvider

from forex_trainer.carry import CarryError, augment_cache_with_carry, main

_SYMBOLS = ("JPY/USD", "JPY/EUR")


def _base_cache(tmp_path: Path) -> Path:
    """Write a synthetic two-pair cache without CarryAnnual.

    Args:
        tmp_path: Test-scoped directory.

    Returns:
        Path to the cache parquet.
    """
    data = SyntheticDataProvider(seed=7).get_data(
        _SYMBOLS, "2020-01-01", "2020-03-01", "1h"
    )
    data = data[[column for column in data.columns if column[1] != "CarryAnnual"]]
    path = tmp_path / "base.parquet"
    save_ohlcv_parquet(data, "1h", "2020-01-01", "2020-03-01", path)
    return path


def _fake_series(series_id: str) -> pd.Series:
    """Deterministic monthly rate series per FRED id (percent units)."""
    index = pd.date_range("2019-01-01", "2020-04-01", freq="MS")
    base = {"FEDFUNDS": 2.0, "ECBDFR": -0.5, "IR3TIB01JPM156N": 0.1}[series_id]
    return pd.Series(base, index=index)


def _replace_cache_metadata(path: Path, updates: dict[bytes, bytes | None]) -> None:
    """Replace selected parquet schema metadata values.

    Args:
        path: Cache parquet to modify.
        updates: Metadata values to set, or None to remove a key.
    """
    table = pq.read_table(path)
    metadata = dict(table.schema.metadata or {})
    for key, value in updates.items():
        if value is None:
            metadata.pop(key, None)
        else:
            metadata[key] = value
    pq.write_table(table.replace_schema_metadata(metadata), path)


def test_augment_adds_lagged_carry(tmp_path: Path) -> None:
    """CarryAnnual equals the decimal rate differential for every symbol."""
    base = _base_cache(tmp_path)
    output = tmp_path / "carry.parquet"
    augment_cache_with_carry(base, output, lag_days=30, fetch_series=_fake_series)
    loaded = FileDataProvider(str(output)).get_data(
        _SYMBOLS, "2020-01-01", "2020-03-01", "1h"
    )
    usd = loaded[("JPY/USD", "CarryAnnual")]
    eur = loaded[("JPY/EUR", "CarryAnnual")]
    assert usd.iloc[-1] == pytest.approx((2.0 - 0.1) / 100.0)
    assert eur.iloc[-1] == pytest.approx((-0.5 - 0.1) / 100.0)
    assert np.isfinite(usd).all() and np.isfinite(eur).all()


def test_carry_augmentation_rejects_legacy_cache_without_schema(
    tmp_path: Path,
) -> None:
    """Carry regeneration must start from an explicitly current cache."""
    base = _base_cache(tmp_path)
    _replace_cache_metadata(
        base,
        {
            b"forex_env_cache_schema_version": None,
            b"forex_env_carry_contract": None,
        },
    )

    with pytest.raises(CarryError, match="lacks forex-env schema metadata"):
        augment_cache_with_carry(
            base,
            tmp_path / "carry.parquet",
            lag_days=30,
            fetch_series=_fake_series,
        )


def test_carry_augmentation_rejects_unsupported_carry_contract(
    tmp_path: Path,
) -> None:
    """Carry regeneration must not accept unknown input semantics."""
    base = _base_cache(tmp_path)
    _replace_cache_metadata(
        base,
        {b"forex_env_carry_contract": b"counter-minus-jpy-v1"},
    )

    with pytest.raises(CarryError, match="unsupported carry contract"):
        augment_cache_with_carry(
            base,
            tmp_path / "carry.parquet",
            lag_days=30,
            fetch_series=_fake_series,
        )


def test_carry_augmentation_rejects_negative_lag_days(tmp_path: Path) -> None:
    """A negative lag would expose future short-rate observations."""
    base = _base_cache(tmp_path)
    with pytest.raises(CarryError, match="lag_days must be non-negative"):
        augment_cache_with_carry(
            base,
            tmp_path / "carry.parquet",
            lag_days=-1,
            fetch_series=_fake_series,
        )


def test_carry_cli_rejects_negative_lag_days(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The CLI must reject lookahead lags before accessing FRED."""
    base = _base_cache(tmp_path)
    result = main(
        [
            "--input",
            str(base),
            "--output",
            str(tmp_path / "carry.parquet"),
            "--lag-days",
            "-1",
        ]
    )

    assert result == 1
    assert "lag_days must be non-negative" in capsys.readouterr().err


def test_augmented_cache_runs_signed_env(tmp_path: Path) -> None:
    """A signed-mode env consumes the augmented cache end to end."""
    base = _base_cache(tmp_path)
    output = tmp_path / "carry.parquet"
    augment_cache_with_carry(base, output, lag_days=30, fetch_series=_fake_series)
    raw = {
        "environment": {
            "seed": 5,
            "initial_balance_jpy": 1_000_000.0,
            "episode_max_steps": 16,
            "window_size": 8,
            "max_leverage": 5.0,
            "margin_call_threshold": 0.2,
            "allow_action_leverage": False,
            "random_start": False,
            "currency_pairs": list(_SYMBOLS),
        },
        "data": {
            "provider": "file",
            "start_date": "2020-01-01",
            "end_date": "2020-03-01",
            "timeframe": "1h",
            "path": str(output),
        },
        "features": {
            "volatility_window": 8,
            "normalize": False,
            "selected": ["log_return", "carry_annual"],
        },
        "transaction_costs": {
            "commission_rate": 0.0,
            "overnight_rate": 0.0,
            "carry_mode": "signed",
            "spreads": {s: 0.0 for s in _SYMBOLS},
        },
    }
    from forex_trainer.features import FEATURE_REGISTRY

    env = ForexEnv(
        parse_config(raw),
        custom_features={"carry_annual": FEATURE_REGISTRY["carry_annual"]},
    )
    _, _ = env.reset(seed=0)
    action = np.array([[1.0, 1.0], [0.0, 1.0]], dtype=np.float32)
    _, _, _, _, info = env.step(action)
    # Long JPY/USD with USD rate 1.9% above JPY: financing is a payment.
    assert info["financing_jpy"] < 0.0


def test_unmapped_currency_rejected(tmp_path: Path) -> None:
    """Caches containing currencies without a FRED mapping must fail fast."""
    data = SyntheticDataProvider(seed=8).get_data(
        ("JPY/SGD",), "2020-01-01", "2020-02-01", "1h"
    )
    data = data[[column for column in data.columns if column[1] != "CarryAnnual"]]
    path = tmp_path / "sgd.parquet"
    save_ohlcv_parquet(data, "1h", "2020-01-01", "2020-02-01", path)
    with pytest.raises(CarryError, match="SGD"):
        augment_cache_with_carry(
            path, tmp_path / "out.parquet", lag_days=30, fetch_series=_fake_series
        )


def test_augment_generalizes_to_non_jpy_base(tmp_path: Path) -> None:
    """A USD-denominated pair (env ADR-0010) resolves via the currency-code
    mapping, not a JPY-only assumption."""
    symbols = ("USD/JPY", "USD/EUR")
    data = SyntheticDataProvider(seed=13).get_data(
        symbols, "2020-01-01", "2020-03-01", "1h"
    )
    data = data[[column for column in data.columns if column[1] != "CarryAnnual"]]
    path = tmp_path / "usd_base.parquet"
    save_ohlcv_parquet(data, "1h", "2020-01-01", "2020-03-01", path)
    output = tmp_path / "usd_carry.parquet"
    augment_cache_with_carry(path, output, lag_days=30, fetch_series=_fake_series)
    loaded = FileDataProvider(str(output)).get_data(
        symbols, "2020-01-01", "2020-03-01", "1h"
    )
    jpy_leg = loaded[("USD/JPY", "CarryAnnual")]
    eur_leg = loaded[("USD/EUR", "CarryAnnual")]
    # USD/JPY: counter(JPY)=0.1 - base(USD)=2.0; USD/EUR: counter(EUR)=-0.5 - base(USD)=2.0.
    assert jpy_leg.iloc[-1] == pytest.approx((0.1 - 2.0) / 100.0)
    assert eur_leg.iloc[-1] == pytest.approx((-0.5 - 2.0) / 100.0)
