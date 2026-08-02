"""Tests for the external-factor augmentation (term rates, PPP, global series)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pytest
from forex_env.data.file_provider import FileDataProvider, save_ohlcv_parquet
from forex_env.data.synthetic import SyntheticDataProvider

from forex_trainer.factors import (
    CURRENCY_CPI_SERIES,
    CURRENCY_RATE10Y_SERIES,
    FactorError,
    FetchSeriesFn,
    add_global_series,
    add_ppp_misalignment,
    add_term_carry,
    main,
)

_SYMBOLS = ("JPY/USD", "JPY/EUR")


def _base_cache(tmp_path: Path) -> Path:
    """Write a synthetic two-pair cache without any factor fields."""
    data = SyntheticDataProvider(seed=9).get_data(
        _SYMBOLS, "2020-01-01", "2020-03-01", "1h"
    )
    data = data[[column for column in data.columns if column[1] != "CarryAnnual"]]
    path = tmp_path / "base.parquet"
    save_ohlcv_parquet(data, "1h", "2020-01-01", "2020-03-01", path)
    return path


def _flat_series(value: float) -> FetchSeriesFn:
    levels = {
        CURRENCY_RATE10Y_SERIES["USD"]: 2.0,
        CURRENCY_RATE10Y_SERIES["EUR"]: 0.5,
        CURRENCY_RATE10Y_SERIES["JPY"]: 0.1,
    }

    def fetch(series_id: str) -> pd.Series:
        base = levels.get(series_id, value)
        index = pd.date_range("2019-01-01", "2020-04-01", freq="MS")
        return pd.Series(base, index=index)

    return fetch


def _cpi_series() -> FetchSeriesFn:
    rates = {
        CURRENCY_CPI_SERIES["USD"]: 0.002,
        CURRENCY_CPI_SERIES["EUR"]: 0.0005,
        CURRENCY_CPI_SERIES["JPY"]: 0.0001,
    }

    def fetch(series_id: str) -> pd.Series:
        index = pd.date_range("2019-01-01", "2020-04-01", freq="MS")
        n = len(index)
        # Distinct, monotonically rising index levels per series id so the
        # relative growth rates differ across symbols.
        rate = rates[series_id]
        return pd.Series(100.0 * (1.0 + rate) ** np.arange(n), index=index)

    return fetch


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


def test_add_term_carry_matches_expected_differential(tmp_path: Path) -> None:
    base = _base_cache(tmp_path)
    output = tmp_path / "term.parquet"
    add_term_carry(base, output, lag_days=30, fetch_series=_flat_series(0.0))
    loaded = FileDataProvider(str(output)).get_data(
        _SYMBOLS, "2020-01-01", "2020-03-01", "1h"
    )
    usd = loaded[("JPY/USD", "TermCarryAnnual")]
    eur = loaded[("JPY/EUR", "TermCarryAnnual")]
    assert usd.iloc[-1] == pytest.approx((2.0 - 0.1) / 100.0)
    assert eur.iloc[-1] == pytest.approx((0.5 - 0.1) / 100.0)
    assert np.isfinite(usd).all() and np.isfinite(eur).all()


def test_add_ppp_misalignment_is_zero_at_anchor(tmp_path: Path) -> None:
    base = _base_cache(tmp_path)
    output = tmp_path / "ppp.parquet"
    add_ppp_misalignment(base, output, lag_days=30, fetch_series=_cpi_series())
    loaded = FileDataProvider(str(output)).get_data(
        _SYMBOLS, "2020-01-01", "2020-03-01", "1h"
    )
    usd = loaded[("JPY/USD", "PppGap")]
    eur = loaded[("JPY/EUR", "PppGap")]
    # Anchored at the first cache row: the gap must start at exactly zero.
    assert usd.iloc[0] == pytest.approx(0.0, abs=1e-9)
    assert eur.iloc[0] == pytest.approx(0.0, abs=1e-9)
    assert np.isfinite(usd).all() and np.isfinite(eur).all()
    # The two symbols reference different CPI series, so the paths diverge.
    assert not np.allclose(usd.to_numpy(), eur.to_numpy())


def test_add_global_series_broadcasts_equally(tmp_path: Path) -> None:
    base = _base_cache(tmp_path)
    output = tmp_path / "global.parquet"

    def fetch(series_id: str) -> pd.Series:
        index = pd.date_range("2019-01-01", "2020-04-01", freq="D")
        return pd.Series(np.linspace(10.0, 20.0, len(index)), index=index)

    add_global_series(
        base, output, lag_days=1, fetch_series=fetch, series_map={"VixLevel": "VIXCLS"}
    )
    loaded = FileDataProvider(str(output)).get_data(
        _SYMBOLS, "2020-01-01", "2020-03-01", "1h"
    )
    usd = loaded[("JPY/USD", "VixLevel")]
    eur = loaded[("JPY/EUR", "VixLevel")]
    np.testing.assert_allclose(usd.to_numpy(), eur.to_numpy())
    assert np.isfinite(usd).all()


def test_factor_augmentation_rejects_legacy_cache_without_schema(
    tmp_path: Path,
) -> None:
    """Factor augmentation must not bless an unversioned cache as current."""
    base = _base_cache(tmp_path)
    _replace_cache_metadata(
        base,
        {
            b"forex_env_cache_schema_version": None,
            b"forex_env_carry_contract": None,
        },
    )

    with pytest.raises(FactorError, match="lacks forex-env schema metadata"):
        add_term_carry(
            base,
            tmp_path / "term.parquet",
            lag_days=30,
            fetch_series=_flat_series(0.0),
        )


def test_factor_augmentation_rejects_unsupported_carry_contract(
    tmp_path: Path,
) -> None:
    """Existing CarryAnnual semantics must be explicit before preservation."""
    base = _base_cache(tmp_path)
    _replace_cache_metadata(
        base,
        {b"forex_env_carry_contract": b"counter-minus-jpy-v1"},
    )

    with pytest.raises(FactorError, match="unsupported carry contract"):
        add_global_series(
            base,
            tmp_path / "global.parquet",
            lag_days=1,
            fetch_series=_flat_series(0.0),
            series_map={"VixLevel": "VIXCLS"},
        )


def test_add_term_carry_rejects_negative_lag_days(tmp_path: Path) -> None:
    """A negative lag would expose future rate observations."""
    base = _base_cache(tmp_path)
    with pytest.raises(FactorError, match="lag_days must be non-negative"):
        add_term_carry(
            base,
            tmp_path / "term.parquet",
            lag_days=-1,
            fetch_series=_flat_series(0.0),
        )


def test_add_ppp_misalignment_rejects_negative_lag_days(tmp_path: Path) -> None:
    """A negative lag would expose future CPI observations."""
    base = _base_cache(tmp_path)
    with pytest.raises(FactorError, match="lag_days must be non-negative"):
        add_ppp_misalignment(
            base,
            tmp_path / "ppp.parquet",
            lag_days=-1,
            fetch_series=_cpi_series(),
        )


def test_add_global_series_rejects_negative_lag_days(tmp_path: Path) -> None:
    """A negative lag would expose future global observations."""
    base = _base_cache(tmp_path)
    with pytest.raises(FactorError, match="lag_days must be non-negative"):
        add_global_series(
            base,
            tmp_path / "global.parquet",
            lag_days=-1,
            fetch_series=_flat_series(0.0),
            series_map={"VixLevel": "VIXCLS"},
        )


def test_factor_cli_rejects_negative_lag_days(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """The CLI must reject lookahead lags before accessing FRED."""
    base = _base_cache(tmp_path)
    result = main(
        [
            "--input",
            str(base),
            "--output",
            str(tmp_path / "term.parquet"),
            "--which",
            "term",
            "--lag-days",
            "-1",
        ]
    )

    assert result == 1
    assert "lag_days must be non-negative" in capsys.readouterr().err


def test_unmapped_currency_rejected_for_term_carry(tmp_path: Path) -> None:
    data = SyntheticDataProvider(seed=3).get_data(
        ("JPY/SGD",), "2020-01-01", "2020-02-01", "1h"
    )
    data = data[[column for column in data.columns if column[1] != "CarryAnnual"]]
    path = tmp_path / "sgd.parquet"
    save_ohlcv_parquet(data, "1h", "2020-01-01", "2020-02-01", path)
    with pytest.raises(FactorError, match="SGD"):
        add_term_carry(
            path, tmp_path / "out.parquet", lag_days=30, fetch_series=_flat_series(0.0)
        )


def test_add_term_carry_generalizes_to_non_jpy_base(tmp_path: Path) -> None:
    """A USD-denominated pair (env ADR-0010) resolves via the same currency-
    code-keyed mapping, not a JPY-only assumption."""
    symbols = ("USD/JPY", "USD/EUR")
    data = SyntheticDataProvider(seed=12).get_data(
        symbols, "2020-01-01", "2020-03-01", "1h"
    )
    data = data[[column for column in data.columns if column[1] != "CarryAnnual"]]
    path = tmp_path / "usd_base.parquet"
    save_ohlcv_parquet(data, "1h", "2020-01-01", "2020-03-01", path)
    output = tmp_path / "usd_term.parquet"
    add_term_carry(path, output, lag_days=30, fetch_series=_flat_series(0.0))
    loaded = FileDataProvider(str(output)).get_data(
        symbols, "2020-01-01", "2020-03-01", "1h"
    )
    jpy_leg = loaded[("USD/JPY", "TermCarryAnnual")]
    eur_leg = loaded[("USD/EUR", "TermCarryAnnual")]
    # USD/JPY: counter(JPY)=0.1 - base(USD)=2.0; USD/EUR: counter(EUR)=0.5 - base(USD)=2.0.
    assert jpy_leg.iloc[-1] == pytest.approx((0.1 - 2.0) / 100.0)
    assert eur_leg.iloc[-1] == pytest.approx((0.5 - 2.0) / 100.0)
