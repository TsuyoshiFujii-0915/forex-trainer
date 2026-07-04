"""Tests for the FRED carry augmentation (ADR-0008 / env ADR-0009)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from forex_env import ForexEnv, parse_config
from forex_env.data.file_provider import FileDataProvider, save_ohlcv_parquet
from forex_env.data.synthetic import SyntheticDataProvider

from forex_trainer.carry import CarryError, augment_cache_with_carry

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


def test_unmapped_symbol_rejected(tmp_path: Path) -> None:
    """Caches containing pairs without a FRED mapping must fail fast."""
    data = SyntheticDataProvider(seed=8).get_data(
        ("JPY/SGD",), "2020-01-01", "2020-02-01", "1h"
    )
    data = data[[column for column in data.columns if column[1] != "CarryAnnual"]]
    path = tmp_path / "sgd.parquet"
    save_ohlcv_parquet(data, "1h", "2020-01-01", "2020-02-01", path)
    with pytest.raises(CarryError, match="JPY/SGD"):
        augment_cache_with_carry(
            path, tmp_path / "out.parquet", lag_days=30, fetch_series=_fake_series
        )
