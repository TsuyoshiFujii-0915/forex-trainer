"""Attribution input preflight uses real environment data without replay."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from forex_env.data.file_provider import save_ohlcv_parquet

from forex_trainer.input_recovery import preflight_train_input
from test_full_period import config_raw, market_calendar


def raw_fixture(tmp_path: Path, bars: int) -> dict[str, Any]:
    """Write a deterministic source with the frozen feature geometry."""
    market, _ = market_calendar('2023-01-02', bars)
    cache = tmp_path / 'market.parquet'
    save_ohlcv_parquet(market, '1d', '2023-01-01', '2025-01-01', cache)
    raw = config_raw()
    raw['env']['data']['path'] = str(cache)
    raw['train_range'] = {'start': '2023-01-01', 'end': '2024-01-01'}
    raw['val_range'] = {'start': '2024-01-01', 'end': '2024-07-01'}
    raw['eval_range'] = {'start': '2024-07-01', 'end': '2025-01-01'}
    return raw


def test_train_input_is_read_and_reset_without_policy_inference(tmp_path: Path) -> None:
    result = preflight_train_input(raw_fixture(tmp_path, 100))
    assert result['status'] == 'input_ready'
    assert result['available_bars'] == 100
    assert result['inference_calls'] == 0
    assert result['initial_assets_zero'] is True


def test_train_shortage_remains_explicit(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match='65 bars'):
        preflight_train_input(raw_fixture(tmp_path, 64))
