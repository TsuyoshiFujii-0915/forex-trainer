"""Behavioral checks for bounded historical input recovery."""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from forex_trainer.input_recovery import audit_missing_pairs, select_contiguous_range, write_immutable_json
from forex_trainer.full_period import prepare_period
from test_full_period import market_calendar


def bounds(calendar: dict[str, Any], first: int) -> tuple[str, str]:
    """Return explicit measurement bounds for fixture sessions."""
    return calendar['sessions'][first]['session_close'], (pd.Timestamp(calendar['sessions'][-1]['session_close']) + pd.Timedelta(seconds=1)).isoformat()


def test_missing_holiday_is_unknown_for_every_pair_and_includes_history() -> None:
    market, calendar = market_calendar('2024-09-01', 110)
    missing = market.index[60]
    market = market.drop(missing)
    start, end = bounds(calendar, 63)
    rows = audit_missing_pairs(market, calendar, start, end)
    assert len(rows) == 9
    assert {r['pair'] for r in rows} == set(calendar['symbols'])
    assert {r['classification'] for r in rows} == {'unknown_prejoin_or_provider'}
    assert {r['role'] for r in rows} == {'history'}
    assert {r['bar_label'] for r in rows} == {missing.isoformat()}
    assert all(r['provider_closed'] is None for r in rows)


def test_gap_requires_63_new_history_bars_and_never_bridges_missing() -> None:
    market, calendar = market_calendar('2024-01-01', 170)
    market = market.drop(market.index[70])
    start, end = bounds(calendar, 63)
    result = select_contiguous_range(market, calendar, start, end)
    assert result['measurement_start'] == pd.Timestamp(calendar['sessions'][134]['session_close']).tz_convert('UTC').isoformat()
    assert result['measured_bars'] == 36
    assert [r['measured_bars'] for r in result['candidates']] == [7, 36]
    plan = prepare_period(market, calendar, result['measurement_start'], result['measurement_end'], tuple(calendar['symbols']), Path('fixture'))
    assert len(plan.history_labels) == 63
    assert len(plan.timestamps) == 36


def test_history_shortage_is_explicit_and_does_not_shorten_warmup() -> None:
    market, calendar = market_calendar('2024-01-01', 100)
    market = market.drop(market.index[40])
    start, end = bounds(calendar, 63)
    with pytest.raises(ValueError, match='63.*history'):
        select_contiguous_range(market, calendar, start, end)


def test_longest_coverage_selection_is_independent_of_prices() -> None:
    market, calendar = market_calendar('2024-01-01', 210)
    market = market.drop(market.index[105])
    start, end = bounds(calendar, 63)
    result = select_contiguous_range(market, calendar, start, end)
    changed = market.copy()
    for symbol in calendar['symbols']:
        for field in ('Open', 'High', 'Low', 'Close'):
            changed[symbol, field] *= np.linspace(1, 2, len(changed))
    assert select_contiguous_range(changed, calendar, start, end) == result
    assert result['measured_bars'] == 42


def test_declared_holiday_does_not_consume_a_history_bar() -> None:
    market, calendar = market_calendar('2024-01-01', 100)
    holiday = calendar['sessions'].pop(40)
    calendar['holidays'] = [holiday['bar_label'][:10]]
    market = market.drop(pd.Timestamp(holiday['bar_label']))
    start, end = bounds(calendar, 63)
    result = select_contiguous_range(market, calendar, start, end)
    assert result['measurement_start'] == pd.Timestamp(start).tz_convert('UTC').isoformat()
    assert audit_missing_pairs(market, calendar, start, end) == []


def test_dst_preserves_close_instants_and_financing_elapsed_time() -> None:
    market, calendar = market_calendar('2023-12-01', 100)
    start, end = bounds(calendar, 63)
    result = select_contiguous_range(market, calendar, start, end)
    plan = prepare_period(market, calendar, result['measurement_start'], result['measurement_end'], tuple(calendar['symbols']), Path('fixture'))
    hours = np.diff(pd.to_datetime(plan.timestamps).as_unit('s').asi8) / 3600
    assert 71 in hours
    assert 72 in hours


@pytest.mark.parametrize('bad', ['missing_pair', 'reordered_pair', 'nan', 'duplicate', 'unexpected', 'late_available', 'holiday_conflict'])
def test_invalid_input_fails_instead_of_becoming_a_shorter_success(bad: str) -> None:
    market, calendar = market_calendar('2024-01-01', 100)
    start, end = bounds(calendar, 63)
    if bad == 'missing_pair':
        market = market.drop(columns=calendar['symbols'][0], level=0)
    elif bad == 'reordered_pair':
        market = market.loc[:, list(reversed(market.columns))]
    elif bad == 'nan':
        market.iloc[80, 0] = np.nan
    elif bad == 'duplicate':
        market = pd.concat([market, market.iloc[80:81]]).sort_index()
    elif bad == 'unexpected':
        extra = market.iloc[80:81].copy()
        extra.index = extra.index + pd.Timedelta(hours=1)
        market = pd.concat([market, extra]).sort_index()
    elif bad == 'late_available':
        calendar['sessions'][80]['available_at'] = end
    else:
        calendar['holidays'] = [calendar['sessions'][80]['bar_label'][:10]]
    with pytest.raises((ValueError, Exception), match='pair|finite|duplicate|Unexpected|available_at|holiday|OHLCV|NaN|non-finite'):
        select_contiguous_range(market, calendar, start, end)


def test_output_is_immutable_even_when_content_is_identical(tmp_path: Path) -> None:
    output = tmp_path / 'manifest.json'
    write_immutable_json(output, {'identity': 'one'})
    original = output.read_bytes()
    with pytest.raises(FileExistsError):
        write_immutable_json(output, {'identity': 'one'})
    assert output.read_bytes() == original
