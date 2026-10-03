"""Reject unsafe handoffs without opening new synthetic accounts."""
from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest

from forex_trainer.common_allocation import load_forecasts
from forex_trainer.common_quantity_study import load_config, require_unused_budget, verify_fixture
from forex_trainer.common_quantity import marks_from_period
from forex_trainer.full_period import prepare_period
from forex_trainer.common_basket import PAIRS
from forex_trainer.common_basket_study import digest
from test_common_basket import market_fixture

CONFIG = Path('configs/research/issue42_common_quantity.json')


def test_config_rejects_unregistered_scope(tmp_path: Path) -> None:
    raw = json.loads(CONFIG.read_text())
    raw['fixture_account_limit'] = 17
    path = tmp_path / 'config.json'
    path.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match='contract'):
        load_config(path)


def test_saved_consumption_prevents_new_clone_rerun(tmp_path: Path) -> None:
    path = tmp_path / 'saved'
    path.mkdir()
    (path / 'manifest.json').write_text(json.dumps({'fixture_accounts': 13}))
    with pytest.raises(ValueError, match='consumed|reserved'):
        require_unused_budget(tmp_path / 'absent-run', path)


def test_interrupted_attempt_still_consumes_budget(tmp_path: Path) -> None:
    path = tmp_path / 'run'
    path.mkdir()
    (path / 'account-ledger.jsonl').write_text('{"event":"reserved","work_item":"fixed-h1"}\n')
    with pytest.raises(ValueError, match='consumed|reserved'):
        require_unused_budget(path, tmp_path / 'absent-saved')


def test_unsealed_bundle_rejected_before_account(tmp_path: Path) -> None:
    (tmp_path / 'incident.json').write_text('{}')
    with pytest.raises(ValueError, match='incident'):
        load_forecasts(tmp_path)


def test_saved_fixture_tamper_is_detected(tmp_path: Path) -> None:
    import shutil
    target = tmp_path / 'fixture'
    shutil.copytree('docs/research/results/issue42/fixture', target)
    path = target / 'accounts/hold.json'
    raw = json.loads(path.read_text())
    raw['trace'][1]['actual_traded_notional'] = 1.
    path.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match='hash|artifact'):
        verify_fixture(target)


def test_history_is_not_a_decision_and_future_windows_do_not_leak() -> None:
    market = market_fixture().iloc[:70].copy()
    dates = market.index
    calendar = {'version': 'synthetic-london', 'timezone': 'Europe/London', 'symbols': list(PAIRS),
                'holidays': [], 'label_rule': 'London midnight historical assumed close',
                'sessions': [{'bar_label': day.isoformat(), 'session_open': (day - __import__('pandas').Timedelta(hours=12)).isoformat(),
                              'session_close': day.isoformat(), 'available_at': None} for day in dates]}
    start, end = dates[63].isoformat(), (dates[-1] + __import__('pandas').Timedelta(days=1)).isoformat()
    plan = prepare_period(market, calendar, start, end, PAIRS, Path('synthetic-history'))
    marks = marks_from_period(plan)
    assert len(marks) == 7
    assert marks[0].at == dates[63].tz_convert('UTC').isoformat()
    changed = market.copy()
    changed.iloc[66:] *= 2
    other_plan = prepare_period(changed, calendar, start, end, PAIRS, Path('synthetic-future'))
    other = marks_from_period(other_plan)
    for a, b in zip(marks[:3], other[:3], strict=True):
        np.testing.assert_array_equal(a.window, b.window)
        np.testing.assert_array_equal(a.carry, b.carry)
        np.testing.assert_array_equal(a.prices, b.prices)
    assert digest(marks[3].window.tolist()) != digest(other[3].window.tolist())
