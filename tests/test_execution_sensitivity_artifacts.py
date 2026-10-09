"""Save failure evidence and reject misleading execution diagnostics."""
from __future__ import annotations

import copy
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from forex_trainer.common_allocation import Costs, Decision, State
from forex_trainer.common_basket_study import digest, read, write
from forex_trainer.common_quantity import instruction
from forex_trainer.common_quantity_study import append_event
from forex_trainer.execution_sensitivity import audit_proxy, replay_next_close
from forex_trainer.execution_sensitivity_study import (
    CONFIG, cell_name, finish, metrics, registered_cells, seal, verify,
)
from test_execution_sensitivity import hold_after_entry, marks


def test_short_partial_and_close_use_current_fill_notional() -> None:
    axis = marks([100., 110., 90., 95.], date(2025, 1, 6))

    def decisions(state: State, observation: dict[str, np.ndarray], window: np.ndarray) -> Decision:
        value = {axis[0].at: -100., axis[1].at: -50., axis[2].at: 0.}[state.at]
        return instruction(state, np.full(9, value), 'close' if value == 0 else 'partial')

    costs = Costs(np.full(9, .001), .001, .001)
    result = replay_next_close(axis, costs, decisions, 'a' * 64, {})
    assert [r['actual_traded_notional'] for r in result['trace']] == [99_000., 40_500., 42_750.]
    assert [r['price_pnl'] for r in result['trace']] == pytest.approx([0., 18_000., -2250.])
    assert result['trace'][1]['financing'] > 0
    assert result['trace'][1]['markup'] > 0
    assert result['final_quantities'] == [0.] * 9
    audit_proxy(result, axis, costs, 'short')


def test_nonpositive_equity_does_not_fabricate_flat_exposure() -> None:
    axis = marks([100., 100., 1.], date(2025, 1, 6))

    def leveraged(state: State, observation: dict[str, np.ndarray], window: np.ndarray) -> Decision:
        return instruction(state, np.full(9, 2000.), 'partial')

    costs = Costs(np.zeros(9), 0., 0.)
    result = replay_next_close(axis, costs, leveraged, 'a' * 64, {})
    audit_proxy(result, axis, costs, 'bankrupt')
    value = metrics(result, 'next_close')
    assert value['period_net_log'] is None
    assert value['mean_gross_exposure'] is None
    assert value['exposure_status'] == 'undefined_nonpositive_equity'
    assert value['order_status_counts']['cancelled_terminal'] == 1


@pytest.mark.parametrize('field,value', [('literal_hold', False), ('risk_override', True),
                                        ('constraint_residual', 99.), ('marked_weights_after', [0.] * 9)])
def test_audit_rejects_false_execution_diagnostics(field: str, value: Any) -> None:
    axis = marks([100., 110., 120.], date(2025, 1, 6))
    costs = Costs(np.zeros(9), 0., 0.)
    result = replay_next_close(axis, costs, hold_after_entry, 'a' * 64, {})
    result['trace'][1][field] = value
    with pytest.raises(ValueError, match='accounting|reconciliation|diagnostic'):
        audit_proxy(result, axis, costs, 'tamper')


def test_failed_attempt_keeps_all_unexecuted_cells_and_hashes(tmp_path: Path) -> None:
    cells = registered_cells()
    cells[0].update(status='execution_error', error='fixture failure')
    ledger = tmp_path / 'account-ledger.jsonl'
    append_event(ledger, {'event': 'budget_reserved', 'proxy_limit': 306, 'quote_limit': 0})
    append_event(ledger, {'event': 'reserved', 'work_item': cell_name(cells[0])})
    append_event(ledger, {'event': 'finished', 'work_item': cell_name(cells[0]), 'status': 'execution_error'})
    incident = {'error': 'fixture failure', 'cell': cell_name(cells[0]), 'partial': None, 'automatic_retry': False}
    write(tmp_path / 'incident.json', incident)
    report = finish(tmp_path, cells, incident)
    assert report['status_counts']['execution_error'] == 1
    assert report['status_counts']['not_run'] == 305
    assert report['issue45_complete'] is False
    assert report['incident'] == incident
    manifest = read(tmp_path / 'manifest.json')
    assert manifest['proxy_accounts_consumed'] == 1
    assert 'incident.json' in manifest['artifacts']
    assert 'account-ledger.jsonl' in manifest['artifacts']
    with pytest.raises(FileExistsError):
        finish(tmp_path, cells, incident)


def test_verification_rejects_tamper_before_loading_models(tmp_path: Path) -> None:
    write(tmp_path / 'report.json', {'status': 'forged'})
    manifest = {'artifacts': {'report.json': '0' * 64}}
    manifest['content_sha256'] = digest(copy.deepcopy(manifest))
    write(tmp_path / 'manifest.json', manifest)
    with pytest.raises(ValueError, match='artifact hash'):
        verify(tmp_path)


def test_market_seal_cannot_use_parent_merge_without_registration(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match='merge|registration'):
        seal(CONFIG, 'e33d3417f0c5e98fa348d86eecc31a9ce16a3adc', tmp_path / 'activation')
    assert not (tmp_path / 'activation').exists()
