"""Behavioral regressions for frozen next-close orders and separate routes."""
from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any
import copy
import json

import numpy as np
import pytest

from forex_trainer.common_allocation import Costs, Decision, State
from forex_trainer.common_basket import advance, utc
from forex_trainer.common_quantity import Mark, QuantityAccount, QuantityExecutionError, instruction, replay
from forex_trainer.execution_sensitivity import (
    MEASUREMENT_ID, audit_proxy, fill_order, freeze_order, replay_next_close,
)
from forex_trainer.execution_sensitivity_study import (
    CONFIG, contract, registered_cells, require_registration_merge, summarize,
)


def marks(values: list[float], start: date) -> tuple[Mark, ...]:
    """Build known synthetic prices and causal windows."""
    return tuple(Mark(utc(advance(start, i)), np.full(9, p), np.full(9, .0365),
                      np.full((9, 32, 8), float(i))) for i, p in enumerate(values))


def fixed_quantity(state: State, observation: dict[str, np.ndarray], window: np.ndarray) -> Decision:
    """Request a fixed exposure using only current price and equity."""
    return instruction(state, np.full(9, state.equity / state.prices[0] / 18), 'legacy_target')


def hold_after_entry(state: State, observation: dict[str, np.ndarray], window: np.ndarray) -> Decision:
    """Enter once and then preserve the actual quantities."""
    if not np.any(state.quantities):
        return instruction(state, np.full(9, 100.), 'partial')
    return instruction(state, state.quantities, 'hold_quantity')


def test_frozen_quantity_old_gap_and_fill_costs() -> None:
    axis = marks([100., 120., 90.], date(2025, 1, 6))
    costs = Costs(np.full(9, .001), .002, .0001)
    result = replay_next_close(axis, costs, hold_after_entry, 'a' * 64, {'fixture': True})
    assert result['measurement_id'] == MEASUREMENT_ID
    first, second = result['trace']
    assert first['price_pnl'] == 0
    assert first['financing'] == 0
    assert first['markup'] == 0
    np.testing.assert_allclose(first['orders'][0]['quantities'], 100.)
    assert first['orders'][0]['status'] == 'filled'
    assert first['orders'][0]['fill_at'] == axis[1].at
    assert first['orders'][0]['equity_before_fill'] == 1_000_000
    assert first['actual_traded_notional'] == 108_000
    assert first['trades'][-1]['spread'] == pytest.approx(108.)
    assert first['trades'][-1]['commission'] == pytest.approx(216.)
    assert second['price_pnl'] == pytest.approx(-27_000)
    assert second['gap_price_pnl'] == second['price_pnl']
    assert second['financing'] == pytest.approx(-8.1)
    assert second['markup'] == pytest.approx(10.8)
    assert second['actual_traded_notional'] == 0
    assert result['final_equity'] == pytest.approx(972_657.1)
    assert result['remaining_decisions'] == 0
    assert result['virtual_terminal_liquidation'] is False
    assert len(result['trace']) == 2
    audit_proxy(result, axis, costs, 'fixture')


def test_future_price_does_not_resize_frozen_order() -> None:
    costs = Costs(np.zeros(9), 0., 0.)
    a = replay_next_close(marks([100., 110.], date(2025, 1, 6)), costs, fixed_quantity, 'a' * 64, {})
    b = replay_next_close(marks([100., 130.], date(2025, 1, 6)), costs, fixed_quantity, 'a' * 64, {})
    assert a['trace'][0]['orders'][0]['quantities'] == b['trace'][0]['orders'][0]['quantities']
    assert a['final_quantities'] == b['final_quantities']
    assert a['trace'][0]['decision'] == b['trace'][0]['decision']
    assert a['final_equity'] == b['final_equity'] == 1_000_000


def test_dst_gap_uses_actual_seconds_and_independent_assets() -> None:
    axis = marks([100., 100., 110., 110.], date(2025, 3, 27))
    seen: list[np.ndarray] = []

    def observe(state: State, observation: dict[str, np.ndarray], window: np.ndarray) -> Decision:
        seen.append(observation['assets'].copy())
        np.testing.assert_array_equal(observation['market'], window)
        return hold_after_entry(state, observation, window)

    costs = Costs(np.zeros(9), 0., .001)
    result = replay_next_close(axis, costs, observe, 'a' * 64, {})
    row = result['trace'][1]
    assert row['elapsed_days'] == pytest.approx(71 / 24)
    assert row['markup'] == pytest.approx(90 * 71 / 24)
    assert row['financing'] == pytest.approx(-9.9 * 71 / 24)
    np.testing.assert_array_equal(seen[0], np.zeros((9, 3)))
    np.testing.assert_allclose(seen[2][:, 2], 1000 / 1_000_000)
    assert not np.array_equal(seen[1], seen[0])
    audit_proxy(result, axis, costs, 'dst')


def test_same_close_reference_is_reinferred_on_its_own_account() -> None:
    axis = marks([100., 110., 120.], date(2025, 1, 6))
    costs = Costs(np.zeros(9), 0., 0.)
    same = replay(axis, costs, fixed_quantity, 'a' * 64, {})
    delayed = replay_next_close(axis, costs, fixed_quantity, 'a' * 64, {})
    assert same['trace'][1]['equity_at_decision'] != delayed['trace'][1]['equity_at_decision']
    assert same['trace'][1]['decision']['quantities'] != delayed['trace'][1]['decision']['quantities']
    assert same['measurement_id'] != delayed['measurement_id']


def test_expiry_does_not_trade_or_charge() -> None:
    axis = marks([100., 100.], date(2025, 1, 6))
    account = QuantityAccount(axis[0], Costs(np.full(9, .01), .01, .01))
    state = account.prepare()
    order = freeze_order(state, fixed_quantity(state, {}, axis[0].window))
    result = fill_order(account, order, axis[0].at)
    assert result['status'] == 'expired'
    assert result['reason'] == 'after_final_mark'
    assert result['trades'] == []
    assert account.account.equity_jpy == 1_000_000
    np.testing.assert_array_equal(account.quantities, np.zeros(9))


def test_wrong_fill_timestamp_and_foreign_decision_fail() -> None:
    axis = marks([100., 101., 102.], date(2025, 1, 6))
    account = QuantityAccount(axis[0], Costs(np.zeros(9), 0., 0.))
    state = account.prepare()
    decision = fixed_quantity(state, {}, axis[0].window)
    order = freeze_order(state, decision)
    with pytest.raises(ValueError, match='expected.*fill|fill.*expected'):
        fill_order(account, order, axis[-1].at)
    foreign = State(axis[1].at, state.pairs, state.equity, state.quantities, state.prices, state.carry, state.costs)
    with pytest.raises(ValueError, match='stale|foreign'):
        freeze_order(foreign, decision)


def test_missing_expected_bar_never_falls_back() -> None:
    axis = marks([100., 110., 120.], date(2025, 1, 6))
    with pytest.raises(ValueError, match='expected|gap'):
        replay_next_close((axis[0], axis[2]), Costs(np.zeros(9), 0., 0.), fixed_quantity, 'a' * 64, {})


def test_fill_risk_is_a_separate_trade_not_order_resizing() -> None:
    axis = marks([100., 200.], date(2025, 1, 6))

    def concentrated(state: State, observation: dict[str, np.ndarray], window: np.ndarray) -> Decision:
        q = np.zeros(9)
        q[0] = 9_000
        return instruction(state, q, 'partial')

    costs = Costs(np.full(9, .001), 0., 0.)
    result = replay_next_close(axis, costs, concentrated, 'a' * 64, {})
    row = result['trace'][0]
    assert row['orders'][0]['quantities'][0] == 9_000
    assert row['trades'][0]['quantities_after'][0] == 9_000
    assert row['trades'][1]['reason'] == 'risk_fill'
    assert row['quantities'][0] < 5_000
    assert max(abs(x) for x in row['execution_weights_after_cost']) <= 1 + 1e-10
    audit_proxy(result, axis, costs, 'risk')


def test_gap_margin_cancels_pending_fill_and_retains_unexecuted_tail() -> None:
    axis = marks([100., 100., 1., 1.], date(2025, 1, 6))

    def leveraged(state: State, observation: dict[str, np.ndarray], window: np.ndarray) -> Decision:
        return instruction(state, np.full(9, 2_000.), 'partial')

    result = replay_next_close(axis, Costs(np.zeros(9), 0., 0.), leveraged, 'a' * 64, {})
    assert result['status'] == 'strategy_terminal'
    assert result['terminal_reason'] == 'nonpositive_equity'
    assert result['trace'][-1]['orders'][0]['status'] == 'cancelled_terminal'
    assert result['trace'][-1]['trades'] == []
    assert result['remaining_decisions'] == 1
    assert result['net_log_return'] is None


def test_policy_error_preserves_completed_trace_and_origin() -> None:
    axis = marks([100., 101., 102.], date(2025, 1, 6))

    def fails(state: State, observation: dict[str, np.ndarray], window: np.ndarray) -> Decision:
        if state.at != axis[0].at:
            raise RuntimeError('fixture inference failed')
        return fixed_quantity(state, observation, window)

    with pytest.raises(QuantityExecutionError, match='fixture inference failed') as caught:
        replay_next_close(axis, Costs(np.zeros(9), 0., 0.), fails, 'a' * 64, {})
    assert caught.value.evidence['at'] == axis[1].at
    assert len(caught.value.evidence['trace']) == 1
    assert caught.value.__cause__ is not None


def test_audit_rejects_consistent_hash_but_wrong_gap_accounting() -> None:
    axis = marks([100., 110., 120.], date(2025, 1, 6))
    costs = Costs(np.zeros(9), 0., 0.)
    result = replay_next_close(axis, costs, hold_after_entry, 'a' * 64, {})
    result['trace'][1]['price_pnl'] += 1
    with pytest.raises(ValueError, match='accounting|reconciliation'):
        audit_proxy(result, axis, costs, 'tampered')


def test_registration_separates_306_proxy_and_blocked_six_quote_cells() -> None:
    config = contract(CONFIG)
    cells = registered_cells()
    assert len(cells) == 306
    assert len({(c['fold'], c['scenario'], c['policy'], c['execution']) for c in cells}) == 306
    assert config['proxy']['account_limit_after_merge'] == 306
    assert config['quote']['effective_accounts'] == 0
    report = summarize(cells)
    assert len(report['quote_cells']) == 6
    assert all(c['status'] == 'blocked_input' for c in report['quote_cells'])
    assert all(c['status'] == 'pending_measurement' for c in report['comparisons'])
    assert report['candidate_validation'] == 'not_performed'


def test_registration_rejects_expanded_budget(tmp_path: Path) -> None:
    raw = json.loads(CONFIG.read_text())
    raw['proxy']['account_limit_after_merge'] = 307
    path = tmp_path / 'config.json'
    path.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match='contract|budget'):
        contract(path)


def test_unmerged_registration_cannot_activate() -> None:
    with pytest.raises(ValueError, match='merge|registration'):
        require_registration_merge('c549746')


def test_complete_fold_statistics_are_withheld_for_any_missing_cell() -> None:
    cells = registered_cells()
    for c in cells:
        c['status'] = 'complete'
        c['metrics'] = {'period_net_log': .1 if c['execution'] == 'same_close' else .09,
                        'period_gross_log': .12, 'gap_price_pnl': 0., 'spread': 1.,
                        'commission': 0., 'markup': 2., 'actual_traded_notional': 10.,
                        'max_drawdown': .1, 'mean_gross_exposure': .5}
    complete = summarize(cells)
    assert all(c['status'] == 'available' for c in complete['comparisons'])
    assert complete['comparisons'][0]['net']['mean_difference'] == pytest.approx(-.01)
    missing = copy.deepcopy(cells)
    missing[0]['status'] = 'execution_error'
    missing[0]['metrics'] = None
    report = summarize(missing)
    assert report['comparisons'][0]['status'] == 'pending_measurement'
    assert report['comparisons'][0]['net'] is None
    with pytest.raises(ValueError, match='306|matrix'):
        summarize(cells[:-1])
