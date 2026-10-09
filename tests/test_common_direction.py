"""Short campaign decisions from numerical observations and saved cash ledgers."""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from forex_trainer.common_basket_study import read
from forex_trainer.common_direction import account_metrics, registered_cells, summarize
from forex_trainer.common_direction_study import require_unused_budget, scenario_costs
from forex_trainer.common_quantity_study import fixture_marks, audit_account

RULES = read(Path('docs/research/protocols/issue40/revisions/v3/registration.json'))['risk_and_decision']
PREDICTION = {'1': {'met': True}, '5': {'met': True}}


def observations() -> list[dict[str, Any]]:
    """Provide explicit numerical fold observations without opening accounts.

    Returns:
        Complete matrix with an identifiable allocation improvement.
    """
    cells = registered_cells()
    levels = {'A1-fixed': .01, 'A1-cost': .02, 'A5-fixed': .008, 'A5-cost': .015,
              'canonical': .002, 'ridge': .003, 'ppo_ens3': .004}
    for cell in cells:
        cell.update(status='complete', metrics={
            'period_net_log': levels[cell['policy']], 'max_drawdown': .05,
            'annual_net_log_volatility': .1, 'actual_notional_ratio': 3.,
            'annualized_net_log': levels[cell['policy']] * 2,
        })
    return cells


def cell_at(cells: list[dict[str, Any]], policy: str, scenario: str, fold: str) -> dict[str, Any]:
    """Find one explicit numerical observation.

    Args:
        cells: Complete matrix.
        policy: Registered policy.
        scenario: Registered cost scenario.
        fold: Registered interval.

    Returns:
        Unique cell.
    """
    return next(c for c in cells if (c['policy'], c['scenario'], c['fold']) == (policy, scenario, fold))


def test_full_matrix_and_period_primary_comparisons() -> None:
    cells = observations()
    report = summarize(cells, RULES, PREDICTION)
    assert len(cells) == 357
    assert len({(c['policy'], c['scenario'], c['fold']) for c in cells}) == 357
    assert report['status'] == 'complete'
    for scenario in ('F0', 'F1', 'F2'):
        comparisons = report['comparisons'][scenario]
        assert len(comparisons) == 17
        assert comparisons['A1-cost_minus_A1-fixed']['evidence']['mean_difference'] == pytest.approx(.01)
        assert comparisons['interaction']['evidence']['mean_difference'] == pytest.approx(-.003)
        assert len(comparisons['interaction']['evidence']['leave_one_fold_out_means']) == 17
    assert report['selected_candidate'] == 'A1-cost'
    assert report['allocation_added_value']['1']['status'] == 'supported'
    assert report['independent_profitability'] == 'not_established'
    assert report['handoff_issue49']['candidate'] == 'A1-cost'
    assert report['handoff_issue46']['horizon'] == 1
    assert report['handoff_issue46']['training_authorized'] is False


@pytest.mark.parametrize('status', ['blocked_input', 'execution_error', 'not_run', 'strategy_terminal'])
def test_incomplete_comparison_never_drops_a_fold(status: str) -> None:
    cells = observations()
    cell_at(cells, 'A1-cost', 'F0', '2009').update(status=status, metrics=None)
    report = summarize(cells, RULES, PREDICTION)
    assert report['status'] == 'incomplete'
    comparison = report['comparisons']['F0']['A1-cost_minus_A1-fixed']
    assert comparison['status'] == 'pending_measurement'
    assert comparison['evidence'] is None
    assert report['candidates']['A1-cost']['status'] == ('rejected_terminal' if status == 'strategy_terminal' else 'pending_measurement')
    assert report['selected_candidate'] is None
    assert report['handoff_issue46']['status'] == 'pending_measurement'
    assert len(report['cells']) == 357
    assert report['comparisons']['F0']['A5-cost_minus_A5-fixed']['status'] == 'available'


@pytest.mark.parametrize('mutation', ['missing', 'duplicate', 'unknown', 'nan'])
def test_invalid_matrix_is_an_exception(mutation: str) -> None:
    cells = observations()
    if mutation == 'missing':
        cells.pop()
    elif mutation == 'duplicate':
        cells[-1] = copy.deepcopy(cells[0])
    elif mutation == 'unknown':
        cells[0]['policy'] = 'unregistered'
    else:
        cells[0]['metrics']['period_net_log'] = float('nan')
    with pytest.raises(ValueError):
        summarize(cells, RULES, PREDICTION)


@pytest.mark.parametrize('cause', ['stress_loss', 'stress_drop', 'drawdown', 'volatility', 'era', 'loo', 'flat'])
def test_candidate_rejection_uses_all_registered_conditions(cause: str) -> None:
    cells = observations()
    for cell in cells:
        if cell['policy'] != 'A1-cost':
            continue
        metric = cell['metrics']
        if cause == 'stress_loss' and cell['scenario'] == 'F1':
            metric['period_net_log'] = -.001
        elif cause == 'stress_drop' and cell['scenario'] == 'F2':
            metric['period_net_log'] = .01
        elif cause == 'drawdown' and cell['fold'] == '2009':
            metric['max_drawdown'] = .251
        elif cause == 'volatility':
            metric['annual_net_log_volatility'] = .201
        elif cause == 'era' and int(cell['fold']) >= 2019:
            metric['period_net_log'] = -.001
        elif cause == 'loo':
            metric['period_net_log'] = .2 if cell['fold'] == '2009' else -.001
        elif cause == 'flat':
            metric['period_net_log'] = 0.
    report = summarize(cells, RULES, PREDICTION)
    assert report['candidates']['A1-cost']['status'] == 'rejected'
    assert report['allocation_added_value']['1']['status'] != 'supported'


def test_concentrated_difference_does_not_support_added_value() -> None:
    cells = observations()
    for cell in cells:
        if cell['policy'] == 'A1-cost':
            cell['metrics']['period_net_log'] = .21 if cell['fold'] == '2009' else .011
    report = summarize(cells, RULES, PREDICTION)
    assert report['candidates']['A1-cost']['status'] == 'eligible'
    assert report['allocation_added_value']['1']['status'] == 'not_supported'
    assert report['comparisons']['F0']['A1-cost_minus_A1-fixed']['evidence']['largest_absolute_contribution_fraction'] > .5


def test_tie_prefers_h1_then_fixed_and_zero_difference_is_undefined_share() -> None:
    cells = observations()
    for cell in cells:
        cell['metrics']['period_net_log'] = .02 + (5e-13 if cell['policy'] == 'A5-cost' else 0.)
    report = summarize(cells, RULES, PREDICTION)
    assert report['selected_candidate'] == 'A1-fixed'
    evidence = report['comparisons']['F0']['A1-cost_minus_A1-fixed']['evidence']
    assert evidence['largest_absolute_contribution_fraction'] is None
    assert evidence['contribution_status'] == 'undefined_zero_absolute_sum'


def test_failed_prediction_gate_stops_rl_separately_from_simple_candidate() -> None:
    report = summarize(observations(), RULES, {'1': {'met': False}, '5': {'met': False}})
    assert report['handoff_issue46']['status'] == 'not_planned'
    assert 'no_predictive_room' in report['stop_reasons']
    assert report['handoff_issue49']['candidate'] == 'A1-cost'


def test_saved_ledger_metrics_keep_cash_and_nonpositive_log_semantics() -> None:
    root = Path('docs/research/results/issue42/fixture')
    raw = read(root / 'fixture.json')
    account = read(root / 'accounts/hold.json')
    costs = scenario_costs('F0')
    assert costs.markup == .00002
    metrics = account_metrics(account)
    rows = account['trace']
    assert metrics['period_net_log'] == pytest.approx(np.log(account['final_equity'] / 1_000_000))
    total_cost = sum(r['markup'] + sum(t['spread'] + t['commission'] for t in r['trades']) for r in rows)
    assert metrics['cost_ratio'] == pytest.approx(total_cost / 1_000_000)
    assert metrics['actual_notional_ratio'] == pytest.approx(sum(r['actual_traded_notional'] for r in rows) / 1_000_000)
    assert metrics['literal_hold_decisions'] == sum(r['literal_hold'] for r in rows)
    assert metrics['price_pnl'] + metrics['financing'] - total_cost == pytest.approx(account['final_equity'] - 1_000_000)
    bankruptcy = account_metrics(read(root / 'accounts/bankruptcy.json'))
    assert bankruptcy['period_net_log'] is None
    assert bankruptcy['annualized_net_log'] is None
    assert bankruptcy['annual_net_log_volatility'] is None
    assert len(fixture_marks(raw)) == 6


def test_stress_changes_only_registered_cost_component() -> None:
    f0, f1, f2 = [scenario_costs(s) for s in ('F0', 'F1', 'F2')]
    np.testing.assert_array_equal(f1.spreads, 2 * f0.spreads)
    np.testing.assert_array_equal(f2.spreads, f0.spreads)
    assert f1.markup == f0.markup
    assert f2.markup == 2 * f0.markup
    assert f0.commission == f1.commission == f2.commission == 0.
    with pytest.raises(ValueError):
        scenario_costs('F3')


def test_used_or_reserved_budget_cannot_move_to_a_new_output(tmp_path: Path) -> None:
    run, saved = tmp_path / 'run', tmp_path / 'saved'
    require_unused_budget(run, saved)
    saved.mkdir()
    with pytest.raises(ValueError, match='budget'):
        require_unused_budget(run, saved)
