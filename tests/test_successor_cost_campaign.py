"""The frozen successor panel must not become a completed annual campaign."""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from forex_trainer.cost_campaign import POLICIES, SCENARIOS
from forex_trainer.successor_cost_campaign import CAMPAIGN_ID, summarize_successor, validate_config
from forex_trainer.supervised_portfolio import FOLDS, paired_evidence


def panel_fixture() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Construct complete labelled observations with nine partial ranges."""
    coverage = [{'fold': f, 'successor_scope': 'partial_year' if i < 9 else 'full_year',
                 'first_decision': f'{f}-05-01T00:00:00+00:00' if i < 9 else f'{f}-01-01T00:00:00+00:00',
                 'last_mark': f'{f}-12-31T00:00:00+00:00'} for i, f in enumerate(FOLDS)]
    cells = []
    for row in coverage:
        for si, scenario in enumerate(SCENARIOS):
            for pi, policy in enumerate(POLICIES):
                net = (int(row['fold']) - 2015) * .01 + pi * .02 - si * .03
                metrics = {name: .1 for name in ('sharpe_annualized', 'max_drawdown', 'mean_gross_leverage',
                    'mean_target_gross_exposure', 'mean_target_net_exposure', 'total_weight_turnover',
                    'total_cost_ratio', 'annualized_gross_minus_net', 'total_price_pnl_jpy', 'total_financing_jpy',
                    'total_spread_jpy', 'total_commission_jpy', 'total_overnight_jpy', 'total_actual_traded_notional_jpy',
                    'period_net_return', 'period_gross_return', 'annualized_net_volatility')}
                metrics.update({'annualized_net_return': net, 'annualized_gross_return': net + .05})
                cells.append({'fold': row['fold'], 'scenario': scenario, 'policy': policy, 'status': 'complete',
                              'coverage': {'first_decision': row['first_decision'], 'last_mark': row['last_mark'], 'elapsed_seconds': 86400 * 240},
                              'metrics': metrics})
    return cells, coverage


def test_successor_completion_never_completes_old_annual_attempt() -> None:
    cells, coverage = panel_fixture()
    report = summarize_successor(cells, coverage)
    assert report['campaign_id'] == CAMPAIGN_ID
    assert report['status'] == 'complete'
    assert report['completed_policy_folds'] == 153
    assert report['original_annual_campaign_complete'] is False
    assert report['scope_counts'] == {'full_year': 8, 'partial_year': 9}
    values = np.array([(int(f) - 2015) * .01 for f in FOLDS])
    assert report['absolute_evidence']['F0']['canonical']['annualized_net_return'] == paired_evidence(values, np.zeros(17))
    assert report['summaries']['F0']['canonical']['all']['fold_count'] == 17
    assert len(report['paired_evidence']['F0']['canonical_minus_ridge']['annualized_net_return']['leave_one_fold_out_means']) == 17
    assert report['original_blocked_policy_folds'] == 81


@pytest.mark.parametrize('status', ['incomplete_margin_call', 'execution_error', 'blocked_input'])
def test_any_incomplete_successor_cell_withholds_aggregate_evidence(status: str) -> None:
    cells, coverage = panel_fixture()
    cells[0]['status'] = status
    report = summarize_successor(cells, coverage)
    assert report['status'] == 'incomplete'
    assert report['completed_policy_folds'] == 152
    for name in ('summaries', 'paired_evidence', 'absolute_evidence', 'sensitivity_evidence'):
        assert report[name] is None
    assert len(report['cells']) == 153


def test_duplicate_cell_or_changed_scope_cannot_be_aggregated() -> None:
    cells, coverage = panel_fixture()
    wrong = copy.deepcopy(cells)
    wrong[-1] = wrong[0]
    with pytest.raises(ValueError, match='153|unique|duplicate'):
        summarize_successor(wrong, coverage)
    cells[0]['coverage']['first_decision'] = '2009-01-01T00:00:00+00:00'
    with pytest.raises(ValueError, match='coverage|range'):
        summarize_successor(cells, coverage)
    with pytest.raises(ValueError, match='coverage|fold'):
        summarize_successor(cells[1:], coverage[:-1])


def test_config_cannot_add_costs_or_override_registered_ranges() -> None:
    from forex_trainer.full_period import read_json
    config = read_json(Path('configs/research/issue29_successor_cost.json'))
    validate_config(config)
    for key, value in (('scenarios', {'F0': SCENARIOS['F0']}), ('folds', []), ('campaign_id', 'issue29-full-period-cost-v1')):
        bad = copy.deepcopy(config)
        bad[key] = value
        with pytest.raises(ValueError, match='fields|scenario|campaign'):
            validate_config(bad)
