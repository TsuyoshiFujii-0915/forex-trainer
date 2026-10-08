"""Registered period-log decisions for the complete short 2-by-2 matrix."""
from __future__ import annotations

from datetime import datetime
from functools import lru_cache
from typing import Any

import numpy as np

from .supervised_portfolio import FOLDS, paired_evidence

CANDIDATES = ('A1-fixed', 'A1-cost', 'A5-fixed', 'A5-cost')
CONTROLS = ('canonical', 'ridge', 'ppo_ens3')
POLICIES = (*CANDIDATES, *CONTROLS)
SCENARIOS = ('F0', 'F1', 'F2')
STATUSES = ('not_run', 'blocked_input', 'execution_error', 'complete', 'strategy_terminal')


def registered_cells() -> list[dict[str, Any]]:
    """Register every account before any inference.

    Returns:
        The fixed 357 cells, including their initial unexecuted status.
    """
    return [{'fold': fold, 'scenario': scenario, 'policy': policy, 'status': 'not_run',
             'metrics': None, 'error': None, 'result_path': None}
            for fold in FOLDS for scenario in SCENARIOS for policy in POLICIES]


def account_metrics(account: dict[str, Any]) -> dict[str, Any]:
    """Describe a saved independent account, preserving undefined terminal logs.

    Args:
        account: Reconciled quantity-account trace.

    Returns:
        Period, cash, exposure, trading, risk, and separately annualized measures.
    """
    rows = account['trace']
    if not rows:
        raise ValueError('Issue 43 metrics: empty account trace')
    equity = np.array([rows[0]['equity_start'], *[r['equity_end'] for r in rows]])
    costs = np.array([r['markup'] + sum(t['spread'] + t['commission'] for t in r['trades']) for r in rows])
    if not np.isfinite(equity).all() or not np.isfinite(costs).all():
        raise ValueError('Issue 43 metrics: nonfinite cash path')
    elapsed = (datetime.fromisoformat(account['actual_last_mark']) - datetime.fromisoformat(account['first_decision'])).total_seconds()
    years = elapsed / (365.25 * 86400)
    positive = bool(np.all(equity > 0))
    logs = np.log(equity[1:] / equity[:-1]) if positive else None
    gross_valid = bool(np.all(equity[:-1] > 0) and np.all(equity[1:] + costs > 0))
    gross_log = float(np.log((equity[1:] + costs) / equity[:-1]).sum()) if gross_valid else None
    period_log = float(np.log(equity[-1] / equity[0])) if positive else None
    annual = account['status'] == 'complete' and positive and years > 0
    volatility = float(np.std(logs, ddof=0) * np.sqrt(len(rows) / years)) if annual else None
    sharpe = None
    if annual and len(rows) > 1 and float(np.std(logs, ddof=1)) > 0:
        sharpe = float(np.mean(logs) / np.std(logs, ddof=1) * np.sqrt(len(rows) / years))
    targets = np.array([np.array(r['decision']['quantities']) * r['prices'] / r['equity_at_decision'] for r in rows])
    weights = np.array([r['execution_weights_after_cost'] for r in rows])
    turnover = float(np.abs(np.diff(np.vstack([np.zeros(9), targets]), axis=0)).sum())
    return {
        'elapsed_seconds': elapsed, 'elapsed_years': years, 'decisions': len(rows),
        'period_net_log': period_log, 'period_net_return': float(equity[-1] / equity[0] - 1),
        'period_gross_log': gross_log, 'period_gross_return': float(np.expm1(gross_log)) if gross_log is not None else None,
        'annualized_net_log': period_log / years if annual else None,
        'annualized_net_return': float(np.expm1(period_log / years)) if annual else None,
        'annualized_gross_return': float(np.expm1(gross_log / years)) if annual and gross_log is not None else None,
        'annualization_status': 'defined' if annual else 'undefined_terminal_or_nonpositive_equity',
        'annual_net_log_volatility': volatility, 'sharpe': sharpe,
        'sharpe_status': 'defined' if sharpe is not None else 'undefined_terminal_or_zero_variance',
        'max_drawdown': float(np.max(1 - equity / np.maximum.accumulate(equity))),
        'price_pnl': sum(r['price_pnl'] for r in rows), 'financing': sum(r['financing'] for r in rows),
        'spread': sum(t['spread'] for r in rows for t in r['trades']),
        'commission': sum(t['commission'] for r in rows for t in r['trades']),
        'markup': sum(r['markup'] for r in rows), 'cost_ratio': float(costs.sum() / equity[0]),
        'target_turnover': turnover, 'actual_traded_notional': sum(r['actual_traded_notional'] for r in rows),
        'actual_notional_ratio': sum(r['actual_traded_notional'] for r in rows) / equity[0],
        'literal_hold_decisions': sum(r['literal_hold'] for r in rows),
        'partial_allocation_decisions': sum(r['decision']['u'] is not None and 0 < abs(r['decision']['u']) < 1 for r in rows),
        'quantity_change_decisions': sum(any(t['quantities_before'] != t['quantities_after'] for t in r['trades']) for r in rows),
        'risk_override_decisions': sum(r['risk_override'] for r in rows),
        'mean_gross_exposure': float(np.abs(weights).sum(axis=1).mean()),
        'max_gross_exposure': float(np.abs(weights).sum(axis=1).max()),
        'mean_net_exposure': float(weights.sum(axis=1).mean()),
        'mean_pair_exposures': weights.mean(axis=0).tolist(),
        'terminal_reason': account['terminal_reason'], 'remaining_decisions': account['remaining_decisions'],
    }


@lru_cache(maxsize=512)
def evidence(values: tuple[float, ...]) -> dict[str, Any]:
    """Reuse deterministic fold statistics for identical observation vectors.

    Args:
        values: Exactly seventeen period logs or paired differences.

    Returns:
        Registered bootstrap, era, LOO, and concentration evidence.
    """
    result = paired_evidence(np.array(values), np.zeros(17))
    result['contribution_status'] = 'defined' if result['largest_absolute_contribution_fraction'] is not None else 'undefined_zero_absolute_sum'
    return result


def persistent_positive(item: dict[str, Any]) -> bool:
    """Check both eras and every leave-one-out mean.

    Args:
        item: Seventeen-fold evidence.

    Returns:
        Whether all registered robustness signs are strictly positive.
    """
    return min(item['eras'].values()) > 0 and item['minimum_leave_one_fold_out_mean'] > 0


def comparison_terms() -> dict[str, dict[str, float]]:
    """List the two primary and fifteen secondary contrasts.

    Returns:
        Policy coefficients for every registered comparison.
    """
    pairs = [(f'A{h}-cost', f'A{h}-fixed') for h in (1, 5)]
    pairs += [(f'A5-{mode}', f'A1-{mode}') for mode in ('fixed', 'cost')]
    pairs += [(candidate, control) for candidate in CANDIDATES for control in CONTROLS]
    return {**{f'{a}_minus_{b}': {a: 1., b: -1.} for a, b in pairs},
            'interaction': {'A5-cost': 1., 'A5-fixed': -1., 'A1-cost': -1., 'A1-fixed': 1.}}


def summarize(cells: list[dict[str, Any]], rules: dict[str, Any], predictive: dict[str, Any]) -> dict[str, Any]:
    """Apply sealed period-log selection without dropping any failed interval.

    Args:
        cells: All 357 registered account outcomes, including non-success states.
        rules: Immutable v3 numerical risk and decision contract.
        predictive: Separate sealed predictive gate for both horizons.

    Returns:
        Full comparisons and separate simple-policy, allocation, and RL decisions.
    """
    expected = {(c['fold'], c['scenario'], c['policy']) for c in registered_cells()}
    indexed = {(c['fold'], c['scenario'], c['policy']): c for c in cells}
    if len(cells) != 357 or len(indexed) != 357 or set(indexed) != expected:
        raise ValueError('Issue 43 matrix must contain all 357 unique registered cells')
    for cell in cells:
        if cell['status'] not in STATUSES:
            raise ValueError(f'Issue 43 matrix: unknown status {cell}')
        if cell['status'] == 'complete':
            metric = cell['metrics']
            if metric is None or any(metric[k] is None or not np.isfinite(metric[k]) for k in
                                     ('period_net_log', 'max_drawdown', 'annual_net_log_volatility', 'actual_notional_ratio')):
                raise ValueError(f'Issue 43 complete cell has undefined metrics: {cell}')
    vectors: dict[tuple[str, str], np.ndarray] = {}
    absolute: dict[str, Any] = {s: {} for s in SCENARIOS}
    risk: dict[str, Any] = {s: {} for s in SCENARIOS}
    for scenario in SCENARIOS:
        for policy in POLICIES:
            selected = [indexed[f, scenario, policy] for f in FOLDS]
            if all(c['status'] == 'complete' for c in selected):
                values = np.array([c['metrics']['period_net_log'] for c in selected])
                vectors[scenario, policy] = values
                absolute[scenario][policy] = {'status': 'available', 'evidence': evidence(tuple(values))}
                dd = [c['metrics']['max_drawdown'] for c in selected]
                vol = [c['metrics']['annual_net_log_volatility'] for c in selected]
                risk[scenario][policy] = {'status': 'available', 'worst_drawdown': max(dd),
                    'mean_drawdown': float(np.mean(dd)), 'mean_annual_net_log_volatility': float(np.mean(vol)),
                    'met': bool(max(dd) <= rules['max_worst_drawdown'] and np.mean(dd) <= rules['max_mean_drawdown']
                                and np.mean(vol) <= rules['max_mean_annual_net_log_volatility'])}
            else:
                absolute[scenario][policy] = {'status': 'pending_measurement', 'evidence': None,
                    'unavailable_folds': [c['fold'] for c in selected if c['status'] != 'complete']}
                risk[scenario][policy] = {'status': 'pending_measurement', 'met': False}
    comparisons: dict[str, Any] = {s: {} for s in SCENARIOS}
    for scenario in SCENARIOS:
        for name, terms in comparison_terms().items():
            available = all((scenario, p) in vectors for p in terms)
            comparisons[scenario][name] = {'status': 'available' if available else 'pending_measurement',
                'evidence': evidence(tuple(sum(coef * vectors[scenario, p] for p, coef in terms.items()))) if available else None}
    candidates: dict[str, Any] = {}
    stress: dict[str, Any] = {}
    for policy in POLICIES:
        stress[policy] = {}
        for scenario in ('F1', 'F2'):
            available = all((s, policy) in vectors for s in ('F0', scenario))
            stress[policy][scenario] = {'status': 'available' if available else 'pending_measurement',
                'evidence': evidence(tuple(vectors[scenario, policy] - vectors['F0', policy])) if available else None}
    for policy in CANDIDATES:
        selected = [c for c in cells if c['policy'] == policy]
        if any(c['status'] == 'strategy_terminal' for c in selected):
            candidates[policy] = {'status': 'rejected_terminal', 'failed_conditions': ['strategy_terminal']}
            continue
        if any(c['status'] != 'complete' for c in selected):
            candidates[policy] = {'status': 'pending_measurement', 'failed_conditions': ['measurement_incomplete']}
            continue
        f0 = absolute['F0'][policy]['evidence']
        checks = {'risk': all(risk[s][policy]['met'] for s in SCENARIOS),
                  'F0_level': f0['mean_difference'] >= rules['candidate_F0_min_mean_net_log'],
                  'eras_and_loo': persistent_positive(f0),
                  'stress_level': all(float(vectors[s, policy].mean()) >= rules['candidate_stress_min_mean_net_log'] for s in ('F1', 'F2')),
                  'stress_degradation': all(stress[policy][s]['evidence']['mean_difference'] >= rules['min_stress_minus_F0_mean_log'] for s in ('F1', 'F2'))}
        candidates[policy] = {'status': 'eligible' if all(checks.values()) else 'rejected',
                             'failed_conditions': [k for k, v in checks.items() if not v], 'checks': checks}
    added: dict[str, Any] = {}
    for h in (1, 5):
        name = f'A{h}-cost_minus_A{h}-fixed'
        items = [comparisons[s][name] for s in SCENARIOS]
        if any(x['evidence'] is None for x in items):
            added[str(h)] = {'status': 'pending_measurement'}
            continue
        item = items[0]['evidence']
        share = item['largest_absolute_contribution_fraction']
        checks = {'own_candidate': candidates[f'A{h}-cost']['status'] == 'eligible',
                  'improvement': item['mean_difference'] >= rules['min_mean_improvement_log'],
                  'eras_and_loo': persistent_positive(item),
                  'intervals': item['intervals']['fold_low'] > 0 and item['intervals']['moving_block_low'] > 0,
                  'stress': all(x['evidence']['mean_difference'] >= 0 for x in items[1:]),
                  'concentration': share is not None and share <= rules['max_absolute_fold_contribution_share']}
        added[str(h)] = {'status': 'supported' if all(checks.values()) else 'not_supported', 'checks': checks}
    eligible = [p for p in CANDIDATES if candidates[p]['status'] == 'eligible']
    unresolved = any(c['status'] != 'complete' for c in cells if c['policy'] in CANDIDATES)
    chosen: str | None = None
    if eligible and not unresolved:
        ranking = {p: (-absolute['F0'][p]['evidence']['mean_difference'],
                       max(risk[s][p]['worst_drawdown'] for s in SCENARIOS),
                       float(np.mean([indexed[f, 'F0', p]['metrics']['actual_notional_ratio'] for f in FOLDS])),
                       int(p[1]), 0 if p.endswith('fixed') else 1) for p in eligible}
        for index in range(5):
            best = min(ranking[p][index] for p in eligible)
            eligible = [p for p in eligible if ranking[p][index] - best <= rules['tie_tolerance']]
        chosen = eligible[0]
    reasons = []
    if not any(predictive[str(h)]['met'] for h in (1, 5)):
        reasons.append('no_predictive_room')
    if unresolved:
        reasons.append('measurement_incomplete')
    if any('risk' in c['failed_conditions'] or c['status'] == 'rejected_terminal' for c in candidates.values()):
        reasons.append('risk_failed')
    if any(any(k in c['failed_conditions'] for k in ('F0_level', 'eras_and_loo', 'stress_level', 'stress_degradation')) for c in candidates.values()):
        reasons.append('net_economic_conditions_failed')
    horizon = int(chosen[1]) if chosen is not None else None
    rl_ready = chosen is not None and predictive[str(horizon)]['met']
    return {'status': 'complete' if all(c['status'] == 'complete' for c in cells) else 'incomplete',
            'cells': cells, 'status_counts': {s: sum(c['status'] == s for c in cells) for s in STATUSES},
            'primary_unit': rules['primary_unit'], 'primary_comparisons': ['A1-cost_minus_A1-fixed', 'A5-cost_minus_A5-fixed'],
            'absolute': absolute, 'comparisons': comparisons, 'risk': risk, 'stress_sensitivity': stress,
            'candidates': candidates, 'allocation_added_value': added, 'selected_candidate': chosen,
            'selected_horizon': horizon, 'predictive_gate': predictive, 'stop_reasons': reasons,
            'independent_profitability': 'not_established',
            'handoff_issue49': {'status': 'candidate' if chosen else ('pending_measurement' if unresolved else 'not_planned'), 'candidate': chosen},
            'handoff_issue46': {'status': 'hypothesis_for_registration' if rl_ready else ('pending_measurement' if unresolved else 'not_planned'),
                'horizon': horizon if rl_ready else None, 'training_authorized': False,
                'hypothesis': 'Can the same forecast and four actions reduce future replanning costs beyond myopic utility?' if rl_ready else None},
            'limitations': ['Unadjusted development evidence; total independent historical trial count unknown.',
                           '17 fixed ranges include 9 partial years; equal-interval means are not continuous CAGR.',
                           'Same caps do not establish matched risk. Same-close inverse-price quantity is a research assumption.',
                           'Repaired prices and unknown carry vintage remain source limitations; no independent profitability recognition.']}
