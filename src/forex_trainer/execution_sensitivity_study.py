"""Independent registration and bounded execution of Issue 45 bar sensitivity."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import subprocess
from typing import Any

import numpy as np

from .artifact_provenance import sha256_file
from .common_basket_study import ROOT, checked, digest, execution_runtime, git, read, write
from .common_direction import CONTROLS, SCENARIOS, evidence
from .common_direction_study import scenario_costs
from .common_quantity import MEASUREMENT_ID as SAME_CLOSE_ID
from .common_quantity import Mark, QuantityExecutionError, legacy_policy, marks_from_period, replay
from .common_quantity_study import append_event, audit_account, verify_fixture
from .execution_sensitivity import MEASUREMENT_ID, audit_proxy, input_identity, replay_next_close
from .full_period import canonical_action
from .full_period_sources import FoldSources
from .input_recovery import load_snapshot_sources
from .supervised_portfolio import FOLDS

CONFIG = ROOT / 'configs/research/issue45_execution_sensitivity.json'
CONFIG_SHA256 = '6678a37ac38695e16d7a2cd0af50ad8989ecf240b949970b0758a9a3f0f36819'
RUN = ROOT / 'runs/issue45-execution-sensitivity-v1'
SAVED = ROOT / 'docs/research/results/issue45/campaign'
EXECUTIONS = ('same_close', 'next_close')
STATUSES = ('not_run', 'blocked_input', 'execution_error', 'complete', 'strategy_terminal')


@dataclass(frozen=True)
class Inputs:
    """Validated fixed inputs; loading never infers or trains a policy."""

    config: dict[str, Any]
    sources: list[FoldSources]
    marks: dict[str, tuple[Mark, ...]]
    identities: dict[str, Any]


def contract(path: Path) -> dict[str, Any]:
    """Validate the independently registered exact scope and dependencies.

    Args:
        path: Explicit registration JSON.

    Returns:
        Fixed proxy and blocked quote contracts.
    """
    if sha256_file(path) != CONFIG_SHA256:
        raise ValueError(f'Issue 45 contract/budget bytes differ: {path}')
    config = read(path)
    for key in ('snapshot', 'quantity_fixture', 'parent_registration'):
        checked(config[key])
    checked(config['quote']['source_bundle'])
    return config


def registered_cells() -> list[dict[str, Any]]:
    """Enumerate every proxy account before any inference.

    Returns:
        All 306 cells, including both independently inferred conditions.
    """
    return [{'fold': f, 'scenario': s, 'policy': p, 'execution': e, 'status': 'not_run',
             'metrics': None, 'result_path': None, 'error': None}
            for f in FOLDS for s in SCENARIOS for p in CONTROLS for e in EXECUTIONS]


def cell_name(cell: dict[str, Any]) -> str:
    """Identify one registered account.

    Args:
        cell: Fixed fold, cost, policy and execution tuple.

    Returns:
        Unique safe artifact basename.
    """
    return '-'.join(cell[k] for k in ('fold', 'scenario', 'policy', 'execution'))


def require_registration_merge(merge_commit: str) -> None:
    """Reject activation until this registration and implementation are merged.

    Args:
        merge_commit: Full reviewed merge SHA on fetched origin/main.
    """
    if len(merge_commit) != 40 or any(c not in '0123456789abcdef' for c in merge_commit):
        raise ValueError('Issue 45 registration merge requires a full SHA')
    try:
        git(['merge-base', '--is-ancestor', merge_commit, 'origin/main'], ROOT)
        git(['merge-base', '--is-ancestor', merge_commit, 'HEAD'], ROOT)
        saved = git(['show', f'{merge_commit}:{CONFIG.relative_to(ROOT)}'], ROOT)
        if json.loads(saved) != contract(CONFIG):
            raise ValueError('Issue 45 merge contains a different registration')
        git(['diff', '--exit-code', merge_commit, 'HEAD', '--',
             'src/forex_trainer/execution_sensitivity.py',
             'src/forex_trainer/execution_sensitivity_study.py'], ROOT)
    except subprocess.CalledProcessError as exc:
        raise ValueError('Issue 45 registration/implementation merge is not verified on origin/main') from exc


def load_inputs(config_path: Path) -> Inputs:
    """Validate all source bytes, causal windows, models and 17 fixed ranges.

    Args:
        config_path: Fixed independent execution registration.

    Returns:
        Complete proxy inputs with no model inference or account evaluation.
    """
    config = contract(config_path)
    verify_fixture(checked(config['quantity_fixture']).parent)
    sources = load_snapshot_sources(checked(config['snapshot']).parent)
    if [s.fold for s in sources] != list(FOLDS):
        raise ValueError('Issue 45 sources must contain all 17 ordered folds')
    marks: dict[str, tuple[Mark, ...]] = {}
    identities: dict[str, Any] = {}
    for source, spec in zip(sources, config['proxy']['folds'], strict=True):
        axis = marks_from_period(source.plan)
        if (spec['fold'] != source.fold or axis[0].at != spec['first_decision']
                or axis[-1].at != spec['last_mark'] or len(axis) != spec['measured_bars']
                or source.plan.measurement_start != spec['measurement_start']
                or source.plan.measurement_end != spec['measurement_end']):
            raise ValueError(f'Issue 45 registered range mismatch: {source.fold}')
        marks[source.fold] = axis
        identities[source.fold] = {'source': source.identity, 'input_sha256': input_identity(axis),
                                   'ridge_parameter_sha256': source.ridge.parameter_sha256, 'range': spec}
    return Inputs(config, sources, marks, identities)


def summarize(cells: list[dict[str, Any]]) -> dict[str, Any]:
    """Compare full fold vectors while retaining all non-success states.

    Args:
        cells: All registered cells, including failures and not-run accounts.

    Returns:
        Paired descriptive fold statistics and separate blocked quote handoff.
    """
    expected = {cell_name(c) for c in registered_cells()}
    indexed = {cell_name(c): c for c in cells}
    if len(cells) != 306 or set(indexed) != expected:
        raise ValueError('Issue 45 matrix requires all 306 unique cells')
    for c in cells:
        if c['status'] not in STATUSES:
            raise ValueError(f'Issue 45 unknown status: {cell_name(c)}')
        if c['status'] == 'complete':
            if c['metrics'] is None or any(c['metrics'][k] is None or not np.isfinite(c['metrics'][k])
                                          for k in ('period_net_log', 'period_gross_log')):
                raise ValueError(f'Issue 45 complete cell has undefined metrics: {cell_name(c)}')
    comparisons: list[dict[str, Any]] = []
    for s in SCENARIOS:
        for p in CONTROLS:
            selected = [[indexed[f'{f}-{s}-{p}-{e}'] for f in FOLDS] for e in EXECUTIONS]
            ready = all(c['status'] == 'complete' for group in selected for c in group)
            comparison: dict[str, Any] = {'scenario': s, 'policy': p,
                'contrast': 'next_close_minus_same_close', 'status': 'available' if ready else 'pending_measurement',
                'net': None, 'gross': None, 'cash_and_risk_mean_differences': None,
                'missing_cells': [cell_name(c) for group in selected for c in group if c['status'] != 'complete']}
            if ready:
                for name, metric in [('net', 'period_net_log'), ('gross', 'period_gross_log')]:
                    comparison[name] = evidence(tuple(b['metrics'][metric] - a['metrics'][metric]
                                                       for a, b in zip(*selected, strict=True)))
                comparison['cash_and_risk_mean_differences'] = {
                    metric: float(np.mean([b['metrics'][metric] - a['metrics'][metric]
                                           for a, b in zip(*selected, strict=True)]))
                    for metric in ('gap_price_pnl', 'spread', 'commission', 'markup',
                                   'actual_traded_notional', 'max_drawdown', 'mean_gross_exposure')}
            comparisons.append(comparison)
    quote_cells = [{'fold': '2025', 'scenario': 'F0', 'policy': p, 'execution': e,
                    'status': 'blocked_input', 'reason': 'issue44_observed_inputs_and_source_contract_missing'}
                   for p in CONTROLS for e in ('same_close', 'observed_quote')]
    return {'experiment_id': 'issue45-execution-sensitivity-v1',
            'proxy_status': 'complete' if all(c['status'] in ('complete', 'strategy_terminal') for c in cells) else 'incomplete',
            'quote_status': 'blocked_input', 'issue45_complete': False, 'cells': cells,
            'status_counts': {s: sum(c['status'] == s for c in cells) for s in STATUSES},
            'comparisons': comparisons, 'quote_cells': quote_cells, 'candidate_validation': 'not_performed',
            'interpretation': 'Fixed historical execution-rule sensitivity; not observed fill or causal market latency.',
            'handoff': {'issues': [43, 48, 50], 'new_candidates_validated': 0,
                        'additional_trial_required': 'At most one candidate with a separately registered finite budget.',
                        'unresolved': ['observed_quotes', 'provider_product_lot_financing_depth', 'asynchronous_baskets_and_partial_fills']}}


def metrics(result: dict[str, Any], execution: str) -> dict[str, Any]:
    """Describe saved cash, exposure, cost, delays and complete coverage.

    Args:
        result: Audited account trace.
        execution: Registered same-close or next-close condition.

    Returns:
        Explicit net/gross and execution diagnostics without hidden flat values.
    """
    rows = result['trace']
    equity = np.array([rows[0]['equity_start'], *[r['equity_end'] for r in rows]])
    costs = np.array([r['markup'] + sum(t['spread'] + t['commission'] for t in r['trades']) for r in rows])
    positive = bool(np.all(equity > 0))
    gross_valid = bool(np.all(equity[:-1] > 0) and np.all(equity[1:] + costs > 0))
    weights_defined = all(r['execution_weights_after_cost'] is not None for r in rows)
    weights = np.array([r['execution_weights_after_cost'] for r in rows]) if weights_defined else None
    orders = [o for r in rows for o in r['orders']] if execution == 'next_close' else []
    return {'period_net_log': float(np.log(equity[-1] / equity[0])) if positive else None,
            'period_gross_log': float(np.log((equity[1:] + costs) / equity[:-1]).sum()) if gross_valid else None,
            'gross_definition': 'cost_addback_on_realized_path_not_counterfactual_cost_free_policy',
            'price_pnl': sum(r['price_pnl'] for r in rows), 'financing': sum(r['financing'] for r in rows),
            'gap_price_pnl': sum(r['gap_price_pnl'] for r in rows) if execution == 'next_close' else 0.,
            'spread': sum(t['spread'] for r in rows for t in r['trades']),
            'commission': sum(t['commission'] for r in rows for t in r['trades']),
            'markup': sum(r['markup'] for r in rows),
            'actual_traded_notional': sum(r['actual_traded_notional'] for r in rows),
            'max_drawdown': float(np.max(1 - equity / np.maximum.accumulate(equity))),
            'mean_gross_exposure': float(np.abs(weights).sum(axis=1).mean()) if weights is not None else None,
            'max_gross_exposure': float(np.abs(weights).sum(axis=1).max()) if weights is not None else None,
            'exposure_status': 'defined' if weights_defined else 'undefined_nonpositive_equity',
            'decisions': len(rows), 'remaining_decisions': result['remaining_decisions'],
            'decision_coverage': len(rows) / (len(rows) + result['remaining_decisions']),
            'order_status_counts': {s: sum(o['status'] == s for o in orders)
                                    for s in ('filled', 'expired', 'cancelled_terminal')},
            'fill_delays_seconds': [r['elapsed_days'] * 86400 for r in rows if r['orders'][0]['status'] == 'filled']
                                  if execution == 'next_close' else [0.] * len(rows),
            'terminal_reason': result['terminal_reason']}


def preflight(config_path: Path, output: Path) -> dict[str, Any]:
    """Save input readiness without activating or consuming market accounts.

    Args:
        config_path: Independent fixed registration.
        output: New result directory.

    Returns:
        Full not-run matrix, source hashes, and blocked quote status.
    """
    output.mkdir(parents=True, exist_ok=False)
    cells = registered_cells()
    try:
        inputs = load_inputs(config_path)
    except Exception as exc:
        for c in cells:
            c.update(status='blocked_input', error=f'{type(exc).__name__}: {exc}')
        write(output / 'preflight.json', {**summarize(cells), 'input_status': 'blocked_input', 'accounts_executed': 0})
        raise
    result = {**summarize(cells), 'input_status': 'ready', 'identities': inputs.identities,
              'config_sha256': sha256_file(config_path), 'accounts_executed': 0,
              'activation_status': 'requires_registration_merge_and_clean_runtime_seal',
              'fit_calls': 0, 'external_requests': 0}
    write(output / 'preflight.json', result)
    return result


def require_unused_budget() -> None:
    """Reject reuse of reserved accounts in local runs or committed evidence."""
    for path in (RUN, SAVED):
        if path.exists():
            raise ValueError(f'Issue 45 budget already reserved: {path}; no automatic retry')


def seal(config_path: Path, merge_commit: str, output: Path) -> dict[str, Any]:
    """Activate only this merged experiment at a clean runtime.

    Args:
        config_path: Fixed registration.
        merge_commit: Reviewed registration/implementation merge SHA.
        output: New activation directory.

    Returns:
        Runtime, source, scope and budget seal.
    """
    require_registration_merge(merge_commit)
    require_unused_budget()
    inputs = load_inputs(config_path)
    activation = {'config': inputs.config, 'config_sha256': sha256_file(config_path),
                  'registration_merge_commit': merge_commit, 'runtime': execution_runtime(),
                  'identities': inputs.identities, 'cells': registered_cells(),
                  'effective_proxy_accounts': 306, 'effective_quote_accounts': 0}
    output.mkdir(parents=True, exist_ok=False)
    write(output / 'activation.json', activation)
    return activation


def finish(directory: Path, cells: list[dict[str, Any]], incident: dict[str, Any] | None) -> dict[str, Any]:
    """Persist complete or incomplete experiment status and every artifact hash.

    Args:
        directory: Reserved run directory.
        cells: All 306 cells with current status.
        incident: Explicit execution failure, or None.

    Returns:
        Persisted report, with independent quote and candidate handoffs.
    """
    report = {**summarize(cells), 'incident': incident}
    write(directory / 'report.json', report)
    write(directory / 'handoff.json', report['handoff'])
    events = [json.loads(line) for line in (directory / 'account-ledger.jsonl').read_text().splitlines()]
    manifest = {'experiment_id': 'issue45-execution-sensitivity-v1',
                'proxy_accounts_consumed': sum(e['event'] == 'reserved' for e in events),
                'quote_accounts_consumed': 0, 'fit_calls': 0, 'retry_accounts': 0,
                'artifacts': {str(p.relative_to(directory)): sha256_file(p)
                              for p in sorted(directory.rglob('*')) if p.is_file()}}
    manifest['content_sha256'] = digest(manifest)
    write(directory / 'manifest.json', manifest)
    return report


def run(config_path: Path, activation_path: Path) -> dict[str, Any]:
    """Execute each registered proxy account once, saving failures before raising.

    Args:
        config_path: Fixed registration.
        activation_path: Merged clean-runtime activation.

    Returns:
        Complete report, unless an explicit incident stops the finite attempt.
    """
    require_unused_budget()
    config = contract(config_path)
    activation = read(activation_path)
    require_registration_merge(activation['registration_merge_commit'])
    if (activation['config'] != config or activation['config_sha256'] != sha256_file(config_path)
            or activation['runtime'] != execution_runtime() or activation['cells'] != registered_cells()
            or activation['effective_proxy_accounts'] != 306 or activation['effective_quote_accounts'] != 0):
        raise ValueError('Issue 45 activation/runtime/scope mismatch')
    inputs = load_inputs(config_path)
    if activation['identities'] != inputs.identities:
        raise ValueError('Issue 45 activated inputs changed')
    RUN.mkdir(parents=True, exist_ok=False)
    (RUN / 'accounts').mkdir()
    write(RUN / 'activation.json', activation)
    ledger = RUN / 'account-ledger.jsonl'
    append_event(ledger, {'event': 'budget_reserved', 'proxy_limit': 306, 'quote_limit': 0})
    cells = registered_cells()
    active: dict[str, Any] | None = None
    runtime_hash = digest(activation['runtime'])
    try:
        source_by_fold = {s.fold: s for s in inputs.sources}
        for cell in cells:
            source = source_by_fold[cell['fold']]
            predictors = {'canonical': canonical_action, 'ridge': source.ridge.action, 'ppo_ens3': source.ppo.action}
            costs = scenario_costs(cell['scenario'])
            active = cell
            name = cell_name(cell)
            append_event(ledger, {'event': 'reserved', 'work_item': name})
            policy_source = {'policy': cell['policy'], 'fold': cell['fold'], 'scenario': cell['scenario'],
                             'source': inputs.identities[cell['fold']], 'training_lineage_changed': False}
            engine = replay if cell['execution'] == 'same_close' else replay_next_close
            result = engine(inputs.marks[cell['fold']], costs, legacy_policy(predictors[cell['policy']]),
                            runtime_hash, policy_source)
            path = RUN / 'accounts' / f'{name}.json.gz'
            write(path, result)
            cell['result_path'] = str(path.relative_to(RUN))
            audit = audit_account if cell['execution'] == 'same_close' else audit_proxy
            if cell['execution'] == 'same_close':
                audit(result, name, inputs.marks[cell['fold']], costs)
            else:
                audit(result, inputs.marks[cell['fold']], costs, name)
            cell.update(status=result['status'], metrics=metrics(result, cell['execution']))
            append_event(ledger, {'event': 'finished', 'work_item': name, 'status': cell['status']})
            active = None
        if execution_runtime() != activation['runtime']:
            raise ValueError('Issue 45 runtime changed during execution')
    except Exception as exc:
        error = f'{type(exc).__name__}: {exc}'
        if active is not None:
            active.update(status='execution_error', error=error, metrics=None)
            append_event(ledger, {'event': 'finished', 'work_item': cell_name(active), 'status': 'execution_error'})
        incident = {'error': error, 'cell': cell_name(active) if active is not None else None,
                    'partial': exc.evidence if isinstance(exc, QuantityExecutionError) else None,
                    'automatic_retry': False}
        write(RUN / 'incident.json', incident)
        finish(RUN, cells, incident)
        raise
    return finish(RUN, cells, None)


def verify(directory: Path) -> dict[str, Any]:
    """Verify saved complete or incomplete artifacts without policy inference.

    Args:
        directory: Existing campaign directory.

    Returns:
        Explicit validation result and consumed account count.
    """
    manifest = read(directory / 'manifest.json')
    content = {k: v for k, v in manifest.items() if k != 'content_sha256'}
    if digest(content) != manifest['content_sha256']:
        raise ValueError('Issue 45 manifest content hash mismatch')
    for name, expected in manifest['artifacts'].items():
        path = directory / name
        if not path.resolve().is_relative_to(directory.resolve()) or sha256_file(path) != expected:
            raise ValueError(f'Issue 45 artifact hash/path mismatch: {name}')
    inventory = {str(p.relative_to(directory)) for p in directory.rglob('*') if p.is_file() and p != directory / 'manifest.json'}
    if inventory != set(manifest['artifacts']):
        raise ValueError('Issue 45 artifact inventory mismatch')
    activation, report = read(directory / 'activation.json'), read(directory / 'report.json')
    inputs = load_inputs(CONFIG)
    if (activation['config'] != inputs.config or activation['config_sha256'] != CONFIG_SHA256
            or activation['identities'] != inputs.identities or activation['cells'] != registered_cells()
            or activation['effective_proxy_accounts'] != 306 or activation['effective_quote_accounts'] != 0):
        raise ValueError('Issue 45 saved activation/input mismatch')
    expected_report = {**summarize(report['cells']), 'incident': report['incident']}
    if report != expected_report or read(directory / 'handoff.json') != report['handoff']:
        raise ValueError('Issue 45 saved report reconciliation mismatch')
    if report['incident'] is not None and read(directory / 'incident.json') != report['incident']:
        raise ValueError('Issue 45 incident mismatch')
    events = [json.loads(line) for line in (directory / 'account-ledger.jsonl').read_text().splitlines()]
    reserved = [e['work_item'] for e in events if e['event'] == 'reserved']
    consumed = [cell_name(c) for c in report['cells'] if c['status'] in ('complete', 'strategy_terminal', 'execution_error')]
    if (reserved != consumed or reserved != [cell_name(c) for c in registered_cells()][:len(reserved)]
            or len(reserved) != manifest['proxy_accounts_consumed'] or manifest['quote_accounts_consumed'] != 0):
        raise ValueError('Issue 45 account ledger/budget mismatch')
    expected_events = [{'event': 'budget_reserved', 'proxy_limit': 306, 'quote_limit': 0}]
    for cell in report['cells']:
        if cell['status'] not in ('not_run', 'blocked_input'):
            expected_events += [{'event': 'reserved', 'work_item': cell_name(cell)},
                                {'event': 'finished', 'work_item': cell_name(cell), 'status': cell['status']}]
        if cell['status'] in ('complete', 'strategy_terminal'):
            result = read(directory / cell['result_path'])
            if (result['runtime_sha256'] != digest(activation['runtime'])
                    or result['policy_source'] != {'policy': cell['policy'], 'fold': cell['fold'], 'scenario': cell['scenario'],
                        'source': inputs.identities[cell['fold']], 'training_lineage_changed': False}
                    or result['measurement_id'] != (SAME_CLOSE_ID if cell['execution'] == 'same_close' else MEASUREMENT_ID)):
                raise ValueError(f'Issue 45 account identity mismatch: {cell_name(cell)}')
            costs = scenario_costs(cell['scenario'])
            axis = inputs.marks[cell['fold']]
            if cell['execution'] == 'same_close':
                audit_account(result, cell_name(cell), axis, costs)
            else:
                audit_proxy(result, axis, costs, cell_name(cell))
            if result['status'] != cell['status'] or metrics(result, cell['execution']) != cell['metrics']:
                raise ValueError(f'Issue 45 metrics mismatch: {cell_name(cell)}')
    if events != expected_events:
        raise ValueError('Issue 45 ledger event order mismatch')
    return {'status': 'verified', 'proxy_status': report['proxy_status'], 'quote_status': 'blocked_input',
            'proxy_accounts_consumed': len(reserved), 'accounts_reexecuted': 0}


def main() -> int:
    """Expose preflight, merge-gated seal, bounded run and read-only verification.

    Returns:
        Zero on success; errors propagate with artifact origins.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    for name in ('preflight', 'seal', 'run', 'verify'):
        command = sub.add_parser(name)
        if name == 'verify':
            command.add_argument('--directory', type=Path, required=True)
        else:
            command.add_argument('--config', type=Path, required=True)
            if name == 'run':
                command.add_argument('--activation', type=Path, required=True)
            else:
                command.add_argument('--output', type=Path, required=True)
                if name == 'seal':
                    command.add_argument('--registration-merge-commit', required=True)
    args = parser.parse_args()
    if args.command == 'preflight':
        result = preflight(args.config, args.output)
    elif args.command == 'seal':
        result = seal(args.config, args.registration_merge_commit, args.output)
    elif args.command == 'run':
        result = run(args.config, args.activation)
    else:
        result = verify(args.directory)
    print(json.dumps({k: v for k, v in result.items() if k not in ('cells', 'identities', 'config', 'runtime')}, indent=2, allow_nan=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
