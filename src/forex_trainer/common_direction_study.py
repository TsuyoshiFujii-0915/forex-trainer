"""Seal, execute once, and verify the Issue 43 short comparison campaign."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Literal

import numpy as np

from .artifact_provenance import sha256_file
from .common_allocation import Costs, Forecast, State, load_forecasts, proportional_cap
from .common_basket import PAIRS
from .common_basket_study import CAMPAIGN, ROOT, checked, digest, execution_runtime, git, read, require_merge, write
from .common_direction import CANDIDATES, POLICIES, SCENARIOS, account_metrics, comparison_terms, registered_cells, summarize
from .common_quantity import MEASUREMENT_ID, Mark, QuantityExecutionError, QuantityPolicy, legacy_policy, marks_from_period, replay
from .common_quantity_study import allocation_policy, append_event, audit_account, verify_fixture
from .full_period import canonical_action
from .full_period_sources import FoldSources
from .input_recovery import load_snapshot_sources
from .supervised_portfolio import FOLDS

RUN = ROOT / 'runs/issue43-common-direction-v3'
SAVED = ROOT / 'docs/research/results/issue43/campaign'
CONFIG = ROOT / 'configs/research/issue43_common_direction.json'
DEPENDENCY = 'e33d3417f0c5e98fa348d86eecc31a9ce16a3adc'


@dataclass(frozen=True)
class Inputs:
    """Complete preflight inputs; constructing these performs no inference."""

    config: dict[str, Any]
    registration: dict[str, Any]
    sources: list[FoldSources]
    forecasts: dict[tuple[str, int, str], Forecast]
    marks: dict[str, tuple[Mark, ...]]
    identities: dict[str, Any]
    predictive: dict[str, Any]


def scenario_costs(scenario: str) -> Costs:
    """Construct the registered independent-account cost treatment.

    Args:
        scenario: F0, spread-only F1, or markup-only F2.

    Returns:
        Exact nine-pair spread, commission, and holding markup rates.
    """
    if scenario not in SCENARIOS:
        raise ValueError(f'Issue 43 unknown scenario: {scenario}')
    spreads = np.array([.000015, .000025, .000030, .000030, .000035, .000035, .000040, .000060, .000060])
    return Costs(spreads * (2 if scenario == 'F1' else 1), 0., .00002 * (2 if scenario == 'F2' else 1))


def require_unused_budget(run: Path, saved: Path) -> None:
    """Reject previous reservations across local runs and committed artifacts.

    Args:
        run: Campaign-fixed output location.
        saved: Committed output location checked also in fresh clones.
    """
    for path in (run, saved):
        if path.exists():
            raise ValueError(f'Issue 43 budget already reserved: {path}; retry requires separate registration')


def contract(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """Validate scope, sealed dependencies, and unchanged economic thresholds.

    Args:
        path: Explicit Issue 43 config.

    Returns:
        Config and immutable v3 parent registration.
    """
    config = read(path)
    fixed = {'campaign_id': CAMPAIGN, 'measurement_id': MEASUREMENT_ID,
             'run_directory': str(RUN.relative_to(ROOT)), 'policies': list(POLICIES),
             'scenarios': list(SCENARIOS), 'candidate_accounts': 204, 'control_accounts': 153,
             'fit_calls': 0, 'ppo_training': 0, 'shared_retry_accounts': 0}
    references = {'registration', 'snapshot', 'forecast_manifest', 'quantity_manifest'}
    if set(config) != set(fixed) | references or any(config[k] != v for k, v in fixed.items()):
        raise ValueError(f'Issue 43 scope/budget mismatch: {path}')
    for key in references:
        checked(config[key])
    if config['registration'] != {'path': 'docs/research/protocols/issue40/revisions/v3/registration.json',
                                  'sha256': '5d096705e68536faa48eadf7a701e7d034f918e5203b45fb8191bbefeae6e8aa'}:
        raise ValueError('Issue 43 requires the immutable v3 economic and split contract')
    registration = read(checked(config['registration']))
    if registration['gates']['short_data_ready']['status'] != 'ready':
        raise ValueError('Issue 43 short data gate is not ready')
    for key, limit in (('issue43_candidate_accounts', 204), ('issue43_control_reevaluations', 153)):
        if registration['counters'][key] != {'limit': limit, 'consumed': 0}:
            raise ValueError(f'Issue 43 parent budget changed: {key}')
    if [f['fold'] for f in registration['folds']] != list(FOLDS):
        raise ValueError('Issue 43 requires the 17 fixed intervals')
    parent_config = read(ROOT / 'configs/research/issue41_common_basket.json')
    if config['snapshot'] != parent_config['source_snapshot']:
        raise ValueError('Issue 43 source snapshot differs from Issue 41')
    return config, registration


def load_inputs(config_path: Path) -> Inputs:
    """Validate all data, models, forecasts, calendars and accounting evidence.

    Args:
        config_path: Registered campaign config.

    Returns:
        Full 17-fold inputs without fits, policy inference, or account execution.
    """
    config, registration = contract(config_path)
    forecasts = load_forecasts(checked(config['forecast_manifest']).parent)
    verify_fixture(checked(config['quantity_manifest']).parent)
    sources = load_snapshot_sources(checked(config['snapshot']).parent)
    if [s.fold for s in sources] != list(FOLDS):
        raise ValueError('Issue 43 control sources require all 17 ordered folds')
    marks: dict[str, tuple[Mark, ...]] = {}
    identities: dict[str, Any] = {}
    for source, fold in zip(sources, registration['folds'], strict=True):
        axis = marks_from_period(source.plan)
        if (axis[0].at != fold['first_decision'] or axis[-1].at != fold['last_mark']
                or len(axis) != fold['measured_bars'] or source.plan.symbols != PAIRS):
            raise ValueError(f'Issue 43 market coverage/pair mismatch: {source.fold}')
        for h in (1, 5):
            expected = {(source.fold, h, m.at) for m in axis[:-1]}
            if {k for k in forecasts if k[:2] == (source.fold, h)} != expected:
                raise ValueError(f'Issue 43 decision/forecast axis mismatch: {source.fold}/h{h}')
        marks[source.fold] = axis
        identities[source.fold] = {'source': source.identity, 'ridge_parameter_sha256': source.ridge.parameter_sha256,
            'input_sha256': digest([{'at': m.at, 'prices': m.prices.tolist(), 'carry': m.carry.tolist(), 'window': m.window.tolist()} for m in axis]),
            'pairs': list(PAIRS), 'first_decision': axis[0].at, 'last_mark': axis[-1].at,
            'measurement_start': source.plan.measurement_start, 'measurement_end': source.plan.measurement_end,
            'history_labels': list(source.plan.history_labels), 'scope': fold['scope']}
    diagnostics = read(checked(config['forecast_manifest']).parent / 'diagnostics.json')['cells']
    predictive: dict[str, Any] = {}
    for h in (1, 5):
        values = []
        for fold in FOLDS:
            found = [c for c in diagnostics if (c['fold'], c['horizon']) == (fold, h)]
            if len(found) != 1 or found[0]['diagnostics']['status'] != 'available':
                raise ValueError(f'Issue 43 predictive diagnostic missing: {fold}/h{h}')
            values.append(found[0]['diagnostics']['mse_minus_zero'])
        if not np.isfinite(values).all():
            raise ValueError(f'Issue 43 predictive diagnostic nonfinite: h{h}')
        means = {'all': float(np.mean(values)), '2009-2018': float(np.mean(values[:10])), '2019-2025': float(np.mean(values[10:]))}
        predictive[str(h)] = {'met': all(v < 0 for v in means.values()), 'mean_mse_minus_zero': means}
    return Inputs(config, registration, sources, forecasts, marks, identities, predictive)


def policy_source(inputs: Inputs, cell: dict[str, Any]) -> dict[str, Any]:
    """Bind each account to its original model lineage and treatment.

    Args:
        inputs: Verified complete inputs.
        cell: Registered account key.

    Returns:
        Shared forecast identity or unchanged frozen-control source identity.
    """
    fold, policy = cell['fold'], cell['policy']
    source = {'policy': policy, 'fold': fold, 'scenario': cell['scenario'],
              'training_lineage_changed': False, 'source': inputs.identities[fold]}
    if policy in CANDIDATES:
        h = int(policy[1])
        source['forecast_manifest'] = inputs.config['forecast_manifest']
        source['model_sha256'] = inputs.forecasts[fold, h, inputs.marks[fold][0].at].model_sha256
    return source


def bind_policy(inputs: Inputs, source: FoldSources, policy: str) -> QuantityPolicy:
    """Connect a frozen forecast or control to a fresh scenario account.

    Args:
        inputs: Verified market and forecast bundle.
        source: This fold's original trained controls.
        policy: Registered treatment.

    Returns:
        Causal callable, with no fitted or replayed historical actions.
    """
    if policy in CANDIDATES:
        mode: Literal['fixed', 'cost'] = 'fixed' if policy.endswith('fixed') else 'cost'
        return allocation_policy({at: f for (fold, h, at), f in inputs.forecasts.items()
                                  if fold == source.fold and h == int(policy[1])}, mode)
    controls = {'canonical': canonical_action, 'ridge': source.ridge.action, 'ppo_ens3': source.ppo.action}
    return legacy_policy(controls[policy])


def audit_result(account: dict[str, Any], inputs: Inputs, cell: dict[str, Any], runtime_hash: str, costs: Costs) -> None:
    """Audit accounting, decision state, model identity and exact market coverage.

    Args:
        account: Saved account, never inferred again during verification.
        inputs: Independently validated source inputs.
        cell: Registered account identity.
        runtime_hash: Required common measurement/runtime seal.
        costs: Independently registered scenario rates, including fixture rates.
    """
    origin = f"{cell['fold']}/{cell['scenario']}/{cell['policy']}"
    if (account['measurement_id'] != MEASUREMENT_ID or account['runtime_sha256'] != runtime_hash
            or account['policy_source'] != policy_source(inputs, cell)):
        raise ValueError(f'Issue 43 account source/runtime mismatch: {origin}')
    marks = inputs.marks[cell['fold']]
    audit_account(account, origin, marks, costs)
    before = np.zeros(9)
    for i, row in enumerate(account['trace']):
        trades = row['trades']
        if not trades or len(trades) > 2:
            raise ValueError(f'Issue 43 invalid trade count: {origin}/{i}')
        pre = State(row['at'], PAIRS, row['equity_start'], before, marks[i].prices, marks[i].carry, costs)
        capped = proportional_cap(pre, before)
        drift = not np.array_equal(capped, before)
        if drift != (len(trades) == 2) or (drift and trades[0]['reason'] != 'risk_drift'):
            raise ValueError(f'Issue 43 risk drift mismatch: {origin}/{i}')
        equity = row['equity_start'] - (trades[0]['spread'] + trades[0]['commission'] if drift else 0.)
        state = State(row['at'], PAIRS, equity, capped, marks[i].prices, marks[i].carry, costs)
        decision = row['decision']
        applied = proportional_cap(state, np.array(decision['quantities']))
        override = not np.array_equal(applied, decision['quantities'])
        expected_reason = 'risk_proposal' if override else decision['mode']
        if (decision['state_sha256'] != state.identity or row['equity_at_decision'] != equity
                or trades[-1]['reason'] != expected_reason or not np.allclose(applied, row['quantities'], rtol=1e-12, atol=1e-8)
                or row['risk_override'] != (drift or override)
                or row['literal_hold'] != (decision['mode'] == 'hold_quantity' and row['actual_traded_notional'] == 0)):
            raise ValueError(f'Issue 43 decision/risk mismatch: {origin}/{i}')
        expected_weights = applied * marks[i].prices / (equity - trades[-1]['spread'] - trades[-1]['commission'])
        if not np.allclose(row['execution_weights_after_cost'], expected_weights, rtol=1e-12, atol=1e-12):
            raise ValueError(f'Issue 43 exposure mismatch: {origin}/{i}')
        if cell['policy'] in CANDIDATES:
            forecast = inputs.forecasts[cell['fold'], int(cell['policy'][1]), row['at']]
            if digest(decision['forecast_identity']) != digest(forecast.identity):
                raise ValueError(f'Issue 43 forecast/model changed: {origin}/{i}')
        elif decision['forecast_identity'] is not None:
            raise ValueError(f'Issue 43 control has a foreign forecast: {origin}/{i}')
        before = np.array(row['quantities'])


def activation_record(inputs: Inputs, runtime: dict[str, Any]) -> dict[str, Any]:
    """Describe the entire experiment before its first evaluation.

    Args:
        inputs: Complete, read-only preflight inputs.
        runtime: Clean execution runtime.

    Returns:
        Fixed inputs, budgets, cell inventory, risk rules and measurement seal.
    """
    return {'campaign_id': CAMPAIGN, 'measurement_id': MEASUREMENT_ID, 'dependency_merge_commit': DEPENDENCY,
            'runtime': runtime, 'config': inputs.config, 'config_sha256': sha256_file(CONFIG),
            'identities': inputs.identities, 'rules': inputs.registration['risk_and_decision'],
            'predictive_gate': inputs.predictive, 'cells': registered_cells(),
            'candidate_accounts': 204, 'control_accounts': 153, 'fit_calls': 0, 'ppo_training': 0}


def preflight(config_path: Path, output: Path) -> dict[str, Any]:
    """Save readiness of all cells without spending the evaluation budget.

    Args:
        config_path: Explicit registered configuration.
        output: New directory for a complete or blocked preflight.

    Returns:
        Input readiness, including independence from the middle-term RL gate.
    """
    output.mkdir(parents=True, exist_ok=False)
    cells = registered_cells()
    try:
        inputs = load_inputs(config_path)
    except Exception as exc:
        error = f'{type(exc).__name__}: {exc}'
        for cell in cells:
            cell.update(status='blocked_input', error=error)
        write(output / 'preflight.json', {'status': 'blocked_input', 'cells': cells, 'error': error, 'accounts_executed': 0})
        raise
    result = {'status': 'ready', 'cells': cells, 'accounts_executed': 0,
              'input_identity_sha256': digest(inputs.identities), 'predictive_gate': inputs.predictive,
              'short_data_ready': inputs.registration['gates']['short_data_ready'],
              'rl_data_ready': inputs.registration['gates']['rl_data_ready'], 'rl_is_a_dependency': False}
    write(output / 'preflight.json', result)
    return result


def seal(config_path: Path, output: Path) -> dict[str, Any]:
    """Seal clean implementation and all source bytes before evaluation.

    Args:
        config_path: Explicit configuration.
        output: New activation directory.

    Returns:
        Activation required by the one permitted normal run.
    """
    require_unused_budget(RUN, SAVED)
    inputs = load_inputs(config_path)
    require_merge(inputs.registration, DEPENDENCY)
    git(['merge-base', '--is-ancestor', DEPENDENCY, 'HEAD'], ROOT)
    activation = activation_record(inputs, execution_runtime())
    output.mkdir(parents=True, exist_ok=False)
    write(output / 'activation.json', activation)
    return activation


def render_report(report: dict[str, Any]) -> str:
    """Render every account and the separate adoption decisions.

    Args:
        report: Full report with all successful and unsuccessful cells.

    Returns:
        Reviewable Markdown, with links to the complete statistics artifact.
    """
    lines = ['# Issue 43 short comparison', '', f"Status: {report['status']}",
             f"Selected simple candidate: {report['selected_candidate']}",
             f"Issue 46: {report['handoff_issue46']['status']}; Issue 49: {report['handoff_issue49']['status']}",
             f"Stop reasons: {', '.join(report['stop_reasons'])}", '',
             'Primary unit: equal-weight mean of 17 period net log returns. See [report.json](report.json) for every CI, era, LOO, risk, exposure and cost statistic.', '',
             '| Scenario | Contrast | Mean period log difference | IID 95% | Block 95% |',
             '|---|---|---:|---|---|']
    for scenario in SCENARIOS:
        for name in comparison_terms():
            item = report['comparisons'][scenario][name]
            value = item['evidence']
            detail = 'pending | unavailable | unavailable' if value is None else f"{value['mean_difference']:.8f} | {value['intervals']['fold_low']:.8f}, {value['intervals']['fold_high']:.8f} | {value['intervals']['moving_block_low']:.8f}, {value['intervals']['moving_block_high']:.8f}"
            lines.append(f'| {scenario} | {name} | {detail} |')
    lines += ['', '| Fold | Scenario | Policy | Status | Period net log | Annualized net log | MDD |', '|---|---|---|---|---:|---:|---:|']
    for cell in report['cells']:
        metrics = cell['metrics']
        values = ['unavailable' if metrics is None or metrics[k] is None else f'{metrics[k]:.8f}'
                  for k in ('period_net_log', 'annualized_net_log', 'max_drawdown')]
        lines.append(f"| {cell['fold']} | {cell['scenario']} | {cell['policy']} | {cell['status']} | " + ' | '.join(values) + ' |')
    lines += ['', *report['limitations']]
    return '\n'.join(lines) + '\n'


def report_with_incident(cells: list[dict[str, Any]], activation: dict[str, Any], incident: dict[str, Any] | None) -> dict[str, Any]:
    """Prevent integrity failures from authorizing decisions on completed paths.

    Args:
        cells: Entire registered matrix.
        activation: Pre-execution seal with immutable rules.
        incident: Explicit execution/integrity failure, if one occurred.

    Returns:
        Statistical report with any unresolved execution failure made explicit.
    """
    report = summarize(cells, activation['rules'], activation['predictive_gate'])
    report['incident'] = incident
    if incident is not None:
        report.update(status='incomplete', selected_candidate=None, selected_horizon=None)
        report['handoff_issue49'] = {'status': 'pending_measurement', 'candidate': None}
        report['handoff_issue46'] = {'status': 'pending_measurement', 'horizon': None, 'training_authorized': False, 'hypothesis': None}
    return report


def finish(directory: Path, cells: list[dict[str, Any]], activation: dict[str, Any], incident: dict[str, Any] | None) -> dict[str, Any]:
    """Persist complete or incomplete evidence without overwriting prior results.

    Args:
        directory: Reserved campaign directory.
        cells: Every registered cell and its actual status.
        activation: Frozen pre-execution contract.
        incident: Recorded failure or explicit absence.

    Returns:
        Saved report, with separate Issue 46 and Issue 49 handoff artifacts.
    """
    report = report_with_incident(cells, activation, incident)
    write(directory / 'report.json', report)
    (directory / 'report.md').write_text(render_report(report))
    for issue in (46, 49):
        write(directory / f'issue{issue}-handoff.json', report[f'handoff_issue{issue}'])
    events = [json.loads(line) for line in (directory / 'account-ledger.jsonl').read_text().splitlines()]
    reserved = [e for e in events if e['event'] == 'reserved']
    manifest = {'campaign_id': CAMPAIGN, 'measurement_id': MEASUREMENT_ID, 'status': report['status'],
                'runtime_sha256': digest(activation['runtime']), 'candidate_accounts': sum(e['counter'] == 'issue43_candidate_accounts' for e in reserved),
                'control_accounts': sum(e['counter'] == 'issue43_control_reevaluations' for e in reserved),
                'fit_calls': 0, 'ppo_training': 0, 'shared_retry_accounts': 0,
                'artifacts': {str(p.relative_to(directory)): sha256_file(p) for p in sorted(directory.rglob('*')) if p.is_file()}}
    manifest['content_sha256'] = digest(manifest)
    write(directory / 'manifest.json', manifest)
    return report


def run(config_path: Path, activation_path: Path) -> dict[str, Any]:
    """Execute each independent scenario account once and preserve any failure.

    Args:
        config_path: Registered config, unchanged since sealing.
        activation_path: Clean runtime and input seal.

    Returns:
        Complete result or a report including strategy terminals.
    """
    require_unused_budget(RUN, SAVED)
    activation = read(activation_path)
    config, registration = contract(config_path)
    require_merge(registration, DEPENDENCY)
    if activation['runtime'] != execution_runtime() or activation['config'] != config:
        raise ValueError('Issue 43 activation differs from current runtime/config')
    RUN.mkdir(parents=True, exist_ok=False)
    (RUN / 'accounts').mkdir()
    ledger = RUN / 'account-ledger.jsonl'
    ledger.touch(exist_ok=False)
    write(RUN / 'activation.json', activation)
    cells = registered_cells()
    write(RUN / 'registered-cells.json', cells)
    active: dict[str, Any] | None = None
    started = False
    try:
        inputs = load_inputs(config_path)
        if activation != activation_record(inputs, activation['runtime']):
            raise ValueError('Issue 43 input/measurement/budget activation mismatch')
        runtime_hash = digest(activation['runtime'])
        sources = {s.fold: s for s in inputs.sources}
        for cell in cells:
            active = cell
            name = f"{cell['fold']}-{cell['scenario']}-{cell['policy']}"
            counter = 'issue43_candidate_accounts' if cell['policy'] in CANDIDATES else 'issue43_control_reevaluations'
            append_event(ledger, {'event': 'reserved', 'work_item': name, 'counter': counter, 'attempt': 1, 'runtime_sha256': runtime_hash})
            started = True
            account = replay(inputs.marks[cell['fold']], scenario_costs(cell['scenario']),
                             bind_policy(inputs, sources[cell['fold']], cell['policy']), runtime_hash, policy_source(inputs, cell))
            result_path = f'accounts/{name}.json.gz'
            write(RUN / result_path, account)
            cell['result_path'] = result_path
            audit_result(account, inputs, cell, runtime_hash, scenario_costs(cell['scenario']))
            cell.update(status=account['status'], metrics=account_metrics(account))
            append_event(ledger, {'event': 'finished', 'work_item': name, 'status': cell['status']})
            active = None
            if cell['policy'] == POLICIES[-1]:
                print(f"{cell['fold']}/{cell['scenario']}: 7 independent accounts saved", flush=True)
        if execution_runtime() != activation['runtime']:
            raise ValueError('Issue 43 runtime changed during evaluation')
        refreshed = load_inputs(config_path)
        if activation_record(refreshed, activation['runtime']) != activation:
            raise ValueError('Issue 43 source bytes changed during evaluation')
    except (Exception, KeyboardInterrupt) as exc:
        error = f'{type(exc).__name__}: {exc}'
        if active is not None:
            active.update(status='execution_error', error=error)
            append_event(ledger, {'event': 'finished', 'work_item': f"{active['fold']}-{active['scenario']}-{active['policy']}", 'status': 'execution_error'})
        if not started:
            for cell in cells:
                cell.update(status='blocked_input', error=error)
        incident = {'error': error, 'account_evidence': exc.evidence if isinstance(exc, QuantityExecutionError) else None,
                    'automatic_retry': False, 'resume_condition': 'Resolve the explicit incident; any consumed account retry requires a separate shared-budget registration.'}
        finish(RUN, cells, activation, incident)
        raise
    return finish(RUN, cells, activation, None)


def verify(directory: Path) -> dict[str, Any]:
    """Verify complete or incomplete saved evidence without account inference.

    Args:
        directory: Original run or byte-identical committed copy.

    Returns:
        Fully reconciled report; hash or cash violations remain exceptions.
    """
    manifest = read(directory / 'manifest.json')
    if digest({k: v for k, v in manifest.items() if k != 'content_sha256'}) != manifest['content_sha256']:
        raise ValueError('Issue 43 manifest hash mismatch')
    inventory = {str(p.relative_to(directory)) for p in directory.rglob('*') if p.is_file()} - {'manifest.json'}
    if inventory != set(manifest['artifacts']):
        raise ValueError('Issue 43 artifact inventory mismatch')
    for name, expected in manifest['artifacts'].items():
        path = (directory / name).resolve()
        if not path.is_relative_to(directory.resolve()) or sha256_file(path) != expected:
            raise ValueError(f'Issue 43 artifact hash mismatch: {name}')
    activation = read(directory / 'activation.json')
    config, registration = contract(CONFIG)
    if (activation['config'] != config or activation['rules'] != registration['risk_and_decision']
            or activation['cells'] != registered_cells() or read(directory / 'registered-cells.json') != registered_cells()
            or manifest['runtime_sha256'] != digest(activation['runtime'])
            or manifest['campaign_id'] != CAMPAIGN or manifest['measurement_id'] != MEASUREMENT_ID
            or any(manifest[k] != 0 for k in ('fit_calls', 'ppo_training', 'shared_retry_accounts'))):
        raise ValueError('Issue 43 saved scope/runtime/rules mismatch')
    inputs = load_inputs(CONFIG)
    if activation_record(inputs, activation['runtime']) != activation:
        raise ValueError('Issue 43 saved source activation mismatch')
    report = read(directory / 'report.json')
    required = {'activation.json', 'registered-cells.json', 'account-ledger.jsonl',
                'report.json', 'report.md', 'issue46-handoff.json', 'issue49-handoff.json'}
    required.update(c['result_path'] for c in report['cells'] if c['result_path'] is not None)
    if inventory != required:
        raise ValueError('Issue 43 unregistered or missing report artifact')
    events = [json.loads(line) for line in (directory / 'account-ledger.jsonl').read_text().splitlines()]
    reservations = [e for e in events if e['event'] == 'reserved']
    completions = [e for e in events if e['event'] == 'finished']
    expected_names = [f"{c['fold']}-{c['scenario']}-{c['policy']}" for c in report['cells'] if c['status'] in ('complete', 'strategy_terminal', 'execution_error')]
    if ([e['work_item'] for e in reservations] != expected_names or [e['work_item'] for e in completions] != expected_names
            or len(events) != len(reservations) * 2):
        raise ValueError('Issue 43 reservation/outcome ledger mismatch')
    for i, (reservation, completion) in enumerate(zip(reservations, completions, strict=True)):
        if events[2 * i] != reservation or events[2 * i + 1] != completion:
            raise ValueError('Issue 43 reservation must precede its outcome')
        policy = reservation['work_item'].split('-', 2)[2]
        counter = 'issue43_candidate_accounts' if policy in CANDIDATES else 'issue43_control_reevaluations'
        if reservation['counter'] != counter or reservation['attempt'] != 1 or reservation['runtime_sha256'] != manifest['runtime_sha256']:
            raise ValueError('Issue 43 ledger counter/runtime mismatch')
    for cell in report['cells']:
        name = f"{cell['fold']}-{cell['scenario']}-{cell['policy']}"
        if cell['status'] in ('complete', 'strategy_terminal'):
            if cell['result_path'] != f'accounts/{name}.json.gz':
                raise ValueError(f'Issue 43 account path mismatch: {name}')
            account = read(directory / cell['result_path'])
            audit_result(account, inputs, cell, manifest['runtime_sha256'], scenario_costs(cell['scenario']))
            if cell['metrics'] != account_metrics(account) or cell['status'] != account['status']:
                raise ValueError(f'Issue 43 cell metrics/status mismatch: {name}')
        if cell['status'] in ('complete', 'strategy_terminal', 'execution_error'):
            if next(e for e in completions if e['work_item'] == name)['status'] != cell['status']:
                raise ValueError(f'Issue 43 ledger completion status mismatch: {name}')
    for kind, counter in (('candidate_accounts', 'issue43_candidate_accounts'), ('control_accounts', 'issue43_control_reevaluations')):
        if manifest[kind] != sum(e['counter'] == counter for e in reservations):
            raise ValueError('Issue 43 consumed budget mismatch')
    rebuilt = report_with_incident(report['cells'], activation, report['incident'])
    if rebuilt != report or manifest['status'] != report['status'] or (directory / 'report.md').read_text() != render_report(report):
        raise ValueError('Issue 43 statistical report differs from saved observations')
    for issue in (46, 49):
        if read(directory / f'issue{issue}-handoff.json') != report[f'handoff_issue{issue}']:
            raise ValueError(f'Issue 43 handoff mismatch: {issue}')
    return report


def main() -> int:
    """Expose bounded execution and read-only verification.

    Returns:
        Zero for readiness/completion, two for an explicitly incomplete report.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    for command in ('preflight', 'seal', 'run'):
        entry = sub.add_parser(command)
        entry.add_argument('--config', type=Path, required=True)
        entry.add_argument('--activation' if command == 'run' else '--output', type=Path, required=True)
    entry = sub.add_parser('verify')
    entry.add_argument('--directory', type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'verify':
        result = verify(args.directory)
    elif args.command == 'run':
        result = run(args.config, args.activation)
    elif args.command == 'seal':
        seal(args.config, args.output)
        print('Issue 43 activation sealed; zero accounts executed')
        return 0
    else:
        result = preflight(args.config, args.output)
    print(json.dumps({k: result[k] for k in ('status', 'selected_candidate', 'status_counts') if k in result}, sort_keys=True))
    return 0 if result['status'] in ('ready', 'complete') else 2


if __name__ == '__main__':
    raise SystemExit(main())
