"""Frozen successor F0 attribution with the two original independent controls."""
from __future__ import annotations

import argparse
import copy
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from .artifact_provenance import sha256_file
from .cost_campaign import account_trace, write_json
from .full_period import (
    Policy, build_period_env, canonical_action, checked_path, evaluate_policy,
    read_json, require_comparable, runtime_identity,
)
from .full_period_sources import FoldSources
from .input_recovery import load_snapshot_sources
from .profit_attribution import (
    BASE_POLICIES, COMPONENTS, CONTROLS, POLICIES, CommonProjection, ConstantAllocation,
    attribute_account, effective_proposal, read_compressed, summarize_attribution,
    train_allocation, verify_reproduction, write_compressed,
)
from .successor_cost_campaign import successor_metrics, verify_successor
from .supervised_portfolio import FOLDS

CAMPAIGN_ID = 'issue30-contiguous-attribution-v1'
TRAIN_GAP_POLICY = 'original_train_replay_with_observed_gaps'
REFERENCES = ('parent_manifest', 'handoff', 'snapshot', 'registration', 'decision', 'parent_contract', 'lock')
ParentAccounts = dict[str, tuple[dict[str, Any], list[dict[str, Any]]]]
STEP_COLUMNS = [
    'fold', 'policy', 'decision_timestamp', 'target_timestamp', 'equity_before', 'equity_jpy',
    'price_simple', 'net_simple', 'net_log', 'log_allocation_factor', 'log_undefined_reason',
    'gross_exposure', 'net_exposure', 'actual_traded_notional_jpy', 'weight_turnover',
    *[name + suffix for name in COMPONENTS for suffix in ('_simple', '_log', '_pnl_jpy')],
    'source_result_path', 'source_result_sha256', 'source_pairs_sha256',
]


def interrupted_fold_cells(cells: list[dict[str, Any]], failure: dict[str, Any]) -> list[dict[str, Any]]:
    """Complete the failed fold's state table without fabricating account results.

    Args:
        cells: Successfully saved prefix, including any terminal train block.
        failure: Explicit failing operation and original exception evidence.

    Returns:
        Five cells separating execution errors, dependencies and unstarted work.
    """
    fold, stage, policy = failure['fold'], failure['stage'], failure['policy']
    if stage == 'train_replay' and policy is None:
        count = len(BASE_POLICIES)
    elif stage == 'policy_replay' and policy in POLICIES:
        count = POLICIES.index(policy)
    else:
        raise ValueError(f'Unknown failed operation: {fold}/{stage}/{policy}')
    if ([c['policy'] for c in cells] != list(POLICIES[:count])
            or any(c['fold'] != fold for c in cells)):
        raise ValueError(f'Interrupted fold has an invalid saved prefix: {fold}')
    pending = []
    for name in POLICIES[count:]:
        if stage == 'policy_replay' and name == policy:
            status, reason = 'execution_error', 'policy_replay_failed'
        elif ((stage == 'train_replay' and name == 'train_constant')
              or (policy in BASE_POLICIES and name in CONTROLS)):
            status, reason = 'blocked_dependency', 'required_replay_failed'
        else:
            status, reason = 'not_run', 'stopped_after_execution_error'
        pending.append({'fold': fold, 'policy': name, 'status': status, 'reason': reason,
                        'failure_path': f'fold-{fold}/execution-error.json'})
    return [*cells, *pending]


def record_execution_error(work: Path, fold: str, cells: list[dict[str, Any]], stage: str,
                           policy: str | None, error: RuntimeError) -> list[dict[str, Any]]:
    """Persist an exception at the replay boundary and stop this fold.

    Args:
        work: Active fold directory.
        fold: Registered fold identifier, independent of the directory name.
        cells: Already saved cells, which must remain unchanged.
        stage: Policy replay or train replay.
        policy: Failing policy; null for the separate train operation.
        error: Original runtime error, retaining its traceback and cause chain.

    Returns:
        Explicit five-cell interrupted fold panel.
    """
    failure = {'fold': fold, 'stage': stage, 'policy': policy,
               'exception_type': type(error).__name__, 'message': str(error),
               'traceback': ''.join(traceback.format_exception(error))}
    result = interrupted_fold_cells(cells, failure)
    write_json(work / 'execution-error.json', failure)
    return result


def unstarted_fold_cells(fold: str, failure: dict[str, Any]) -> list[dict[str, Any]]:
    """Describe later folds without executing or assigning them market metrics.

    Args:
        fold: Unstarted registered fold.
        failure: Earlier error that stopped the campaign.

    Returns:
        Five explicit not-run cells linked to their stopping cause.
    """
    return [{'fold': fold, 'policy': policy, 'status': 'not_run',
             'reason': 'stopped_after_execution_error',
             'failure_path': f"fold-{failure['fold']}/execution-error.json"} for policy in POLICIES]


def verify_interrupted_fold(work: Path, cells: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Require recorded causes for every unsaved cell in an interrupted fold.

    Args:
        work: Fold artifact directory.
        cells: Five declared cell states.

    Returns:
        Validated runtime error, or null for a fold without a runtime failure.
    """
    path = work / 'execution-error.json'
    if not path.exists():
        if any(c['status'] in ('execution_error', 'blocked_dependency', 'not_run') or 'failure_path' in c for c in cells):
            raise ValueError(f'Missing execution failure evidence: {work}')
        return None
    failure = read_json(path)
    if (set(failure) != {'fold', 'stage', 'policy', 'exception_type', 'message', 'traceback'}
            or failure['fold'] != cells[0]['fold']
            or not isinstance(failure['exception_type'], str) or not failure['exception_type']
            or not isinstance(failure['message'], str)
            or not isinstance(failure['traceback'], str) or not failure['traceback']):
        raise ValueError(f'Invalid execution failure evidence: {path}')
    saved = [c for c in cells if c['status'] not in ('execution_error', 'blocked_dependency', 'not_run')]
    if interrupted_fold_cells(saved, failure) != cells:
        raise ValueError(f'Interrupted fold states contradict the recorded error: {work}')
    return failure


def validate_config(config: dict[str, Any]) -> None:
    """Reject unregistered scopes, controls and training-gap treatments.

    Args:
        config: Explicit successor registration.
    """
    if set(config) != {'campaign_id', 'scenario', 'folds', 'controls', 'train_gap_policy', *REFERENCES}:
        raise ValueError('Expected exact successor attribution config fields')
    if (config['campaign_id'] != CAMPAIGN_ID or config['scenario'] != 'F0'
            or config['folds'] != list(FOLDS) or config['controls'] != list(CONTROLS)
            or config['train_gap_policy'] != TRAIN_GAP_POLICY):
        raise ValueError('Successor attribution requires registered F0, 17 folds, two controls and original train gaps')


def summarize_successor_attribution(cells: list[dict[str, Any]], coverage: list[dict[str, Any]]) -> dict[str, Any]:
    """Retain existing fold statistics while explicitly limiting their scope.

    Args:
        cells: All 85 account observations or explicit failures.
        coverage: The unchanged 17 ordered successor ranges.

    Returns:
        Conditional successor evidence, never annual-attempt completion.
    """
    if [row['fold'] for row in coverage] != list(FOLDS):
        raise ValueError('Successor coverage requires 17 ordered folds')
    indexed = {row['fold']: row for row in coverage}
    for cell in cells:
        if cell['status'] == 'complete':
            for field in ('first_decision', 'last_mark'):
                if cell['coverage'][field] != indexed[cell['fold']][field]:
                    raise ValueError(f"Successor coverage differs: {cell['fold']}/{field}")
    if any(row['successor_scope'] not in ('full_year', 'partial_year') for row in coverage):
        raise ValueError('Unknown successor coverage scope')
    report = summarize_attribution(cells)
    report.update({'campaign_id': CAMPAIGN_ID, 'original_annual_campaign_complete': False,
                   'original_blocked_policy_folds': 45, 'coverage': coverage,
                   'scope_counts': {kind: sum(r['successor_scope'] == kind for r in coverage) for kind in ('full_year', 'partial_year')},
                   'train_gap_policy': TRAIN_GAP_POLICY,
                   'statistics_interpretation': 'Equal-weight observations conditional on the 17 frozen contiguous ranges; not full-year population evidence or continuous-operation CAGR.'})
    if report['status'] == 'complete':
        rows = {(c['fold'], c['policy']): c for c in cells}
        for policy in POLICIES:
            for era, folds in (('all', FOLDS), ('2009-2018', FOLDS[:10]), ('2019-2025', FOLDS[10:])):
                for name in ('period_net_return', 'period_gross_return', 'annualized_net_volatility'):
                    report['summaries'][policy][era]['mean_metrics'][name] = float(np.mean([rows[f, policy]['metrics'][name] for f in folds]))
    return report


def replay_account(source: FoldSources, policy: str, predict: Policy, work: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Evaluate one independently reset account with existing accounting.

    Args:
        source: Hash-verified frozen market and policies.
        policy: Registered policy name.
        predict: This account's closed-loop policy callable.
        work: New fold artifact directory.

    Returns:
        Reconciled account and its pair-level monetary trace.
    """
    cache = work / f'{policy}.parquet'
    env, windows = build_period_env(source.env_raw, source.plan, cache)
    try:
        result = evaluate_policy(env, windows, source.plan, predict)
    finally:
        env.close()
        cache.unlink()
    result.update({'fold': source.fold, 'policy': policy, 'scenario': 'F0'})
    return result, account_trace(result, source.plan)


def save_account(result: dict[str, Any], pairs: list[dict[str, Any]], work: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Seal one account and retain exact source hashes in its attribution.

    Args:
        result: Reconciled F0 account.
        pairs: Ordered pair trace.
        work: Fold output directory.

    Returns:
        Fold observation and per-step attribution with source identities.
    """
    steps, attribution = attribute_account(result, pairs)
    path = work / f"{result['policy']}.json.gz"
    pair_path = work / f"{result['policy']}-pairs.csv.gz"
    write_compressed(path, result)
    pd.DataFrame(pairs).to_csv(pair_path, index=False, compression={'method': 'gzip', 'mtime': 0})
    identity = {'source_result_path': str(path), 'source_result_sha256': sha256_file(path),
                'source_pairs_sha256': sha256_file(pair_path)}
    cell = {key: result[key] for key in ('fold', 'policy', 'status', 'coverage')}
    cell.update({'metrics': successor_metrics(result), 'attribution': attribution,
                 'result_path': path.name, 'pairs_path': pair_path.name, **identity})
    return cell, [{**step, **identity} for step in steps]


def evaluate_fold(source: FoldSources, parents: ParentAccounts, work: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Reproduce three parent accounts and replay the original two controls.

    Args:
        source: One frozen successor fold.
        parents: All three complete sealed F0 accounts and pair traces.
        work: Previously nonexistent fold output directory.

    Returns:
        Five explicit cell statuses and all available attribution steps.
    """
    if set(parents) != set(BASE_POLICIES) or any(r['status'] != 'complete' for r, _ in parents.values()):
        raise ValueError(f'Complete parent F0 required before successor controls: {source.fold}')
    work.mkdir()
    cells, steps, reproduced = [], [], []
    for policy, predict in (('canonical', canonical_action), ('ridge', source.ridge.action), ('ppo_ens3', source.ppo.action)):
        original, original_pairs = parents[policy]
        attribute_account(original, original_pairs)
        try:
            result, pairs = replay_account(source, policy, predict, work)
        except RuntimeError as error:
            return record_execution_error(work, source.fold, cells, 'policy_replay', policy, error), steps
        verify_reproduction(original, result)
        pd.testing.assert_frame_equal(pd.DataFrame(original_pairs), pd.DataFrame(pairs), check_exact=True)
        cell, rows = save_account(result, pairs, work)
        cells.append(cell)
        steps.extend(rows)
        reproduced.append(result)
    raw = yaml.safe_load(checked_path({'path': source.identity['config_path'], 'sha256': source.identity['config_sha256']}).read_text())
    try:
        training = train_allocation(raw, source.ppo.action)
    except RuntimeError as error:
        return record_execution_error(work, source.fold, cells, 'train_replay', None, error), steps
    write_compressed(work / 'train-replay.json.gz', training)
    cap = float(source.env_raw['environment']['max_leverage'])
    adapters: dict[str, Policy] = {'common_projected': CommonProjection(source.ppo.action, cap).action}
    if training['status'] == 'complete':
        adapters['train_constant'] = ConstantAllocation(np.asarray(training['weights']), cap).action
    else:
        cells.append({'fold': source.fold, 'policy': 'train_constant', 'status': 'blocked_train_terminal',
                      'error': 'Train PPO terminated; partial train means are prohibited'})
    for policy in CONTROLS:
        if policy not in adapters:
            continue
        try:
            result, pairs = replay_account(source, policy, adapters[policy], work)
        except RuntimeError as error:
            return record_execution_error(work, source.fold, cells, 'policy_replay', policy, error), steps
        if result['status'] == 'complete':
            require_comparable([*reproduced, result])
        cell, rows = save_account(result, pairs, work)
        cells.append(cell)
        steps.extend(rows)
    cells.sort(key=lambda cell: POLICIES.index(cell['policy']))
    return cells, steps


def verify_training(source: FoldSources, work: Path) -> dict[str, Any]:
    """Verify a required completed or terminal train replay without inference.

    Args:
        source: Verified original train inputs.
        work: Fold directory containing the train replay.

    Returns:
        Reconciled training result; missing or changed evidence raises an error.
    """
    training = read_compressed(work / 'train-replay.json.gz')
    raw = yaml.safe_load(checked_path({'path': source.identity['config_path'], 'sha256': source.identity['config_sha256']}).read_text())
    if training['train_range'] != raw['train_range'] or training['symbols'] != raw['env']['environment']['currency_pairs']:
        raise ValueError(f'Train source boundary/pair mismatch: {source.fold}')
    data = source.identity['data_identity']
    market = pd.read_parquet(checked_path({'path': data['path'], 'sha256': data['sha256']}))
    dates = market.index.tz_localize(None) if market.index.tz is not None else market.index
    start, end = (pd.Timestamp(raw['train_range'][k]) for k in ('start', 'end'))
    selected = dates[(dates >= start) & (dates < end)]
    trace = training['trace']
    if not trace or trace[0]['equity_before'] != 1_000_000 or np.any(trace[0]['assets_before']):
        raise ValueError(f'Train replay reset differs: {source.fold}')
    cap = float(source.env_raw['environment']['max_leverage'])
    for i, row in enumerate(trace):
        decision, target = (pd.Timestamp(row[k]) for k in ('decision_timestamp', 'target_timestamp'))
        decision = decision.tz_localize(None) if decision.tz is not None else decision
        target = target.tz_localize(None) if target.tz is not None else target
        if i + 64 >= len(selected) or decision != selected[i + 63] or target != selected[i + 64] or not start <= decision < target < end:
            raise ValueError(f'Train-only timestamp boundary differs: {source.fold}/{i}')
        np.testing.assert_allclose(row['target_weights'], effective_proposal(np.asarray(row['action']).reshape(9, 1), cap), rtol=1e-10, atol=1e-12)
        if i and row['equity_before'] != trace[i - 1]['equity_jpy']:
            raise ValueError(f'Train equity continuity differs: {source.fold}/{i}')
    if training['status'] == 'complete':
        if len(trace) != len(selected) - 64 or any(r['terminated'] for r in trace):
            raise ValueError(f'Train replay incomplete: {source.fold}')
        np.testing.assert_array_equal(training['weights'], np.mean([r['target_weights'] for r in trace], axis=0))
        ConstantAllocation(np.asarray(training['weights']), cap)
    elif training['status'] != 'incomplete_margin_call' or training['weights'] is not None or not trace[-1]['terminated']:
        raise ValueError(f'Invalid train failure: {source.fold}')
    return training


def verify_fold(source: FoldSources, parents: ParentAccounts, work: Path, cells: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Reconcile saved accounts, train means, control actions and all fold metrics.

    Args:
        source: Original verified fold inputs.
        parents: Sealed parent F0 accounts.
        work: Existing fold directory.
        cells: Saved five-policy observations.

    Returns:
        Reconstructed attribution steps, without policy inference.
    """
    if len(cells) != 5 or {c['policy'] for c in cells} != set(POLICIES) or any(c['fold'] != source.fold for c in cells):
        raise ValueError(f'Invalid attribution cell panel: {source.fold}')
    failure = verify_interrupted_fold(work, cells)
    training_required = (failure is None or (failure['stage'] == 'policy_replay' and failure['policy'] in CONTROLS))
    training = verify_training(source, work) if training_required else None
    if not training_required and (work / 'train-replay.json.gz').exists():
        raise ValueError(f'Unexpected train result after earlier replay failure: {work}')
    cap = float(source.env_raw['environment']['max_leverage'])
    results, steps = [], []
    for cell in cells:
        policy = cell['policy']
        if cell['status'] in ('execution_error', 'blocked_dependency', 'not_run'):
            continue
        if cell['status'] == 'blocked_train_terminal':
            if policy != 'train_constant' or training is None or training['status'] != 'incomplete_margin_call':
                raise ValueError(f'Invalid blocked attribution cell: {source.fold}/{policy}')
            continue
        path = checked_path({'path': str(work / cell['result_path']), 'sha256': cell['source_result_sha256']})
        pair_path = checked_path({'path': str(work / cell['pairs_path']), 'sha256': cell['source_pairs_sha256']})
        result = read_compressed(path)
        if result['fold'] != source.fold or result['policy'] != policy or result['status'] != cell['status']:
            raise ValueError(f'Account identity differs: {path}')
        reconstructed = copy.deepcopy(result)
        pairs = account_trace(reconstructed, source.plan)
        if reconstructed != result:
            raise ValueError(f'Account reconciliation differs: {path}')
        pd.testing.assert_frame_equal(pd.read_csv(pair_path, float_precision='round_trip'), pd.DataFrame(pairs), check_exact=True)
        rows, attribution = attribute_account(result, pairs)
        if attribution != cell['attribution'] or successor_metrics(result) != cell['metrics'] or result['coverage'] != cell['coverage']:
            raise ValueError(f'Attribution cell differs: {path}')
        if policy in BASE_POLICIES:
            verify_reproduction(parents[policy][0], result)
        for row in result['trace']:
            if policy == 'common_projected':
                expected = np.full(9, effective_proposal(np.asarray(row['scores']).reshape(9, 1), cap).mean(), dtype=np.float32)
                np.testing.assert_array_equal(row['action'], expected)
            elif policy == 'train_constant':
                if training is None or training['status'] != 'complete':
                    raise ValueError(f'Constant action without complete training: {path}')
                np.testing.assert_array_equal(row['action'], np.asarray(training['weights']).astype(np.float32))
        identity = {key: cell[key] for key in ('source_result_path', 'source_result_sha256', 'source_pairs_sha256')}
        steps.extend({**row, **identity} for row in rows)
        if result['status'] == 'complete':
            results.append(result)
    if results:
        require_comparable(results)
    return steps


def load_inputs(config: dict[str, Any]) -> tuple[list[FoldSources], list[dict[str, Any]], dict[str, Any]]:
    """Verify the parent handoff, source seal and unchanged research contract.

    Args:
        config: Registered successor input references.

    Returns:
        Frozen sources, coverage and complete parent report.
    """
    validate_config(config)
    for name in REFERENCES:
        checked_path(config[name])
    parent_root = Path(config['parent_manifest']['path']).parent
    parent = verify_successor(parent_root)
    parent_config = read_json(parent_root / 'campaign_snapshot.json')
    for name in ('snapshot', 'parent_contract', 'lock'):
        if config[name] != parent_config[name]:
            raise ValueError(f'Parent successor contract differs: {name}')
    if Path(config['handoff']['path']) != parent_root / 'attribution-handoff.json':
        raise ValueError('Handoff must belong to the sealed parent')
    handoff = read_json(checked_path(config['handoff']))
    f0 = [cell for cell in parent['cells'] if cell['scenario'] == 'F0']
    if handoff['F0_cells'] != f0 or len(f0) != 51 or any(c['status'] != 'complete' for c in f0):
        raise ValueError('Complete sealed successor F0 handoff required')
    snapshot_root = Path(config['snapshot']['path']).parent
    sources = load_snapshot_sources(snapshot_root)
    parent_manifest = read_json(parent_root / 'manifest.json')
    for source in sources:
        if source.identity != parent_manifest['folds'][source.fold]['source']:
            raise ValueError(f'Frozen parent source differs: {source.fold}')
    return sources, read_json(snapshot_root / 'coverage.json')['folds'], parent


def parent_accounts(parent_root: Path, fold: str) -> ParentAccounts:
    """Read the already verified F0 account and pair inventory for one fold.

    Args:
        parent_root: Hash-verified successor cost result directory.
        fold: Registered source fold.

    Returns:
        Three sealed accounts and exact float pair traces.
    """
    return {policy: (read_compressed(parent_root / f'fold-{fold}/F0-{policy}.json.gz'),
                     pd.read_csv(parent_root / f'fold-{fold}/F0-{policy}-pairs.csv.gz', float_precision='round_trip').to_dict('records'))
            for policy in BASE_POLICIES}


def run_successor_attribution(config_path: Path, output: Path) -> dict[str, Any]:
    """Execute one preregistered diagnostic, without retraining or reselection.

    Args:
        config_path: Immutable successor config.
        output: New result directory.

    Returns:
        Complete or explicitly incomplete successor attribution evidence.
    """
    if output.exists():
        raise FileExistsError(f'Output already exists: {output}')
    config = read_json(config_path)
    sources, coverage, _ = load_inputs(config)
    return run_prepared_attribution(config_path, output, sources, coverage)


def run_prepared_attribution(config_path: Path, output: Path, sources: list[FoldSources],
                             coverage: list[dict[str, Any]]) -> dict[str, Any]:
    """Execute and seal the panel after strict input preflight has succeeded.

    Args:
        config_path: Preflighted configuration and parent references.
        output: New attempt directory.
        sources: All 17 preflighted frozen fold sources.
        coverage: Registered coverage, in the same fold order.

    Returns:
        Sealed complete or interrupted report; runtime failure stops new work.
    """
    if output.exists():
        raise FileExistsError(f'Output already exists: {output}')
    if [s.fold for s in sources] != list(FOLDS) or [r['fold'] for r in coverage] != list(FOLDS):
        raise ValueError('Prepared attribution requires all 17 ordered source/coverage folds')
    config = read_json(config_path)
    validate_config(config)
    parent_root = Path(config['parent_manifest']['path']).parent
    output.mkdir(parents=True)
    runtime = runtime_identity()
    manifest = {'campaign_id': CAMPAIGN_ID, 'config_sha256': sha256_file(config_path), 'runtime': runtime,
                'parent_manifest': config['parent_manifest'], 'parent_runtime': read_json(parent_root / 'manifest.json')['runtime'],
                'started_at': datetime.now(timezone.utc).isoformat(),
                'command': ['forex-successor-attribution', 'run', '--config', str(config_path), '--output', str(output)],
                'fold_sources': {s.fold: s.identity for s in sources}}
    write_json(output / 'campaign_snapshot.json', config)
    write_json(output / 'coverage.json', {'folds': coverage})
    write_json(output / 'started.json', manifest)
    cells, steps = [], []
    failure: dict[str, Any] | None = None
    for source in sources:
        if failure is not None:
            cells.extend(unstarted_fold_cells(source.fold, failure))
            continue
        fold_cells, fold_steps = evaluate_fold(source, parent_accounts(parent_root, source.fold), output / f'fold-{source.fold}')
        cells.extend(fold_cells)
        steps.extend(fold_steps)
        if any('failure_path' in c for c in fold_cells):
            failure = read_json(output / f'fold-{source.fold}/execution-error.json')
        print(f"{source.fold}: {sum(c['status'] == 'complete' for c in fold_cells)}/5 successor attribution accounts", flush=True)
    if runtime_identity() != runtime:
        raise ValueError('Runtime changed during attribution; preserve the incomplete attempt')
    report = summarize_successor_attribution(cells, coverage)
    write_json(output / 'report.json', report)
    pd.json_normalize(cells).to_csv(output / 'folds.csv', index=False)
    pd.DataFrame(steps, columns=STEP_COLUMNS).to_csv(output / 'steps.csv.gz', index=False, compression={'method': 'gzip', 'mtime': 0})
    manifest['finished_at'] = datetime.now(timezone.utc).isoformat()
    manifest['generated_artifact_sha256'] = {str(p.relative_to(output)): sha256_file(p) for p in sorted(output.rglob('*')) if p.is_file()}
    write_json(output / 'manifest.json', manifest)
    return report


def verify_successor_attribution(output: Path) -> dict[str, Any]:
    """Verify saved provenance, all accounts and aggregate evidence without inference.

    Args:
        output: Existing sealed successor attribution directory.

    Returns:
        Reconciled report, or an explicit exception at the damaged source.
    """
    sources, coverage, _ = load_inputs(read_json(output / 'campaign_snapshot.json'))
    return verify_prepared_attribution(output, sources, coverage)


def verify_prepared_attribution(output: Path, sources: list[FoldSources],
                                coverage: list[dict[str, Any]]) -> dict[str, Any]:
    """Reconcile attempt artifacts against preflighted inputs without inference.

    Args:
        output: Complete or interrupted attempt directory.
        sources: Preflighted sources, also for intentionally unstarted folds.
        coverage: Original fixed coverage.

    Returns:
        Verified report, preserving incomplete status and its recorded cause.
    """
    if [s.fold for s in sources] != list(FOLDS):
        raise ValueError('Verification requires all 17 ordered sources')
    manifest = read_json(output / 'manifest.json')
    if manifest['campaign_id'] != CAMPAIGN_ID:
        raise ValueError('Unexpected attribution manifest identity')
    inventory = {str(p.relative_to(output)) for p in output.rglob('*') if p.is_file() and p.name != 'manifest.json'}
    if inventory != set(manifest['generated_artifact_sha256']):
        raise ValueError('Attribution artifact inventory differs')
    for name, digest in manifest['generated_artifact_sha256'].items():
        checked_path({'path': str(output / name), 'sha256': digest})
    config_path = output / 'campaign_snapshot.json'
    checked_path({'path': str(config_path), 'sha256': manifest['config_sha256']})
    config = read_json(config_path)
    if read_json(output / 'coverage.json') != {'folds': coverage}:
        raise ValueError('Saved coverage differs from frozen successor')
    report = read_json(output / 'report.json')
    if summarize_successor_attribution(report['cells'], coverage) != report:
        raise ValueError('Attribution summary differs')
    steps = []
    stopped: dict[str, Any] | None = None
    for source in sources:
        if source.identity != manifest['fold_sources'][source.fold]:
            raise ValueError(f'Saved source identity differs: {source.fold}')
        work = output / f'fold-{source.fold}'
        cells = [c for c in report['cells'] if c['fold'] == source.fold]
        if stopped is not None:
            if cells != unstarted_fold_cells(source.fold, stopped) or work.exists():
                raise ValueError(f'Unexpected execution or states after campaign stop: {source.fold}')
            continue
        stopped = verify_interrupted_fold(work, cells)
        steps.extend(verify_fold(source, parent_accounts(Path(config['parent_manifest']['path']).parent, source.fold), work, cells))
        for cell in cells:
            if 'result_path' in cell and read_compressed(work / cell['result_path'])['runtime'] != manifest['runtime']:
                raise ValueError(f'Attribution runtime differs: {source.fold}/{cell["policy"]}')
    expected = pd.DataFrame(steps, columns=STEP_COLUMNS)
    actual = pd.read_csv(output / 'steps.csv.gz', float_precision='round_trip', dtype={'fold': str})
    pd.testing.assert_frame_equal(actual, expected, check_exact=True, check_dtype=False)
    return report


def main() -> int:
    """Run or verify the explicit successor diagnostic.

    Returns:
        Zero for 85 complete successor cells, two for an incomplete panel.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    run = commands.add_parser('run')
    run.add_argument('--config', type=Path, required=True)
    run.add_argument('--output', type=Path, required=True)
    verify = commands.add_parser('verify')
    verify.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = run_successor_attribution(args.config, args.output) if args.command == 'run' else verify_successor_attribution(args.output)
    return report_exit_status(report)


def report_exit_status(report: dict[str, Any]) -> int:
    """Print the command outcome and keep interrupted attempts unsuccessful.

    Args:
        report: Evaluated or verified attempt report.

    Returns:
        Zero for completion, two for explicitly incomplete evidence.
    """
    print(f"{report['status']}: {report['completed_policy_folds']}/85 successor cells; old annual attempt remains incomplete")
    return 0 if report['status'] == 'complete' else 2


if __name__ == '__main__':
    raise SystemExit(main())
