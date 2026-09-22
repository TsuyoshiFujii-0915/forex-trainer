"""Issue #29 cost evaluation on the immutable Issue #39 successor panel."""
from __future__ import annotations

import argparse
import copy
import gzip
import hashlib
import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .artifact_provenance import sha256_file
from .cost_campaign import (
    POLICIES, SCENARIOS, account_trace, compare_sensitivity, describe_period_change,
    fold_metrics, run_fold_scenarios, summarize_campaign, verify_campaign, write_json,
)
from .full_period import checked_path, prepare_period, read_json, require_comparable, runtime_identity
from .full_period_sources import FoldSources
from .input_recovery import load_snapshot_sources, read_market, reference, verify_snapshot
from .supervised_portfolio import FOLDS, paired_evidence

CAMPAIGN_ID = 'issue29-contiguous-cost-v1'
INPUT_REFERENCES = ('snapshot', 'parent_manifest', 'portfolio', 'registration', 'decision', 'parent_contract', 'lock')


def validate_config(config: dict[str, Any]) -> None:
    """Require the fixed successor inputs and original three cost treatments.

    Args:
        config: Explicit campaign registration.
    """
    if set(config) != {'campaign_id', 'scenarios', *INPUT_REFERENCES} or config['campaign_id'] != CAMPAIGN_ID:
        raise ValueError('Invalid successor campaign fields/ID; ranges must come from the sealed snapshot')
    if config['scenarios'] != SCENARIOS:
        raise ValueError('Successor campaign requires the original F0/F1/F2 scenarios')


def summarize_successor(cells: list[dict[str, Any]], coverage: list[dict[str, Any]]) -> dict[str, Any]:
    """Reuse registered statistics with an explicit conditional-coverage scope.

    Args:
        cells: All 153 successor observations or failures.
        coverage: Ordered frozen coverage records, one per fold.

    Returns:
        Successor-only evidence, with old annual incompleteness preserved.
    """
    if [row['fold'] for row in coverage] != list(FOLDS):
        raise ValueError('Successor coverage requires all 17 ordered folds')
    indexed = {row['fold']: row for row in coverage}
    for cell in cells:
        if cell['status'] == 'complete':
            if cell['fold'] not in indexed:
                raise ValueError(f"Unknown coverage fold: {cell['fold']}")
            for field in ('first_decision', 'last_mark'):
                if cell['coverage'][field] != indexed[cell['fold']][field]:
                    raise ValueError(f"Successor coverage differs in {cell['fold']}/{field}")
    report = summarize_campaign(cells)
    report.update({'campaign_id': CAMPAIGN_ID, 'original_annual_campaign_complete': False,
                   'original_blocked_policy_folds': 81, 'coverage': coverage,
                   'scope_counts': {kind: sum(row['successor_scope'] == kind for row in coverage) for kind in ('full_year', 'partial_year')},
                   'statistics_interpretation': 'Equal-weight annualized observations conditional on the 17 frozen contiguous ranges; not full-year population evidence or continuous-operation CAGR.',
                   'absolute_evidence': None})
    if report['status'] != 'complete':
        return report
    metrics = {(r['scenario'], r['fold'], r['policy']): r['metrics'] for r in cells}
    report['absolute_evidence'] = {
        scenario: {policy: {name: paired_evidence(np.array([metrics[scenario, f, policy][name] for f in FOLDS]), np.zeros(17))
                             for name in ('annualized_net_return', 'annualized_gross_return')}
                   for policy in POLICIES} for scenario in SCENARIOS}
    for scenario in SCENARIOS:
        for policy in POLICIES:
            for era, folds in (('all', FOLDS), ('2009-2018', FOLDS[:10]), ('2019-2025', FOLDS[10:])):
                summary = report['summaries'][scenario][policy][era]
                for name in ('period_net_return', 'period_gross_return', 'annualized_net_volatility'):
                    summary['mean_' + name] = float(np.mean([metrics[scenario, f, policy][name] for f in folds]))
            report['policy_assessments'][policy]['reason'] = 'Conditional development triage on the frozen successor ranges; inspect absolute/paired uncertainty, eras and LOO. No full-year completion, independent profitability recognition, or change to Issue #16.'
    return report


def successor_metrics(result: dict[str, Any]) -> dict[str, Any]:
    """Include nonannualized returns alongside existing monetary and risk metrics.

    Args:
        result: Account evaluated and reconciled by the existing cost evaluator.

    Returns:
        Existing metrics plus period returns and population log-return volatility.
    """
    metrics = fold_metrics(result)
    metric = result['metrics']
    gross_log = metric['gross_cumulative_log_return']
    years = result['coverage']['elapsed_seconds'] / (365.25 * 86400)
    equity = np.array([result['trace'][0]['equity_before'], *[row['equity_jpy'] for row in result['trace']]])
    volatility = None
    if np.all(equity > 0):
        logs = np.log(equity[1:] / equity[:-1])
        volatility = float(np.std(logs, ddof=0) * np.sqrt(len(logs) / years))
    return {**metrics, 'period_net_return': metric['final_equity_ratio'] - 1,
            'period_gross_return': float(np.expm1(gross_log)) if gross_log is not None else None,
            'annualized_net_volatility': volatility}


def evaluate_sources(sources: list[FoldSources], output: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Evaluate the fixed panel using the existing nine-account fold runner.

    Args:
        sources: Ordered frozen inputs for all 17 registered ranges.
        output: New campaign directory with no previous fold results.

    Returns:
        All account cells and valid within-policy cost comparisons.
    """
    if [source.fold for source in sources] != list(FOLDS):
        raise ValueError('Successor evaluation requires 17 ordered source folds')
    for source in sources:
        if (output / f'fold-{source.fold}').exists():
            raise FileExistsError(f'Fold output already exists: {output}/fold-{source.fold}')
    cells: list[dict[str, Any]] = []
    sensitivities: list[dict[str, Any]] = []
    for source in sources:
        work = output / f'fold-{source.fold}'
        work.mkdir()
        results = run_fold_scenarios(source, work)
        for result in results:
            stem = f"{result['scenario']}-{result['policy']}"
            cell = {name: result[name] for name in ('scenario', 'fold', 'policy', 'status')}
            cell['result_path'] = f'fold-{source.fold}/{stem}.json.gz'
            if result['status'] == 'execution_error':
                cell['error'] = result['error']
            else:
                cell.update({'coverage': result['coverage'], 'metrics': successor_metrics(result),
                             'pairs_path': f'fold-{source.fold}/{stem}-pairs.csv.gz'})
            cells.append(cell)
        indexed = {(r['scenario'], r['policy']): r for r in results}
        for scenario in ('F1', 'F2'):
            for policy in POLICIES:
                f0, stress = indexed['F0', policy], indexed[scenario, policy]
                if f0['status'] == stress['status'] == 'complete':
                    sensitivities.append(compare_sensitivity(f0, stress))
        print(f"{source.fold}: {sum(r['status'] == 'complete' for r in results)}/9 successor accounts complete", flush=True)
    return cells, sensitivities


def read_account(path: Path) -> dict[str, Any]:
    """Read an existing compressed account without rerunning its policy.

    Args:
        path: Sealed account trace.

    Returns:
        Saved account mapping.
    """
    return json.loads(gzip.decompress(path.read_bytes()))


def render_report(report: dict[str, Any]) -> str:
    """Render the successor scope and every account status explicitly.

    Args:
        report: Verified panel report.

    Returns:
        Markdown evidence summary.
    """
    lines = ['# Issue #29 contiguous successor cost evaluation', '',
             f"Status: {report['status']}; {report['completed_policy_folds']}/153 successor accounts.",
             'Old annual campaign remains incomplete: 81 blocked cells. This panel contains 9 partial-year and 8 full-coverage ranges.',
             report['statistics_interpretation'], '',
             '| Scenario | Fold | Policy | Status | Net annualized | Gross annualized | Period net | MDD |',
             '|---|---|---|---|---:|---:|---:|---:|']
    for cell in report['cells']:
        values = [f"{cell['metrics'][name]:.6f}" if cell['status'] == 'complete' else 'unavailable'
                  for name in ('annualized_net_return', 'annualized_gross_return', 'period_net_return', 'max_drawdown')]
        lines.append(f"| {cell['scenario']} | {cell['fold']} | {cell['policy']} | {cell['status']} | " + ' | '.join(values) + ' |')
    return '\n'.join(lines) + '\n'


def run_successor(config_path: Path, output: Path) -> dict[str, Any]:
    """Run exactly one newly registered successor campaign without training.

    Args:
        config_path: Frozen successor registration and references.
        output: New directory for results and their seal.

    Returns:
        Full or explicitly incomplete successor report.
    """
    config = read_json(config_path)
    validate_config(config)
    for key in INPUT_REFERENCES:
        checked_path(config[key])
    if output.exists():
        raise FileExistsError(f'Campaign output already exists: {output}')
    snapshot_root = Path(config['snapshot']['path']).parent
    snapshot = verify_snapshot(snapshot_root)
    sources = load_snapshot_sources(snapshot_root)
    coverage = read_json(snapshot_root / 'coverage.json')['folds']
    old_root = Path(config['parent_manifest']['path']).parent
    old_report = verify_campaign(old_root)
    if old_report['status'] != 'incomplete' or old_report['completed_policy_folds'] != 72:
        raise ValueError('Expected the unchanged 72/153 old annual attempt')
    portfolio = read_json(Path(config['portfolio']['path']))
    portfolio_root = Path(config['portfolio']['path']).parent
    for name, digest in portfolio['generated_artifact_sha256'].items():
        checked_path({'path': str(portfolio_root / name), 'sha256': digest})
    for source in sources:
        baseline = portfolio['fold_sources'][source.fold]
        expected = {**baseline['source'], **{name: source.identity[name] for name in ('lineage', 'calendar_source', 'successor_snapshot')}}
        if source.identity != expected or source.env_raw['transaction_costs'] != baseline['resolved_eval_env']['transaction_costs']:
            raise ValueError(f'Frozen source or F0 costs changed: {source.fold}')
    output.mkdir(parents=True)
    write_json(output / 'campaign_snapshot.json', config)
    write_json(output / 'evaluation.json', read_json(snapshot_root / 'evaluation.json'))
    write_json(output / 'coverage.json', {'folds': coverage})
    runtime = runtime_identity()
    manifest: dict[str, Any] = {'campaign_id': CAMPAIGN_ID, 'config_sha256': sha256_file(config_path),
        'snapshot': config['snapshot'], 'snapshot_identity': snapshot['identity_sha256'], 'runtime': runtime,
        'python': platform.python_version(), 'started_at': datetime.now(timezone.utc).isoformat(),
        'git_status': subprocess.check_output(['git', 'status', '--porcelain'], text=True),
        'command': ['forex-successor-cost-campaign', 'run', '--config', str(config_path), '--output', str(output)],
        'folds': {source.fold: {'source': source.identity, 'ridge_parameter_sha256': source.ridge.parameter_sha256,
            'history_labels': list(source.plan.history_labels), 'effective_timestamps': list(source.plan.timestamps),
            'measurement_start': source.plan.measurement_start, 'measurement_end': source.plan.measurement_end}
            for source in sources}}
    write_json(output / 'started.json', manifest)
    cells, sensitivities = evaluate_sources(sources, output)
    report = summarize_successor(cells, coverage)
    report['fold_cost_sensitivity'] = sensitivities
    legacy = pd.read_csv(portfolio_root / 'fold_metrics.csv')
    legacy.to_csv(output / 'legacy_reference.csv', index=False)
    changes: list[dict[str, Any]] = []
    for cell in cells:
        if cell['scenario'] == 'F0' and cell['status'] == 'complete':
            name = dict(zip(POLICIES, ('reversal', 'supervised', 'ppo')))[cell['policy']]
            row = legacy[(legacy['fold'].astype(str) == cell['fold']) & (legacy['policy'] == name)]
            if len(row) != 1:
                raise ValueError(f'Expected one sealed legacy row: {cell["fold"]}/{name}')
            changes.append({'fold': cell['fold'], 'policy': cell['policy'], **describe_period_change(read_account(output / cell['result_path']), row.iloc[0].to_dict())})
    report['descriptive_period_changes'] = changes
    report['legacy_reference'] = 'Different periods; descriptive only. Zero new_year_start_steps means no newly measured prefix in the successor, not a recovered annual beginning.'
    if runtime_identity() != runtime:
        raise ValueError('Runtime changed during successor evaluation; preserve this incomplete attempt')
    write_json(output / 'report.json', report)
    (output / 'report.md').write_text(render_report(report), encoding='utf-8')
    pd.json_normalize(cells).to_csv(output / 'status.csv', index=False)
    write_json(output / 'attribution-handoff.json', {'campaign_id': CAMPAIGN_ID, 'snapshot_identity': snapshot['identity_sha256'],
        'scope': 'Successor ranges; requires an explicit successor #30 registration, not the old 85-cell annual attempt.',
        'evaluation': reference(output / 'evaluation.json'),
        'F0_cells': [cell for cell in cells if cell['scenario'] == 'F0'],
        'model_training_lineage_changed': False, 'new_training_calls': 0})
    manifest['finished_at'] = datetime.now(timezone.utc).isoformat()
    manifest['generated_artifact_sha256'] = {str(p.relative_to(output)): sha256_file(p) for p in sorted(output.rglob('*')) if p.is_file()}
    write_json(output / 'manifest.json', manifest)
    return report


def verify_successor(output: Path) -> dict[str, Any]:
    """Recheck saved hashes, market accounting and statistics without inference.

    Args:
        output: Sealed successor results.

    Returns:
        Reconciled report; any changed source or account raises an exception.
    """
    manifest = read_json(output / 'manifest.json')
    if manifest['campaign_id'] != CAMPAIGN_ID:
        raise ValueError('Unexpected successor manifest campaign ID')
    inventory = {str(p.relative_to(output)) for p in output.rglob('*') if p.is_file() and p != output / 'manifest.json'}
    if inventory != set(manifest['generated_artifact_sha256']):
        raise ValueError(f'Successor artifact inventory differs: {output}')
    for name, digest in manifest['generated_artifact_sha256'].items():
        checked_path({'path': str(output / name), 'sha256': digest})
    config = read_json(output / 'campaign_snapshot.json')
    validate_config(config)
    for key in INPUT_REFERENCES:
        checked_path(config[key])
    snapshot_root = Path(checked_path(manifest['snapshot'])).parent
    snapshot = verify_snapshot(snapshot_root)
    if snapshot['identity_sha256'] != manifest['snapshot_identity']:
        raise ValueError('Successor snapshot identity differs')
    report = read_json(output / 'report.json')
    coverage = read_json(snapshot_root / 'coverage.json')['folds']
    summary = summarize_successor(report['cells'], coverage)
    for field, value in summary.items():
        if report[field] != value:
            raise ValueError(f'Successor summary differs: {field}')
    evaluation = read_json(output / 'evaluation.json')
    plans = {}
    for fold in evaluation['folds']:
        name = fold['fold']
        calendar = read_json(checked_path(fold['calendar']))
        path = checked_path(snapshot['derived_data'][name])
        plans[name] = prepare_period(read_market(path), calendar, fold['measurement_start'], fold['measurement_end'], tuple(calendar['symbols']), path)
    results: dict[tuple[str, str, str], dict[str, Any]] = {}
    for cell in report['cells']:
        result = read_account(output / cell['result_path'])
        if any(result[key] != cell[key] for key in ('status', 'fold', 'scenario', 'policy')):
            raise ValueError(f'Account identity differs: {cell["result_path"]}')
        if cell['status'] == 'execution_error':
            if result['error'] != cell['error']:
                raise ValueError('Account error differs')
            continue
        if result['runtime'] != manifest['runtime'] or successor_metrics(result) != cell['metrics'] or result['coverage'] != cell['coverage']:
            raise ValueError(f'Account metrics/runtime differ: {cell["result_path"]}')
        plan = plans[cell['fold']]
        if hashlib.sha256(plan.market.to_numpy(dtype=np.float64).tobytes()).hexdigest() != result['market_sha256']:
            raise ValueError(f'Market bytes differ: {cell["result_path"]}')
        reconstructed = copy.deepcopy(result)
        expected_pairs = account_trace(reconstructed, plan)
        if reconstructed != result:
            raise ValueError(f'Account reconciliation differs: {cell["result_path"]}')
        pairs = pd.read_csv(output / cell['pairs_path'], float_precision='round_trip')
        pd.testing.assert_frame_equal(pairs, pd.DataFrame(expected_pairs), check_exact=True)
        results[cell['scenario'], cell['fold'], cell['policy']] = result
    sensitivities = []
    for fold in FOLDS:
        for scenario in SCENARIOS:
            same = [results[scenario, fold, p] for p in POLICIES if (scenario, fold, p) in results]
            if len(same) == 3 and all(r['status'] == 'complete' for r in same):
                require_comparable(same)
        for scenario in ('F1', 'F2'):
            for policy in POLICIES:
                keys = [('F0', fold, policy), (scenario, fold, policy)]
                if all(key in results and results[key]['status'] == 'complete' for key in keys):
                    sensitivities.append(compare_sensitivity(results[keys[0]], results[keys[1]]))
    if sensitivities != report['fold_cost_sensitivity']:
        raise ValueError('Saved cost sensitivity differs')
    return report


def main() -> int:
    """Run or verify the explicit successor campaign.

    Returns:
        Zero for a complete successor panel; two for an incomplete panel.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    run = commands.add_parser('run')
    run.add_argument('--config', type=Path, required=True)
    run.add_argument('--output', type=Path, required=True)
    verify = commands.add_parser('verify')
    verify.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = run_successor(args.config, args.output) if args.command == 'run' else verify_successor(args.output)
    print(f"{report['status']}: {report['completed_policy_folds']}/153 successor accounts; old annual attempt remains incomplete")
    return 0 if report['status'] == 'complete' else 2


if __name__ == '__main__':
    raise SystemExit(main())
