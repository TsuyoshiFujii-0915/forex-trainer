"""One registered continuation of Issue 45 after its saved audit incident."""
from __future__ import annotations

import argparse
import ast
from copy import deepcopy
import json
from pathlib import Path
import shutil
import subprocess
from typing import Any

from . import execution_sensitivity_study as study
from .artifact_provenance import sha256_file
from .common_basket_study import ROOT, digest, execution_runtime, git, read, write
from .common_quantity import QuantityExecutionError
from .common_quantity_study import append_event

CONFIG = ROOT / 'configs/research/issue45_audit_continuation.json'
ORIGINAL = ROOT / 'docs/research/results/issue45/campaign'
RUN = ROOT / 'runs/issue45-audit-continuation-v1'
SAVED = ROOT / 'docs/research/results/issue45/continuation'
ORIGINAL_COMMIT = 'fcec20ab3a90763661a31c691967bbd8f47979ed'
ORIGINAL_MANIFEST = '03d8e5e59a2873c50299abc385be58a3f219c75a66af94e596c313f4afe1c9c9'
AUDIT_FILE = ROOT / 'src/forex_trainer/execution_sensitivity.py'


def scope(config: dict[str, Any]) -> None:
    """Reject budget expansion or any change to the inherited cell order.

    Args:
        config: Explicit one-time continuation registration.
    """
    names = [study.cell_name(c) for c in study.registered_cells()]
    required = {'recovery_id': 'issue45-audit-continuation-v1', 'original_accounts_consumed': 2,
                'new_account_limit': 304, 'fit_calls': 0, 'retry_accounts': 0, 'quote_accounts': 0,
                'cells': names, 'remaining_cells': names[2:]}
    if any(config[k] != v for k, v in required.items()):
        raise ValueError('Issue 45 continuation registration scope/budget mismatch')
    expected = {f'accounts/{name}.json.gz' for name in names[:2]}
    if set(config['original_accounts']) != expected:
        raise ValueError('Issue 45 continuation must reuse exactly the first two accounts')


def compare_runtime(old: dict[str, Any], new: dict[str, Any], allowed: dict[str, str]) -> None:
    """Allow only the registered audit and continuation source changes.

    Args:
        old: Original account execution runtime.
        new: Proposed continuation runtime.
        allowed: Exact hashes for the two permitted trainer modules.
    """
    if set(allowed) != {'execution_sensitivity.py', 'execution_sensitivity_recovery.py'}:
        raise ValueError('Issue 45 runtime source allowlist mismatch')
    expected = deepcopy(old)
    expected['git']['forex_trainer'] = new['git']['forex_trainer']
    expected['source_sha256']['forex_trainer'].update(allowed)
    if new != expected:
        raise ValueError('Issue 45 runtime changed outside registered audit/recovery source')


def contract(path: Path) -> dict[str, Any]:
    """Check frozen provenance and ensure trading code is unchanged.

    Args:
        path: Registered continuation JSON.

    Returns:
        Validated registration, without activating any market accounts.
    """
    config = read(path)
    scope(config)
    if (config['original_manifest_sha256'] != ORIGINAL_MANIFEST
            or config['original_execution_commit'] != ORIGINAL_COMMIT
            or config['original_registration_sha256'] != study.CONFIG_SHA256
            or config['run_directory'] != str(RUN.relative_to(ROOT))
            or config['saved_directory'] != str(SAVED.relative_to(ROOT))):
        raise ValueError('Issue 45 continuation original provenance/directory mismatch')
    for name, expected in config['permitted_source_sha256'].items():
        if name not in ('execution_sensitivity.py', 'execution_sensitivity_recovery.py'):
            raise ValueError(f'Issue 45 continuation unregistered source: {name}')
        if sha256_file(ROOT / 'src/forex_trainer' / name) != expected:
            raise ValueError(f'Issue 45 continuation source hash mismatch: {name}')
    original_source = git(['show', f'{ORIGINAL_COMMIT}:src/forex_trainer/execution_sensitivity.py'], ROOT)
    trees = [ast.parse(original_source), ast.parse(AUDIT_FILE.read_text())]
    for tree in trees:
        auditors = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'audit_proxy']
        if len(auditors) != 1:
            raise ValueError('Issue 45 continuation requires exactly one proxy auditor')
        tree.body.remove(auditors[0])
    if ast.dump(trees[0], include_attributes=False) != ast.dump(trees[1], include_attributes=False):
        raise ValueError('Issue 45 continuation changes proxy execution outside audit_proxy')
    return config


def require_merge(config_path: Path, merge_commit: str) -> None:
    """Require the reviewed registration and implementation on origin/main.

    Args:
        config_path: Exact registration file to compare with the merge.
        merge_commit: Full merged registration SHA.
    """
    if len(merge_commit) != 40 or any(c not in '0123456789abcdef' for c in merge_commit):
        raise ValueError('Issue 45 continuation merge requires a full commit SHA')
    try:
        git(['merge-base', '--is-ancestor', merge_commit, 'origin/main'], ROOT)
        git(['merge-base', '--is-ancestor', merge_commit, 'HEAD'], ROOT)
        git(['merge-base', '--is-ancestor', ORIGINAL_COMMIT, merge_commit], ROOT)
        merged = subprocess.check_output(['git', 'show', f'{merge_commit}:{config_path.relative_to(ROOT)}'], cwd=ROOT)
        if merged != config_path.read_bytes():
            raise ValueError('Issue 45 continuation merge registration bytes differ')
        git(['diff', '--exit-code', merge_commit, 'HEAD', '--', 'src', 'uv.lock', 'pyproject.toml'], ROOT)
    except subprocess.CalledProcessError as exc:
        raise ValueError('Issue 45 continuation merge not verified on origin/main') from exc


def audit_result(result: dict[str, Any], cell: dict[str, Any], inputs: study.Inputs, runtime: dict[str, Any]) -> None:
    """Audit all transitions and bind a saved account to its actual runtime.

    Args:
        result: Saved engine output, never inferred again.
        cell: Registered fold, policy, scenario and execution.
        inputs: Independently loaded original market inputs.
        runtime: Actual old or new execution runtime for this account.
    """
    name = study.cell_name(cell)
    source = {'policy': cell['policy'], 'fold': cell['fold'], 'scenario': cell['scenario'],
              'source': inputs.identities[cell['fold']], 'training_lineage_changed': False}
    if result['runtime_sha256'] != digest(runtime) or result['policy_source'] != source:
        raise ValueError(f'Issue 45 continuation account runtime/source mismatch: {name}')
    axis, costs = inputs.marks[cell['fold']], study.scenario_costs(cell['scenario'])
    if cell['execution'] == 'same_close':
        study.audit_account(result, name, axis, costs)
    else:
        study.audit_proxy(result, axis, costs, name)


def audit_original(inputs: study.Inputs, original: Path, config: dict[str, Any], runtime: dict[str, Any]) -> dict[str, Any]:
    """Reaudit both entire original accounts without changing their status.

    Args:
        inputs: Original loaded inputs.
        original: Immutable failed attempt directory.
        config: Exact source files and bounded continuation registration.
        runtime: Registered verification and continuation implementation.

    Returns:
        Separate recovery evidence bound to the old hashes and new verifier.
    """
    scope(config)
    if sha256_file(original / 'manifest.json') != config['original_manifest_sha256']:
        raise ValueError('Issue 45 continuation original manifest mismatch')
    records = study.verify_manifest(original)
    study.verify_accounts(original, inputs, records)
    manifest, activation, report = records
    compare_runtime(activation['runtime'], runtime, config['permitted_source_sha256'])
    if (manifest['proxy_accounts_consumed'] != 2 or report['incident'] is None
            or [c['status'] for c in report['cells']] != ['complete', 'execution_error'] + ['not_run'] * 304):
        raise ValueError('Issue 45 continuation original attempt status/consumption mismatch')
    audited = []
    for cell in report['cells'][:2]:
        path = original / cell['result_path']
        if sha256_file(path) != config['original_accounts'][cell['result_path']]:
            raise ValueError(f'Issue 45 continuation inherited account hash mismatch: {path}')
        result = read(path)
        audit_result(result, cell, inputs, activation['runtime'])
        if result['status'] != 'complete' or result['remaining_decisions'] != 0:
            raise ValueError(f'Issue 45 continuation inherited account not complete: {path}')
        audited.append({'cell': study.cell_name(cell), 'path': cell['result_path'], 'sha256': sha256_file(path),
                        'transitions_audited': len(result['trace']), 'original_status': cell['status'],
                        'reaudit_status': 'verified'})
    return {'original_manifest_sha256': config['original_manifest_sha256'], 'original_incident': report['incident'],
            'original_runtime': activation['runtime'], 'verification_runtime': runtime,
            'accounts': audited, 'accounts_reexecuted': 0, 'policy_inference_calls': 0}


def unused_budget() -> None:
    """Reject a second continuation reservation in local or published evidence."""
    for path in (RUN, SAVED):
        if path.exists():
            raise ValueError(f'Issue 45 continuation budget already reserved: {path}; retry 0')


def require_activation_commit(activation: dict[str, Any]) -> None:
    """Bind the new execution runtime to the reviewed registration merge.

    Args:
        activation: Proposed or saved continuation seal.
    """
    if activation['runtime']['git']['forex_trainer'] != activation['registration_merge_commit']:
        raise ValueError('Issue 45 continuation runtime must equal registration merge commit')


def seal(config_path: Path, merge_commit: str, output: Path) -> dict[str, Any]:
    """Activate only the merged, clean, reaudit-approved 304-account scope.

    Args:
        config_path: Reviewed finite continuation registration.
        merge_commit: Registration and implementation merge commit.
        output: New activation directory.

    Returns:
        Exact old/new identities, recovery proof and new runtime seal.
    """
    require_merge(config_path, merge_commit)
    unused_budget()
    config, inputs, runtime = contract(config_path), study.load_inputs(study.CONFIG), execution_runtime()
    record = audit_original(inputs, ORIGINAL, config, runtime)
    activation = {'config': config, 'runtime': runtime, 'registration_merge_commit': merge_commit,
                  'identities': inputs.identities, 'recovery': record}
    require_activation_commit(activation)
    output.mkdir(parents=True, exist_ok=False)
    write(output / 'activation.json', activation)
    return activation


def finish(directory: Path, cells: list[dict[str, Any]], incident: dict[str, Any] | None) -> dict[str, Any]:
    """Save all inherited/new cells and withhold comparisons on any incident.

    Args:
        directory: Reserved continuation directory.
        cells: Full 306-cell recovered matrix.
        incident: Explicit unresolved continuation failure, or None.

    Returns:
        Complete or incomplete recovery report with original lineage retained.
    """
    report = study.report_with_incident(cells, incident)
    write(directory / 'report.json', report)
    write(directory / 'handoff.json', report['handoff'])
    events = [json.loads(line) for line in (directory / 'account-ledger.jsonl').read_text().splitlines()]
    consumed = sum(e['event'] == 'reserved' for e in events)
    manifest = {'recovery_id': 'issue45-audit-continuation-v1', 'original_accounts_consumed': 2,
                'new_accounts_consumed': consumed, 'total_accounts_consumed': 2 + consumed,
                'quote_accounts_consumed': 0, 'fit_calls': 0, 'retry_accounts': 0,
                'artifacts': {str(p.relative_to(directory)): sha256_file(p)
                              for p in sorted(directory.rglob('*')) if p.is_file()}}
    manifest['content_sha256'] = digest(manifest)
    write(directory / 'manifest.json', manifest)
    return report


def execute(inputs: study.Inputs, original: Path, activation: dict[str, Any], directory: Path) -> dict[str, Any]:
    """Copy the two verified accounts and execute only the remaining 304 once.

    Args:
        inputs: Original market inputs or explicit synthetic fixture inputs.
        original: Failed attempt, preserved without modification.
        activation: Checked registration, recovery evidence and runtime seal.
        directory: New fixed campaign directory or isolated synthetic directory.

    Returns:
        Persisted recovered report; recorded failures propagate as exceptions.
    """
    record = audit_original(inputs, original, activation['config'], activation['runtime'])
    if activation['recovery'] != record or activation['identities'] != inputs.identities:
        raise ValueError('Issue 45 continuation recovery/input activation mismatch')
    directory.mkdir(parents=True, exist_ok=False)
    (directory / 'accounts').mkdir()
    write(directory / 'activation.json', activation)
    write(directory / 'recovery.json', record)
    ledger = directory / 'account-ledger.jsonl'
    append_event(ledger, {'event': 'budget_reserved', 'original_consumed': 2, 'new_limit': 304, 'quote_limit': 0})
    cells = study.registered_cells()
    active: dict[str, Any] | None = None
    try:
        for cell, inherited in zip(cells[:2], record['accounts'], strict=True):
            path = inherited['path']
            shutil.copyfile(original / path, directory / path)
            if sha256_file(directory / path) != inherited['sha256']:
                raise ValueError(f'Issue 45 continuation inherited copy mismatch: {path}')
            result = read(directory / path)
            cell.update(status=result['status'], result_path=path, metrics=study.metrics(result, cell['execution']))
            append_event(ledger, {'event': 'inherited', 'work_item': study.cell_name(cell), 'sha256': inherited['sha256']})
        source_by_fold = {s.fold: s for s in inputs.sources}
        for cell in cells[2:]:
            active = cell
            name = study.cell_name(cell)
            append_event(ledger, {'event': 'reserved', 'work_item': name})
            source = source_by_fold[cell['fold']]
            predictors = {'canonical': study.canonical_action, 'ridge': source.ridge.action, 'ppo_ens3': source.ppo.action}
            identity = {'policy': cell['policy'], 'fold': cell['fold'], 'scenario': cell['scenario'],
                        'source': inputs.identities[cell['fold']], 'training_lineage_changed': False}
            engine = study.replay if cell['execution'] == 'same_close' else study.replay_next_close
            result = engine(inputs.marks[cell['fold']], study.scenario_costs(cell['scenario']),
                            study.legacy_policy(predictors[cell['policy']]), digest(activation['runtime']), identity)
            path = f'accounts/{name}.json.gz'
            write(directory / path, result)
            cell['result_path'] = path
            audit_result(result, cell, inputs, activation['runtime'])
            cell.update(status=result['status'], metrics=study.metrics(result, cell['execution']))
            append_event(ledger, {'event': 'finished', 'work_item': name, 'status': cell['status']})
            active = None
        if execution_runtime() != activation['runtime']:
            raise ValueError('Issue 45 continuation runtime changed during execution')
    except Exception as exc:
        error = f'{type(exc).__name__}: {exc}'
        if active is not None:
            active.update(status='execution_error', error=error, metrics=None)
            append_event(ledger, {'event': 'finished', 'work_item': study.cell_name(active), 'status': 'execution_error'})
        incident = {'error': error, 'cell': study.cell_name(active) if active is not None else None,
                    'partial': exc.evidence if isinstance(exc, QuantityExecutionError) else None, 'automatic_retry': False}
        write(directory / 'incident.json', incident)
        finish(directory, cells, incident)
        raise
    return finish(directory, cells, None)


def run(config_path: Path, activation_path: Path) -> dict[str, Any]:
    """Check merged registration and unused scope before market inference.

    Args:
        config_path: Fixed continuation registration.
        activation_path: Matching clean-runtime seal.

    Returns:
        Persisted finite continuation result.
    """
    unused_budget()
    activation = read(activation_path)
    require_activation_commit(activation)
    require_merge(config_path, activation['registration_merge_commit'])
    if activation['config'] != contract(config_path) or activation['runtime'] != execution_runtime():
        raise ValueError('Issue 45 continuation config/runtime activation mismatch')
    return execute(study.load_inputs(study.CONFIG), ORIGINAL, activation, RUN)


def verify_saved(directory: Path, original: Path, inputs: study.Inputs) -> dict[str, Any]:
    """Reconcile inherited and new accounts without inference or rewriting history.

    Args:
        directory: Saved continuation campaign.
        original: Original failed campaign referenced by its exact hashes.
        inputs: Independently validated original inputs.

    Returns:
        Verification status and separate original/new account consumption.
    """
    manifest, activation, report = study.verify_manifest(directory)
    config, runtime = activation['config'], activation['runtime']
    record = audit_original(inputs, original, config, runtime)
    if (activation['recovery'] != record or read(directory / 'recovery.json') != record
            or activation['identities'] != inputs.identities):
        raise ValueError('Issue 45 continuation saved recovery/input mismatch')
    incident_path = directory / 'incident.json'
    incident = read(incident_path) if incident_path.exists() else None
    if (incident_path.exists() and not isinstance(incident, dict)) or report['incident'] != incident:
        raise ValueError('Issue 45 continuation saved incident mismatch')
    if report != study.report_with_incident(report['cells'], incident) or read(directory / 'handoff.json') != report['handoff']:
        raise ValueError('Issue 45 continuation saved report mismatch')
    expected = [{'event': 'budget_reserved', 'original_consumed': 2, 'new_limit': 304, 'quote_limit': 0}]
    consumed = 0
    stopped = False
    for i, (cell, registered) in enumerate(zip(report['cells'], study.registered_cells(), strict=True)):
        name = study.cell_name(cell)
        if name != study.cell_name(registered):
            raise ValueError('Issue 45 continuation saved cell order mismatch')
        if i < 2:
            inherited = record['accounts'][i]
            if cell['result_path'] != inherited['path'] or sha256_file(directory / inherited['path']) != inherited['sha256']:
                raise ValueError(f'Issue 45 continuation inherited bytes changed: {name}')
            if cell['status'] != 'complete':
                raise ValueError(f'Issue 45 continuation inherited status changed: {name}')
            expected.append({'event': 'inherited', 'work_item': name, 'sha256': inherited['sha256']})
        elif cell['status'] != 'not_run':
            if stopped:
                raise ValueError('Issue 45 continuation skipped or resumed a cell')
            consumed += 1
            expected += [{'event': 'reserved', 'work_item': name}, {'event': 'finished', 'work_item': name, 'status': cell['status']}]
            if cell['status'] == 'execution_error':
                stopped = True
        else:
            stopped = True
        if cell['status'] in ('complete', 'strategy_terminal'):
            if cell['result_path'] != f'accounts/{name}.json.gz':
                raise ValueError(f'Issue 45 continuation account path mismatch: {name}')
            result = read(directory / cell['result_path'])
            audit_result(result, cell, inputs, record['original_runtime'] if i < 2 else runtime)
            if result['status'] != cell['status'] or study.metrics(result, cell['execution']) != cell['metrics']:
                raise ValueError(f'Issue 45 continuation account metrics mismatch: {name}')
    events = [json.loads(line) for line in (directory / 'account-ledger.jsonl').read_text().splitlines()]
    counts = {'original_accounts_consumed': 2, 'new_accounts_consumed': consumed,
              'total_accounts_consumed': consumed + 2, 'quote_accounts_consumed': 0, 'fit_calls': 0, 'retry_accounts': 0}
    if events != expected or any(manifest[k] != v for k, v in counts.items()):
        raise ValueError('Issue 45 continuation ledger/budget mismatch')
    return {'status': 'verified', 'proxy_status': report['proxy_status'], 'quote_status': 'blocked_input',
            **counts, 'accounts_reexecuted': 0, 'incident': incident}


def main() -> int:
    """Expose read-only reaudit, merged seal, finite run and saved verification.

    Returns:
        Zero on success; explicit failures propagate with origins.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    for name in ('reaudit', 'seal', 'run', 'verify'):
        command = sub.add_parser(name)
        command.add_argument('--config', type=Path, required=True)
        if name in ('reaudit', 'seal'):
            command.add_argument('--output', type=Path, required=True)
        if name == 'seal':
            command.add_argument('--registration-merge-commit', required=True)
        elif name == 'run':
            command.add_argument('--activation', type=Path, required=True)
        elif name == 'verify':
            command.add_argument('--directory', type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'seal':
        result = seal(args.config, args.registration_merge_commit, args.output)
    elif args.command == 'run':
        result = run(args.config, args.activation)
    else:
        config, inputs = contract(args.config), study.load_inputs(study.CONFIG)
        if args.command == 'reaudit':
            result = audit_original(inputs, ORIGINAL, config, execution_runtime())
            args.output.mkdir(parents=True, exist_ok=False)
            write(args.output / 'recovery.json', result)
        else:
            _, activation, _ = study.verify_manifest(args.directory)
            require_activation_commit(activation)
            if activation['config'] != config:
                raise ValueError('Issue 45 continuation saved registration mismatch')
            require_merge(args.config, activation['registration_merge_commit'])
            result = verify_saved(args.directory, ORIGINAL, inputs)
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
