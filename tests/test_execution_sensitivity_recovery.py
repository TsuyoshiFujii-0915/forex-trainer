"""Exercise one bounded continuation with real synthetic accounting and files."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest

from forex_trainer import execution_sensitivity_recovery as recovery
from forex_trainer import execution_sensitivity_study as study
from forex_trainer.artifact_provenance import sha256_file
from forex_trainer.common_basket_study import digest, read, write
from forex_trainer.common_quantity_study import append_event
from test_execution_sensitivity_runner import synthetic_inputs, activation, account_hashes


def runtimes() -> tuple[dict[str, Any], dict[str, Any]]:
    """Build explicit old/new runtime records for synthetic-only tests.

    Returns:
        Records differing only in the allowed trainer code and commit.
    """
    old = {'git': {'forex_trainer': 'old', 'forex_env': 'same-env'}, 'lock_sha256': 'same-lock',
           'versions': {'numpy': 'same'}, 'source_sha256': {
               'forex_trainer': {'execution_sensitivity.py': 'old-audit', 'common_quantity.py': 'same-engine'},
               'forex_env': {'accounting.py': 'same-accounting'}}}
    new = deepcopy(old)
    new['git']['forex_trainer'] = 'new'
    new['source_sha256']['forex_trainer'].update({
        'execution_sensitivity.py': 'new-audit', 'execution_sensitivity_recovery.py': 'new-recovery'})
    return old, new


@pytest.fixture(scope='module')
def original_campaign(tmp_path_factory: pytest.TempPathFactory, synthetic_inputs: study.Inputs) -> tuple[Path, dict[str, Any]]:
    """Save exactly two actual synthetic accounts and an explicit audit incident.

    Args:
        tmp_path_factory: Isolated storage.
        synthetic_inputs: Original synthetic prices and untrained models.

    Returns:
        Failed campaign and matching one-time continuation registration.
    """
    directory = tmp_path_factory.mktemp('original-recovery') / 'campaign'
    directory.mkdir()
    (directory / 'accounts').mkdir()
    old, new = runtimes()
    sealed = activation(synthetic_inputs)
    sealed['runtime'] = old
    write(directory / 'activation.json', sealed)
    cells = study.registered_cells()
    ledger = directory / 'account-ledger.jsonl'
    append_event(ledger, {'event': 'budget_reserved', 'proxy_limit': 306, 'quote_limit': 0})
    for cell in cells[:2]:
        name = study.cell_name(cell)
        append_event(ledger, {'event': 'reserved', 'work_item': name})
        source = {'policy': cell['policy'], 'fold': cell['fold'], 'scenario': cell['scenario'],
                  'source': synthetic_inputs.identities[cell['fold']], 'training_lineage_changed': False}
        engine = study.replay if cell['execution'] == 'same_close' else study.replay_next_close
        result = engine(synthetic_inputs.marks[cell['fold']], study.scenario_costs(cell['scenario']),
                        study.legacy_policy(study.canonical_action), digest(old), source)
        path = directory / 'accounts' / f'{name}.json.gz'
        write(path, result)
        cell.update(status='complete', result_path=str(path.relative_to(directory)),
                    metrics=study.metrics(result, cell['execution']))
        if cell['execution'] == 'next_close':
            cell.update(status='execution_error', metrics=None, error='synthetic saved audit incident')
        append_event(ledger, {'event': 'finished', 'work_item': name, 'status': cell['status']})
    incident = {'error': 'synthetic saved audit incident', 'cell': study.cell_name(cells[1]),
                'partial': None, 'automatic_retry': False}
    write(directory / 'incident.json', incident)
    study.finish(directory, cells, incident)
    names = [study.cell_name(c) for c in study.registered_cells()]
    config = {'recovery_id': 'issue45-audit-continuation-v1', 'original_accounts_consumed': 2,
              'new_account_limit': 304, 'fit_calls': 0, 'retry_accounts': 0, 'quote_accounts': 0,
              'cells': names, 'remaining_cells': names[2:],
              'original_manifest_sha256': sha256_file(directory / 'manifest.json'),
              'original_accounts': {c['result_path']: sha256_file(directory / c['result_path']) for c in cells[:2]},
              'permitted_source_sha256': {k: new['source_sha256']['forex_trainer'][k]
                  for k in ('execution_sensitivity.py', 'execution_sensitivity_recovery.py')}}
    return directory, config


@pytest.mark.parametrize('incident', [False, True])
def test_continuation_reuses_two_bytes_and_consumes_only_304_new_accounts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, original_campaign: tuple[Path, dict[str, Any]],
    synthetic_inputs: study.Inputs, incident: bool,
) -> None:
    original, config = original_campaign
    _, runtime = runtimes()
    before = {str(p.relative_to(original)): p.read_bytes() for p in original.rglob('*') if p.is_file()}
    record = recovery.audit_original(synthetic_inputs, original, config, runtime)
    sealed = {'config': config, 'runtime': runtime, 'recovery': record,
              'identities': synthetic_inputs.identities, 'registration_merge_commit': 'synthetic-only'}
    directory = tmp_path / 'continued'

    def current_runtime() -> dict[str, Any]:
        if incident:
            raise ValueError('synthetic continuation runtime failure')
        return runtime

    monkeypatch.setattr(recovery, 'execution_runtime', current_runtime)
    if incident:
        with pytest.raises(ValueError, match='synthetic continuation runtime failure'):
            recovery.execute(synthetic_inputs, original, sealed, directory)
    else:
        recovery.execute(synthetic_inputs, original, sealed, directory)
    verified = recovery.verify_saved(directory, original, synthetic_inputs)
    assert verified['status'] == 'verified'
    assert verified['new_accounts_consumed'] == 304
    assert verified['original_accounts_consumed'] == 2
    assert verified['total_accounts_consumed'] == 306
    assert verified['accounts_reexecuted'] == 0
    assert verified['proxy_status'] == ('incomplete' if incident else 'complete')
    report = read(directory / 'report.json')
    assert all(c['status'] == ('pending_integrity' if incident else 'available') for c in report['comparisons'])
    for relative, data in before.items():
        assert (original / relative).read_bytes() == data
    for relative in config['original_accounts']:
        assert (directory / relative).read_bytes() == (original / relative).read_bytes()
    hashes = account_hashes(directory)
    with pytest.raises((ValueError, FileExistsError)):
        recovery.execute(synthetic_inputs, original, sealed, directory)
    assert account_hashes(directory) == hashes


@pytest.mark.parametrize('field', ['lock', 'engine', 'environment', 'unexpected_source'])
def test_execution_runtime_changes_beyond_audit_are_rejected(field: str) -> None:
    old, new = runtimes()
    allowed = {k: new['source_sha256']['forex_trainer'][k]
               for k in ('execution_sensitivity.py', 'execution_sensitivity_recovery.py')}
    recovery.compare_runtime(old, new, allowed)
    if field == 'lock':
        new['lock_sha256'] = 'changed'
    elif field == 'engine':
        new['source_sha256']['forex_trainer']['common_quantity.py'] = 'changed'
    elif field == 'environment':
        new['git']['forex_env'] = 'changed'
    else:
        new['source_sha256']['forex_trainer']['unregistered.py'] = 'added'
    with pytest.raises(ValueError, match='runtime'):
        recovery.compare_runtime(old, new, allowed)


@pytest.mark.parametrize('mutation', ['budget', 'order', 'missing', 'duplicate', 'original_hash'])
def test_registration_cannot_expand_budget_or_reorder_cells(
    original_campaign: tuple[Path, dict[str, Any]], synthetic_inputs: study.Inputs, mutation: str,
) -> None:
    original, registered = original_campaign
    config = deepcopy(registered)
    if mutation == 'budget':
        config['new_account_limit'] = 306
    elif mutation == 'order':
        config['remaining_cells'].reverse()
    elif mutation == 'missing':
        config['remaining_cells'].pop()
    elif mutation == 'duplicate':
        config['remaining_cells'][1] = config['remaining_cells'][0]
    else:
        config['original_manifest_sha256'] = '0' * 64
    with pytest.raises(ValueError):
        recovery.audit_original(synthetic_inputs, original, config, runtimes()[1])


def test_unmerged_continuation_is_not_activated(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match='merge'):
        recovery.seal(recovery.CONFIG, '0' * 64, tmp_path / 'activation')
    assert not (tmp_path / 'activation').exists()
