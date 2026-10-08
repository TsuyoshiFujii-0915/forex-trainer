"""Verify incomplete reports against real sealed inputs without account replay."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from forex_trainer.artifact_provenance import sha256_file
from forex_trainer.common_basket_study import digest, read, write
from forex_trainer.common_direction import registered_cells
from forex_trainer.common_direction_study import CONFIG, Inputs, activation_record, finish, load_inputs, verify


@pytest.fixture(scope='module')
def inputs() -> Inputs:
    """Load frozen models and market data without prediction.

    Returns:
        The campaign's complete validated dependencies.
    """
    return load_inputs(CONFIG)


def blocked_report(directory: Path, inputs: Inputs) -> dict[str, Any]:
    """Record a dependency failure before any budget is consumed.

    Args:
        directory: Isolated temporary output.
        inputs: Verified real source data.

    Returns:
        Persisted report including all 357 blocked cells.
    """
    runtime = read(Path('docs/research/results/issue42/fixture/activation.json'))['runtime']
    activation = activation_record(inputs, runtime)
    write(directory / 'activation.json', activation)
    write(directory / 'registered-cells.json', registered_cells())
    (directory / 'account-ledger.jsonl').touch()
    cells = registered_cells()
    for cell in cells:
        cell.update(status='blocked_input', error='missing external input')
    return finish(directory, cells, activation, {'error': 'missing external input', 'automatic_retry': False})


def test_incomplete_saved_report_is_verifiable_without_accounts(tmp_path: Path, inputs: Inputs) -> None:
    report = blocked_report(tmp_path, inputs)
    verified = verify(tmp_path)
    assert verified == report
    assert verified['status_counts']['blocked_input'] == 357
    assert verified['handoff_issue49']['status'] == 'pending_measurement'
    manifest = read(tmp_path / 'manifest.json')
    assert manifest['candidate_accounts'] == manifest['control_accounts'] == 0


@pytest.mark.parametrize('mutation', ['hash', 'counter', 'rules', 'handoff', 'unregistered_artifact'])
def test_saved_report_detects_tampering(tmp_path: Path, inputs: Inputs, mutation: str) -> None:
    blocked_report(tmp_path, inputs)
    manifest = read(tmp_path / 'manifest.json')
    if mutation == 'hash':
        (tmp_path / 'report.json').write_text('{}')
    elif mutation == 'counter':
        manifest['candidate_accounts'] = 1
    elif mutation == 'rules':
        value = read(tmp_path / 'activation.json')
        value['rules']['candidate_F0_min_mean_net_log'] = 0.
        (tmp_path / 'activation.json').unlink()
        write(tmp_path / 'activation.json', value)
        manifest['artifacts']['activation.json'] = sha256_file(tmp_path / 'activation.json')
    elif mutation == 'handoff':
        (tmp_path / 'issue49-handoff.json').write_text('{"status":"candidate","candidate":"A1-cost"}')
        manifest['artifacts']['issue49-handoff.json'] = sha256_file(tmp_path / 'issue49-handoff.json')
    else:
        write(tmp_path / 'old-metrics.json', {'reused': True})
        manifest['artifacts']['old-metrics.json'] = sha256_file(tmp_path / 'old-metrics.json')
    del manifest['content_sha256']
    manifest['content_sha256'] = digest(manifest)
    (tmp_path / 'manifest.json').unlink()
    write(tmp_path / 'manifest.json', manifest)
    with pytest.raises(ValueError, match='Issue 43'):
        verify(tmp_path)


def test_recorded_execution_exception_keeps_remaining_cells_unrun(tmp_path: Path, inputs: Inputs) -> None:
    runtime = read(Path('docs/research/results/issue42/fixture/activation.json'))['runtime']
    activation = activation_record(inputs, runtime)
    write(tmp_path / 'activation.json', activation)
    write(tmp_path / 'registered-cells.json', registered_cells())
    from forex_trainer.common_quantity_study import append_event
    ledger = tmp_path / 'account-ledger.jsonl'
    append_event(ledger, {'event': 'reserved', 'work_item': '2009-F0-A1-fixed',
                         'counter': 'issue43_candidate_accounts', 'attempt': 1,
                         'runtime_sha256': digest(runtime)})
    append_event(ledger, {'event': 'finished', 'work_item': '2009-F0-A1-fixed', 'status': 'execution_error'})
    cells = registered_cells()
    cells[0].update(status='execution_error', error='solver failure')
    finish(tmp_path, cells, activation, {'error': 'solver failure', 'automatic_retry': False})
    result = verify(tmp_path)
    assert result['status_counts']['execution_error'] == 1
    assert result['status_counts']['not_run'] == 356
    assert read(tmp_path / 'manifest.json')['candidate_accounts'] == 1
