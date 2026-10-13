"""Rehashed metadata must not conceal mixed runtimes or unresolved incidents."""
from __future__ import annotations

from pathlib import Path
import gzip
import shutil
from typing import Any

import pytest

from forex_trainer import execution_sensitivity_recovery as recovery
from forex_trainer import execution_sensitivity_study as study
from forex_trainer.artifact_provenance import sha256_file
from forex_trainer.common_basket_study import digest, encode, read
from test_execution_sensitivity_recovery import original_campaign, runtimes
from test_execution_sensitivity_runner import synthetic_inputs


@pytest.fixture(scope='module')
def saved_continuation(
    tmp_path_factory: pytest.TempPathFactory, original_campaign: tuple[Path, dict[str, Any]],
    synthetic_inputs: study.Inputs,
) -> Path:
    """Save a full synthetic continuation with a final runtime incident.

    Args:
        tmp_path_factory: Isolated storage.
        original_campaign: Two original synthetic account outputs.
        synthetic_inputs: Original synthetic input and model identities.

    Returns:
        Completed account files with an unresolved final incident.
    """
    original, config = original_campaign
    _, runtime = runtimes()
    record = recovery.audit_original(synthetic_inputs, original, config, runtime)
    sealed = {'config': config, 'runtime': runtime, 'recovery': record,
              'identities': synthetic_inputs.identities, 'registration_merge_commit': 'synthetic-only'}
    directory = tmp_path_factory.mktemp('integrity-continuation') / 'saved'

    def failed_runtime() -> dict[str, Any]:
        raise ValueError('synthetic final runtime incident')

    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(recovery, 'execution_runtime', failed_runtime)
        with pytest.raises(ValueError, match='synthetic final runtime incident'):
            recovery.execute(synthetic_inputs, original, sealed, directory)
    return directory


@pytest.mark.parametrize('mutation', ['inherited_runtime', 'new_runtime', 'ledger', 'recovery_record', 'incident_null', 'cell_order'])
def test_rehashed_continuation_disagreements_are_rejected(
    tmp_path: Path, saved_continuation: Path, original_campaign: tuple[Path, dict[str, Any]],
    synthetic_inputs: study.Inputs, mutation: str,
) -> None:
    directory = tmp_path / 'tampered'
    shutil.copytree(saved_continuation, directory)
    if mutation in ('inherited_runtime', 'new_runtime'):
        index = 0 if mutation == 'inherited_runtime' else 2
        cell = read(directory / 'report.json')['cells'][index]
        path = directory / cell['result_path']
        account = read(path)
        account['runtime_sha256'] = '0' * 64
        path.write_bytes(gzip.compress(encode(account), mtime=0))
    elif mutation == 'ledger':
        path = directory / 'account-ledger.jsonl'
        lines = path.read_text().splitlines()
        del lines[3]
        path.write_text('\n'.join(lines)+'\n')
    elif mutation == 'recovery_record':
        record = read(directory / 'recovery.json')
        record['original_incident'] = None
        (directory / 'recovery.json').write_bytes(encode(record))
    else:
        report = read(directory / 'report.json')
        if mutation == 'incident_null':
            report = {**study.summarize(report['cells']), 'incident': None}
        else:
            report['cells'][2], report['cells'][3] = report['cells'][3], report['cells'][2]
        (directory / 'report.json').write_bytes(encode(report))
        (directory / 'handoff.json').write_bytes(encode(report['handoff']))
    manifest = read(directory / 'manifest.json')
    manifest['artifacts'] = {str(p.relative_to(directory)): sha256_file(p)
                             for p in sorted(directory.rglob('*')) if p.is_file() and p.name != 'manifest.json'}
    del manifest['content_sha256']
    manifest['content_sha256'] = digest(manifest)
    (directory / 'manifest.json').write_bytes(encode(manifest))
    with pytest.raises(ValueError):
        recovery.verify_saved(directory, original_campaign[0], synthetic_inputs)
