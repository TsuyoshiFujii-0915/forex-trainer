"""Saved incident presence cannot be overridden by a report's self-description."""
from __future__ import annotations

from pathlib import Path
import shutil
from typing import Any

import pytest

from forex_trainer import execution_sensitivity_study as study
from forex_trainer.artifact_provenance import sha256_file
from forex_trainer.common_basket_study import digest, encode, read
from test_execution_sensitivity_runner import (
    account_hashes, activation, synthetic_inputs, verify_synthetic,
)


@pytest.fixture(scope='module')
def saved_incident_campaign(tmp_path_factory: pytest.TempPathFactory, synthetic_inputs: study.Inputs) -> Path:
    """Save one real synthetic run for read-only incident consistency checks.

    Args:
        tmp_path_factory: Temporary fixture storage.
        synthetic_inputs: Synthetic prices and untrained fixture models.

    Returns:
        Completed account artifacts with a final runtime incident.
    """
    directory = tmp_path_factory.mktemp('saved-incident') / 'campaign'

    def failed_runtime() -> dict[str, Any]:
        raise ValueError('synthetic final runtime failure')

    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(study, 'execution_runtime', failed_runtime)
        with pytest.raises(ValueError, match='synthetic final runtime failure'):
            study.execute_accounts(synthetic_inputs, activation(synthetic_inputs), directory)
    return directory


def refresh_manifest(directory: Path) -> None:
    """Rehash changed metadata so tests reach semantic consistency validation.

    Args:
        directory: Isolated copy of a saved synthetic campaign.
    """
    manifest = read(directory / 'manifest.json')
    manifest['artifacts'] = {str(p.relative_to(directory)): sha256_file(p)
                             for p in sorted(directory.rglob('*'))
                             if p.is_file() and p != directory / 'manifest.json'}
    del manifest['content_sha256']
    manifest['content_sha256'] = digest(manifest)
    (directory / 'manifest.json').write_bytes(encode(manifest))


def test_saved_incident_remains_verifiable_without_account_reexecution(
    saved_incident_campaign: Path, synthetic_inputs: study.Inputs,
) -> None:
    before = account_hashes(saved_incident_campaign)
    verified = verify_synthetic(saved_incident_campaign, synthetic_inputs)
    assert verified['status'] == 'verified'
    assert verified['proxy_status'] == 'incomplete'
    assert verified['incident'] == read(saved_incident_campaign / 'incident.json')
    assert verified['proxy_accounts_consumed'] == 306
    assert verified['accounts_reexecuted'] == 0
    assert account_hashes(saved_incident_campaign) == before


@pytest.mark.parametrize('mutation', ['report_null', 'missing_file', 'different_content', 'null_file'])
def test_rehashed_incident_disagreement_is_rejected(
    tmp_path: Path, saved_incident_campaign: Path, synthetic_inputs: study.Inputs, mutation: str,
) -> None:
    directory = tmp_path / 'campaign'
    shutil.copytree(saved_incident_campaign, directory)
    accounts_before = account_hashes(directory)
    ledger_before = (directory / 'account-ledger.jsonl').read_bytes()
    incident_before = (directory / 'incident.json').read_bytes()
    if mutation in ('report_null', 'null_file'):
        report = read(directory / 'report.json')
        misleading = {**study.summarize(report['cells']), 'incident': None}
        (directory / 'report.json').write_bytes(encode(misleading))
        (directory / 'handoff.json').write_bytes(encode(misleading['handoff']))
        if mutation == 'null_file':
            (directory / 'incident.json').write_bytes(encode(None))
    elif mutation == 'missing_file':
        (directory / 'incident.json').unlink()
    else:
        incident = read(directory / 'incident.json')
        incident['error'] = 'different incident origin'
        (directory / 'incident.json').write_bytes(encode(incident))
    refresh_manifest(directory)

    with pytest.raises(ValueError, match='incident'):
        verify_synthetic(directory, synthetic_inputs)
    assert account_hashes(directory) == accounts_before
    assert (directory / 'account-ledger.jsonl').read_bytes() == ledger_before
    assert read(directory / 'manifest.json')['proxy_accounts_consumed'] == 306
    if mutation == 'report_null':
        assert (directory / 'incident.json').read_bytes() == incident_before
        assert 'incident.json' in read(directory / 'manifest.json')['artifacts']
