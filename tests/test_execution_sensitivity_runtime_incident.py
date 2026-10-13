"""Final runtime failures invalidate evidence without rewriting completed paths."""
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


@pytest.mark.parametrize('failure', ['mismatch', 'exception'])
def test_final_runtime_incident_preserves_accounts_and_withholds_all_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, synthetic_inputs: study.Inputs, failure: str,
) -> None:
    sealed = activation(synthetic_inputs)
    directory = tmp_path / 'campaign'
    snapshots: list[tuple[dict[str, str], bytes]] = []

    def final_runtime() -> dict[str, Any]:
        snapshots.append((account_hashes(directory), (directory / 'account-ledger.jsonl').read_bytes()))
        assert len(snapshots[-1][0]) == 306
        if failure == 'exception':
            raise ValueError('Execution seal requires clean repository: synthetic')
        return {'fixture_runtime': 'changed_after_all_accounts'}

    monkeypatch.setattr(study, 'execution_runtime', final_runtime)
    error = ('Execution seal requires clean repository: synthetic' if failure == 'exception'
             else 'Issue 45 runtime changed during execution')
    with pytest.raises(ValueError, match=error):
        study.execute_accounts(synthetic_inputs, sealed, directory)

    assert len(snapshots) == 1
    hashes, ledger_bytes = snapshots[0]
    report = read(directory / 'report.json')
    incident = read(directory / 'incident.json')
    assert incident['cell'] is None
    assert incident['error'] == f'ValueError: {error}'
    assert report['incident'] == incident
    assert report['proxy_status'] == 'incomplete'
    assert report['status_counts']['complete'] == 306
    assert report['status_counts']['execution_error'] == report['status_counts']['not_run'] == 0
    for cell in report['cells']:
        account = read(directory / cell['result_path'])
        assert cell['status'] == account['status'] == 'complete'
        assert cell['metrics'] == study.metrics(account, cell['execution'])
    assert len(report['comparisons']) == 9
    for comparison in report['comparisons']:
        assert comparison['status'] == 'pending_integrity'
        assert comparison['reason'] == 'unresolved_campaign_incident'
        assert comparison['net'] is comparison['gross'] is comparison['cash_and_risk_mean_differences'] is None
        assert comparison['missing_cells'] == []
    handoff = read(directory / 'handoff.json')
    assert handoff == report['handoff']
    assert handoff['proxy_status'] == 'incomplete'
    assert handoff['proxy_evidence_status'] == 'withheld_unresolved_incident'
    assert handoff['incident'] == incident
    assert handoff['new_candidates_validated'] == 0
    manifest = read(directory / 'manifest.json')
    assert manifest['proxy_accounts_consumed'] == 306
    assert manifest['quote_accounts_consumed'] == manifest['fit_calls'] == manifest['retry_accounts'] == 0

    verified = verify_synthetic(directory, synthetic_inputs)
    assert verified['status'] == 'verified'
    assert verified['proxy_status'] == 'incomplete'
    assert verified['incident'] == incident
    assert verified['proxy_accounts_consumed'] == 306
    assert verified['accounts_reexecuted'] == 0
    assert account_hashes(directory) == hashes
    assert (directory / 'account-ledger.jsonl').read_bytes() == ledger_bytes
    assert len(snapshots) == 1

    # Recomputed hashes must not restore the old misleading evidence classification.
    tampered = tmp_path / 'tampered'
    shutil.copytree(directory, tampered)
    misleading = {**study.summarize(report['cells']), 'incident': incident}
    (tampered / 'report.json').write_bytes(encode(misleading))
    (tampered / 'handoff.json').write_bytes(encode(misleading['handoff']))
    changed_manifest = read(tampered / 'manifest.json')
    for name in ('report.json', 'handoff.json'):
        changed_manifest['artifacts'][name] = sha256_file(tampered / name)
    del changed_manifest['content_sha256']
    changed_manifest['content_sha256'] = digest(changed_manifest)
    (tampered / 'manifest.json').write_bytes(encode(changed_manifest))
    with pytest.raises(ValueError, match='report reconciliation'):
        verify_synthetic(tampered, synthetic_inputs)
