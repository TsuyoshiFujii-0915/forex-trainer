"""Snapshot integrity must be checked before frozen-policy use."""
from __future__ import annotations

from pathlib import Path

import pytest

from forex_trainer.full_period import json_hash
from forex_trainer.input_recovery import CONTRACT_ID, reference, verify_snapshot, write_immutable_json


def snapshot_fixture(root: Path) -> Path:
    """Create byte-sealed inputs without remote APIs or model inference."""
    root.mkdir()
    source = root / 'source.json'
    write_immutable_json(source, {'input': 'fixed'})
    registration = root / 'registration.json'
    write_immutable_json(registration, {key: reference(source) for key in ('campaign', 'attribution', 'budget', 'evidence', 'decision')})
    artifact = root / 'coverage.json'
    write_immutable_json(artifact, {'folds': []})
    manifest = {'contract_id': CONTRACT_ID, 'registration': reference(registration),
                'artifact_sha256': {'coverage.json': reference(artifact)['sha256']},
                'derived_data': {'2009': reference(source)},
                'source_data': {key: [reference(source)] for key in ('raw', 'clean', 'carry')}}
    manifest['identity_sha256'] = json_hash(manifest)
    write_immutable_json(root / 'manifest.json', manifest)
    return root


def test_verified_snapshot_rejects_modified_prices_and_registration(tmp_path: Path) -> None:
    root = snapshot_fixture(tmp_path / 'snapshot')
    assert verify_snapshot(root)['contract_id'] == CONTRACT_ID
    (root / 'source.json').write_text('{"input":"changed"}')
    with pytest.raises(ValueError, match='source hash mismatch'):
        verify_snapshot(root)


def test_verified_snapshot_rejects_modified_coverage(tmp_path: Path) -> None:
    root = snapshot_fixture(tmp_path / 'snapshot')
    (root / 'coverage.json').write_text('{"folds":["fake"]}')
    with pytest.raises(ValueError, match='source hash mismatch'):
        verify_snapshot(root)


def test_manifest_identity_cannot_be_relabelled(tmp_path: Path) -> None:
    root = snapshot_fixture(tmp_path / 'snapshot')
    path = root / 'manifest.json'
    path.write_text(path.read_text().replace(CONTRACT_ID, 'old-attempt-complete'))
    with pytest.raises(ValueError, match='Snapshot identity mismatch'):
        verify_snapshot(root)
