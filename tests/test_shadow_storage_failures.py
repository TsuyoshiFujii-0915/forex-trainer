"""A failed append is acknowledged explicitly before the trial can continue."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pytest

from forex_trainer.development_shadow import initialize, run_cycle, append_correction, recover
from test_development_shadow import setup_trial, events, write


def test_recovery_imports_external_failure_as_an_incident(tmp_path: Path) -> None:
    path, manifest = setup_trial(tmp_path)
    initialize(path)
    root = Path(manifest['storage'])
    run_cycle(root, 'day-0')
    failure_dir = Path(manifest['failure_storage'])
    failure_dir.mkdir()
    failure = failure_dir / 'failed-checkpoint.json'
    write(failure, {'trial_id': manifest['trial_id'], 'storage': str(root),
                    'origin': 'checkpoint', 'detected_at': '2026-01-06T22:00:00Z',
                    'type': 'OSError', 'message': 'disk full after durable outcomes',
                    'response': 'stop; explicit recovery required'})
    with pytest.raises(ValueError, match='recover|failure'):
        run_cycle(root, 'day-1')
    recover(root)
    assert any(e['payload']['code'] == 'recovered_failure' for e in events(root, 'incident'))
    assert run_cycle(root, 'day-1')['status'] == 'complete'


def test_correction_disk_failure_also_has_independent_failure_record(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path, manifest = setup_trial(tmp_path)
    initialize(path)
    root = Path(manifest['storage'])
    run_cycle(root, 'day-0')
    proof = tmp_path / 'proof.json'
    write(proof, {'quality': 'correction evidence'})
    real_link = os.link

    def fail(src: str | Path, dst: str | Path, **kwargs: Any) -> None:
        if Path(dst).parent == root / 'events' and b'"event_type":"correction"' in Path(src).read_bytes():
            raise OSError('disk full while recording correction')
        real_link(src, dst, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(os, 'link', fail)
        with pytest.raises(OSError, match='disk'):
            append_correction(root, 'correction-disk', 'day-0/decision/canonical_reversal', 'quality note', proof)
    assert not events(root, 'correction')
    failures = list(Path(manifest['failure_storage']).glob('*.json'))
    assert failures
    assert 'correction' in json.loads(failures[0].read_text())['origin']
