"""Adversarial recovery, clock, and registration failures for the shadow CLI."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pytest

from forex_trainer.development_shadow import initialize, run_cycle, verify_log, recover
from test_development_shadow import setup_trial, events, write


def test_missing_raw_is_a_main_log_incident(tmp_path: Path) -> None:
    path, manifest = setup_trial(tmp_path)
    initialize(path)
    Path(manifest['schedule'][0]['raw_path']).unlink()
    root = Path(manifest['storage'])
    with pytest.raises(ValueError, match='raw|missing'):
        run_cycle(root, 'day-0')
    assert events(root, 'incident')
    assert not events(root, 'decision')


def test_malformed_execution_contract_rejected_at_registration(tmp_path: Path) -> None:
    path, manifest = setup_trial(tmp_path)
    manifest['execution']['costs']['fixed_spread_debit'] = .01
    write(path, manifest)
    with pytest.raises(ValueError, match='execution|spread|schema'):
        initialize(path)
    assert not Path(manifest['storage']).exists()


def test_no_second_day_backfill_after_crash_during_raw_recording(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path, manifest = setup_trial(tmp_path)
    initialize(path)
    root = Path(manifest['storage'])
    run_cycle(root, 'day-0')
    real_link = os.link

    def crash(src: str | Path, dst: str | Path, **kwargs: Any) -> None:
        if Path(dst).parent == root / 'events' and b'"event_type":"decision"' in Path(src).read_bytes():
            raise KeyboardInterrupt('crash before second decision commit')
        real_link(src, dst, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(os, 'link', crash)
        with pytest.raises(KeyboardInterrupt):
            run_cycle(root, 'day-1')
    recover(root)
    with pytest.raises(ValueError, match='halt|backfill|retrospective'):
        run_cycle(root, 'day-1')
    assert not [e for e in events(root, 'decision') if e['cycle_id'] == 'day-1']


def test_committed_torn_batch_is_never_truncated_by_recovery(tmp_path: Path) -> None:
    path, manifest = setup_trial(tmp_path)
    initialize(path)
    root = Path(manifest['storage'])
    run_cycle(root, 'day-0')
    batch = sorted((root / 'events').glob('*.jsonl'))[-1]
    content = batch.read_bytes()[:-7]
    batch.write_bytes(content)
    with pytest.raises(ValueError, match='Torn'):
        recover(root)
    assert batch.read_bytes() == content


def test_observed_decision_deadline_uses_wall_clock(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path, manifest = setup_trial(tmp_path)
    from forex_trainer.full_period import runtime_identity
    from forex_trainer.artifact_provenance import sha256_file
    source = Path('docs/research/protocols/issue20/artifact-identities.json')
    identities = json.loads(source.read_text())
    if not Path(identities['forward_selection']['ppo']['ensemble_dir']).exists():
        pytest.skip('Pinned real bundle unavailable')
    runtime = runtime_identity()
    sources = tmp_path / 'sources.json'
    write(sources, {'source': identities['artifacts'][0], 'fold': '2025', 'runtime': {
        'forex_env_sha': runtime['git']['forex_env'], 'versions': runtime['versions'], 'device': 'cpu'}})
    manifest.update(mode='observed_file', policy_mode='frozen_2025',
                    policy_sources={'path': str(sources), 'sha256': sha256_file(sources)})
    write(path, manifest)

    def before_start() -> str:
        return '2026-01-04T22:00:01Z'

    def after_deadline() -> str:
        return '2026-01-05T22:00:11Z'

    monkeypatch.setattr('forex_trainer.development_shadow.now', before_start)
    initialize(path)
    monkeypatch.setattr('forex_trainer.development_shadow.now', after_deadline)
    root = Path(manifest['storage'])
    with pytest.raises(ValueError, match='deadline'):
        run_cycle(root, 'day-0')
    assert not events(root, 'decision')
    assert events(root, 'incident')


def test_late_observed_barrier_is_recorded_as_invalid_without_apply(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path, manifest = setup_trial(tmp_path)
    manifest['schedule'][0]['decision_deadline'] = manifest['schedule'][0]['decision_not_before']
    write(path, manifest)
    initialize(path)
    root = Path(manifest['storage'])
    run_cycle(root, 'day-0')
    batches = sorted((root / 'events').glob('*.jsonl'))
    barrier_batch = next(p for p in batches if b'"event_type":"decision_barrier"' in p.read_bytes())
    barrier = json.loads(barrier_batch.read_text())
    barrier['payload']['durable_at'] = '2026-01-05T22:00:12Z'
    from forex_trainer.shadow_store import encode, digest
    previous: str | None = None
    for batch in batches:
        revised = []
        for line in batch.read_bytes().splitlines():
            row = json.loads(line)
            if row['event_id'] == barrier['event_id']:
                row = barrier
            row['previous_event_sha256'] = previous
            serialized = encode(row)
            previous = digest(serialized)
            revised.append(serialized + b'\n')
        batch.write_bytes(b''.join(revised))
    with pytest.raises(ValueError, match='deadline'):
        verify_log(root)


def test_published_outcome_with_failed_directory_sync_recovers_once(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path, manifest = setup_trial(tmp_path)
    initialize(path)
    root = Path(manifest['storage'])
    real_fsync = os.fsync
    real_link = os.link
    published = False

    def track_link(src: str | Path, dst: str | Path, **kwargs: Any) -> None:
        nonlocal published
        real_link(src, dst, **kwargs)
        if Path(dst).parent == root / 'events' and b'"event_type":"outcome"' in Path(src).read_bytes():
            published = True

    def fail_once(descriptor: int) -> None:
        nonlocal published
        if published:
            published = False
            raise OSError('injected outcome directory fsync failure')
        real_fsync(descriptor)

    with monkeypatch.context() as patch:
        patch.setattr(os, 'link', track_link)
        patch.setattr(os, 'fsync', fail_once)
        with pytest.raises(OSError, match='fsync'):
            run_cycle(root, 'day-0')
    recover(root)
    assert run_cycle(root, 'day-0')['status'] == 'already_complete'
    assert len(events(root, 'outcome')) == 3
    assert list(Path(manifest['failure_storage']).glob('*.json'))
