"""User-visible shadow recording, crash recovery and causal audit contracts."""

from __future__ import annotations

import copy
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from forex_trainer.artifact_provenance import sha256_file
from forex_trainer.development_shadow import (
    initialize, run_cycle, verify_log, export_head, append_correction,
    record_trial, recover, build_window,
)
from forex_trainer.shadow_store import EventLog

FIXTURE = Path(__file__).parent / 'fixtures/issue32/manifest.json'


def write(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, allow_nan=False) + '\n')


def setup_trial(tmp_path: Path) -> tuple[Path, dict[str, Any]]:
    manifest = json.loads(FIXTURE.read_text())
    manifest['storage'] = str(tmp_path / 'trial')
    manifest['failure_storage'] = str(tmp_path / 'failures')
    for step in manifest['schedule']:
        for field in ('raw_path', 'settlement_path'):
            source = Path(step[field])
            target = tmp_path / source.name
            target.write_bytes(source.read_bytes())
            step[field] = str(target)
    path = tmp_path / 'manifest.json'
    write(path, manifest)
    return path, manifest


def events(root: Path, kind: str) -> list[dict[str, Any]]:
    return [e for e in verify_log(root)['events'] if e['event_type'] == kind]


def test_two_cycles_restart_duplicate_and_account_continuity(tmp_path: Path) -> None:
    path, manifest = setup_trial(tmp_path)
    root = Path(manifest['storage'])
    initialize(path)
    first = run_cycle(root, 'day-0')
    assert first['status'] == 'complete'
    before = {p: p.read_bytes() for p in (root / 'events').glob('*.jsonl')}
    assert run_cycle(root, 'day-0')['status'] == 'already_complete'
    assert {p: p.read_bytes() for p in (root / 'events').glob('*.jsonl')} == before
    assert run_cycle(root, 'day-1')['status'] == 'complete'
    audit = verify_log(root)
    assert len(events(root, 'decision')) == 6
    assert len(events(root, 'outcome')) == 6
    assert audit['completed_cycles'] == ['day-0', 'day-1']
    outcomes = events(root, 'outcome')
    for i in range(3):
        assert outcomes[i + 3]['payload']['equity_before_jpy'] == outcomes[i]['payload']['equity_after_jpy']
    assert len({json.dumps(e['payload']['state_after']['assets']) for e in outcomes[:3]}) == 3
    head_path = tmp_path / 'head.json'
    export_head(root, head_path)
    head = json.loads(head_path.read_text())
    assert head['head_sha256'] == audit['head_sha256']
    assert head['event_count'] == audit['event_count']
    with pytest.raises((ValueError, FileExistsError), match='exist'):
        export_head(root, head_path)


@pytest.mark.parametrize('bad', ['future', 'unavailable', 'carry_unavailable', 'missing', 'pairs', 'source'])
def test_bad_raw_preserved_as_incident_without_decisions(tmp_path: Path, bad: str) -> None:
    path, manifest = setup_trial(tmp_path)
    raw_path = Path(manifest['schedule'][0]['raw_path'])
    raw = json.loads(raw_path.read_text())
    if bad == 'future':
        raw['rows'][-1]['event_time'] = '2027-01-01T00:00:00Z'
    elif bad == 'unavailable':
        raw['rows'][-1]['retrieved_at'] = '2027-01-01T00:00:00Z'
    elif bad == 'carry_unavailable':
        raw['rows'][-1]['carry_available_at'] = '2027-01-01T00:00:00Z'
    elif bad == 'missing':
        raw['rows'].pop(1)
    elif bad == 'pairs':
        raw['pairs'].reverse()
    else:
        raw['source'] = 'unregistered-source'
    write(raw_path, raw)
    initialize(path)
    root = Path(manifest['storage'])
    with pytest.raises(ValueError):
        run_cycle(root, 'day-0')
    assert not events(root, 'decision')
    assert events(root, 'raw')
    assert events(root, 'incident')[-1]['payload']['origin']
    assert verify_log(root)['accounts'] == {}


@pytest.mark.parametrize('bad', ['late_quote', 'missing_quote', 'funding', 'mark'])
def test_settlement_failure_keeps_three_decisions_without_outcomes(tmp_path: Path, bad: str) -> None:
    path, manifest = setup_trial(tmp_path)
    settlement_path = Path(manifest['schedule'][0]['settlement_path'])
    raw = json.loads(settlement_path.read_text())
    if bad == 'late_quote':
        for field in ('timestamp', 'available_at', 'retrieved_at'):
            raw['quotes'][0][field] = '2026-01-05T22:02:00Z'
    elif bad == 'missing_quote':
        raw['quotes'] = []
    elif bad == 'funding':
        raw['funding'] = []
    else:
        raw['mark']['timestamp'] = '2026-01-05T22:00:02Z'
    write(settlement_path, raw)
    initialize(path)
    root = Path(manifest['storage'])
    with pytest.raises(ValueError):
        run_cycle(root, 'day-0')
    assert len(events(root, 'decision')) == 3
    assert not events(root, 'outcome')
    assert verify_log(root)['accounts'] == {}


def test_missing_settlement_can_resume_from_durable_decisions(tmp_path: Path) -> None:
    path, manifest = setup_trial(tmp_path)
    settlement = Path(manifest['schedule'][0]['settlement_path'])
    content = settlement.read_bytes()
    settlement.unlink()
    initialize(path)
    root = Path(manifest['storage'])
    assert run_cycle(root, 'day-0')['status'] == 'awaiting_settlement'
    decisions = copy.deepcopy(events(root, 'decision'))
    settlement.write_bytes(content)
    assert run_cycle(root, 'day-0')['status'] == 'complete'
    assert events(root, 'decision') == decisions
    assert len(events(root, 'outcome')) == 3


def test_hash_modification_and_artifact_modification_are_detected(tmp_path: Path) -> None:
    path, manifest = setup_trial(tmp_path)
    initialize(path)
    root = Path(manifest['storage'])
    run_cycle(root, 'day-0')
    data = events(root, 'decision')[0]['payload']['data']
    artifact = root / data['path']
    original = artifact.read_bytes()
    artifact.write_bytes(original + b' ')
    with pytest.raises(ValueError, match='hash|sha256'):
        verify_log(root)
    artifact.write_bytes(original)
    batch = sorted((root / 'events').glob('*.jsonl'))[1]
    batch.write_bytes(batch.read_bytes().replace(b'"raw"', b'"RAW"', 1))
    with pytest.raises(ValueError):
        verify_log(root)


def test_semantic_audit_rejects_rehashed_bad_equity(tmp_path: Path) -> None:
    path, manifest = setup_trial(tmp_path)
    initialize(path)
    root = Path(manifest['storage'])
    run_cycle(root, 'day-0')
    batch = sorted((root / 'events').glob('*.jsonl'))[-1]
    rows = [json.loads(line) for line in batch.read_text().splitlines()]
    assert all(row['event_type'] == 'outcome' for row in rows)
    rows[-1]['payload']['equity_after_jpy'] += 100
    batch.write_text(''.join(json.dumps(row, sort_keys=True, separators=(',', ':')) + '\n' for row in rows))
    with pytest.raises(ValueError, match='equity|account|checkpoint|hash'):
        verify_log(root)


def test_corrections_and_trial_ledger_never_rewrite_bytes(tmp_path: Path) -> None:
    path, manifest = setup_trial(tmp_path)
    initialize(path)
    root = Path(manifest['storage'])
    run_cycle(root, 'day-0')
    originals = {p: p.read_bytes() for p in (root / 'events').glob('*.jsonl')}
    evidence = tmp_path / 'correction.json'
    write(evidence, {'reason': 'source quality description corrected; no account restatement'})
    target = events(root, 'outcome')[0]['event_id']
    append_correction(root, 'correction-1', target, 'quality note', evidence)
    record_trial(root, 'view-1', 'performance_view', 'developer reviewed completed outcomes')
    record_trial(root, 'design-1', 'design_change', 'new candidate must use a different trial')
    assert all(p.read_bytes() == content for p, content in originals.items())
    assert len(events(root, 'correction')) == 1
    assert len(events(root, 'trial_entry')) == 2
    with pytest.raises(ValueError, match='duplicate'):
        append_correction(root, 'correction-1', target, 'quality note', evidence)
    with pytest.raises(ValueError, match='reference|unknown'):
        append_correction(root, 'correction-2', 'absent', 'bad reference', evidence)


def test_single_writer_and_duplicate_event_rejection(tmp_path: Path) -> None:
    path, manifest = setup_trial(tmp_path)
    initialize(path)
    root = Path(manifest['storage'])
    with EventLog(root) as log:
        with pytest.raises(ValueError, match='writer|lock'):
            with EventLog(root):
                pass
        with pytest.raises(ValueError, match='duplicate'):
            log.append([log.events[0]])


def test_torn_staging_is_preserved_and_explicit_recovery_is_required(tmp_path: Path) -> None:
    path, manifest = setup_trial(tmp_path)
    initialize(path)
    root = Path(manifest['storage'])
    staged = root / 'staging' / 'interrupted.jsonl'
    staged.write_bytes(b'{"event_id":"unfinished')
    with pytest.raises(ValueError, match='recover|staging'):
        run_cycle(root, 'day-0')
    recover(root)
    assert any(p.read_bytes() == b'{"event_id":"unfinished' for p in (root / 'orphans').iterdir())
    assert events(root, 'incident')
    with pytest.raises(ValueError, match='halt|incomplete'):
        run_cycle(root, 'day-0')


def test_disk_failure_before_decision_commit_never_applies_accounts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path, manifest = setup_trial(tmp_path)
    initialize(path)
    root = Path(manifest['storage'])
    real_link = os.link

    def fail_decision_link(src: str | Path, dst: str | Path, **kwargs: Any) -> None:
        if Path(dst).parent == root / 'events' and b'"event_type":"decision"' in Path(src).read_bytes():
            raise OSError('injected full disk during decision publication')
        real_link(src, dst, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(os, 'link', fail_decision_link)
        with pytest.raises((OSError, ValueError), match='disk|publication'):
            run_cycle(root, 'day-0')
    assert verify_log(root)['accounts'] == {}
    assert not events(root, 'outcome')
    assert list(Path(manifest['failure_storage']).glob('*.json'))


def test_crash_after_decisions_resumes_without_backfilling(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path, manifest = setup_trial(tmp_path)
    initialize(path)
    root = Path(manifest['storage'])
    real_link = os.link

    def crash_outcome(src: str | Path, dst: str | Path, **kwargs: Any) -> None:
        if Path(dst).parent == root / 'events' and b'"event_type":"outcome"' in Path(src).read_bytes():
            raise KeyboardInterrupt('simulated process loss')
        real_link(src, dst, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(os, 'link', crash_outcome)
        with pytest.raises(KeyboardInterrupt):
            run_cycle(root, 'day-0')
    before = events(root, 'decision')
    assert len(before) == 3
    assert not events(root, 'outcome')
    recover(root)
    assert run_cycle(root, 'day-0')['status'] == 'complete'
    assert events(root, 'decision') == before
    assert len(events(root, 'outcome')) == 3


def test_registration_is_immutable_and_observed_cannot_start_retroactively(tmp_path: Path) -> None:
    path, manifest = setup_trial(tmp_path)
    initialize(path)
    with pytest.raises((ValueError, FileExistsError), match='exist'):
        initialize(path)
    manifest['storage'] = str(tmp_path / 'observed')
    manifest['mode'] = 'observed_file'
    write(path, manifest)
    with pytest.raises(ValueError, match='fixture|retroactive|registration|future'):
        initialize(path)


def test_registered_raw_window_has_exact_geometry_and_no_history_trades(tmp_path: Path) -> None:
    _, manifest = setup_trial(tmp_path)
    step = manifest['schedule'][0]
    raw = json.loads(Path(step['raw_path']).read_text())
    result = build_window(manifest, step, raw, step['decision_not_before'])
    assert np.asarray(result['window']).shape == (9, 32, 8)
    assert result['feature_names'][0] == 'log_return_lag_31'
    assert result['feature_names'][-1] == 'xz_carry_lag_0'
    assert result['history_count'] == 63


def test_real_2025_frozen_bundle_runs_on_raw_fixture(tmp_path: Path) -> None:
    path, manifest = setup_trial(tmp_path)
    from forex_trainer.full_period import runtime_identity
    identities = json.loads(Path('docs/research/protocols/issue20/artifact-identities.json').read_text())
    ensemble_dir = Path(identities['forward_selection']['ppo']['ensemble_dir'])
    if not ensemble_dir.exists():
        pytest.skip(f'Real frozen bundle unavailable: {ensemble_dir}')
    runtime = runtime_identity()
    source = tmp_path / 'sources.json'
    write(source, {'source': identities['artifacts'][0], 'fold': '2025', 'runtime': {
        'forex_env_sha': runtime['git']['forex_env'], 'versions': runtime['versions'], 'device': 'cpu'}})
    manifest['policy_mode'] = 'frozen_2025'
    manifest['policy_sources'] = {'path': str(source), 'sha256': sha256_file(source)}
    write(path, manifest)
    initialize(path)
    root = Path(manifest['storage'])
    assert run_cycle(root, 'day-0')['status'] == 'complete'
    assert run_cycle(root, 'day-1')['status'] == 'complete'
    assert len(events(root, 'outcome')) == 6
    assert all(e['payload']['execution_basis'] == 'shadow_quote' for e in events(root, 'outcome'))
