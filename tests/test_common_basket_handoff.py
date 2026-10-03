"""Artifact and activation failures must be detected before downstream use."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from forex_trainer.common_basket_study import (
    FitLedger, ROOT, load_contract, require_merge, verify_bundle, write,
)


def test_fit_ledger_is_a_durable_json_line_stream(tmp_path: Path) -> None:
    path = tmp_path / 'fits.jsonl'
    ledger = FitLedger(path, 'fixture', 1)
    ledger.reserve('failed-attempt')
    events = [json.loads(line) for line in path.read_text().splitlines()]
    assert [event['event'] for event in events] == ['budget_reserved', 'reserved']
    assert events[-1]['consumed'] == 1


def test_registered_but_unmerged_revision_cannot_activate() -> None:
    _, registration = load_contract(ROOT / 'configs/research/issue41_common_basket.json')
    with pytest.raises(ValueError, match='merge'):
        require_merge(registration, '5766d3e' + '0' * 33)


def test_incomplete_bundle_and_incident_are_not_handoff_ready(tmp_path: Path) -> None:
    write(tmp_path / 'incident.json', {'status': 'execution_error'})
    with pytest.raises(ValueError, match='incident'):
        verify_bundle(tmp_path)


def test_modified_contract_is_rejected_without_fit(tmp_path: Path) -> None:
    config = json.loads((ROOT / 'configs/research/issue41_common_basket.json').read_text())
    config['horizons'] = [1]
    path = tmp_path / 'config.json'
    write(path, config)
    with pytest.raises(ValueError, match='horizons'):
        load_contract(path)
