"""Boundary failures and saved-ledger reconciliation without another replay."""
from __future__ import annotations

import json
from pathlib import Path
import shutil

import numpy as np
import pytest

from forex_trainer.artifact_provenance import sha256_file
from forex_trainer.common_allocation import allocate
from forex_trainer.common_basket_study import digest
from forex_trainer.common_quantity import instruction
from forex_trainer.common_quantity_study import verify_fixture
from test_common_allocation import state, forecast


def test_literal_hold_and_close_reject_quantity_substitution() -> None:
    current = state(np.ones(9) / 9)
    with pytest.raises(ValueError, match='hold'):
        instruction(current, current.quantities / 2, 'hold_quantity')
    with pytest.raises(ValueError, match='close'):
        instruction(current, current.quantities, 'close')


def test_unreduced_risk_must_not_reach_solver() -> None:
    current = state(np.array([1.01, *([0.] * 8)]))
    with pytest.raises(ValueError, match='risk'):
        allocate(current, forecast(.01, 1), 'cost')


def test_saved_cash_inconsistency_fails_even_if_outer_hashes_are_recomputed(tmp_path: Path) -> None:
    path = tmp_path / 'fixture'
    shutil.copytree('docs/research/results/issue42/fixture', path)
    account_path = path / 'accounts/hold.json'
    account = json.loads(account_path.read_text())
    account['trace'][1]['financing'] += 100.
    account_path.write_text(json.dumps(account))
    manifest_path = path / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    manifest['artifacts']['accounts/hold.json'] = sha256_file(account_path)
    manifest['content_sha256'] = digest({k: v for k, v in manifest.items() if k != 'content_sha256'})
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='accounting|reconciliation'):
        verify_fixture(path)


def test_saved_trace_requires_all_account_artifacts(tmp_path: Path) -> None:
    path = tmp_path / 'fixture'
    shutil.copytree('docs/research/results/issue42/fixture', path)
    manifest_path = path / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    del manifest['artifacts']['accounts/hold.json']
    manifest['content_sha256'] = digest({k: v for k, v in manifest.items() if k != 'content_sha256'})
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='artifact'):
        verify_fixture(path)
