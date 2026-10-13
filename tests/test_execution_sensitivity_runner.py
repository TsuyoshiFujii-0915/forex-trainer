"""Exercise real campaign persistence with synthetic data and untrained models."""
from __future__ import annotations

from dataclasses import replace
from datetime import timedelta
from pathlib import Path
from types import MethodType
from typing import Any, cast

import numpy as np
import pytest

from forex_trainer import execution_sensitivity_study as study
from forex_trainer.artifact_provenance import sha256_file
from forex_trainer.common_basket import PAIRS
from forex_trainer.common_basket_study import read
from forex_trainer.common_quantity import marks_from_period
from forex_trainer.full_period import FrozenRidge, canonical_action, prepare_period
from forex_trainer.full_period_sources import FoldSources, FrozenEnsemble
from forex_trainer.quote_replay_policies import fixture_policies
from forex_trainer.supervised_portfolio import FOLDS
from test_common_basket import market_fixture


@pytest.fixture(scope='module')
def synthetic_inputs() -> study.Inputs:
    """Build synthetic marks and actual frozen fixture models without fitting.

    Returns:
        Seventeen synthetic folds for the real 306-cell runner.
    """
    market = market_fixture().iloc[:66].copy()
    dates = market.index
    calendar = {'version': 'synthetic-london', 'timezone': 'Europe/London', 'symbols': list(PAIRS),
                'holidays': [], 'label_rule': 'London midnight historical assumed close',
                'sessions': [{'bar_label': at.isoformat(),
                              'session_open': (at - timedelta(hours=12)).isoformat(),
                              'session_close': at.isoformat(), 'available_at': None} for at in dates]}
    plan = prepare_period(market, calendar, dates[63].isoformat(),
                          (dates[-1] + timedelta(days=1)).isoformat(), PAIRS, Path('synthetic-runner'))
    axis = marks_from_period(plan)
    policies, identity = fixture_policies()
    ridge = cast(FrozenRidge, cast(MethodType, policies['ridge']).__self__)
    ensemble = cast(FrozenEnsemble, cast(MethodType, policies['ppo_ens3']).__self__)
    sources = [FoldSources(fold, {}, plan, ridge, ensemble, identity) for fold in FOLDS]
    return study.Inputs(read(study.CONFIG), sources, {fold: axis for fold in FOLDS},
                        {fold: {'fixture': True, 'models': identity} for fold in FOLDS})


def activation(inputs: study.Inputs) -> dict[str, Any]:
    """Bind a synthetic runner invocation to its explicit fixture identities.

    Args:
        inputs: Synthetic inputs, never passed through market activation.

    Returns:
        Isolated fixture activation with a deterministic runtime sample.
    """
    return {'config': inputs.config, 'config_sha256': study.CONFIG_SHA256,
            'registration_merge_commit': 'synthetic-runner-only',
            'runtime': {'fixture_runtime': 'sealed'}, 'identities': inputs.identities,
            'cells': study.registered_cells(), 'effective_proxy_accounts': 306,
            'effective_quote_accounts': 0}


def account_hashes(directory: Path) -> dict[str, str]:
    """Snapshot every saved account before the runner's final runtime check.

    Args:
        directory: Temporary synthetic campaign directory.

    Returns:
        Account basenames and hashes for preservation assertions.
    """
    return {p.name: sha256_file(p) for p in sorted((directory / 'accounts').glob('*.json.gz'))}


def verify_synthetic(directory: Path, inputs: study.Inputs) -> dict[str, Any]:
    """Use the same inventory and reconciliation stages as the verify CLI.

    Args:
        directory: Saved synthetic campaign.
        inputs: Original synthetic input identities and prices.

    Returns:
        Read-only verification result from the production verifier.
    """
    manifest = study.verify_manifest(directory)
    return study.verify_accounts(directory, inputs, manifest)


@pytest.mark.parametrize('outcome', ['complete', 'strategy_terminal'])
def test_real_runner_preserves_normal_and_terminal_outcomes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, synthetic_inputs: study.Inputs, outcome: str,
) -> None:
    inputs = synthetic_inputs
    if outcome == 'strategy_terminal':
        axis = inputs.marks[FOLDS[0]]
        observation = {'market': axis[0].window.astype(np.float32), 'assets': np.zeros((9, 3), dtype=np.float32)}
        _, action = canonical_action(observation, axis[0].window)
        adverse = np.where(action[:, 0] > 0, .001, np.where(action[:, 0] < 0, 100., 1.))
        terminal_axis = (axis[0], axis[1], replace(axis[2], prices=axis[0].prices * adverse))
        inputs = replace(inputs, marks={fold: terminal_axis for fold in FOLDS})
    sealed = activation(inputs)
    directory = tmp_path / 'campaign'
    snapshots: list[dict[str, str]] = []

    def current_runtime() -> dict[str, Any]:
        snapshots.append(account_hashes(directory))
        return sealed['runtime']

    monkeypatch.setattr(study, 'execution_runtime', current_runtime)
    report = study.execute_accounts(inputs, sealed, directory)
    assert report['proxy_status'] == 'complete'
    assert report['incident'] is None
    assert len(snapshots) == 1 and len(snapshots[0]) == 306
    assert account_hashes(directory) == snapshots[0]
    assert read(directory / 'manifest.json')['proxy_accounts_consumed'] == 306
    if outcome == 'complete':
        assert report['status_counts']['complete'] == 306
        assert all(c['status'] == 'available' for c in report['comparisons'])
    else:
        assert report['status_counts']['strategy_terminal'] > 0
        assert any(c['status'] == 'pending_measurement' for c in report['comparisons'])
        assert report['status_counts']['complete'] + report['status_counts']['strategy_terminal'] == 306
    verified = verify_synthetic(directory, inputs)
    assert verified['status'] == 'verified'
    assert verified['proxy_status'] == 'complete'
    assert verified['proxy_accounts_consumed'] == 306
    assert verified['accounts_reexecuted'] == 0
    assert account_hashes(directory) == snapshots[0]
