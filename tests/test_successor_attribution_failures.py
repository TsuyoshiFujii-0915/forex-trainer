"""Runtime failures preserve an inspectable, explicitly incomplete campaign."""
from __future__ import annotations

import copy
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
import yaml

from forex_trainer.artifact_provenance import sha256_file
from forex_trainer.cost_campaign import account_trace, write_json
from forex_trainer.full_period import build_period_env, canonical_action, evaluate_policy, read_json, runtime_identity
from forex_trainer.full_period_sources import FoldSources, FrozenEnsemble, load_campaign_sources
from forex_trainer.input_recovery import reference
from forex_trainer.profit_attribution import BASE_POLICIES, POLICIES, read_compressed, write_compressed
from forex_trainer.successor_attribution import (
    evaluate_fold, parent_accounts, report_exit_status, run_prepared_attribution,
    run_successor_attribution, verify_prepared_attribution,
)
from forex_trainer.supervised_portfolio import FOLDS
from test_full_period import prepare_fixture, sealed_fixture


class FailingInference:
    """A test policy that actually raises after a fixed number of real predictions."""

    def __init__(self, model: FrozenEnsemble, fail_after: int, output: Path) -> None:
        self.model = model
        self.fail_after = fail_after
        self.calls = 0
        self.output = output
        self.preserved: dict[Path, bytes] = {}

    def action(self, observation: dict[str, np.ndarray], window: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Preserve existing output bytes at the exact failing inference."""
        if self.calls == self.fail_after:
            self.preserved = {p: p.read_bytes() for p in self.output.rglob('*.gz')}
            raise RuntimeError('injected inference failure after real environment steps')
        self.calls += 1
        return self.model.action(observation, window)


def prepared_fixture(sealed_fixture: Path, tmp_path: Path) -> tuple[Path, list[FoldSources], list[dict[str, Any]], int]:
    """Create real parent accounts and models without mocking the evaluator or I/O."""
    source = load_campaign_sources(read_json(sealed_fixture), sealed_fixture)[0]
    plan, _, _ = prepare_fixture(tmp_path, 68)
    raw = yaml.safe_load(Path(source.identity['config_path']).read_text())
    raw['train_range'] = {'start': '2023-10-02', 'end': '2024-02-01'}
    raw['val_range'] = {'start': '2024-02-01', 'end': '2024-03-01'}
    raw['eval_range'] = {'start': '2024-03-01', 'end': '2025-01-01'}
    config_path = tmp_path / 'train-config.yaml'
    config_path.write_text(yaml.safe_dump(raw))
    source = replace(source, plan=plan, identity={**source.identity, 'config_path': str(config_path),
                                               'config_sha256': sha256_file(config_path)})
    market = pd.read_parquet(raw['env']['data']['path'])
    labels = market.index.tz_localize(None) if market.index.tz is not None else market.index
    train_steps = len(labels[(labels >= '2023-10-02') & (labels < '2024-02-01')]) - 64
    parent = tmp_path / 'parent'
    parent.mkdir()
    write_json(parent / 'manifest.json', {'runtime': runtime_identity()})
    results = []
    for policy, predict in [('canonical', canonical_action), ('ridge', source.ridge.action), ('ppo_ens3', source.ppo.action)]:
        env, windows = build_period_env(source.env_raw, plan, tmp_path / f'{policy}.parquet')
        try:
            result = evaluate_policy(env, windows, plan, predict)
        finally:
            env.close()
        result.update({'fold': '2009', 'policy': policy, 'scenario': 'F0'})
        pairs = account_trace(result, plan)
        results.append((result, pairs))
    for fold in FOLDS[:2]:
        work = parent / f'fold-{fold}'
        work.mkdir()
        for original, pairs in results:
            result = {**original, 'fold': fold}
            policy = result['policy']
            write_compressed(work / f'F0-{policy}.json.gz', result)
            pd.DataFrame(pairs).to_csv(work / f'F0-{policy}-pairs.csv.gz', index=False)
    config = read_json(Path('configs/research/issue30_successor_attribution.json'))
    config['parent_manifest'] = reference(parent / 'manifest.json')
    path = tmp_path / 'campaign.json'
    write_json(path, config)
    sources = [replace(source, fold=fold) for fold in FOLDS]
    coverage = [{'fold': fold, 'successor_scope': 'partial_year',
                 'first_decision': plan.timestamps[0], 'last_mark': plan.timestamps[-1]} for fold in FOLDS]
    return path, sources, coverage, train_steps


@pytest.mark.parametrize('stage', ['ppo_replay', 'train_replay', 'common_projected'])
def test_actual_runtime_failure_seals_all_cells_and_verifies_without_replay(
    sealed_fixture: Path, tmp_path: Path, stage: str, capsys: pytest.CaptureFixture[str],
) -> None:
    path, sources, coverage, train_steps = prepared_fixture(sealed_fixture, tmp_path)
    output = tmp_path / 'output'
    fail_after = {'ppo_replay': 1, 'train_replay': 5, 'common_projected': 4 + train_steps + 1}[stage]
    failing = FailingInference(sources[1].ppo, fail_after, output)
    sources[1] = replace(sources[1], ppo=failing)
    report = run_prepared_attribution(path, output, sources, coverage)
    assert report['status'] == 'incomplete'
    assert len(report['cells']) == 85
    assert report['completed_policy_folds'] == {'ppo_replay': 7, 'train_replay': 8, 'common_projected': 9}[stage]
    assert report['summaries'] is report['component_evidence'] is report['paired_evidence'] is None
    indexed = {(c['fold'], c['policy']): c for c in report['cells']}
    assert all(indexed['2009', p]['status'] == 'complete' for p in POLICIES)
    assert all(indexed[f, p]['status'] == 'not_run' for f in FOLDS[2:] for p in POLICIES)
    if stage == 'train_replay':
        assert indexed['2010', 'train_constant']['status'] == 'blocked_dependency'
        assert indexed['2010', 'common_projected']['status'] == 'not_run'
        assert not (output / 'fold-2010/train-replay.json.gz').exists()
    elif stage == 'ppo_replay':
        assert indexed['2010', 'ppo_ens3']['status'] == 'execution_error'
        assert all(indexed['2010', p]['status'] == 'blocked_dependency' for p in POLICIES[3:])
    else:
        assert indexed['2010', 'common_projected']['status'] == 'execution_error'
        assert indexed['2010', 'train_constant']['status'] == 'complete'
    failure = read_json(output / 'fold-2010/execution-error.json')
    assert failure['exception_type'] == 'RuntimeError'
    assert 'injected inference failure' in failure['message']
    assert 'action' in failure['traceback']
    assert failure['stage'] == ('train_replay' if stage == 'train_replay' else 'policy_replay')
    assert len(pd.read_csv(output / 'folds.csv')) == 85
    assert not list(output.rglob('*.parquet'))
    assert not (output / 'fold-2011').exists()
    for saved, before in failing.preserved.items():
        assert saved.read_bytes() == before
    assert failing.preserved
    calls = failing.calls
    verified = verify_prepared_attribution(output, sources, coverage)
    assert verified == report and failing.calls == calls
    assert report_exit_status(report) == report_exit_status(verified) == 2
    assert 'incomplete' in capsys.readouterr().out
    before = {p: p.read_bytes() for p in output.rglob('*') if p.is_file()}
    with pytest.raises(FileExistsError):
        run_prepared_attribution(path, output, sources, coverage)
    assert all(p.read_bytes() == data for p, data in before.items())
    (output / 'fold-2010/execution-error.json').write_text('{}')
    with pytest.raises(ValueError, match='hash mismatch'):
        verify_prepared_attribution(output, sources, coverage)


def test_bad_input_hash_still_fails_before_creating_an_attempt(tmp_path: Path) -> None:
    config = read_json(Path('configs/research/issue30_successor_attribution.json'))
    config['parent_manifest']['sha256'] = '0' * 64
    path = tmp_path / 'bad.json'
    write_json(path, config)
    output = tmp_path / 'output'
    with pytest.raises(ValueError, match='hash mismatch'):
        run_successor_attribution(path, output)
    assert not output.exists()


def test_parent_account_mismatch_is_not_recorded_as_runtime_failure(sealed_fixture: Path, tmp_path: Path) -> None:
    path, sources, _, _ = prepared_fixture(sealed_fixture, tmp_path)
    parents = parent_accounts(Path(read_json(path)['parent_manifest']['path']).parent, '2009')
    parents['canonical'][0]['trace'][0]['equity_before'] += 1
    output = tmp_path / 'broken'
    with pytest.raises((ValueError, AssertionError)):
        evaluate_fold(sources[0], parents, output)
    assert not (output / 'execution-error.json').exists()
