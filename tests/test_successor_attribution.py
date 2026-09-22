"""Successor attribution preserves scope, accounting and independent controls."""
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
from forex_trainer.cost_campaign import account_trace
from forex_trainer.full_period import build_period_env, canonical_action, evaluate_policy, read_json
from forex_trainer.full_period_sources import load_campaign_sources
from forex_trainer.profit_attribution import POLICIES, attribute_account, read_compressed, write_compressed
from forex_trainer.successor_attribution import (
    CAMPAIGN_ID, evaluate_fold, summarize_successor_attribution, validate_config, verify_fold,
)
from forex_trainer.successor_cost_campaign import successor_metrics
from forex_trainer.supervised_portfolio import FOLDS
from test_full_period import prepare_fixture, sealed_fixture
from test_profit_attribution import evaluated


def attribution_panel(tmp_path: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Use reconciled account metrics in a labelled complete panel."""
    result, pairs = evaluated(tmp_path)
    _, attribution = attribute_account(result, pairs)
    cells = [{'fold': f, 'policy': p, 'status': 'complete', 'coverage': result['coverage'],
              'metrics': successor_metrics(result), 'attribution': attribution} for f in FOLDS for p in POLICIES]
    coverage = [{'fold': f, 'successor_scope': 'partial_year' if i < 9 else 'full_year',
                 'first_decision': result['coverage']['first_decision'], 'last_mark': result['coverage']['last_mark']}
                for i, f in enumerate(FOLDS)]
    return cells, coverage


def test_successor_85_cells_keep_old_annual_incomplete(tmp_path: Path) -> None:
    cells, coverage = attribution_panel(tmp_path)
    report = summarize_successor_attribution(cells, coverage)
    assert report['campaign_id'] == CAMPAIGN_ID
    assert report['completed_policy_folds'] == 85 and report['status'] == 'complete'
    assert report['original_annual_campaign_complete'] is False
    assert report['original_blocked_policy_folds'] == 45
    assert report['scope_counts'] == {'full_year': 8, 'partial_year': 9}
    assert len(report['component_evidence']['ppo_ens3']['common_annual_log']['leave_one_fold_out_means']) == 17
    assert report['paired_evidence']['common_projected_minus_ppo_ens3']['annualized_net_return']['mean_difference'] == 0
    assert report['summaries']['ppo_ens3']['all']['mean_metrics']['period_net_return'] == pytest.approx(cells[0]['metrics']['period_net_return'])
    assert report['summaries']['ppo_ens3']['all']['mean_metrics']['annualized_net_volatility'] > 0


@pytest.mark.parametrize('status', ['blocked_input', 'blocked_train_terminal', 'execution_error', 'incomplete_margin_call'])
def test_partial_successor_never_aggregates(tmp_path: Path, status: str) -> None:
    cells, coverage = attribution_panel(tmp_path)
    cells[-1]['status'] = status
    report = summarize_successor_attribution(cells, coverage)
    assert report['completed_policy_folds'] == 84
    assert report['summaries'] is report['paired_evidence'] is report['component_evidence'] is None


def test_changed_coverage_or_duplicate_cell_is_rejected(tmp_path: Path) -> None:
    cells, coverage = attribution_panel(tmp_path)
    wrong = copy.deepcopy(cells)
    wrong[0]['coverage']['first_decision'] = '2009-01-01'
    with pytest.raises(ValueError, match='coverage'):
        summarize_successor_attribution(wrong, coverage)
    with pytest.raises(ValueError, match='coverage|fold'):
        summarize_successor_attribution(cells, coverage[:-1])
    cells[-1] = cells[0]
    with pytest.raises(ValueError, match='85|unique|panel'):
        summarize_successor_attribution(cells, coverage)


def test_only_registered_parent_f0_and_two_controls_are_accepted() -> None:
    config = read_json(Path('configs/research/issue30_successor_attribution.json'))
    validate_config(config)
    for key, value in [('scenario', 'F1'), ('controls', ['relative_only']), ('folds', ['2024']),
                       ('train_gap_policy', 'drop_gaps'), ('campaign_id', 'issue30-profit-attribution-v1')]:
        with pytest.raises(ValueError):
            validate_config({**config, key: value})


def test_frozen_fold_reproduces_parent_and_seals_train_and_closed_loop_controls(sealed_fixture: Path, tmp_path: Path) -> None:
    source = load_campaign_sources(read_json(sealed_fixture), sealed_fixture)[0]
    plan, _, _ = prepare_fixture(tmp_path, 68)
    raw = yaml.safe_load(Path(source.identity['config_path']).read_text())
    raw['train_range'] = {'start': '2023-10-02', 'end': '2024-02-01'}
    raw['val_range'] = {'start': '2024-02-01', 'end': '2024-03-01'}
    raw['eval_range'] = {'start': '2024-03-01', 'end': '2025-01-01'}
    config_path = tmp_path / 'train-config.yaml'
    config_path.write_text(yaml.safe_dump(raw))
    identity = {**source.identity, 'config_path': str(config_path), 'config_sha256': sha256_file(config_path)}
    source = replace(source, plan=plan, identity=identity)
    parents = {}
    for policy, predict in [('canonical', canonical_action), ('ridge', source.ridge.action), ('ppo_ens3', source.ppo.action)]:
        env, windows = build_period_env(source.env_raw, plan, tmp_path / f'{policy}.parquet')
        try:
            result = evaluate_policy(env, windows, plan, predict)
        finally:
            env.close()
        result.update({'fold': source.fold, 'policy': policy, 'scenario': 'F0'})
        pairs = account_trace(result, plan)
        parents[policy] = (result, pairs)
    output = tmp_path / 'attribution'
    cells, steps = evaluate_fold(source, parents, output)
    assert len(cells) == 5 and len(steps) == 20
    assert {c['status'] for c in cells} == {'complete'}
    assert len(list(output.glob('*-pairs.csv.gz'))) == 5
    assert not list(output.glob('*.parquet'))
    training = read_compressed(output / 'train-replay.json.gz')
    assert all(pd.Timestamp(r['target_timestamp']).tz_localize(None) < pd.Timestamp('2024-02-01') for r in training['trace'])
    np.testing.assert_array_equal(training['weights'], np.mean([r['target_weights'] for r in training['trace']], axis=0))
    projected = read_compressed(output / 'common_projected.json.gz')
    direct = read_compressed(output / 'ppo_ens3.json.gz')
    assert projected['trace'][0]['assets_before'] == np.zeros((9, 3)).tolist()
    assert projected['trace'][1]['assets_before'] != direct['trace'][1]['assets_before']
    assert projected['trace'][1]['scores'] != direct['trace'][1]['scores']
    verify_fold(source, parents, output, cells)
    with pytest.raises((FileExistsError, ValueError), match='exist'):
        evaluate_fold(source, parents, output)
    changed = copy.deepcopy(cells)
    changed[0]['attribution']['common_annual_log'] += .1
    with pytest.raises(ValueError, match='attribution|cell'):
        verify_fold(source, parents, output, changed)
    path = output / 'common_projected.json.gz'
    projected['trace'][0]['target_weights'][0] += .1
    path.unlink()
    write_compressed(path, projected)
    with pytest.raises((ValueError, AssertionError), match='hash|weight|account|reconcil'):
        verify_fold(source, parents, output, cells)
