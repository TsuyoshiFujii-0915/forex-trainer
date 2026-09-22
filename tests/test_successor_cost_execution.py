"""Exercise the successor panel through real frozen-policy inference."""
from __future__ import annotations

import copy
from dataclasses import replace
from pathlib import Path

import pytest

from forex_trainer.full_period import read_json
from forex_trainer.full_period_sources import load_campaign_sources
from forex_trainer.successor_cost_campaign import evaluate_sources, run_successor
from forex_trainer.supervised_portfolio import FOLDS
from test_full_period import prepare_fixture, sealed_fixture


def test_all_153_accounts_use_separate_outputs_and_preserve_frozen_models(sealed_fixture: Path, tmp_path: Path) -> None:
    source = load_campaign_sources(read_json(sealed_fixture), sealed_fixture)[0]
    plan, _, _ = prepare_fixture(tmp_path, 66)
    sources = [replace(source, fold=fold, plan=plan, identity=copy.deepcopy(source.identity)) for fold in FOLDS]
    parameters = source.ridge.parameter_sha256
    output = tmp_path / 'panel'
    output.mkdir()
    cells, sensitivities = evaluate_sources(sources, output)
    assert len(cells) == 153
    assert {cell['status'] for cell in cells} == {'complete'}
    assert len(sensitivities) == 102
    assert len(list(output.glob('fold-*/*-pairs.csv.gz'))) == 153
    assert all(cell['metrics']['steps'] == 2 for cell in cells)
    assert all(cell['coverage']['elapsed_seconds'] > 0 for cell in cells)
    assert source.ridge.parameter_sha256 == parameters
    assert not source.ridge.model.coefficients.flags.writeable
    assert not list(output.rglob('*.parquet'))
    with pytest.raises((FileExistsError, ValueError), match='exist'):
        evaluate_sources(sources, output)


def test_bad_snapshot_hash_fails_before_any_output(tmp_path: Path) -> None:
    import json
    config = read_json(Path('configs/research/issue29_successor_cost.json'))
    config['snapshot']['sha256'] = '0' * 64
    path = tmp_path / 'campaign.json'
    path.write_text(json.dumps(config))
    output = tmp_path / 'output'
    with pytest.raises(ValueError, match='hash mismatch'):
        run_successor(path, output)
    assert not output.exists()
