"""Review regressions using saved market artifacts without fitting or mocks."""
from __future__ import annotations

import gzip
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from forex_trainer.common_basket_study import ROOT, digest, encode, read, verify_bundle
from forex_trainer.artifact_provenance import sha256_file

BUNDLE = ROOT / 'docs/research/results/issue41/bundle'


def overwrite(path: Path, value: Any) -> None:
    """Replace only a temporary fixture's JSON bytes.

    Args:
        path: Temporary copy of an artifact.
        value: Mutated fixture value.
    """
    data = encode(value)
    path.write_bytes(gzip.compress(data, mtime=0) if path.suffix == '.gz' else data)


@pytest.mark.parametrize('mutation', [
    'model_fold', 'model_horizon', 'model_asof', 'row_asof', 'row_provenance',
    'cutoff', 'train_range', 'validation_range', 'train_label', 'validation_label',
    'covariance_asof', 'covariance_rows', 'config',
])
def test_semantic_mismatch_rejected_even_with_consistent_artifact_hashes(tmp_path: Path, mutation: str) -> None:
    directory = tmp_path / 'bundle'
    shutil.copytree(BUNDLE, directory)
    model_path = directory / 'model-2009-h1.json'
    model = read(model_path)
    p = model['provenance']
    rows = read(directory / 'forecasts.json.gz')
    affected = [r for r in rows if r['fold'] == '2009' and r['horizon_business_days'] == 1]
    if mutation == 'model_fold':
        model['fold'] = '2010'
    elif mutation == 'model_horizon':
        model['horizon_business_days'] = 5
    elif mutation == 'model_asof':
        p['parameter_as_of'] = '2030-01-01T00:00:00+00:00'
    elif mutation == 'row_asof':
        affected[0]['parameter_as_of'] = '2008-01-01T00:00:00+00:00'
    elif mutation == 'row_provenance':
        affected[0]['provenance_sha256'] = '0' * 64
    elif mutation == 'cutoff':
        next(r for r in affected if r['decision_at_utc'].startswith('2009-06-15'))['information_cutoff_utc'] = '2009-06-01T00:00:00+00:00'
    elif mutation == 'train_range':
        p['train_range_london'][1] = '2008-01-01'
    elif mutation == 'validation_range':
        p['validation_range_london'][1] = '2010-01-01'
    elif mutation == 'train_label':
        p['train']['target_end_utc'][-1] = '2007-01-01T00:00:00+00:00'
        p['covariance_as_of'] = p['train']['target_end_utc'][-1]
    elif mutation == 'validation_label':
        p['validation']['target_end_utc'][-1] = '2009-01-01T00:00:00+00:00'
        p['parameter_as_of'] = p['validation']['target_end_utc'][-1]
        for r in affected:
            r['parameter_as_of'] = '2008-01-01T00:00:00+00:00'
    elif mutation == 'covariance_asof':
        p['covariance_as_of'] = p['parameter_as_of']
    elif mutation == 'covariance_rows':
        p['covariance_rows_sha256'] = p['validation']['row_sha256']
    else:
        config = read(directory / 'config.json')
        config['features'] = list(reversed(config['features']))
        overwrite(directory / 'config.json', config)
    overwrite(model_path, model)
    for r in affected:
        r['model_sha256'] = sha256_file(model_path)
        if mutation != 'row_provenance':
            r['provenance_sha256'] = digest(p)
    overwrite(directory / 'forecasts.json.gz', rows)
    manifest = read(directory / 'manifest.json')
    del manifest['bundle_content_sha256']
    manifest['artifacts'] = {name: sha256_file(directory / name) for name in manifest['artifacts']}
    manifest['bundle_content_sha256'] = digest(manifest)
    overwrite(directory / 'manifest.json', manifest)
    with pytest.raises(ValueError, match='model|provenance|cutoff|label|range|covariance|config|as-of'):
        verify_bundle(directory)


def test_completed_market_bundle_remains_readable() -> None:
    assert len(verify_bundle(BUNDLE)) == 7138


@pytest.mark.parametrize('command', ['seal', 'run'])
@pytest.mark.parametrize('record', ['execution-status.json', 'bundle/manifest.json', 'bundle/fit-ledger.jsonl'])
def test_new_checkout_without_runs_cannot_reacquire_consumed_budget(tmp_path: Path, command: str, record: str) -> None:
    checkout = tmp_path / 'checkout'
    shutil.copytree(ROOT / 'src', checkout / 'src')
    result_path = Path('docs/research/results/issue41') / record
    destination = checkout / result_path
    destination.parent.mkdir(parents=True)
    shutil.copyfile(ROOT / result_path, destination)
    assert not (checkout / 'runs').exists()
    args = [sys.executable, '-m', 'forex_trainer.common_basket_study', command,
            '--config', str(ROOT / 'configs/research/issue41_common_basket.json')]
    if command == 'seal':
        args += ['--dependency-merge-commit', '1498af1b751ef0d7c4557246e21748f665d70cb2',
                 '--output', str(checkout / 'runs/seal')]
    else:
        args += ['--activation', str(BUNDLE / 'activation.json')]
    env = dict(os.environ, PYTHONPATH=str(checkout / 'src'))
    result = subprocess.run(args, cwd=checkout, env=env, capture_output=True, text=True)
    assert result.returncode == 2
    assert 'normal fit budget' in result.stderr, result.stderr
    assert not (checkout / 'runs').exists()
