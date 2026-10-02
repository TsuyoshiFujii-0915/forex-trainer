"""Issue 41 input preflight, bounded fitting, and sealed forecast handoff CLI."""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import os
import platform
import subprocess
import sys
from dataclasses import asdict
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .artifact_provenance import sha256_file
from .common_basket import (
    ALPHAS, FEATURES, PAIRS, InputError, Model, Panel, Rows, build_panel,
    choose_alpha, diagnostics, fit_candidates, forecast_rows, standardize,
    training_rows, utc,
)

ROOT = Path(__file__).resolve().parents[2]
CAMPAIGN = 'issue40-common-direction-development-v3'
BUNDLE = 'issue40-short-forecast-v3'
RUN_DIRECTORY = 'runs/issue41-common-basket-v3'


def encode(value: Any) -> bytes:
    """Encode a finite JSON artifact deterministically.

    Args:
        value: JSON-compatible data.

    Returns:
        Canonical UTF-8 bytes.
    """
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n').encode()


def digest(value: Any) -> str:
    """Hash canonical JSON content.

    Args:
        value: JSON-compatible data.

    Returns:
        Content SHA-256.
    """
    return hashlib.sha256(encode(value)).hexdigest()


def write(path: Path, value: Any) -> None:
    """Durably create a new JSON artifact without overwriting any earlier bytes.

    Args:
        path: New artifact path.
        value: JSON-compatible data.
    """
    data = encode(value)
    if path.suffix == '.gz':
        data = gzip.compress(data, mtime=0)
    with path.open('xb') as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())


def read(path: Path) -> Any:
    """Read explicit JSON or gzip JSON.

    Args:
        path: Existing artifact.

    Returns:
        Decoded JSON value.
    """
    data = path.read_bytes()
    return json.loads(gzip.decompress(data) if path.suffix == '.gz' else data)


def checked(reference: dict[str, str]) -> Path:
    """Verify a repository-relative source reference before reading it.

    Args:
        reference: Exact path and SHA-256 source identity.

    Returns:
        Verified absolute path.
    """
    path = ROOT / reference['path']
    if sha256_file(path) != reference['sha256']:
        raise ValueError(f'SHA-256 mismatch: {path}')
    return path


class FitLedger:
    """Append-only reservations; interrupted and failed fits still consume budget."""

    def __init__(self, path: Path, counter: str, limit: int) -> None:
        self.path = path
        self.limit = limit
        self.consumed = 0
        self.items: set[str] = set()
        self.completed: set[str] = set()
        with path.open('x') as handle:
            handle.write(json.dumps({'event': 'budget_reserved', 'counter': counter, 'limit': limit}) + '\n')
            handle.flush()
            os.fsync(handle.fileno())

    def event(self, value: dict[str, Any]) -> None:
        """Append and fsync an execution event.

        Args:
            value: Event to persist before the next action.
        """
        with self.path.open('a') as handle:
            handle.write(json.dumps(value, sort_keys=True, allow_nan=False) + '\n')
            handle.flush()
            os.fsync(handle.fileno())

    def reserve(self, work_item: str) -> None:
        """Consume an attempt before fitting, never on success alone.

        Args:
            work_item: Unique fold/horizon/alpha work identity.
        """
        if self.consumed >= self.limit or work_item in self.items:
            raise ValueError(f'Fit budget exhausted or duplicate work item: {work_item}')
        self.event({'event': 'reserved', 'work_item': work_item, 'consumed': self.consumed + 1})
        self.items.add(work_item)
        self.consumed += 1

    def complete(self, work_item: str) -> None:
        """Record successful completion separately from consumption.

        Args:
            work_item: Previously reserved candidate identity.
        """
        if work_item not in self.items or work_item in self.completed:
            raise ValueError(f'Fit completion without unique reservation: {work_item}')
        self.event({'event': 'completed', 'work_item': work_item})
        self.completed.add(work_item)


def load_contract(config_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """Resolve the frozen v3 contract and recursively verify parent source seals.

    Args:
        config_path: Issue 41 CLI configuration.

    Returns:
        Configuration and verified v3 registration.
    """
    config = read(config_path)
    expected = {'campaign_id': CAMPAIGN, 'forecast_bundle_id': BUNDLE,
                'run_directory': RUN_DIRECTORY, 'horizons': [1, 5],
                'alpha_grid': list(ALPHAS), 'features': list(FEATURES), 'pairs': list(PAIRS)}
    if set(config) != set(expected) | {'registration', 'source_snapshot'}:
        raise ValueError('Unexpected Issue 41 configuration fields')
    for key, value in expected.items():
        if config[key] != value:
            raise ValueError(f'Issue 41 fixed contract differs: {key}')
    registration = read(checked(config['registration']))
    if registration['campaign_id'] != CAMPAIGN or registration['gates']['short_data_ready']['status'] != 'ready':
        raise ValueError('Issue 41 requires the v3 short_data_ready gate')
    if not all(registration['gate_requirements']['short_data_ready'].values()):
        raise ValueError('Issue 41 short gate requirements are not satisfied')
    for ref in registration['sources']:
        checked(ref)
    parent = read(checked(registration['parent_registration']))
    for ref in parent['sources']:
        checked(ref)
    snapshot_path = checked(config['source_snapshot'])
    if config['source_snapshot'] not in parent['sources']:
        raise ValueError('Snapshot not in inherited v2 source seals')
    snapshot = read(snapshot_path)
    checked(snapshot['source_data']['carry'][0])
    if tuple(f['fold'] for f in registration['folds']) != tuple(str(y) for y in range(2009, 2026)):
        raise ValueError('Issue 41 requires the complete ordered 17-fold panel')
    split_report = read(ROOT / 'docs/research/protocols/issue40/revisions/v3/preflight/report.json')
    for fold, split in zip(registration['folds'], split_report['fold_splits'], strict=True):
        if any(fold[key] != split[key] for key in ('fold', 'train_range_london', 'validation_range_london')):
            raise ValueError(f'Adopted split differs: {fold["fold"]}')
        if fold['train_range_london'][1] != fold['validation_range_london'][0] or fold['validation_range_london'][1] != fold['information_cutoff_london']:
            raise ValueError(f'Overlapping split or cutoff mismatch: {fold["fold"]}')
    if registration['counters']['issue41_regular_ridge_fit_calls'] != {'consumed': 0, 'limit': 102}:
        raise ValueError('Issue 41 normal fit reservation is not available')
    return config, registration


def market(reference: dict[str, str]) -> pd.DataFrame:
    """Load the sealed flattened parquet format without sorting or filling.

    Args:
        reference: Existing market cache path/hash.

    Returns:
        Market frame in its original pair order.
    """
    frame = pd.read_parquet(checked(reference))
    frame.columns = pd.MultiIndex.from_tuples([tuple(c.rsplit('|', 1)) for c in frame.columns])
    return frame


def row_identity(rows: Rows) -> dict[str, Any]:
    """Seal the exact training/validation row set, values and label ends.

    Args:
        rows: Purged observations.

    Returns:
        Complete reproducible fit range identity.
    """
    dates = [utc(day) for day in rows.dates]
    ends = [utc(day) for day in rows.target_dates]
    return {'rows': len(dates), 'decision_at_utc': dates, 'target_end_utc': ends,
            'row_sha256': digest({'dates': dates, 'target_ends': ends, 'features': rows.features.tolist(),
                                  'pair_returns': rows.pair_returns.tolist()}),
            'counts': rows.counts}


def evaluation_panel(config: dict[str, Any], fold: dict[str, Any]) -> tuple[Panel, tuple[date, ...], dict[str, Any]]:
    """Require the unchanged derived cache and complete account decision axis.

    Args:
        config: Verified CLI config.
        fold: Registered evaluation interval.

    Returns:
        Evaluation panel, all account decisions, and source identities.
    """
    snapshot = read(checked(config['source_snapshot']))
    reference = snapshot['derived_data'][fold['fold']]
    frame = market(reference)
    panel = build_panel(frame)
    evaluation = read(ROOT / 'docs/research/results/issue39/snapshot/evaluation.json')
    spec = next(row for row in evaluation['folds'] if row['fold'] == fold['fold'])
    calendar = read(checked(spec['calendar']))
    if tuple(calendar['symbols']) != PAIRS or calendar['timezone'] != 'Europe/London' or calendar['holidays'] != []:
        raise ValueError(f'Calendar contract differs: {fold["fold"]}')
    labels = tuple(pd.Timestamp(row['bar_label']).tz_convert('Europe/London').date() for row in calendar['sessions'])
    for row in calendar['sessions']:
        day = pd.Timestamp(row['bar_label']).tz_convert('Europe/London').date()
        if pd.Timestamp(row['session_close']) != pd.Timestamp(utc(day)):
            raise ValueError(f'Calendar close differs from registered London midnight: {day}')
    first = pd.Timestamp(fold['first_decision']).tz_convert('Europe/London').date()
    last = pd.Timestamp(fold['last_mark']).tz_convert('Europe/London').date()
    first_index = labels.index(first)
    if first_index < 63 or labels[first_index-63:labels.index(last)+1] != panel.dates:
        raise ValueError(f'Evaluation cache/calendar history axis differs: {fold["fold"]}')
    measured = tuple(day for day in labels if first <= day <= last)
    if len(measured) != fold['measured_bars'] or len(measured) < 2 or measured[-1] != last or measured[0] != first:
        raise ValueError(f'Evaluation measured bars differ: {fold["fold"]}')
    expected = tuple(pd.bdate_range(first, last).date)
    if measured != expected:
        raise ValueError(f'Evaluation gap in {fold["fold"]}')
    indices = [panel.dates.index(day) for day in measured[:-1]]
    if np.any(panel.history[indices] < 63):
        raise InputError(f'Evaluation history missing: {fold["fold"]}', [{'reason': 'evaluation_history'}])
    return panel, measured[:-1], {'data': reference, 'calendar': spec['calendar'],
                                'decision_sha256': digest([utc(day) for day in measured[:-1]])}


def preflight(config_path: Path, output: Path) -> dict[str, Any]:
    """Check all real inputs and all 34 cases without fitting or market inference.

    Args:
        config_path: Issue 41 config.
        output: New audit directory.

    Returns:
        Complete status and all case counts, or explicit blocked-input evidence.
    """
    output.mkdir(parents=True, exist_ok=False)
    report: dict[str, Any] = {'campaign_id': CAMPAIGN, 'status': 'blocked_input', 'fit_calls': 0,
                              'market_inference_calls': 0, 'cases': [],
                              'execution_status': 'requires_merge_and_runtime_seal'}
    exclusions: list[dict[str, Any]] = []
    try:
        config, registration = load_contract(config_path)
        report['cases'] = [{'fold': f['fold'], 'horizon_business_days': h, 'status': 'blocked_input'}
                           for f in registration['folds'] for h in (1, 5)]
        snapshot = read(checked(config['source_snapshot']))
        source = snapshot['source_data']['carry'][0]
        panel = build_panel(market(source))
        from docs.research.protocols.issue40.preflight import calendar_blocks
        coverage = read(ROOT / 'docs/research/results/issue39/snapshot/coverage.json')
        expected_dates = tuple(day for block in calendar_blocks(coverage) for day in block)
        observed_dates = tuple(day for day in panel.dates if date(2003, 6, 1) <= day < date(2026, 1, 1))
        if observed_dates != expected_dates:
            raise ValueError('Training cache differs from registered London calendar coverage')
        all_dates = tuple(pd.bdate_range('2003-06-01', '2025-12-31').date)
        present = set(observed_dates)
        write(output / 'missing-calendar.json.gz', [utc(day) for day in all_dates if day not in present])
        case_inputs = []
        with (ROOT / 'docs/research/protocols/issue40/revisions/v3/preflight/counts.csv').open() as handle:
            calendar_counts = {(r['fold'], int(r['horizon'])): r for r in csv.DictReader(handle)}
        for fold in registration['folds']:
            eval_panel, decisions, eval_source = evaluation_panel(config, fold)
            for horizon in (1, 5):
                case = next(c for c in report['cases'] if c['fold'] == fold['fold'] and c['horizon_business_days'] == horizon)
                case['expected_decisions'] = len(decisions)
                train = training_rows(panel, *(date.fromisoformat(v) for v in fold['train_range_london']), horizon)
                validation = training_rows(panel, *(date.fromisoformat(v) for v in fold['validation_range_london']), horizon)
                for kind, rows in (('train', train), ('validation', validation)):
                    case[f'{kind}_rows'] = len(rows.targets)
                    case[f'{kind}_counts'] = rows.counts
                    exclusions.extend({'fold': fold['fold'], 'horizon': horizon, 'range': kind, **row}
                                      for row in rows.excluded)
                    registered = calendar_counts[fold['fold'], horizon]
                    for key, value in rows.counts.items():
                        suffix = 'rows' if key == 'eligible_rows' else key
                        if int(registered[f'{kind}_{suffix}']) != value:
                            raise ValueError(f'Calendar count mismatch: {fold["fold"]}/h{horizon}/{kind}/{key}')
                eligible = np.arange(63, len(eval_panel.dates)-horizon)
                diagnostic_targets = eval_panel.prices[eligible+horizon] / eval_panel.prices[eligible] - 1
                if not np.isfinite(diagnostic_targets).all():
                    raise InputError(f'Nonfinite evaluation target: {fold["fold"]}/h{horizon}',
                                     [{'reason': 'nonfinite_evaluation_target', 'fold': fold['fold'], 'horizon': horizon}])
                if len(train.targets) < 252 or len(validation.targets) < 60:
                    case['reason'] = 'minimum_rows'
                    continue
                try:
                    standardize(train.features)
                except InputError as exc:
                    exclusions.extend({'fold': fold['fold'], 'horizon': horizon, **row} for row in exc.rows)
                    case['reason'] = str(exc)
                    continue
                case['status'] = 'input_ready'
                case_inputs.append({'fold': fold['fold'], 'horizon': horizon,
                                    'train': row_identity(train), 'validation': row_identity(validation),
                                    'evaluation': eval_source})
        write(output / 'row-identities.json.gz', case_inputs)
        report.update({'status': 'input_ready' if all(c['status'] == 'input_ready' for c in report['cases']) else 'blocked_input',
                       'config_sha256': sha256_file(config_path), 'registration': config['registration'],
                       'source_data': source, 'features': list(FEATURES), 'pairs': list(PAIRS)})
    except (ValueError, OSError, KeyError) as exc:
        report['error'] = f'{type(exc).__name__}: {exc}'
        if isinstance(exc, InputError):
            exclusions.extend(exc.rows)
    write(output / 'excluded-rows.json.gz', exclusions)
    report['artifacts'] = {path.name: sha256_file(path) for path in sorted(output.iterdir())}
    write(output / 'report.json', report)
    return report


def git(args: list[str], directory: Path) -> str:
    """Run a read-only Git query, propagating its exact failure.

    Args:
        args: Git query arguments.
        directory: Repository location.

    Returns:
        Stripped standard output.
    """
    return subprocess.run(['git', *args], cwd=directory, text=True, capture_output=True, check=True).stdout.strip()


def execution_runtime() -> dict[str, Any]:
    """Require clean repositories and identify numerical runtime and source bytes.

    Returns:
        Clean Git, lock, Python, dependencies and CPU identities.
    """
    import forex_env
    from .full_period import runtime_identity
    env_root = Path(forex_env.__file__).resolve().parents[2]
    for repo in (ROOT, env_root):
        status = git(['status', '--porcelain', '--untracked-files=all'], repo)
        if status:
            raise ValueError(f'Execution seal requires clean repository: {repo}\n{status}')
    return {**runtime_identity(), 'python': sys.version, 'platform': platform.platform(),
            'machine': platform.machine(), 'processor': platform.processor(),
            'cpu_model': sorted({line.strip() for line in Path('/proc/cpuinfo').read_text().splitlines()
                                 if line.startswith(('model name', 'vendor_id', 'flags'))}),
            'lock_sha256': sha256_file(ROOT / 'uv.lock')}


def require_merge(registration: dict[str, Any], merge_commit: str) -> None:
    """Require a v3-containing commit already present on the fetched main branch.

    Args:
        registration: Frozen v3 registration.
        merge_commit: Reviewed and merged dependency commit.
    """
    if len(merge_commit) != 40 or any(c not in '0123456789abcdef' for c in merge_commit):
        raise ValueError('A full dependency merge commit SHA is required')
    try:
        git(['merge-base', '--is-ancestor', registration['registration_commit'], merge_commit], ROOT)
        git(['merge-base', '--is-ancestor', merge_commit, 'origin/main'], ROOT)
        registered = git(['show', f'{merge_commit}:docs/research/protocols/issue40/revisions/v3/registration.json'], ROOT)
        if json.loads(registered) != registration:
            raise ValueError('Dependency merge does not contain the adopted v3 registration')
    except subprocess.CalledProcessError as exc:
        raise ValueError('Dependency v3 merge is not verified on origin/main') from exc


def require_activation(config: dict[str, Any], registration: dict[str, Any], activation: dict[str, Any] | None) -> None:
    """Validate the separate pre-execution seal without gating on RL readiness.

    Args:
        config: Fixed Issue 41 configuration.
        registration: Verified short-campaign registration.
        activation: Explicit execution seal, absent before dependency merge.
    """
    if activation is None:
        raise ValueError('Missing activation: dependency review/merge and execution seal required')
    if activation['campaign_id'] != CAMPAIGN or activation['config_content_sha256'] != digest(config):
        raise ValueError('Execution activation/config identity mismatch')
    if activation['registration'] != config['registration'] or activation['regular_fit_limit'] != 102:
        raise ValueError('Execution activation registration/budget mismatch')
    require_merge(registration, activation['dependency_merge_commit'])
    if activation['runtime'] != execution_runtime():
        raise ValueError('Execution activation runtime differs from current clean runtime')


def seal(config_path: Path, merge_commit: str, output: Path) -> dict[str, Any]:
    """Seal the verified inputs/runtime after the dependency has merged.

    Args:
        config_path: Fixed CLI config.
        merge_commit: Dependency commit verified against origin/main.
        output: Fresh directory for preflight and activation.

    Returns:
        Execution activation for the one fixed normal-fit attempt.
    """
    config, registration = load_contract(config_path)
    require_merge(registration, merge_commit)
    runtime = execution_runtime()
    report = preflight(config_path, output)
    if report['status'] != 'input_ready':
        raise ValueError(f'Execution blocked by input preflight: {output / "report.json"}')
    activation = {'campaign_id': CAMPAIGN, 'config_content_sha256': digest(config),
                  'config_sha256': sha256_file(config_path), 'registration': config['registration'],
                  'dependency_merge_commit': merge_commit, 'runtime': runtime,
                  'regular_fit_limit': 102, 'refits': 0, 'shared_retry_consumed': 0,
                  'seal_command': ['forex-common-basket', 'seal', '--config', str(config_path),
                                   '--dependency-merge-commit', merge_commit, '--output', str(output)],
                  'run_command': ['forex-common-basket', 'run', '--config', str(config_path),
                                  '--activation', str(output / 'activation.json')],
                  'preflight_report_sha256': sha256_file(output / 'report.json'),
                  'row_identities_sha256': sha256_file(output / 'row-identities.json.gz')}
    write(output / 'activation.json', activation)
    return activation


def model_payload(model: Model) -> dict[str, Any]:
    """Serialize a candidate without pickles or another fit.

    Args:
        model: Fitted candidate.

    Returns:
        JSON numeric parameters including train-only covariance.
    """
    return {key: value.tolist() if isinstance(value, np.ndarray) else value for key, value in asdict(model).items()}


def run(config_path: Path, activation_path: Path) -> dict[str, Any]:
    """Fit all 34 models before producing any evaluation prediction.

    Args:
        config_path: Fixed Issue 41 config.
        activation_path: Previously sealed activation.

    Returns:
        Complete bundle manifest, or an incident retained in the fixed run directory.
    """
    config, registration = load_contract(config_path)
    activation = read(activation_path)
    require_activation(config, registration, activation)
    if sha256_file(config_path) != activation['config_sha256']:
        raise ValueError('Execution configuration bytes differ from activation seal')
    if sha256_file(activation_path.parent / 'report.json') != activation['preflight_report_sha256']:
        raise ValueError('Activation preflight report hash mismatch')
    output = ROOT / RUN_DIRECTORY
    output.mkdir(parents=True, exist_ok=False)
    ledger: FitLedger | None = None
    cells = [{'fold': f['fold'], 'horizon': h, 'status': 'not_started'} for f in registration['folds'] for h in (1, 5)]
    try:
        report = preflight(config_path, output / 'preflight')
        if report['status'] != 'input_ready' or sha256_file(output / 'preflight/row-identities.json.gz') != activation['row_identities_sha256']:
            raise InputError('Run preflight differs from activation or is blocked', [{'reason': 'input_seal'}])
        write(output / 'activation.json', activation)
        write(output / 'config.json', config)
        snapshot = read(checked(config['source_snapshot']))
        identities = {(row['fold'], row['horizon']): row
                      for row in read(output / 'preflight/row-identities.json.gz')}
        panel = build_panel(market(snapshot['source_data']['carry'][0]))
        ledger = FitLedger(output / 'fit-ledger.jsonl', 'issue41_regular_ridge_fit_calls', 102)
        selected: dict[tuple[str, int], tuple[Model, str, dict[str, Any]]] = {}
        for cell in cells:
            fold = next(f for f in registration['folds'] if f['fold'] == cell['fold'])
            horizon = cell['horizon']
            cell['status'] = 'fitting'
            train = training_rows(panel, *(date.fromisoformat(v) for v in fold['train_range_london']), horizon)
            validation = training_rows(panel, *(date.fromisoformat(v) for v in fold['validation_range_london']), horizon)
            candidates = fit_candidates(train, validation, ledger, f'{fold["fold"]}/h={horizon}')
            ledger.event({'event': 'validation_results_opened', 'fold': fold['fold'], 'horizon': horizon,
                          'candidate_mse': {str(m.alpha): m.validation_mse for m in candidates}})
            model = choose_alpha(candidates)
            ledger.event({'event': 'alpha_selected_without_refit', 'fold': fold['fold'], 'horizon': horizon,
                          'alpha': model.alpha})
            provenance = {'train': row_identity(train), 'validation': row_identity(validation),
                          'train_range_london': fold['train_range_london'],
                          'validation_range_london': fold['validation_range_london'],
                          'parameter_as_of': utc(max(validation.target_dates)),
                          'covariance_as_of': utc(max(train.target_dates)),
                          'covariance_formula': '0.9*S+0.1*diag(S)+1e-8*I; S=sample_cov(pair_h_simple_returns,ddof=1)',
                          'covariance_rows_sha256': row_identity(train)['row_sha256'],
                          'covariance_sha256': digest(model.covariance.tolist()),
                          'prediction_unit': 'basket_h_business_day_price_simple_return',
                          'pair_order': list(PAIRS), 'feature_order': list(FEATURES),
                          'registration': config['registration'], 'source_data': snapshot['source_data']['carry'][0],
                          'config_sha256': sha256_file(config_path), 'runtime_sha256': digest(activation['runtime'])}
            provenance['evaluation'] = identities[fold['fold'], horizon]['evaluation']
            payload = {'fold': fold['fold'], 'horizon_business_days': horizon,
                       'selected': model_payload(model), 'candidates': [model_payload(m) for m in candidates],
                       'provenance': provenance, 'refits': 0,
                       'train_diagnostics': diagnostics(train.targets, model.predict(train.features)),
                       'validation_diagnostics': diagnostics(validation.targets, model.predict(validation.features))}
            filename = f'model-{fold["fold"]}-h{horizon}.json'
            write(output / filename, payload)
            model_hash = sha256_file(output / filename)
            selected[fold['fold'], horizon] = model, model_hash, provenance
            cell.update(status='model_sealed', model_path=filename, model_sha256=model_hash,
                        alpha=model.alpha, train_rows=len(train.targets), validation_rows=len(validation.targets))
        write(output / 'models-sealed.json', {'cells': cells, 'fit_calls': ledger.consumed})
        ledger.event({'event': 'all_models_sealed_before_evaluation', 'model_count': len(selected)})
        forecasts = []
        for cell in cells:
            fold = next(f for f in registration['folds'] if f['fold'] == cell['fold'])
            panel, decisions, source = evaluation_panel(config, fold)
            model, model_hash, provenance = selected[fold['fold'], cell['horizon']]
            rows = forecast_rows(panel, model, decisions, cell['horizon'],
                                 date.fromisoformat(fold['information_cutoff_london']), fold['fold'], model_hash)
            for row in rows:
                row.update(campaign_id=CAMPAIGN, forecast_bundle_id=BUNDLE,
                           parameter_as_of=provenance['parameter_as_of'], provenance_sha256=digest(provenance))
            forecasts.extend(rows)
            labeled = [row for row in rows if row['diagnostic_target_status'] == 'available']
            cell.update(status='complete', account_decisions=len(rows), diagnostic_labels=len(labeled),
                        evaluation=source, diagnostics=diagnostics(
                            np.array([r['diagnostic_target'] for r in labeled]),
                            np.array([r['predicted_basket_simple_return'] for r in labeled])))
        if execution_runtime() != activation['runtime']:
            raise ValueError('Runtime changed during fitting or evaluation')
        load_contract(config_path)
        write(output / 'forecasts.json.gz', forecasts)
        summaries = []
        for horizon in (1, 5):
            for era, start, end in (('all', 2009, 2025), ('early', 2009, 2018), ('late', 2019, 2025)):
                members = [c for c in cells if c['horizon'] == horizon and start <= int(c['fold']) <= end]
                fields = ('mse', 'zero_mse', 'mse_minus_zero', 'direction_agreement', 'prediction_variance')
                complete = all(c['diagnostics']['status'] == 'available' for c in members)
                summaries.append({'horizon': horizon, 'era': era, 'folds': len(members),
                                  'status': 'available' if complete else 'undefined_missing_fold_labels',
                                  'equal_fold_mean': {k: float(np.mean([c['diagnostics'][k] for c in members]))
                                                      for k in fields} if complete else None})
        write(output / 'diagnostics.json', {'cells': cells, 'era_summary': summaries,
                                           'sample_unit': 'fold; overlapping labels and pairs are not independent observations'})
        ledger.event({'event': 'evaluation_results_opened', 'portfolio_decision': 'deferred_to_issue43'})
        manifest = {'campaign_id': CAMPAIGN, 'forecast_bundle_id': BUNDLE, 'status': 'complete',
                    'fit_calls': ledger.consumed, 'selected_models': len(selected), 'refits': 0,
                    'portfolio_decision': 'deferred_to_issue43', 'cells': cells,
                    'forecast_rows': len(forecasts), 'activation': activation,
                    'artifacts': {str(p.relative_to(output)): sha256_file(p) for p in sorted(output.rglob('*')) if p.is_file()}}
        manifest['bundle_content_sha256'] = digest(manifest)
        write(output / 'manifest.json', manifest)
        verify_bundle(output)
        return manifest
    except Exception as exc:
        write(output / 'incident.json', {'status': 'blocked_input' if isinstance(exc, InputError) else 'execution_error',
                                        'error': f'{type(exc).__name__}: {exc}', 'cells': cells,
                                        'fit_calls': ledger.consumed if ledger is not None else 0,
                                        'retry_calls': 0, 'automatic_retry': False})
        raise


def verify_bundle(directory: Path) -> list[dict[str, Any]]:
    """Validate the exact complete forecast interface consumed by fixed and cost.

    Args:
        directory: Completed immutable Issue 41 run directory.

    Returns:
        All forecast rows, never a successful-cell intersection.
    """
    if (directory / 'incident.json').exists():
        raise ValueError('Forecast bundle has an unresolved execution incident')
    manifest = read(directory / 'manifest.json')
    content_hash = manifest.pop('bundle_content_sha256')
    if digest(manifest) != content_hash or manifest['status'] != 'complete' or manifest['forecast_bundle_id'] != BUNDLE:
        raise ValueError('Forecast bundle manifest identity/status mismatch')
    if manifest['fit_calls'] != 102 or manifest['selected_models'] != 34 or manifest['refits'] != 0:
        raise ValueError('Forecast bundle fit/model counts mismatch')
    for name, expected in manifest['artifacts'].items():
        path = (directory / name).resolve()
        if not path.is_relative_to(directory.resolve()) or sha256_file(path) != expected:
            raise ValueError(f'Forecast bundle artifact mismatch: {name}')
    config = read(directory / 'config.json')
    _, registration = load_contract(ROOT / 'configs/research/issue41_common_basket.json')
    rows = read(directory / 'forecasts.json.gz')
    expected_keys: set[tuple[str, int, str]] = set()
    model_hashes = {}
    for fold in registration['folds']:
        _, decisions, _ = evaluation_panel(config, fold)
        for horizon in (1, 5):
            model_hashes[fold['fold'], horizon] = sha256_file(directory / f'model-{fold["fold"]}-h{horizon}.json')
            expected_keys.update((fold['fold'], horizon, utc(day)) for day in decisions)
    seen = set()
    for row in rows:
        key = (row['fold'], row['horizon_business_days'], row['decision_at_utc'])
        if key in seen or key not in expected_keys:
            raise ValueError(f'Unexpected/duplicate forecast key: {key}')
        seen.add(key)
        if row['model_sha256'] != model_hashes[key[:2]] or not np.isfinite(row['predicted_basket_simple_return']):
            raise ValueError(f'Forecast model/value mismatch: {key}')
        if not pd.Timestamp(row['parameter_as_of']) < pd.Timestamp(row['information_cutoff_utc']) <= pd.Timestamp(row['decision_at_utc']):
            raise ValueError(f'Forecast as-of violation: {key}')
        if row['campaign_id'] != CAMPAIGN or row['forecast_bundle_id'] != BUNDLE:
            raise ValueError(f'Forecast campaign mismatch: {key}')
    if seen != expected_keys or len(rows) != manifest['forecast_rows']:
        raise ValueError('Missing forecast decisions; intersections are prohibited')
    return rows


def main() -> int:
    """Execute the explicitly selected input, seal, fit, or verification stage.

    Returns:
        Zero on success, two for an explicit blocked/error state.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    for name in ('preflight', 'seal', 'run'):
        command = commands.add_parser(name)
        command.add_argument('--config', type=Path, required=True)
        if name != 'run':
            command.add_argument('--output', type=Path, required=True)
        if name == 'seal':
            command.add_argument('--dependency-merge-commit', required=True)
        if name == 'run':
            command.add_argument('--activation', type=Path, required=True)
    verify = commands.add_parser('verify')
    verify.add_argument('--bundle', type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.command == 'preflight':
            report = preflight(args.config, args.output)
            print(json.dumps({'status': report['status'], 'cases': len(report['cases']), 'fit_calls': 0}))
            return 0 if report['status'] == 'input_ready' else 2
        if args.command == 'seal':
            seal(args.config, args.dependency_merge_commit, args.output)
        elif args.command == 'run':
            run(args.config, args.activation)
        else:
            verify_bundle(args.bundle)
        return 0
    except (ValueError, OSError, KeyError, subprocess.CalledProcessError) as exc:
        print(f'{type(exc).__name__}: {exc}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
