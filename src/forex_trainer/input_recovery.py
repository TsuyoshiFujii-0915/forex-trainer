"""Finite input audit and immutable contiguous research snapshot (Issue #39)."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from forex_env.data.base import validate_ohlcv
from forex_env.data.file_provider import save_ohlcv_parquet

from .artifact_provenance import sha256_file
from .cost_campaign import validate_campaign
from .config import RangeConfig, parse_experiment_config, resolve_env_raw
from .env_factory import GateEvaluationMode, build_single_env
from .features import CROSS_FEATURE_REGISTRY, FEATURE_REGISTRY
from .full_period import (
    FEATURES, build_period_env, canonical_action, checked_path, evaluate_policy, json_hash,
    prepare_period, read_json, require_comparable, runtime_identity, utc_instant,
)
from .full_period_sources import FoldSources, load_campaign_sources
from .profit_attribution import validate_config as validate_attribution_config

CONTRACT_ID = 'issue39-contiguous-snapshot-v1'
POLICIES = ('canonical', 'ridge', 'ppo_ens3')


def write_immutable_json(path: Path, value: Any) -> None:
    """Create a finite JSON artifact without overwriting an earlier attempt.

    Args:
        path: New output path.
        value: Serializable artifact.
    """
    encoded = json.dumps(value, indent=2, allow_nan=False) + '\n'
    with path.open('x', encoding='utf-8') as stream:
        stream.write(encoded)


def reference(path: Path) -> dict[str, str]:
    """Identify exact bytes at an explicit path.

    Args:
        path: Existing artifact.

    Returns:
        Path and SHA-256 reference.
    """
    return {'path': str(path), 'sha256': sha256_file(path)}


def read_market(path: Path) -> pd.DataFrame:
    """Read the existing flattened forex cache format.

    Args:
        path: Verified parquet file.

    Returns:
        Market with ordered pair and field columns.
    """
    market = pd.read_parquet(path)
    market.columns = pd.MultiIndex.from_tuples([tuple(c.rsplit('|', 1)) for c in market.columns])
    return market


def calendar_axes(calendar: dict[str, Any]) -> tuple[pd.DatetimeIndex, pd.DatetimeIndex]:
    """Resolve labels independently from assumed close instants.

    Args:
        calendar: Explicit existing calendar.

    Returns:
        Label and UTC close axes.
    """
    labels = [pd.Timestamp(row['bar_label']) for row in calendar['sessions']]
    if all(label.tzinfo is not None for label in labels):
        labels = [label.tz_convert('UTC') for label in labels]
    return pd.DatetimeIndex(labels), pd.DatetimeIndex([
        utc_instant(row['session_close'], 'input recovery calendar') for row in calendar['sessions']
    ])


def audit_missing_pairs(
    market: pd.DataFrame, calendar: dict[str, Any], start: str, end: str,
) -> list[dict[str, Any]]:
    """Enumerate every absent expected date/pair, including the history prefix.

    Args:
        market: Joined cache, not a substitute for pair-level raw responses.
        calendar: Registered expected sessions.
        start: Inclusive measurement instant.
        end: Exclusive measurement instant.

    Returns:
        Explicit unknowns; a missing joined row never proves provider closure.
    """
    labels, closes = calendar_axes(calendar)
    positions = np.flatnonzero((closes >= utc_instant(start, 'audit')) & (closes < utc_instant(end, 'audit')))
    if len(positions) < 2 or positions[0] < 63:
        raise ValueError('Input audit requires 63 history bars and two measured bars')
    rows: list[dict[str, Any]] = []
    for i in range(int(positions[0]) - 63, int(positions[-1]) + 1):
        for pair in calendar['symbols']:
            fields = [(pair, field) for field in ('Open', 'High', 'Low', 'Close', 'CarryAnnual')]
            absent = labels[i] not in market.index
            missing_fields = [field for field in fields if field not in market.columns]
            invalid = not absent and not missing_fields and not np.isfinite(market.loc[labels[i], fields].to_numpy(dtype=float)).all()
            if absent or missing_fields or invalid:
                rows.append({'bar_label': calendar['sessions'][i]['bar_label'],
                             'session_close': closes[i].isoformat(), 'pair': pair,
                             'role': 'history' if i < positions[0] else 'evaluation',
                             'classification': 'unknown_prejoin_or_provider' if absent else 'pair_or_field_unavailable',
                             'provider_closed': None, 'original_pair_response': None,
                             'join_loss': None, 'available_at': None,
                             'evidence': 'Joined cache only; cannot identify prejoin presence, provider closure, or historical publication.'})
    return rows


def select_contiguous_range(
    market: pd.DataFrame, calendar: dict[str, Any], start: str, end: str,
) -> dict[str, Any]:
    """Choose maximal coverage without looking at returns or relaxing validation.

    Args:
        market: All available joined data, with unchanged values.
        calendar: Full expected calendar, including missing dates.
        start: Original fold start.
        end: Original fold end.

    Returns:
        Longest eligible range and every eligible candidate; earliest breaks ties.

    Raises:
        ValueError: On malformed data or insufficient contiguous history.
    """
    symbols = tuple(calendar['symbols'])
    actual = tuple(dict.fromkeys(pair for pair, _ in market.columns))
    if actual != symbols:
        raise ValueError(f'Input recovery pair order mismatch: {actual}; expected {symbols}')
    if not market.index.is_unique or not market.index.is_monotonic_increasing:
        raise ValueError('Input recovery duplicate or unsorted market labels')
    try:
        validate_ohlcv(market, symbols)
    except Exception as exc:
        raise ValueError(f'Input recovery invalid OHLCV: {exc}') from exc
    carry = market.loc[:, [(pair, 'CarryAnnual') for pair in symbols]].to_numpy(dtype=float)
    if not np.isfinite(carry).all():
        raise ValueError('Input recovery CarryAnnual must be finite')
    labels, closes = calendar_axes(calendar)
    if not labels.is_unique or not labels.is_monotonic_increasing:
        raise ValueError('Input recovery duplicate or unsorted calendar labels')
    if (market.index.tz is None) != (labels.tz is None):
        raise ValueError('Input recovery market/calendar timezone mismatch')
    within = market.index[(market.index >= labels[0]) & (market.index <= labels[-1])]
    if len(within.difference(labels)):
        raise ValueError('Unexpected market bar outside registered calendar')
    begin, stop = utc_instant(start, 'successor'), utc_instant(end, 'successor')
    measured_positions = np.flatnonzero((closes >= begin) & (closes < stop))
    if len(measured_positions) < 2:
        raise ValueError('Successor range requires at least two measured calendar bars')
    present = labels.isin(market.index)
    candidates: list[dict[str, Any]] = []
    run_start = 0
    for boundary in range(len(labels) + 1):
        if boundary < len(labels) and present[boundary]:
            continue
        eligible = [i for i in range(run_start + 63, boundary) if begin <= closes[i] < stop]
        if len(eligible) >= 2:
            first, last = eligible[0], eligible[-1]
            candidate_start = begin if first == measured_positions[0] else closes[first]
            candidate_end = min(stop, closes[last + 1]) if last + 1 < len(closes) else stop
            candidates.append({'measurement_start': candidate_start.isoformat(),
                               'measurement_end': candidate_end.isoformat(),
                               'first_decision': closes[first].isoformat(), 'last_mark': closes[last].isoformat(),
                               'first_history': closes[first - 63].isoformat(), 'measured_bars': len(eligible)})
        run_start = boundary + 1
    if not candidates:
        raise ValueError('No range with 63 contiguous history bars and at least two measured bars')
    chosen = max(candidates, key=lambda row: row['measured_bars'])
    # Reuse the strict evaluator, including all session/holiday/availability checks.
    prepare_period(market, calendar, chosen['measurement_start'], chosen['measurement_end'], symbols, Path('input-recovery'))
    return {**chosen, 'candidates': candidates}


def range_audit(market: pd.DataFrame, raw: dict[str, Any]) -> dict[str, Any]:
    """Expose missing labels in original train/validation/evaluation ranges.

    Args:
        market: Historical joined source.
        raw: Original sealed experiment config.

    Returns:
        Requested ranges, actual bounds and all missing weekday labels.
    """
    dates = market.index.tz_convert('Europe/London').tz_localize(None).normalize()
    result: dict[str, Any] = {}
    for name in ('train_range', 'val_range', 'eval_range'):
        bounds = raw[name]
        expected = pd.date_range(bounds['start'], bounds['end'], freq='B', inclusive='left')
        observed = dates.intersection(expected)
        result[name] = {'requested': bounds, 'audit_boundary': '[start,end)',
                        'first_present': observed[0].isoformat() if len(observed) else None,
                        'last_present': observed[-1].isoformat() if len(observed) else None,
                        'missing_labels': [t.date().isoformat() for t in expected.difference(dates)],
                        'training_lineage_changed': False}
    return result


def build_snapshot(config_path: Path, output: Path, data_output: Path) -> dict[str, Any]:
    """Build a sealed successor dataset without remote requests or inference.

    Args:
        config_path: Explicit preregistered input references.
        output: New public metadata directory.
        data_output: New private derived-cache directory.

    Returns:
        Immutable manifest for one research data/calendar contract.
    """
    config = read_json(config_path)
    if set(config) != {'contract_id', 'campaign', 'attribution', 'budget', 'evidence', 'decision'} or config['contract_id'] != CONTRACT_ID:
        raise ValueError(f'Invalid Issue #39 registration: {config_path}')
    for key in ('campaign', 'attribution', 'budget', 'evidence', 'decision'):
        checked_path(config[key])
    campaign = read_json(Path(config['campaign']['path']))
    validate_campaign(campaign)
    attribution = read_json(Path(config['attribution']['path']))
    validate_attribution_config(attribution)
    evaluation = campaign['evaluation']
    lineage = read_json(checked_path(evaluation['lineage']))
    if any(len(lineage[name]) != 1 for name in ('raw', 'clean', 'carry')):
        raise ValueError('Issue39 requires exactly one registered raw/clean/carry source')
    frames = {name: read_market(checked_path(lineage[name][0])) for name in ('raw', 'clean', 'carry')}
    market = frames['carry']
    if not frames['raw'].index.equals(frames['clean'].index) or not frames['clean'].index.equals(market.index):
        raise ValueError('raw/clean/carry timestamp lineage differs')
    if not frames['raw'].columns.equals(frames['clean'].columns) or not frames['clean'].equals(market.loc[:, frames['clean'].columns]):
        raise ValueError('clean/carry OHLCV lineage differs')
    parent = read_json(checked_path(evaluation['source']))
    if output.exists() or data_output.exists():
        raise FileExistsError(f'Snapshot output already exists: {output} or {data_output}')
    output.mkdir(parents=True)
    data_output.mkdir(parents=True)
    successor = copy.deepcopy(evaluation)
    successor['folds'] = []
    missing: list[dict[str, Any]] = []
    coverage: list[dict[str, Any]] = []
    derived: dict[str, Any] = {}
    for fold in evaluation['folds']:
        name = fold['fold']
        calendar = read_json(checked_path(fold['calendar']))
        rows = audit_missing_pairs(market, calendar, fold['measurement_start'], fold['measurement_end'])
        for row in rows:
            label = pd.Timestamp(row['bar_label'])
            row.update({'fold': name, 'raw_joined_present': label in frames['raw'].index,
                        'clean_present': label in frames['clean'].index, 'carry_present': label in market.index,
                        'cause_bound': 'Absent before spike repair and carry; original pair loss vs provider loss unidentified.'})
        missing.extend(rows)
        selected = select_contiguous_range(market, calendar, fold['measurement_start'], fold['measurement_end'])
        new_calendar = copy.deepcopy(calendar)
        new_calendar['version'] = CONTRACT_ID
        new_calendar['label_rule'] += ' Issue39 successor: unchanged expected sessions; measured range selected by contiguous coverage only.'
        calendar_path = output / f'calendar-{name}.json'
        write_immutable_json(calendar_path, new_calendar)
        new_fold = {**fold, 'measurement_start': selected['measurement_start'], 'measurement_end': selected['measurement_end'], 'calendar': reference(calendar_path)}
        plan = prepare_period(market, new_calendar, new_fold['measurement_start'], new_fold['measurement_end'], tuple(calendar['symbols']), Path(lineage['carry'][0]['path']))
        data_path = data_output / f'fold-{name}.parquet'
        save_ohlcv_parquet(plan.market, '1d', str(plan.market.index[0].date()), str(plan.market.index[-1].date()), data_path)
        derived[name] = reference(data_path)
        identity = parent['fold_sources'][name]
        raw = yaml.safe_load(checked_path({'path': identity['config_path'], 'sha256': identity['config_sha256']}).read_text())
        labels, closes = calendar_axes(calendar)
        expected = closes[(closes >= utc_instant(fold['measurement_start'], name)) & (closes < utc_instant(fold['measurement_end'], name))]
        coverage.append({'fold': name, 'original_start': fold['measurement_start'], 'original_end': fold['measurement_end'],
                         'original_status': 'blocked_input' if rows else 'input_ready',
                         'successor_scope': 'partial_year' if rows else 'full_year', **selected,
                         'excluded_measurement_instants': [t.isoformat() for t in expected if t.isoformat() not in plan.timestamps],
                         'original_ranges': range_audit(market, raw)})
        successor['folds'].append(new_fold)
    write_immutable_json(output / 'evaluation.json', successor)
    write_immutable_json(output / 'missing-pairs.json', {'rows': missing})
    write_immutable_json(output / 'coverage.json', {'folds': coverage})
    changed = frames['raw'].ne(frames['clean'])
    repairs = [{'bar_label': changed.index[row].isoformat(), 'pair': changed.columns[column][0], 'field': changed.columns[column][1],
                'information_limit': 'Retrospective repair uses future neighbors; not causal.'}
               for row, column in np.argwhere(changed.to_numpy())]
    write_immutable_json(output / 'lineage.json', {'sources': lineage, 'changed_ohlc_cells': repairs,
        'retrieved_at': None, 'historical_available_at': None, 'pair_prejoin_presence': None,
        'carry_values': 'Exact existing CarryAnnual bytes/values; FRED response and vintage unavailable; documented lag 60 days does not prove vintage.'})
    manifest = {'contract_id': CONTRACT_ID, 'exit': 'B', 'registration': reference(config_path),
        'command': ['forex-input-recovery', 'build', '--config', str(config_path), '--output', str(output), '--data-output', str(data_output)],
        'training_lineage': evaluation['source'], 'source_data': lineage, 'derived_data': derived,
        'runtime': runtime_identity(), 'original_attempts_complete': False,
        'input_schema': {'symbols': list(market.columns.get_level_values(0).unique()),
            'fields': ['Open', 'High', 'Low', 'Close', 'Volume', 'CarryAnnual'],
            'price_coordinate': 'Existing JPY/COUNTER inverse Yahoo XXXJPY=X quotes; High=1/raw Low, Low=1/raw High.',
            'features': list(FEATURES), 'window_size': 32, 'volatility_window': 32,
            'normalize': False, 'decision_interval': 1},
        'comparability': 'Same frozen models, values, pair order, features, execution and costs. New range/calendar identity. Never pool partial-year successor observations with old annual folds.',
        'successor_decision': 'Adopt these explicit partial ranges and a new statistical scope in #29/#30, or register a separate finite acquisition/training budget. Old 17-fold attempts remain incomplete.',
        'timestamp_contract': {'timezone': 'Europe/London', 'bar_label': 'London midnight',
            'session_close': 'Assumed equal to label; historical same-close research only', 'available_at': None, 'retrieved_at': None,
            'holidays': [], 'history_bars': 63, 'gap': 'Stop before missing bar; restart flat only after 63 contiguous history bars.',
            'financing': 'Unchanged evaluator actual UTC elapsed time, including weekends/DST; no missing-bar bridge.',
            'last_mark': 'No virtual liquidation; no concatenation of independent ranges.'},
        'artifact_sha256': {p.name: sha256_file(p) for p in sorted(output.iterdir()) if p.is_file()}}
    manifest['identity_sha256'] = json_hash(manifest)
    write_immutable_json(output / 'manifest.json', manifest)
    return manifest


def verify_snapshot(output: Path) -> dict[str, Any]:
    """Validate all snapshot bytes before any model or evaluator is used.

    Args:
        output: Snapshot metadata directory.

    Returns:
        Verified manifest.
    """
    manifest = read_json(output / 'manifest.json')
    body = {k: v for k, v in manifest.items() if k != 'identity_sha256'}
    if manifest['contract_id'] != CONTRACT_ID or json_hash(body) != manifest['identity_sha256']:
        raise ValueError(f'Snapshot identity mismatch: {output}')
    checked_path(manifest['registration'])
    registration = read_json(Path(manifest['registration']['path']))
    for key in ('campaign', 'attribution', 'budget', 'evidence', 'decision'):
        checked_path(registration[key])
    for name, digest in manifest['artifact_sha256'].items():
        checked_path({'path': str(output / name), 'sha256': digest})
    for ref in manifest['derived_data'].values():
        checked_path(ref)
    for name in ('raw', 'clean', 'carry'):
        for ref in manifest['source_data'][name]:
            checked_path(ref)
    return manifest


def load_snapshot_sources(output: Path) -> list[FoldSources]:
    """Read a successor through the existing frozen-source preflight.

    Args:
        output: Immutable snapshot directory.

    Returns:
        Sources with identical market values and explicit successor identity.
    """
    manifest = verify_snapshot(output)
    path = output / 'evaluation.json'
    sources = load_campaign_sources(read_json(path), path)
    for source in sources:
        derived = read_market(checked_path(manifest['derived_data'][source.fold]))
        pd.testing.assert_frame_equal(source.plan.market, derived, check_freq=False)
        source.identity['successor_snapshot'] = {'identity_sha256': manifest['identity_sha256'],
                                                'data': manifest['derived_data'][source.fold], 'training_lineage_changed': False}
    return sources


def preflight_snapshot(output: Path, report_path: Path) -> dict[str, Any]:
    """Preflight old and successor cells without fitting or policy inference.

    Args:
        output: Snapshot directory.
        report_path: New machine-readable handoff path.

    Returns:
        Separate original/successor input statuses, never campaign completion.
    """
    manifest = verify_snapshot(output)
    registration = read_json(Path(manifest['registration']['path']))
    campaign_path = Path(registration['campaign']['path'])
    campaign = read_json(campaign_path)
    attribution = read_json(Path(registration['attribution']['path']))
    validate_campaign(campaign)
    validate_attribution_config(attribution)
    for field in ('parent_contract', 'parent_portfolio', 'lock'):
        checked_path(campaign[field])
    for field in ('parent_manifest', 'registration', 'parent_contract', 'lock'):
        checked_path(attribution[field])
    parent_directory = Path(attribution['parent_manifest']['path']).parent
    parent_manifest = read_json(parent_directory / 'manifest.json')
    for name, digest in parent_manifest['generated_artifact_sha256'].items():
        checked_path({'path': str(parent_directory / name), 'sha256': digest})
    cells: list[dict[str, Any]] = []
    for fold in campaign['evaluation']['folds']:
        selected = copy.deepcopy(campaign['evaluation'])
        selected['folds'] = [fold]
        try:
            load_campaign_sources(selected, campaign_path)
            status, error = 'input_ready', None
        except (ValueError, OSError, KeyError, TypeError) as exc:
            status, error = 'blocked_input', f'{type(exc).__name__}: {exc}'
        cells.extend({'issue': 29, 'fold': fold['fold'], 'policy': policy, 'scenario': scenario, 'status': status, 'error': error}
                     for scenario in ('F0', 'F1', 'F2') for policy in POLICIES)
        cells.extend({'issue': 30, 'fold': fold['fold'], 'policy': policy, 'scenario': 'F0', 'status': status, 'error': error}
                     for policy in (*POLICIES, 'train_constant', 'common_projected'))
    sources = load_snapshot_sources(output)
    coverage = read_json(output / 'coverage.json')['folds']
    train_inputs: dict[str, Any] = {}
    for source in sources:
        raw = yaml.safe_load(checked_path({'path': source.identity['config_path'], 'sha256': source.identity['config_sha256']}).read_text())
        train_inputs[source.fold] = preflight_train_input(raw)
    successor = [{'fold': source.fold, 'policies': list(POLICIES), 'status': 'input_ready',
                  'measurement_start': source.plan.measurement_start, 'measurement_end': source.plan.measurement_end,
                  'train_constant': train_inputs[source.fold],
                  'common_projected': 'same_observation_and_action_geometry; requires successor parent F0'} for source in sources]
    report = {'snapshot_identity': manifest['identity_sha256'], 'inference_calls': 0, 'training_calls': 0,
              'attribution_parent_artifacts_verified': len(parent_manifest['generated_artifact_sha256']),
              'original_cells': cells, 'original_resolved_blocked_cells': 0,
              'original_remaining_blocked': {str(issue): sum(c['issue'] == issue and c['status'] == 'blocked_input' for c in cells) for issue in (29, 30)},
              'successor': successor, 'coverage': coverage, 'original_attempts_complete': False,
              'decision_required': manifest['successor_decision'], 'runtime': runtime_identity()}
    write_immutable_json(report_path, report)
    return report


def preflight_train_input(raw: dict[str, Any]) -> dict[str, Any]:
    """Read and reset the existing train-constant input without inference.

    Args:
        raw: Original hash-verified training configuration.

    Returns:
        Input readiness only; no constant weights or model fit are created.
    """
    config = parse_experiment_config(raw)
    start, end = pd.Timestamp(config.train_range.start), pd.Timestamp(config.train_range.end)
    market = read_market(Path(config.env['data']['path']))
    dates = market.index.tz_localize(None) if market.index.tz is not None else market.index
    selected = dates[(dates >= start) & (dates < end)]
    if len(selected) < 65:
        raise ValueError(f'Train range {start}/{end} requires at least 65 bars; got {len(selected)}')
    resolved = resolve_env_raw(config.env, RangeConfig(config.train_range.start, (end - pd.Timedelta(days=1)).date().isoformat()), for_eval=True)
    names = tuple(config.env['features']['selected'])
    env = build_single_env(resolved, tuple(n for n in names if n in FEATURE_REGISTRY),
                           tuple(n for n in names if n in CROSS_FEATURE_REGISTRY),
                           seed=0, decision_interval=1, residual=None, rank_allocation=None,
                           apply_hold_gate=None, gate_evaluation_mode=GateEvaluationMode.LEARNED)
    try:
        observation, info = env.reset(seed=0)
        if np.any(observation['assets']) or info['equity_jpy'] != 1_000_000:
            raise ValueError('Train input preflight requires flat 1000000 JPY reset')
        timestamp = pd.Timestamp(info['timestamp'])
        date = timestamp.tz_localize(None) if timestamp.tzinfo is not None else timestamp
        if date != selected[63] or not np.isfinite(observation['market']).all():
            raise ValueError('Train input preflight observation alignment/values differ')
    finally:
        env.close()
    return {'status': 'input_ready', 'available_bars': len(selected), 'inference_calls': 0,
            'initial_assets_zero': True, 'first_decision': timestamp.isoformat(),
            'scope_limit': 'Original training data and gap semantics retained; successor use needs #30 scope decision.'}


def smoke_snapshot(output: Path, report_path: Path, private_output: Path) -> dict[str, Any]:
    """Replay two transitions of the first successor fold with three frozen policies.

    Args:
        output: Sealed input metadata.
        report_path: New public smoke summary.
        private_output: New directory for small traces and staged prices.

    Returns:
        Geometry/accounting smoke evidence, not investment results.
    """
    sources = load_snapshot_sources(output)
    source = sources[0]
    if private_output.exists() or report_path.exists():
        raise FileExistsError('Smoke outputs must be new')
    private_output.mkdir(parents=True)
    calendar = source.plan.calendar
    stop = (pd.Timestamp(source.plan.timestamps[2]) + pd.Timedelta(seconds=1)).isoformat()
    market = read_market(Path(source.identity['data_identity']['path']))
    plan = prepare_period(market, calendar, source.plan.measurement_start, stop, source.plan.symbols, Path('issue39-smoke'))
    results: list[dict[str, Any]] = []
    for name, policy in (('canonical', canonical_action), ('ridge', source.ridge.action), ('ppo_ens3', source.ppo.action)):
        env, windows = build_period_env(source.env_raw, plan, private_output / f'{name}.parquet')
        try:
            result = evaluate_policy(env, windows, plan, policy)
        finally:
            env.close()
        results.append(result)
        write_immutable_json(private_output / f'{name}.json', result)
    require_comparable(results)
    report = {'snapshot_identity': verify_snapshot(output)['identity_sha256'], 'fold': source.fold,
              'purpose': 'Two transitions only; no full-market adoption statistics.', 'training_calls': 0,
              'policies': [{'policy': name, 'status': result['status'], 'steps': len(result['trace']),
                            'trace': reference(private_output / f'{name}.json')} for name, result in zip(POLICIES, results)],
              'timestamps': plan.timestamps, 'source': source.identity, 'runtime': runtime_identity()}
    write_immutable_json(report_path, report)
    return report


def main() -> int:
    """Run explicit build, preflight, smoke, or verification phases.

    Returns:
        Zero after success; exceptions preserve their origin and stop execution.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    build = commands.add_parser('build')
    build.add_argument('--config', type=Path, required=True)
    build.add_argument('--output', type=Path, required=True)
    build.add_argument('--data-output', type=Path, required=True)
    for name in ('preflight', 'smoke', 'verify'):
        command = commands.add_parser(name)
        command.add_argument('--snapshot', type=Path, required=True)
        if name != 'verify':
            command.add_argument('--report', type=Path, required=True)
        if name == 'smoke':
            command.add_argument('--private-output', type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'build':
        result = build_snapshot(args.config, args.output, args.data_output)
    elif args.command == 'preflight':
        result = preflight_snapshot(args.snapshot, args.report)
    elif args.command == 'smoke':
        result = smoke_snapshot(args.snapshot, args.report, args.private_output)
    else:
        result = verify_snapshot(args.snapshot)
    print(json.dumps({'command': args.command, 'status': 'complete', 'contract_id': CONTRACT_ID}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
