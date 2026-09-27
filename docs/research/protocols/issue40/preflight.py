"""Audit the frozen Issue 40 learning schedule using calendar metadata only."""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
from collections import Counter
from datetime import date, timedelta
from pathlib import Path
from typing import Any

HISTORY = 63
COVERAGE_SOURCE = 'docs/research/results/issue39/snapshot/coverage.json'


def calendar_blocks(coverage: dict[str, Any]) -> tuple[tuple[date, ...], ...]:
    """Reconstruct observed weekday blocks from overlapping sealed audits.

    Args:
        coverage: Issue 39 coverage document with complete missing-label lists.

    Returns:
        Contiguous observed London weekday blocks, without prices.

    Raises:
        ValueError: Audits conflict or do not cover the expected calendar.
    """
    states: dict[date, bool] = {}
    for fold in coverage['folds']:
        for name, audit in fold['original_ranges'].items():
            start = date.fromisoformat(audit['requested']['start'])
            end = date.fromisoformat(audit['requested']['end'])
            missing = {date.fromisoformat(value) for value in audit['missing_labels']}
            if start >= end or len(missing) != len(audit['missing_labels']):
                raise ValueError(f"Invalid range/missing labels: {fold['fold']} {name}")
            if any(not start <= value < end or value.weekday() >= 5 for value in missing):
                raise ValueError(f"Missing label outside weekday range: {fold['fold']} {name}")
            current = start
            while current < end:
                if current.weekday() < 5:
                    present = current not in missing
                    if current in states and states[current] != present:
                        raise ValueError(f'conflicting coverage for {current}: fold {fold["fold"]} {name}')
                    states[current] = present
                current += timedelta(days=1)
    if not states:
        raise ValueError('Coverage has no audited weekdays')
    blocks: list[tuple[date, ...]] = []
    block: list[date] = []
    current = min(states)
    while current <= max(states):
        if current.weekday() < 5:
            if current not in states:
                raise ValueError(f'Unaudited expected weekday: {current}')
            if states[current]:
                block.append(current)
            elif block:
                blocks.append(tuple(block))
                block = []
        current += timedelta(days=1)
    if block:
        blocks.append(tuple(block))
    return tuple(blocks)


def audit_range(
    blocks: tuple[tuple[date, ...], ...], start: date, end: date, horizon: int,
) -> dict[str, Any]:
    """Count decisions whose prior history and future label stay in one block.

    Args:
        blocks: Observed, contiguous expected weekday sequences.
        start: Inclusive London decision-date boundary.
        end: Exclusive decision and label-end boundary.
        horizon: Positive number of expected business days to the target.

    Returns:
        Eligible and excluded counts plus the earliest longest decision block.

    Raises:
        ValueError: The requested range or horizon is invalid.
    """
    if start >= end or horizon <= 0:
        raise ValueError(f'Invalid preflight range/horizon: {start}, {end}, {horizon}')
    present = history = gap = boundary = eligible = 0
    longest: list[date] = []
    last_mark: date | None = None
    for block in blocks:
        decisions: list[date] = []
        target: date | None = None
        for index, decision in enumerate(block):
            if not start <= decision < end:
                continue
            present += 1
            if index < HISTORY:
                history += 1
            elif index + horizon >= len(block):
                gap += 1
            elif block[index + horizon] >= end:
                boundary += 1
            else:
                decisions.append(decision)
                target = block[index + horizon]
        eligible += len(decisions)
        if len(decisions) > len(longest):
            longest = decisions
            last_mark = target
    return {
        'start': start.isoformat(), 'end': end.isoformat(),
        'present_rows': present, 'history_excluded': history,
        'label_gap_excluded': gap, 'boundary_purged': boundary,
        'eligible_rows': eligible, 'longest_decisions': len(longest),
        'first_decision': longest[0].isoformat() if longest else None,
        'last_mark': last_mark.isoformat() if last_mark is not None else None,
    }


def row(
    stage: str, fold: str, cutoff: str, horizon: int,
    train: dict[str, Any] | None, validation: dict[str, Any] | None,
) -> dict[str, Any]:
    """Describe one required execution unit and its unchanged admission rule.

    Args:
        stage: Outer fit, cross-fit, PPO validation, or RL training availability.
        fold: Registered outer fold identifier.
        cutoff: Date by which labels must be known, or the outer train end.
        horizon: Label horizon; one for a single portfolio transition.
        train: Training counts, absent for a validation-only replay.
        validation: Validation counts, absent for training availability.

    Returns:
        CSV-compatible counts and an explicit eligibility decision.
    """
    train_rows = train['eligible_rows'] if train is not None else 0
    validation_rows = validation['eligible_rows'] if validation is not None else 0
    if stage == 'ppo_validation':
        if validation is None:
            raise ValueError(f'Missing PPO validation audit: {fold}')
        validation_rows = validation['longest_decisions']
        ready = validation_rows >= 60
    elif stage == 'rl_train':
        ready = train_rows >= 1
    elif stage in {'outer', 'cross_fit'}:
        ready = train_rows >= 252 and validation_rows >= 60
    else:
        raise ValueError(f'Unknown preflight stage: {stage}')
    result: dict[str, Any] = {
        'stage': stage, 'fold': fold, 'cutoff': cutoff, 'horizon': horizon,
        'train_rows': train_rows, 'validation_rows': validation_rows,
        'status': 'calendar_eligible' if ready else 'blocked_input',
    }
    for name, audit in [('train', train), ('validation', validation)]:
        for key in ['start', 'end', 'present_rows', 'history_excluded', 'label_gap_excluded',
                    'boundary_purged', 'longest_decisions', 'first_decision', 'last_mark']:
            result[f'{name}_{key}'] = audit[key] if audit is not None else None
    return result


def build_rows(registration: dict[str, Any], coverage: dict[str, Any]) -> list[dict[str, Any]]:
    """Audit every outer fold, both possible cross-fit horizons, and PPO ranges.

    Args:
        registration: Frozen v1 split and update schedule.
        coverage: Sealed Issue 39 weekday coverage.

    Returns:
        All planned cases, including blocked cases and repeated cutoffs by fold.

    Raises:
        ValueError: Fold schedules disagree with the parent calendar audit.
    """
    blocks = calendar_blocks(coverage)
    parent = {item['fold']: item for item in coverage['folds']}
    if [f['fold'] for f in registration['folds']] != [f['fold'] for f in coverage['folds']]:
        raise ValueError('Registration fold order does not match coverage')
    rows: list[dict[str, Any]] = []
    for fold in registration['folds']:
        year = fold['fold']
        train_start, train_end = map(date.fromisoformat, fold['train_range_london'])
        val_start, val_end = map(date.fromisoformat, fold['validation_range_london'])
        for key, registered in [('train_range', fold['train_range_london']),
                                ('val_range', fold['validation_range_london'])]:
            original = parent[year]['original_ranges'][key]['requested']
            if registered != [original['start'], original['end']]:
                raise ValueError(f'Registered {key} differs from source: fold {year}')
        for horizon in [1, 5]:
            rows.append(row('outer', year, val_end.isoformat(), horizon,
                            audit_range(blocks, train_start, train_end, horizon),
                            audit_range(blocks, val_start, val_end, horizon)))
        for value in fold['cross_fit_cutoffs_london']:
            cutoff = date.fromisoformat(value)
            if cutoff.day != 1 or cutoff.month not in {1, 7}:
                raise ValueError(f'Unexpected registered cutoff: {year} {value}')
            inner_start = date(cutoff.year - 1, 7, 1) if cutoff.month == 1 else date(cutoff.year, 1, 1)
            for horizon in [1, 5]:
                rows.append(row('cross_fit', year, value, horizon,
                                audit_range(blocks, date(2005, 7, 19), inner_start, horizon),
                                audit_range(blocks, inner_start, cutoff, horizon)))
        rows.append(row('ppo_validation', year, val_end.isoformat(), 1, None,
                        audit_range(blocks, val_start, val_end, 1)))
        rows.append(row('rl_train', year, train_end.isoformat(), 1,
                        audit_range(blocks, date.fromisoformat(fold['cross_fit_cutoffs_london'][0]), train_end, 1), None))
    return rows


def main() -> int:
    """Write a reproducible calendar report; return 2 for a blocked schedule.

    Returns:
        Zero for calendar eligibility, two for identified input blockers.

    Raises:
        ValueError: Input identity or output ownership checks fail.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--registration', type=Path, required=True)
    parser.add_argument('--coverage', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError(f'Preflight output already exists: {args.output}')
    registration_bytes = args.registration.read_bytes()
    registration = json.loads(registration_bytes)
    pins = [source for source in registration['sources'] if source['path'] == COVERAGE_SOURCE]
    coverage_bytes = args.coverage.read_bytes()
    coverage_hash = hashlib.sha256(coverage_bytes).hexdigest()
    if len(pins) != 1 or pins[0]['sha256'] != coverage_hash:
        raise ValueError(f'coverage SHA-256 mismatch: {args.coverage}')
    rows = build_rows(registration, json.loads(coverage_bytes))
    stream = io.StringIO(newline='')
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator='\n')
    writer.writeheader()
    writer.writerows(rows)
    csv_bytes = stream.getvalue().encode()
    blocked = [value for value in rows if value['status'] == 'blocked_input']
    report = {
        'campaign_id': registration['campaign_id'],
        'status': 'blocked_input' if blocked else 'calendar_eligible',
        'scope': 'Calendar-only upper bounds; feature finiteness and price validity are not certified.',
        'training_calls': 0, 'price_values_read': 0, 'model_inference_calls': 0,
        'history_bars': HISTORY, 'minimum_train_rows': 252, 'minimum_validation_rows': 60,
        'minimum_ppo_validation_decisions': 60, 'rl_train_check': 'At least one transition after first forecast cutoff; not a sufficiency claim.',
        'registration_sha256': hashlib.sha256(registration_bytes).hexdigest(),
        'coverage_sha256': coverage_hash,
        'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'counts_sha256': hashlib.sha256(csv_bytes).hexdigest(),
        'cases_by_stage': dict(sorted(Counter(value['stage'] for value in rows).items())),
        'blocked_by_stage': dict(sorted(Counter(value['stage'] for value in blocked).items())),
        'blocked_outer_folds': sorted({value['fold'] for value in blocked if value['stage'] == 'outer'}),
        'blocked_cross_fit_cutoffs': sorted({value['cutoff'] for value in blocked if value['stage'] == 'cross_fit'}),
        'blocked_ppo_validation_folds': sorted({value['fold'] for value in blocked if value['stage'] == 'ppo_validation'}),
        'blocked_rl_train_folds': sorted({value['fold'] for value in blocked if value['stage'] == 'rl_train'}),
        'decision': 'Return to input remediation; no automatic threshold reduction, gap bridging, fold exclusion, or training activation.',
    }
    args.output.mkdir(parents=True)
    (args.output / 'counts.csv').write_bytes(csv_bytes)
    (args.output / 'report.json').write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n')
    print(f"{report['status']}: {len(blocked)}/{len(rows)} cases blocked; no training or inference")
    return 2 if blocked else 0


if __name__ == '__main__':
    raise SystemExit(main())
