"""Extend the sealed calendar preflight with Issue 55's four short splits."""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
from datetime import date
from pathlib import Path
from typing import Any

from docs.research.protocols.issue40.preflight import (
    COVERAGE_SOURCE, HISTORY, audit_range, calendar_blocks, row,
)

PARENT_SHA256 = 'ec90830f4abb811073106cdfb38e214e7609b24ad1e905542f2defb57f8603c4'
CAMPAIGN_ID = 'issue40-common-direction-development-v3'
LOOKBACK_MONTHS = (6, 12, 18, 24)
GATE_REQUIREMENTS = {
    'short_data_ready': {
        'all_34_calendar_cases', 'input_validation_contract',
        'fixed_17_evaluation_intervals', 'budget_comparison_decision_seal',
    },
    'rl_data_ready': {
        'issue43_one_horizon_and_sequential_hypothesis', 'past_only_ranges',
        'continuous_ppo_validation', 'justified_rl_training_amount', 'budget_seal',
    },
    'execution_ready': {
        'issue44_source_permissions_timestamps_finite_collection',
        'issue45_separate_execution_contract',
    },
}


def select_fold(
    fold: str, train_start: date, blocks: tuple[tuple[date, ...], ...],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Choose the shortest common boundary meeting both horizon counts.

    Args:
        fold: Outer evaluation year with a January 1 London cutoff.
        train_start: Unchanged requested expanding-training start.
        blocks: Observed weekday blocks from sealed calendar metadata.

    Returns:
        Eight candidate cases and two selected or explicitly blocked cases.
        Failed cases retain the last candidate's counts for diagnosis only.

    Raises:
        ValueError: A requested training range is invalid.
    """
    cutoff = date(int(fold), 1, 1)
    candidates: list[dict[str, Any]] = []
    selected: list[dict[str, Any]] = []
    for months in LOOKBACK_MONTHS:
        year, month = divmod(cutoff.year * 12 - months, 12)
        boundary = date(year, month + 1, 1)
        pair = []
        for horizon in (1, 5):
            case = row('outer', fold, cutoff.isoformat(), horizon,
                       audit_range(blocks, train_start, boundary, horizon),
                       audit_range(blocks, boundary, cutoff, horizon))
            case['lookback_months'] = months
            case['boundary_role'] = 'candidate'
            pair.append(case)
        candidates.extend(pair)
        if not selected and all(case['status'] == 'calendar_eligible' for case in pair):
            selected = [dict(case, boundary_role='selected') for case in pair]
    if not selected:
        selected = [dict(case, lookback_months=None, status='blocked_input',
                         boundary_role='last_candidate_diagnostic_only') for case in candidates[-2:]]
    return candidates, selected


def build_report(
    registration: dict[str, Any], coverage: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Inspect the fixed 17 folds using dates only, without activating work.

    Args:
        registration: Frozen v2 parent manifest.
        coverage: Sealed Issue 39 calendar coverage metadata.

    Returns:
        Every candidate, all 34 final decisions, and the decision report.

    Raises:
        ValueError: The parent fold matrix or original boundaries disagree.
    """
    expected = [str(year) for year in range(2009, 2026)]
    if [f['fold'] for f in registration['folds']] != expected or [f['fold'] for f in coverage['folds']] != expected:
        raise ValueError('Short preflight requires the ordered 17 folds 2009 through 2025')
    blocks = calendar_blocks(coverage)
    candidates: list[dict[str, Any]] = []
    selected: list[dict[str, Any]] = []
    for fold, audit in zip(registration['folds'], coverage['folds'], strict=True):
        year = int(fold['fold'])
        for original, registered in [('train_range', 'train_range_london'), ('val_range', 'validation_range_london')]:
            requested = audit['original_ranges'][original]['requested']
            if fold[registered] != [requested['start'], requested['end']]:
                raise ValueError(f"Parent {registered} disagrees with coverage: fold {year}")
        if fold['validation_range_london'] != [f'{year - 1}-07-01', f'{year}-01-01']:
            raise ValueError(f'Parent validation does not end at the London January cutoff: {year}')
        if fold['train_range_london'] != ['2003-06-01', f'{year - 1}-07-01']:
            raise ValueError(f'Parent expanding training boundary differs: {year}')
        examined, chosen = select_fold(fold['fold'], date(2003, 6, 1), blocks)
        candidates.extend(examined)
        selected.extend(chosen)
    blocked = sorted({case['fold'] for case in selected if case['status'] == 'blocked_input'})
    report = {
        'campaign_id': CAMPAIGN_ID,
        'status': 'blocked_input' if blocked else 'calendar_eligible',
        'scope': 'Calendar eligibility only; prices, features, targets, and training success are not certified.',
        'training_calls': 0, 'price_values_read': 0, 'model_inference_calls': 0,
        'market_evaluation_accounts': 0, 'external_data_requests': 0,
        'baseline': {'cases': 442, 'blocked_cases': 97,
                     'report': 'docs/research/protocols/issue40/preflight/report.json'},
        'history_bars': HISTORY, 'minimum_train_rows': 252, 'minimum_validation_rows': 60,
        'lookback_months': list(LOOKBACK_MONTHS), 'candidate_cases': len(candidates),
        'selected_cases': len(selected), 'blocked_folds': blocked,
        'selection': 'Shortest lookback satisfying both horizons; expanding train ends at validation start.',
        'fold_splits': [
            {'fold': case['fold'], 'lookback_months': case['lookback_months'],
             'train_range_london': [case['train_start'], case['train_end']],
             'validation_range_london': [case['validation_start'], case['validation_end']],
             'boundary_role': case['boundary_role']}
            for case in selected if case['horizon'] == 1
        ],
        'next_decision': {
            'recommended_change': 'new_data_identity_with_bounded_input_recovery',
            'status': 'requires_separate_review' if blocked else 'not_needed',
            'invariant_to_change': 'Existing data bytes only; retain 17 intervals, 63 history bars, and 252/60 minima.',
            'proposal': 'Register one replacement input snapshot with a finite acquisition budget before any fetch or fit; do not add lookbacks or relax counts.',
            'blocked_folds': blocked,
        },
        'decision': 'Register short split; RL and execution require separate gates.' if not blocked
                    else 'Stop training; request the single bounded input-identity decision, without further rule search.',
    }
    return candidates, selected, report


def evaluate_gates(requirements: dict[str, dict[str, bool]]) -> dict[str, dict[str, Any]]:
    """Evaluate the three independent campaign contracts without execution.

    Args:
        requirements: Explicit evidence booleans for every registered condition.

    Returns:
        Ready or blocked per gate, with all missing conditions named.

    Raises:
        ValueError: A condition is missing, unknown, or not a boolean.
    """
    if set(requirements) != set(GATE_REQUIREMENTS):
        raise ValueError('Expected short_data_ready, rl_data_ready, and execution_ready requirements')
    result: dict[str, dict[str, Any]] = {}
    for gate, keys in GATE_REQUIREMENTS.items():
        evidence = requirements[gate]
        if set(evidence) != keys or any(type(value) is not bool for value in evidence.values()):
            raise ValueError(f'Incomplete or invalid gate evidence: {gate}')
        missing = sorted(key for key, ready in evidence.items() if not ready)
        result[gate] = {'status': 'blocked' if missing else 'ready', 'missing': missing}
    return result


def csv_bytes(rows: list[dict[str, Any]]) -> bytes:
    """Serialize a complete ordered case table.

    Args:
        rows: Nonempty calendar cases with the same column layout.

    Returns:
        Deterministic UTF-8 CSV bytes.
    """
    stream = io.StringIO(newline='')
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator='\n')
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode()


def main() -> int:
    """Save the finite audit in a fresh directory after verifying input seals.

    Returns:
        Zero for calendar eligibility or two for an infeasible short split.

    Raises:
        ValueError: Input identity or output ownership checks fail.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--registration', type=Path, required=True)
    parser.add_argument('--coverage', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError(f'Short preflight output already exists: {args.output}')
    parent_bytes = args.registration.read_bytes()
    if hashlib.sha256(parent_bytes).hexdigest() != PARENT_SHA256:
        raise ValueError(f'Parent v2 registration SHA-256 mismatch: {args.registration}')
    registration = json.loads(parent_bytes)
    coverage_bytes = args.coverage.read_bytes()
    coverage_hash = hashlib.sha256(coverage_bytes).hexdigest()
    pins = [s for s in registration['sources'] if s['path'] == COVERAGE_SOURCE]
    if len(pins) != 1 or pins[0]['sha256'] != coverage_hash:
        raise ValueError(f'Coverage SHA-256 mismatch: {args.coverage}')
    legacy_path = Path(__file__).with_name('preflight.py')
    legacy_key = 'docs/research/protocols/issue40/preflight.py'
    pins = [s for s in registration['sources'] if s['path'] == legacy_key]
    legacy_hash = hashlib.sha256(legacy_path.read_bytes()).hexdigest()
    if len(pins) != 1 or pins[0]['sha256'] != legacy_hash:
        raise ValueError(f'Legacy preflight SHA-256 mismatch: {legacy_path}')
    candidates, selected, report = build_report(registration, json.loads(coverage_bytes))
    artifacts = {'candidates.csv': csv_bytes(candidates), 'counts.csv': csv_bytes(selected)}
    report.update({
        'parent_registration_sha256': PARENT_SHA256, 'coverage_sha256': coverage_hash,
        'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'legacy_script_sha256': legacy_hash,
        'artifacts': {name: hashlib.sha256(content).hexdigest() for name, content in artifacts.items()},
    })
    args.output.mkdir(parents=True)
    for name, content in artifacts.items():
        (args.output / name).write_bytes(content)
    (args.output / 'report.json').write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n')
    print(f"{report['status']}: {len(report['blocked_folds'])}/17 folds blocked; 136 candidates, 34 cases; no fit or inference")
    return 2 if report['blocked_folds'] else 0


if __name__ == '__main__':
    raise SystemExit(main())
