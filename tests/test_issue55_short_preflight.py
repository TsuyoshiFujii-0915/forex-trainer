"""Behavioral checks for the bounded short-campaign calendar revision."""
from __future__ import annotations

import copy
import csv
import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from datetime import date, timedelta
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
PARENT = ROOT / 'docs/research/protocols/issue40/registration.json'
COVERAGE = ROOT / 'docs/research/results/issue39/snapshot/coverage.json'
REVISION = ROOT / 'docs/research/protocols/issue40/revisions/v3'


def run_audit(parent: Path, coverage: Path, output: Path) -> subprocess.CompletedProcess[str]:
    """Run the bounded public calendar command.

    Args:
        parent: Frozen v2 registration.
        coverage: Sealed weekday metadata.
        output: Fresh report directory.

    Returns:
        Captured command result.
    """
    return subprocess.run(
        [sys.executable, '-m', 'docs.research.protocols.issue40.short_preflight',
         '--registration', str(parent), '--coverage', str(coverage), '--output', str(output)],
        cwd=ROOT, capture_output=True, text=True,
    )


class ShortCalendarTests(unittest.TestCase):
    """Observe boundaries, complete panels, failure handling, and stage isolation."""

    def test_all_four_candidates_and_all_34_selected_cases_are_saved(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'audit'
            result = run_audit(PARENT, COVERAGE, output)
            self.assertEqual(result.returncode, 0, result.stderr)
            report = json.loads((output / 'report.json').read_text())
            self.assertEqual(report['status'], 'calendar_eligible')
            self.assertEqual(report['candidate_cases'], 136)
            self.assertEqual(report['selected_cases'], 34)
            self.assertEqual(report['blocked_folds'], [])
            self.assertEqual(report['training_calls'], 0)
            self.assertEqual(report['price_values_read'], 0)
            self.assertEqual(report['model_inference_calls'], 0)
            with (output / 'candidates.csv').open() as handle:
                candidates = list(csv.DictReader(handle))
            with (output / 'counts.csv').open() as handle:
                selected = list(csv.DictReader(handle))
            self.assertEqual(len(candidates), 136)
            self.assertEqual(len(selected), 34)
            for fold in range(2009, 2026):
                rows = [r for r in selected if r['fold'] == str(fold)]
                expected_months = 24 if fold == 2009 else 12 if fold == 2018 else 6
                self.assertEqual([int(r['lookback_months']) for r in rows], [expected_months] * 2)
                self.assertEqual([int(r['horizon']) for r in rows], [1, 5])
                for row in rows:
                    self.assertEqual(row['train_end'], row['validation_start'])
                    self.assertEqual(row['cutoff'], f'{fold}-01-01')
                    self.assertGreaterEqual(int(row['train_rows']), 252)
                    self.assertGreaterEqual(int(row['validation_rows']), 60)
                    for prefix in ('train', 'validation'):
                        self.assertEqual(int(row[f'{prefix}_present_rows']),
                                         sum(int(row[f'{prefix}_{key}']) for key in
                                             ('rows', 'history_excluded', 'label_gap_excluded', 'boundary_purged')))
            oldest = [r for r in selected if r['fold'] == '2009']
            self.assertEqual([int(r['train_rows']) for r in oldest], [268, 264])
            self.assertEqual([int(r['validation_rows']) for r in oldest], [189, 181])
            self.assertNotEqual(run_audit(PARENT, COVERAGE, output).returncode, 0)

    def test_both_horizons_share_the_first_eligible_boundary(self) -> None:
        from docs.research.protocols.issue40.short_preflight import select_fold
        labels: list[date] = []
        current = date(2005, 1, 3)
        while current < date(2009, 1, 1):
            if current.weekday() < 5:
                labels.append(current)
            current += timedelta(days=1)
        previous = tuple(d for d in labels if d < date(2008, 7, 1))
        recent = tuple(d for d in labels if d >= date(2008, 7, 1))[-125:]
        candidates, selected = select_fold('2009', date(2003, 6, 1), (previous, recent))
        self.assertEqual([r['validation_rows'] for r in candidates[:2]], [61, 57])
        self.assertEqual([r['lookback_months'] for r in selected], [12, 12])
        self.assertEqual(len(candidates), 8)

    def test_no_candidate_does_not_lower_threshold_or_drop_failed_fold(self) -> None:
        from docs.research.protocols.issue40.short_preflight import build_report
        coverage = copy.deepcopy(json.loads(COVERAGE.read_text()))
        for fold in coverage['folds']:
            for audit in fold['original_ranges'].values():
                current = date.fromisoformat(audit['requested']['start'])
                end = date.fromisoformat(audit['requested']['end'])
                audit['missing_labels'] = []
                while current < end:
                    if current.weekday() < 5:
                        audit['missing_labels'].append(current.isoformat())
                    current += timedelta(days=1)
        candidates, selected, report = build_report(json.loads(PARENT.read_text()), coverage)
        self.assertEqual(len(candidates), 136)
        self.assertEqual(len(selected), 34)
        self.assertEqual(report['blocked_folds'], [str(y) for y in range(2009, 2026)])
        self.assertTrue(all(row['status'] == 'blocked_input' for row in selected))
        self.assertTrue(all(row['lookback_months'] is None for row in selected))
        self.assertEqual(report['status'], 'blocked_input')
        self.assertEqual(report['next_decision']['recommended_change'], 'new_data_identity_with_bounded_input_recovery')
        self.assertEqual(report['training_calls'], 0)

    def test_future_calendar_changes_do_not_change_an_earlier_split(self) -> None:
        from docs.research.protocols.issue40.short_preflight import select_fold
        from docs.research.protocols.issue40.preflight import calendar_blocks
        blocks = calendar_blocks(json.loads(COVERAGE.read_text()))
        before = select_fold('2009', date(2003, 6, 1), blocks)[1]
        past = tuple(tuple(d for d in block if d < date(2009, 1, 1)) for block in blocks)
        after = select_fold('2009', date(2003, 6, 1), tuple(b for b in past if b))[1]
        self.assertEqual([(r['lookback_months'], r['train_rows'], r['validation_rows']) for r in before],
                         [(r['lookback_months'], r['train_rows'], r['validation_rows']) for r in after])

    def test_missing_or_reordered_folds_are_rejected(self) -> None:
        from docs.research.protocols.issue40.short_preflight import build_report
        parent = json.loads(PARENT.read_text())
        parent['folds'].pop()
        with self.assertRaisesRegex(ValueError, '17|fold'):
            build_report(parent, json.loads(COVERAGE.read_text()))

    def test_cli_rejects_changed_coverage_and_parent_before_writing(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            for name, original in [('coverage', COVERAGE), ('parent', PARENT)]:
                changed = base / f'{name}.json'
                changed.write_bytes(original.read_bytes() + b'\n')
                result = run_audit(changed if name == 'parent' else PARENT,
                                   changed if name == 'coverage' else COVERAGE, base / name)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn('SHA-256 mismatch', result.stderr)
                self.assertFalse((base / name).exists())

    def test_short_ready_does_not_wait_for_rl_or_execution(self) -> None:
        from docs.research.protocols.issue40.short_preflight import evaluate_gates
        requirements = json.loads((REVISION / 'registration.json').read_text())['gate_requirements']
        gates = evaluate_gates(requirements)
        self.assertEqual(gates['short_data_ready']['status'], 'ready')
        self.assertEqual(gates['rl_data_ready']['status'], 'blocked')
        self.assertEqual(gates['execution_ready']['status'], 'blocked')
        requirements['short_data_ready']['all_34_calendar_cases'] = False
        self.assertEqual(evaluate_gates(requirements)['short_data_ready']['status'], 'blocked')
        del requirements['rl_data_ready']['past_only_ranges']
        with self.assertRaisesRegex(ValueError, 'rl_data_ready'):
            evaluate_gates(requirements)

    def test_revision_is_reproducible_sealed_and_preserves_economics_and_caps(self) -> None:
        parent = json.loads(PARENT.read_text())
        revision = json.loads((REVISION / 'registration.json').read_text())
        self.assertEqual(revision['risk_and_decision'], parent['risk_and_decision'])
        self.assertEqual(revision['budgets']['ridge_candidate_fits'], 2 * 3 * 17)
        self.assertEqual(revision['budgets']['ridge_refits'], 0)
        self.assertEqual(revision['budgets']['allocation_evaluation_accounts'], (4 + 3) * 17 * 3)
        self.assertEqual(revision['budgets']['allocation_control_reevaluations'], 3 * 17 * 3)
        self.assertEqual(revision['legacy_rl_schedule']['cross_fit_candidate_fits'], 561)
        self.assertEqual(revision['legacy_rl_schedule']['status'], 'historical_not_required_for_short')
        self.assertEqual(revision['active_market_training_budget'], 0)
        self.assertEqual(revision['active_market_evaluation_budget'], 0)
        self.assertFalse(revision['original_annual_attempts_complete'])
        for source in revision['sources']:
            self.assertEqual(hashlib.sha256((ROOT / source['path']).read_bytes()).hexdigest(), source['sha256'])
        expected = [{k: f[k] for k in ('fold', 'scope', 'measurement_start', 'measurement_end',
                                     'first_decision', 'last_mark', 'measured_bars')} for f in parent['folds']]
        actual = [{k: f[k] for k in expected[0]} for f in revision['folds']]
        self.assertEqual(actual, expected)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'audit'
            result = run_audit(PARENT, COVERAGE, output)
            self.assertEqual(result.returncode, 0, result.stderr)
            for name in ('report.json', 'counts.csv', 'candidates.csv'):
                self.assertEqual((output / name).read_bytes(), (REVISION / 'preflight' / name).read_bytes())
        for source in revision['sources']:
            result = subprocess.run(['git', 'show', f"{revision['registration_commit']}:{source['path']}"],
                                    capture_output=True, cwd=ROOT)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(hashlib.sha256(result.stdout).hexdigest(), source['sha256'])
