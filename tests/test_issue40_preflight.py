"""Behavioral checks for the calendar-only Issue 40 registration preflight."""
from __future__ import annotations

import csv
import hashlib
import json
import runpy
import subprocess
import sys
import tempfile
import unittest
from datetime import date, timedelta
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / 'docs/research/protocols/issue40/preflight.py'
REGISTRATION = ROOT / 'docs/research/protocols/issue40/revisions/v1/registration.json'
COVERAGE = ROOT / 'docs/research/results/issue39/snapshot/coverage.json'


def weekdays(start: date, count: int) -> tuple[date, ...]:
    """Build deterministic London weekday labels for a fixture.

    Args:
        start: First calendar date to consider.
        count: Number of weekday labels.

    Returns:
        Consecutive expected weekday labels.
    """
    result: list[date] = []
    while len(result) < count:
        if start.weekday() < 5:
            result.append(start)
        start += timedelta(days=1)
    return tuple(result)


class CalendarPreflightTests(unittest.TestCase):
    """Observe counts and CLI status without fitting or reading market prices."""

    def test_history_and_label_are_required_in_the_same_block(self) -> None:
        api = runpy.run_path(str(SCRIPT))
        block = weekdays(date(2023, 1, 2), 89)
        one = api['audit_range']((block,), block[0], block[-1] + timedelta(days=1), 1)
        five = api['audit_range']((block,), block[0], block[-1] + timedelta(days=1), 5)
        self.assertEqual(one['eligible_rows'], 25)
        self.assertEqual(five['eligible_rows'], 21)
        self.assertEqual(five['history_excluded'], 63)
        self.assertEqual(five['label_gap_excluded'], 5)

    def test_gap_restarts_history_instead_of_compressing_missing_days(self) -> None:
        api = runpy.run_path(str(SCRIPT))
        labels = weekdays(date(2023, 1, 2), 131)
        blocks = (labels[:65], labels[66:])
        result = api['audit_range'](blocks, labels[0], labels[-1] + timedelta(days=1), 1)
        self.assertEqual(result['eligible_rows'], 2)
        self.assertEqual(result['history_excluded'], 126)
        self.assertEqual(result['longest_decisions'], 1)
        self.assertEqual(result['first_decision'], labels[63].isoformat())

    def test_range_boundary_purges_labels_but_allows_past_history(self) -> None:
        api = runpy.run_path(str(SCRIPT))
        block = weekdays(date(2023, 1, 2), 100)
        result = api['audit_range']((block,), block[70], block[90], 5)
        self.assertEqual(result['eligible_rows'], 15)
        self.assertEqual(result['history_excluded'], 0)
        self.assertEqual(result['boundary_purged'], 5)
        later_missing = api['audit_range']((block[:90],), block[70], block[90], 5)
        self.assertEqual(later_missing['eligible_rows'], result['eligible_rows'])

    def test_overlapping_audits_cannot_disagree_about_missing_days(self) -> None:
        api = runpy.run_path(str(SCRIPT))
        original = json.loads(COVERAGE.read_text())
        original['folds'][1]['original_ranges']['train_range']['missing_labels'].remove('2005-09-21')
        with self.assertRaisesRegex(ValueError, 'conflicting.*2005-09-21'):
            api['calendar_blocks'](original)

    def test_cli_reports_every_registered_split_and_the_known_blockers(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'audit'
            result = self.run_preflight(REGISTRATION, COVERAGE, output)
            self.assertEqual(result.returncode, 2, result.stderr)
            report = json.loads((output / 'report.json').read_text())
            self.assertEqual(report['status'], 'blocked_input')
            self.assertEqual(report['training_calls'], 0)
            self.assertEqual(report['price_values_read'], 0)
            with (output / 'counts.csv').open() as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(sum(r['stage'] == 'outer' for r in rows), 34)
            self.assertEqual(sum(r['stage'] == 'cross_fit' for r in rows), 374)
            self.assertEqual(sum(r['stage'] == 'ppo_validation' for r in rows), 17)
            self.assertEqual(sum(r['stage'] == 'rl_train' for r in rows), 17)
            outer = [r for r in rows if r['stage'] == 'outer' and r['fold'] == '2009']
            self.assertEqual([int(r['validation_rows']) for r in outer], [25, 21])
            self.assertTrue(all(r['status'] == 'blocked_input' for r in outer))
            zero = [r for r in rows if r['stage'] == 'cross_fit' and r['cutoff'] == '2008-07-01']
            self.assertEqual(len(zero), 2)
            self.assertTrue(all(int(r['validation_rows']) == 0 for r in zero))
            ppo = next(r for r in rows if r['stage'] == 'ppo_validation' and r['fold'] == '2018')
            self.assertEqual(int(ppo['validation_rows']), 27)
            train = next(r for r in rows if r['stage'] == 'rl_train' and r['fold'] == '2009')
            self.assertEqual(int(train['train_rows']), 0)
            self.assertEqual(train['status'], 'blocked_input')
            duplicate = self.run_preflight(REGISTRATION, COVERAGE, output)
            self.assertNotEqual(duplicate.returncode, 0)
            self.assertIn('already exists', duplicate.stderr)

    def test_cli_rejects_coverage_bytes_that_do_not_match_the_source_seal(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            coverage = Path(directory) / 'coverage.json'
            coverage.write_bytes(COVERAGE.read_bytes() + b'\n')
            result = self.run_preflight(REGISTRATION, coverage, Path(directory) / 'audit')
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('coverage SHA-256 mismatch', result.stderr)
            self.assertFalse((Path(directory) / 'audit').exists())

    def test_saved_preflight_is_reproducible_and_sealed_by_revision(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'audit'
            result = self.run_preflight(REGISTRATION, COVERAGE, output)
            self.assertEqual(result.returncode, 2, result.stderr)
            for name in ['report.json', 'counts.csv']:
                saved = ROOT / 'docs/research/protocols/issue40/preflight' / name
                self.assertEqual((output / name).read_bytes(), saved.read_bytes())
            current = json.loads((ROOT / 'docs/research/protocols/issue40/registration.json').read_text())
            self.assertEqual(current['status'], 'registered_blocked_input')
            self.assertEqual(current['active_market_training_budget'], 0)
            self.assertEqual(current['budgets']['continuous_accounts'], 9)
            self.assertEqual(current['budgets']['annual_reset_accounts'], 153)
            self.assertEqual(current['continuation_policy_sets']['rl_selected'], ['selected_rl', 'same_forecast_cost_allocator', 'canonical'])
            for source in current['sources']:
                self.assertEqual(hashlib.sha256((ROOT / source['path']).read_bytes()).hexdigest(), source['sha256'], source['path'])

    def run_preflight(self, registration: Path, coverage: Path, output: Path) -> subprocess.CompletedProcess[str]:
        """Run the public CLI with explicit inputs and a fresh destination.

        Args:
            registration: Frozen schedule and source identities.
            coverage: Sealed calendar audit bytes.
            output: New report directory.

        Returns:
            Process exit status and captured diagnostics.
        """
        return subprocess.run([sys.executable, str(SCRIPT), '--registration', str(registration), '--coverage', str(coverage), '--output', str(output)], capture_output=True, text=True, cwd=ROOT)
