"""Persist complete reports and recover reporting failures without inference."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from forex_trainer.common_basket_study import encode, read
from forex_trainer.common_direction import summarize
from forex_trainer.common_direction_study import recover_report
from test_common_direction import PREDICTION, RULES, observations


def test_complete_report_is_strict_finite_json() -> None:
    report = summarize(observations(), RULES, PREDICTION)
    saved = json.loads(encode(report))
    assert saved == report
    assert saved['risk']['F0']['A1-cost']['met'] is True


def test_report_recovery_rejects_missing_failure_record(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match='reporting incident'):
        recover_report(tmp_path)


def test_finished_campaign_cannot_be_reported_again(tmp_path: Path) -> None:
    (tmp_path / 'manifest.json').write_text('{}')
    with pytest.raises(ValueError, match='already finalized'):
        recover_report(tmp_path)
