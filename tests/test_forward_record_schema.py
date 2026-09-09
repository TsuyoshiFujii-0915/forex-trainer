"""Validate the documented forward-record exchange contract."""

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest
from jsonschema import Draft202012Validator, FormatChecker, ValidationError


CONTRACT_DIR = Path(__file__).resolve().parents[1] / "docs/research/protocols/issue20"


def validator() -> Draft202012Validator:
    schema = json.loads((CONTRACT_DIR / "forward-record.schema.json").read_text())
    Draft202012Validator.check_schema(schema)
    return Draft202012Validator(schema, format_checker=FormatChecker())


def records() -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in (CONTRACT_DIR / "forward-record.examples.jsonl").read_text().splitlines()
    ]


def test_documented_decision_outcome_incident_and_correction_are_valid() -> None:
    examples = records()
    assert {record["event_type"] for record in examples} == {
        "decision", "outcome", "incident", "correction"
    }
    for record in examples:
        validator().validate(record)


@pytest.mark.parametrize("field", ["policy", "config", "data", "inputs"])
def test_decision_requires_each_reproducibility_artifact(field: str) -> None:
    decision = deepcopy(records()[0])
    del decision["payload"][field]
    with pytest.raises(ValidationError):
        validator().validate(decision)


@pytest.mark.parametrize("field", ["scores", "proposed_action", "effective_action"])
def test_decision_cannot_silently_omit_a_pair(field: str) -> None:
    decision = deepcopy(records()[0])
    decision["payload"][field].pop()
    with pytest.raises(ValidationError):
        validator().validate(decision)


@pytest.mark.parametrize("timestamp", ["2027-02-30T12:00:00Z", "2027-01-04T12:00:00"])
def test_record_rejects_invalid_or_timezone_free_timestamps(timestamp: str) -> None:
    decision = deepcopy(records()[0])
    decision["recorded_at"] = timestamp
    with pytest.raises(ValidationError):
        validator().validate(decision)


def test_decision_rejects_unidentified_input_snapshot() -> None:
    decision = deepcopy(records()[0])
    decision["payload"]["inputs"]["sha256"] = "unknown"
    with pytest.raises(ValidationError):
        validator().validate(decision)


def test_missing_observed_cost_requires_an_incident_instead_of_a_zero_fill() -> None:
    outcome = deepcopy(records()[1])
    outcome["payload"]["observed_transaction_cost_jpy"] = None
    with pytest.raises(ValidationError):
        validator().validate(outcome)


def test_non_genesis_record_requires_previous_hash() -> None:
    outcome = deepcopy(records()[1])
    outcome["previous_event_sha256"] = None
    with pytest.raises(ValidationError):
        validator().validate(outcome)


def test_correction_requires_original_event_and_reason() -> None:
    for field in ("corrects_event_id", "reason"):
        correction = deepcopy(records()[3])
        del correction["payload"][field]
        with pytest.raises(ValidationError):
            validator().validate(correction)


def test_record_rejects_undocumented_fields() -> None:
    decision = deepcopy(records()[0])
    decision["payload"]["selected_after_viewing_profit"] = True
    with pytest.raises(ValidationError):
        validator().validate(decision)
