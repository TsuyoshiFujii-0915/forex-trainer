"""Behavior tests for the historical period audit, without policy replay."""

from pathlib import Path

import pandas as pd
import pytest

from forex_trainer.execution_audit import audit_period, run_audit


def _steps(index: pd.DatetimeIndex, first: int) -> pd.DataFrame:
    """Build three policy traces over the same market transitions.

    Args:
        index: Complete raw market labels.
        first: First decision position.

    Returns:
        Ordered transitions for all three controls.
    """
    return pd.DataFrame([
        {"policy": policy, "decision_timestamp": d.isoformat(),
         "target_timestamp": t.isoformat()}
        for policy in ("supervised", "reversal", "ppo")
        for d, t in zip(index[first:-1], index[first + 1:])
    ])


def test_period_counts_warmup_once_and_uses_elapsed_utc_time() -> None:
    index = pd.DatetimeIndex([
        "2024-03-26", "2024-03-27", "2024-03-28", "2024-03-29",
        "2024-04-01", "2024-04-02",
    ], tz="Europe/London")
    row, excluded = audit_period("2024", index, _steps(index, 3), 2, 2)
    assert row["raw_bars"] == 6
    assert row["decision_count_per_policy"] == 2
    assert row["equity_observations_per_policy"] == 3
    assert row["first_decision"] == "2024-03-29T00:00:00+00:00"
    assert row["first_target"] == "2024-04-01T00:00:00+01:00"
    assert row["elapsed_seconds"] == 95 * 3600
    assert row["elapsed_years"] == 95 / (365.25 * 24)
    assert row["steps_per_year"] == 2 / row["elapsed_years"]
    assert excluded["reason"].tolist() == ["feature_warmup", "feature_warmup", "observation_history"]
    assert excluded["timestamp"].tolist() == [t.isoformat() for t in index[:3]]


def test_inclusive_end_bar_is_reported_as_next_year_target() -> None:
    index = pd.date_range("2023-12-27", periods=6, tz="Europe/London")
    row, _ = audit_period("2023", index, _steps(index, 3), 2, 2)
    assert row["last_decision"] == "2023-12-31T00:00:00+00:00"
    assert row["last_target"] == "2024-01-01T00:00:00+00:00"
    assert row["targets_in_next_year"] == 1


@pytest.mark.parametrize("mutation", ["missing_policy", "missing_step", "duplicate", "target", "order"])
def test_incomplete_or_misaligned_controls_fail_without_intersection(mutation: str) -> None:
    index = pd.date_range("2024-01-01", periods=8, tz="UTC")
    steps = _steps(index, 3)
    if mutation == "missing_policy":
        steps = steps[steps.policy != "ppo"]
    elif mutation == "missing_step":
        steps = steps.drop(1)
    elif mutation == "duplicate":
        steps = pd.concat([steps, steps.iloc[:1]])
    elif mutation == "target":
        steps.loc[0, "target_timestamp"] = index[-1].isoformat()
    else:
        steps = steps.iloc[::-1]
    with pytest.raises(ValueError, match="fold 2024"):
        audit_period("2024", index, steps, 2, 2)


@pytest.mark.parametrize("mutation", ["short", "duplicate", "naive", "unordered"])
def test_unusable_market_history_fails_early(mutation: str) -> None:
    index = pd.date_range("2024-01-01", periods=8, tz="UTC")
    steps = _steps(index, 3)
    if mutation == "short":
        index = index[:4]
    elif mutation == "duplicate":
        index = index.insert(1, index[0])
    elif mutation == "naive":
        index = index.tz_localize(None)
    else:
        index = index[::-1]
    with pytest.raises(ValueError, match="fold 2024"):
        audit_period("2024", index, steps, 2, 2)


def test_wrong_source_seal_produces_no_audit(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "provenance.json").write_text("{}", encoding="utf-8")
    output = tmp_path / "output"
    with pytest.raises(ValueError, match="source provenance hash mismatch"):
        run_audit(source, "0" * 64, tmp_path / "data.parquet", output)
    assert not output.exists()


def test_existing_output_is_never_overwritten(tmp_path: Path) -> None:
    output = tmp_path / "output"
    output.mkdir()
    marker = output / "marker.txt"
    marker.write_text("original", encoding="utf-8")
    with pytest.raises(FileExistsError):
        run_audit(tmp_path / "missing", "0" * 64, tmp_path / "data.parquet", output)
    assert marker.read_text(encoding="utf-8") == "original"
