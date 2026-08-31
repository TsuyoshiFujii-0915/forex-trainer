"""Regression tests for regime-effect edge cases found in review."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from forex_trainer.regime import (
    CandidateRegime,
    FoldEffect,
    StepRecord,
    aggregate_fold_effects,
    compute_fold_effects,
)


def _record(index: int, candidate: float, response: float) -> StepRecord:
    """Build one finite trace record for a tied-candidate regression.

    Args:
        index: Timestamp offset.
        candidate: Regime candidate value.
        response: Following return.

    Returns:
        Complete trace record.
    """
    regime = CandidateRegime(candidate, 0.1, 0.2, 0.3, 0.4, 0.5)
    return StepRecord(
        fold="2020",
        era="recent",
        timestamp=datetime(2020, 1, 1, tzinfo=UTC) + timedelta(days=index),
        regime=regime,
        decision_gross_exposure=1.0 + index / 100.0,
        decision_turnover=0.1 + index / 100.0,
        next_net_return=response,
        next_gross_return=response + index / 1_000.0,
        next_cost_ratio=0.01 + index / 10_000.0,
        next_drawdown_change=-response,
    )


def test_equal_candidate_values_are_never_split_across_buckets() -> None:
    """A bucket boundary must not turn timestamp order into a regime effect."""
    records = [
        _record(index, candidate, float(index))
        for index, candidate in enumerate((0.0, 0.0, 0.0, 0.0, 1.0, 2.0))
    ]

    buckets, _ = compute_fold_effects(
        records, "realized_market_volatility", 3
    )

    assert [(row.minimum_candidate_value, row.maximum_candidate_value) for row in buckets] == [
        (0.0, 0.0),
        (1.0, 1.0),
        (2.0, 2.0),
    ]


def _effect(fold: str, year: int, value: float) -> FoldEffect:
    """Build one fold effect whose mean may cancel exactly.

    Args:
        fold: Fold identifier.
        year: Chronological year.
        value: Effect for every response and rank association.

    Returns:
        Complete fold effect.
    """
    return FoldEffect(
        fold=fold,
        era="all",
        first_timestamp=datetime(year, 1, 1, tzinfo=UTC),
        high_minus_low_next_net_return=value,
        high_minus_low_next_gross_return=value,
        high_minus_low_next_cost_ratio=value,
        high_minus_low_next_drawdown_change=value,
        rank_association_next_net_return=value,
        rank_association_next_gross_return=value,
        rank_association_next_cost_ratio=value,
        rank_association_next_drawdown_change=value,
    )


def test_zero_mean_effect_is_reported_as_unstable_not_an_error() -> None:
    """An exact cancellation is valid evidence of no stable relationship."""
    evidence = aggregate_fold_effects(
        (_effect("2019", 2019, 1.0), _effect("2020", 2020, -1.0)),
        "next_net_return",
        500,
        7,
        1,
        0.5,
        ("all",),
        2,
    )

    assert evidence.mean_high_minus_low == pytest.approx(0.0)
    assert not evidence.is_stable
    assert "overall mean direction is neutral" in evidence.instability_reasons
