"""Behavior tests for fold-level regime diagnostics."""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime, timedelta

import numpy as np
import pytest

from forex_trainer.regime import (
    CandidateRegime,
    FoldEffect,
    StepRecord,
    aggregate_fold_effects,
    align_agent_rule_records,
    compute_candidate_regime,
    compute_fold_effects,
)


def _regime(level: float) -> CandidateRegime:
    return CandidateRegime(
        realized_market_volatility=level,
        mean_cross_pair_correlation=level + 0.1,
        cross_sectional_return_dispersion=level + 0.2,
        momentum_dispersion=level + 0.3,
        carry_dispersion=level + 0.4,
        trend_reversal_proxy=level + 0.5,
    )


def _record(
    fold: str,
    era: str,
    day: int,
    level: float,
    response: float,
) -> StepRecord:
    return StepRecord(
        fold=fold,
        era=era,
        timestamp=datetime(2020, 1, 1, tzinfo=UTC) + timedelta(days=day),
        regime=_regime(level),
        decision_gross_exposure=0.5 + level,
        decision_turnover=0.1 + level,
        next_net_return=response,
        next_gross_return=response + 0.01,
        next_cost_ratio=0.01 + response / 100.0,
        forward_max_drawdown=abs(response),
    )


def test_candidate_regime_uses_only_the_decision_time_market_window() -> None:
    market_history = np.array(
        [
            [0.01, 0.04, -0.01],
            [0.02, 0.01, 0.03],
            [-0.01, 0.03, 0.02],
            [0.04, -0.02, 0.01],
            [0.03, 0.02, -0.03],
            [9.00, -8.00, 7.00],
        ],
        dtype=np.float64,
    )
    carry = np.array([0.02, -0.01, 0.04], dtype=np.float64)

    actual = compute_candidate_regime(market_history[:5], carry)
    changed_future = market_history.copy()
    changed_future[5] = np.array([-99.0, 88.0, -77.0])
    unchanged = compute_candidate_regime(changed_future[:5], carry)

    market_returns = market_history[:5].mean(axis=1)
    expected_correlation = np.corrcoef(market_history[:5], rowvar=False)
    expected_off_diagonal = expected_correlation[np.triu_indices(3, k=1)].mean()
    assert actual == unchanged
    assert actual.realized_market_volatility == pytest.approx(
        market_returns.std(ddof=1)
    )
    assert actual.mean_cross_pair_correlation == pytest.approx(
        expected_off_diagonal
    )
    assert actual.cross_sectional_return_dispersion == pytest.approx(
        market_history[:5].std(axis=1, ddof=1).mean()
    )
    assert actual.momentum_dispersion == pytest.approx(
        market_history[:5].sum(axis=0).std(ddof=1)
    )
    assert actual.carry_dispersion == pytest.approx(carry.std(ddof=1))
    assert actual.trend_reversal_proxy == pytest.approx(
        np.corrcoef(market_returns[:-1], market_returns[1:])[0, 1]
    )


@pytest.mark.parametrize(
    ("market", "carry", "message"),
    [
        (np.ones((4, 2)), np.array([0.1, 0.2]), "constant market return"),
        (
            np.array([[0.0, 1.0], [1.0, 0.0], [2.0, -1.0], [3.0, -2.0]]),
            np.array([0.1, 0.1]),
            "constant carry",
        ),
        (
            np.array([[0.0, 1.0], [1.0, 0.0], [2.0, -1.0], [np.nan, -2.0]]),
            np.array([0.1, 0.2]),
            "finite",
        ),
    ],
)
def test_candidate_regime_rejects_invalid_inputs_explicitly(
    market: np.ndarray,
    carry: np.ndarray,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        compute_candidate_regime(market, carry)


def test_alignment_requires_exact_fold_timestamp_and_market_regime() -> None:
    agent = [_record("fold-a", "early", 0, 0.1, 0.01)]
    rule = [replace(agent[0], next_net_return=0.02)]

    aligned = align_agent_rule_records(agent, rule)

    assert aligned[0].agent == agent[0]
    assert aligned[0].rule == rule[0]

    with pytest.raises(ValueError, match="fold/timestamp mismatch"):
        align_agent_rule_records(agent, [replace(rule[0], timestamp=rule[0].timestamp + timedelta(days=1))])
    with pytest.raises(ValueError, match="duplicate agent fold/timestamp"):
        align_agent_rule_records([agent[0], agent[0]], rule)
    with pytest.raises(ValueError, match="regime mismatch"):
        align_agent_rule_records(agent, [replace(rule[0], regime=_regime(0.2))])
    with pytest.raises(ValueError, match="forward_max_drawdown must be non-negative"):
        align_agent_rule_records(
            [replace(agent[0], forward_max_drawdown=-0.01)], rule
        )


def test_fold_buckets_precede_high_minus_low_and_rank_effects() -> None:
    records = [
        _record("fold-a", "early", index, float(index + 1), float(index + 1))
        for index in range(4)
    ]

    buckets, effects = compute_fold_effects(
        records,
        "realized_market_volatility",
        2,
    )

    assert [(row.bucket_index, row.observation_count) for row in buckets] == [
        (0, 2),
        (1, 2),
    ]
    assert buckets[0].mean_next_net_return == pytest.approx(1.5)
    assert buckets[1].mean_next_net_return == pytest.approx(3.5)
    assert buckets[1].mean_next_gross_return == pytest.approx(3.51)
    assert buckets[1].mean_next_cost_ratio == pytest.approx(0.045)
    assert buckets[1].mean_forward_max_drawdown == pytest.approx(3.5)
    assert effects[0].high_minus_low_next_net_return == pytest.approx(2.0)
    assert effects[0].rank_association_next_net_return == pytest.approx(1.0)
    assert effects[0].high_minus_low_forward_max_drawdown == pytest.approx(2.0)
    assert effects[0].rank_association_forward_max_drawdown == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("records", "message"),
    [
        (
            [
                _record("fold-a", "early", index, 1.0, float(index))
                for index in range(4)
            ],
            "constant candidate",
        ),
        (
            [
                _record("fold-a", "early", index, float(index), 1.0)
                for index in range(4)
            ],
            "constant response",
        ),
    ],
)
def test_fold_effects_reject_constant_candidate_or_response(
    records: list[StepRecord],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        compute_fold_effects(records, "realized_market_volatility", 2)


def _effect(
    fold: str,
    era: str,
    year: int,
    value: float,
) -> FoldEffect:
    return FoldEffect(
        fold=fold,
        era=era,
        first_timestamp=datetime(year, 1, 1, tzinfo=UTC),
        high_minus_low_next_net_return=value,
        high_minus_low_next_gross_return=value,
        high_minus_low_next_cost_ratio=value,
        high_minus_low_forward_max_drawdown=value,
        rank_association_next_net_return=value,
        rank_association_next_gross_return=value,
        rank_association_next_cost_ratio=value,
        rank_association_forward_max_drawdown=value,
    )


def test_stability_requires_fold_bootstraps_direction_rate_and_every_era() -> None:
    effects = [
        _effect("f1", "early", 2008, 0.10),
        _effect("f2", "early", 2009, 0.20),
        _effect("f3", "recent", 2015, 0.15),
        _effect("f4", "recent", 2016, 0.25),
    ]

    evidence = aggregate_fold_effects(
        effects,
        "next_net_return",
        2_000,
        17,
        2,
        0.75,
        ("early", "recent"),
        2,
    )

    assert evidence.mean_high_minus_low == pytest.approx(0.175)
    assert evidence.fold_direction_rate == pytest.approx(1.0)
    assert evidence.intervals.fold_low > 0.0
    assert evidence.intervals.moving_block_low > 0.0
    assert [(row.era, row.direction) for row in evidence.era_directions] == [
        ("early", "positive"),
        ("recent", "positive"),
    ]
    assert evidence.is_stable
    assert evidence.instability_reasons == ()


def test_isolated_tail_era_is_never_called_stable() -> None:
    evidence = aggregate_fold_effects(
        [
            _effect("f1", "development", 2013, 0.10),
            _effect("f2", "development", 2014, 0.20),
            _effect("tail-2015", "tail", 2015, 0.30),
        ],
        "next_net_return",
        1_000,
        11,
        2,
        0.5,
        ("development", "tail"),
        2,
    )

    assert not evidence.is_stable
    assert evidence.instability_reasons == (
        "era 'tail' has 1 fold; at least 2 are required",
    )


def test_opposite_era_direction_is_not_stable() -> None:
    evidence = aggregate_fold_effects(
        [
            _effect("f1", "early", 2008, 0.40),
            _effect("f2", "early", 2009, 0.30),
            _effect("f3", "recent", 2015, -0.10),
            _effect("f4", "recent", 2016, -0.05),
        ],
        "next_net_return",
        1_000,
        19,
        2,
        0.5,
        ("early", "recent"),
        2,
    )

    assert not evidence.is_stable
    assert "era 'recent' direction is negative, expected positive" in (
        evidence.instability_reasons
    )
