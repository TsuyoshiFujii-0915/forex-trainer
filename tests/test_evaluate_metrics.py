"""Behavioral tests for evaluation metrics."""

from __future__ import annotations

import math

import pytest

from forex_trainer.evaluate import compute_metrics


def test_sharpe_uses_observed_bars_per_elapsed_year() -> None:
    """Annualize Sharpe from the full observed span, including weekends."""
    rewards = [0.01, -0.005, 0.015, 0.002]
    elapsed_years = 6.0 / 365.25
    mean = sum(rewards) / len(rewards)
    sample_variance = sum((value - mean) ** 2 for value in rewards) / (
        len(rewards) - 1
    )
    expected = mean / math.sqrt(sample_variance) * math.sqrt(
        len(rewards) / elapsed_years
    )
    equities = [1_000_000.0]
    for reward in rewards:
        equities.append(equities[-1] * math.exp(reward))

    metrics = compute_metrics(
        rewards=rewards,
        equities=equities,
        timestamps=[
            "2025-01-02T00:00:00",
            "2025-01-03T00:00:00",
            "2025-01-06T00:00:00",
            "2025-01-07T00:00:00",
            "2025-01-08T00:00:00",
        ],
        step_costs_jpy=[0.0, 0.0, 0.0, 0.0],
        gross_leverages=[1.0, 1.0, 1.0, 1.0],
        weight_turnovers=[0.0, 0.0, 0.0, 0.0],
        terminated=False,
    )

    assert metrics["sharpe_annualized"] == pytest.approx(expected)
