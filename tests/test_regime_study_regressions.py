"""Regression tests for regime-study artifact comparisons."""

from __future__ import annotations

import pytest

from forex_trainer.regime_study import compare_legacy_current


def test_sanity_comparison_keeps_mean_and_worst_drawdown_distinct() -> None:
    """Current worst drawdown must not silently resolve to its mean drawdown."""
    legacy = {
        "overall": {
            "mean_annualized_net_return": 0.1,
            "mean_annualized_gross_return": 0.2,
            "mean_max_drawdown": 0.15,
            "worst_max_drawdown": 0.4,
        },
        "eras": {},
        "folds": {},
    }
    current = {
        "overall": {
            "annualized_net_return": 0.1,
            "annualized_gross_return": 0.2,
            "max_drawdown": 0.15,
            "mean_max_drawdown": 0.15,
            "worst_max_drawdown": 0.4,
        },
        "eras": {},
        "folds": {},
    }

    rows = compare_legacy_current(legacy, current)

    worst = next(row for row in rows if row["metric"] == "worst_max_drawdown")
    assert worst["legacy"] == pytest.approx(0.4)
    assert worst["current"] == pytest.approx(0.4)
    assert worst["current_minus_legacy"] == pytest.approx(0.0)
