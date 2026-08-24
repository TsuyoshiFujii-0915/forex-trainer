"""Statistical aggregation tests for data-scaling conclusions."""

from __future__ import annotations

import numpy as np

from forex_trainer.data_scaling import bootstrap_annualized_fold_returns


def test_bootstrap_preserves_fold_and_two_year_block_units() -> None:
    """Uncertainty resamples evaluation years, never individual daily bars."""
    fold_log_returns = np.array([-0.08, -0.03, 0.01, 0.02, 0.04, 0.05, 0.09])
    fold_years = np.ones(7, dtype=float)

    intervals = bootstrap_annualized_fold_returns(
        fold_log_returns,
        fold_years,
        samples=2_000,
        seed=17,
        block_length=2,
    )

    observed = float(np.expm1(fold_log_returns.sum() / fold_years.sum()))
    assert intervals.fold_low < observed < intervals.fold_high
    assert intervals.moving_block_low < observed < intervals.moving_block_high
