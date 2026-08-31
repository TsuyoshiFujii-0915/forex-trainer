"""Shared fold-aware statistics for reproducible research reports."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class BootstrapIntervals:
    """Fold-level and moving-block percentile intervals."""

    fold_low: float
    fold_high: float
    moving_block_low: float
    moving_block_high: float


def _validate_bootstrap_controls(
    observation_count: int,
    samples: int,
    seed: int,
    block_length: int,
) -> None:
    """Validate controls shared by every bootstrap calculation.

    Args:
        observation_count: Number of fold-level observations.
        samples: Number of bootstrap draws.
        seed: NumPy generator seed.
        block_length: Circular moving-block length.

    Raises:
        ValueError: If any control is outside its valid range.
    """
    if observation_count < 1:
        raise ValueError("bootstrap requires at least one observation.")
    if isinstance(samples, bool) or not isinstance(samples, int) or samples < 1:
        raise ValueError(f"samples must be a positive integer, got {samples!r}.")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError(f"seed must be a non-negative integer, got {seed!r}.")
    if (
        isinstance(block_length, bool)
        or not isinstance(block_length, int)
        or not 1 <= block_length <= observation_count
    ):
        raise ValueError(
            f"block_length must be in [1, {observation_count}], got {block_length!r}."
        )


def _bootstrap_indices(
    observation_count: int,
    samples: int,
    seed: int,
    block_length: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Draw IID-fold and circular moving-block index matrices.

    Args:
        observation_count: Number of ordered folds.
        samples: Number of bootstrap draws.
        seed: NumPy generator seed.
        block_length: Circular moving-block length.

    Returns:
        IID-fold and moving-block matrices shaped samples by folds.
    """
    _validate_bootstrap_controls(observation_count, samples, seed, block_length)
    generator = np.random.default_rng(seed)
    fold_indices = generator.integers(
        0, observation_count, size=(samples, observation_count)
    )
    block_indices = np.empty((samples, observation_count), dtype=int)
    for sample_index in range(samples):
        selected: list[int] = []
        while len(selected) < observation_count:
            start = int(generator.integers(0, observation_count))
            selected.extend(
                (start + offset) % observation_count for offset in range(block_length)
            )
        block_indices[sample_index] = selected[:observation_count]
    return fold_indices, block_indices


def bootstrap_mean_intervals(
    fold_values: Sequence[float] | np.ndarray,
    samples: int,
    seed: int,
    block_length: int,
) -> BootstrapIntervals:
    """Bootstrap a fold-level arithmetic mean.

    Args:
        fold_values: One finite value per ordered evaluation fold.
        samples: Number of bootstrap draws.
        seed: Deterministic NumPy generator seed.
        block_length: Circular moving-block length in adjacent folds.

    Returns:
        IID-fold and moving-block 95% percentile intervals.

    Raises:
        ValueError: If values or controls are invalid.
    """
    values = np.asarray(fold_values, dtype=np.float64)
    if values.ndim != 1 or len(values) < 1 or not np.isfinite(values).all():
        raise ValueError("fold_values must contain at least one finite value.")
    fold_indices, block_indices = _bootstrap_indices(
        len(values), samples, seed, block_length
    )
    fold_draws = values[fold_indices].mean(axis=1)
    block_draws = values[block_indices].mean(axis=1)
    return BootstrapIntervals(
        fold_low=float(np.quantile(fold_draws, 0.025)),
        fold_high=float(np.quantile(fold_draws, 0.975)),
        moving_block_low=float(np.quantile(block_draws, 0.025)),
        moving_block_high=float(np.quantile(block_draws, 0.975)),
    )


def bootstrap_annualized_fold_returns(
    fold_log_returns: np.ndarray,
    fold_years: np.ndarray,
    samples: int,
    seed: int,
    block_length: int,
) -> BootstrapIntervals:
    """Bootstrap annualized returns using folds as the sampling unit.

    Args:
        fold_log_returns: One seed-averaged cumulative log return per fold.
        fold_years: Evaluation duration in years for each fold.
        samples: Number of bootstrap draws.
        seed: Deterministic NumPy generator seed.
        block_length: Circular moving-block length in adjacent folds.

    Returns:
        Percentile intervals from IID-fold and moving-block resampling.

    Raises:
        ValueError: If arrays or bootstrap controls are invalid.
    """
    returns = np.asarray(fold_log_returns, dtype=np.float64)
    years = np.asarray(fold_years, dtype=np.float64)
    if returns.ndim != 1 or years.ndim != 1 or returns.shape != years.shape:
        raise ValueError(
            "fold_log_returns and fold_years must be equal-length 1D arrays."
        )
    if len(returns) < 1 or not np.isfinite(returns).all():
        raise ValueError("fold_log_returns must contain at least one finite fold.")
    if not np.isfinite(years).all() or bool(np.any(years <= 0.0)):
        raise ValueError("fold_years must contain finite positive durations.")
    fold_indices, block_indices = _bootstrap_indices(
        len(returns), samples, seed, block_length
    )
    fold_draws = np.expm1(
        returns[fold_indices].sum(axis=1) / years[fold_indices].sum(axis=1)
    )
    block_draws = np.expm1(
        returns[block_indices].sum(axis=1) / years[block_indices].sum(axis=1)
    )
    return BootstrapIntervals(
        fold_low=float(np.quantile(fold_draws, 0.025)),
        fold_high=float(np.quantile(fold_draws, 0.975)),
        moving_block_low=float(np.quantile(block_draws, 0.025)),
        moving_block_high=float(np.quantile(block_draws, 0.975)),
    )
