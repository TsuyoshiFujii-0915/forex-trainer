"""Tests for sparse score-ranked portfolio allocation (ADR-0010)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from helpers import make_experiment_raw

from forex_trainer.config import parse_experiment_config, resolve_env_raw
from forex_trainer.env_factory import GateEvaluationMode, build_single_env

_PAIRS = ["JPY/USD", "JPY/EUR", "JPY/GBP", "JPY/AUD"]


def _rank_env(top_k: int = 1, gross_exposure: float = 2.0) -> Any:
    """Build a deterministic four-pair environment using rank allocation.

    Args:
        top_k: Number of score tails selected on each side.
        gross_exposure: Total absolute portfolio weight.

    Returns:
        Monitor-wrapped evaluation environment.
    """
    raw = make_experiment_raw()
    raw["env"]["environment"]["currency_pairs"] = list(_PAIRS)
    raw["env"]["transaction_costs"]["spreads"] = {pair: 0.0 for pair in _PAIRS}
    raw["run"]["rank_allocation"] = {
        "top_k": top_k,
        "gross_exposure": gross_exposure,
    }
    config = parse_experiment_config(raw)
    resolved = resolve_env_raw(config.env, config.train_range, for_eval=True)
    return build_single_env(
        resolved,
        config.custom_feature_names,
        config.custom_cross_feature_names,
        seed=0,
        decision_interval=1,
        residual=config.run.residual,
        rank_allocation=config.run.rank_allocation,
        apply_hold_gate=config.run.apply_hold_gate,
        gate_evaluation_mode=GateEvaluationMode.LEARNED,
    )


def _weights(info: dict[str, Any]) -> np.ndarray:
    """Read the applied target weights in configured pair order.

    Args:
        info: ForexEnv step information.

    Returns:
        Target weights in configured pair order.
    """
    return np.asarray(info["target_weights"], dtype=np.float64)


def test_action_space_shape_matches_pair_scores() -> None:
    """The policy may emit every finite float32 score for each pair."""
    env = _rank_env()
    assert env.action_space.shape == (len(_PAIRS), 1)
    max_score = np.finfo(np.float32).max
    np.testing.assert_array_equal(env.action_space.low, -max_score)
    np.testing.assert_array_equal(env.action_space.high, max_score)
    env.close()


def test_scores_convert_to_sparse_weights_in_pair_order() -> None:
    """Highest scores are long, lowest short, and middle pairs stay flat."""
    env = _rank_env()
    env.reset(seed=0)
    scores = np.array([[0.25], [-0.75], [0.5], [0.0]], dtype=np.float32)
    _, _, _, _, info = env.step(scores)
    np.testing.assert_allclose(_weights(info), [0.0, -1.0, 1.0, 0.0], atol=1e-6)
    env.close()


def test_ties_use_configured_pair_order_deterministically() -> None:
    """Stable ordering sends the first tied pair short and the last long."""
    env = _rank_env()
    env.reset(seed=0)
    _, _, _, _, info = env.step(np.zeros((len(_PAIRS), 1), dtype=np.float32))
    np.testing.assert_allclose(_weights(info), [-1.0, 0.0, 0.0, 1.0], atol=1e-6)
    env.close()


def test_unbounded_scores_preserve_strict_rank_order() -> None:
    """Scores above one remain distinct instead of collapsing into a tie."""
    env = _rank_env()
    env.reset(seed=0)
    scores = np.array([[5.0], [2.0], [1.0], [-1.0]], dtype=np.float32)
    _, _, _, _, info = env.step(scores)
    np.testing.assert_allclose(_weights(info), [1.0, 0.0, 0.0, -1.0], atol=1e-6)
    env.close()


def test_fixed_gross_is_split_equally_across_both_tails() -> None:
    """Top-k changes sparsity without changing configured total gross."""
    env = _rank_env(top_k=2, gross_exposure=2.0)
    env.reset(seed=0)
    scores = np.array([[-1.0], [-0.5], [0.5], [1.0]], dtype=np.float32)
    _, _, _, _, info = env.step(scores)
    weights = _weights(info)
    np.testing.assert_allclose(weights, [-0.5, -0.5, 0.5, 0.5], atol=1e-6)
    assert np.abs(weights).sum() == pytest.approx(2.0, abs=1e-6)
    assert weights.sum() == pytest.approx(0.0, abs=1e-6)
    env.close()


@pytest.mark.parametrize(
    "scores",
    [
        np.zeros((len(_PAIRS),), dtype=np.float32),
        np.zeros((1, len(_PAIRS)), dtype=np.float32),
        np.full((len(_PAIRS), 1), np.nan, dtype=np.float32),
    ],
)
def test_invalid_score_arrays_fail_explicitly(scores: np.ndarray) -> None:
    """Wrong shapes and non-finite scores are never reshaped or ranked."""
    env = _rank_env()
    env.reset(seed=0)
    with pytest.raises(ValueError, match="rank allocation scores"):
        env.step(scores)
    env.close()
