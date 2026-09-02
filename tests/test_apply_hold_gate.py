"""Tests for the learned direct-policy apply/hold gate (ADR-0025)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from helpers import make_experiment_raw

from forex_trainer.config import parse_experiment_config, resolve_env_raw
from forex_trainer.env_factory import GateEvaluationMode, build_single_env

_PAIRS = ["JPY/USD", "JPY/EUR"]


def _gate_env(mode: GateEvaluationMode, decision_interval: int = 1) -> Any:
    """Build a deterministic direct-policy environment with gating enabled.

    Args:
        mode: Learned or forced-apply evaluation behavior.
        decision_interval: Bars spanned by one gate decision.

    Returns:
        Monitor-wrapped evaluation environment.
    """
    raw = make_experiment_raw()
    raw["env"]["environment"]["currency_pairs"] = list(_PAIRS)
    raw["env"]["transaction_costs"]["spreads"] = {pair: 0.0001 for pair in _PAIRS}
    raw["run"]["apply_hold_gate"] = "zero_threshold"
    config = parse_experiment_config(raw)
    resolved = resolve_env_raw(config.env, config.train_range, for_eval=True)
    return build_single_env(
        resolved,
        config.custom_feature_names,
        config.custom_cross_feature_names,
        seed=0,
        decision_interval=decision_interval,
        residual=config.run.residual,
        rank_allocation=config.run.rank_allocation,
        apply_hold_gate=config.run.apply_hold_gate,
        gate_evaluation_mode=mode,
    )


def test_gate_action_space_appends_one_bounded_scalar() -> None:
    """The gate adds exactly one [-1, 1] scalar after direct pair weights."""
    env = _gate_env(GateEvaluationMode.LEARNED)
    assert env.action_space.shape == (len(_PAIRS) + 1, 1)
    np.testing.assert_array_equal(env.action_space.low, -1.0)
    np.testing.assert_array_equal(env.action_space.high, 1.0)
    env.close()


def test_negative_gate_preserves_current_allocation_exactly() -> None:
    """A hold resends the prior effective target without shrinking exposure."""
    env = _gate_env(GateEvaluationMode.LEARNED)
    env.reset(seed=0)
    applied = np.array([[0.6], [-0.4], [0.25]], dtype=np.float32)
    _, _, _, _, apply_info = env.step(applied)
    held = np.array([[-0.8], [0.9], [-0.25]], dtype=np.float32)
    _, _, _, _, hold_info = env.step(held)

    np.testing.assert_allclose(
        hold_info["target_weights"], apply_info["target_weights"], atol=1e-7
    )
    np.testing.assert_allclose(
        hold_info["proposed_target_weights"], [-0.8, 0.9], atol=1e-7
    )
    assert hold_info["gate_signal"] == pytest.approx(-0.25)
    assert hold_info["gate_learned_apply"] is False
    assert hold_info["gate_applied"] is False
    assert hold_info["proposal_distance_from_current"] == pytest.approx(2.7)
    assert hold_info["turnover_avoided_by_hold"] == pytest.approx(2.7)
    np.testing.assert_allclose(
        hold_info["current_target_weights_before"],
        apply_info["target_weights"],
        atol=1e-7,
    )
    np.testing.assert_allclose(
        hold_info["applied_target_weights"], apply_info["target_weights"], atol=1e-7
    )
    assert hold_info["immediate_transaction_cost_paid_jpy"] == pytest.approx(
        hold_info["costs_jpy"]["spread"] + hold_info["costs_jpy"]["commission"],
        abs=1e-9,
    )
    assert hold_info["immediate_transaction_cost_avoided_by_hold_jpy"] == (
        pytest.approx(
            hold_info["proposed_immediate_transaction_cost_jpy"]
            - hold_info["held_immediate_transaction_cost_jpy"]
        )
    )
    assert hold_info["immediate_transaction_cost_avoided_by_hold_jpy"] > 0.0
    env.close()


def test_zero_gate_applies_proposal_once_per_decision_interval() -> None:
    """The zero boundary applies, and diagnostics describe the whole decision."""
    env = _gate_env(GateEvaluationMode.LEARNED, decision_interval=3)
    env.reset(seed=0)
    action = np.array([[0.5], [-0.25], [0.0]], dtype=np.float32)
    _, _, _, _, info = env.step(action)
    np.testing.assert_allclose(info["target_weights"], [0.5, -0.25], atol=1e-7)
    assert info["gate_applied"] is True
    assert info["proposal_distance_from_current"] == pytest.approx(0.75)
    assert info["turnover_avoided_by_hold"] == 0.0
    env.close()


def test_forced_apply_ignores_negative_gate_without_changing_proposal() -> None:
    """The same learned action head can be evaluated with every update applied."""
    env = _gate_env(GateEvaluationMode.FORCED_APPLY)
    env.reset(seed=0)
    action = np.array([[0.7], [-0.2], [-0.9]], dtype=np.float32)
    _, _, _, _, info = env.step(action)
    np.testing.assert_allclose(info["target_weights"], [0.7, -0.2], atol=1e-7)
    assert info["gate_learned_apply"] is False
    assert info["gate_applied"] is True
    assert info["gate_forced_apply"] is True
    env.close()


@pytest.mark.parametrize(
    "action",
    [
        np.zeros((len(_PAIRS), 1), dtype=np.float32),
        np.zeros((1, len(_PAIRS) + 1), dtype=np.float32),
        np.full((len(_PAIRS) + 1, 1), np.nan, dtype=np.float32),
        np.array([[1.1], [0.0], [0.0]], dtype=np.float32),
        np.array([[0.0], [0.0], [-1.1]], dtype=np.float32),
    ],
)
def test_invalid_gate_actions_fail_explicitly(action: np.ndarray) -> None:
    """Malformed proposals and gates are rejected at their origin."""
    env = _gate_env(GateEvaluationMode.LEARNED)
    env.reset(seed=0)
    with pytest.raises(ValueError, match="apply/hold gate action"):
        env.step(action)
    env.close()
