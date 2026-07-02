"""Tests for the decision-interval action-repeat wrapper (ADR-0004)."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from forex_trainer.config import parse_experiment_config, resolve_env_raw
from forex_trainer.env_factory import build_single_env
from helpers import make_experiment_raw

_INTERVAL = 3
_ACTION = np.array([[0.5]], dtype=np.float32)


def _eval_env_raw() -> tuple[dict, tuple[str, ...]]:
    """Build a deterministic full-walk env config over the train range.

    Returns:
        Tuple of (resolved raw env config, custom feature names).
    """
    raw = make_experiment_raw()
    config = parse_experiment_config(raw)
    resolved = resolve_env_raw(config.env, config.train_range, for_eval=True)
    return resolved, config.custom_feature_names


def test_reward_is_log_equity_change_over_interval() -> None:
    """Each wrapped step's reward equals the log equity change over k bars."""
    resolved, feature_names = _eval_env_raw()
    env = build_single_env(resolved, feature_names, seed=0, decision_interval=_INTERVAL)
    _, info = env.reset(seed=0)
    equity_before = info["equity_jpy"]
    for _ in range(20):
        _, reward, terminated, truncated, info = env.step(_ACTION)
        equity_after = info["equity_jpy"]
        assert reward == pytest.approx(
            math.log(equity_after / equity_before), abs=1e-12
        )
        equity_before = equity_after
        if terminated or truncated:
            break
    env.close()


def test_wrapped_walk_matches_manual_action_repeat() -> None:
    """A k-interval env reproduces a 1-interval env driven with repeated actions."""
    resolved, feature_names = _eval_env_raw()
    wrapped = build_single_env(
        resolved, feature_names, seed=0, decision_interval=_INTERVAL
    )
    manual = build_single_env(resolved, feature_names, seed=0, decision_interval=1)
    wrapped.reset(seed=0)
    manual.reset(seed=0)

    for _ in range(15):
        _, wrapped_reward, w_term, w_trunc, wrapped_info = wrapped.step(_ACTION)

        manual_reward = 0.0
        manual_costs = 0.0
        for _ in range(_INTERVAL):
            _, reward, m_term, m_trunc, manual_info = manual.step(_ACTION)
            manual_reward += float(reward)
            manual_costs += float(manual_info["costs_jpy"]["total"])
            if m_term or m_trunc:
                break

        assert wrapped_reward == pytest.approx(manual_reward, abs=1e-12)
        assert wrapped_info["equity_jpy"] == pytest.approx(
            manual_info["equity_jpy"], abs=1e-6
        )
        assert wrapped_info["costs_jpy"]["total"] == pytest.approx(
            manual_costs, abs=1e-9
        )
        assert (w_term or w_trunc) == (m_term or m_trunc)
        if w_term or w_trunc:
            break
    wrapped.close()
    manual.close()


def test_eval_steps_scale_with_decision_interval(tmp_path: Path) -> None:
    """Evaluating with interval k walks the same bars in ceil(bars/k) decisions."""
    from forex_trainer.evaluate import run_evaluation
    from forex_trainer.train import run_training
    from helpers import write_experiment_yaml

    raw_k1 = make_experiment_raw()
    raw_k1["experiment"] = "interval_k1"
    config_path = write_experiment_yaml(tmp_path, raw_k1, name="k1.yaml")
    metrics_k1 = run_evaluation(run_training(config_path, tmp_path / "runs", None))

    raw_k3 = make_experiment_raw()
    raw_k3["experiment"] = "interval_k3"
    raw_k3["run"]["decision_interval"] = _INTERVAL
    config_path = write_experiment_yaml(tmp_path, raw_k3, name="k3.yaml")
    metrics_k3 = run_evaluation(run_training(config_path, tmp_path / "runs", None))

    assert metrics_k3["steps"] == math.ceil(metrics_k1["steps"] / _INTERVAL)
