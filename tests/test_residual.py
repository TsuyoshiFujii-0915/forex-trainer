"""Tests for the residual action scheme (ADR-0009)."""

from __future__ import annotations

import numpy as np
import pytest

from forex_trainer.config import parse_experiment_config, resolve_env_raw
from forex_trainer.env_factory import build_single_env
from helpers import make_experiment_raw

_PAIRS = ["JPY/USD", "JPY/EUR", "JPY/GBP", "JPY/AUD"]


def _residual_env(scale: float):
    raw = make_experiment_raw()
    raw["env"]["environment"]["currency_pairs"] = list(_PAIRS)
    raw["env"]["transaction_costs"]["spreads"] = {p: 0.0 for p in _PAIRS}
    raw["env"]["features"]["normalize"] = False
    raw["env"]["features"]["selected"] = ["log_return", "volatility", "mom24"]
    raw["run"]["residual"] = {
        "feature": "mom24",
        "top_k": 1,
        "base_size": 0.8,
        "scale": scale,
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
    )


def test_zero_residual_reproduces_base_rule() -> None:
    """With a zero agent action the applied weights equal the base rule."""
    env = _residual_env(scale=0.3)
    obs, _ = env.reset(seed=0)
    momentum = obs["market"][:, -1, 2]
    _, _, _, _, info = env.step(np.zeros((len(_PAIRS), 1), dtype=np.float32))
    exposures = np.array([info["exposures_jpy"][p] for p in _PAIRS])
    lowest = int(np.argmin(momentum))
    highest = int(np.argmax(momentum))
    assert exposures[lowest] > 0  # long the biggest loser (reversal)
    assert exposures[highest] < 0
    middle = [i for i in range(len(_PAIRS)) if i not in (lowest, highest)]
    assert np.allclose(exposures[middle], 0.0)
    env.close()


def test_residual_shifts_weights_within_scale() -> None:
    """A full positive residual adds `scale` to the base weight (no clipping).

    scale=0.1 keeps every combined weight inside [-1, 1] so the expected
    shift is uniform across pairs.
    """
    env_zero = _residual_env(scale=0.1)
    env_full = _residual_env(scale=0.1)
    env_zero.reset(seed=0)
    env_full.reset(seed=0)
    _, _, _, _, info_zero = env_zero.step(np.zeros((len(_PAIRS), 1), dtype=np.float32))
    _, _, _, _, info_full = env_full.step(np.ones((len(_PAIRS), 1), dtype=np.float32))
    for pair in _PAIRS:
        shift = info_full["exposures_jpy"][pair] - info_zero["exposures_jpy"][pair]
        assert shift == pytest.approx(0.1 * 1_000_000.0, rel=1e-3)
    env_zero.close()
    env_full.close()
