"""Tests for the network registry: shape contract over real observations."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from forex_env import ForexEnv, parse_config

from forex_trainer.config import parse_experiment_config, resolve_env_raw
from forex_trainer.networks import NETWORK_REGISTRY
from helpers import make_experiment_raw


def _make_env() -> ForexEnv:
    """Build a two-pair env so extractors are exercised with N > 1.

    Returns:
        ForexEnv instance.
    """
    raw = make_experiment_raw()
    raw["env"]["environment"]["currency_pairs"] = ["JPY/USD", "JPY/EUR"]
    raw["env"]["transaction_costs"]["spreads"] = {"JPY/USD": 0.0, "JPY/EUR": 0.0}
    raw["env"]["features"]["selected"] = ["log_return", "volatility"]
    config = parse_experiment_config(raw)
    env_raw = resolve_env_raw(config.env, config.train_range, for_eval=False)
    return ForexEnv(parse_config(env_raw))


@pytest.mark.parametrize("name", sorted(NETWORK_REGISTRY))
def test_extractor_maps_observations_to_feature_vectors(name: str) -> None:
    """Every extractor maps a batch of real observations to (B, features_dim)."""
    env = _make_env()
    obs_a, _ = env.reset(seed=0)
    obs_b, _, _, _, _ = env.step(env.action_space.sample())

    extractor = NETWORK_REGISTRY[name](env.observation_space, features_dim=32)
    batch = {
        key: torch.as_tensor(np.stack([obs_a[key], obs_b[key]]))
        for key in ("market", "assets")
    }
    with torch.no_grad():
        output = extractor(batch)
    assert output.shape == (2, 32)
    assert torch.isfinite(output).all()
