"""Environment construction for training and evaluation."""

from __future__ import annotations

from functools import partial
from typing import Any, Mapping

import gymnasium
import numpy as np
from forex_env import ForexEnv
from forex_env import parse_config as parse_env_config
from gymnasium import spaces
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecEnv

from .features import FEATURE_REGISTRY


class PinnedLeverageAction(gymnasium.ActionWrapper):
    """Removes the pinned leverage column from the agent-facing action space.

    With allow_action_leverage: false, ForexEnv collapses the leverage bounds
    to [1, 1]. That zero-width Box dimension breaks SB3's off-policy action
    rescaling ((action - low) / (high - low) divides by zero), so this wrapper
    exposes only the weight column (N, 1) to the agent and re-attaches the
    pinned leverage value before forwarding the action to the env.
    """

    def __init__(self, env: gymnasium.Env) -> None:
        """Wrap an env whose leverage bounds are collapsed.

        Args:
            env: ForexEnv with action space (N, 2) and low[:,1] == high[:,1].
        """
        super().__init__(env)
        num_pairs = env.action_space.shape[0]
        self._pinned_leverage = np.asarray(env.action_space.low[:, 1], dtype=np.float32)
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(num_pairs, 1), dtype=np.float32
        )

    def action(self, action: np.ndarray) -> np.ndarray:
        """Re-attach the pinned leverage column.

        Args:
            action: Agent action of shape (N, 1) holding target weights.

        Returns:
            Full env action of shape (N, 2).
        """
        weights = np.asarray(action, dtype=np.float32).reshape(-1, 1)
        return np.concatenate([weights, self._pinned_leverage.reshape(-1, 1)], axis=1)


def build_single_env(
    env_raw: Mapping[str, Any], feature_names: tuple[str, ...], seed: int
) -> Monitor:
    """Build one Monitor-wrapped ForexEnv.

    Args:
        env_raw: Complete raw forex-env configuration (dates already injected).
        feature_names: Custom feature names to inject from FEATURE_REGISTRY.
        seed: Seed applied via an initial reset so episode randomness
            (random_start) is reproducible per worker.

    Returns:
        Monitor-wrapped environment (with PinnedLeverageAction applied when
        the env pins leverage to a constant).
    """
    custom_features = {name: FEATURE_REGISTRY[name] for name in feature_names}
    env: gymnasium.Env = ForexEnv(
        parse_env_config(env_raw), custom_features=custom_features
    )
    box = env.action_space
    if bool(np.all(box.low[:, 1] == box.high[:, 1])):
        env = PinnedLeverageAction(env)
    monitored = Monitor(env)
    monitored.reset(seed=seed)
    return monitored


def build_vec_env(
    env_raw: Mapping[str, Any],
    feature_names: tuple[str, ...],
    n_envs: int,
    vec_env_kind: str,
    base_seed: int,
) -> VecEnv:
    """Build the vectorized training environment.

    Args:
        env_raw: Complete raw forex-env configuration.
        feature_names: Custom feature names to inject.
        n_envs: Number of parallel environments.
        vec_env_kind: "dummy" (in-process) or "subproc" (one process each).
        base_seed: Worker i is seeded with base_seed + i.

    Returns:
        SB3 VecEnv instance.

    Raises:
        ValueError: If vec_env_kind is unknown (config validation should have
            rejected it already).
    """
    factories = [
        partial(build_single_env, dict(env_raw), feature_names, base_seed + rank)
        for rank in range(n_envs)
    ]
    if vec_env_kind == "dummy":
        return DummyVecEnv(factories)
    if vec_env_kind == "subproc":
        return SubprocVecEnv(factories)
    raise ValueError(f"Unknown vec_env kind '{vec_env_kind}'.")
