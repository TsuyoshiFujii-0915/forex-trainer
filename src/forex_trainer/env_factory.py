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

from .config import ResidualConfig
from .features import CROSS_FEATURE_REGISTRY, FEATURE_REGISTRY


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


class DecisionInterval(gymnasium.Wrapper):
    """Repeats each agent action for k consecutive env bars (ADR-0004).

    One agent decision spans `interval` bars: the same target allocation is
    forwarded to the env at every bar (holding the allocation constant),
    rewards (log equity returns) are summed into one decision reward, and the
    per-bar `info["costs_jpy"]` breakdowns are accumulated so the returned
    info reports the costs of the whole interval. Episode termination or
    truncation cuts the interval short.
    """

    def __init__(self, env: gymnasium.Env, interval: int) -> None:
        """Wrap an env with a fixed decision interval.

        Args:
            env: Environment whose step reward is a log equity return.
            interval: Number of env bars per agent decision (>= 1, validated
                by config parsing).
        """
        super().__init__(env)
        self._interval = interval

    def step(self, action: np.ndarray) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        """Apply one action for the whole interval.

        Args:
            action: Agent action, re-applied at every bar of the interval.

        Returns:
            Tuple of (observation, summed reward, terminated, truncated, info)
            where observation/info reflect the last bar and info["costs_jpy"]
            holds interval-total costs.
        """
        total_reward = 0.0
        total_costs: dict[str, float] = {}
        for _ in range(self._interval):
            observation, reward, terminated, truncated, info = self.env.step(action)
            total_reward += float(reward)
            for key, value in info["costs_jpy"].items():
                total_costs[key] = total_costs.get(key, 0.0) + float(value)
            if terminated or truncated:
                break
        info = dict(info)
        info["costs_jpy"] = total_costs
        return observation, total_reward, terminated, truncated, info


class ResidualAction(gymnasium.Wrapper):
    """Adds the agent action as a residual on a rank-based rule (ADR-0009).

    At every decision the base weights are the cross-sectional reversal rule
    on the configured feature's latest bar: the `top_k` lowest values get
    +base_size (long), the `top_k` highest get -base_size (short). The final
    weight per pair is clip(base + scale * agent_action, -1, 1).
    """

    def __init__(
        self,
        env: gymnasium.Env,
        feature_index: int,
        top_k: int,
        base_size: float,
        scale: float,
    ) -> None:
        """Wrap an env whose agent-facing action is (num_pairs, 1) weights.

        Args:
            env: Environment after PinnedLeverageAction.
            feature_index: Column of obs["market"] the base rule ranks on.
            top_k: Rank-tail slots per side.
            base_size: Base weight magnitude (validated by config parsing).
            scale: Residual half-range (validated by config parsing).
        """
        super().__init__(env)
        self._feature_index = feature_index
        self._top_k = top_k
        self._base_size = base_size
        self._scale = scale
        self._last_market: np.ndarray | None = None

    def reset(self, **kwargs: Any) -> tuple[Any, dict[str, Any]]:
        """Reset and cache the observation for the first base computation."""
        observation, info = self.env.reset(**kwargs)
        self._last_market = observation["market"]
        return observation, info

    def step(self, action: np.ndarray) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        """Combine the base rule with the agent residual and step.

        Args:
            action: Agent residual of shape (num_pairs, 1) in [-1, 1].

        Returns:
            Standard Gymnasium step tuple.

        Raises:
            RuntimeError: If called before reset().
        """
        if self._last_market is None:
            raise RuntimeError("step() called before reset().")
        values = self._last_market[:, -1, self._feature_index]
        order = np.argsort(values)
        base = np.zeros(len(values), dtype=np.float64)
        base[order[: self._top_k]] = self._base_size
        base[order[-self._top_k :]] = -self._base_size
        residual = np.asarray(action, dtype=np.float64).reshape(-1)
        weights = np.clip(base + self._scale * residual, -1.0, 1.0)
        observation, reward, terminated, truncated, info = self.env.step(
            weights.reshape(-1, 1).astype(np.float32)
        )
        self._last_market = observation["market"]
        return observation, reward, terminated, truncated, info


def build_single_env(
    env_raw: Mapping[str, Any],
    feature_names: tuple[str, ...],
    cross_feature_names: tuple[str, ...],
    seed: int,
    decision_interval: int,
    residual: "ResidualConfig | None",
) -> Monitor:
    """Build one Monitor-wrapped ForexEnv.

    Args:
        env_raw: Complete raw forex-env configuration (dates already injected).
        feature_names: Custom feature names to inject from FEATURE_REGISTRY.
        cross_feature_names: Cross-sectional feature names to inject from
            CROSS_FEATURE_REGISTRY (env ADR-0008).
        seed: Seed applied via an initial reset so episode randomness
            (random_start) is reproducible per worker.
        decision_interval: Env bars per agent decision (ADR-0004).
        residual: Residual action scheme (ADR-0009), or None for direct
            weight actions.

    Returns:
        Monitor-wrapped environment (with PinnedLeverageAction applied when
        the env pins leverage to a constant, then ResidualAction when
        configured, and DecisionInterval outside it).
    """
    custom_features = {name: FEATURE_REGISTRY[name] for name in feature_names}
    custom_cross_features = {
        name: CROSS_FEATURE_REGISTRY[name] for name in cross_feature_names
    }
    env: gymnasium.Env = ForexEnv(
        parse_env_config(env_raw),
        custom_features=custom_features,
        custom_cross_features=custom_cross_features,
    )
    box = env.action_space
    if bool(np.all(box.low[:, 1] == box.high[:, 1])):
        env = PinnedLeverageAction(env)
    if residual is not None:
        selected = list(env_raw["features"]["selected"])
        env = ResidualAction(
            env,
            feature_index=selected.index(residual.feature),
            top_k=residual.top_k,
            base_size=residual.base_size,
            scale=residual.scale,
        )
    env = DecisionInterval(env, decision_interval)
    monitored = Monitor(env)
    monitored.reset(seed=seed)
    return monitored


def build_vec_env(
    env_raw: Mapping[str, Any],
    feature_names: tuple[str, ...],
    cross_feature_names: tuple[str, ...],
    n_envs: int,
    vec_env_kind: str,
    base_seed: int,
    decision_interval: int,
    residual: ResidualConfig | None,
) -> VecEnv:
    """Build the vectorized training environment.

    Args:
        env_raw: Complete raw forex-env configuration.
        feature_names: Custom feature names to inject.
        cross_feature_names: Cross-sectional feature names to inject.
        n_envs: Number of parallel environments.
        vec_env_kind: "dummy" (in-process) or "subproc" (one process each).
        base_seed: Worker i is seeded with base_seed + i.
        decision_interval: Env bars per agent decision (ADR-0004).

    Returns:
        SB3 VecEnv instance.

    Raises:
        ValueError: If vec_env_kind is unknown (config validation should have
            rejected it already).
    """
    factories = [
        partial(
            build_single_env,
            dict(env_raw),
            feature_names,
            cross_feature_names,
            base_seed + rank,
            decision_interval,
            residual,
        )
        for rank in range(n_envs)
    ]
    if vec_env_kind == "dummy":
        return DummyVecEnv(factories)
    if vec_env_kind == "subproc":
        return SubprocVecEnv(factories)
    raise ValueError(f"Unknown vec_env kind '{vec_env_kind}'.")
