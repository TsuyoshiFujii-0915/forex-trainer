"""Environment construction for training and evaluation."""

from __future__ import annotations

from collections.abc import Mapping
from enum import Enum
from functools import partial
from typing import Any

import gymnasium
import numpy as np
from forex_env import ForexEnv
from forex_env import parse_config as parse_env_config
from gymnasium import spaces
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecEnv

from .config import ApplyHoldGateConfig, RankAllocationConfig, ResidualConfig
from .features import CROSS_FEATURE_REGISTRY, FEATURE_REGISTRY

_MAX_FINITE_FLOAT32_SCORE = np.finfo(np.float32).max
_ACTION_BOUND_TOLERANCE = 1e-6


class GateEvaluationMode(str, Enum):
    """How an enabled learned gate is applied during an environment walk."""

    LEARNED = "learned"
    FORCED_APPLY = "forced_apply"


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


class TargetWeightInfo(gymnasium.Wrapper):
    """Report the effective target exposure weights applied by ForexEnv."""

    def __init__(self, env: gymnasium.Env, max_leverage: float) -> None:
        """Wrap the full two-column ForexEnv action interface.

        Args:
            env: ForexEnv receiving weight and leverage columns.
            max_leverage: Global gross cap from the resolved environment config.
        """
        super().__init__(env)
        self._max_leverage = max_leverage

    def step(self, action: np.ndarray) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        """Apply a full action and expose its capped target exposure weights.

        Args:
            action: Full ForexEnv action of shape (N, 2).

        Returns:
            Standard Gymnasium step tuple with ``target_weights`` in info.
        """
        array = np.asarray(action, dtype=np.float64)
        observation, reward, terminated, truncated, info = self.env.step(array)
        weights = np.clip(array[:, 0], -1.0, 1.0)
        leverage = np.clip(
            array[:, 1], self.action_space.low[:, 1], self.action_space.high[:, 1]
        )
        gross_request = float(np.sum(np.abs(weights) * leverage))
        if gross_request > self._max_leverage:
            weights = weights * (self._max_leverage / gross_request)
        enriched_info = dict(info)
        enriched_info["target_weights"] = (weights * leverage).astype(float).tolist()
        return observation, reward, terminated, truncated, enriched_info


class RankAllocationAction(gymnasium.ActionWrapper):
    """Convert learned pair scores into a sparse market-neutral portfolio.

    The highest ``top_k`` scores are long and the lowest ``top_k`` are short.
    Stable sorting makes configured pair order the deterministic tie-breaker.
    No observation or trading feature is inspected by this wrapper.
    """

    def __init__(
        self,
        env: gymnasium.Env,
        top_k: int,
        gross_exposure: float,
    ) -> None:
        """Wrap an environment accepting one direct weight per pair.

        Args:
            env: Environment after PinnedLeverageAction.
            top_k: Number of score-ranked pairs selected on each side.
            gross_exposure: Fixed total absolute portfolio weight.
        """
        super().__init__(env)
        num_pairs = env.action_space.shape[0]
        self._top_k = top_k
        self._weight_magnitude = gross_exposure / (2 * top_k)
        self.action_space = spaces.Box(
            low=-_MAX_FINITE_FLOAT32_SCORE,
            high=_MAX_FINITE_FLOAT32_SCORE,
            shape=(num_pairs, 1),
            dtype=np.float32,
        )

    def action(self, action: np.ndarray) -> np.ndarray:
        """Rank scores and return sparse weights in configured pair order.

        Args:
            action: Pair scores with exactly the advertised action-space shape.

        Returns:
            Sparse direct weights with fixed total gross exposure.

        Raises:
            ValueError: If scores have the wrong shape or contain non-finite
                values.
        """
        scores = np.asarray(action, dtype=np.float64)
        if scores.shape != self.action_space.shape:
            raise ValueError(
                "rank allocation scores must have shape "
                f"{self.action_space.shape}, got {scores.shape}."
            )
        if not np.isfinite(scores).all():
            raise ValueError("rank allocation scores must contain only finite values.")
        order = np.argsort(scores[:, 0], kind="stable")
        weights = np.zeros(len(order), dtype=np.float32)
        weights[order[: self._top_k]] = -self._weight_magnitude
        weights[order[-self._top_k :]] = self._weight_magnitude
        return weights.reshape(-1, 1)


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


class ApplyHoldGate(gymnasium.Wrapper):
    """Apply proposed direct weights or preserve the current allocation.

    The agent-facing action appends one gate scalar to the direct pair weights.
    The wrapper sits outside ``DecisionInterval`` so one threshold decision is
    made for the whole agent decision, including action-repeat configurations.
    """

    def __init__(
        self,
        env: gymnasium.Env,
        max_leverage: float,
        evaluation_mode: GateEvaluationMode,
        currency_pairs: tuple[str, ...],
        commission_rate: float,
        spread_rates: tuple[float, ...],
    ) -> None:
        """Wrap a pinned-leverage direct-weight environment.

        Args:
            env: Environment accepting direct pair weights of shape (N, 1).
            max_leverage: Global cap used to calculate proposed target weights.
            evaluation_mode: Learned threshold behavior or forced apply.
            currency_pairs: Configured pair order used by exposure diagnostics.
            commission_rate: Proportional commission applied to JPY turnover.
            spread_rates: Per-pair proportional spread costs in pair order.

        Raises:
            ValueError: If the wrapped action space is not direct weights.
        """
        super().__init__(env)
        box = env.action_space
        if not isinstance(box, spaces.Box) or len(box.shape) != 2 or box.shape[1] != 1:
            raise ValueError(
                "apply/hold gate requires a direct-weight Box action space "
                f"with shape (N, 1), got {box!r}."
            )
        self._num_pairs = box.shape[0]
        if len(currency_pairs) != self._num_pairs:
            raise ValueError(
                "apply/hold gate currency-pair count must match the action space, "
                f"got {len(currency_pairs)} pairs and {self._num_pairs} actions."
            )
        if len(spread_rates) != self._num_pairs:
            raise ValueError(
                "apply/hold gate spread count must match the action space, "
                f"got {len(spread_rates)} spreads and {self._num_pairs} actions."
            )
        self._max_leverage = max_leverage
        self._evaluation_mode = evaluation_mode
        self._currency_pairs = currency_pairs
        self._commission_rate = commission_rate
        self._spread_rates = np.asarray(spread_rates, dtype=np.float64)
        self._current_target_weights = np.zeros(self._num_pairs, dtype=np.float64)
        self._current_equity_jpy = 0.0
        self._current_exposures_jpy = np.zeros(self._num_pairs, dtype=np.float64)
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self._num_pairs + 1, 1),
            dtype=np.float32,
        )

    def reset(self, **kwargs: Any) -> tuple[Any, dict[str, Any]]:
        """Reset the allocation state to the environment's flat portfolio."""
        observation, info = self.env.reset(**kwargs)
        self._current_target_weights = np.zeros(self._num_pairs, dtype=np.float64)
        self._cache_account_state(info)
        return observation, info

    def _cache_account_state(self, info: Mapping[str, Any]) -> None:
        """Cache the account state needed for exact cost counterfactuals.

        Args:
            info: Wrapped environment info at the current decision boundary.

        Raises:
            TypeError: If equity or pair exposures have invalid types.
            ValueError: If equity or pair exposures are absent or non-finite.
        """
        equity = info.get("equity_jpy")
        exposures = info.get("exposures_jpy")
        if isinstance(equity, bool) or not isinstance(equity, (int, float)):
            raise TypeError("apply/hold gate requires numeric equity_jpy in info.")
        if not isinstance(exposures, Mapping):
            raise TypeError("apply/hold gate requires exposures_jpy in info.")
        try:
            exposure_array = np.asarray(
                [exposures[pair] for pair in self._currency_pairs], dtype=np.float64
            )
        except KeyError as exc:
            raise ValueError(
                f"apply/hold gate exposures_jpy is missing pair {exc.args[0]!r}."
            ) from exc
        if (
            not np.isfinite(float(equity))
            or float(equity) <= 0.0
            or not np.isfinite(exposure_array).all()
        ):
            raise ValueError("apply/hold gate received a non-finite account state.")
        self._current_equity_jpy = float(equity)
        self._current_exposures_jpy = exposure_array

    def _proposed_target_weights(self, proposal: np.ndarray) -> np.ndarray:
        """Apply direct-policy clipping and the global gross cap.

        Args:
            proposal: Raw direct pair weights of shape (N, 1).

        Returns:
            Effective target weights before the gate decision.
        """
        weights = np.clip(proposal[:, 0].astype(np.float64), -1.0, 1.0)
        gross_request = float(np.abs(weights).sum())
        if gross_request > self._max_leverage:
            weights = weights * (self._max_leverage / gross_request)
        return weights

    def _immediate_rebalance_cost(self, target_weights: np.ndarray) -> float:
        """Calculate the deterministic spread and commission for one rebalance.

        Args:
            target_weights: Effective direct targets at the decision boundary.

        Returns:
            Immediate JPY spread plus commission cost.
        """
        target_exposures = target_weights * self._current_equity_jpy
        turnover = np.abs(target_exposures - self._current_exposures_jpy)
        return float(
            turnover @ self._spread_rates + turnover.sum() * self._commission_rate
        )

    def step(self, action: np.ndarray) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        """Make one apply/hold decision and report its mechanism diagnostics.

        Args:
            action: Direct pair-weight proposal followed by one gate scalar.

        Returns:
            Standard Gymnasium step tuple with gate diagnostics in ``info``.

        Raises:
            ValueError: If the action shape is wrong or contains non-finite data.
        """
        array = np.asarray(action, dtype=np.float64)
        if array.shape != self.action_space.shape:
            raise ValueError(
                "apply/hold gate action must have shape "
                f"{self.action_space.shape}, got {array.shape}."
            )
        if not np.isfinite(array).all():
            raise ValueError("apply/hold gate action must contain only finite values.")
        if np.any(array < -1.0 - _ACTION_BOUND_TOLERANCE) or np.any(
            array > 1.0 + _ACTION_BOUND_TOLERANCE
        ):
            raise ValueError(
                "apply/hold gate action values must lie in [-1, 1], "
                f"got minimum={float(array.min())} and maximum={float(array.max())}."
            )
        array = np.clip(array, -1.0, 1.0)

        proposal = array[: self._num_pairs]
        gate_signal = float(array[self._num_pairs, 0])
        proposed_target_weights = self._proposed_target_weights(proposal)
        current_before = self._current_target_weights.copy()
        proposed_immediate_cost = self._immediate_rebalance_cost(
            proposed_target_weights
        )
        held_immediate_cost = self._immediate_rebalance_cost(current_before)
        learned_apply = gate_signal >= 0.0
        forced_apply = self._evaluation_mode is GateEvaluationMode.FORCED_APPLY
        applied = learned_apply or forced_apply
        forwarded = proposed_target_weights if applied else current_before
        observation, reward, terminated, truncated, info = self.env.step(
            forwarded.reshape(-1, 1).astype(np.float32)
        )
        if "target_weights" not in info:
            raise ValueError(
                "apply/hold gate requires target_weights in wrapped environment info."
            )
        applied_target_weights = np.asarray(info["target_weights"], dtype=np.float64)
        if applied_target_weights.shape != (self._num_pairs,):
            raise ValueError(
                "apply/hold gate received malformed target_weights with shape "
                f"{applied_target_weights.shape}."
            )
        self._current_target_weights = applied_target_weights.copy()
        if not (terminated or truncated):
            self._cache_account_state(info)
        proposal_distance = float(
            np.abs(proposed_target_weights - current_before).sum()
        )
        enriched_info = dict(info)
        enriched_info.update(
            {
                "gate_signal": gate_signal,
                "gate_learned_apply": bool(learned_apply),
                "gate_applied": bool(applied),
                "gate_forced_apply": bool(forced_apply),
                "proposed_target_weights": proposed_target_weights.tolist(),
                "current_target_weights_before": current_before.tolist(),
                "applied_target_weights": applied_target_weights.tolist(),
                "proposal_distance_from_current": proposal_distance,
                "turnover_avoided_by_hold": 0.0 if applied else proposal_distance,
                "proposed_immediate_transaction_cost_jpy": proposed_immediate_cost,
                "held_immediate_transaction_cost_jpy": held_immediate_cost,
                "immediate_transaction_cost_paid_jpy": (
                    proposed_immediate_cost if applied else held_immediate_cost
                ),
                "immediate_transaction_cost_avoided_by_hold_jpy": (
                    0.0 if applied else proposed_immediate_cost - held_immediate_cost
                ),
            }
        )
        return observation, reward, terminated, truncated, enriched_info


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
    residual: ResidualConfig | None,
    rank_allocation: RankAllocationConfig | None,
    apply_hold_gate: ApplyHoldGateConfig | None,
    gate_evaluation_mode: GateEvaluationMode,
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
        rank_allocation: Sparse score-ranked allocation (ADR-0010), or None
            for direct weight actions.
        apply_hold_gate: Learned zero-threshold direct-policy gate (ADR-0025),
            or None.
        gate_evaluation_mode: Learned gate decisions or forced-apply control.

    Returns:
        Monitor-wrapped environment (with PinnedLeverageAction applied when
        the env pins leverage to a constant, then ResidualAction when
        configured, and DecisionInterval outside it).
    """
    custom_features = {name: FEATURE_REGISTRY[name] for name in feature_names}
    custom_cross_features = {
        name: CROSS_FEATURE_REGISTRY[name] for name in cross_feature_names
    }
    env: gymnasium.Env = TargetWeightInfo(
        ForexEnv(
            parse_env_config(env_raw),
            custom_features=custom_features,
            custom_cross_features=custom_cross_features,
        ),
        max_leverage=float(env_raw["environment"]["max_leverage"]),
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
    if rank_allocation is not None:
        env = RankAllocationAction(
            env,
            top_k=rank_allocation.top_k,
            gross_exposure=rank_allocation.gross_exposure,
        )
    env = DecisionInterval(env, decision_interval)
    if apply_hold_gate is not None:
        currency_pairs = tuple(env_raw["environment"]["currency_pairs"])
        costs = env_raw["transaction_costs"]
        env = ApplyHoldGate(
            env,
            max_leverage=float(env_raw["environment"]["max_leverage"]),
            evaluation_mode=gate_evaluation_mode,
            currency_pairs=currency_pairs,
            commission_rate=float(costs["commission_rate"]),
            spread_rates=tuple(
                float(costs["spreads"][pair]) for pair in currency_pairs
            ),
        )
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
    rank_allocation: RankAllocationConfig | None,
    apply_hold_gate: ApplyHoldGateConfig | None,
    gate_evaluation_mode: GateEvaluationMode,
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
        residual: Residual action scheme (ADR-0009), or None.
        rank_allocation: Sparse score-ranked allocation (ADR-0010), or None.
        apply_hold_gate: Learned zero-threshold direct-policy gate, or None.
        gate_evaluation_mode: Learned gate decisions or forced-apply control.

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
            rank_allocation,
            apply_hold_gate,
            gate_evaluation_mode,
        )
        for rank in range(n_envs)
    ]
    if vec_env_kind == "dummy":
        return DummyVecEnv(factories)
    if vec_env_kind == "subproc":
        return SubprocVecEnv(factories)
    raise ValueError(f"Unknown vec_env kind '{vec_env_kind}'.")
