"""Evaluation CLI: forex-eval --run <run_dir>.

Walks the entire eval range once with the deterministic policy and writes
metrics.json and equity_curve.csv into the run directory.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from forex_env.errors import ConfigError, DataError, FeatureError

from .algorithms import ALGO_REGISTRY, resolve_device
from .artifact_provenance import sha256_file
from .config import TrainerConfigError, parse_experiment_config
from .env_factory import build_single_env

_SECONDS_PER_YEAR = 365.25 * 86400.0


def compute_metrics(
    rewards: list[float],
    equities: list[float],
    timestamps: list[str],
    step_costs_jpy: list[float],
    gross_leverages: list[float],
    weight_turnovers: list[float],
    terminated: bool,
) -> dict[str, Any]:
    """Compute evaluation metrics from one full eval-range walk.

    Args:
        rewards: Per-step log returns of equity.
        equities: Equity curve including the initial value (len = steps + 1).
        timestamps: ISO timestamps aligned with equities.
        step_costs_jpy: Per-step total transaction costs, aligned with rewards.
        gross_leverages: Per-step gross leverage readings.
        weight_turnovers: Per-decision sums of absolute target-weight changes.
        terminated: Whether the walk ended in a margin call.

    Returns:
        Metrics dictionary (all values JSON-serializable).
    """
    reward_array = np.asarray(rewards, dtype=float)
    equity_array = np.asarray(equities, dtype=float)
    cost_array = np.asarray(step_costs_jpy, dtype=float)
    # Pre-cost (gross) log return per step: add the step's costs back onto
    # the end-of-step equity before taking the log.
    gross_log_returns = np.log((equity_array[1:] + cost_array) / equity_array[:-1])
    parsed_times = [datetime.fromisoformat(value) for value in timestamps]
    elapsed_years = (
        parsed_times[-1] - parsed_times[0]
    ).total_seconds() / _SECONDS_PER_YEAR
    if elapsed_years <= 0.0:
        raise ValueError("Evaluation timestamps must span a positive duration.")
    steps_per_year = len(reward_array) / elapsed_years
    std = float(reward_array.std(ddof=1)) if len(reward_array) > 1 else 0.0
    # A zero std means the policy never took exposure; Sharpe is reported as
    # 0.0 in that case (documented, not a silent division fallback).
    sharpe = (
        0.0
        if std == 0.0
        else float(reward_array.mean() / std * math.sqrt(steps_per_year))
    )
    running_peak = np.maximum.accumulate(equity_array)
    max_drawdown = float(np.max(1.0 - equity_array / running_peak))
    return {
        "steps": int(len(reward_array)),
        "cumulative_log_return": float(reward_array.sum()),
        "gross_cumulative_log_return": float(gross_log_returns.sum()),
        "annualized_net_return": float(
            math.expm1(float(reward_array.sum()) / elapsed_years)
        ),
        "annualized_gross_return": float(
            math.expm1(float(gross_log_returns.sum()) / elapsed_years)
        ),
        "final_equity_ratio": float(equity_array[-1] / equity_array[0]),
        "sharpe_annualized": sharpe,
        "max_drawdown": max_drawdown,
        "total_cost_ratio": float(cost_array.sum() / equity_array[0]),
        "mean_gross_leverage": float(np.mean(gross_leverages))
        if gross_leverages
        else 0.0,
        "mean_weight_turnover": float(np.mean(weight_turnovers)),
        "total_weight_turnover": float(np.sum(weight_turnovers)),
        "terminated_by_margin_call": bool(terminated),
        "eval_start": timestamps[0],
        "eval_end": timestamps[-1],
    }


def walk_eval_range(
    env: Any,
    predict: Any,
) -> tuple[
    list[float],
    list[float],
    list[str],
    list[float],
    list[float],
    list[float],
    bool,
]:
    """Walk one full eval range with a deterministic policy function.

    Args:
        env: Monitor-wrapped evaluation environment (deterministic full walk).
        predict: Callable mapping (observation, episode_start array) to the
            action to take; owns any recurrent state internally.

    Returns:
        Tuple of (rewards, equities, timestamps, step_costs_jpy,
        gross_leverages, weight_turnovers, terminated) in the argument order
        of compute_metrics.
    """
    observation, info = env.reset(seed=0)
    timestamps = [info["timestamp"]]
    equities = [info["equity_jpy"]]
    rewards: list[float] = []
    gross_leverages: list[float] = []
    step_costs: list[float] = []
    weight_turnovers: list[float] = []
    previous_target_weights: np.ndarray | None = None
    episode_start = np.ones((1,), dtype=bool)
    terminated = False
    while True:
        action = predict(observation, episode_start)
        observation, reward, terminated, truncated, info = env.step(action)
        rewards.append(float(reward))
        equities.append(float(info["equity_jpy"]))
        timestamps.append(str(info["timestamp"]))
        gross_leverages.append(float(info["gross_leverage"]))
        step_costs.append(float(info["costs_jpy"]["total"]))
        if "target_weights" not in info:
            raise ValueError(
                "Evaluation info is missing target_weights; allocation turnover "
                "cannot be measured."
            )
        target_weights = np.asarray(info["target_weights"], dtype=np.float64)
        if previous_target_weights is None:
            previous_target_weights = np.zeros_like(target_weights)
        weight_turnovers.append(
            float(np.abs(target_weights - previous_target_weights).sum())
        )
        previous_target_weights = target_weights
        episode_start = np.array([terminated or truncated], dtype=bool)
        if terminated or truncated:
            break
    return (
        rewards,
        equities,
        timestamps,
        step_costs,
        gross_leverages,
        weight_turnovers,
        terminated,
    )


def run_evaluation(run_dir: Path) -> dict[str, Any]:
    """Evaluate a trained run on its held-out eval range.

    Args:
        run_dir: Run directory produced by forex-train.

    Returns:
        Metrics dictionary; also written to run_dir/metrics.json together
        with run_dir/equity_curve.csv.

    Raises:
        TrainerConfigError: If the run directory lacks required artifacts.
    """
    snapshot_path = run_dir / "config_snapshot.yaml"
    model_path = run_dir / "model_final.zip"
    eval_env_path = run_dir / "env_eval.yaml"
    for required in (snapshot_path, model_path, eval_env_path):
        if not required.is_file():
            raise TrainerConfigError(
                f"Run directory is missing {required.name}: {run_dir}"
            )

    config = parse_experiment_config(
        yaml.safe_load(snapshot_path.read_text(encoding="utf-8"))
    )
    resolved_eval = yaml.safe_load(eval_env_path.read_text(encoding="utf-8"))

    env = build_single_env(
        resolved_eval,
        config.custom_feature_names,
        config.custom_cross_feature_names,
        seed=0,
        decision_interval=config.run.decision_interval,
        residual=config.run.residual,
        rank_allocation=config.run.rank_allocation,
    )
    spec = ALGO_REGISTRY[config.algorithm.name]
    model = spec.algo_class.load(model_path, device=resolve_device(config.run.device))

    state = None

    def predict(observation: dict[str, Any], episode_start: np.ndarray) -> np.ndarray:
        nonlocal state
        action, state = model.predict(
            observation, state=state, episode_start=episode_start, deterministic=True
        )
        return action

    walk = walk_eval_range(env, predict)
    env.close()

    metrics = compute_metrics(*walk)
    timestamps, equities = walk[2], walk[1]
    (run_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    pd.DataFrame({"timestamp": timestamps, "equity_jpy": equities}).to_csv(
        run_dir / "equity_curve.csv", index=False
    )
    evaluation = {
        "manifest_version": 1,
        "model_selection": "validation_best",
        "model_path": model_path.name,
        "model_sha256": sha256_file(model_path),
        "metrics_sha256": sha256_file(run_dir / "metrics.json"),
    }
    (run_dir / "evaluation.json").write_text(
        json.dumps(evaluation, indent=2), encoding="utf-8"
    )
    return metrics


def main(argv: list[str] | None = None) -> int:
    """CLI entry point.

    Args:
        argv: CLI arguments; None lets argparse read sys.argv (testability).

    Returns:
        Process exit code: 0 on success, 1 on errors.
    """
    parser = argparse.ArgumentParser(
        prog="forex-eval", description="Evaluate a trained run on its eval range."
    )
    parser.add_argument("--run", type=str, required=True, help="Run directory path.")
    args = parser.parse_args(argv)
    try:
        metrics = run_evaluation(Path(args.run))
        print(json.dumps(metrics, indent=2))
        return 0
    except (TrainerConfigError, ConfigError, DataError, FeatureError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
