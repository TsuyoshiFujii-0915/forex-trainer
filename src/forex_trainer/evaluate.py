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
from .artifact_provenance import (
    evaluation_runtime_provenance,
    require_current_training_provenance,
    sha256_file,
)
from .config import (
    TrainerConfigError,
    parse_experiment_config,
    require_matching_resolved_eval_env,
)
from .env_factory import GateEvaluationMode, build_single_env

_SECONDS_PER_YEAR = 365.25 * 86400.0

EvaluationWalk = tuple[
    list[float],
    list[float],
    list[str],
    list[float],
    list[float],
    list[float],
    bool,
]

_GATE_TRACE_FIELDS: tuple[str, ...] = (
    "decision_timestamp",
    "gate_signal",
    "gate_learned_apply",
    "gate_applied",
    "gate_forced_apply",
    "current_target_weights_before",
    "proposed_target_weights",
    "applied_target_weights",
    "proposal_distance_from_current",
    "turnover_avoided_by_hold",
    "proposed_immediate_transaction_cost_jpy",
    "held_immediate_transaction_cost_jpy",
    "immediate_transaction_cost_paid_jpy",
    "immediate_transaction_cost_avoided_by_hold_jpy",
    "realized_weight_turnover",
    "realized_total_cost_jpy",
    "gross_leverage",
)


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
        "steps": len(reward_array),
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


def _walk_eval_range_core(
    env: Any,
    predict: Any,
) -> tuple[EvaluationWalk, list[dict[str, Any]]]:
    """Walk one full eval range and retain any explicit gate diagnostics.

    Args:
        env: Monitor-wrapped evaluation environment (deterministic full walk).
        predict: Callable mapping (observation, episode_start array) to the
            action to take; owns any recurrent state internally.

    Returns:
        Evaluation metrics inputs and gate trace rows. The trace is empty for
        an environment without an apply/hold gate.
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
    gate_trace: list[dict[str, Any]] = []
    while True:
        decision_timestamp = str(info["timestamp"])
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
        realized_turnover = float(
            np.abs(target_weights - previous_target_weights).sum()
        )
        weight_turnovers.append(realized_turnover)
        previous_target_weights = target_weights
        if "gate_signal" in info:
            row: dict[str, Any] = {"decision_timestamp": decision_timestamp}
            for field in _GATE_TRACE_FIELDS[1:-3]:
                if field not in info:
                    raise ValueError(
                        f"Gated evaluation info is missing required field {field}."
                    )
                row[field] = info[field]
            row["realized_weight_turnover"] = realized_turnover
            row["realized_total_cost_jpy"] = float(info["costs_jpy"]["total"])
            row["gross_leverage"] = float(info["gross_leverage"])
            gate_trace.append(row)
        episode_start = np.array([terminated or truncated], dtype=bool)
        if terminated or truncated:
            break
    return (
        (
            rewards,
            equities,
            timestamps,
            step_costs,
            gross_leverages,
            weight_turnovers,
            terminated,
        ),
        gate_trace,
    )


def walk_eval_range(env: Any, predict: Any) -> EvaluationWalk:
    """Walk one full eval range using the established metrics contract.

    Args:
        env: Monitor-wrapped deterministic evaluation environment.
        predict: Deterministic policy callable.

    Returns:
        Metrics inputs in the argument order of ``compute_metrics``.
    """
    walk, _ = _walk_eval_range_core(env, predict)
    return walk


def walk_gated_eval_range(
    env: Any, predict: Any
) -> tuple[EvaluationWalk, list[dict[str, Any]]]:
    """Walk an enabled gate and require one complete trace row per decision.

    Args:
        env: Monitor-wrapped gated evaluation environment.
        predict: Deterministic gated policy callable.

    Returns:
        Metrics inputs and complete per-decision gate trace.

    Raises:
        ValueError: If the environment omits gate diagnostics.
    """
    walk, trace = _walk_eval_range_core(env, predict)
    if len(trace) != len(walk[0]):
        raise ValueError(
            "Gated evaluation must emit one gate trace row per decision, "
            f"got {len(trace)} rows for {len(walk[0])} decisions."
        )
    return walk, trace


def _hold_run_lengths(applied: list[bool]) -> list[int]:
    """Return consecutive learned hold-run lengths.

    Args:
        applied: Learned apply decisions in time order.

    Returns:
        Positive lengths for every consecutive hold run.
    """
    runs: list[int] = []
    current = 0
    for decision in applied:
        if decision:
            if current > 0:
                runs.append(current)
                current = 0
        else:
            current += 1
    if current > 0:
        runs.append(current)
    return runs


def compute_gate_metrics(trace: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate behavior and mechanism diagnostics from a complete gate trace.

    Args:
        trace: Non-empty per-decision gate trace.

    Returns:
        JSON-serializable gate metrics.

    Raises:
        ValueError: If no gate decisions were recorded.
    """
    if not trace:
        raise ValueError("Gate metrics require at least one decision trace row.")
    learned = [bool(row["gate_learned_apply"]) for row in trace]
    effective = [bool(row["gate_applied"]) for row in trace]
    hold_runs = _hold_run_lengths(learned)
    proposed_gross = [
        float(np.abs(np.asarray(row["proposed_target_weights"], dtype=float)).sum())
        for row in trace
    ]
    applied_gross = [
        float(np.abs(np.asarray(row["applied_target_weights"], dtype=float)).sum())
        for row in trace
    ]
    return {
        "gate_decision_count": len(trace),
        "gate_apply_count": int(sum(learned)),
        "gate_hold_count": int(len(learned) - sum(learned)),
        "gate_apply_fraction": float(np.mean(learned)),
        "gate_hold_fraction": float(1.0 - np.mean(learned)),
        "effective_gate_apply_fraction": float(np.mean(effective)),
        "hold_run_lengths": hold_runs,
        "mean_hold_run_length": float(np.mean(hold_runs)) if hold_runs else 0.0,
        "max_hold_run_length": max(hold_runs, default=0),
        "mean_proposal_distance_from_current": float(
            np.mean([row["proposal_distance_from_current"] for row in trace])
        ),
        "total_turnover_avoided_by_hold": float(
            np.sum([row["turnover_avoided_by_hold"] for row in trace])
        ),
        "total_immediate_transaction_cost_paid_jpy": float(
            np.sum([row["immediate_transaction_cost_paid_jpy"] for row in trace])
        ),
        "total_immediate_transaction_cost_avoided_by_hold_jpy": float(
            np.sum(
                [row["immediate_transaction_cost_avoided_by_hold_jpy"] for row in trace]
            )
        ),
        "mean_proposed_gross_exposure": float(np.mean(proposed_gross)),
        "mean_applied_gross_exposure": float(np.mean(applied_gross)),
        "mean_gross_exposure_drift": float(
            np.mean(np.abs(np.asarray(proposed_gross) - np.asarray(applied_gross)))
        ),
    }


def write_gate_trace(path: Path, trace: list[dict[str, Any]]) -> None:
    """Write a stable CSV gate trace with JSON-encoded vector columns.

    Args:
        path: Destination CSV path.
        trace: Complete per-decision gate trace.
    """
    rows: list[dict[str, Any]] = []
    vector_fields = {
        "current_target_weights_before",
        "proposed_target_weights",
        "applied_target_weights",
    }
    for source in trace:
        row = dict(source)
        for field in vector_fields:
            row[field] = json.dumps(row[field], separators=(",", ":"))
        rows.append(row)
    pd.DataFrame(rows, columns=_GATE_TRACE_FIELDS).to_csv(path, index=False)


def _run_evaluation(
    run_dir: Path,
    output_dir: Path,
    gate_evaluation_mode: GateEvaluationMode,
) -> dict[str, Any]:
    """Evaluate one source model under an explicit gate treatment.

    Args:
        run_dir: Run directory produced by forex-train.
        output_dir: Directory receiving evaluation artifacts.
        gate_evaluation_mode: Learned gate or same-model forced apply.

    Returns:
        Metrics dictionary written to ``output_dir``.

    Raises:
        TrainerConfigError: If the run directory lacks required artifacts.
    """
    snapshot_path = run_dir / "config_snapshot.yaml"
    model_path = run_dir / "model_final.zip"
    eval_env_path = run_dir / "env_eval.yaml"
    meta_path = run_dir / "meta.json"
    for required in (snapshot_path, model_path, eval_env_path, meta_path):
        if not required.is_file():
            raise TrainerConfigError(
                f"Run directory is missing {required.name}: {run_dir}"
            )

    raw_config = yaml.safe_load(snapshot_path.read_text(encoding="utf-8"))
    config = parse_experiment_config(raw_config)
    if (
        gate_evaluation_mode is GateEvaluationMode.FORCED_APPLY
        and config.run.apply_hold_gate is None
    ):
        raise TrainerConfigError(
            "Forced-apply evaluation requires a trained apply/hold gate."
        )
    require_current_training_provenance(
        json.loads(meta_path.read_text(encoding="utf-8")),
        raw_config,
        meta_path,
    )
    resolved_eval = require_matching_resolved_eval_env(
        raw_config,
        yaml.safe_load(eval_env_path.read_text(encoding="utf-8")),
        eval_env_path,
    )

    env = build_single_env(
        resolved_eval,
        config.custom_feature_names,
        config.custom_cross_feature_names,
        seed=0,
        decision_interval=config.run.decision_interval,
        residual=config.run.residual,
        rank_allocation=config.run.rank_allocation,
        apply_hold_gate=config.run.apply_hold_gate,
        gate_evaluation_mode=gate_evaluation_mode,
    )
    spec = ALGO_REGISTRY[config.algorithm.name]
    evaluation_device = resolve_device(config.run.device)
    model = spec.algo_class.load(model_path, device=evaluation_device)

    state = None

    def predict(observation: dict[str, Any], episode_start: np.ndarray) -> np.ndarray:
        nonlocal state
        action, state = model.predict(
            observation, state=state, episode_start=episode_start, deterministic=True
        )
        return action

    try:
        if config.run.apply_hold_gate is None:
            walk = walk_eval_range(env, predict)
            gate_trace: list[dict[str, Any]] = []
        else:
            walk, gate_trace = walk_gated_eval_range(env, predict)
    finally:
        env.close()

    metrics = compute_metrics(*walk)
    if gate_trace:
        metrics.update(compute_gate_metrics(gate_trace))
    timestamps, equities = walk[2], walk[1]
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    pd.DataFrame({"timestamp": timestamps, "equity_jpy": equities}).to_csv(
        output_dir / "equity_curve.csv", index=False
    )
    evaluation: dict[str, Any] = {
        "model_selection": "validation_best",
        "model_path": model_path.name,
        "model_sha256": sha256_file(model_path),
        "metrics_sha256": sha256_file(output_dir / "metrics.json"),
        "config_snapshot_sha256": sha256_file(snapshot_path),
        "env_eval_sha256": sha256_file(eval_env_path),
        "meta_sha256": sha256_file(meta_path),
        **evaluation_runtime_provenance(evaluation_device, raw_config, snapshot_path),
    }
    if config.run.apply_hold_gate is None:
        evaluation = {"manifest_version": 2, **evaluation}
    else:
        trace_path = output_dir / "gate_trace.csv"
        write_gate_trace(trace_path, gate_trace)
        evaluation = {
            "manifest_version": 3,
            "source_run_dir": str(run_dir.resolve()),
            "gate_evaluation_mode": gate_evaluation_mode.value,
            "gate_trace_sha256": sha256_file(trace_path),
            **evaluation,
        }
    (output_dir / "evaluation.json").write_text(
        json.dumps(evaluation, indent=2), encoding="utf-8"
    )
    return metrics


def run_evaluation(run_dir: Path) -> dict[str, Any]:
    """Evaluate a trained run using its learned gate behavior.

    Args:
        run_dir: Run directory produced by forex-train.

    Returns:
        Metrics written into the source run directory.
    """
    return _run_evaluation(run_dir, run_dir, GateEvaluationMode.LEARNED)


def run_forced_apply_evaluation(run_dir: Path) -> tuple[Path, dict[str, Any]]:
    """Evaluate a gated model while applying every direct-weight proposal.

    Args:
        run_dir: Gated run directory produced by forex-train.

    Returns:
        Forced-apply artifact directory and metrics.

    Raises:
        TrainerConfigError: If the output already exists or gating is disabled.
    """
    output_dir = run_dir / "forced_apply"
    try:
        output_dir.mkdir()
    except FileExistsError as exc:
        raise TrainerConfigError(
            f"Forced-apply evaluation directory already exists: {output_dir}"
        ) from exc
    try:
        metrics = _run_evaluation(run_dir, output_dir, GateEvaluationMode.FORCED_APPLY)
    except (
        TrainerConfigError,
        ConfigError,
        DataError,
        FeatureError,
        OSError,
        RuntimeError,
        ValueError,
    ):
        if not any(output_dir.iterdir()):
            output_dir.rmdir()
        raise
    return output_dir, metrics


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
    parser.add_argument(
        "--gate-mode",
        choices=tuple(mode.value for mode in GateEvaluationMode),
        default=GateEvaluationMode.LEARNED.value,
        help="Gate attribution mode (default: learned).",
    )
    args = parser.parse_args(argv)
    try:
        if args.gate_mode == GateEvaluationMode.LEARNED.value:
            metrics = run_evaluation(Path(args.run))
        else:
            output_dir, metrics = run_forced_apply_evaluation(Path(args.run))
            print(f"forced-apply evaluation: {output_dir}")
        print(json.dumps(metrics, indent=2))
        return 0
    except (
        TrainerConfigError,
        ConfigError,
        DataError,
        FeatureError,
        ValueError,
    ) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
