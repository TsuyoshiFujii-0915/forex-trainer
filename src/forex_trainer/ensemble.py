"""Seed-ensemble evaluation CLI: forex-ensemble-eval (ADR-0007).

Loads the validation-selected models of several runs trained against the same
eval env, walks that env once, and forwards the arithmetic mean of the
members' deterministic actions at every decision. Metrics use the same
contract as forex-eval; artifacts land in runs/<experiment>_ens<N>/<ts>/.
"""

from __future__ import annotations

import argparse
import json
import sys
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
from .evaluate import (
    compute_gate_metrics,
    compute_metrics,
    walk_eval_range,
    walk_gated_eval_range,
    write_gate_trace,
)
from .run_dir import create_run_dir


def _load_member_artifacts(
    run_dir: Path,
) -> tuple[Any, dict[str, Any], dict[str, Any], dict[str, Any], Path]:
    """Validate one member's immutable artifacts without choosing a device.

    Args:
        run_dir: Run directory produced by forex-train.

    Returns:
        Typed config, eval env, raw config, meta, and model path.

    Raises:
        TrainerConfigError: If required artifacts are missing.
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
    resolved_eval = require_matching_resolved_eval_env(
        raw_config,
        yaml.safe_load(eval_env_path.read_text(encoding="utf-8")),
        eval_env_path,
    )
    meta = require_current_training_provenance(
        json.loads(meta_path.read_text(encoding="utf-8")),
        raw_config,
        meta_path,
    )
    return config, resolved_eval, raw_config, meta, model_path


def load_member_for_device(
    run_dir: Path,
    evaluation_device: str,
) -> tuple[Any, Any, dict[str, Any], dict[str, Any], dict[str, Any], str]:
    """Load one validated member on an explicitly sealed evaluation device.

    Args:
        run_dir: Run directory produced by forex-train.
        evaluation_device: Concrete device recorded by the source evaluation.

    Returns:
        Typed config, model, eval env, raw config, meta, and sealed device.

    Raises:
        TrainerConfigError: If the sealed device is not a concrete supported device.
    """
    if evaluation_device not in {"cpu", "cuda", "mps"}:
        raise TrainerConfigError(
            "Sealed evaluation device must be cpu, cuda, or mps, got "
            f"{evaluation_device!r}."
        )
    config, resolved_eval, raw_config, meta, model_path = _load_member_artifacts(
        run_dir
    )
    spec = ALGO_REGISTRY[config.algorithm.name]
    model = spec.algo_class.load(model_path, device=evaluation_device)
    return config, model, resolved_eval, raw_config, meta, evaluation_device


def _load_member(
    run_dir: Path,
) -> tuple[Any, Any, dict[str, Any], dict[str, Any], dict[str, Any], str]:
    """Load one member using the current evaluation-device resolution policy.

    Args:
        run_dir: Run directory produced by forex-train.

    Returns:
        Typed config, model, eval env, raw config, meta, and actual model device.
    """
    config, resolved_eval, raw_config, meta, model_path = _load_member_artifacts(
        run_dir
    )
    evaluation_device = resolve_device(config.run.device)
    spec = ALGO_REGISTRY[config.algorithm.name]
    model = spec.algo_class.load(model_path, device=evaluation_device)
    return config, model, resolved_eval, raw_config, meta, evaluation_device


def _run_ensemble_evaluation(
    run_dirs: list[Path],
    runs_root: Path,
    gate_evaluation_mode: GateEvaluationMode,
) -> tuple[Path, dict[str, Any]]:
    """Evaluate an action-mean ensemble under an explicit gate treatment.

    Args:
        run_dirs: Member run directories; all must share an identical
            resolved eval env and decision interval.
        runs_root: Root directory under which the ensemble run directory is
            created.
        gate_evaluation_mode: Learned gate or same-model forced apply.

    Returns:
        Tuple of (ensemble run directory, metrics dict).

    Raises:
        TrainerConfigError: If no members are given or members disagree on
            the eval env or decision interval.
    """
    if not run_dirs:
        raise TrainerConfigError("At least one member run directory is required.")
    members = [_load_member(run_dir) for run_dir in run_dirs]
    reference_config, _, reference_eval, reference_raw, _, evaluation_device = members[
        0
    ]
    if (
        gate_evaluation_mode is GateEvaluationMode.FORCED_APPLY
        and reference_config.run.apply_hold_gate is None
    ):
        raise TrainerConfigError(
            "Forced-apply ensemble evaluation requires gated member runs."
        )
    for run_dir, (
        config,
        _,
        resolved_eval,
        _,
        _,
        member_device,
    ) in zip(run_dirs[1:], members[1:]):
        if resolved_eval != reference_eval:
            raise TrainerConfigError(
                f"Member {run_dir} has a different resolved eval env than "
                f"{run_dirs[0]}; ensemble members must share one eval env."
            )
        if config.run.decision_interval != reference_config.run.decision_interval:
            raise TrainerConfigError(
                f"Member {run_dir} has decision_interval "
                f"{config.run.decision_interval}, expected "
                f"{reference_config.run.decision_interval}."
            )
        if config.run.residual != reference_config.run.residual:
            raise TrainerConfigError(
                f"Member {run_dir} has a different residual action scheme."
            )
        if config.run.rank_allocation != reference_config.run.rank_allocation:
            raise TrainerConfigError(
                f"Member {run_dir} has a different rank allocation scheme."
            )
        if config.run.apply_hold_gate != reference_config.run.apply_hold_gate:
            raise TrainerConfigError(
                f"Member {run_dir} has a different apply/hold gate scheme."
            )
        if member_device != evaluation_device:
            raise TrainerConfigError(
                f"Member {run_dir} resolves evaluation device {member_device}, "
                f"expected {evaluation_device}."
            )

    env = build_single_env(
        reference_eval,
        reference_config.custom_feature_names,
        reference_config.custom_cross_feature_names,
        seed=0,
        decision_interval=reference_config.run.decision_interval,
        residual=reference_config.run.residual,
        rank_allocation=reference_config.run.rank_allocation,
        apply_hold_gate=reference_config.run.apply_hold_gate,
        gate_evaluation_mode=gate_evaluation_mode,
    )
    states: list[Any] = [None] * len(members)

    def predict(observation: dict[str, Any], episode_start: np.ndarray) -> np.ndarray:
        actions = []
        for index, (_, model, _, _, _, _) in enumerate(members):
            action, states[index] = model.predict(
                observation,
                state=states[index],
                episode_start=episode_start,
                deterministic=True,
            )
            actions.append(np.asarray(action, dtype=np.float64))
        return np.mean(np.stack(actions, axis=0), axis=0).astype(np.float32)

    try:
        if reference_config.run.apply_hold_gate is None:
            walk = walk_eval_range(env, predict)
            gate_trace: list[dict[str, Any]] = []
        else:
            walk, gate_trace = walk_gated_eval_range(env, predict)
    finally:
        env.close()
    metrics = compute_metrics(*walk)
    if gate_trace:
        metrics.update(compute_gate_metrics(gate_trace))
    equities, timestamps = walk[1], walk[2]

    experiment = f"{reference_config.experiment}_ens{len(members)}"
    if gate_evaluation_mode is GateEvaluationMode.FORCED_APPLY:
        experiment = f"{experiment}_forced_apply"
    ensemble_dir = create_run_dir(runs_root, experiment)
    (ensemble_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    pd.DataFrame({"timestamp": timestamps, "equity_jpy": equities}).to_csv(
        ensemble_dir / "equity_curve.csv", index=False
    )
    (ensemble_dir / "env_eval.yaml").write_text(
        yaml.safe_dump(dict(reference_eval), sort_keys=False), encoding="utf-8"
    )
    member_records = []
    for run_dir, (config, _, _, _, meta, _) in zip(run_dirs, members):
        member_records.append(
            {
                "run_dir": str(run_dir.resolve()),
                "experiment": config.experiment,
                "seed": config.run.seed,
                "model_path": "model_final.zip",
                "model_sha256": sha256_file(run_dir / "model_final.zip"),
                "config_snapshot_sha256": sha256_file(run_dir / "config_snapshot.yaml"),
                "meta_sha256": sha256_file(run_dir / "meta.json"),
            }
        )
    evaluation: dict[str, Any] = {
        **evaluation_runtime_provenance(
            evaluation_device,
            reference_raw,
            run_dirs[0] / "config_snapshot.yaml",
        ),
        "metrics_sha256": sha256_file(ensemble_dir / "metrics.json"),
        "env_eval_sha256": sha256_file(ensemble_dir / "env_eval.yaml"),
    }
    manifest: dict[str, Any] = {
        "experiment": experiment,
        "policy": "action_mean",
        "model_selection": "validation_best",
        "decision_interval": reference_config.run.decision_interval,
        "members": member_records,
        "evaluation": evaluation,
    }
    if reference_config.run.apply_hold_gate is None:
        manifest = {"manifest_version": 2, **manifest}
    else:
        trace_path = ensemble_dir / "gate_trace.csv"
        write_gate_trace(trace_path, gate_trace)
        evaluation["gate_trace_sha256"] = sha256_file(trace_path)
        manifest = {
            "manifest_version": 3,
            "gate_evaluation_mode": gate_evaluation_mode.value,
            **manifest,
        }
    (ensemble_dir / "ensemble.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return ensemble_dir, metrics


def run_ensemble_evaluation(
    run_dirs: list[Path], runs_root: Path
) -> tuple[Path, dict[str, Any]]:
    """Evaluate members with their learned action-mean gate behavior.

    Args:
        run_dirs: Compatible member run directories.
        runs_root: Root directory for ensemble artifacts.

    Returns:
        Ensemble artifact directory and metrics.
    """
    return _run_ensemble_evaluation(run_dirs, runs_root, GateEvaluationMode.LEARNED)


def run_forced_apply_ensemble_evaluation(
    run_dirs: list[Path], runs_root: Path
) -> tuple[Path, dict[str, Any]]:
    """Evaluate the same gated ensemble while applying every mean proposal.

    Args:
        run_dirs: Compatible gated member run directories.
        runs_root: Root directory for ensemble artifacts.

    Returns:
        Forced-apply ensemble artifact directory and metrics.
    """
    return _run_ensemble_evaluation(
        run_dirs, runs_root, GateEvaluationMode.FORCED_APPLY
    )


def main(argv: list[str] | None = None) -> int:
    """CLI entry point.

    Args:
        argv: CLI arguments; None lets argparse read sys.argv (testability).

    Returns:
        Process exit code: 0 on success, 1 on errors.
    """
    parser = argparse.ArgumentParser(
        prog="forex-ensemble-eval",
        description="Evaluate the action-mean ensemble of several runs.",
    )
    parser.add_argument(
        "--runs", type=str, nargs="+", required=True, help="Member run directories."
    )
    parser.add_argument(
        "--gate-mode",
        choices=tuple(mode.value for mode in GateEvaluationMode),
        default=GateEvaluationMode.LEARNED.value,
        help="Gate attribution mode (default: learned).",
    )
    parser.add_argument(
        "--runs-root",
        type=str,
        default="runs",
        help="Root directory for the ensemble artifacts (default: runs, the "
        "repo convention).",
    )
    args = parser.parse_args(argv)
    try:
        run_dirs = [Path(run) for run in args.runs]
        runs_root = Path(args.runs_root)
        if args.gate_mode == GateEvaluationMode.LEARNED.value:
            ensemble_dir, metrics = run_ensemble_evaluation(run_dirs, runs_root)
        else:
            ensemble_dir, metrics = run_forced_apply_ensemble_evaluation(
                run_dirs, runs_root
            )
    except (
        TrainerConfigError,
        ConfigError,
        DataError,
        FeatureError,
        ValueError,
    ) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"ensemble run: {ensemble_dir}")
    print(json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
