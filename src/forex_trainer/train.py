"""Training CLI: forex-train --config <experiment.yaml>."""

from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

from forex_env.errors import ConfigError, DataError, FeatureError
from stable_baselines3.common.callbacks import EvalCallback

from .algorithms import build_model, resolve_device
from .config import (
    TrainerConfigError,
    load_experiment_config,
    parse_experiment_config,
    resolve_env_raw,
)
from .env_factory import build_single_env, build_vec_env
from .run_dir import (
    capture_run_data_provenance,
    create_run_dir,
    resolve_file_data_path,
    write_run_metadata,
)

# Target number of validation walks per training run (ADR-0005).
_VALIDATION_WALKS = 20


def run_training(config_path: Path, runs_root: Path, seed_override: int | None) -> Path:
    """Run one training experiment and return its run directory.

    Args:
        config_path: Path to the experiment YAML.
        runs_root: Root directory under which the run directory is created.
        seed_override: If given, replaces run.seed (CLI convenience for seed
            sweeps without editing the YAML).

    Returns:
        Path to the run directory containing all artifacts.
    """
    config, raw = load_experiment_config(config_path)
    if seed_override is not None:
        raw = copy.deepcopy(raw)
        raw["run"]["seed"] = seed_override
        config = parse_experiment_config(raw)

    resolved_train = resolve_env_raw(config.env, config.train_range, for_eval=False)
    resolved_val = resolve_env_raw(config.env, config.val_range, for_eval=True)
    resolved_eval = resolve_env_raw(config.env, config.eval_range, for_eval=True)
    working_directory = Path.cwd()
    resolved_train = resolve_file_data_path(resolved_train, working_directory)
    resolved_val = resolve_file_data_path(resolved_val, working_directory)
    resolved_eval = resolve_file_data_path(resolved_eval, working_directory)
    data_provenance = capture_run_data_provenance(
        resolved_train, resolved_val, resolved_eval
    )
    device = resolve_device(config.run.device)

    run_dir = create_run_dir(runs_root, config.experiment)
    write_run_metadata(
        run_dir,
        config,
        raw,
        resolved_train,
        resolved_val,
        resolved_eval,
        device,
        data_provenance,
    )

    vec_env = build_vec_env(
        resolved_train,
        config.custom_feature_names,
        config.custom_cross_feature_names,
        config.run.n_envs,
        config.run.vec_env,
        config.run.seed,
        config.run.decision_interval,
        config.run.residual,
    )
    val_env = build_single_env(
        resolved_val,
        config.custom_feature_names,
        config.custom_cross_feature_names,
        seed=0,
        decision_interval=config.run.decision_interval,
        residual=config.run.residual,
    )
    # ~20 validation walks over the run; eval_freq counts per-env steps, and
    # the max(1, ...) floor guarantees at least one walk even on tiny budgets,
    # so best_model.zip always exists afterwards.
    eval_freq = max(
        1, config.run.total_timesteps // (_VALIDATION_WALKS * config.run.n_envs)
    )
    callback = EvalCallback(
        val_env,
        best_model_save_path=str(run_dir),
        log_path=str(run_dir),
        eval_freq=eval_freq,
        n_eval_episodes=1,
        deterministic=True,
        verbose=0,
    )
    try:
        model = build_model(config, vec_env, device, run_dir / "tensorboard")
        model.learn(total_timesteps=config.run.total_timesteps, callback=callback)
        model.save(run_dir / "model_last")
    finally:
        vec_env.close()
        val_env.close()

    best_model = run_dir / "best_model.zip"
    if not best_model.is_file():
        raise RuntimeError(
            f"EvalCallback produced no best model in {run_dir}; validation never "
            f"ran (total_timesteps={config.run.total_timesteps}, "
            f"eval_freq={eval_freq})."
        )
    # The run's deliverable is the validation-selected model (ADR-0005).
    best_model.replace(run_dir / "model_final.zip")
    return run_dir


def main(argv: list[str] | None = None) -> int:
    """CLI entry point.

    Args:
        argv: CLI arguments; None lets argparse read sys.argv (testability).

    Returns:
        Process exit code: 0 on success, 1 on configuration/data errors.
    """
    parser = argparse.ArgumentParser(
        prog="forex-train", description="Train an RL agent per an experiment YAML."
    )
    parser.add_argument(
        "--config", type=str, required=True, help="Experiment YAML path."
    )
    parser.add_argument(
        "--runs-root",
        type=str,
        default="runs",
        help="Root directory for run artifacts (default: runs, the repo convention).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override run.seed for seed sweeps without editing the YAML.",
    )
    args = parser.parse_args(argv)
    try:
        run_dir = run_training(Path(args.config), Path(args.runs_root), args.seed)
        print(f"run completed: {run_dir}")
        return 0
    except (TrainerConfigError, ConfigError, DataError, FeatureError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
