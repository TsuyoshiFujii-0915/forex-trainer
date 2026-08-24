"""Controlled data-scaling studies for RL generalization (Issue #7).

The study runner materializes fixed-history and expanding-history variants of
existing experiment configurations.  It deliberately delegates training and
evaluation to the established commands so the study does not introduce a
second learning or scoring protocol.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import html
import json
import math
import multiprocessing
import os
import sys
from concurrent.futures import Future, ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import yaml
from forex_env.data import FileDataProvider
from forex_env.errors import ConfigError, DataError, FeatureError

from .config import (
    DEVICES,
    TrainerConfigError,
    load_experiment_config,
    parse_experiment_config,
)
from .ensemble import run_ensemble_evaluation
from .evaluate import run_evaluation
from .features import mom24, xr_mom24
from .train import run_training

_STUDY_KEYS: tuple[str, ...] = (
    "name",
    "fold_configs",
    "audit_fold_configs",
    "seeds",
    "history_years",
    "device",
    "workers",
    "bootstrap_samples",
    "bootstrap_seed",
)
_MANIFEST_VERSION = 1
_PPO_DEFAULT_N_EPOCHS = 10
_SECONDS_PER_YEAR = 365.25 * 86_400.0


@dataclass(frozen=True)
class ScalingStudy:
    """Validated definition of one controlled data-scaling study."""

    name: str
    fold_configs: tuple[Path, ...]
    audit_fold_configs: tuple[Path, ...]
    seeds: tuple[int, ...]
    history_years: tuple[int, ...]
    device: str
    workers: int
    bootstrap_samples: int
    bootstrap_seed: int
    source_path: Path


@dataclass(frozen=True)
class RolloutAccounting:
    """PPO sampling and optimization counts for a requested step budget."""

    requested_steps: int
    actual_steps: int
    rollouts: int
    episode_equivalents: float
    optimizer_minibatch_steps: int
    sample_presentations: int


@dataclass(frozen=True)
class BootstrapIntervals:
    """Fold-level and moving-block annualized-return intervals."""

    fold_low: float
    fold_high: float
    moving_block_low: float
    moving_block_high: float


@dataclass(frozen=True)
class _TrainingJob:
    """One materialized fold, history condition, and seed."""

    key: str
    fold: str
    condition: str
    seed: int
    config_path: Path
    data_path: Path
    data_sha256: str


def _require_exact_keys(raw: Mapping[str, Any]) -> None:
    """Require the study root to contain exactly the supported controls.

    Args:
        raw: Raw study mapping.

    Raises:
        TrainerConfigError: If required keys are missing or unknown keys exist.
    """
    actual = set(raw)
    expected = set(_STUDY_KEYS)
    if actual != expected:
        missing = sorted(expected - actual)
        unknown = sorted(actual - expected)
        raise TrainerConfigError(
            "Scaling study must contain exactly "
            f"{list(_STUDY_KEYS)}; missing={missing}, unknown={unknown}."
        )


def _require_int(value: Any, field: str, minimum: int | None) -> int:
    """Validate an integer study field.

    Args:
        value: Candidate value.
        field: Field name for errors.
        minimum: Inclusive minimum, or None when no lower bound applies.

    Returns:
        Validated integer.

    Raises:
        TrainerConfigError: If the value has the wrong type or is too small.
    """
    if isinstance(value, bool) or not isinstance(value, int):
        raise TrainerConfigError(f"Scaling study {field} must be an integer.")
    if minimum is not None and value < minimum:
        raise TrainerConfigError(
            f"Scaling study {field} must be >= {minimum}, got {value}."
        )
    return value


def _require_int_sequence(value: Any, field: str, minimum: int) -> tuple[int, ...]:
    """Validate a non-empty sequence of unique integers.

    Args:
        value: Candidate sequence.
        field: Field name for errors.
        minimum: Inclusive lower bound for each item.

    Returns:
        Validated tuple preserving declared order.

    Raises:
        TrainerConfigError: If the sequence is invalid.
    """
    if not isinstance(value, list) or not value:
        raise TrainerConfigError(
            f"Scaling study {field} must be a non-empty list of integers."
        )
    result = tuple(
        _require_int(item, f"{field}[{index}]", minimum)
        for index, item in enumerate(value)
    )
    if len(set(result)) != len(result):
        raise TrainerConfigError(f"Scaling study {field} must not contain duplicates.")
    return result


def _resolve_config_paths(
    value: Any, field: str, study_dir: Path
) -> tuple[Path, ...]:
    """Validate and resolve experiment config paths against the study directory.

    Args:
        value: Candidate list of YAML paths.
        field: Field name for errors.
        study_dir: Directory containing the study YAML.

    Returns:
        Tuple of absolute, existing paths.

    Raises:
        TrainerConfigError: If the list or one of its paths is invalid.
    """
    if not isinstance(value, list) or not value:
        raise TrainerConfigError(
            f"Scaling study {field} must be a non-empty list of paths."
        )
    paths: list[Path] = []
    for index, item in enumerate(value):
        if not isinstance(item, str) or not item:
            raise TrainerConfigError(
                f"Scaling study {field}[{index}] must be a non-empty string."
            )
        candidate = Path(item)
        path = candidate if candidate.is_absolute() else study_dir / candidate
        path = path.resolve()
        if not path.is_file():
            raise TrainerConfigError(
                f"Scaling study {field}[{index}] does not exist: {path}"
            )
        paths.append(path)
    if len(set(paths)) != len(paths):
        raise TrainerConfigError(f"Scaling study {field} must not contain duplicates.")
    return tuple(paths)


def load_scaling_study(study_path: str | Path) -> ScalingStudy:
    """Load and strictly validate a data-scaling study YAML.

    Args:
        study_path: Study YAML path.

    Returns:
        Typed study definition with resolved config paths.

    Raises:
        TrainerConfigError: If the file or any field is invalid.
    """
    path = Path(study_path).resolve()
    if not path.is_file():
        raise TrainerConfigError(f"Scaling study file not found: {path}")
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise TrainerConfigError(f"Failed to parse scaling study {path}: {exc}") from exc
    if not isinstance(raw, Mapping):
        raise TrainerConfigError(f"Scaling study root must be a mapping: {path}")
    _require_exact_keys(raw)
    name = raw["name"]
    if not isinstance(name, str) or not name:
        raise TrainerConfigError("Scaling study name must be a non-empty string.")
    device = raw["device"]
    if not isinstance(device, str) or device not in DEVICES:
        raise TrainerConfigError(
            f"Scaling study device must be one of {list(DEVICES)}, got {device!r}."
        )
    return ScalingStudy(
        name=name,
        fold_configs=_resolve_config_paths(
            raw["fold_configs"], "fold_configs", path.parent
        ),
        audit_fold_configs=_resolve_config_paths(
            raw["audit_fold_configs"], "audit_fold_configs", path.parent
        ),
        seeds=_require_int_sequence(raw["seeds"], "seeds", 0),
        history_years=_require_int_sequence(
            raw["history_years"], "history_years", 1
        ),
        device=device,
        workers=_require_int(raw["workers"], "workers", 1),
        bootstrap_samples=_require_int(
            raw["bootstrap_samples"], "bootstrap_samples", 1
        ),
        bootstrap_seed=_require_int(raw["bootstrap_seed"], "bootstrap_seed", 0),
        source_path=path,
    )


def rollout_accounting(
    requested_steps: int,
    n_steps: int,
    n_envs: int,
    episode_max_steps: int,
    batch_size: int,
    n_epochs: int,
) -> RolloutAccounting:
    """Calculate the complete-rollout work performed by PPO.

    Args:
        requested_steps: Lower-bound environment-step budget passed to SB3.
        n_steps: Steps collected by each environment per rollout.
        n_envs: Number of vector environments.
        episode_max_steps: Maximum environment decisions per episode.
        batch_size: PPO optimizer minibatch size.
        n_epochs: Optimizer passes over every rollout.

    Returns:
        Exact rollout, optimizer, and sample-presentation counts.

    Raises:
        ValueError: If an argument is non-positive or the rollout cannot be
            partitioned into complete minibatches.
    """
    arguments = {
        "requested_steps": requested_steps,
        "n_steps": n_steps,
        "n_envs": n_envs,
        "episode_max_steps": episode_max_steps,
        "batch_size": batch_size,
        "n_epochs": n_epochs,
    }
    for name, value in arguments.items():
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer, got {value!r}.")
    rollout_size = n_steps * n_envs
    if rollout_size % batch_size != 0:
        raise ValueError(
            f"PPO rollout size {rollout_size} is not divisible by batch_size "
            f"{batch_size}; exact optimizer accounting is impossible."
        )
    rollouts = math.ceil(requested_steps / rollout_size)
    actual_steps = rollouts * rollout_size
    optimizer_minibatch_steps = rollouts * (rollout_size // batch_size) * n_epochs
    return RolloutAccounting(
        requested_steps=requested_steps,
        actual_steps=actual_steps,
        rollouts=rollouts,
        episode_equivalents=actual_steps / episode_max_steps,
        optimizer_minibatch_steps=optimizer_minibatch_steps,
        sample_presentations=actual_steps * n_epochs,
    )


def effective_sample_size(values: Sequence[float] | np.ndarray, max_lag: int) -> float:
    """Estimate MCMC-style ESS with Geyer's initial-positive-pair sequence.

    Args:
        values: One-dimensional finite scalar series.
        max_lag: Largest autocorrelation lag to examine.

    Returns:
        Effective sample size in ``[1, len(values)]``.

    Raises:
        ValueError: If the series or lag bound is invalid, or variance is zero.
    """
    series = np.asarray(values, dtype=np.float64)
    if series.ndim != 1 or len(series) < 2:
        raise ValueError("values must be a one-dimensional series of length >= 2.")
    if not np.isfinite(series).all():
        raise ValueError("values must contain only finite numbers.")
    if isinstance(max_lag, bool) or not isinstance(max_lag, int) or max_lag < 1:
        raise ValueError(f"max_lag must be a positive integer, got {max_lag!r}.")
    centered = series - float(series.mean())
    variance = float(np.dot(centered, centered) / len(centered))
    if variance == 0.0:
        raise ValueError("effective sample size is undefined for a constant series.")
    lag_limit = min(max_lag, len(series) - 1)
    autocorrelations = np.empty(lag_limit + 1, dtype=np.float64)
    autocorrelations[0] = 1.0
    for lag in range(1, lag_limit + 1):
        autocovariance = float(
            np.dot(centered[:-lag], centered[lag:]) / len(centered)
        )
        autocorrelations[lag] = autocovariance / variance
    paired_sum = 0.0
    first_lag = 1
    while first_lag <= lag_limit:
        second_lag = first_lag + 1
        pair = float(autocorrelations[first_lag])
        if second_lag <= lag_limit:
            pair += float(autocorrelations[second_lag])
        if pair <= 0.0:
            break
        paired_sum += pair
        first_lag += 2
    integrated_autocorrelation = 1.0 + 2.0 * paired_sum
    estimate = len(series) / integrated_autocorrelation
    return float(min(len(series), max(1.0, estimate)))


def bootstrap_annualized_fold_returns(
    fold_log_returns: np.ndarray,
    fold_years: np.ndarray,
    samples: int,
    seed: int,
    block_length: int,
) -> BootstrapIntervals:
    """Bootstrap annualized returns using folds as the sampling unit.

    Args:
        fold_log_returns: One seed-averaged cumulative log return per fold.
        fold_years: Evaluation duration in years for each fold.
        samples: Number of bootstrap draws.
        seed: Deterministic NumPy generator seed.
        block_length: Circular moving-block length in adjacent folds.

    Returns:
        Percentile intervals from IID-fold and moving-block resampling.

    Raises:
        ValueError: If arrays or bootstrap controls are invalid.
    """
    returns = np.asarray(fold_log_returns, dtype=np.float64)
    years = np.asarray(fold_years, dtype=np.float64)
    if returns.ndim != 1 or years.ndim != 1 or returns.shape != years.shape:
        raise ValueError(
            "fold_log_returns and fold_years must be equal-length 1D arrays."
        )
    if len(returns) < 1 or not np.isfinite(returns).all():
        raise ValueError("fold_log_returns must contain at least one finite fold.")
    if not np.isfinite(years).all() or bool(np.any(years <= 0.0)):
        raise ValueError("fold_years must contain finite positive durations.")
    if isinstance(samples, bool) or not isinstance(samples, int) or samples < 1:
        raise ValueError(f"samples must be a positive integer, got {samples!r}.")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError(f"seed must be a non-negative integer, got {seed!r}.")
    if (
        isinstance(block_length, bool)
        or not isinstance(block_length, int)
        or not 1 <= block_length <= len(returns)
    ):
        raise ValueError(
            f"block_length must be in [1, {len(returns)}], got {block_length!r}."
        )

    generator = np.random.default_rng(seed)
    fold_indices = generator.integers(
        0, len(returns), size=(samples, len(returns))
    )
    fold_draws = np.expm1(
        returns[fold_indices].sum(axis=1) / years[fold_indices].sum(axis=1)
    )
    block_indices = np.empty((samples, len(returns)), dtype=int)
    for sample_index in range(samples):
        selected: list[int] = []
        while len(selected) < len(returns):
            start = int(generator.integers(0, len(returns)))
            selected.extend(
                (start + offset) % len(returns) for offset in range(block_length)
            )
        block_indices[sample_index] = selected[: len(returns)]
    block_draws = np.expm1(
        returns[block_indices].sum(axis=1) / years[block_indices].sum(axis=1)
    )
    return BootstrapIntervals(
        fold_low=float(np.quantile(fold_draws, 0.025)),
        fold_high=float(np.quantile(fold_draws, 0.975)),
        moving_block_low=float(np.quantile(block_draws, 0.025)),
        moving_block_high=float(np.quantile(block_draws, 0.975)),
    )


def _subtract_years(value: str, years: int) -> str:
    """Subtract whole calendar years from an ISO date.

    Args:
        value: ISO date.
        years: Positive number of years.

    Returns:
        Shifted ISO date. February 29 maps to February 28 in a non-leap year.
    """
    parsed = date.fromisoformat(value)
    target_year = parsed.year - years
    if parsed.month == 2 and parsed.day == 29:
        return date(target_year, 2, 28).isoformat()
    return parsed.replace(year=target_year).isoformat()


def _condition_names(study: ScalingStudy) -> tuple[str, ...]:
    """Return fixed-history conditions followed by the expanding control.

    Args:
        study: Validated study.

    Returns:
        Ordered condition names.
    """
    return tuple(f"{years}y" for years in study.history_years) + ("expanding",)


def _materialize_raw(
    source_raw: Mapping[str, Any], condition: str, seed: int, device: str
) -> dict[str, Any]:
    """Create one controlled experiment variant.

    Args:
        source_raw: Validated source experiment mapping.
        condition: Fixed-year label such as ``1y`` or ``expanding``.
        seed: Training seed.
        device: Requested training device.

    Returns:
        Deep-copied materialized experiment mapping.

    Raises:
        TrainerConfigError: If a fixed history exceeds available source history.
    """
    raw: dict[str, Any] = copy.deepcopy(dict(source_raw))
    original_start = str(raw["train_range"]["start"])
    if condition != "expanding":
        years_text = condition.removesuffix("y")
        if not years_text.isdigit():
            raise TrainerConfigError(f"Invalid scaling condition: {condition}")
        fixed_start = _subtract_years(str(raw["train_range"]["end"]), int(years_text))
        if date.fromisoformat(fixed_start) < date.fromisoformat(original_start):
            raise TrainerConfigError(
                f"Condition {condition} requires data from {fixed_start}, before "
                f"source train_range.start {original_start}."
            )
        raw["train_range"]["start"] = fixed_start
    raw["experiment"] = (
        f"{raw['experiment']}__scale_{condition}__seed{seed}"
    )
    raw["run"]["seed"] = seed
    raw["run"]["device"] = device
    return raw


def _atomic_write_text(path: Path, content: str) -> None:
    """Atomically replace a UTF-8 text file in its destination directory.

    Args:
        path: Destination path.
        content: Complete new file content.
    """
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _atomic_write_json(path: Path, value: Any) -> None:
    """Atomically serialize JSON.

    Args:
        path: Destination path.
        value: JSON-serializable value.
    """
    _atomic_write_text(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def _sha256_file(path: Path) -> str:
    """Hash a file without loading it entirely into memory.

    Args:
        path: Existing file path.

    Returns:
        Lowercase SHA-256 digest.
    """
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _study_identity(study: ScalingStudy) -> dict[str, Any]:
    """Build immutable manifest identity fields for resume validation.

    Args:
        study: Validated study.

    Returns:
        JSON-serializable identity dictionary.
    """
    source_paths = tuple(
        dict.fromkeys(study.fold_configs + study.audit_fold_configs)
    )
    data_paths: dict[str, Path] = {}
    for config_path in source_paths:
        _, raw = load_experiment_config(config_path)
        data_path = _resolve_data_path(raw)
        data_paths[str(data_path)] = data_path
    return {
        "manifest_version": _MANIFEST_VERSION,
        "study_path": str(study.source_path),
        "study_sha256": _sha256_file(study.source_path),
        "name": study.name,
        "fold_config_sha256": {
            str(path): _sha256_file(path) for path in study.fold_configs
        },
        "audit_fold_config_sha256": {
            str(path): _sha256_file(path) for path in study.audit_fold_configs
        },
        "data_sha256": {
            path_text: _sha256_file(path) for path_text, path in data_paths.items()
        },
    }


def _load_or_create_manifest(study: ScalingStudy, output_dir: Path) -> dict[str, Any]:
    """Load a compatible resume manifest or create a new one.

    Args:
        study: Validated study.
        output_dir: Study artifact directory.

    Returns:
        Mutable manifest dictionary.

    Raises:
        TrainerConfigError: If existing output belongs to different inputs.
    """
    manifest_path = output_dir / "manifest.json"
    identity = _study_identity(study)
    if manifest_path.is_file():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise TrainerConfigError(
                f"Malformed scaling manifest at {manifest_path}: {exc}"
            ) from exc
        for key, expected in identity.items():
            if manifest.get(key) != expected:
                raise TrainerConfigError(
                    f"Cannot resume {output_dir}: manifest field {key!r} does "
                    "not match the current study inputs."
                )
        jobs = manifest.get("jobs")
        if not isinstance(jobs, dict):
            raise TrainerConfigError(
                f"Malformed scaling manifest jobs at {manifest_path}."
            )
        return manifest
    manifest = {**identity, "status": "running", "jobs": {}, "ensembles": {}}
    _atomic_write_json(manifest_path, manifest)
    return manifest


def _job_key(fold: Path, condition: str, seed: int) -> str:
    """Build a stable key for one source training run.

    Args:
        fold: Absolute source fold path.
        condition: History condition.
        seed: Training seed.

    Returns:
        SHA-prefixed human-readable job key.
    """
    identity = f"{fold}|{condition}|{seed}"
    prefix = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:12]
    return f"{fold.stem}:{condition}:seed{seed}:{prefix}"


def _protocol_projection(raw: Mapping[str, Any]) -> dict[str, Any]:
    """Remove only fold identity, date ranges, and run seed from a protocol.

    Args:
        raw: Validated experiment mapping.

    Returns:
        Deep protocol projection used for strict equality checks.
    """
    projected: dict[str, Any] = copy.deepcopy(dict(raw))
    del projected["experiment"]
    del projected["train_range"]
    del projected["val_range"]
    del projected["eval_range"]
    del projected["run"]["seed"]
    return projected


def _validate_shared_protocol(study: ScalingStudy) -> None:
    """Require all source and audit folds to share one learning protocol.

    Args:
        study: Validated study definition.

    Raises:
        TrainerConfigError: If folds differ outside name, ranges, or run seed.
    """
    paths = tuple(dict.fromkeys(study.fold_configs + study.audit_fold_configs))
    reference_path = paths[0]
    _, reference_raw = load_experiment_config(reference_path)
    reference = _protocol_projection(reference_raw)
    for path in paths[1:]:
        _, raw = load_experiment_config(path)
        if _protocol_projection(raw) != reference:
            raise TrainerConfigError(
                f"Scaling fold {path} does not share the protocol of "
                f"{reference_path}; only experiment, date ranges, and run.seed "
                "may differ across source folds."
            )


def _materialize_jobs(
    study: ScalingStudy, output_dir: Path
) -> tuple[list[_TrainingJob], dict[tuple[str, str], list[str]]]:
    """Validate source folds and write every controlled experiment YAML.

    Args:
        study: Validated study.
        output_dir: Study artifact directory.

    Returns:
        Training jobs and grouping of job keys by fold/condition.
    """
    config_dir = output_dir / "materialized_configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    jobs: list[_TrainingJob] = []
    groups: dict[tuple[str, str], list[str]] = {}
    for fold_index, fold_path in enumerate(study.fold_configs):
        _, source_raw = load_experiment_config(fold_path)
        data_path = _resolve_data_path(source_raw)
        data_sha256 = _sha256_file(data_path)
        for condition in _condition_names(study):
            group = (str(fold_path), condition)
            groups[group] = []
            for seed in study.seeds:
                key = _job_key(fold_path, condition, seed)
                materialized = _materialize_raw(
                    source_raw, condition, seed, study.device
                )
                # Parsing here proves that the four controlled mutations did
                # not invalidate the existing experiment contract.
                parse_experiment_config(materialized)
                destination = config_dir / (
                    f"fold{fold_index:03d}_{condition}_seed{seed}.yaml"
                )
                _atomic_write_text(
                    destination,
                    yaml.safe_dump(materialized, sort_keys=False),
                )
                jobs.append(
                    _TrainingJob(
                        key=key,
                        fold=str(fold_path),
                        condition=condition,
                        seed=seed,
                        config_path=destination,
                        data_path=data_path,
                        data_sha256=data_sha256,
                    )
                )
                groups[group].append(key)
    return jobs, groups


def _completed_job(entry: Any) -> bool:
    """Verify that a manifest entry has complete train/eval artifacts.

    Args:
        entry: Candidate manifest job entry.

    Returns:
        True only when every required artifact exists.
    """
    if not isinstance(entry, Mapping) or entry.get("status") != "complete":
        return False
    run_dir_value = entry.get("run_dir")
    if not isinstance(run_dir_value, str):
        return False
    run_dir = Path(run_dir_value)
    return all(
        (run_dir / name).is_file()
        for name in ("model_final.zip", "metrics.json", "training_stats.json")
    )


def _execute_job(job: _TrainingJob, runs_root: Path) -> dict[str, Any]:
    """Train and evaluate one materialized experiment.

    Args:
        job: Materialized job.
        runs_root: Shared root for run artifacts.

    Returns:
        Completed manifest entry.
    """
    before_sha = _sha256_file(job.data_path)
    if before_sha != job.data_sha256:
        raise TrainerConfigError(
            f"Input data changed before job {job.key}: expected "
            f"{job.data_sha256}, observed {before_sha}."
        )
    run_dir = run_training(job.config_path, runs_root, seed_override=None)
    metrics = run_evaluation(run_dir)
    after_sha = _sha256_file(job.data_path)
    if after_sha != job.data_sha256:
        raise TrainerConfigError(
            f"Input data changed during job {job.key}: expected "
            f"{job.data_sha256}, observed {after_sha}."
        )
    stats_path = run_dir / "training_stats.json"
    try:
        training_stats = json.loads(stats_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise TrainerConfigError(
            f"Malformed training statistics at {stats_path}: {exc}"
        ) from exc
    return {
        "status": "complete",
        "fold": job.fold,
        "condition": job.condition,
        "seed": job.seed,
        "config_path": str(job.config_path),
        "run_dir": str(run_dir.resolve()),
        "metrics": metrics,
        "training_stats": training_stats,
        "data_path": str(job.data_path),
        "data_sha256": job.data_sha256,
    }


def _write_source_runs(output_dir: Path, jobs: Mapping[str, Any]) -> None:
    """Write the public source-run index atomically.

    Args:
        output_dir: Study artifact directory.
        jobs: Manifest jobs mapping.
    """
    completed = {
        key: value for key, value in jobs.items() if _completed_job(value)
    }
    _atomic_write_json(output_dir / "source_runs.json", completed)


def _run_pending_jobs(
    jobs: Sequence[_TrainingJob],
    study: ScalingStudy,
    runs_root: Path,
    output_dir: Path,
    manifest: dict[str, Any],
) -> None:
    """Run incomplete jobs with the study's declared worker count.

    Args:
        jobs: Complete desired job matrix.
        study: Validated study.
        runs_root: Root for run artifacts.
        output_dir: Study artifact directory.
        manifest: Mutable resume manifest.
    """
    manifest_jobs = manifest["jobs"]
    pending = [job for job in jobs if not _completed_job(manifest_jobs.get(job.key))]
    if not pending:
        _write_source_runs(output_dir, manifest_jobs)
        return
    # Spawn avoids inheriting CUDA contexts and process-global Torch/NumPy RNG
    # state.  Independent jobs therefore remain independent when workers > 1.
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(
        max_workers=study.workers, mp_context=context
    ) as executor:
        futures: dict[Future[dict[str, Any]], _TrainingJob] = {
            executor.submit(_execute_job, job, runs_root): job for job in pending
        }
        for future in as_completed(futures):
            job = futures[future]
            try:
                entry = future.result()
            except Exception as exc:
                manifest_jobs[job.key] = {
                    "status": "failed",
                    "fold": job.fold,
                    "condition": job.condition,
                    "seed": job.seed,
                    "config_path": str(job.config_path),
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
                _atomic_write_json(output_dir / "manifest.json", manifest)
                _write_source_runs(output_dir, manifest_jobs)
                raise
            manifest_jobs[job.key] = entry
            _atomic_write_json(output_dir / "manifest.json", manifest)
            _write_source_runs(output_dir, manifest_jobs)


def _run_ensembles(
    groups: Mapping[tuple[str, str], list[str]],
    study: ScalingStudy,
    runs_root: Path,
    output_dir: Path,
    manifest: dict[str, Any],
) -> None:
    """Run three-seed action-mean ensembles when the study declares three seeds.

    Args:
        groups: Source job keys grouped by fold and condition.
        study: Validated study.
        runs_root: Root for ensemble artifacts.
        output_dir: Study output directory.
        manifest: Mutable resume manifest.
    """
    if len(study.seeds) != 3:
        manifest["ensemble_status"] = (
            f"not_run: ens3 requires exactly 3 seeds, study declares "
            f"{len(study.seeds)}"
        )
        _atomic_write_json(output_dir / "manifest.json", manifest)
        return
    ensembles = manifest["ensembles"]
    for (fold, condition), keys in groups.items():
        ensemble_key = f"{fold}|{condition}|ens3"
        existing = ensembles.get(ensemble_key)
        if isinstance(existing, Mapping) and existing.get("status") == "complete":
            directory = existing.get("run_dir")
            if isinstance(directory, str) and (Path(directory) / "metrics.json").is_file():
                continue
        member_dirs = [Path(manifest["jobs"][key]["run_dir"]) for key in keys]
        ensemble_dir, metrics = run_ensemble_evaluation(member_dirs, runs_root)
        ensembles[ensemble_key] = {
            "status": "complete",
            "fold": fold,
            "condition": condition,
            "members": keys,
            "run_dir": str(ensemble_dir.resolve()),
            "metrics": metrics,
        }
        _atomic_write_json(output_dir / "manifest.json", manifest)


def _resolve_data_path(raw: Mapping[str, Any]) -> Path:
    """Require a file-backed experiment and return its cache path.

    Args:
        raw: Experiment mapping.

    Returns:
        Existing cache path.

    Raises:
        TrainerConfigError: If the study cannot produce a file-data SHA.
    """
    data = raw["env"]["data"]
    if data["provider"] != "file":
        raise TrainerConfigError(
            "Data-scaling audits require env.data.provider: file so the exact "
            "source dataset can be identified by SHA-256."
        )
    path_value = data.get("path")
    if not isinstance(path_value, str) or not path_value:
        raise TrainerConfigError(
            "File-backed data-scaling audit requires env.data.path."
        )
    path = Path(path_value).resolve()
    if not path.is_file():
        raise TrainerConfigError(f"Audit data cache does not exist: {path}")
    return path


def _effective_rank(returns: np.ndarray) -> float:
    """Calculate participation-ratio rank of the return correlation matrix.

    Args:
        returns: Matrix shaped observations by symbols.

    Returns:
        Participation-ratio effective rank in ``[1, number of symbols]``.

    Raises:
        ValueError: If returns are malformed or correlation is non-finite.
    """
    if returns.ndim != 2 or returns.shape[0] < 2 or returns.shape[1] < 1:
        raise ValueError(
            f"returns must have shape (observations >= 2, symbols >= 1), got "
            f"{returns.shape}."
        )
    if returns.shape[1] == 1:
        if float(np.var(returns[:, 0])) == 0.0:
            raise ValueError("effective rank is undefined for a constant return series.")
        return 1.0
    correlation = np.asarray(np.corrcoef(returns, rowvar=False), dtype=np.float64)
    if not np.isfinite(correlation).all():
        raise ValueError("effective rank requires finite return correlations.")
    trace = float(np.trace(correlation))
    squared_trace = float(np.trace(correlation @ correlation))
    if squared_trace <= 0.0:
        raise ValueError("effective rank requires positive tr(R^2).")
    return trace * trace / squared_trace


def _audit_one(
    fold_path: Path, source_raw: Mapping[str, Any], condition: str
) -> dict[str, Any]:
    """Audit one fold-history condition against its exact file cache.

    Args:
        fold_path: Source fold path.
        source_raw: Source experiment mapping.
        condition: History condition.

    Returns:
        Flat audit row suitable for CSV and JSON.
    """
    seed = int(source_raw["run"]["seed"])
    device = str(source_raw["run"]["device"])
    raw = _materialize_raw(source_raw, condition, seed, device)
    data_path = _resolve_data_path(raw)
    data = raw["env"]["data"]
    environment = raw["env"]["environment"]
    symbols = tuple(str(symbol) for symbol in environment["currency_pairs"])
    frame = FileDataProvider(str(data_path)).get_data(
        symbols,
        str(raw["train_range"]["start"]),
        str(raw["train_range"]["end"]),
        str(data["timeframe"]),
    )
    bars = len(frame.index)
    if bars < 2:
        raise TrainerConfigError(
            f"Audit fold {fold_path} condition {condition} has fewer than two bars."
        )
    span_seconds = float((frame.index[-1] - frame.index[0]).total_seconds())
    years = span_seconds / _SECONDS_PER_YEAR
    warmup = int(raw["env"]["features"]["volatility_window"])
    window_size = int(environment["window_size"])
    episode_max_steps = int(environment["episode_max_steps"])
    usable_rows = bars - warmup
    usable_transitions = usable_rows - window_size
    if usable_transitions < 1:
        raise TrainerConfigError(
            f"Audit fold {fold_path} condition {condition} has "
            f"usable_transitions={usable_transitions}; at least one is required."
        )
    candidate_starts = max(
        1, usable_rows - episode_max_steps - window_size + 1
    )
    closes = frame.xs("Close", axis=1, level=1).loc[:, list(symbols)]
    log_returns = np.log(closes / closes.shift(1)).iloc[1:]
    return_matrix = log_returns.to_numpy(dtype=np.float64)
    lag_limit = min(252, len(log_returns) - 1)
    if lag_limit < 1:
        raise TrainerConfigError(
            f"Audit fold {fold_path} condition {condition} has too few returns for ESS."
        )
    return_ess = [
        effective_sample_size(return_matrix[:, index], lag_limit)
        for index in range(return_matrix.shape[1])
    ]
    absolute_return_ess = [
        effective_sample_size(np.abs(return_matrix[:, index]), lag_limit)
        for index in range(return_matrix.shape[1])
    ]
    squared_return_ess = [
        effective_sample_size(np.square(return_matrix[:, index]), lag_limit)
        for index in range(return_matrix.shape[1])
    ]
    momentum = pd.DataFrame(
        {
            symbol: mom24(frame.xs(symbol, axis=1, level=0))
            for symbol in symbols
        }
    ).iloc[24:]
    momentum_matrix = momentum.to_numpy(dtype=np.float64)
    momentum_lag_limit = min(252, len(momentum) - 1)
    if momentum_lag_limit < 1:
        raise TrainerConfigError(
            f"Audit fold {fold_path} condition {condition} has too few rows "
            "for mom24 ESS."
        )
    momentum_ess = [
        effective_sample_size(momentum_matrix[:, index], momentum_lag_limit)
        for index in range(momentum_matrix.shape[1])
    ]
    rank_ess_median: float | None = None
    if len(symbols) > 1:
        ranked = xr_mom24(frame, symbols).iloc[24:]
        ranked_matrix = ranked.to_numpy(dtype=np.float64)
        rank_ess = [
            effective_sample_size(ranked_matrix[:, index], momentum_lag_limit)
            for index in range(ranked_matrix.shape[1])
        ]
        rank_ess_median = float(np.median(np.asarray(rank_ess)))
    carry_columns = [(symbol, "CarryAnnual") for symbol in symbols]
    carry_available = all(column in frame.columns for column in carry_columns)
    selected_features = set(raw["env"]["features"]["selected"])
    if "carry_annual" in selected_features and not carry_available:
        raise TrainerConfigError(
            f"Audit fold {fold_path} selects carry_annual but {data_path} does "
            "not contain CarryAnnual for every symbol."
        )
    carry_update_events = 0
    carry_updated_values = 0
    if carry_available:
        carry_values = frame.loc[:, carry_columns].to_numpy(dtype=np.float64)
        carry_changes = np.not_equal(carry_values[1:], carry_values[:-1])
        carry_update_events = int(np.any(carry_changes, axis=1).sum())
        carry_updated_values = int(carry_changes.sum())
    effective_rank = _effective_rank(return_matrix)
    hyperparams = raw["algorithm"]["hyperparams"]
    if raw["algorithm"]["name"] != "ppo":
        raise TrainerConfigError(
            f"Data-scaling rollout accounting requires PPO, got "
            f"{raw['algorithm']['name']!r} in {fold_path}."
        )
    for required in ("n_steps", "batch_size"):
        if required not in hyperparams:
            raise TrainerConfigError(
                f"PPO data-scaling audit requires algorithm.hyperparams.{required} "
                f"in {fold_path}."
            )
    n_epochs_value = hyperparams.get("n_epochs", _PPO_DEFAULT_N_EPOCHS)
    accounting = rollout_accounting(
        requested_steps=int(raw["run"]["total_timesteps"]),
        n_steps=int(hyperparams["n_steps"]),
        n_envs=int(raw["run"]["n_envs"]),
        episode_max_steps=episode_max_steps,
        batch_size=int(hyperparams["batch_size"]),
        n_epochs=int(n_epochs_value),
    )
    median_ess = float(np.median(np.asarray(return_ess)))
    return {
        "fold": str(fold_path),
        "condition": condition,
        "train_start": raw["train_range"]["start"],
        "train_end": raw["train_range"]["end"],
        "data_path": str(data_path),
        "data_sha256": _sha256_file(data_path),
        "symbols": len(symbols),
        "bars": bars,
        "years": years,
        "usable_transitions": usable_transitions,
        "candidate_episode_starts": candidate_starts,
        "requested_steps": accounting.requested_steps,
        "actual_steps": accounting.actual_steps,
        "rollouts": accounting.rollouts,
        "episode_equivalents": accounting.episode_equivalents,
        "optimizer_minibatch_steps": accounting.optimizer_minibatch_steps,
        "sample_presentations": accounting.sample_presentations,
        "transition_reuse": accounting.actual_steps / usable_transitions,
        "candidate_start_reuse": accounting.episode_equivalents / candidate_starts,
        "ess_max_lag": 252,
        "return_ess_min": float(min(return_ess)),
        "return_ess_median": median_ess,
        "return_ess_max": float(max(return_ess)),
        "absolute_return_ess_median": float(
            np.median(np.asarray(absolute_return_ess))
        ),
        "squared_return_ess_median": float(
            np.median(np.asarray(squared_return_ess))
        ),
        "mom24_ess_median": float(np.median(np.asarray(momentum_ess))),
        "xr_mom24_ess_median": rank_ess_median,
        "carry_available": carry_available,
        "carry_update_events": carry_update_events,
        "carry_updated_values": carry_updated_values,
        "effective_rank": effective_rank,
        "effective_independent_samples": median_ess * effective_rank,
    }


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    """Write homogeneous dictionaries as CSV atomically.

    Args:
        path: Destination CSV path.
        rows: Non-empty row sequence.

    Raises:
        TrainerConfigError: If rows are empty or have inconsistent keys.
    """
    if not rows:
        raise TrainerConfigError(f"Cannot write empty CSV report: {path}")
    fieldnames = list(rows[0])
    expected = set(fieldnames)
    for index, row in enumerate(rows):
        if set(row) != expected:
            raise TrainerConfigError(
                f"CSV row {index} for {path} has inconsistent columns."
            )
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _audit_requests(study: ScalingStudy) -> list[tuple[Path, str]]:
    """Build the protocol audit matrix without impossible early fixed windows.

    Every audit fold receives the expanding condition. Fixed-history conditions
    apply only to folds selected for scaling experiments.

    Args:
        study: Validated study definition.

    Returns:
        Ordered ``(fold, condition)`` audit requests.
    """
    scaling_folds = set(study.fold_configs)
    requests: list[tuple[Path, str]] = []
    for fold_path in study.audit_fold_configs:
        if fold_path in scaling_folds:
            requests.extend(
                (fold_path, condition)
                for condition in _condition_names(study)
                if condition != "expanding"
            )
        requests.append((fold_path, "expanding"))
    return requests


def _result_rows(manifest: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Flatten completed source and ensemble metrics for CSV output.

    Args:
        manifest: Completed study manifest.

    Returns:
        Homogeneous result rows.
    """
    rows: list[dict[str, Any]] = []
    metric_names: set[str] = set()
    training_stat_names: set[str] = set()
    entries: list[tuple[str, Mapping[str, Any]]] = []
    for value in manifest["jobs"].values():
        if _completed_job(value):
            entries.append(("seed", value))
            metric_names.update(value["metrics"])
            training_stat_names.update(value["training_stats"])
    for value in manifest["ensembles"].values():
        if isinstance(value, Mapping) and value.get("status") == "complete":
            entries.append(("ens3", value))
            metric_names.update(value["metrics"])
    ordered_metrics = sorted(metric_names)
    ordered_training_stats = sorted(training_stat_names)
    for kind, value in entries:
        metrics = value["metrics"]
        row: dict[str, Any] = {
            "fold": value["fold"],
            "condition": value["condition"],
            "result_kind": kind,
            "seed": value.get("seed", ""),
            "run_dir": value["run_dir"],
        }
        row.update({name: metrics.get(name, "") for name in ordered_metrics})
        stats = value.get("training_stats", {})
        row.update(
            {
                f"training_{name}": stats.get(name, "")
                for name in ordered_training_stats
            }
        )
        rows.append(row)
    rows.sort(key=lambda row: (row["fold"], row["condition"], row["result_kind"], str(row["seed"])))
    return rows


def _bootstrap_summary(
    result_rows: Sequence[Mapping[str, Any]], study: ScalingStudy
) -> dict[str, dict[str, Any]]:
    """Summarize net returns, seed variance, and ens3 by condition.

    Args:
        result_rows: Flat result rows.
        study: Validated study containing bootstrap controls.

    Returns:
        Mapping from condition to primary net-return statistics and diagnostics.

    Raises:
        TrainerConfigError: If a condition has no finite seed net returns.
    """
    summary: dict[str, dict[str, Any]] = {}
    for condition in _condition_names(study):
        seed_rows = [
            row
            for row in result_rows
            if row["condition"] == condition and row["result_kind"] == "seed"
        ]
        log_returns = np.asarray(
            [float(row["cumulative_log_return"]) for row in seed_rows],
            dtype=np.float64,
        )
        eval_years = np.asarray(
            [
                (
                    datetime.fromisoformat(str(row["eval_end"]))
                    - datetime.fromisoformat(str(row["eval_start"]))
                ).total_seconds()
                / _SECONDS_PER_YEAR
                for row in seed_rows
            ],
            dtype=np.float64,
        )
        if (
            len(log_returns) == 0
            or not np.isfinite(log_returns).all()
            or not np.isfinite(eval_years).all()
            or bool(np.any(eval_years <= 0.0))
        ):
            raise TrainerConfigError(
                f"Condition {condition} has no complete finite seed net returns."
            )
        fold_log_returns: list[float] = []
        fold_years: list[float] = []
        fold_seed_standard_deviations: list[float] = []
        for fold in sorted({str(row["fold"]) for row in seed_rows}):
            fold_indices = [
                index
                for index, row in enumerate(seed_rows)
                if str(row["fold"]) == fold
            ]
            if len(fold_indices) != len(study.seeds):
                raise TrainerConfigError(
                    f"Condition {condition} fold {fold} has {len(fold_indices)} "
                    f"seed results, expected {len(study.seeds)}."
                )
            durations = eval_years[fold_indices]
            if not np.allclose(durations, durations[0], rtol=0.0, atol=1e-12):
                raise TrainerConfigError(
                    f"Condition {condition} fold {fold} has inconsistent eval durations."
                )
            fold_values = log_returns[fold_indices]
            fold_log_returns.append(float(fold_values.mean()))
            fold_years.append(float(durations[0]))
            fold_seed_standard_deviations.append(
                float(np.std(fold_values, ddof=1))
            )
        intervals = bootstrap_annualized_fold_returns(
            np.asarray(fold_log_returns, dtype=np.float64),
            np.asarray(fold_years, dtype=np.float64),
            study.bootstrap_samples,
            study.bootstrap_seed,
            block_length=min(2, len(fold_log_returns)),
        )
        seed_annualized: list[float] = []
        for seed in study.seeds:
            indices = [
                index for index, row in enumerate(seed_rows) if row["seed"] == seed
            ]
            if indices:
                seed_annualized.append(
                    float(
                        np.expm1(
                            log_returns[indices].sum() / eval_years[indices].sum()
                        )
                    )
                )
        seed_variance = (
            float(np.var(np.asarray(seed_annualized), ddof=1))
            if len(seed_annualized) > 1
            else 0.0
        )
        ensemble_rows = [
            row
            for row in result_rows
            if row["condition"] == condition and row["result_kind"] == "ens3"
        ]
        ensemble_annualized: float | None = None
        if ensemble_rows:
            ensemble_logs = np.asarray(
                [float(row["cumulative_log_return"]) for row in ensemble_rows]
            )
            ensemble_years = np.asarray(
                [
                    (
                        datetime.fromisoformat(str(row["eval_end"]))
                        - datetime.fromisoformat(str(row["eval_start"]))
                    ).total_seconds()
                    / _SECONDS_PER_YEAR
                    for row in ensemble_rows
                ]
            )
            ensemble_annualized = float(
                np.expm1(ensemble_logs.sum() / ensemble_years.sum())
            )
        summary[condition] = {
            "seed_fold_observations": int(len(log_returns)),
            "mean_cumulative_net_log_return": float(log_returns.mean()),
            "annualized_net_return": float(
                np.expm1(log_returns.sum() / eval_years.sum())
            ),
            "annualized_net_return_bootstrap_95_low": float(
                intervals.fold_low
            ),
            "annualized_net_return_bootstrap_95_high": float(
                intervals.fold_high
            ),
            "annualized_net_return_moving_block_95_low": (
                intervals.moving_block_low
            ),
            "annualized_net_return_moving_block_95_high": (
                intervals.moving_block_high
            ),
            "seed_annualized_net_returns": seed_annualized,
            "seed_variance": seed_variance,
            "mean_fold_seed_standard_deviation": float(
                np.mean(np.asarray(fold_seed_standard_deviations))
            ),
            "median_fold_seed_standard_deviation": float(
                np.median(np.asarray(fold_seed_standard_deviations))
            ),
            "ens3_annualized_net_return": ensemble_annualized,
            "mean_sharpe_annualized": float(
                np.mean(
                    np.asarray(
                        [float(row["sharpe_annualized"]) for row in seed_rows]
                    )
                )
            ),
        }
    return summary


def _paired_differences(
    result_rows: Sequence[Mapping[str, Any]], study: ScalingStudy
) -> list[dict[str, Any]]:
    """Compute adjacent-condition paired net-log-return differences.

    Args:
        result_rows: Flat result rows.
        study: Validated study containing condition order and bootstrap controls.

    Returns:
        One paired summary for every adjacent condition pair.

    Raises:
        TrainerConfigError: If adjacent conditions do not have matching pairs.
    """
    lookup = {
        (str(row["fold"]), int(row["seed"]), str(row["condition"])): float(
            row["cumulative_log_return"]
        )
        for row in result_rows
        if row["result_kind"] == "seed"
    }
    rng = np.random.default_rng(study.bootstrap_seed)
    output: list[dict[str, Any]] = []
    conditions = _condition_names(study)
    for baseline, larger in zip(conditions[:-1], conditions[1:]):
        baseline_keys = {
            (fold, seed)
            for fold, seed, condition in lookup
            if condition == baseline
        }
        larger_keys = {
            (fold, seed) for fold, seed, condition in lookup if condition == larger
        }
        paired_keys = sorted(baseline_keys & larger_keys)
        if not paired_keys:
            raise TrainerConfigError(
                f"No paired results exist for adjacent conditions {baseline} and "
                f"{larger}."
            )
        differences = np.asarray(
            [
                lookup[(fold, seed, larger)] - lookup[(fold, seed, baseline)]
                for fold, seed in paired_keys
            ],
            dtype=np.float64,
        )
        draw_indices = rng.choice(
            len(differences),
            size=(study.bootstrap_samples, len(differences)),
            replace=True,
        )
        draw_means = differences[draw_indices].mean(axis=1)
        output.append(
            {
                "baseline": baseline,
                "larger": larger,
                "pairs": len(differences),
                "mean_cumulative_net_log_return_difference": float(
                    differences.mean()
                ),
                "bootstrap_95_low": float(np.quantile(draw_means, 0.025)),
                "bootstrap_95_high": float(np.quantile(draw_means, 0.975)),
            }
        )
    return output


def _report_markdown(report: Mapping[str, Any]) -> str:
    """Render the compact human-readable scaling report.

    Args:
        report: Complete report dictionary.

    Returns:
        Markdown document.
    """
    lines = [
        f"# {report['name']}",
        "",
        "![Scaling curve](scaling_curve.svg)",
        "",
        "## Net-return scaling summary",
        "",
        "| Condition | Observations | Mean net log return | Annualized net return | Fold bootstrap 95% CI | 2-fold block 95% CI | Median fold seed SD | ens3 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for condition in report["conditions"]:
        row = report["summary"][condition]
        lines.append(
            f"| {condition} | {row['seed_fold_observations']} | "
            f"{row['mean_cumulative_net_log_return']:.6f} | "
            f"{row['annualized_net_return']:.6f} | "
            f"[{row['annualized_net_return_bootstrap_95_low']:.6f}, "
            f"{row['annualized_net_return_bootstrap_95_high']:.6f}] | "
            f"[{row['annualized_net_return_moving_block_95_low']:.6f}, "
            f"{row['annualized_net_return_moving_block_95_high']:.6f}] | "
            f"{row['median_fold_seed_standard_deviation']:.6f} | "
            f"{row['ens3_annualized_net_return']} |"
        )
    lines.extend(["", "## Adjacent paired differences", ""])
    for paired in report["paired_differences"]:
        lines.append(
            f"- {paired['baseline']} → {paired['larger']}: "
            f"mean Δ log return {paired['mean_cumulative_net_log_return_difference']:.6f} "
            f"(n={paired['pairs']}, 95% CI "
            f"[{paired['bootstrap_95_low']:.6f}, "
            f"{paired['bootstrap_95_high']:.6f}])"
        )
    lines.extend(
        [
            "",
            "## Reproducibility",
            "",
            f"- Study SHA-256: `{report['study_sha256']}`",
            f"- Completed source runs: {report['completed_source_runs']}",
            f"- Completed ensembles: {report['completed_ensembles']}",
            f"- Audit rows: {report['audit_rows']}",
            "",
        ]
    )
    return "\n".join(lines)


def write_scaling_curve_svg(
    report: Mapping[str, Any],
    audit_rows: Sequence[Mapping[str, Any]],
    output_path: Path,
) -> None:
    """Write net-return and seed-dispersion curves against unique history.

    Args:
        report: Completed scaling report containing condition summaries.
        audit_rows: Audit rows for the folds used in the scaling experiment.
        output_path: Destination SVG path.

    Raises:
        TrainerConfigError: If a condition lacks numeric summary or audit data.
    """
    conditions_value = report.get("conditions")
    summary_value = report.get("summary")
    if not isinstance(conditions_value, list) or not isinstance(summary_value, Mapping):
        raise TrainerConfigError("Scaling curve requires report conditions and summary.")
    points: list[dict[str, float | str]] = []
    for condition_value in conditions_value:
        condition = str(condition_value)
        condition_audits = [
            row for row in audit_rows if str(row.get("condition")) == condition
        ]
        if not condition_audits:
            raise TrainerConfigError(
                f"Scaling curve lacks audit rows for condition {condition}."
            )
        summary = summary_value.get(condition)
        if not isinstance(summary, Mapping):
            raise TrainerConfigError(
                f"Scaling curve lacks summary for condition {condition}."
            )
        try:
            point = {
                "condition": condition,
                "years": float(
                    np.median([float(row["years"]) for row in condition_audits])
                ),
                "bars": float(
                    np.median([float(row["bars"]) for row in condition_audits])
                ),
                "net": float(summary["annualized_net_return"]),
                "low": float(summary["annualized_net_return_bootstrap_95_low"]),
                "high": float(summary["annualized_net_return_bootstrap_95_high"]),
                "seed_sd": float(summary["median_fold_seed_standard_deviation"]),
            }
        except (KeyError, TypeError, ValueError) as exc:
            raise TrainerConfigError(
                f"Scaling curve has non-numeric data for condition {condition}."
            ) from exc
        numeric = [
            float(point[name])
            for name in ("years", "bars", "net", "low", "high", "seed_sd")
        ]
        if not np.isfinite(np.asarray(numeric)).all():
            raise TrainerConfigError(
                f"Scaling curve has non-finite data for condition {condition}."
            )
        points.append(point)
    points.sort(key=lambda point: float(point["years"]))

    width = 900.0
    height = 720.0
    left = 92.0
    right = 850.0
    top_first = 70.0
    bottom_first = 350.0
    top_second = 425.0
    bottom_second = 650.0
    years = np.asarray([float(point["years"]) for point in points])
    x_min = float(years.min())
    x_max = float(years.max())
    if x_min == x_max:
        x_min -= 0.5
        x_max += 0.5
    else:
        padding = 0.05 * (x_max - x_min)
        x_min -= padding
        x_max += padding
    return_values = np.asarray(
        [
            value
            for point in points
            for value in (
                float(point["low"]),
                float(point["net"]),
                float(point["high"]),
            )
        ]
        + [0.0]
    )
    y_min = float(return_values.min())
    y_max = float(return_values.max())
    if y_min == y_max:
        y_min -= 0.01
        y_max += 0.01
    else:
        padding = 0.08 * (y_max - y_min)
        y_min -= padding
        y_max += padding
    seed_max = max(float(point["seed_sd"]) for point in points)
    seed_max = max(seed_max * 1.12, 0.01)

    def x_coordinate(value: float) -> float:
        """Map a history duration to the shared horizontal axis."""
        return left + (value - x_min) / (x_max - x_min) * (right - left)

    def return_coordinate(value: float) -> float:
        """Map annualized return to the first panel."""
        return bottom_first - (value - y_min) / (y_max - y_min) * (
            bottom_first - top_first
        )

    def seed_coordinate(value: float) -> float:
        """Map seed standard deviation to the second panel."""
        return bottom_second - value / seed_max * (bottom_second - top_second)

    lines = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="900" height="720" viewBox="0 0 900 720">',
        '<rect width="900" height="720" fill="white"/>',
        '<style>text{font-family:system-ui,sans-serif;fill:#1f2937} .axis{stroke:#64748b;stroke-width:1} .grid{stroke:#e2e8f0;stroke-width:1} .series{fill:none;stroke:#2563eb;stroke-width:2.5} .seed{fill:none;stroke:#d97706;stroke-width:2.5}</style>',
        '<text x="450" y="28" text-anchor="middle" font-size="18" font-weight="600">longf data-scaling generalization</text>',
        f'<rect x="{left}" y="{top_first}" width="{right-left}" height="{bottom_first-top_first}" fill="none" class="axis"/>',
        f'<rect x="{left}" y="{top_second}" width="{right-left}" height="{bottom_second-top_second}" fill="none" class="axis"/>',
        '<text x="20" y="210" text-anchor="middle" font-size="13" transform="rotate(-90 20 210)">Annualized OOS net return</text>',
        '<text x="20" y="538" text-anchor="middle" font-size="13" transform="rotate(-90 20 538)">Median fold seed SD (log return)</text>',
        '<text x="471" y="704" text-anchor="middle" font-size="13">Unique training history (years)</text>',
    ]
    for tick in np.linspace(y_min, y_max, 5):
        coordinate = return_coordinate(float(tick))
        lines.extend(
            [
                f'<line x1="{left}" y1="{coordinate:.2f}" x2="{right}" y2="{coordinate:.2f}" class="grid"/>',
                f'<text x="{left-8}" y="{coordinate+4:.2f}" text-anchor="end" font-size="11">{tick:.1%}</text>',
            ]
        )
    for tick in np.linspace(0.0, seed_max, 4):
        coordinate = seed_coordinate(float(tick))
        lines.extend(
            [
                f'<line x1="{left}" y1="{coordinate:.2f}" x2="{right}" y2="{coordinate:.2f}" class="grid"/>',
                f'<text x="{left-8}" y="{coordinate+4:.2f}" text-anchor="end" font-size="11">{tick:.3f}</text>',
            ]
        )
    zero_coordinate = return_coordinate(0.0)
    lines.append(
        f'<line x1="{left}" y1="{zero_coordinate:.2f}" x2="{right}" y2="{zero_coordinate:.2f}" stroke="#475569" stroke-dasharray="5 4"/>'
    )
    return_path = " ".join(
        f'{"M" if index == 0 else "L"}{x_coordinate(float(point["years"])):.2f},{return_coordinate(float(point["net"])):.2f}'
        for index, point in enumerate(points)
    )
    seed_path = " ".join(
        f'{"M" if index == 0 else "L"}{x_coordinate(float(point["years"])):.2f},{seed_coordinate(float(point["seed_sd"])):.2f}'
        for index, point in enumerate(points)
    )
    lines.extend(
        [f'<path d="{return_path}" class="series"/>', f'<path d="{seed_path}" class="seed"/>']
    )
    for point in points:
        x_value = x_coordinate(float(point["years"]))
        net_value = return_coordinate(float(point["net"]))
        low_value = return_coordinate(float(point["low"]))
        high_value = return_coordinate(float(point["high"]))
        seed_value = seed_coordinate(float(point["seed_sd"]))
        label = html.escape(str(point["condition"]))
        lines.extend(
            [
                f'<line x1="{x_value:.2f}" y1="{high_value:.2f}" x2="{x_value:.2f}" y2="{low_value:.2f}" stroke="#2563eb" stroke-width="1.5"/>',
                f'<line x1="{x_value-5:.2f}" y1="{high_value:.2f}" x2="{x_value+5:.2f}" y2="{high_value:.2f}" stroke="#2563eb"/>',
                f'<line x1="{x_value-5:.2f}" y1="{low_value:.2f}" x2="{x_value+5:.2f}" y2="{low_value:.2f}" stroke="#2563eb"/>',
                f'<circle cx="{x_value:.2f}" cy="{net_value:.2f}" r="4.5" fill="#2563eb"/>',
                f'<circle cx="{x_value:.2f}" cy="{seed_value:.2f}" r="4.5" fill="#d97706"/>',
                f'<text x="{x_value:.2f}" y="{bottom_second+18:.2f}" text-anchor="middle" font-size="11">{float(point["years"]):.1f}y</text>',
                f'<text x="{x_value:.2f}" y="{bottom_second+34:.2f}" text-anchor="middle" font-size="10">{label} · {float(point["bars"]):.0f} bars</text>',
            ]
        )
    lines.append("</svg>")
    _atomic_write_text(output_path, "\n".join(lines) + "\n")


def run_data_scaling_study(
    study_path: Path, runs_root: Path, output_dir: Path
) -> dict[str, Any]:
    """Run or resume a complete fixed-versus-expanding data-scaling study.

    Args:
        study_path: Strict study YAML path.
        runs_root: Root directory for individual and ensemble run artifacts.
        output_dir: Durable, resumable study artifact directory.

    Returns:
        JSON-serializable final report.
    """
    study = load_scaling_study(study_path)
    _validate_shared_protocol(study)
    output_dir.mkdir(parents=True, exist_ok=True)
    runs_root.mkdir(parents=True, exist_ok=True)
    manifest = _load_or_create_manifest(study, output_dir)
    jobs, groups = _materialize_jobs(study, output_dir)
    _run_pending_jobs(jobs, study, runs_root, output_dir, manifest)
    _run_ensembles(groups, study, runs_root, output_dir, manifest)

    audit_rows: list[dict[str, Any]] = []
    for fold_path, condition in _audit_requests(study):
        _, source_raw = load_experiment_config(fold_path)
        audit_rows.append(_audit_one(fold_path, source_raw, condition))
    _write_csv(output_dir / "data_audit.csv", audit_rows)

    result_rows = _result_rows(manifest)
    _write_csv(output_dir / "scaling_results.csv", result_rows)
    summary = _bootstrap_summary(result_rows, study)
    paired_differences = _paired_differences(result_rows, study)
    report: dict[str, Any] = {
        "name": study.name,
        "conditions": list(_condition_names(study)),
        "study_path": str(study.source_path),
        "study_sha256": manifest["study_sha256"],
        "workers": study.workers,
        "bootstrap_samples": study.bootstrap_samples,
        "bootstrap_seed": study.bootstrap_seed,
        "summary": summary,
        "paired_differences": paired_differences,
        "completed_source_runs": len(
            [value for value in manifest["jobs"].values() if _completed_job(value)]
        ),
        "completed_ensembles": len(
            [
                value
                for value in manifest["ensembles"].values()
                if isinstance(value, Mapping) and value.get("status") == "complete"
            ]
        ),
        "ensemble_status": manifest.get("ensemble_status", "complete"),
        "audit_rows": len(audit_rows),
    }
    _atomic_write_json(output_dir / "report.json", report)
    scaling_folds = {str(path) for path in study.fold_configs}
    scaling_audit_rows = [
        row for row in audit_rows if str(row["fold"]) in scaling_folds
    ]
    write_scaling_curve_svg(
        report,
        scaling_audit_rows,
        output_dir / "scaling_curve.svg",
    )
    _atomic_write_text(output_dir / "report.md", _report_markdown(report))
    manifest["status"] = "complete"
    _atomic_write_json(output_dir / "manifest.json", manifest)
    return report


def main(argv: list[str] | None = None) -> int:
    """Run the data-scaling study CLI.

    Args:
        argv: Explicit arguments for tests; None reads the process arguments.

    Returns:
        Process exit code.
    """
    parser = argparse.ArgumentParser(
        prog="forex-data-scaling",
        description="Run or resume a controlled RL data-scaling study.",
    )
    parser.add_argument("--study", required=True, help="Scaling study YAML path.")
    parser.add_argument("--runs-root", required=True, help="Run artifact root.")
    parser.add_argument("--output-dir", required=True, help="Study artifact directory.")
    args = parser.parse_args(argv)
    try:
        report = run_data_scaling_study(
            Path(args.study), Path(args.runs_root), Path(args.output_dir)
        )
    except (TrainerConfigError, ConfigError, DataError, FeatureError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
