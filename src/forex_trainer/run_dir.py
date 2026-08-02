"""Run directory creation and immutable provenance capture (ADR-0011)."""

from __future__ import annotations

import copy
import hashlib
import json
import subprocess
from datetime import date, datetime, timezone
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any, Mapping

import forex_env
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import yaml
from forex_env.data.file_provider import load_ohlcv_parquet
from forex_env.errors import DataError

from .config import ExperimentConfig, TrainerConfigError

_TRAINER_REPO_ROOT = Path(__file__).resolve().parents[2]
_ENV_REPO_ROOT = Path(forex_env.__file__).resolve().parents[2]
_RUN_PROVENANCE_CONTRACT_VERSION = 1
_REQUIRED_PACKAGES = (
    "forex-env-v3",
    "stable-baselines3",
    "sb3-contrib",
    "torch",
    "gymnasium",
)
_CACHE_METADATA_KEYS = {
    "forex_env_timeframe": "timeframe",
    "forex_env_start_date": "declared_start",
    "forex_env_end_date": "declared_end",
    "forex_env_cache_schema_version": "schema_version",
    "forex_env_carry_contract": "carry_contract",
}


def create_run_dir(runs_root: Path, experiment: str) -> Path:
    """Create a fresh run directory runs_root/<experiment>/<UTC timestamp>/.

    Args:
        runs_root: Root directory for all runs.
        experiment: Experiment name (validated by config parsing).

    Returns:
        Path to the created directory.
    """
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S-%f")
    run_dir = runs_root / experiment / timestamp
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def resolve_file_data_path(
    env_config: Mapping[str, Any], working_directory: Path
) -> dict[str, Any]:
    """Resolve a file-provider path relative to the launch directory.

    Args:
        env_config: Resolved forex-env configuration.
        working_directory: Training process working directory, preserving the
            existing relative-path contract.

    Returns:
        A deep copy with an absolute file-provider path. Non-file providers
        are returned as an unchanged deep copy.

    Raises:
        TrainerConfigError: If file-provider path is missing or not a string.
    """
    resolved = copy.deepcopy(dict(env_config))
    data = resolved.get("data")
    if not isinstance(data, Mapping):
        raise TrainerConfigError("Resolved environment data must be a mapping.")
    resolved_data = dict(data)
    resolved["data"] = resolved_data
    if resolved_data.get("provider") != "file":
        return resolved
    raw_path = resolved_data.get("path")
    if not isinstance(raw_path, str) or not raw_path:
        raise TrainerConfigError(
            f"File-provider data.path must be a non-empty string, got {raw_path!r}."
        )
    path = Path(raw_path)
    if not path.is_absolute():
        path = working_directory.resolve() / path
    resolved_data["path"] = str(path.resolve())
    return resolved


def _sha256_file(path: Path) -> str:
    """Hash a file without loading it entirely into memory.

    Args:
        path: File to hash.

    Returns:
        Lowercase SHA-256 hexadecimal digest.

    Raises:
        TrainerConfigError: If the file cannot be read.
    """
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            while chunk := stream.read(1024 * 1024):
                digest.update(chunk)
    except OSError as exc:
        raise TrainerConfigError(f"Failed to hash data cache {path}: {exc}") from exc
    return digest.hexdigest()


def _decode_metadata(metadata: Mapping[bytes, bytes], path: Path) -> dict[str, str]:
    """Decode provenance-relevant Parquet metadata.

    Args:
        metadata: Raw Arrow schema metadata.
        path: Cache path used in diagnostics.

    Returns:
        UTF-8 metadata whose keys belong to forex-env or forex-trainer.

    Raises:
        TrainerConfigError: If relevant metadata is not valid UTF-8.
    """
    decoded: dict[str, str] = {}
    for raw_key, raw_value in metadata.items():
        try:
            key = raw_key.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise TrainerConfigError(
                f"Cache {path} has a non-UTF-8 metadata key."
            ) from exc
        if not (key.startswith("forex_env_") or key.startswith("forex_trainer_")):
            continue
        try:
            decoded[key] = raw_value.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise TrainerConfigError(
                f"Cache {path} metadata {key!r} is not valid UTF-8."
            ) from exc
    return decoded


def _requested_coverage(data: Mapping[str, Any]) -> dict[str, str]:
    """Extract a split's requested date coverage.

    Args:
        data: Resolved data section.

    Returns:
        Start and end date strings.

    Raises:
        TrainerConfigError: If either boundary is missing or not a string.
    """
    coverage: dict[str, str] = {}
    for key, label in (("start_date", "start"), ("end_date", "end")):
        value = data.get(key)
        if not isinstance(value, str) or not value:
            raise TrainerConfigError(
                f"Resolved environment data.{key} must be a non-empty string."
            )
        coverage[label] = value
    return coverage


def _capture_file_data_provenance(
    data: Mapping[str, Any], path: Path
) -> dict[str, Any]:
    """Capture immutable provenance for one Parquet cache.

    Args:
        data: Resolved file-provider data section.
        path: Absolute cache path.

    Returns:
        JSON-serializable provenance mapping.

    Raises:
        TrainerConfigError: If the cache or its contract is unreadable.
    """
    if not path.is_file():
        raise TrainerConfigError(f"Data cache file not found: {path}")
    sha256_before = _sha256_file(path)
    try:
        cache = load_ohlcv_parquet(path)
        schema = pq.read_schema(path)
    except (DataError, pa.ArrowException, OSError) as exc:
        raise TrainerConfigError(f"Failed to validate data cache {path}: {exc}") from exc
    metadata = _decode_metadata(dict(schema.metadata or {}), path)
    missing = sorted(set(_CACHE_METADATA_KEYS) - set(metadata))
    if missing:
        raise TrainerConfigError(
            f"Cache {path} lacks required provenance metadata {missing}; regenerate "
            "it with the current forex-env cache writer before training."
        )
    frame = cache.data
    if frame.empty:
        raise TrainerConfigError(f"Cache {path} contains no rows.")
    if not isinstance(frame.index, pd.DatetimeIndex):
        raise TrainerConfigError(
            f"Cache {path} must have a DatetimeIndex, got "
            f"{type(frame.index).__name__}."
        )
    if frame.index.hasnans:
        raise TrainerConfigError(f"Cache {path} index contains NaT values.")
    cache_contract = {
        output_key: metadata[input_key]
        for input_key, output_key in _CACHE_METADATA_KEYS.items()
    }
    augmentation_metadata = {
        key: value
        for key, value in sorted(metadata.items())
        if key.startswith("forex_trainer_")
    }
    timeframe = data.get("timeframe")
    if timeframe != cache.timeframe:
        raise TrainerConfigError(
            f"Data cache {path} timeframe mismatch: cache {cache.timeframe!r}, "
            f"requested {timeframe!r}."
        )
    requested = _requested_coverage(data)
    if date.fromisoformat(requested["start"]) < date.fromisoformat(cache.start_date):
        raise TrainerConfigError(
            f"Data cache {path} does not cover requested start {requested['start']}; "
            f"declared start is {cache.start_date}."
        )
    if date.fromisoformat(requested["end"]) > date.fromisoformat(cache.end_date):
        raise TrainerConfigError(
            f"Data cache {path} does not cover requested end {requested['end']}; "
            f"declared end is {cache.end_date}."
        )
    sha256_after = _sha256_file(path)
    if sha256_before != sha256_after:
        raise TrainerConfigError(
            f"Data cache {path} changed while provenance was being captured; "
            "stop the writer and retry."
        )
    return {
        "provider": "file",
        "path": str(path),
        "sha256": sha256_after,
        "requested_coverage": requested,
        "actual_coverage": {
            "start": pd.Timestamp(frame.index.min()).isoformat(),
            "end": pd.Timestamp(frame.index.max()).isoformat(),
            "rows": int(len(frame.index)),
        },
        "cache_contract": cache_contract,
        "augmentation_metadata": augmentation_metadata,
    }


def _canonical_sha256(value: Mapping[str, Any]) -> str:
    """Hash a mapping's canonical JSON representation.

    Args:
        value: JSON-compatible mapping.

    Returns:
        Lowercase SHA-256 hexadecimal digest.
    """
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def capture_data_provenance(env_config: Mapping[str, Any]) -> dict[str, Any]:
    """Capture immutable data provenance for one resolved environment split.

    Args:
        env_config: Resolved forex-env configuration including dates.

    Returns:
        JSON-serializable provenance mapping.

    Raises:
        TrainerConfigError: If provenance cannot be made immutable.
    """
    data = env_config.get("data")
    if not isinstance(data, Mapping):
        raise TrainerConfigError("Resolved environment data must be a mapping.")
    provider = data.get("provider")
    if provider == "file":
        raw_path = data.get("path")
        if not isinstance(raw_path, str) or not raw_path:
            raise TrainerConfigError(
                f"File-provider data.path must be a non-empty string, got {raw_path!r}."
            )
        path = Path(raw_path)
        if not path.is_absolute():
            raise TrainerConfigError(
                f"File-provider data.path must be resolved before provenance capture: "
                f"{raw_path!r}."
            )
        return _capture_file_data_provenance(data, path)
    if provider == "synthetic":
        canonical_input = copy.deepcopy(dict(env_config))
        return {
            "provider": "synthetic",
            "sha256": _canonical_sha256(canonical_input),
            "requested_coverage": _requested_coverage(data),
            "actual_coverage": _requested_coverage(data),
        }
    raise TrainerConfigError(
        f"Data provider {provider!r} cannot produce immutable run provenance; "
        "materialize it as a current-contract Parquet cache and use provider 'file'."
    )


def capture_run_data_provenance(
    resolved_train_env: Mapping[str, Any],
    resolved_val_env: Mapping[str, Any],
    resolved_eval_env: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    """Capture data provenance for all run splits.

    Repeated file paths are hashed and read once; only their requested split
    coverage differs.

    Args:
        resolved_train_env: Resolved training environment.
        resolved_val_env: Resolved validation environment.
        resolved_eval_env: Resolved evaluation environment.

    Returns:
        Mapping with train, validation, and evaluation provenance.
    """
    splits = {
        "train": resolved_train_env,
        "validation": resolved_val_env,
        "evaluation": resolved_eval_env,
    }
    file_cache: dict[str, dict[str, Any]] = {}
    captured: dict[str, dict[str, Any]] = {}
    for split, env_config in splits.items():
        data = env_config.get("data")
        if not isinstance(data, Mapping):
            raise TrainerConfigError("Resolved environment data must be a mapping.")
        path = data.get("path") if data.get("provider") == "file" else None
        if isinstance(path, str) and path in file_cache:
            provenance = copy.deepcopy(file_cache[path])
            provenance["requested_coverage"] = _requested_coverage(data)
        else:
            provenance = capture_data_provenance(env_config)
            if isinstance(path, str):
                file_cache[path] = copy.deepcopy(provenance)
        captured[split] = provenance
    return captured


def _run_git(repo_root: Path, args: tuple[str, ...]) -> bytes:
    """Run Git and return stdout, failing explicitly.

    Args:
        repo_root: Repository root.
        args: Git arguments after ``git -C <repo>``.

    Returns:
        Raw stdout bytes.

    Raises:
        TrainerConfigError: If Git cannot inspect the repository.
    """
    result = subprocess.run(
        ["git", "-C", str(repo_root), *args],
        capture_output=True,
    )
    if result.returncode != 0:
        diagnostic = result.stderr.decode("utf-8", errors="replace").strip()
        raise TrainerConfigError(
            f"Failed to inspect Git repository {repo_root}: {diagnostic}"
        )
    return result.stdout


def capture_repository_provenance(repo_root: Path) -> dict[str, Any]:
    """Capture commit and exact dirty-worktree identity for a repository.

    Args:
        repo_root: Git repository root.

    Returns:
        Mapping containing commit, dirty flag, and worktree SHA-256.
    """
    commit = _run_git(repo_root, ("rev-parse", "HEAD")).decode("ascii").strip()
    tracked_diff = _run_git(
        repo_root,
        (
            "diff",
            "--binary",
            "HEAD",
            "--",
            "src/**",
            "pyproject.toml",
            "uv.lock",
        ),
    )
    untracked_output = _run_git(
        repo_root,
        (
            "ls-files",
            "--others",
            "--exclude-standard",
            "-z",
            "--",
            "src/**",
            "pyproject.toml",
            "uv.lock",
        ),
    )
    untracked_paths = [
        item.decode("utf-8") for item in untracked_output.split(b"\0") if item
    ]
    digest = hashlib.sha256()
    digest.update(b"tracked-diff\0")
    digest.update(tracked_diff)
    for relative in sorted(untracked_paths):
        path = repo_root / relative
        digest.update(b"untracked\0")
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        try:
            digest.update(path.read_bytes())
        except OSError as exc:
            raise TrainerConfigError(
                f"Failed to hash untracked repository file {path}: {exc}"
            ) from exc
    return {
        "commit": commit,
        "dirty": bool(tracked_diff or untracked_paths),
        "worktree_sha256": digest.hexdigest(),
    }


def verify_repository_provenance(
    recorded: Mapping[str, Any], repo_root: Path, label: str
) -> None:
    """Require the current repository to match recorded provenance.

    Args:
        recorded: Provenance stored in meta.json.
        repo_root: Current repository root.
        label: Repository label used in diagnostics.

    Raises:
        TrainerConfigError: If commit, dirty state, or worktree content differs.
    """
    current = capture_repository_provenance(repo_root)
    for key in ("commit", "dirty", "worktree_sha256"):
        if recorded.get(key) != current[key]:
            display = "worktree SHA-256" if key == "worktree_sha256" else key
            raise TrainerConfigError(
                f"Run {label} {display} mismatch: recorded "
                f"{recorded.get(key)!r}, current {current[key]!r}. Evaluate with "
                "the exact code state used to create the run."
            )


def capture_package_versions() -> dict[str, str]:
    """Capture package versions that affect training and evaluation.

    Returns:
        Package name to installed version mapping.
    """
    return {name: importlib_metadata.version(name) for name in _REQUIRED_PACKAGES}


def verify_data_provenance(
    recorded: Mapping[str, Any], resolved_eval_env: Mapping[str, Any]
) -> None:
    """Require current evaluation data to match its recorded identity.

    Args:
        recorded: Evaluation split provenance from meta.json.
        resolved_eval_env: Resolved evaluation environment snapshot.

    Raises:
        TrainerConfigError: If any provenance field differs.
    """
    current = capture_data_provenance(resolved_eval_env)
    keys = sorted(set(recorded) | set(current))
    for key in keys:
        if recorded.get(key) != current.get(key):
            raise TrainerConfigError(
                f"Run data provenance mismatch for {key}: recorded "
                f"{recorded.get(key)!r}, current {current.get(key)!r}. Restore the "
                "exact cache used for training or create a new run."
            )


def _require_meta_mapping(
    value: Any, field: str, run_origin: str
) -> Mapping[str, Any]:
    """Require one meta.json field to be a mapping.

    Args:
        value: Field value.
        field: Field name.
        run_origin: Run path text for diagnostics.

    Returns:
        Validated mapping.

    Raises:
        TrainerConfigError: If the field is malformed.
    """
    if not isinstance(value, Mapping):
        raise TrainerConfigError(
            f"Run {run_origin} has malformed provenance field {field!r}; expected "
            "a mapping."
        )
    return value


def verify_run_provenance(
    meta: Mapping[str, Any], resolved_eval_env: Mapping[str, Any]
) -> None:
    """Verify the full ADR-0011 contract before evaluation.

    Args:
        meta: Parsed run meta.json.
        resolved_eval_env: Parsed env_eval.yaml.

    Raises:
        TrainerConfigError: If the run is legacy, malformed, or no longer
            matches its data, repositories, or package versions.
    """
    version = meta.get("run_provenance_contract_version")
    if version != _RUN_PROVENANCE_CONTRACT_VERSION:
        if version is None:
            raise TrainerConfigError(
                "Cannot evaluate legacy run without immutable provenance. Create "
                "a new run under ADR-0011; provenance must not be inferred or "
                "silently backfilled."
            )
        raise TrainerConfigError(
            f"Unsupported run provenance contract version {version!r}; expected "
            f"{_RUN_PROVENANCE_CONTRACT_VERSION}."
        )
    data_provenance = _require_meta_mapping(
        meta.get("data_provenance"), "data_provenance", "meta.json"
    )
    expected_splits = {"train", "validation", "evaluation"}
    if set(data_provenance) != expected_splits:
        raise TrainerConfigError(
            "Run meta.json data_provenance must contain exactly train, validation, "
            f"and evaluation; got {sorted(data_provenance)}."
        )
    for split in ("train", "validation", "evaluation"):
        _require_meta_mapping(
            data_provenance[split], f"data_provenance.{split}", "meta.json"
        )
    evaluation = _require_meta_mapping(
        data_provenance.get("evaluation"),
        "data_provenance.evaluation",
        "meta.json",
    )
    verify_data_provenance(evaluation, resolved_eval_env)

    git = _require_meta_mapping(meta.get("git"), "git", "meta.json")
    repositories = {
        "forex_trainer": _TRAINER_REPO_ROOT,
        "forex_env": _ENV_REPO_ROOT,
    }
    for label, repo_root in repositories.items():
        recorded = _require_meta_mapping(git.get(label), f"git.{label}", "meta.json")
        verify_repository_provenance(recorded, repo_root, label)

    versions = _require_meta_mapping(meta.get("versions"), "versions", "meta.json")
    current_versions = capture_package_versions()
    if dict(versions) != current_versions:
        raise TrainerConfigError(
            f"Run package version provenance mismatch: recorded {dict(versions)!r}, "
            f"current {current_versions!r}."
        )


def write_run_metadata(
    run_dir: Path,
    config: ExperimentConfig,
    raw_config: Mapping[str, Any],
    resolved_train_env: Mapping[str, Any],
    resolved_val_env: Mapping[str, Any],
    resolved_eval_env: Mapping[str, Any],
    device: str,
    data_provenance: Mapping[str, Mapping[str, Any]],
) -> None:
    """Persist everything needed to reproduce the run.

    Args:
        run_dir: Run directory.
        config: Typed experiment configuration.
        raw_config: Raw experiment YAML contents.
        resolved_train_env: Env config with train dates injected.
        resolved_val_env: Env config with validation dates injected (ADR-0005).
        resolved_eval_env: Env config with eval dates injected.
        device: Resolved torch device string.
        data_provenance: Immutable provenance for every data split (ADR-0011).
    """
    (run_dir / "config_snapshot.yaml").write_text(
        yaml.safe_dump(dict(raw_config), sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    (run_dir / "env_train.yaml").write_text(
        yaml.safe_dump(dict(resolved_train_env), sort_keys=False), encoding="utf-8"
    )
    (run_dir / "env_val.yaml").write_text(
        yaml.safe_dump(dict(resolved_val_env), sort_keys=False), encoding="utf-8"
    )
    (run_dir / "env_eval.yaml").write_text(
        yaml.safe_dump(dict(resolved_eval_env), sort_keys=False), encoding="utf-8"
    )
    meta: dict[str, Any] = {
        "run_provenance_contract_version": _RUN_PROVENANCE_CONTRACT_VERSION,
        "experiment": config.experiment,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "seed": config.run.seed,
        "device": device,
        "algorithm": config.algorithm.name,
        "network": config.network.name,
        "decision_interval": config.run.decision_interval,
        "data_provenance": copy.deepcopy(dict(data_provenance)),
        "git": {
            "forex_trainer": capture_repository_provenance(_TRAINER_REPO_ROOT),
            "forex_env": capture_repository_provenance(_ENV_REPO_ROOT),
        },
        "versions": capture_package_versions(),
    }
    (run_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
