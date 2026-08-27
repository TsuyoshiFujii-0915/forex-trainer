"""Immutable data and file identities recorded by research artifacts."""

from __future__ import annotations

import hashlib
import json
import subprocess
from collections.abc import Mapping
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any

import forex_env

from .config import TrainerConfigError

_TRAINER_REPO_ROOT = Path(__file__).resolve().parents[2]
_ENV_REPO_ROOT = Path(forex_env.__file__).resolve().parents[2]
_VERSION_PACKAGES: tuple[str, ...] = (
    "forex-env-v3",
    "stable-baselines3",
    "sb3-contrib",
    "torch",
    "gymnasium",
)


def sha256_file(path: Path) -> str:
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


def data_identity_from_config(
    raw_config: Mapping[str, Any], origin: Path
) -> dict[str, str]:
    """Capture the exact data identity used by an experiment config.

    Args:
        raw_config: Raw experiment config snapshot.
        origin: Config path used in validation errors.

    Returns:
        File path and SHA-256, or a deterministic non-file config digest.

    Raises:
        TrainerConfigError: If data metadata or a file-backed dataset is invalid.
    """
    env = raw_config.get("env")
    if not isinstance(env, Mapping):
        raise TrainerConfigError(f"Config {origin} requires mapping field 'env'.")
    data = env.get("data")
    if not isinstance(data, Mapping):
        raise TrainerConfigError(f"Config {origin} requires mapping field 'env.data'.")
    provider = data.get("provider")
    if not isinstance(provider, str) or not provider:
        raise TrainerConfigError(
            f"Config {origin} requires non-empty env.data.provider."
        )
    if provider == "file":
        path_value = data.get("path")
        if not isinstance(path_value, str) or not path_value:
            raise TrainerConfigError(
                f"Config {origin} requires non-empty env.data.path."
            )
        data_path = Path(path_value).resolve()
        if not data_path.is_file():
            raise TrainerConfigError(
                f"Data file recorded by {origin} does not exist: {data_path}"
            )
        return {
            "provider": provider,
            "path": str(data_path),
            "sha256": sha256_file(data_path),
        }
    encoded = json.dumps(dict(data), sort_keys=True, separators=(",", ":")).encode()
    return {
        "provider": provider,
        "config_sha256": hashlib.sha256(encoded).hexdigest(),
    }


def git_commits() -> dict[str, str]:
    """Capture the current trainer and environment Git commits.

    Returns:
        Repository names mapped to HEAD SHA or an explicit unavailable marker.
    """
    return {
        "forex_trainer": _git_commit(_TRAINER_REPO_ROOT),
        "forex_env": _git_commit(_ENV_REPO_ROOT),
    }


def dependency_versions() -> dict[str, str]:
    """Capture dependency versions that can affect training or evaluation.

    Returns:
        Distribution names mapped to installed versions.
    """
    return {name: importlib_metadata.version(name) for name in _VERSION_PACKAGES}


def evaluation_runtime_provenance(
    resolved_device: str,
    raw_config: Mapping[str, Any],
    origin: Path,
) -> dict[str, Any]:
    """Capture the runtime conditions that produced evaluation metrics.

    Args:
        resolved_device: Concrete inference device actually passed to the model.
        raw_config: Raw experiment snapshot used for evaluation.
        origin: Snapshot path used in data-identity errors.

    Returns:
        Evaluation device, software revisions, versions, and data identity.
    """
    return {
        "resolved_device": resolved_device,
        "git": git_commits(),
        "versions": dependency_versions(),
        "data_identity": data_identity_from_config(raw_config, origin),
    }


def _git_commit(repo_root: Path) -> str:
    """Return a repository HEAD SHA or an explicit unavailable marker.

    Args:
        repo_root: Repository root directory.

    Returns:
        Commit SHA, or unavailable when the directory is not a Git repository.
    """
    result = subprocess.run(
        ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return "unavailable"
    return result.stdout.strip()
