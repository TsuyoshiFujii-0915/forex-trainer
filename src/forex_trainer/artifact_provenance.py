"""Immutable data and file identities recorded by research artifacts."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .config import TrainerConfigError


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
