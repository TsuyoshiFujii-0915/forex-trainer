"""Run directory creation and metadata capture (ADR-0003)."""

from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from .artifact_provenance import (
    data_identity_from_config,
    dependency_versions,
    git_commits,
)
from .config import ExperimentConfig


def create_run_dir(runs_root: Path, experiment: str) -> Path:
    """Create a fresh run directory runs_root/<experiment>/<UTC timestamp>/.

    Args:
        runs_root: Root directory for all runs.
        experiment: Experiment name (validated by config parsing).

    Returns:
        Path to the created directory.
    """
    timestamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S-%f")
    run_dir = runs_root / experiment / timestamp
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def write_run_metadata(
    run_dir: Path,
    config: ExperimentConfig,
    raw_config: Mapping[str, Any],
    resolved_train_env: Mapping[str, Any],
    resolved_val_env: Mapping[str, Any],
    resolved_eval_env: Mapping[str, Any],
    device: str,
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
        "experiment": config.experiment,
        "created_utc": datetime.now(UTC).isoformat(),
        "seed": config.run.seed,
        "requested_device": config.run.device,
        "device": device,
        "algorithm": config.algorithm.name,
        "network": config.network.name,
        "decision_interval": config.run.decision_interval,
        "git": git_commits(),
        "versions": dependency_versions(),
        "data_identity": data_identity_from_config(
            raw_config, run_dir / "config_snapshot.yaml"
        ),
    }
    (run_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
