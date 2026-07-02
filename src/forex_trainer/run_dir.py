"""Run directory creation and metadata capture (ADR-0003)."""

from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any, Mapping

import forex_env
import yaml

from .config import ExperimentConfig

_TRAINER_REPO_ROOT = Path(__file__).resolve().parents[2]
_ENV_REPO_ROOT = Path(forex_env.__file__).resolve().parents[2]


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


def _git_commit(repo_root: Path) -> str:
    """Return the HEAD commit of a repository, or an explicit marker.

    Args:
        repo_root: Repository root directory.

    Returns:
        Commit SHA, or "unavailable" when the directory is not a usable git
        repository (recorded explicitly rather than failing the run, since
        metadata capture must not block training).
    """
    result = subprocess.run(
        ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return "unavailable"
    return result.stdout.strip()


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
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "seed": config.run.seed,
        "device": device,
        "algorithm": config.algorithm.name,
        "network": config.network.name,
        "decision_interval": config.run.decision_interval,
        "git": {
            "forex_trainer": _git_commit(_TRAINER_REPO_ROOT),
            "forex_env": _git_commit(_ENV_REPO_ROOT),
        },
        "versions": {
            name: importlib_metadata.version(name)
            for name in (
                "forex-env-v3",
                "stable-baselines3",
                "sb3-contrib",
                "torch",
                "gymnasium",
            )
        },
    }
    (run_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
