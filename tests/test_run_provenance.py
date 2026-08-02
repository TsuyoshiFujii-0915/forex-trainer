"""Run provenance contract tests (ADR-0011)."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow.parquet as pq
import pytest
from forex_env.data.file_provider import save_ohlcv_parquet

from forex_trainer.config import TrainerConfigError
from forex_trainer.evaluate import run_evaluation
from forex_trainer.run_dir import (
    capture_data_provenance,
    capture_repository_provenance,
    resolve_file_data_path,
    verify_data_provenance,
    verify_repository_provenance,
)
from helpers import make_experiment_raw


def _write_cache(path: Path, close_offset: float = 0.0) -> None:
    """Write a small current-contract cache.

    Args:
        path: Destination Parquet path.
        close_offset: Value added to Close so tests can alter file content.
    """
    index = pd.date_range("2020-01-01", periods=4, freq="D", tz="UTC")
    fields = {
        "Open": [100.0, 101.0, 102.0, 103.0],
        "High": [101.0, 102.0, 103.0, 104.0],
        "Low": [99.0, 100.0, 101.0, 102.0],
        "Close": [100.5 + close_offset, 101.5, 102.5, 103.5],
        "Volume": [1.0, 1.0, 1.0, 1.0],
    }
    frame = pd.DataFrame(
        {
            ("JPY/USD", field): values
            for field, values in fields.items()
        },
        index=index,
    )
    frame.columns = pd.MultiIndex.from_tuples(frame.columns)
    save_ohlcv_parquet(frame, "1d", "2020-01-01", "2020-01-04", path)


def _add_augmentation_metadata(path: Path) -> None:
    """Add representative trainer-owned FRED provenance to a cache.

    Args:
        path: Existing Parquet cache.
    """
    table = pq.read_table(path)
    metadata = dict(table.schema.metadata or {})
    metadata[b"forex_trainer_fred_series"] = b'{"USD":"DFF"}'
    metadata[b"forex_trainer_lag_days"] = b"30"
    pq.write_table(table.replace_schema_metadata(metadata), path)


def _file_env(path: str) -> dict[str, Any]:
    """Return a resolved file-provider environment mapping.

    Args:
        path: Cache path inserted into the data section.

    Returns:
        Resolved environment mapping.
    """
    raw = make_experiment_raw()["env"]
    raw["data"] = {
        "provider": "file",
        "timeframe": "1d",
        "path": path,
        "start_date": "2020-01-01",
        "end_date": "2020-01-04",
    }
    return raw


def _run_git(repo: Path, *args: str) -> None:
    """Run a successful Git command in a test repository.

    Args:
        repo: Repository directory.
        *args: Git arguments.
    """
    subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True)


def test_file_provenance_records_content_contract_and_actual_coverage(
    tmp_path: Path,
) -> None:
    """File provenance identifies bytes, coverage, contract, and augmentation."""
    config_path = tmp_path / "configs" / "experiment.yaml"
    config_path.parent.mkdir()
    cache_path = config_path.parent / "cache.parquet"
    _write_cache(cache_path)
    _add_augmentation_metadata(cache_path)

    resolved = resolve_file_data_path(_file_env("cache.parquet"), config_path.parent)
    assert resolved["data"]["path"] == str(cache_path.resolve())

    provenance = capture_data_provenance(resolved)
    assert provenance["provider"] == "file"
    assert provenance["path"] == str(cache_path.resolve())
    assert provenance["sha256"] == hashlib.sha256(cache_path.read_bytes()).hexdigest()
    assert provenance["actual_coverage"] == {
        "start": "2020-01-01T00:00:00+00:00",
        "end": "2020-01-04T00:00:00+00:00",
        "rows": 4,
    }
    assert provenance["cache_contract"]["schema_version"] == "2"
    assert provenance["cache_contract"]["carry_contract"] == "absent"
    assert provenance["augmentation_metadata"] == {
        "forex_trainer_fred_series": '{"USD":"DFF"}',
        "forex_trainer_lag_days": "30",
    }


def test_repository_verification_detects_dirty_content_change(tmp_path: Path) -> None:
    """A changed dirty worktree is rejected even when HEAD is unchanged."""
    repo = tmp_path / "repo"
    repo.mkdir()
    _run_git(repo, "init")
    tracked = repo / "src" / "tracked.py"
    tracked.parent.mkdir()
    tracked.write_text("committed\n", encoding="utf-8")
    _run_git(repo, "add", "src/tracked.py")
    _run_git(
        repo,
        "-c",
        "user.name=Test",
        "-c",
        "user.email=test@example.com",
        "commit",
        "-m",
        "initial",
    )
    tracked.write_text("dirty one\n", encoding="utf-8")
    recorded = capture_repository_provenance(repo)
    verify_repository_provenance(recorded, repo, "test_repo")

    tracked.write_text("dirty two\n", encoding="utf-8")
    with pytest.raises(TrainerConfigError, match="worktree SHA-256 mismatch"):
        verify_repository_provenance(recorded, repo, "test_repo")


def test_run_verification_detects_cache_replacement(tmp_path: Path) -> None:
    """Evaluation provenance rejects bytes changed at an unchanged cache path."""
    cache_path = tmp_path / "cache.parquet"
    _write_cache(cache_path)
    resolved = _file_env(str(cache_path))
    meta = {
        "run_provenance_contract_version": 1,
        "data_provenance": {"evaluation": capture_data_provenance(resolved)},
        "git": {},
        "versions": {},
    }
    _write_cache(cache_path, close_offset=0.25)

    with pytest.raises(TrainerConfigError, match="data provenance mismatch.*sha256"):
        verify_data_provenance(meta["data_provenance"]["evaluation"], resolved)


def test_run_evaluation_rejects_legacy_run_before_model_loading(tmp_path: Path) -> None:
    """Unversioned runs require explicit migration instead of an implicit fallback."""
    run_dir = tmp_path / "legacy"
    run_dir.mkdir()
    raw = make_experiment_raw()
    (run_dir / "config_snapshot.yaml").write_text(
        json.dumps(raw), encoding="utf-8"
    )
    resolved_eval = raw["env"] | {
        "data": raw["env"]["data"]
        | {"start_date": "2020-02-15", "end_date": "2020-03-01"}
    }
    (run_dir / "env_eval.yaml").write_text(
        json.dumps(resolved_eval), encoding="utf-8"
    )
    (run_dir / "model_final.zip").write_bytes(b"not a real model")
    (run_dir / "meta.json").write_text(
        json.dumps({"git": {}, "versions": {}}), encoding="utf-8"
    )

    with pytest.raises(TrainerConfigError, match="legacy run.*provenance"):
        run_evaluation(run_dir)
