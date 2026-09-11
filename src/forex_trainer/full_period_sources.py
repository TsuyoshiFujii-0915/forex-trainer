"""Strict frozen-source loading for the bounded full-period evaluator."""

from __future__ import annotations

import copy
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from forex_env.data.file_provider import FileDataProvider

from .artifact_provenance import dependency_versions, git_commits
from .config import parse_experiment_config, resolve_env_raw
from .ensemble import load_member_for_device
from .full_period import (
    MEASUREMENT_ID,
    FrozenRidge,
    PeriodPlan,
    checked_path,
    prepare_period,
    read_json,
    runtime_identity,
)


@dataclass(frozen=True)
class FrozenEnsemble:
    """Three loaded models sharing the current ensemble account observation."""

    models: tuple[Any, ...]

    def action(
        self, observation: dict[str, np.ndarray], window: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Infer a fresh deterministic action mean with current PPO assets.

        Args:
            observation: This policy's market and account state.
            window: Float64 ridge window; unused by PPO.

        Returns:
            Diagnostic direct scores and the mean float32 action.
        """
        actions = [
            np.asarray(
                model.predict(observation, deterministic=True)[0], dtype=np.float64
            )
            for model in self.models
        ]
        if any(
            action.shape != (9, 1) or not np.isfinite(action).all()
            for action in actions
        ):
            raise ValueError("Frozen PPO member returned malformed nine-pair action")
        action = np.mean(np.stack(actions), axis=0).astype(np.float32)
        return action[:, 0].astype(float), action


@dataclass(frozen=True)
class FoldSources:
    """Validated independent fold inputs; no implicit source discovery."""

    fold: str
    env_raw: dict[str, Any]
    plan: PeriodPlan
    ridge: FrozenRidge
    ppo: FrozenEnsemble
    identity: dict[str, Any]


def _hash(path: Path, expected: str) -> Path:
    """Verify one explicit path against its sealed hash.

    Args:
        path: Source file.
        expected: Required digest.

    Returns:
        Verified path.
    """
    return checked_path({"path": str(path), "sha256": expected})


def load_frozen_ensemble(
    source: Mapping[str, Any], raw: dict[str, Any]
) -> FrozenEnsemble:
    """Load only the sealed current-provenance PPO ensemble.

    Args:
        source: Parent seal's fold PPO identity.
        raw: Hash-verified fold experiment config.

    Returns:
        Three validated frozen models, without training or old CSV actions.
    """
    directory = Path(source["ensemble_dir"])
    path = _hash(directory / "ensemble.json", source["ensemble_manifest_sha256"])
    manifest = read_json(path)
    for field, expected in (
        ("manifest_version", 2),
        ("policy", "action_mean"),
        ("model_selection", "validation_best"),
        ("decision_interval", 1),
    ):
        if manifest[field] != expected:
            raise ValueError(f"Invalid frozen ensemble {field}: {path}")
    members = manifest["members"]
    if [member["seed"] for member in members] != [42, 43, 44]:
        raise ValueError(f"Frozen ensemble requires ordered seeds 42/43/44: {path}")
    if [member["model_sha256"] for member in members] != source["member_model_sha256"]:
        raise ValueError(f"Parent seal and ensemble model hashes differ: {path}")
    evaluation = manifest["evaluation"]
    comparisons = {
        "git": "evaluation_git",
        "versions": "evaluation_versions",
        "data_identity": "data_identity",
        "metrics_sha256": "metrics_sha256",
        "env_eval_sha256": "eval_env_sha256",
        "resolved_device": "sealed_device",
    }
    for field, parent_field in comparisons.items():
        if evaluation[field] != source[parent_field]:
            raise ValueError(f"Ensemble {field} differs from parent seal: {path}")
    if evaluation["resolved_device"] != "cpu":
        raise ValueError(f"Current frozen ensemble requires sealed CPU device: {path}")
    _hash(directory / "metrics.json", evaluation["metrics_sha256"])
    eval_path = _hash(directory / "env_eval.yaml", evaluation["env_eval_sha256"])
    config = parse_experiment_config(raw)
    expected_eval = resolve_env_raw(config.env, config.eval_range, for_eval=True)
    if yaml.safe_load(eval_path.read_text()) != expected_eval:
        raise ValueError(f"Ensemble evaluation config differs from fold config: {path}")
    models: list[Any] = []
    for member in members:
        run = Path(member["run_dir"])
        if member["model_path"] != "model_final.zip":
            raise ValueError(f"Expected validation-selected model_final.zip: {run}")
        for name, field in (
            ("model_final.zip", "model_sha256"),
            ("config_snapshot.yaml", "config_snapshot_sha256"),
            ("meta.json", "meta_sha256"),
        ):
            _hash(run / name, member[field])
        loaded_config, model, eval_raw, _member_raw, meta, _device = (
            load_member_for_device(run, "cpu")
        )
        if (
            loaded_config.run.seed != member["seed"]
            or meta["seed"] != member["seed"]
            or loaded_config.experiment != member["experiment"]
            or meta["experiment"] != member["experiment"]
        ):
            raise ValueError(f"PPO member config/meta identity mismatch: {run}")
        if (
            loaded_config.algorithm.name != "ppo"
            or loaded_config.network.name != "mlp"
            or loaded_config.run.decision_interval != 1
            or loaded_config.run.residual is not None
            or loaded_config.run.rank_allocation is not None
            or loaded_config.run.apply_hold_gate is not None
        ):
            raise ValueError(f"Expected direct ungated PPO longf k=1: {run}")
        if eval_raw != expected_eval:
            raise ValueError(f"PPO member evaluation settings differ: {run}")
        if (
            model.action_space.shape != (9, 1)
            or model.observation_space["market"].shape != (9, 32, 8)
            or model.observation_space["assets"].shape != (9, 3)
        ):
            raise ValueError(f"Frozen PPO model space mismatch: {run}")
        models.append(model)
    return FrozenEnsemble(tuple(models))


def load_campaign_sources(config: dict[str, Any], origin: Path) -> list[FoldSources]:
    """Preflight every selected source and calendar before creating outputs.

    Args:
        config: Explicit full-period campaign mapping.
        origin: Campaign file for errors.

    Returns:
        Ordered verified fold inputs.
    """
    required = {
        "mode",
        "measurement_id",
        "evidence_class",
        "source",
        "lineage",
        "runtime",
        "folds",
    }
    if set(config) != required:
        raise ValueError(f"Campaign requires exact fields {sorted(required)}: {origin}")
    if config["mode"] != "full_period" or config["measurement_id"] != MEASUREMENT_ID:
        raise ValueError(f"Unsupported measurement mode/ID: {origin}")
    if config["evidence_class"] != "development_historical":
        raise ValueError(
            f"Full-period evaluator only supports development_historical: {origin}"
        )
    runtime = config["runtime"]
    if set(runtime) != {"forex_env_sha", "versions", "device"}:
        raise ValueError(f"Malformed runtime contract: {origin}")
    actual_versions = runtime_identity()["versions"]
    version_keys = set(runtime["versions"])
    valid_version_keys = version_keys == set(
        dependency_versions()
    ) or version_keys == set(actual_versions)
    if (
        runtime["device"] != "cpu"
        or runtime["forex_env_sha"] != git_commits()["forex_env"]
        or not valid_version_keys
        or any(
            actual_versions[name] != expected
            for name, expected in runtime["versions"].items()
        )
    ):
        raise ValueError(
            f"runtime differs from pinned CPU/env/dependency contract: {origin}; actual git={git_commits()}, versions={dependency_versions()}"
        )
    source_path = checked_path(config["source"])
    source = read_json(source_path)
    if source["artifact_version"] != 1:
        raise ValueError(f"Unsupported parent seal version: {source_path}")
    models_path = _hash(
        source_path.parent / "models.json",
        source["generated_artifact_sha256"]["models.json"],
    )
    models = read_json(models_path)
    lineage_path = checked_path(config["lineage"])
    lineage = read_json(lineage_path)
    lineage_fields = {
        "raw",
        "clean",
        "carry",
        "transformations",
        "rate_vintage",
        "available_at_evidence",
    }
    if set(lineage) != lineage_fields:
        raise ValueError(f"lineage requires {sorted(lineage_fields)}: {lineage_path}")
    for name in ("raw", "clean", "carry"):
        if not isinstance(lineage[name], list) or not lineage[name]:
            raise ValueError(
                f"lineage.{name} requires explicit path/hash sources: {lineage_path}"
            )
        for reference in lineage[name]:
            checked_path(reference)
    for name in ("transformations", "rate_vintage", "available_at_evidence"):
        if not isinstance(lineage[name], str) or not lineage[name].strip():
            raise ValueError(
                f"lineage.{name} requires explicit evidence or unknown declaration: {lineage_path}"
            )
    folds = config["folds"]
    if not isinstance(folds, list) or not folds:
        raise ValueError(f"At least one explicit fold required: {origin}")
    seen: set[str] = set()
    sources: list[FoldSources] = []
    for fold_config in folds:
        if set(fold_config) != {
            "fold",
            "measurement_start",
            "measurement_end",
            "calendar",
        }:
            raise ValueError(f"Malformed fold configuration: {origin}: {fold_config}")
        fold = fold_config["fold"]
        if (
            not isinstance(fold, str)
            or not re.fullmatch(r"[A-Za-z0-9_-]+", fold)
            or fold in seen
        ):
            raise ValueError(f"Duplicate or invalid fold {fold!r}: {origin}")
        seen.add(fold)
        if fold not in source["fold_sources"] or fold not in models:
            raise ValueError(
                f"missing frozen fold {fold} in {source_path} or {models_path}"
            )
        identity = source["fold_sources"][fold]
        config_file = _hash(Path(identity["config_path"]), identity["config_sha256"])
        raw = yaml.safe_load(config_file.read_text())
        parsed = parse_experiment_config(raw)
        data_identity = identity["data_identity"]
        if data_identity["provider"] != "file":
            raise ValueError(f"Full-period requires a frozen file cache: {config_file}")
        data_path = _hash(Path(data_identity["path"]), data_identity["sha256"])
        if (
            Path(parsed.env["data"]["path"]).resolve() != data_path
            or identity["ppo"]["data_identity"] != data_identity
        ):
            raise ValueError(
                f"Config, source data, and PPO data identities differ: {config_file}"
            )
        if not any(
            Path(item["path"]).resolve() == data_path
            and item["sha256"] == data_identity["sha256"]
            for item in lineage["carry"]
        ):
            raise ValueError(
                f"lineage.carry does not identify evaluation cache {data_path}"
            )
        calendar_path = checked_path(fold_config["calendar"])
        calendar = read_json(calendar_path)
        symbols = parsed.env["environment"]["currency_pairs"]
        sessions = calendar["sessions"]
        if not isinstance(sessions, list) or not sessions:
            raise ValueError(f"Missing calendar sessions: {calendar_path}")
        begin = str(pd.Timestamp(sessions[0]["bar_label"]).date())
        end = str(pd.Timestamp(sessions[-1]["bar_label"]).date())
        FileDataProvider(str(data_path)).get_data(
            tuple(symbols), begin, end, parsed.env["data"]["timeframe"]
        )
        market = pd.read_parquet(data_path)
        market.columns = pd.MultiIndex.from_tuples(
            [tuple(column.rsplit("|", 1)) for column in market.columns]
        )
        plan = prepare_period(
            market,
            calendar,
            fold_config["measurement_start"],
            fold_config["measurement_end"],
            tuple(symbols),
            data_path,
        )
        ridge = FrozenRidge(models[fold], models_path)
        ensemble = load_frozen_ensemble(identity["ppo"], raw)
        sources.append(
            FoldSources(
                fold,
                copy.deepcopy(parsed.env),
                plan,
                ridge,
                ensemble,
                {
                    **copy.deepcopy(identity),
                    "lineage": lineage,
                    "calendar_source": dict(fold_config["calendar"]),
                },
            )
        )
    return sources
