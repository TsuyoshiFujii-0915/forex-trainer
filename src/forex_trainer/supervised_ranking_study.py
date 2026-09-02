"""Sealed 17-fold supervised ranking study for Issue #15."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import sys
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from forex_env import parse_config as parse_env_config
from forex_env.data import build_provider
from forex_env.errors import ConfigError, DataError, FeatureError
from forex_env.features import FeaturePipeline

from .artifact_provenance import (
    data_identity_from_config,
    dependency_versions,
    git_commits,
    sha256_file,
)
from .config import (
    ExperimentConfig,
    TrainerConfigError,
    load_experiment_config,
    resolve_env_raw,
)
from .ensemble import load_member_for_device
from .env_factory import GateEvaluationMode, build_single_env
from .evaluate import compute_metrics
from .features import CROSS_FEATURE_REGISTRY, FEATURE_REGISTRY
from .research_statistics import bootstrap_mean_intervals
from .supervised_ranking import (
    ClassificationInputs,
    ScoreDiagnostics,
    SupervisedDataset,
    apply_standardizer,
    build_aligned_dataset,
    canonical_reversal_scores,
    classify_learnability,
    compute_score_diagnostics,
    cross_sectional_spearman,
    fit_ridge,
    fit_standardizer,
    select_validation_alpha,
)

_STUDY_KEYS: tuple[str, ...] = (
    "name",
    "folds",
    "alpha_grid",
    "top_k",
    "member_seeds",
    "eras",
    "bootstrap_samples",
    "bootstrap_seed",
    "moving_block_length",
)
_FOLD_KEYS: tuple[str, ...] = ("config", "ppo_ensemble")
_ERA_KEYS: tuple[str, ...] = ("start", "end")
_EXPECTED_FOLDS: tuple[str, ...] = tuple(str(year) for year in range(2009, 2026))
_EXPECTED_ALPHAS: tuple[float, ...] = (0.0, 0.1, 1.0, 10.0)
_EXPECTED_SEEDS: tuple[int, ...] = (42, 43, 44)
_EXPECTED_PAIRS: tuple[str, ...] = (
    "JPY/USD",
    "JPY/EUR",
    "JPY/GBP",
    "JPY/AUD",
    "JPY/CHF",
    "JPY/CAD",
    "JPY/NZD",
    "JPY/NOK",
    "JPY/SEK",
)
_EXPECTED_FEATURES: tuple[str, ...] = (
    "log_return",
    "volatility",
    "sma20_ratio",
    "mom24",
    "xz_mom24",
    "xr_mom24",
    "carry_annual",
    "xz_carry",
)
_EXPECTED_ERAS: tuple[tuple[str, int, int], ...] = (
    ("2009-2018", 2009, 2018),
    ("2019-2025", 2019, 2025),
)
_EXPECTED_EVALUATION_GIT_KEYS: frozenset[str] = frozenset(
    {"forex_trainer", "forex_env"}
)
_EXPECTED_EVALUATION_VERSION_KEYS: frozenset[str] = frozenset(
    {"forex-env-v3", "stable-baselines3", "sb3-contrib", "torch", "gymnasium"}
)
_DIAGNOSTIC_FIELDS: tuple[str, ...] = tuple(ScoreDiagnostics.__dataclass_fields__)


@dataclass(frozen=True)
class FoldSource:
    """Explicit source config and sealed PPO ensemble for one evaluation fold."""

    config: Path
    ppo_ensemble: Path


@dataclass(frozen=True)
class EraSpec:
    """Inclusive ordered evaluation-fold era."""

    name: str
    start: int
    end: int


@dataclass(frozen=True)
class SupervisedStudy:
    """Strict preregistered Issue #15 study definition."""

    name: str
    folds: Mapping[str, FoldSource]
    alpha_grid: tuple[float, ...]
    top_k: int
    member_seeds: tuple[int, ...]
    eras: tuple[EraSpec, ...]
    bootstrap_samples: int
    bootstrap_seed: int
    moving_block_length: int
    source_path: Path


@dataclass(frozen=True)
class PpoScoreTrace:
    """Sealed action-mean PPO ordering at aligned decision timestamps."""

    scores: np.ndarray
    decision_timestamps: tuple[datetime, ...]
    symbols: tuple[str, ...]

    def __post_init__(self) -> None:
        """Require a complete finite decision-by-pair score matrix."""
        scores = np.asarray(self.scores, dtype=np.float64)
        if scores.shape != (len(self.decision_timestamps), len(self.symbols)):
            raise ValueError(
                "PPO scores must have shape decision timestamps by symbols."
            )
        if scores.ndim != 2 or not np.isfinite(scores).all():
            raise ValueError("PPO scores must be a finite two-dimensional array.")
        if len(set(self.symbols)) != len(self.symbols):
            raise ValueError("PPO symbols must be unique.")
        object.__setattr__(self, "scores", scores)


def _require_mapping(value: Any, origin: str) -> Mapping[str, Any]:
    """Require a mapping with an origin-rich error.

    Args:
        value: Candidate mapping value.
        origin: Human-readable source for error messages.

    Returns:
        Validated mapping.

    Raises:
        TrainerConfigError: If value is not a mapping.
    """
    if not isinstance(value, Mapping):
        raise TrainerConfigError(f"{origin} must be a mapping.")
    return value


def _require_exact_keys(
    value: Mapping[str, Any], expected: Sequence[str], origin: str
) -> None:
    """Require exactly the declared mapping keys.

    Args:
        value: Mapping to validate.
        expected: Complete allowed key sequence.
        origin: Human-readable source for error messages.

    Raises:
        TrainerConfigError: If keys are missing or unknown.
    """
    missing = sorted(set(expected) - set(value))
    unknown = sorted(set(value) - set(expected))
    if missing or unknown:
        raise TrainerConfigError(
            f"{origin} keys are invalid: missing={missing}, unknown={unknown}."
        )


def _require_string(value: Any, origin: str) -> str:
    """Require a non-empty string.

    Args:
        value: Candidate string value.
        origin: Human-readable source for error messages.

    Returns:
        Validated string.

    Raises:
        TrainerConfigError: If value is not a non-empty string.
    """
    if not isinstance(value, str) or not value:
        raise TrainerConfigError(f"{origin} must be a non-empty string.")
    return value


def _require_integer(value: Any, origin: str, minimum: int) -> int:
    """Require an integer at or above a minimum.

    Args:
        value: Candidate integer value.
        origin: Human-readable source for error messages.
        minimum: Inclusive minimum value.

    Returns:
        Validated integer.

    Raises:
        TrainerConfigError: If value is not an integer in range.
    """
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise TrainerConfigError(
            f"{origin} must be an integer >= {minimum}, got {value!r}."
        )
    return value


def _load_mapping(path: Path, artifact: str) -> Mapping[str, Any]:
    """Load a required JSON or YAML mapping.

    Args:
        path: Artifact path.
        artifact: Human-readable artifact type.

    Returns:
        Parsed root mapping.

    Raises:
        TrainerConfigError: If the file is missing, malformed, or not a mapping.
    """
    if not path.is_file():
        raise TrainerConfigError(f"Required {artifact} is missing: {path}")
    try:
        if path.suffix.lower() == ".json":
            value = json.loads(path.read_text(encoding="utf-8"))
        else:
            value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, yaml.YAMLError) as exc:
        raise TrainerConfigError(f"Malformed {artifact} {path}: {exc}") from exc
    return _require_mapping(value, f"{artifact} root")


def _resolve_file(origin: Path, value: Any, field: str) -> Path:
    """Resolve one required file reference relative to a manifest.

    Args:
        origin: Referencing manifest path.
        value: Raw path value.
        field: Field name for errors.

    Returns:
        Absolute existing file path.

    Raises:
        TrainerConfigError: If the reference is invalid or missing.
    """
    text = _require_string(value, field)
    candidate = Path(text)
    resolved = (candidate if candidate.is_absolute() else origin.parent / candidate).resolve()
    if not resolved.is_file():
        raise TrainerConfigError(f"{field} file is missing: {resolved}")
    return resolved


def _resolve_directory(origin: Path, value: Any, field: str) -> Path:
    """Resolve one required directory reference relative to a manifest.

    Args:
        origin: Referencing manifest path.
        value: Raw path value.
        field: Field name for errors.

    Returns:
        Absolute existing directory path.

    Raises:
        TrainerConfigError: If the reference is invalid or missing.
    """
    text = _require_string(value, field)
    candidate = Path(text)
    resolved = (candidate if candidate.is_absolute() else origin.parent / candidate).resolve()
    if not resolved.is_dir():
        raise TrainerConfigError(f"{field} directory is missing: {resolved}")
    return resolved


def load_supervised_study(study_path: Path) -> SupervisedStudy:
    """Load the strict, fixed Issue #15 study manifest.

    Args:
        study_path: YAML path containing all 17 explicit sources.

    Returns:
        Validated immutable study definition.

    Raises:
        TrainerConfigError: If any path or preregistered control differs.
    """
    path = study_path.resolve()
    raw = _load_mapping(path, "supervised ranking study")
    missing = sorted(set(_STUDY_KEYS) - set(raw))
    unknown = sorted(set(raw) - set(_STUDY_KEYS))
    folds_value = raw.get("folds")
    fold_keys = tuple(folds_value) if isinstance(folds_value, Mapping) else ()
    if missing or unknown or fold_keys != _EXPECTED_FOLDS:
        raise TrainerConfigError(
            "Supervised ranking study contract is invalid: "
            f"missing={missing}, unknown={unknown}; folds must be ordered exactly "
            "2009 through 2025."
        )
    folds_raw = _require_mapping(folds_value, "Supervised ranking study folds")
    folds: dict[str, FoldSource] = {}
    for fold, value in folds_raw.items():
        fold_raw = _require_mapping(value, f"folds.{fold}")
        _require_exact_keys(fold_raw, _FOLD_KEYS, f"folds.{fold}")
        folds[fold] = FoldSource(
            config=_resolve_file(path, fold_raw["config"], f"folds.{fold}.config"),
            ppo_ensemble=_resolve_directory(
                path, fold_raw["ppo_ensemble"], f"folds.{fold}.ppo_ensemble"
            ),
        )
    alphas_raw = raw["alpha_grid"]
    if not isinstance(alphas_raw, list):
        raise TrainerConfigError("alpha_grid must be a list.")
    alphas = tuple(float(value) for value in alphas_raw)
    if alphas != _EXPECTED_ALPHAS:
        raise TrainerConfigError(f"alpha_grid must equal {list(_EXPECTED_ALPHAS)}.")
    top_k = _require_integer(raw["top_k"], "top_k", 1)
    if top_k != 2:
        raise TrainerConfigError("top_k must equal the preregistered value 2.")
    seeds_raw = raw["member_seeds"]
    if not isinstance(seeds_raw, list) or any(
        isinstance(seed, bool) or not isinstance(seed, int) for seed in seeds_raw
    ):
        raise TrainerConfigError("member_seeds must be an integer list.")
    seeds = tuple(seeds_raw)
    if seeds != _EXPECTED_SEEDS:
        raise TrainerConfigError(
            f"member_seeds must equal {list(_EXPECTED_SEEDS)} without duplicates."
        )
    eras_raw = _require_mapping(raw["eras"], "eras")
    if tuple(eras_raw) != tuple(item[0] for item in _EXPECTED_ERAS):
        raise TrainerConfigError("eras must be exactly 2009-2018 and 2019-2025.")
    eras: list[EraSpec] = []
    for name, expected_start, expected_end in _EXPECTED_ERAS:
        era_raw = _require_mapping(eras_raw[name], f"eras.{name}")
        _require_exact_keys(era_raw, _ERA_KEYS, f"eras.{name}")
        start = _require_integer(era_raw["start"], f"eras.{name}.start", 0)
        end = _require_integer(era_raw["end"], f"eras.{name}.end", 0)
        if (start, end) != (expected_start, expected_end):
            raise TrainerConfigError(
                f"eras.{name} must span {expected_start} through {expected_end}."
            )
        eras.append(EraSpec(name=name, start=start, end=end))
    bootstrap_samples = _require_integer(
        raw["bootstrap_samples"], "bootstrap_samples", 1
    )
    if bootstrap_samples != 10_000:
        raise TrainerConfigError("bootstrap_samples must equal 10000.")
    bootstrap_seed = _require_integer(raw["bootstrap_seed"], "bootstrap_seed", 0)
    moving_block_length = _require_integer(
        raw["moving_block_length"], "moving_block_length", 1
    )
    if moving_block_length != 3:
        raise TrainerConfigError("moving_block_length must equal 3.")
    return SupervisedStudy(
        name=_require_string(raw["name"], "name"),
        folds=folds,
        alpha_grid=alphas,
        top_k=top_k,
        member_seeds=seeds,
        eras=tuple(eras),
        bootstrap_samples=bootstrap_samples,
        bootstrap_seed=bootstrap_seed,
        moving_block_length=moving_block_length,
        source_path=path,
    )


def build_dataset_from_env_raw(
    env_raw: Mapping[str, Any],
    custom_feature_names: tuple[str, ...],
    custom_cross_feature_names: tuple[str, ...],
) -> SupervisedDataset:
    """Build the exact ``longf`` market-window dataset for one date range.

    Args:
        env_raw: Complete resolved forex-env configuration.
        custom_feature_names: Trainer per-pair feature registry names.
        custom_cross_feature_names: Trainer cross-sectional registry names.

    Returns:
        Causal flattened observation windows and next relative-return labels.
    """
    parsed = parse_env_config(env_raw)
    custom_features = {
        name: FEATURE_REGISTRY[name] for name in custom_feature_names
    }
    custom_cross_features = {
        name: CROSS_FEATURE_REGISTRY[name] for name in custom_cross_feature_names
    }
    pipeline = FeaturePipeline(
        parsed.features,
        custom_features,
        custom_cross_features=custom_cross_features,
    )
    symbols = parsed.environment.currency_pairs
    provider = build_provider(parsed.data, seed=parsed.environment.seed)
    market = provider.get_data(
        symbols,
        parsed.data.start_date,
        parsed.data.end_date,
        parsed.data.timeframe,
    )
    features = pipeline.compute(market, symbols)
    closes = market.xs("Close", axis=1, level=1)[list(symbols)].to_numpy(dtype=float)
    dataset = build_aligned_dataset(
        timestamps=market.index,
        closes=closes,
        features=features,
        feature_names=pipeline.feature_names,
        symbols=symbols,
        warmup=pipeline.warmup,
        window_size=parsed.environment.window_size,
    )
    if parsed.environment.random_start:
        base = parsed.environment.window_size - 1
        usable_rows = len(market.index) - pipeline.warmup
        latest = usable_rows - 1 - parsed.environment.episode_max_steps
        start = (
            int(np.random.default_rng(0).integers(base, latest + 1))
            if latest > base
            else base
        )
        reset_timestamp = market.index[pipeline.warmup + start].to_pydatetime()
        try:
            offset = dataset.decision_timestamps.index(reset_timestamp)
        except ValueError as exc:
            raise ValueError(
                "the deterministic environment reset timestamp is not a usable "
                "supervised decision."
            ) from exc
        order = np.concatenate(
            [
                np.arange(offset, len(dataset.decision_timestamps)),
                np.arange(0, offset),
            ]
        )
        dataset = SupervisedDataset(
            features=dataset.features[order],
            targets=dataset.targets[order],
            decision_timestamps=tuple(dataset.decision_timestamps[index] for index in order),
            target_timestamps=tuple(dataset.target_timestamps[index] for index in order),
            symbols=dataset.symbols,
            feature_names=dataset.feature_names,
        )
    return dataset


def require_score_alignment(
    dataset: SupervisedDataset, ppo_trace: PpoScoreTrace
) -> None:
    """Require exact ordered timestamp and pair alignment with no intersection.

    Args:
        dataset: Supervised evaluation dataset.
        ppo_trace: Sealed PPO action score trace.

    Raises:
        ValueError: If decisions, pairs, or score shape differ.
    """
    if ppo_trace.decision_timestamps != dataset.decision_timestamps:
        raise ValueError("PPO and supervised decision timestamps differ.")
    if ppo_trace.symbols != dataset.symbols:
        raise ValueError("PPO and supervised pair order differs.")
    if ppo_trace.scores.shape != dataset.targets.shape:
        raise ValueError("PPO score shape differs from supervised targets.")


def _transformed_dataset(
    dataset: SupervisedDataset, transformed_features: np.ndarray
) -> SupervisedDataset:
    """Preserve dataset identity while replacing its feature coordinates.

    Args:
        dataset: Source dataset retaining timestamps, targets, and axes.
        transformed_features: Replacement feature tensor with identical shape.

    Returns:
        Dataset with transformed features and unchanged identity fields.
    """
    return SupervisedDataset(
        features=transformed_features,
        targets=dataset.targets,
        decision_timestamps=dataset.decision_timestamps,
        target_timestamps=dataset.target_timestamps,
        symbols=dataset.symbols,
        feature_names=dataset.feature_names,
    )


def _era_name(study: SupervisedStudy, fold: str) -> str:
    """Resolve exactly one declared era for a fold.

    Args:
        study: Validated study containing era boundaries.
        fold: Four-digit evaluation year.

    Returns:
        Matching era name.

    Raises:
        TrainerConfigError: If the fold belongs to zero or multiple eras.
    """
    year = int(fold)
    matches = [era.name for era in study.eras if era.start <= year <= era.end]
    if len(matches) != 1:
        raise TrainerConfigError(f"fold {fold} belongs to {len(matches)} eras.")
    return matches[0]


def _validate_longf_config(config: ExperimentConfig, fold: str) -> None:
    """Require the canonical longf data, feature, and walk-forward contract.

    Args:
        config: Parsed fold experiment config.
        fold: Expected evaluation year.

    Raises:
        TrainerConfigError: If any canonical protocol field differs.
    """
    env = config.env
    environment = _require_mapping(env.get("environment"), "env.environment")
    data = _require_mapping(env.get("data"), "env.data")
    features = _require_mapping(env.get("features"), "env.features")
    if tuple(environment.get("currency_pairs", ())) != _EXPECTED_PAIRS:
        raise TrainerConfigError(f"fold {fold} must use the canonical 9-pair order.")
    if environment.get("window_size") != 32:
        raise TrainerConfigError(f"fold {fold} env window_size must equal 32.")
    if data.get("provider") != "file" or data.get("timeframe") != "1d":
        raise TrainerConfigError(f"fold {fold} must use canonical file-backed daily data.")
    if tuple(features.get("selected", ())) != _EXPECTED_FEATURES:
        raise TrainerConfigError(f"fold {fold} must use the canonical longf features.")
    if features.get("volatility_window") != 32 or features.get("normalize") is not False:
        raise TrainerConfigError(
            f"fold {fold} must use volatility_window 32 and normalize false."
        )
    year = int(fold)
    expected_ranges = (
        (config.train_range.start, "2003-06-01"),
        (config.train_range.end, f"{year - 1}-07-01"),
        (config.val_range.start, f"{year - 1}-07-01"),
        (config.val_range.end, f"{year}-01-01"),
        (config.eval_range.start, f"{year}-01-01"),
        (config.eval_range.end, f"{year + 1}-01-01"),
    )
    if any(actual != expected for actual, expected in expected_ranges):
        raise TrainerConfigError(f"fold {fold} does not use canonical expanding ranges.")
    if (
        config.run.decision_interval != 1
        or config.run.residual is not None
        or config.run.rank_allocation is not None
        or config.run.apply_hold_gate is not None
    ):
        raise TrainerConfigError(f"fold {fold} must use direct one-step actions.")


def _sealed_device(manifest: Mapping[str, Any], origin: Path) -> str:
    """Read the concrete evaluation device sealed in an ensemble manifest.

    Args:
        manifest: Versioned ensemble manifest.
        origin: Manifest path for errors.

    Returns:
        Concrete CPU, CUDA, or MPS device.

    Raises:
        TrainerConfigError: If the device is absent or unresolved.
    """
    evaluation = manifest.get("evaluation")
    device = evaluation.get("resolved_device") if isinstance(evaluation, Mapping) else None
    if device not in {"cpu", "cuda", "mps"}:
        raise TrainerConfigError(
            f"ensemble manifest lacks a concrete sealed device: {origin}"
        )
    return device


def _require_hash(path: Path, expected: Any, field: str) -> None:
    """Require a declared SHA-256 to match an existing file.

    Args:
        path: Existing file to hash.
        expected: Declared lowercase SHA-256 value.
        field: Artifact field name for errors.

    Raises:
        TrainerConfigError: If the hash is invalid or differs.
    """
    if not isinstance(expected, str) or sha256_file(path) != expected:
        raise TrainerConfigError(f"sealed hash mismatch for {field}: {path}")


def _require_string_mapping(value: Any, field: str) -> Mapping[str, str]:
    """Require non-empty string-to-string provenance metadata.

    Args:
        value: Candidate provenance value.
        field: Field name for errors.

    Returns:
        Validated string mapping.

    Raises:
        TrainerConfigError: If the value is empty or has non-string entries.
    """
    if not isinstance(value, Mapping) or not value or not all(
        isinstance(key, str) and isinstance(item, str)
        for key, item in value.items()
    ):
        raise TrainerConfigError(f"{field} must be a non-empty string mapping.")
    return value


def _metrics_match(
    actual: Mapping[str, Any], expected: Mapping[str, Any], origin: Path
) -> None:
    """Require deterministic replay metrics to match the sealed artifact.

    Args:
        actual: Metrics recomputed by deterministic replay.
        expected: Metrics loaded from the sealed artifact.
        origin: Sealed metrics path for errors.

    Raises:
        TrainerConfigError: If fields or values differ.
    """
    if set(actual) != set(expected):
        raise TrainerConfigError(f"replay metrics fields differ from {origin}.")
    for key in actual:
        left = actual[key]
        right = expected[key]
        if isinstance(left, bool) or isinstance(right, bool):
            equal = left is right
        elif isinstance(left, (int, float)) and isinstance(right, (int, float)):
            equal = math.isclose(float(left), float(right), rel_tol=1e-10, abs_tol=1e-12)
        else:
            equal = left == right
        if not equal:
            raise TrainerConfigError(
                f"deterministic replay metric {key!r} differs from {origin}."
            )


def _replay_ppo_scores(
    ensemble_dir: Path,
    expected_eval_raw: Mapping[str, Any],
    expected_symbols: tuple[str, ...],
    expected_seeds: tuple[int, ...],
) -> tuple[PpoScoreTrace, Mapping[str, Any]]:
    """Replay a sealed action-mean ensemble and retain raw pair score ordering.

    Args:
        ensemble_dir: Version-2 action-mean ensemble directory.
        expected_eval_raw: Eval environment reconstructed from the fold config.
        expected_symbols: Canonical stable pair order.
        expected_seeds: Exact member seed sequence.

    Returns:
        Aligned PPO score trace and sealed source provenance.

    Raises:
        TrainerConfigError: If identity, hashes, replay, or provenance differ.
        ValueError: If replay trace values are malformed.
    """
    manifest_path = ensemble_dir / "ensemble.json"
    manifest = _load_mapping(manifest_path, "ensemble manifest")
    if (
        manifest.get("manifest_version") != 2
        or manifest.get("policy") != "action_mean"
        or manifest.get("model_selection") != "validation_best"
        or manifest.get("decision_interval") != 1
    ):
        raise TrainerConfigError(f"ensemble contract is invalid: {manifest_path}")
    members_raw = manifest.get("members")
    if not isinstance(members_raw, list) or len(members_raw) != len(expected_seeds):
        raise TrainerConfigError(f"ensemble member count is invalid: {manifest_path}")
    seeds = tuple(int(member["seed"]) for member in members_raw)
    if seeds != expected_seeds:
        raise TrainerConfigError(f"ensemble seeds differ from {expected_seeds}: {manifest_path}")
    evaluation = _require_mapping(manifest.get("evaluation"), "ensemble evaluation")
    evaluation_git = _require_string_mapping(
        evaluation.get("git"), "ensemble evaluation.git"
    )
    evaluation_versions = _require_string_mapping(
        evaluation.get("versions"), "ensemble evaluation.versions"
    )
    evaluation_data_identity = _require_string_mapping(
        evaluation.get("data_identity"), "ensemble evaluation.data_identity"
    )
    if set(evaluation_git) != _EXPECTED_EVALUATION_GIT_KEYS:
        raise TrainerConfigError(
            f"ensemble evaluation.git has invalid keys: {manifest_path}"
        )
    if set(evaluation_versions) != _EXPECTED_EVALUATION_VERSION_KEYS:
        raise TrainerConfigError(
            f"ensemble evaluation.versions has invalid keys: {manifest_path}"
        )
    if set(evaluation_data_identity) != {"provider", "path", "sha256"}:
        raise TrainerConfigError(
            f"ensemble evaluation.data_identity is not canonical: {manifest_path}"
        )
    _require_hash(ensemble_dir / "metrics.json", evaluation.get("metrics_sha256"), "metrics")
    _require_hash(ensemble_dir / "env_eval.yaml", evaluation.get("env_eval_sha256"), "eval env")
    sealed_eval_raw = _load_mapping(ensemble_dir / "env_eval.yaml", "ensemble eval env")
    if dict(sealed_eval_raw) != dict(expected_eval_raw):
        raise TrainerConfigError(
            f"sealed ensemble eval env differs from the fold config: {ensemble_dir}"
        )
    device = _sealed_device(manifest, manifest_path)
    members: list[tuple[Any, Any, dict[str, Any], dict[str, Any], dict[str, Any], str]] = []
    for member_index, member in enumerate(members_raw):
        member_raw = _require_mapping(member, "ensemble member")
        declared_seed = member_raw.get("seed")
        if declared_seed != expected_seeds[member_index]:
            raise TrainerConfigError(
                f"ensemble member seed differs at index {member_index}: {manifest_path}"
            )
        run_dir = Path(_require_string(member_raw.get("run_dir"), "member.run_dir")).resolve()
        model_name = _require_string(member_raw.get("model_path"), "member.model_path")
        if model_name != "model_final.zip":
            raise TrainerConfigError(
                f"ensemble member model_path must be model_final.zip: {manifest_path}"
            )
        model_path = run_dir / model_name
        _require_hash(model_path, member_raw.get("model_sha256"), "member model")
        _require_hash(
            run_dir / "config_snapshot.yaml",
            member_raw.get("config_snapshot_sha256"),
            "member config",
        )
        _require_hash(run_dir / "meta.json", member_raw.get("meta_sha256"), "member meta")
        loaded = load_member_for_device(run_dir, device)
        loaded_config = loaded[0]
        loaded_meta = loaded[4]
        if loaded_config.run.seed != declared_seed:
            raise TrainerConfigError(
                f"ensemble member seed contradicts its config: {run_dir}"
            )
        if loaded_config.experiment != member_raw.get("experiment"):
            raise TrainerConfigError(
                f"ensemble member experiment contradicts its config: {run_dir}"
            )
        if loaded_meta.get("seed") != declared_seed:
            raise TrainerConfigError(
                f"ensemble member seed contradicts its metadata: {run_dir}"
            )
        if loaded_meta.get("experiment") != member_raw.get("experiment"):
            raise TrainerConfigError(
                f"ensemble member experiment contradicts its metadata: {run_dir}"
            )
        members.append(loaded)
    reference_config, _, eval_raw, raw_config, _, _ = members[0]
    if eval_raw != dict(expected_eval_raw):
        raise TrainerConfigError(f"ensemble eval environment differs: {ensemble_dir}")
    if reference_config.run.residual is not None or reference_config.run.rank_allocation is not None:
        raise TrainerConfigError(f"PPO benchmark must expose direct scores: {ensemble_dir}")
    if (
        reference_config.algorithm.name != "ppo"
        or reference_config.network.name != "mlp"
        or reference_config.run.apply_hold_gate is not None
    ):
        raise TrainerConfigError(
            f"PPO benchmark must use direct ungated PPO+MLP: {ensemble_dir}"
        )
    for member in members[1:]:
        if member[2] != eval_raw:
            raise TrainerConfigError(f"ensemble members have different eval envs: {ensemble_dir}")
    env = build_single_env(
        eval_raw,
        reference_config.custom_feature_names,
        reference_config.custom_cross_feature_names,
        seed=0,
        decision_interval=1,
        residual=None,
        rank_allocation=None,
        apply_hold_gate=None,
        gate_evaluation_mode=GateEvaluationMode.LEARNED,
    )
    states: list[Any] = [None] * len(members)
    observation, info = env.reset(seed=0)
    timestamps = [str(info["timestamp"])]
    equities = [float(info["equity_jpy"])]
    rewards: list[float] = []
    costs: list[float] = []
    leverages: list[float] = []
    turnovers: list[float] = []
    scores: list[np.ndarray] = []
    decision_times: list[datetime] = []
    previous_weights: np.ndarray | None = None
    episode_start = np.ones((1,), dtype=bool)
    terminated = False
    try:
        while True:
            actions: list[np.ndarray] = []
            for index, member in enumerate(members):
                action, states[index] = member[1].predict(
                    observation,
                    state=states[index],
                    episode_start=episode_start,
                    deterministic=True,
                )
                actions.append(np.asarray(action, dtype=np.float64))
            mean_action = np.mean(np.stack(actions), axis=0).astype(np.float32)
            if mean_action.shape != (len(expected_symbols), 1):
                raise TrainerConfigError(
                    f"PPO direct score shape is invalid: {mean_action.shape}"
                )
            decision_times.append(datetime.fromisoformat(str(info["timestamp"])))
            scores.append(mean_action[:, 0].astype(np.float64))
            observation, reward, terminated, truncated, info = env.step(mean_action)
            rewards.append(float(reward))
            equities.append(float(info["equity_jpy"]))
            timestamps.append(str(info["timestamp"]))
            costs.append(float(info["costs_jpy"]["total"]))
            leverages.append(float(info["gross_leverage"]))
            target = np.asarray(info["target_weights"], dtype=np.float64)
            if previous_weights is None:
                previous_weights = np.zeros_like(target)
            turnovers.append(float(np.abs(target - previous_weights).sum()))
            previous_weights = target
            episode_start = np.asarray([terminated or truncated], dtype=bool)
            if terminated or truncated:
                break
    finally:
        env.close()
    replay_metrics = compute_metrics(
        rewards, equities, timestamps, costs, leverages, turnovers, terminated
    )
    sealed_metrics = _load_mapping(ensemble_dir / "metrics.json", "sealed metrics")
    _metrics_match(replay_metrics, sealed_metrics, ensemble_dir / "metrics.json")
    data_identity = data_identity_from_config(raw_config, manifest_path)
    if dict(evaluation_data_identity) != data_identity:
        raise TrainerConfigError(f"ensemble data identity differs: {manifest_path}")
    trace = PpoScoreTrace(
        scores=np.stack(scores),
        decision_timestamps=tuple(decision_times),
        symbols=expected_symbols,
    )
    source = {
        "ensemble_dir": str(ensemble_dir),
        "ensemble_manifest_sha256": sha256_file(manifest_path),
        "metrics_sha256": sha256_file(ensemble_dir / "metrics.json"),
        "eval_env_sha256": sha256_file(ensemble_dir / "env_eval.yaml"),
        "sealed_device": device,
        "evaluation_git": dict(evaluation_git),
        "evaluation_versions": dict(evaluation_versions),
        "data_identity": data_identity,
        "member_model_sha256": [member["model_sha256"] for member in members_raw],
    }
    return trace, source


def _diagnostic_row(
    fold: str,
    era: str,
    score_name: str,
    diagnostics: ScoreDiagnostics,
    selected_alpha: float,
) -> dict[str, Any]:
    """Flatten one score diagnostic for CSV and aggregation.

    Args:
        fold: Evaluation-year identifier.
        era: Declared era name.
        score_name: Supervised, reversal, or PPO score label.
        diagnostics: Fold-reduced predictive diagnostics.
        selected_alpha: Validation-selected supervised penalty.

    Returns:
        Stable flat artifact row.
    """
    return {
        "fold": fold,
        "era": era,
        "score": score_name,
        "selected_alpha": selected_alpha if score_name == "supervised" else "",
        **asdict(diagnostics),
    }


def _mean_score_alignment(left: np.ndarray, right: np.ndarray) -> float | None:
    """Average decision-level rank association between two score matrices.

    Args:
        left: First decision-by-pair score matrix.
        right: Second score matrix with identical shape and pair order.

    Returns:
        Mean Spearman association over decisions with defined ranks, or None
        when neither score has a defined cross-sectional ranking.

    Raises:
        ValueError: If matrix shapes differ.
    """
    left_values = np.asarray(left, dtype=np.float64)
    right_values = np.asarray(right, dtype=np.float64)
    if left_values.ndim != 2 or right_values.shape != left_values.shape:
        raise ValueError("score alignment requires equal decision-by-pair matrices.")
    values = [
        cross_sectional_spearman(left_row, right_row)
        for left_row, right_row in zip(left_values, right_values)
        if float(left_row.std()) > 0.0 and float(right_row.std()) > 0.0
    ]
    return float(np.mean(values)) if values else None


def _aggregate_rows(
    fold_rows: Sequence[Mapping[str, Any]], study: SupervisedStudy
) -> Mapping[str, Any]:
    """Aggregate equal-weight fold observations by score, era, and overall.

    Args:
        fold_rows: One reduced diagnostic row per fold and score.
        study: Validated study controls and era definitions.

    Returns:
        Nested fold-equal aggregate and bootstrap evidence.
    """
    scores = ("supervised", "reversal", "ppo")
    report: dict[str, Any] = {}
    for score in scores:
        rows = [row for row in fold_rows if row["score"] == score]
        rows.sort(key=lambda row: int(str(row["fold"])))
        overall: dict[str, float | None] = {}
        for field in _DIAGNOSTIC_FIELDS:
            values = [float(row[field]) for row in rows if row[field] is not None]
            overall[field] = (
                float(
                    np.median(values)
                    if field == "median_rank_ic"
                    else np.mean(values)
                )
                if values
                else None
            )
        intervals: dict[str, Any] = {}
        for field in _DIAGNOSTIC_FIELDS:
            if field == "median_rank_ic":
                continue
            values = [float(row[field]) for row in rows if row[field] is not None]
            if len(values) < study.moving_block_length:
                intervals[field] = None
                continue
            interval = bootstrap_mean_intervals(
                values,
                study.bootstrap_samples,
                study.bootstrap_seed,
                study.moving_block_length,
            )
            intervals[field] = asdict(interval)
        eras: dict[str, Any] = {}
        for era in study.eras:
            era_rows = [row for row in rows if row["era"] == era.name]
            era_values: dict[str, float | None] = {}
            for field in _DIAGNOSTIC_FIELDS:
                values = [
                    float(row[field]) for row in era_rows if row[field] is not None
                ]
                era_values[field] = (
                    float(
                        np.median(values)
                        if field == "median_rank_ic"
                        else np.mean(values)
                    )
                    if values
                    else None
                )
            eras[era.name] = era_values
        report[score] = {
            "fold_count": len(rows),
            "overall": overall,
            "eras": eras,
            "bootstrap_intervals": intervals,
        }
    return report


def _coefficient_summaries(
    coefficient_rows: Sequence[Mapping[str, Any]], study: SupervisedStudy
) -> list[Mapping[str, Any]]:
    """Summarize standardized coefficients across folds and declared eras.

    Args:
        coefficient_rows: One coefficient row per fold and expanded feature.
        study: Study defining the era boundaries.

    Returns:
        Aggregate and era coefficient sign/magnitude rows for every feature.

    Raises:
        ValueError: If a scope lacks a complete feature observation.
    """
    feature_names = tuple(
        dict.fromkeys(str(row["feature"]) for row in coefficient_rows)
    )
    scopes: list[tuple[str, Sequence[Mapping[str, Any]]]] = [
        ("aggregate", coefficient_rows)
    ]
    for era in study.eras:
        scopes.append(
            (
                f"era:{era.name}",
                [
                    row
                    for row in coefficient_rows
                    if era.start <= int(str(row["fold"])) <= era.end
                ],
            )
        )
    summaries: list[Mapping[str, Any]] = []
    for scope_name, scope_rows in scopes:
        for feature in feature_names:
            values = np.asarray(
                [
                    float(row["standardized_coefficient"])
                    for row in scope_rows
                    if row["feature"] == feature
                ],
                dtype=np.float64,
            )
            if len(values) < 1 or not np.isfinite(values).all():
                raise ValueError(
                    f"coefficient scope {scope_name} lacks finite values for {feature}."
                )
            mean = float(values.mean())
            summaries.append(
                {
                    "scope": scope_name,
                    "feature": feature,
                    "fold_count": len(values),
                    "mean_standardized_coefficient": mean,
                    "sign": "positive" if mean > 0.0 else "negative" if mean < 0.0 else "zero",
                    "mean_absolute_magnitude": float(np.abs(values).mean()),
                    "positive_fold_fraction": float(np.mean(values > 0.0)),
                    "negative_fold_fraction": float(np.mean(values < 0.0)),
                }
            )
    return summaries


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    """Write non-empty mapping rows with stable first-row column order.

    Args:
        path: Output CSV path.
        rows: Non-empty rows sharing one exact field set.

    Raises:
        ValueError: If rows are empty or fields differ.
    """
    if not rows:
        raise ValueError(f"cannot write empty CSV artifact {path}.")
    fields = list(rows[0])
    if any(set(row) != set(fields) for row in rows):
        raise ValueError(f"CSV rows have inconsistent fields for {path}.")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, value: Any) -> None:
    """Write deterministic indented JSON.

    Args:
        path: Output JSON path.
        value: JSON-serializable value.
    """
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _render_markdown(report: Mapping[str, Any]) -> str:
    """Render a compact human-readable predictive report.

    Args:
        report: Complete machine-readable study report.

    Returns:
        Markdown report text.
    """
    lines = [
        f"# {report['study']}",
        "",
        f"Classification: **{report['classification']}**",
        "",
        "| score | mean rank IC | mean top2-bottom2 spread | positive IC fraction |",
        "|---|---:|---:|---:|",
    ]
    aggregates = report["aggregates"]
    for score in ("supervised", "reversal", "ppo"):
        overall = aggregates[score]["overall"]
        mean_rank_ic = overall["mean_rank_ic"]
        mean_rank_ic_text = (
            f"{mean_rank_ic:.6g}" if mean_rank_ic is not None else "undefined"
        )
        lines.append(
            f"| {score} | {mean_rank_ic_text} | "
            f"{overall['mean_tail_spread']:.6g} | "
            f"{overall['positive_rank_ic_fraction']:.6g} |"
        )
    tail_interval = aggregates["supervised"]["bootstrap_intervals"]["mean_tail_spread"]
    lines.extend(
        [
            "",
            "The primary tail-spread 95% intervals use evaluation folds as the sampling unit.",
            "",
            f"- IID fold: [{tail_interval['fold_low']:.6g}, {tail_interval['fold_high']:.6g}]",
            f"- Circular moving block (3 folds): [{tail_interval['moving_block_low']:.6g}, {tail_interval['moving_block_high']:.6g}]",
            "- Scores are predictive diagnostics only; no transaction-cost or portfolio optimization is performed.",
            "",
        ]
    )
    return "\n".join(lines)


def _study_versions() -> Mapping[str, str]:
    """Seal every dependency directly used by prediction artifact generation."""
    versions = dict(dependency_versions())
    for package in ("numpy", "pandas", "PyYAML", "pyarrow"):
        versions[package] = importlib_metadata.version(package)
    return versions


def run_supervised_study(
    study_path: Path, output_dir: Path
) -> tuple[Path, Mapping[str, Any]]:
    """Run all folds and atomically publish sealed predictive artifacts.

    Args:
        study_path: Strict study manifest path.
        output_dir: New destination directory.

    Returns:
        Published directory and complete report mapping.
    """
    study = load_supervised_study(study_path)
    destination = output_dir.resolve()
    if destination.exists():
        raise TrainerConfigError(
            f"supervised study output directory must not exist: {destination}"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent))
    try:
        fold_rows: list[Mapping[str, Any]] = []
        prediction_rows: list[Mapping[str, Any]] = []
        coefficient_rows: list[Mapping[str, Any]] = []
        model_records: dict[str, Any] = {}
        sources: dict[str, Any] = {}
        current_momentum_coefficients: list[float] = []
        reversal_alignment_rows: list[Mapping[str, Any]] = []
        canonical_data_identity: Mapping[str, str] | None = None
        canonical_evaluation_git: Mapping[str, str] | None = None
        canonical_evaluation_versions: Mapping[str, str] | None = None
        for fold, source in study.folds.items():
            config, raw_config = load_experiment_config(source.config)
            _validate_longf_config(config, fold)
            train_raw = resolve_env_raw(config.env, config.train_range, for_eval=True)
            validation_raw = resolve_env_raw(config.env, config.val_range, for_eval=True)
            evaluation_raw = resolve_env_raw(config.env, config.eval_range, for_eval=True)
            training = build_dataset_from_env_raw(
                train_raw, config.custom_feature_names, config.custom_cross_feature_names
            )
            validation = build_dataset_from_env_raw(
                validation_raw, config.custom_feature_names, config.custom_cross_feature_names
            )
            evaluation = build_dataset_from_env_raw(
                evaluation_raw, config.custom_feature_names, config.custom_cross_feature_names
            )
            if not (
                training.symbols == validation.symbols == evaluation.symbols
                and training.feature_names
                == validation.feature_names
                == evaluation.feature_names
            ):
                raise TrainerConfigError(f"fold {fold} dataset axes differ across ranges.")
            standardizer = fit_standardizer(training.features)
            standardized_training = _transformed_dataset(
                training, apply_standardizer(training.features, standardizer)
            )
            standardized_validation = _transformed_dataset(
                validation, apply_standardizer(validation.features, standardizer)
            )
            standardized_evaluation = _transformed_dataset(
                evaluation, apply_standardizer(evaluation.features, standardizer)
            )
            selection = select_validation_alpha(
                standardized_training, standardized_validation, study.alpha_grid
            )
            model = fit_ridge(standardized_training, selection.selected_alpha)
            supervised_scores = model.predict(standardized_evaluation.features)
            reversal_scores = canonical_reversal_scores(evaluation)
            ppo_trace, ppo_source = _replay_ppo_scores(
                source.ppo_ensemble,
                evaluation_raw,
                evaluation.symbols,
                study.member_seeds,
            )
            require_score_alignment(evaluation, ppo_trace)
            era = _era_name(study, fold)
            score_matrices = {
                "supervised": supervised_scores,
                "reversal": reversal_scores,
                "ppo": ppo_trace.scores,
            }
            reversal_alignment_rows.append(
                {
                    "fold": fold,
                    "era": era,
                    "mean_supervised_reversal_rank_alignment": _mean_score_alignment(
                        supervised_scores, reversal_scores
                    ),
                }
            )
            for score_name, matrix in score_matrices.items():
                fold_rows.append(
                    _diagnostic_row(
                        fold,
                        era,
                        score_name,
                        compute_score_diagnostics(matrix, evaluation.targets, study.top_k),
                        selection.selected_alpha,
                    )
                )
            for decision_index, (decision, target_time) in enumerate(
                zip(evaluation.decision_timestamps, evaluation.target_timestamps)
            ):
                for pair_index, symbol in enumerate(evaluation.symbols):
                    prediction_rows.append(
                        {
                            "fold": fold,
                            "era": era,
                            "decision_timestamp": decision.isoformat(),
                            "target_timestamp": target_time.isoformat(),
                            "pair": symbol,
                            "target_relative_log_return": evaluation.targets[
                                decision_index, pair_index
                            ],
                            "supervised_score": supervised_scores[
                                decision_index, pair_index
                            ],
                            "reversal_score": reversal_scores[
                                decision_index, pair_index
                            ],
                            "ppo_score": ppo_trace.scores[decision_index, pair_index],
                        }
                    )
            for feature_name, mean, scale, coefficient in zip(
                evaluation.feature_names,
                standardizer.mean,
                standardizer.scale,
                model.coefficients,
            ):
                coefficient_rows.append(
                    {
                        "fold": fold,
                        "feature": feature_name,
                        "train_mean": mean,
                        "train_scale": scale,
                        "standardized_coefficient": coefficient,
                        "sign": "positive" if coefficient > 0.0 else "negative" if coefficient < 0.0 else "zero",
                        "absolute_magnitude": abs(coefficient),
                    }
                )
            momentum_index = evaluation.feature_names.index("mom24_lag_0")
            current_momentum_coefficients.append(float(model.coefficients[momentum_index]))
            model_records[fold] = {
                "selected_alpha": selection.selected_alpha,
                "validation_mean_rank_ic_by_alpha": {
                    str(alpha): value
                    for alpha, value in selection.mean_rank_ic_by_alpha.items()
                },
                "intercept": model.intercept,
                "feature_names": list(evaluation.feature_names),
                "train_mean": standardizer.mean.tolist(),
                "train_scale": standardizer.scale.tolist(),
                "coefficients": model.coefficients.tolist(),
                "train_decisions": len(training.decision_timestamps),
                "validation_decisions": len(validation.decision_timestamps),
                "evaluation_decisions": len(evaluation.decision_timestamps),
            }
            fold_data_identity = data_identity_from_config(raw_config, source.config)
            ppo_data_identity = ppo_source["data_identity"]
            if fold_data_identity != ppo_data_identity:
                raise TrainerConfigError(
                    f"fold {fold} supervised and PPO data identities differ."
                )
            evaluation_git = ppo_source["evaluation_git"]
            evaluation_versions = ppo_source["evaluation_versions"]
            if canonical_data_identity is None:
                canonical_data_identity = fold_data_identity
                canonical_evaluation_git = evaluation_git
                canonical_evaluation_versions = evaluation_versions
            elif (
                fold_data_identity != canonical_data_identity
                or evaluation_git != canonical_evaluation_git
                or evaluation_versions != canonical_evaluation_versions
            ):
                raise TrainerConfigError(
                    f"fold {fold} differs from the canonical data or PPO evaluation contract."
                )
            sources[fold] = {
                "config_path": str(source.config),
                "config_sha256": sha256_file(source.config),
                "data_identity": fold_data_identity,
                "ppo": ppo_source,
            }
        aggregates = _aggregate_rows(fold_rows, study)
        coefficient_summaries = _coefficient_summaries(coefficient_rows, study)
        supervised_rows = sorted(
            [row for row in fold_rows if row["score"] == "supervised"],
            key=lambda row: int(str(row["fold"])),
        )
        tail_values = np.asarray(
            [float(row["mean_tail_spread"]) for row in supervised_rows], dtype=float
        )
        leave_one_out = tuple(
            float(np.delete(tail_values, index).mean()) for index in range(len(tail_values))
        )
        tail_intervals = aggregates["supervised"]["bootstrap_intervals"][
            "mean_tail_spread"
        ]
        reversal_alignment_by_era: dict[str, float | None] = {}
        for era in study.eras:
            values = [
                float(row["mean_supervised_reversal_rank_alignment"])
                for row in reversal_alignment_rows
                if row["era"] == era.name
                and row["mean_supervised_reversal_rank_alignment"] is not None
            ]
            reversal_alignment_by_era[era.name] = (
                float(np.mean(values)) if values else None
            )
        coherent_reversal = all(
            row["mean_supervised_reversal_rank_alignment"] is not None
            and float(row["mean_supervised_reversal_rank_alignment"]) > 0.0
            for row in reversal_alignment_rows
        ) and all(
            value is not None and value > 0.0
            for value in reversal_alignment_by_era.values()
        )
        classification_inputs = ClassificationInputs(
            aggregate_mean_rank_ic=aggregates["supervised"]["overall"]["mean_rank_ic"],
            aggregate_mean_tail_spread=aggregates["supervised"]["overall"]["mean_tail_spread"],
            iid_tail_spread_low=tail_intervals["fold_low"],
            moving_block_tail_spread_low=tail_intervals["moving_block_low"],
            era_tail_spreads=tuple(
                aggregates["supervised"]["eras"][era.name]["mean_tail_spread"]
                for era in study.eras
            ),
            leave_one_fold_out_tail_spreads=leave_one_out,
            coherent_reversal=coherent_reversal,
            non_degenerate_scores=all(
                float(row["mean_score_dispersion"]) > 0.0 for row in supervised_rows
            ),
        )
        report: Mapping[str, Any] = {
            "artifact_version": 1,
            "study": study.name,
            "classification": classify_learnability(classification_inputs),
            "classification_inputs": asdict(classification_inputs),
            "protocol": {
                "target": "next-decision pair log return minus cross-sectional mean",
                "input": "32-lag by 8-feature per-pair longf market window",
                "pair_identity": False,
                "alpha_grid": list(study.alpha_grid),
                "alpha_selection": "validation mean cross-sectional Spearman rank IC",
                "alpha_tie_break": "stronger regularization",
                "top_k": study.top_k,
                "bootstrap_samples": study.bootstrap_samples,
                "bootstrap_seed": study.bootstrap_seed,
                "moving_block_length": study.moving_block_length,
            },
            "aggregates": aggregates,
            "selected_alpha_by_fold": {
                fold: model_records[fold]["selected_alpha"] for fold in _EXPECTED_FOLDS
            },
            "mean_current_mom24_standardized_coefficient": float(
                np.mean(current_momentum_coefficients)
            ),
            "coefficient_diagnostics": {
                "fold_artifact": "coefficients.csv",
                "era_and_aggregate_artifact": "coefficient_summary.csv",
            },
            "supervised_reversal_rank_alignment": {
                "overall": float(
                    np.mean(
                        [
                            row["mean_supervised_reversal_rank_alignment"]
                            for row in reversal_alignment_rows
                            if row["mean_supervised_reversal_rank_alignment"]
                            is not None
                        ]
                    )
                )
                if any(
                    row["mean_supervised_reversal_rank_alignment"] is not None
                    for row in reversal_alignment_rows
                )
                else None,
                "eras": reversal_alignment_by_era,
                "folds": {
                    str(row["fold"]): row[
                        "mean_supervised_reversal_rank_alignment"
                    ]
                    for row in reversal_alignment_rows
                },
            },
        }
        _write_csv(staging / "predictions.csv", prediction_rows)
        _write_csv(staging / "fold_metrics.csv", fold_rows)
        _write_csv(staging / "coefficients.csv", coefficient_rows)
        _write_csv(staging / "coefficient_summary.csv", coefficient_summaries)
        _write_json(staging / "models.json", model_records)
        _write_json(staging / "report.json", report)
        (staging / "report.md").write_text(_render_markdown(report), encoding="utf-8")
        (staging / "study_snapshot.yaml").write_text(
            study.source_path.read_text(encoding="utf-8"), encoding="utf-8"
        )
        artifact_names = (
            "predictions.csv",
            "fold_metrics.csv",
            "coefficients.csv",
            "coefficient_summary.csv",
            "models.json",
            "report.json",
            "report.md",
            "study_snapshot.yaml",
        )
        provenance = {
            "artifact_version": 1,
            "diagnostic_git": git_commits(),
            "diagnostic_versions": _study_versions(),
            "study_path": str(study.source_path),
            "study_sha256": sha256_file(study.source_path),
            "fold_sources": sources,
            "generated_artifact_sha256": {
                name: sha256_file(staging / name) for name in artifact_names
            },
        }
        _write_json(staging / "provenance.json", provenance)
        os.replace(staging, destination)
    except Exception:
        shutil.rmtree(staging)
        raise
    return destination, report


def main(argv: list[str] | None = None) -> int:
    """Run the sealed supervised ranking study CLI.

    Args:
        argv: CLI arguments, or None to use ``sys.argv``.

    Returns:
        Zero on success and one for explicit validation/runtime errors.
    """
    parser = argparse.ArgumentParser(
        prog="forex-supervised-ranking",
        description="Run the fixed 17-fold supervised ranking diagnostic.",
    )
    parser.add_argument("--study", required=True, help="Strict study YAML path.")
    parser.add_argument("--output-dir", required=True, help="New artifact directory.")
    args = parser.parse_args(argv)
    try:
        output, report = run_supervised_study(Path(args.study), Path(args.output_dir))
    except (
        TrainerConfigError,
        ConfigError,
        DataError,
        FeatureError,
        OSError,
        ValueError,
    ) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"supervised ranking study: {output}")
    print(f"classification: {report['classification']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
