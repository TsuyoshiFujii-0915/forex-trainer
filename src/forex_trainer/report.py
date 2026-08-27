"""Reproducible aggregate reporting for fold/seed research campaigns."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import statistics
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from .artifact_provenance import data_identity_from_config, sha256_file
from .config import TrainerConfigError
from .research_statistics import BootstrapIntervals, bootstrap_mean_intervals

_CAMPAIGN_KEYS: tuple[str, ...] = (
    "name",
    "configurations",
    "comparisons",
    "eras",
    "bootstrap_samples",
    "bootstrap_seed",
    "moving_block_length",
    "trial_count",
)
_CONFIGURATION_KEYS: tuple[str, ...] = ("model_selection", "runs")
_COMPARISON_KEYS: tuple[str, ...] = ("baseline", "candidate")
_ERA_KEYS: tuple[str, ...] = ("start", "end")
_METRICS: tuple[str, ...] = (
    "annualized_net_return",
    "annualized_gross_return",
    "sharpe_annualized",
    "max_drawdown",
)
_SECONDS_PER_YEAR = 365.25 * 86_400.0


@dataclass(frozen=True)
class ConfigurationSpec:
    """One named configuration and its explicit run artifacts."""

    name: str
    model_selection: str
    run_dirs: tuple[Path, ...]


@dataclass(frozen=True)
class ComparisonSpec:
    """One named, direction-preserving paired comparison."""

    baseline: str
    candidate: str


@dataclass(frozen=True)
class EraSpec:
    """Inclusive evaluation-fold year range."""

    name: str
    start: int
    end: int


@dataclass(frozen=True)
class Campaign:
    """Validated aggregate-report campaign definition."""

    name: str
    configurations: tuple[ConfigurationSpec, ...]
    comparisons: tuple[ComparisonSpec, ...]
    eras: tuple[EraSpec, ...]
    bootstrap_samples: int
    bootstrap_seed: int
    moving_block_length: int
    trial_count: int
    source_path: Path


@dataclass(frozen=True)
class Observation:
    """Validated metrics and provenance for one fold/seed run."""

    configuration: str
    fold: str
    seed: int
    run_dir: Path
    experiment: str
    eval_start: str
    eval_end: str
    metrics: Mapping[str, float]
    requested_device: str
    device: str
    protocol_sha256: str
    data_identity: Mapping[str, str]
    git: Mapping[str, str]
    versions: Mapping[str, str]
    model_selection: str


def _require_exact_keys(
    value: Mapping[str, Any], expected: Sequence[str], origin: str
) -> None:
    """Require a mapping to contain exactly the declared keys.

    Args:
        value: Mapping under validation.
        expected: Complete supported key sequence.
        origin: Human-readable source for diagnostics.

    Raises:
        TrainerConfigError: If keys are missing or unknown.
    """
    expected_set = set(expected)
    actual = set(value)
    if actual != expected_set:
        raise TrainerConfigError(
            f"{origin} must contain exactly {list(expected)}; "
            f"missing={sorted(expected_set - actual)}, "
            f"unknown={sorted(actual - expected_set)}."
        )


def _require_string(value: Any, origin: str) -> str:
    """Require one non-empty string.

    Args:
        value: Candidate value.
        origin: Human-readable source for diagnostics.

    Returns:
        Validated string.

    Raises:
        TrainerConfigError: If the value is not a non-empty string.
    """
    if not isinstance(value, str) or not value:
        raise TrainerConfigError(f"{origin} must be a non-empty string.")
    return value


def _require_integer(value: Any, origin: str, minimum: int) -> int:
    """Require an integer at or above a minimum.

    Args:
        value: Candidate value.
        origin: Human-readable source for diagnostics.
        minimum: Inclusive minimum.

    Returns:
        Validated integer.

    Raises:
        TrainerConfigError: If the value is invalid.
    """
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise TrainerConfigError(
            f"{origin} must be an integer >= {minimum}, got {value!r}."
        )
    return value


def load_campaign(campaign_path: Path) -> Campaign:
    """Load a strict campaign YAML without re-parsing legacy experiment schemas.

    Args:
        campaign_path: Campaign YAML path.

    Returns:
        Validated campaign with absolute run paths.

    Raises:
        TrainerConfigError: If the campaign contract is invalid.
    """
    path = campaign_path.resolve()
    if not path.is_file():
        raise TrainerConfigError(f"Campaign file not found: {path}")
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise TrainerConfigError(f"Failed to parse campaign {path}: {exc}") from exc
    if not isinstance(raw, Mapping):
        raise TrainerConfigError(f"Campaign root must be a mapping: {path}")
    _require_exact_keys(raw, _CAMPAIGN_KEYS, "Campaign")
    name = _require_string(raw["name"], "Campaign name")

    raw_configurations = raw["configurations"]
    if not isinstance(raw_configurations, Mapping) or len(raw_configurations) < 2:
        raise TrainerConfigError(
            "Campaign configurations must be a mapping with at least two entries."
        )
    configurations: list[ConfigurationSpec] = []
    for configuration_name, raw_spec in raw_configurations.items():
        validated_name = _require_string(
            configuration_name, "Campaign configuration name"
        )
        if not isinstance(raw_spec, Mapping):
            raise TrainerConfigError(
                f"Configuration {validated_name} must be a mapping."
            )
        _require_exact_keys(
            raw_spec, _CONFIGURATION_KEYS, f"Configuration {validated_name}"
        )
        model_selection = _require_string(
            raw_spec["model_selection"],
            f"Configuration {validated_name} model_selection",
        )
        raw_runs = raw_spec["runs"]
        if not isinstance(raw_runs, list) or not raw_runs:
            raise TrainerConfigError(
                f"Configuration {validated_name} runs must be a non-empty list."
            )
        run_dirs: list[Path] = []
        for index, raw_run in enumerate(raw_runs):
            run_text = _require_string(
                raw_run, f"Configuration {validated_name} runs[{index}]"
            )
            candidate = Path(run_text)
            run_dir = (
                candidate if candidate.is_absolute() else path.parent / candidate
            ).resolve()
            if not run_dir.is_dir():
                raise TrainerConfigError(
                    f"Configuration {validated_name} run directory does not exist: "
                    f"{run_dir}"
                )
            run_dirs.append(run_dir)
        if len(set(run_dirs)) != len(run_dirs):
            raise TrainerConfigError(
                f"Configuration {validated_name} contains duplicate run directories."
            )
        configurations.append(
            ConfigurationSpec(validated_name, model_selection, tuple(run_dirs))
        )

    configuration_names = {item.name for item in configurations}
    raw_comparisons = raw["comparisons"]
    if not isinstance(raw_comparisons, list) or not raw_comparisons:
        raise TrainerConfigError("Campaign comparisons must be a non-empty list.")
    comparisons: list[ComparisonSpec] = []
    comparison_keys: set[tuple[str, str]] = set()
    for index, raw_comparison in enumerate(raw_comparisons):
        if not isinstance(raw_comparison, Mapping):
            raise TrainerConfigError(
                f"Campaign comparisons[{index}] must be a mapping."
            )
        _require_exact_keys(
            raw_comparison, _COMPARISON_KEYS, f"Campaign comparisons[{index}]"
        )
        baseline = _require_string(
            raw_comparison["baseline"], f"Campaign comparisons[{index}] baseline"
        )
        candidate = _require_string(
            raw_comparison["candidate"], f"Campaign comparisons[{index}] candidate"
        )
        if baseline == candidate:
            raise TrainerConfigError(
                f"Campaign comparison {baseline!r} cannot compare itself."
            )
        unknown = {baseline, candidate} - configuration_names
        if unknown:
            raise TrainerConfigError(
                f"Campaign comparison references unknown configurations: "
                f"{sorted(unknown)}."
            )
        key = (baseline, candidate)
        if key in comparison_keys:
            raise TrainerConfigError(f"Duplicate campaign comparison: {key}.")
        comparison_keys.add(key)
        comparisons.append(ComparisonSpec(baseline, candidate))

    raw_eras = raw["eras"]
    if not isinstance(raw_eras, Mapping) or not raw_eras:
        raise TrainerConfigError("Campaign eras must be a non-empty mapping.")
    eras: list[EraSpec] = []
    for era_name, raw_era in raw_eras.items():
        validated_era_name = _require_string(era_name, "Campaign era name")
        if not isinstance(raw_era, Mapping):
            raise TrainerConfigError(
                f"Campaign era {validated_era_name} must be a mapping."
            )
        _require_exact_keys(raw_era, _ERA_KEYS, f"Campaign era {validated_era_name}")
        start = _require_integer(
            raw_era["start"], f"Campaign era {validated_era_name} start", 1
        )
        end = _require_integer(
            raw_era["end"], f"Campaign era {validated_era_name} end", start
        )
        eras.append(EraSpec(validated_era_name, start, end))

    trial_count = _require_integer(raw["trial_count"], "Campaign trial_count", 1)
    if trial_count < len(configurations):
        raise TrainerConfigError(
            f"Campaign trial_count {trial_count} is smaller than the "
            f"{len(configurations)} reported configurations."
        )
    return Campaign(
        name=name,
        configurations=tuple(configurations),
        comparisons=tuple(comparisons),
        eras=tuple(eras),
        bootstrap_samples=_require_integer(
            raw["bootstrap_samples"], "Campaign bootstrap_samples", 1
        ),
        bootstrap_seed=_require_integer(
            raw["bootstrap_seed"], "Campaign bootstrap_seed", 0
        ),
        moving_block_length=_require_integer(
            raw["moving_block_length"], "Campaign moving_block_length", 1
        ),
        trial_count=trial_count,
        source_path=path,
    )


def _load_json_mapping(path: Path, artifact_name: str) -> Mapping[str, Any]:
    """Load a JSON artifact and require a mapping root.

    Args:
        path: JSON path.
        artifact_name: Diagnostic artifact label.

    Returns:
        Parsed mapping.

    Raises:
        TrainerConfigError: If the artifact is missing or malformed.
    """
    if not path.is_file():
        raise TrainerConfigError(
            f"Required {artifact_name} artifact is missing: {path}"
        )
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise TrainerConfigError(
            f"Malformed {artifact_name} artifact {path}: {exc}"
        ) from exc
    if not isinstance(value, Mapping):
        raise TrainerConfigError(
            f"{artifact_name} artifact root must be a mapping: {path}"
        )
    return value


def _load_yaml_mapping(path: Path, artifact_name: str) -> Mapping[str, Any]:
    """Load a YAML artifact and require a mapping root.

    Args:
        path: YAML path.
        artifact_name: Diagnostic artifact label.

    Returns:
        Parsed mapping.

    Raises:
        TrainerConfigError: If the artifact is missing or malformed.
    """
    if not path.is_file():
        raise TrainerConfigError(
            f"Required {artifact_name} artifact is missing: {path}"
        )
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise TrainerConfigError(
            f"Malformed {artifact_name} artifact {path}: {exc}"
        ) from exc
    if not isinstance(value, Mapping):
        raise TrainerConfigError(
            f"{artifact_name} artifact root must be a mapping: {path}"
        )
    return value


def _require_mapping_member(
    value: Mapping[str, Any], key: str, origin: str
) -> Mapping[str, Any]:
    """Read one required mapping member.

    Args:
        value: Parent mapping.
        key: Required member name.
        origin: Human-readable source for diagnostics.

    Returns:
        Child mapping.

    Raises:
        TrainerConfigError: If the member is absent or not a mapping.
    """
    if key not in value or not isinstance(value[key], Mapping):
        raise TrainerConfigError(f"{origin} requires mapping field {key!r}.")
    return value[key]


def _require_finite_metric(metrics: Mapping[str, Any], name: str, path: Path) -> float:
    """Read one required finite numeric metric.

    Args:
        metrics: Metrics artifact mapping.
        name: Required metric name.
        path: Artifact path for diagnostics.

    Returns:
        Metric as a float.

    Raises:
        TrainerConfigError: If the metric is missing, non-numeric, or non-finite.
    """
    if name not in metrics:
        raise TrainerConfigError(
            f"Metrics artifact {path} lacks required metric {name}."
        )
    value = metrics[name]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TrainerConfigError(f"Metric {name} in {path} must be numeric.")
    number = float(value)
    if not math.isfinite(number):
        raise TrainerConfigError(
            f"Metric {name} in {path} must be finite, got {number}."
        )
    return number


def _annualize_log_return(log_return: float, years: float, path: Path) -> float:
    """Convert a finite cumulative log return to a finite annualized return.

    Args:
        log_return: Cumulative evaluation-period log return.
        years: Positive evaluation duration in years.
        path: Metrics artifact path for diagnostics.

    Returns:
        Finite annualized simple return.

    Raises:
        TrainerConfigError: If exponentiation overflows or is non-finite.
    """
    try:
        annualized = math.expm1(log_return / years)
    except OverflowError as exc:
        raise TrainerConfigError(
            f"Annualized return overflows for {path}: "
            f"log_return={log_return}, years={years}."
        ) from exc
    if not math.isfinite(annualized):
        raise TrainerConfigError(
            f"Annualized return is non-finite for {path}: "
            f"log_return={log_return}, years={years}."
        )
    return float(annualized)


def _protocol_identity(config: Mapping[str, Any], config_path: Path) -> str:
    """Hash the invariant learning protocol of a raw config snapshot.

    Args:
        config: Raw config snapshot.
        config_path: Snapshot path for diagnostics.

    Returns:
        SHA-256 of the config after fold, date-range, and seed identity removal.

    Raises:
        TrainerConfigError: If required protocol fields are missing.
    """
    required = {
        "experiment",
        "env",
        "train_range",
        "val_range",
        "eval_range",
        "algorithm",
        "network",
        "run",
    }
    missing = required - set(config)
    if missing:
        raise TrainerConfigError(
            f"Config snapshot {config_path} lacks required fields {sorted(missing)}."
        )
    run = _require_mapping_member(config, "run", str(config_path))
    if "seed" not in run:
        raise TrainerConfigError(f"Config snapshot {config_path} lacks run.seed.")
    projection = {
        key: value
        for key, value in config.items()
        if key not in {"experiment", "train_range", "val_range", "eval_range"}
    }
    projected_run = dict(run)
    del projected_run["seed"]
    projection["run"] = projected_run
    encoded = json.dumps(projection, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _string_mapping(value: Any, origin: str) -> Mapping[str, str]:
    """Require a mapping with string keys and values.

    Args:
        value: Candidate mapping.
        origin: Human-readable source for diagnostics.

    Returns:
        Plain validated dictionary.

    Raises:
        TrainerConfigError: If any entry is not textual.
    """
    if not isinstance(value, Mapping) or not all(
        isinstance(key, str) and isinstance(item, str) for key, item in value.items()
    ):
        raise TrainerConfigError(f"{origin} must be a string-to-string mapping.")
    return dict(value)


def _require_sha256(value: Any, origin: str) -> str:
    """Require one lowercase hexadecimal SHA-256 digest.

    Args:
        value: Candidate digest.
        origin: Human-readable source for diagnostics.

    Returns:
        Validated digest.

    Raises:
        TrainerConfigError: If the digest is malformed.
    """
    digest = _require_string(value, origin)
    if len(digest) != 64 or any(
        character not in "0123456789abcdef" for character in digest
    ):
        raise TrainerConfigError(f"{origin} must be a lowercase SHA-256 digest.")
    return digest


def _load_observation(spec: ConfigurationSpec, run_dir: Path) -> Observation:
    """Load and validate one evaluated training-run artifact.

    Args:
        spec: Declared named configuration.
        run_dir: Run artifact directory.

    Returns:
        Canonical observation.

    Raises:
        TrainerConfigError: If artifact contents are incomplete or inconsistent.
    """
    config_path = run_dir / "config_snapshot.yaml"
    meta_path = run_dir / "meta.json"
    metrics_path = run_dir / "metrics.json"
    evaluation_path = run_dir / "evaluation.json"
    config = _load_yaml_mapping(config_path, "config_snapshot.yaml")
    meta = _load_json_mapping(meta_path, "meta.json")
    metrics = _load_json_mapping(metrics_path, "metrics.json")
    evaluation = _load_json_mapping(evaluation_path, "evaluation.json")
    _require_exact_keys(
        evaluation,
        (
            "manifest_version",
            "model_selection",
            "model_path",
            "model_sha256",
            "metrics_sha256",
        ),
        f"Evaluation manifest {evaluation_path}",
    )
    if evaluation["manifest_version"] != 1:
        raise TrainerConfigError(
            f"Evaluation manifest {evaluation_path} has unsupported "
            f"manifest_version={evaluation['manifest_version']!r}."
        )
    model_selection = _require_string(
        evaluation["model_selection"], f"{evaluation_path} model_selection"
    )
    if model_selection != "validation_best":
        raise TrainerConfigError(
            f"Standard run evaluation {evaluation_path} must record "
            f"model_selection='validation_best', got {model_selection!r}."
        )
    if model_selection != spec.model_selection:
        raise TrainerConfigError(
            f"Configuration {spec.name} declares model_selection "
            f"{spec.model_selection!r}, but {evaluation_path} records "
            f"{model_selection!r}."
        )
    model_name = _require_string(
        evaluation["model_path"], f"{evaluation_path} model_path"
    )
    if model_name != "model_final.zip":
        raise TrainerConfigError(
            f"Standard validation-best evaluation {evaluation_path} must use "
            f"model_final.zip, got {model_name!r}."
        )
    if Path(model_name).name != model_name:
        raise TrainerConfigError(
            f"Evaluation manifest {evaluation_path} model_path must be a file name, "
            f"got {model_name!r}."
        )
    model_path = run_dir / model_name
    if not model_path.is_file():
        raise TrainerConfigError(
            f"Evaluation model recorded by {evaluation_path} is missing: {model_path}"
        )
    expected_model_sha = _require_sha256(
        evaluation["model_sha256"], f"{evaluation_path} model_sha256"
    )
    if sha256_file(model_path) != expected_model_sha:
        raise TrainerConfigError(
            f"Evaluation model changed after {evaluation_path} was written: {model_path}"
        )
    expected_metrics_sha = _require_sha256(
        evaluation["metrics_sha256"], f"{evaluation_path} metrics_sha256"
    )
    if sha256_file(metrics_path) != expected_metrics_sha:
        raise TrainerConfigError(
            f"Metrics changed after {evaluation_path} was written: {metrics_path}"
        )

    eval_range = _require_mapping_member(config, "eval_range", str(config_path))
    if "start" not in eval_range or "end" not in eval_range:
        raise TrainerConfigError(
            f"Config snapshot {config_path} requires eval_range.start and eval_range.end."
        )
    config_eval_start = _require_string(
        eval_range["start"], f"{config_path} eval_range.start"
    )
    _require_string(eval_range["end"], f"{config_path} eval_range.end")
    try:
        fold = str(datetime.fromisoformat(config_eval_start).year)
    except ValueError as exc:
        raise TrainerConfigError(
            f"Invalid eval_range.start in {config_path}: {config_eval_start}"
        ) from exc

    for field in (
        "experiment",
        "seed",
        "requested_device",
        "device",
        "algorithm",
        "network",
        "git",
        "versions",
        "data_identity",
    ):
        if field not in meta:
            raise TrainerConfigError(
                f"Meta artifact {meta_path} lacks required field {field}."
            )
    experiment = _require_string(meta["experiment"], f"{meta_path} experiment")
    if "experiment" not in config:
        raise TrainerConfigError(
            f"Config snapshot {config_path} lacks required field experiment."
        )
    config_experiment = _require_string(
        config["experiment"], f"{config_path} experiment"
    )
    if experiment != config_experiment:
        raise TrainerConfigError(
            f"Experiment mismatch in {run_dir}: meta={experiment!r}, "
            f"config={config_experiment!r}."
        )
    seed = _require_integer(meta["seed"], f"{meta_path} seed", 0)
    requested_device = _require_string(
        meta["requested_device"], f"{meta_path} requested_device"
    )
    device = _require_string(meta["device"], f"{meta_path} device")
    run = _require_mapping_member(config, "run", str(config_path))
    if "seed" not in run or run["seed"] != seed:
        raise TrainerConfigError(
            f"Run seed mismatch in {run_dir}: meta={seed}, config={run.get('seed')!r}."
        )
    if "device" not in run or run["device"] != requested_device:
        raise TrainerConfigError(
            f"Run requested_device mismatch in {run_dir}: "
            f"meta={requested_device!r}, "
            f"config={run.get('device')!r}."
        )
    if device not in {"cpu", "cuda", "mps"}:
        raise TrainerConfigError(
            f"Resolved run device in {meta_path} must be cpu, cuda, or mps, "
            f"got {device!r}."
        )
    for field in ("algorithm", "network"):
        section = _require_mapping_member(config, field, str(config_path))
        if "name" not in section or section["name"] != meta[field]:
            raise TrainerConfigError(
                f"Run {field} mismatch in {run_dir}: meta={meta[field]!r}, "
                f"config={section.get('name')!r}."
            )
    recorded_data_identity = _string_mapping(
        meta["data_identity"], f"{meta_path} data_identity"
    )
    current_data_identity = data_identity_from_config(config, config_path)
    if recorded_data_identity != current_data_identity:
        raise TrainerConfigError(
            f"Run data_identity mismatch in {run_dir}: "
            f"recorded_at_training={recorded_data_identity!r}, "
            f"current={current_data_identity!r}."
        )

    if "eval_start" not in metrics or "eval_end" not in metrics:
        raise TrainerConfigError(
            f"Metrics artifact {metrics_path} requires eval_start and eval_end."
        )
    eval_start = _require_string(metrics["eval_start"], f"{metrics_path} eval_start")
    eval_end = _require_string(metrics["eval_end"], f"{metrics_path} eval_end")
    try:
        start = datetime.fromisoformat(eval_start)
        end = datetime.fromisoformat(eval_end)
    except ValueError as exc:
        raise TrainerConfigError(
            f"Invalid evaluation timestamp in {metrics_path}: "
            f"start={eval_start!r}, end={eval_end!r}."
        ) from exc
    if (start.tzinfo is None) != (end.tzinfo is None):
        raise TrainerConfigError(
            f"Evaluation timestamps in {metrics_path} must use the same timezone "
            f"awareness: start={eval_start!r}, end={eval_end!r}."
        )
    config_eval_end = _require_string(
        eval_range["end"], f"{config_path} eval_range.end"
    )
    try:
        configured_start = datetime.fromisoformat(config_eval_start).date()
        configured_end = datetime.fromisoformat(config_eval_end).date()
    except ValueError as exc:
        raise TrainerConfigError(
            f"Invalid configured evaluation range in {config_path}: "
            f"start={config_eval_start!r}, end={config_eval_end!r}."
        ) from exc
    if start.date() < configured_start or end.date() > configured_end:
        raise TrainerConfigError(
            f"Metrics evaluation interval in {metrics_path} falls outside the "
            f"configured eval_range: metrics=({eval_start}, {eval_end}), "
            f"config=({config_eval_start}, {config_eval_end})."
        )
    duration_years = (end - start).total_seconds() / _SECONDS_PER_YEAR
    if not math.isfinite(duration_years) or duration_years <= 0.0:
        raise TrainerConfigError(
            f"Evaluation duration in {metrics_path} must be positive, got "
            f"{duration_years}."
        )
    net_log = _require_finite_metric(metrics, "cumulative_log_return", metrics_path)
    gross_log = _require_finite_metric(
        metrics, "gross_cumulative_log_return", metrics_path
    )
    canonical_metrics = {
        "annualized_net_return": _annualize_log_return(
            net_log, duration_years, metrics_path
        ),
        "annualized_gross_return": _annualize_log_return(
            gross_log, duration_years, metrics_path
        ),
        "sharpe_annualized": _require_finite_metric(
            metrics, "sharpe_annualized", metrics_path
        ),
        "max_drawdown": _require_finite_metric(metrics, "max_drawdown", metrics_path),
    }
    return Observation(
        configuration=spec.name,
        fold=fold,
        seed=seed,
        run_dir=run_dir,
        experiment=experiment,
        eval_start=eval_start,
        eval_end=eval_end,
        metrics=canonical_metrics,
        requested_device=requested_device,
        device=device,
        protocol_sha256=_protocol_identity(config, config_path),
        data_identity=recorded_data_identity,
        git=_string_mapping(meta["git"], f"{meta_path} git"),
        versions=_string_mapping(meta["versions"], f"{meta_path} versions"),
        model_selection=model_selection,
    )


def _validate_configuration(
    spec: ConfigurationSpec, observations: Sequence[Observation]
) -> Mapping[str, Any]:
    """Require one complete and provenance-consistent fold/seed matrix.

    Args:
        spec: Named configuration declaration.
        observations: Loaded observations for that configuration.

    Returns:
        Common provenance record.

    Raises:
        TrainerConfigError: If matrix or provenance differs inside the group.
    """
    cells: dict[tuple[str, int], Observation] = {}
    for observation in observations:
        key = (observation.fold, observation.seed)
        if key in cells:
            raise TrainerConfigError(
                f"Configuration {spec.name} has duplicate fold/seed cell {key}: "
                f"{cells[key].run_dir} and {observation.run_dir}."
            )
        cells[key] = observation
    folds = sorted({observation.fold for observation in observations})
    seeds = sorted({observation.seed for observation in observations})
    expected = {(fold, seed) for fold in folds for seed in seeds}
    if set(cells) != expected:
        raise TrainerConfigError(
            f"Configuration {spec.name} has an incomplete fold/seed matrix: "
            f"missing={sorted(expected - set(cells))}."
        )
    for fold in folds:
        intervals = {
            (cells[(fold, seed)].eval_start, cells[(fold, seed)].eval_end)
            for seed in seeds
        }
        if len(intervals) != 1:
            raise TrainerConfigError(
                f"Configuration {spec.name} fold {fold} has mismatched effective "
                f"evaluation intervals: {sorted(intervals)}."
            )
    provenance_fields = {
        "protocol": {item.protocol_sha256 for item in observations},
        "requested_device": {item.requested_device for item in observations},
        "device": {item.device for item in observations},
        "data": {
            json.dumps(item.data_identity, sort_keys=True) for item in observations
        },
        "git": {json.dumps(item.git, sort_keys=True) for item in observations},
        "versions": {
            json.dumps(item.versions, sort_keys=True) for item in observations
        },
        "model_selection": {item.model_selection for item in observations},
    }
    for field, values in provenance_fields.items():
        if len(values) != 1:
            raise TrainerConfigError(
                f"Configuration {spec.name} has mismatched {field} provenance: "
                f"{sorted(values)}."
            )
    first = observations[0]
    return {
        "protocol_sha256": first.protocol_sha256,
        "requested_device": first.requested_device,
        "device": first.device,
        "data_identity": dict(first.data_identity),
        "git": dict(first.git),
        "versions": dict(first.versions),
        "model_selection": first.model_selection,
        "folds": folds,
        "seeds": seeds,
        "runs": [
            {
                "fold": item.fold,
                "seed": item.seed,
                "experiment": item.experiment,
                "run_dir": str(item.run_dir),
                "eval_start": item.eval_start,
                "eval_end": item.eval_end,
            }
            for item in sorted(observations, key=lambda row: (row.fold, row.seed))
        ],
    }


def _metric_means(observations: Sequence[Observation]) -> dict[str, float]:
    """Calculate arithmetic means of canonical report metrics.

    Args:
        observations: Non-empty observation sequence.

    Returns:
        Mean of every report metric.
    """
    return {
        name: float(statistics.fmean(item.metrics[name] for item in observations))
        for name in _METRICS
    }


def _interval_mapping(intervals: BootstrapIntervals) -> dict[str, float]:
    """Serialize bootstrap intervals with explicit method names.

    Args:
        intervals: Shared statistics result.

    Returns:
        JSON-serializable interval mapping.
    """
    return {
        "fold_bootstrap_95_low": intervals.fold_low,
        "fold_bootstrap_95_high": intervals.fold_high,
        "moving_block_95_low": intervals.moving_block_low,
        "moving_block_95_high": intervals.moving_block_high,
    }


def _configuration_report(
    observations: Sequence[Observation], campaign: Campaign
) -> Mapping[str, Any]:
    """Build fold, seed, era, overall, and uncertainty aggregates.

    Args:
        observations: Complete matrix for one configuration.
        campaign: Bootstrap and era controls.

    Returns:
        JSON-serializable configuration report.

    Raises:
        TrainerConfigError: If moving-block controls exceed available folds.
    """
    folds = sorted({item.fold for item in observations})
    seeds = sorted({item.seed for item in observations})
    if campaign.moving_block_length > len(folds):
        raise TrainerConfigError(
            f"Campaign moving_block_length {campaign.moving_block_length} exceeds "
            f"the {len(folds)} folds available for configuration "
            f"{observations[0].configuration}."
        )
    fold_report = {
        fold: _metric_means([item for item in observations if item.fold == fold])
        for fold in folds
    }
    seed_report = {
        str(seed): _metric_means([item for item in observations if item.seed == seed])
        for seed in seeds
    }
    overall = {
        name: float(statistics.fmean(fold_report[fold][name] for fold in folds))
        for name in _METRICS
    }
    overall["mean_max_drawdown"] = overall["max_drawdown"]
    overall["worst_max_drawdown"] = max(
        fold_report[fold]["max_drawdown"] for fold in folds
    )
    overall["winning_folds"] = sum(
        fold_report[fold]["annualized_net_return"] > 0.0 for fold in folds
    )
    overall["fold_count"] = len(folds)
    era_report: dict[str, Mapping[str, Any]] = {}
    for era in campaign.eras:
        era_folds = [fold for fold in folds if era.start <= int(fold) <= era.end]
        era_values: dict[str, Any] = {
            "fold_count": len(era_folds),
            "winning_folds": sum(
                fold_report[fold]["annualized_net_return"] > 0.0 for fold in era_folds
            ),
        }
        for name in _METRICS:
            era_values[name] = (
                float(statistics.fmean(fold_report[fold][name] for fold in era_folds))
                if era_folds
                else None
            )
        era_report[era.name] = era_values
    uncertainty = {
        name: _interval_mapping(
            bootstrap_mean_intervals(
                [fold_report[fold][name] for fold in folds],
                campaign.bootstrap_samples,
                campaign.bootstrap_seed,
                campaign.moving_block_length,
            )
        )
        for name in _METRICS
    }
    return {
        "observation_count": len(observations),
        "fold_count": len(folds),
        "seed_count": len(seeds),
        "overall": overall,
        "folds": fold_report,
        "seeds": seed_report,
        "eras": era_report,
        "uncertainty": uncertainty,
    }


def _require_comparable_provenance(
    comparison: ComparisonSpec,
    provenance: Mapping[str, Mapping[str, Any]],
) -> None:
    """Require paired groups to share non-treatment research conditions.

    Args:
        comparison: Directional comparison declaration.
        provenance: Per-configuration provenance records.

    Raises:
        TrainerConfigError: If data, resolved device, or model selection differs.
    """
    baseline = provenance[comparison.baseline]
    candidate = provenance[comparison.candidate]
    for field in ("device", "data_identity", "model_selection"):
        if baseline[field] != candidate[field]:
            raise TrainerConfigError(
                f"Comparison {comparison.baseline} -> {comparison.candidate} has "
                f"mismatched {field}: baseline={baseline[field]!r}, "
                f"candidate={candidate[field]!r}."
            )


def _provenance_differences(
    comparison: ComparisonSpec,
    provenance: Mapping[str, Mapping[str, Any]],
) -> Mapping[str, Mapping[str, Any]]:
    """Record visible non-blocking provenance differences between treatments.

    Args:
        comparison: Directional comparison declaration.
        provenance: Per-configuration provenance records.

    Returns:
        Differing implementation, dependency, device-request, and protocol values.
    """
    baseline = provenance[comparison.baseline]
    candidate = provenance[comparison.candidate]
    differences: dict[str, Mapping[str, Any]] = {}
    for field in ("protocol_sha256", "requested_device", "git", "versions"):
        if baseline[field] != candidate[field]:
            differences[field] = {
                "baseline": baseline[field],
                "candidate": candidate[field],
            }
    return differences


def _comparison_report(
    comparison: ComparisonSpec,
    observations: Mapping[str, Sequence[Observation]],
    provenance: Mapping[str, Mapping[str, Any]],
    campaign: Campaign,
) -> Mapping[str, Any]:
    """Build an exactly aligned candidate-minus-baseline comparison.

    Args:
        comparison: Directional comparison declaration.
        observations: Per-configuration complete matrices.
        provenance: Per-configuration provenance records.
        campaign: Bootstrap controls.

    Returns:
        Fold deltas, mean deltas, and fold-aware uncertainty.

    Raises:
        TrainerConfigError: If matrices or effective intervals differ.
    """
    _require_comparable_provenance(comparison, provenance)
    baseline_lookup = {
        (item.fold, item.seed): item for item in observations[comparison.baseline]
    }
    candidate_lookup = {
        (item.fold, item.seed): item for item in observations[comparison.candidate]
    }
    if set(baseline_lookup) != set(candidate_lookup):
        raise TrainerConfigError(
            f"Comparison {comparison.baseline} -> {comparison.candidate} has "
            "mismatched fold/seed matrix: "
            f"baseline_only={sorted(set(baseline_lookup) - set(candidate_lookup))}, "
            f"candidate_only={sorted(set(candidate_lookup) - set(baseline_lookup))}."
        )
    for key in sorted(baseline_lookup):
        baseline = baseline_lookup[key]
        candidate = candidate_lookup[key]
        if (baseline.eval_start, baseline.eval_end) != (
            candidate.eval_start,
            candidate.eval_end,
        ):
            raise TrainerConfigError(
                f"Comparison {comparison.baseline} -> {comparison.candidate} "
                f"fold/seed {key} has mismatched effective evaluation intervals: "
                f"baseline=({baseline.eval_start}, {baseline.eval_end}), "
                f"candidate=({candidate.eval_start}, {candidate.eval_end})."
            )
    folds = sorted({fold for fold, _ in baseline_lookup})
    fold_differences: dict[str, dict[str, float]] = {}
    for fold in folds:
        keys = sorted(key for key in baseline_lookup if key[0] == fold)
        fold_differences[fold] = {
            name: float(
                statistics.fmean(
                    candidate_lookup[key].metrics[name]
                    - baseline_lookup[key].metrics[name]
                    for key in keys
                )
            )
            for name in _METRICS
        }
    mean_differences = {
        name: float(statistics.fmean(fold_differences[fold][name] for fold in folds))
        for name in _METRICS
    }
    uncertainty = {
        name: _interval_mapping(
            bootstrap_mean_intervals(
                [fold_differences[fold][name] for fold in folds],
                campaign.bootstrap_samples,
                campaign.bootstrap_seed,
                campaign.moving_block_length,
            )
        )
        for name in _METRICS
    }
    return {
        "baseline": comparison.baseline,
        "candidate": comparison.candidate,
        "fold_count": len(folds),
        "seed_fold_pair_count": len(baseline_lookup),
        "folds": fold_differences,
        "mean_differences": mean_differences,
        "improved_folds": {
            name: sum(
                (fold_differences[fold][name] < 0.0)
                if name == "max_drawdown"
                else (fold_differences[fold][name] > 0.0)
                for fold in folds
            )
            for name in _METRICS
        },
        "uncertainty": uncertainty,
        "provenance_differences": _provenance_differences(comparison, provenance),
    }


def _atomic_write_text(path: Path, content: str) -> None:
    """Atomically write UTF-8 text in the destination directory.

    Args:
        path: Destination path.
        content: Complete file content.
    """
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(content, encoding="utf-8")
    os.replace(temporary, path)


def _atomic_write_json(path: Path, value: Any) -> None:
    """Atomically write deterministic JSON.

    Args:
        path: Destination path.
        value: JSON-serializable value.
    """
    _atomic_write_text(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def _write_observations_csv(path: Path, observations: Sequence[Observation]) -> None:
    """Write canonical run-level observations as CSV.

    Args:
        path: Destination CSV path.
        observations: All campaign observations.
    """
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "configuration",
                "fold",
                "seed",
                "experiment",
                "run_dir",
                "eval_start",
                "eval_end",
                *_METRICS,
            ),
            lineterminator="\n",
        )
        writer.writeheader()
        for item in sorted(
            observations,
            key=lambda row: (row.configuration, row.fold, row.seed),
        ):
            writer.writerow(
                {
                    "configuration": item.configuration,
                    "fold": item.fold,
                    "seed": item.seed,
                    "experiment": item.experiment,
                    "run_dir": str(item.run_dir),
                    "eval_start": item.eval_start,
                    "eval_end": item.eval_end,
                    **item.metrics,
                }
            )
    os.replace(temporary, path)


def _report_markdown(report: Mapping[str, Any]) -> str:
    """Render a compact human-readable campaign report.

    Args:
        report: Complete report mapping.

    Returns:
        Markdown document.
    """
    lines = [
        f"# {report['campaign']}",
        "",
        "## Aggregate evidence",
        "",
        "| Configuration | Net/year | Gross/year | Sharpe | Mean / worst drawdown | Winning folds |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, configuration in report["configurations"].items():
        overall = configuration["overall"]
        lines.append(
            f"| {name} | {overall['annualized_net_return']:.4%} | "
            f"{overall['annualized_gross_return']:.4%} | "
            f"{overall['sharpe_annualized']:.3f} | "
            f"{overall['mean_max_drawdown']:.4%} / "
            f"{overall['worst_max_drawdown']:.4%} | "
            f"{overall['winning_folds']}/{overall['fold_count']} |"
        )
    lines.extend(["", "## Paired comparisons", ""])
    for comparison in report["comparisons"]:
        delta = comparison["mean_differences"]["annualized_net_return"]
        interval = comparison["uncertainty"]["annualized_net_return"]
        lines.append(
            f"- {comparison['candidate']} − {comparison['baseline']}: mean net/year "
            f"difference {delta:.4%}; fold-bootstrap 95% CI "
            f"[{interval['fold_bootstrap_95_low']:.4%}, "
            f"{interval['fold_bootstrap_95_high']:.4%}]; moving-block 95% CI "
            f"[{interval['moving_block_95_low']:.4%}, "
            f"{interval['moving_block_95_high']:.4%}]."
        )
    assumptions = report["uncertainty_assumptions"]
    selection_bias = report["selection_bias"]
    lines.extend(
        [
            "",
            "## Statistical assumptions and provenance",
            "",
            f"- Sampling unit: {assumptions['sampling_unit']} after averaging seeds within each fold.",
            f"- Bootstrap draws: {assumptions['samples']}; deterministic seed: {assumptions['seed']}.",
            f"- Moving-block length: {assumptions['moving_block_length']} adjacent folds with circular wrapping.",
            f"- Campaign trials: {report['trial_count']}; reported configurations: {report['configuration_count']}.",
            f"- Selection-bias statistic: {selection_bias['status']}. {selection_bias['reason']}",
            "- Full run, config, data, device, software, and model-selection provenance is in `provenance.json`.",
        ]
    )
    return "\n".join(lines) + "\n"


def run_research_report(
    campaign_path: Path, output_dir: Path
) -> tuple[Path, Mapping[str, Any]]:
    """Generate one aggregate report from explicit run artifacts.

    Args:
        campaign_path: Strict campaign YAML.
        output_dir: New or existing report directory.

    Returns:
        Absolute output directory and complete report.

    Raises:
        TrainerConfigError: If artifacts, matrices, or provenance are invalid.
        OSError: If artifacts cannot be read or output cannot be written.
    """
    campaign = load_campaign(campaign_path)
    observations_by_configuration: dict[str, list[Observation]] = {}
    provenance: dict[str, Mapping[str, Any]] = {}
    for spec in campaign.configurations:
        observations = [_load_observation(spec, run_dir) for run_dir in spec.run_dirs]
        observations_by_configuration[spec.name] = observations
        provenance[spec.name] = _validate_configuration(spec, observations)
    configuration_reports = {
        name: _configuration_report(observations, campaign)
        for name, observations in observations_by_configuration.items()
    }
    comparison_reports = [
        _comparison_report(
            comparison, observations_by_configuration, provenance, campaign
        )
        for comparison in campaign.comparisons
    ]
    report: Mapping[str, Any] = {
        "campaign": campaign.name,
        "trial_count": campaign.trial_count,
        "configuration_count": len(campaign.configurations),
        "configurations": configuration_reports,
        "comparisons": comparison_reports,
        "uncertainty_assumptions": {
            "sampling_unit": "evaluation_fold",
            "seed_handling": "seeds_are_averaged_within_fold_before_resampling",
            "fold_bootstrap": "folds_are_sampled_with_replacement",
            "moving_block": "ordered_adjacent_folds_use_circular_moving_blocks",
            "samples": campaign.bootstrap_samples,
            "seed": campaign.bootstrap_seed,
            "moving_block_length": campaign.moving_block_length,
            "interval": "percentile_95",
        },
        "selection_bias": {
            "status": "not_estimated",
            "declared_trial_count": campaign.trial_count,
            "reason": (
                "Probabilistic/deflated Sharpe is not reported because fold-level "
                "aggregate artifacts do not justify the return-distribution, "
                "autocorrelation, and effective-independent-trial assumptions."
            ),
        },
    }

    destination = output_dir.resolve()
    destination.mkdir(parents=True, exist_ok=True)
    _atomic_write_text(
        destination / "campaign_snapshot.yaml",
        campaign.source_path.read_text(encoding="utf-8"),
    )
    _atomic_write_json(destination / "provenance.json", provenance)
    _atomic_write_json(destination / "report.json", report)
    _atomic_write_text(destination / "report.md", _report_markdown(report))
    all_observations = [
        item
        for observations in observations_by_configuration.values()
        for item in observations
    ]
    _write_observations_csv(destination / "observations.csv", all_observations)
    return destination, report


def main(argv: list[str] | None = None) -> int:
    """Run the aggregate research-report CLI.

    Args:
        argv: CLI arguments; None delegates to argparse for console use.

    Returns:
        Process exit code: 0 on success, 1 on validation or I/O errors.
    """
    parser = argparse.ArgumentParser(
        prog="forex-report",
        description="Aggregate explicit fold/seed run artifacts into research evidence.",
    )
    parser.add_argument(
        "--campaign", type=str, required=True, help="Campaign YAML path."
    )
    parser.add_argument(
        "--output-dir", type=str, required=True, help="Report artifact directory."
    )
    args = parser.parse_args(argv)
    try:
        output_dir, _ = run_research_report(Path(args.campaign), Path(args.output_dir))
    except (TrainerConfigError, OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"research report: {output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
