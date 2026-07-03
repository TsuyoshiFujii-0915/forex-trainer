"""Typed, fail-fast experiment configuration for forex-trainer.

One YAML file defines one experiment. Loading is strict (same philosophy as
forex-env-v3 ADR-0004): every key is required, unknown keys are rejected, and
registry names (algorithm, network, features) are validated up front. The
embedded env block is validated by delegating to forex_env.parse_config after
date-range injection, so a broken experiment fails before any run directory
is created.
"""

from __future__ import annotations

import copy
import re
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Mapping

import yaml
from forex_env import parse_config as parse_env_config
from forex_env.features import BASE_FEATURE_NAMES

from .algorithms import ALGO_REGISTRY
from .features import CROSS_FEATURE_REGISTRY, FEATURE_REGISTRY
from .networks import NETWORK_REGISTRY

_EXPERIMENT_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")
DEVICES: tuple[str, ...] = ("auto", "cpu", "cuda", "mps")
VEC_ENV_KINDS: tuple[str, ...] = ("dummy", "subproc")

# Episode cap applied to evaluation envs so a single episode walks the whole
# eval range; truncation then comes from the end of the data.
EVAL_EPISODE_MAX_STEPS = 1_000_000


class TrainerConfigError(Exception):
    """Raised when an experiment configuration is missing or invalid."""


@dataclass(frozen=True)
class RangeConfig:
    """Inclusive date range."""

    start: str
    end: str


@dataclass(frozen=True)
class AlgorithmConfig:
    """Algorithm axis selection."""

    name: str
    hyperparams: Mapping[str, Any]


@dataclass(frozen=True)
class NetworkConfig:
    """Network axis selection."""

    name: str
    kwargs: Mapping[str, Any]


@dataclass(frozen=True)
class RunConfig:
    """Run execution settings."""

    total_timesteps: int
    seed: int
    device: str
    n_envs: int
    vec_env: str
    decision_interval: int


@dataclass(frozen=True)
class ExperimentConfig:
    """Complete typed experiment configuration."""

    experiment: str
    env: Mapping[str, Any]
    train_range: RangeConfig
    val_range: RangeConfig
    eval_range: RangeConfig
    algorithm: AlgorithmConfig
    network: NetworkConfig
    run: RunConfig
    custom_feature_names: tuple[str, ...]
    custom_cross_feature_names: tuple[str, ...]


def _require_mapping(value: Any, name: str) -> Mapping[str, Any]:
    """Require a value to be a mapping.

    Args:
        value: Value to check.
        name: Name used in error messages.

    Returns:
        The mapping.

    Raises:
        TrainerConfigError: If the value is not a mapping.
    """
    if not isinstance(value, Mapping):
        raise TrainerConfigError(
            f"{name} must be a mapping, got {type(value).__name__}."
        )
    return value


def _check_exact_keys(
    section: Mapping[str, Any], allowed: tuple[str, ...], name: str
) -> None:
    """Require a section to contain exactly the allowed keys.

    Args:
        section: Section mapping.
        allowed: Allowed keys.
        name: Section name for error messages.

    Raises:
        TrainerConfigError: On missing or unknown keys.
    """
    missing = sorted(set(allowed) - set(section))
    if missing:
        raise TrainerConfigError(f"Missing key(s) in {name}: {missing}.")
    unknown = sorted(set(section) - set(allowed))
    if unknown:
        raise TrainerConfigError(f"Unknown key(s) in {name}: {unknown}.")


def _as_int(section: Mapping[str, Any], name: str, key: str) -> int:
    """Read an integer from a section.

    Args:
        section: Section mapping.
        name: Section name for error messages.
        key: Key to read.

    Returns:
        Integer value.

    Raises:
        TrainerConfigError: If the value is not an integer.
    """
    value = section[key]
    if isinstance(value, bool) or not isinstance(value, int):
        raise TrainerConfigError(f"{name}.{key} must be an integer, got {value!r}.")
    return value


def _as_str(section: Mapping[str, Any], name: str, key: str) -> str:
    """Read a non-empty string from a section.

    Args:
        section: Section mapping.
        name: Section name for error messages.
        key: Key to read.

    Returns:
        String value.

    Raises:
        TrainerConfigError: If the value is not a non-empty string.
    """
    value = section[key]
    if not isinstance(value, str) or not value:
        raise TrainerConfigError(
            f"{name}.{key} must be a non-empty string, got {value!r}."
        )
    return value


def _parse_range(section: Mapping[str, Any], name: str) -> RangeConfig:
    """Parse and validate a date range section.

    Args:
        section: Raw range mapping with start/end.
        name: Section name for error messages.

    Returns:
        RangeConfig instance.

    Raises:
        TrainerConfigError: On malformed or inverted dates.
    """
    _check_exact_keys(section, ("start", "end"), name)
    parsed: dict[str, date] = {}
    for key in ("start", "end"):
        text = _as_str(section, name, key)
        try:
            parsed[key] = date.fromisoformat(text)
        except ValueError as exc:
            raise TrainerConfigError(
                f"{name}.{key} must be an ISO date (YYYY-MM-DD), got {text!r}."
            ) from exc
    if parsed["start"] >= parsed["end"]:
        raise TrainerConfigError(
            f"{name}.start ({section['start']}) must be earlier than {name}.end "
            f"({section['end']})."
        )
    return RangeConfig(start=str(section["start"]), end=str(section["end"]))


def resolve_env_raw(
    env_block: Mapping[str, Any],
    date_range: RangeConfig,
    for_eval: bool,
) -> dict[str, Any]:
    """Build a complete forex-env raw config for a date range.

    Args:
        env_block: Experiment env block (without data dates).
        date_range: Range whose dates are injected into the data section.
        for_eval: If True, force a deterministic full-range walk: random_start
            off and the episode cap lifted (EVAL_EPISODE_MAX_STEPS).

    Returns:
        Raw forex-env configuration dictionary.
    """
    resolved: dict[str, Any] = copy.deepcopy(dict(env_block))
    resolved["data"] = dict(resolved["data"])
    resolved["data"]["start_date"] = date_range.start
    resolved["data"]["end_date"] = date_range.end
    if for_eval:
        resolved["environment"] = dict(resolved["environment"])
        resolved["environment"]["random_start"] = False
        resolved["environment"]["episode_max_steps"] = EVAL_EPISODE_MAX_STEPS
    return resolved


def parse_experiment_config(raw: Mapping[str, Any]) -> ExperimentConfig:
    """Parse and validate a raw experiment configuration.

    Args:
        raw: Raw experiment mapping in the YAML shape.

    Returns:
        Typed ExperimentConfig.

    Raises:
        TrainerConfigError: On any structural or registry violation.
        forex_env.ConfigError: If the embedded env block is invalid.
    """
    root = _require_mapping(raw, "experiment configuration root")
    _check_exact_keys(
        root,
        (
            "experiment",
            "env",
            "train_range",
            "val_range",
            "eval_range",
            "algorithm",
            "network",
            "run",
        ),
        "experiment configuration",
    )

    experiment = _as_str(root, "experiment configuration", "experiment")
    if not _EXPERIMENT_NAME_RE.match(experiment):
        raise TrainerConfigError(
            f"experiment must match [A-Za-z0-9][A-Za-z0-9_-]* (used as a directory "
            f"name), got {experiment!r}."
        )

    train_range = _parse_range(
        _require_mapping(root["train_range"], "train_range"), "train_range"
    )
    val_range = _parse_range(
        _require_mapping(root["val_range"], "val_range"), "val_range"
    )
    eval_range = _parse_range(
        _require_mapping(root["eval_range"], "eval_range"), "eval_range"
    )
    if date.fromisoformat(val_range.start) < date.fromisoformat(train_range.end):
        raise TrainerConfigError(
            f"val_range.start ({val_range.start}) must not precede train_range.end "
            f"({train_range.end}); overlapping ranges leak training data into "
            f"model selection."
        )
    if date.fromisoformat(eval_range.start) < date.fromisoformat(val_range.end):
        raise TrainerConfigError(
            f"eval_range.start ({eval_range.start}) must not precede val_range.end "
            f"({val_range.end}); overlapping ranges leak selection data into "
            f"evaluation."
        )

    algo_section = _require_mapping(root["algorithm"], "algorithm")
    _check_exact_keys(algo_section, ("name", "hyperparams"), "algorithm")
    algo_name = _as_str(algo_section, "algorithm", "name")
    if algo_name not in ALGO_REGISTRY:
        raise TrainerConfigError(
            f"algorithm.name '{algo_name}' is not registered; "
            f"available: {sorted(ALGO_REGISTRY)}."
        )
    algorithm = AlgorithmConfig(
        name=algo_name,
        hyperparams=dict(
            _require_mapping(algo_section["hyperparams"], "algorithm.hyperparams")
        ),
    )

    network_section = _require_mapping(root["network"], "network")
    _check_exact_keys(network_section, ("name", "kwargs"), "network")
    network_name = _as_str(network_section, "network", "name")
    if network_name not in NETWORK_REGISTRY:
        raise TrainerConfigError(
            f"network.name '{network_name}' is not registered; "
            f"available: {sorted(NETWORK_REGISTRY)}."
        )
    network = NetworkConfig(
        name=network_name,
        kwargs=dict(_require_mapping(network_section["kwargs"], "network.kwargs")),
    )

    run_section = _require_mapping(root["run"], "run")
    _check_exact_keys(
        run_section,
        ("total_timesteps", "seed", "device", "n_envs", "vec_env", "decision_interval"),
        "run",
    )
    total_timesteps = _as_int(run_section, "run", "total_timesteps")
    if total_timesteps < 1:
        raise TrainerConfigError(
            f"run.total_timesteps must be >= 1, got {total_timesteps}."
        )
    n_envs = _as_int(run_section, "run", "n_envs")
    if n_envs < 1:
        raise TrainerConfigError(f"run.n_envs must be >= 1, got {n_envs}.")
    decision_interval = _as_int(run_section, "run", "decision_interval")
    if decision_interval < 1:
        raise TrainerConfigError(
            f"run.decision_interval must be >= 1, got {decision_interval}."
        )
    device = _as_str(run_section, "run", "device")
    if device not in DEVICES:
        raise TrainerConfigError(
            f"run.device must be one of {list(DEVICES)}, got '{device}'."
        )
    vec_env = _as_str(run_section, "run", "vec_env")
    if vec_env not in VEC_ENV_KINDS:
        raise TrainerConfigError(
            f"run.vec_env must be one of {list(VEC_ENV_KINDS)}, got '{vec_env}'."
        )
    run = RunConfig(
        total_timesteps=total_timesteps,
        seed=_as_int(run_section, "run", "seed"),
        device=device,
        n_envs=n_envs,
        vec_env=vec_env,
        decision_interval=decision_interval,
    )

    env_block = _require_mapping(root["env"], "env")
    _check_exact_keys(
        env_block, ("environment", "data", "features", "transaction_costs"), "env"
    )
    data_block = _require_mapping(env_block["data"], "env.data")
    for forbidden in ("start_date", "end_date"):
        if forbidden in data_block:
            raise TrainerConfigError(
                f"env.data must not define '{forbidden}'; train_range/eval_range "
                f"inject the dates."
            )
    features_block = _require_mapping(env_block["features"], "env.features")
    if "selected" not in features_block:
        raise TrainerConfigError("Missing key(s) in env.features: ['selected'].")
    selected = features_block["selected"]
    if not isinstance(selected, (list, tuple)):
        raise TrainerConfigError(
            f"env.features.selected must be a list, got {selected!r}."
        )
    available = (
        tuple(BASE_FEATURE_NAMES)
        + tuple(FEATURE_REGISTRY)
        + tuple(CROSS_FEATURE_REGISTRY)
    )
    unknown_features = [name for name in selected if name not in available]
    if unknown_features:
        raise TrainerConfigError(
            f"Unknown feature(s) in env.features.selected: {unknown_features}; "
            f"available: {list(available)}."
        )
    custom_feature_names = tuple(name for name in selected if name in FEATURE_REGISTRY)
    custom_cross_feature_names = tuple(
        name for name in selected if name in CROSS_FEATURE_REGISTRY
    )

    config = ExperimentConfig(
        experiment=experiment,
        env=dict(env_block),
        train_range=train_range,
        val_range=val_range,
        eval_range=eval_range,
        algorithm=algorithm,
        network=network,
        run=run,
        custom_feature_names=custom_feature_names,
        custom_cross_feature_names=custom_cross_feature_names,
    )

    # Delegate full env validation (all ranges) to forex-env so that any
    # invalid env setting fails here, before a run directory is created.
    parse_env_config(resolve_env_raw(config.env, train_range, for_eval=False))
    parse_env_config(resolve_env_raw(config.env, val_range, for_eval=True))
    parse_env_config(resolve_env_raw(config.env, eval_range, for_eval=True))
    return config


def load_experiment_config(
    config_path: str | Path,
) -> tuple[ExperimentConfig, dict[str, Any]]:
    """Load an experiment YAML file.

    Args:
        config_path: Path to the experiment YAML.

    Returns:
        Tuple of (typed config, raw dict for snapshotting).

    Raises:
        TrainerConfigError: If the file is missing or malformed.
    """
    path = Path(config_path)
    if not path.is_file():
        raise TrainerConfigError(f"Experiment config file not found: {config_path}")
    try:
        with path.open("r", encoding="utf-8") as handle:
            raw = yaml.safe_load(handle)
    except yaml.YAMLError as exc:
        raise TrainerConfigError(
            f"Failed to parse YAML at {config_path}: {exc}"
        ) from exc
    if not isinstance(raw, dict):
        raise TrainerConfigError(
            f"Experiment config root must be a mapping: {config_path}"
        )
    return parse_experiment_config(raw), raw
