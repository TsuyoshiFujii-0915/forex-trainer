"""Sealed-policy replay and artifact generation for Issue #5 diagnostics."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import sys
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from forex_env.errors import ConfigError, DataError, FeatureError

from .artifact_provenance import dependency_versions, git_commits, sha256_file
from .config import TrainerConfigError
from .ensemble import load_member_for_device
from .env_factory import GateEvaluationMode, build_single_env
from .evaluate import compute_metrics
from .regime import (
    MARKET_CANDIDATE_NAMES,
    METRIC_NAMES,
    POLICY_STATE_CANDIDATE_NAMES,
    CandidateRegime,
    StepRecord,
    aggregate_fold_effects,
    align_agent_rule_records,
    compute_candidate_regime,
    compute_fold_effects,
)
from .report import Campaign, ConfigurationSpec, load_campaign, run_research_report

_STUDY_KEYS: tuple[str, ...] = (
    "name",
    "baseline_campaign",
    "baseline_configuration",
    "legacy_report",
    "legacy_scheme",
    "member_seeds",
    "rule",
    "bucket_count",
    "forward_drawdown_steps",
    "minimum_fold_direction_rate",
    "minimum_folds_per_era",
)
_RULE_KEYS: tuple[str, ...] = ("feature", "top_k", "base_size")
_SANITY_METRICS: tuple[str, ...] = (
    "annualized_net_return",
    "annualized_gross_return",
    "mean_max_drawdown",
    "worst_max_drawdown",
)


@dataclass(frozen=True)
class RuleSpec:
    """Explicit direct-weight momentum-reversal benchmark."""

    feature: str
    top_k: int
    base_size: float


@dataclass(frozen=True)
class RegimeStudy:
    """Validated strict regime-study definition."""

    name: str
    baseline_campaign: Path
    baseline_configuration: str
    legacy_report: Path
    legacy_scheme: str
    member_seeds: tuple[int, ...]
    rule: RuleSpec
    bucket_count: int
    forward_drawdown_steps: int
    minimum_fold_direction_rate: float
    minimum_folds_per_era: int
    source_path: Path


@dataclass(frozen=True)
class ReplayResult:
    """One fold's complete deterministic policy replay."""

    records: tuple[StepRecord, ...]
    metrics: Mapping[str, Any]


def _require_exact_keys(
    value: Mapping[str, Any], expected: Sequence[str], origin: str
) -> None:
    """Require exactly the declared mapping keys.

    Args:
        value: Mapping under validation.
        expected: Complete allowed key sequence.
        origin: Human-readable input location.

    Raises:
        TrainerConfigError: If a required key is absent or unknown.
    """
    expected_set = set(expected)
    actual_set = set(value)
    if actual_set != expected_set:
        raise TrainerConfigError(
            f"{origin} must contain exactly {list(expected)}; "
            f"missing={sorted(expected_set - actual_set)}, "
            f"unknown={sorted(actual_set - expected_set)}."
        )


def _require_string(value: Any, origin: str) -> str:
    """Require a non-empty string.

    Args:
        value: Candidate value.
        origin: Human-readable input location.

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
        origin: Human-readable input location.
        minimum: Inclusive lower bound.

    Returns:
        Validated integer.

    Raises:
        TrainerConfigError: If the value violates the contract.
    """
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise TrainerConfigError(
            f"{origin} must be an integer >= {minimum}, got {value!r}."
        )
    return value


def _require_finite_number(value: Any, origin: str) -> float:
    """Require a finite numeric value.

    Args:
        value: Candidate value.
        origin: Human-readable input location.

    Returns:
        Validated float.

    Raises:
        TrainerConfigError: If the value is boolean, non-numeric, or non-finite.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TrainerConfigError(f"{origin} must be numeric, got {value!r}.")
    number = float(value)
    if not math.isfinite(number):
        raise TrainerConfigError(f"{origin} must be finite, got {number!r}.")
    return number


def _load_mapping(path: Path, artifact: str) -> Mapping[str, Any]:
    """Load a JSON or YAML mapping.

    Args:
        path: Existing artifact path.
        artifact: Human-readable artifact name.

    Returns:
        Parsed mapping.

    Raises:
        TrainerConfigError: If the artifact is missing or malformed.
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
    if not isinstance(value, Mapping):
        raise TrainerConfigError(f"{artifact} root must be a mapping: {path}")
    return value


def load_regime_study(study_path: Path) -> RegimeStudy:
    """Load the strict regime-study YAML and validate legacy evidence identity.

    Args:
        study_path: Study YAML path.

    Returns:
        Validated study with absolute referenced paths.

    Raises:
        TrainerConfigError: If the manifest or referenced legacy report is invalid.
    """
    path = study_path.resolve()
    raw = _load_mapping(path, "regime study YAML")
    _require_exact_keys(raw, _STUDY_KEYS, "Regime study")
    rule_raw = raw["rule"]
    if not isinstance(rule_raw, Mapping):
        raise TrainerConfigError("Regime study rule must be a mapping.")
    _require_exact_keys(rule_raw, _RULE_KEYS, "Regime study rule")
    seeds_raw = raw["member_seeds"]
    if not isinstance(seeds_raw, list) or not seeds_raw:
        raise TrainerConfigError("Regime study member_seeds must be a non-empty list.")
    seeds = tuple(
        _require_integer(seed, f"Regime study member_seeds[{index}]", 0)
        for index, seed in enumerate(seeds_raw)
    )
    if len(set(seeds)) != len(seeds):
        raise TrainerConfigError("Regime study member_seeds contains duplicate seeds.")
    base_size = _require_finite_number(
        rule_raw["base_size"], "Regime study rule base_size"
    )
    if not 0.0 < base_size <= 1.0:
        raise TrainerConfigError(
            f"Regime study rule base_size must be in (0, 1], got {base_size}."
        )
    rate = _require_finite_number(
        raw["minimum_fold_direction_rate"],
        "Regime study minimum_fold_direction_rate",
    )
    if not 0.5 <= rate <= 1.0:
        raise TrainerConfigError(
            "Regime study minimum_fold_direction_rate must be in [0.5, 1.0], "
            f"got {rate}."
        )
    campaign_path = _resolved_reference(
        path, raw["baseline_campaign"], "baseline_campaign"
    )
    legacy_path = _resolved_reference(path, raw["legacy_report"], "legacy_report")
    legacy_scheme = _require_string(raw["legacy_scheme"], "legacy_scheme")
    legacy = _load_mapping(legacy_path, "legacy report")
    schemes = legacy.get("schemes")
    if not isinstance(schemes, Mapping) or legacy_scheme not in schemes:
        raise TrainerConfigError(
            f"legacy report {legacy_path} does not contain requested legacy scheme "
            f"{legacy_scheme!r}."
        )
    selected_scheme = schemes[legacy_scheme]
    if not isinstance(selected_scheme, Mapping) or not all(
        key in selected_scheme for key in ("overall", "eras", "folds")
    ):
        raise TrainerConfigError(
            f"legacy scheme {legacy_scheme!r} in {legacy_path} is malformed."
        )
    return RegimeStudy(
        name=_require_string(raw["name"], "Regime study name"),
        baseline_campaign=campaign_path,
        baseline_configuration=_require_string(
            raw["baseline_configuration"], "baseline_configuration"
        ),
        legacy_report=legacy_path,
        legacy_scheme=legacy_scheme,
        member_seeds=seeds,
        rule=RuleSpec(
            feature=_require_string(rule_raw["feature"], "Regime study rule feature"),
            top_k=_require_integer(rule_raw["top_k"], "Regime study rule top_k", 1),
            base_size=base_size,
        ),
        bucket_count=_require_integer(
            raw["bucket_count"], "Regime study bucket_count", 2
        ),
        forward_drawdown_steps=_require_integer(
            raw["forward_drawdown_steps"],
            "Regime study forward_drawdown_steps",
            1,
        ),
        minimum_fold_direction_rate=rate,
        minimum_folds_per_era=_require_integer(
            raw["minimum_folds_per_era"],
            "Regime study minimum_folds_per_era",
            1,
        ),
        source_path=path,
    )


def _resolved_reference(origin: Path, value: Any, field: str) -> Path:
    """Resolve a required file reference relative to its manifest.

    Args:
        origin: Referencing manifest path.
        value: Raw reference text.
        field: Field name for diagnostics.

    Returns:
        Absolute existing file path.

    Raises:
        TrainerConfigError: If the reference is invalid or missing.
    """
    text = _require_string(value, f"Regime study {field}")
    candidate = Path(text)
    resolved = (candidate if candidate.is_absolute() else origin.parent / candidate).resolve()
    if not resolved.is_file():
        raise TrainerConfigError(f"Regime study {field} file is missing: {resolved}")
    return resolved


def build_momentum_reversal_action(
    observation: Mapping[str, Any],
    feature_names: Sequence[str],
    rule: RuleSpec,
) -> np.ndarray:
    """Build direct weights from the declared cross-sectional reversal rule.

    Args:
        observation: Current environment observation with a market tensor.
        feature_names: Market-feature order represented by the tensor.
        rule: Explicit feature, tail count, and weight magnitude.

    Returns:
        Direct weight action shaped pairs by one.

    Raises:
        ValueError: If the feature or market observation is invalid.
    """
    if rule.feature not in feature_names:
        raise ValueError(
            f"rule feature {rule.feature!r} is absent from selected features "
            f"{list(feature_names)!r}."
        )
    market = np.asarray(observation.get("market"), dtype=np.float64)
    if market.ndim != 3 or market.shape[2] != len(feature_names):
        raise ValueError(
            "observation market must have shape pairs by window by selected features."
        )
    pair_count = market.shape[0]
    if rule.top_k * 2 > pair_count:
        raise ValueError(
            f"rule top_k={rule.top_k} requires at least {rule.top_k * 2} pairs, "
            f"got {pair_count}."
        )
    values = market[:, -1, feature_names.index(rule.feature)]
    if not np.isfinite(values).all():
        raise ValueError("rule feature values must be finite.")
    order = np.argsort(values, kind="stable")
    weights = np.zeros(pair_count, dtype=np.float32)
    weights[order[: rule.top_k]] = rule.base_size
    weights[order[-rule.top_k :]] = -rule.base_size
    return weights.reshape(-1, 1)


def _decision_regime(
    observation: Mapping[str, Any], feature_names: Sequence[str]
) -> CandidateRegime:
    """Compute the causal market regime from an environment observation.

    Args:
        observation: Current environment observation.
        feature_names: Market-feature order.

    Returns:
        Candidate regime at the decision time.

    Raises:
        ValueError: If required unnormalized inputs are unavailable.
    """
    required = ("log_return", "carry_annual")
    missing = [name for name in required if name not in feature_names]
    if missing:
        raise ValueError(
            f"regime replay requires selected features {list(required)!r}; "
            f"missing={missing!r}."
        )
    market = np.asarray(observation.get("market"), dtype=np.float64)
    if market.ndim != 3 or market.shape[2] != len(feature_names):
        raise ValueError("replay observation market has an invalid shape.")
    returns = market[:, :, feature_names.index("log_return")].T
    carry = market[:, -1, feature_names.index("carry_annual")]
    return compute_candidate_regime(returns, carry)


def _parse_timestamp(value: Any, fold: str) -> datetime:
    """Parse a timezone-aware decision timestamp.

    Args:
        value: Environment timestamp value.
        fold: Fold identifier for diagnostics.

    Returns:
        Timezone-aware datetime.

    Raises:
        ValueError: If the timestamp is missing, invalid, or timezone-naive.
    """
    if not isinstance(value, str):
        raise ValueError(f"fold {fold} environment timestamp must be a string.")
    try:
        timestamp = datetime.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"fold {fold} has invalid timestamp {value!r}.") from exc
    if timestamp.tzinfo is None or timestamp.utcoffset() is None:
        raise ValueError(
            f"fold {fold} timestamp {value!r} must include an explicit timezone."
        )
    return timestamp


def _walk_policy(
    env: Any,
    predict: Callable[[Mapping[str, Any], np.ndarray], np.ndarray],
    fold: str,
    era: str,
    feature_names: Sequence[str],
    forward_drawdown_steps: int,
) -> ReplayResult:
    """Walk one policy and retain decision-time state plus next outcomes.

    Args:
        env: Deterministic evaluation environment.
        predict: Policy action callable.
        fold: Evaluation-fold identifier.
        era: Declared era containing the fold.
        feature_names: Unnormalized market-feature order.
        forward_drawdown_steps: Forward equity horizon for drawdown response.

    Returns:
        Complete step records and evaluator-contract metrics.

    Raises:
        ValueError: If environment trace fields are absent or invalid.
    """
    observation, info = env.reset(seed=0)
    timestamps = [str(info["timestamp"])]
    equities = [float(info["equity_jpy"])]
    rewards: list[float] = []
    costs: list[float] = []
    leverages: list[float] = []
    turnovers: list[float] = []
    decision_times: list[datetime] = []
    regimes: list[Any] = []
    gross_responses: list[float] = []
    cost_responses: list[float] = []
    decision_exposures: list[float] = []
    previous_weights: np.ndarray | None = None
    episode_start = np.ones((1,), dtype=bool)
    terminated = False
    while True:
        decision_times.append(_parse_timestamp(info.get("timestamp"), fold))
        regimes.append(_decision_regime(observation, feature_names))
        decision_exposures.append(float(info["gross_leverage"]))
        equity_before = float(info["equity_jpy"])
        action = np.asarray(predict(observation, episode_start), dtype=np.float32)
        observation, reward, terminated, truncated, info = env.step(action)
        cost = float(info["costs_jpy"]["total"])
        equity_after = float(info["equity_jpy"])
        target = np.asarray(info["target_weights"], dtype=np.float64)
        if previous_weights is None:
            previous_weights = np.zeros_like(target)
        turnover = float(np.abs(target - previous_weights).sum())
        previous_weights = target
        rewards.append(float(reward))
        equities.append(equity_after)
        timestamps.append(str(info["timestamp"]))
        costs.append(cost)
        leverages.append(float(info["gross_leverage"]))
        turnovers.append(turnover)
        gross_responses.append(float(math.log((equity_after + cost) / equity_before)))
        cost_responses.append(cost / equity_before)
        episode_start = np.asarray([terminated or truncated], dtype=bool)
        if terminated or truncated:
            break
    complete_record_count = len(regimes) - forward_drawdown_steps + 1
    if complete_record_count < 1:
        raise ValueError(
            f"fold {fold} has {len(regimes)} decisions, fewer than the required "
            f"forward drawdown horizon {forward_drawdown_steps}."
        )
    records: list[StepRecord] = []
    for index in range(complete_record_count):
        regime = regimes[index]
        stop = index + forward_drawdown_steps + 1
        future_minimum = min(equities[index + 1 : stop])
        forward_drawdown = max(0.0, 1.0 - future_minimum / equities[index])
        records.append(
            StepRecord(
                fold=fold,
                era=era,
                timestamp=decision_times[index],
                regime=regime,
                decision_gross_exposure=decision_exposures[index],
                decision_turnover=turnovers[index],
                next_net_return=rewards[index],
                next_gross_return=gross_responses[index],
                next_cost_ratio=cost_responses[index],
                forward_max_drawdown=forward_drawdown,
            )
        )
    metrics = compute_metrics(
        rewards,
        equities,
        timestamps,
        costs,
        leverages,
        turnovers,
        terminated,
    )
    return ReplayResult(records=tuple(records), metrics=metrics)


def _era_for_fold(campaign: Campaign, fold: str) -> str:
    """Resolve exactly one declared era for a fold.

    Args:
        campaign: Baseline campaign.
        fold: Numeric evaluation-year label.

    Returns:
        Unique era name.

    Raises:
        TrainerConfigError: If zero or multiple eras contain the fold.
    """
    year = int(fold)
    matching = [era.name for era in campaign.eras if era.start <= year <= era.end]
    if len(matching) != 1:
        raise TrainerConfigError(
            f"Fold {fold} must belong to exactly one campaign era, got {matching!r}."
        )
    return matching[0]


def _require_baseline_campaign(
    campaign: Campaign, study: RegimeStudy
) -> ConfigurationSpec:
    """Require a baseline-only ensemble campaign matching the study.

    Args:
        campaign: Parsed generic report campaign.
        study: Parsed regime study.

    Returns:
        Sole ensemble configuration.

    Raises:
        TrainerConfigError: If the campaign is not baseline-only.
    """
    if len(campaign.configurations) != 1 or campaign.comparisons:
        raise TrainerConfigError(
            "Regime study baseline_campaign must contain exactly one configuration "
            "and comparisons: []."
        )
    configuration = campaign.configurations[0]
    if configuration.name != study.baseline_configuration:
        raise TrainerConfigError(
            f"baseline_configuration {study.baseline_configuration!r} does not "
            f"match campaign configuration {configuration.name!r}."
        )
    if configuration.result_kind != "ensemble":
        raise TrainerConfigError(
            "Regime study baseline configuration must use result_kind=ensemble."
        )
    return configuration


def _sealed_evaluation_device(
    manifest: Mapping[str, Any], manifest_path: Path
) -> str:
    """Read the concrete replay device sealed by an ensemble evaluation.

    Args:
        manifest: Parsed version 2 ensemble manifest.
        manifest_path: Manifest path for diagnostics.

    Returns:
        Concrete CPU, CUDA, or MPS device.

    Raises:
        TrainerConfigError: If the evaluation device is absent or unresolved.
    """
    evaluation = manifest.get("evaluation")
    device = (
        evaluation.get("resolved_device") if isinstance(evaluation, Mapping) else None
    )
    if device not in {"cpu", "cuda", "mps"}:
        raise TrainerConfigError(
            "Ensemble manifest lacks a concrete sealed resolved_device: "
            f"{manifest_path}; got {device!r}."
        )
    return device


def _load_replay_member(
    run_dir: Path, sealed_device: str
) -> tuple[Any, Any, Mapping[str, Any], Mapping[str, Any], Mapping[str, Any], str]:
    """Load one SB3 replay member on the source evaluation's sealed device.

    Args:
        run_dir: Current-contract member run directory.
        sealed_device: Concrete device recorded by the ensemble manifest.

    Returns:
        Parsed config, model, eval env, raw config, meta, and resolved device.
    """
    return load_member_for_device(run_dir, sealed_device)


def _metrics_match(actual: Mapping[str, Any], expected: Mapping[str, Any], origin: Path) -> None:
    """Require replay metrics to match the sealed evaluator artifact.

    Args:
        actual: Deterministic replay metrics.
        expected: Sealed metrics mapping.
        origin: Metrics path for diagnostics.

    Raises:
        TrainerConfigError: If fields or values differ.
    """
    if set(actual) != set(expected):
        raise TrainerConfigError(
            f"Replay metrics fields differ from sealed metrics {origin}: "
            f"actual_only={sorted(set(actual) - set(expected))}, "
            f"sealed_only={sorted(set(expected) - set(actual))}."
        )
    for key in sorted(actual):
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
                f"Deterministic replay metric {key!r} does not match sealed "
                f"metrics {origin}: replay={left!r}, sealed={right!r}."
            )


def _replay_ensemble_fold(
    ensemble_dir: Path,
    fold: str,
    era: str,
    study: RegimeStudy,
) -> tuple[ReplayResult, ReplayResult, Mapping[str, Any]]:
    """Replay one sealed ensemble and its rule in independent identical envs.

    Args:
        ensemble_dir: Version 2 ensemble artifact directory.
        fold: Evaluation-fold label.
        era: Declared era label.
        study: Study controls and benchmark rule.

    Returns:
        Agent replay, rule replay, and source provenance.

    Raises:
        TrainerConfigError: If source identity, seeds, devices, or replay differ.
        ValueError: If the causal trace cannot be computed.
    """
    manifest_path = ensemble_dir / "ensemble.json"
    manifest = _load_mapping(manifest_path, "ensemble manifest")
    if manifest.get("manifest_version") != 2:
        raise TrainerConfigError(
            f"Legacy ensemble manifest is not valid regime evidence: {manifest_path}"
        )
    raw_members = manifest.get("members")
    if not isinstance(raw_members, list) or not raw_members:
        raise TrainerConfigError(f"Ensemble manifest has no members: {manifest_path}")
    declared_seeds = tuple(sorted(int(member["seed"]) for member in raw_members))
    if declared_seeds != tuple(sorted(study.member_seeds)):
        raise TrainerConfigError(
            f"Ensemble member seeds {declared_seeds!r} differ from study seeds "
            f"{tuple(sorted(study.member_seeds))!r}: {manifest_path}"
        )
    member_dirs = [Path(str(member["run_dir"])).resolve() for member in raw_members]
    sealed_device = _sealed_evaluation_device(manifest, manifest_path)
    members = [
        _load_replay_member(run_dir, sealed_device) for run_dir in member_dirs
    ]
    reference_config, _, eval_raw, raw_config, _, device = members[0]
    if reference_config.run.residual is not None or reference_config.run.rank_allocation is not None:
        raise TrainerConfigError(
            f"Regime baseline must use direct weights without residual/rank allocation: {manifest_path}"
        )
    features_raw = raw_config.get("env", {}).get("features", {})
    selected = features_raw.get("selected") if isinstance(features_raw, Mapping) else None
    if not isinstance(selected, list) or not all(isinstance(item, str) for item in selected):
        raise TrainerConfigError(f"Member config has invalid selected features: {member_dirs[0]}")
    if features_raw.get("normalize") is not False:
        raise TrainerConfigError(
            f"Regime candidates require unnormalized observations: {member_dirs[0]}"
        )
    reference_run_contract = (
        reference_config.run.device,
        reference_config.run.decision_interval,
        reference_config.run.residual,
        reference_config.run.rank_allocation,
    )
    for run_dir, member in zip(member_dirs[1:], members[1:]):
        config, _, member_eval, member_raw, _, member_device = member
        if member_eval != eval_raw or member_device != device:
            raise TrainerConfigError(
                f"Ensemble member {run_dir} does not share the replay eval env/device."
            )
        member_run_contract = (
            config.run.device,
            config.run.decision_interval,
            config.run.residual,
            config.run.rank_allocation,
        )
        if member_run_contract != reference_run_contract:
            raise TrainerConfigError(
                f"Ensemble member {run_dir} has incompatible run configuration."
            )
        if member_raw.get("env", {}).get("features") != features_raw:
            raise TrainerConfigError(
                f"Ensemble member {run_dir} has incompatible feature configuration."
            )
    states: list[Any] = [None] * len(members)

    def agent_predict(
        observation: Mapping[str, Any], episode_start: np.ndarray
    ) -> np.ndarray:
        actions: list[np.ndarray] = []
        for index, member in enumerate(members):
            model = member[1]
            action, states[index] = model.predict(
                observation,
                state=states[index],
                episode_start=episode_start,
                deterministic=True,
            )
            actions.append(np.asarray(action, dtype=np.float64))
        return np.mean(np.stack(actions, axis=0), axis=0).astype(np.float32)

    agent_env = build_single_env(
        eval_raw,
        reference_config.custom_feature_names,
        reference_config.custom_cross_feature_names,
        seed=0,
        decision_interval=reference_config.run.decision_interval,
        residual=None,
        rank_allocation=None,
        apply_hold_gate=None,
        gate_evaluation_mode=GateEvaluationMode.LEARNED,
    )
    try:
        agent = _walk_policy(
            agent_env,
            agent_predict,
            fold,
            era,
            tuple(selected),
            study.forward_drawdown_steps,
        )
    finally:
        agent_env.close()
    sealed_metrics = _load_mapping(ensemble_dir / "metrics.json", "sealed metrics")
    _metrics_match(agent.metrics, sealed_metrics, ensemble_dir / "metrics.json")

    def rule_predict(
        observation: Mapping[str, Any], episode_start: np.ndarray
    ) -> np.ndarray:
        del episode_start
        return build_momentum_reversal_action(observation, tuple(selected), study.rule)

    rule_env = build_single_env(
        eval_raw,
        reference_config.custom_feature_names,
        reference_config.custom_cross_feature_names,
        seed=0,
        decision_interval=reference_config.run.decision_interval,
        residual=None,
        rank_allocation=None,
        apply_hold_gate=None,
        gate_evaluation_mode=GateEvaluationMode.LEARNED,
    )
    try:
        rule = _walk_policy(
            rule_env,
            rule_predict,
            fold,
            era,
            tuple(selected),
            study.forward_drawdown_steps,
        )
    finally:
        rule_env.close()
    align_agent_rule_records(agent.records, rule.records)
    source = {
        "fold": fold,
        "ensemble_dir": str(ensemble_dir),
        "ensemble_manifest_sha256": sha256_file(manifest_path),
        "sealed_metrics_sha256": sha256_file(ensemble_dir / "metrics.json"),
        "eval_env_sha256": sha256_file(ensemble_dir / "env_eval.yaml"),
        "sealed_evaluation_device": sealed_device,
        "replay_device": device,
        "evaluation_git": manifest["evaluation"]["git"],
        "evaluation_versions": manifest["evaluation"]["versions"],
        "data_identity": manifest["evaluation"]["data_identity"],
        "members": [
            {
                "run_dir": str(run_dir),
                "seed": raw_member["seed"],
                "model_sha256": raw_member["model_sha256"],
                "config_snapshot_sha256": raw_member["config_snapshot_sha256"],
                "meta_sha256": raw_member["meta_sha256"],
            }
            for run_dir, raw_member in zip(member_dirs, raw_members)
        ],
    }
    return agent, rule, source


def _scope_metric(scope: Mapping[str, Any], metric: str, legacy: bool) -> float:
    """Read one sanity metric across legacy/current naming conventions.

    Args:
        scope: Aggregate, era, or fold metric mapping.
        metric: Canonical sanity metric.
        legacy: Whether legacy mean-prefixed names are preferred.

    Returns:
        Finite metric value.

    Raises:
        TrainerConfigError: If no explicit supported field exists.
    """
    aliases: Mapping[str, tuple[tuple[str, ...], tuple[str, ...]]] = {
        "annualized_net_return": (
            ("mean_annualized_net_return", "annualized_net_return"),
            ("annualized_net_return",),
        ),
        "annualized_gross_return": (
            ("mean_annualized_gross_return", "annualized_gross_return"),
            ("annualized_gross_return",),
        ),
        "mean_max_drawdown": (
            ("mean_max_drawdown", "max_drawdown"),
            ("mean_max_drawdown", "max_drawdown"),
        ),
        "worst_max_drawdown": (
            ("worst_max_drawdown",),
            ("worst_max_drawdown",),
        ),
    }
    ordered = aliases[metric][0 if legacy else 1]
    for key in ordered:
        if key in scope and scope[key] is not None:
            return _require_finite_number(scope[key], f"sanity metric {key}")
    raise TrainerConfigError(
        f"Sanity scope lacks explicit metric {metric!r}; available={sorted(scope)}."
    )


def compare_legacy_current(
    legacy_scheme: Mapping[str, Any], current_configuration: Mapping[str, Any]
) -> tuple[Mapping[str, Any], ...]:
    """Build descriptive legacy/current level comparisons without inference.

    Args:
        legacy_scheme: Issue 1 study-specific selected scheme.
        current_configuration: Generic current-provenance configuration report.

    Returns:
        Overall, common-era, and common-fold metric rows.

    Raises:
        TrainerConfigError: If required structures or finite metrics are absent.
    """
    for origin, value in (
        ("legacy scheme", legacy_scheme),
        ("current configuration", current_configuration),
    ):
        if not all(isinstance(value.get(key), Mapping) for key in ("overall", "eras", "folds")):
            raise TrainerConfigError(f"{origin} requires overall, eras, and folds mappings.")
    scopes: list[tuple[str, Mapping[str, Any], Mapping[str, Any]]] = [
        ("overall", legacy_scheme["overall"], current_configuration["overall"])
    ]
    legacy_eras = legacy_scheme["eras"]
    current_eras = current_configuration["eras"]
    for name in sorted(set(legacy_eras) & set(current_eras)):
        scopes.append((f"era:{name}", legacy_eras[name], current_eras[name]))
    legacy_folds = legacy_scheme["folds"]
    current_folds = current_configuration["folds"]
    for name in sorted(set(legacy_folds) & set(current_folds)):
        scopes.append((f"fold:{name}", legacy_folds[name], current_folds[name]))
    rows: list[Mapping[str, Any]] = []
    for scope_name, legacy_values, current_values in scopes:
        metrics = _SANITY_METRICS if scope_name == "overall" else _SANITY_METRICS[:3]
        for metric in metrics:
            legacy_value = _scope_metric(legacy_values, metric, True)
            current_value = _scope_metric(current_values, metric, False)
            rows.append(
                {
                    "scope": scope_name,
                    "metric": metric,
                    "legacy": legacy_value,
                    "current": current_value,
                    "current_minus_legacy": current_value - legacy_value,
                    "interpretation": "descriptive_only_not_paired_evidence",
                }
            )
    return tuple(rows)


def _json_value(value: Any) -> Any:
    """Convert dataclasses, tuples, datetimes, and mappings to JSON values.

    Args:
        value: Arbitrarily nested report value.

    Returns:
        JSON-serializable value.
    """
    if hasattr(value, "__dataclass_fields__"):
        return _json_value(asdict(value))
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    return value


def _write_json(path: Path, value: Any) -> None:
    """Write deterministic JSON inside an unpublished staging directory.

    Args:
        path: Destination path.
        value: JSON-serializable report value.
    """
    path.write_text(
        json.dumps(_json_value(value), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    """Write a deterministic CSV inside an unpublished staging directory.

    Args:
        path: Destination path.
        rows: Non-empty homogeneous row mappings.

    Raises:
        ValueError: If no rows exist.
    """
    if not rows:
        raise ValueError(f"CSV artifact requires at least one row: {path.name}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _record_row(policy: str, record: StepRecord) -> Mapping[str, Any]:
    """Flatten one trace record for CSV output.

    Args:
        policy: Agent or rule label.
        record: Complete step record.

    Returns:
        Stable flat CSV row.
    """
    return {
        "policy": policy,
        "fold": record.fold,
        "era": record.era,
        "timestamp": record.timestamp.isoformat(),
        **asdict(record.regime),
        "decision_gross_exposure": record.decision_gross_exposure,
        "decision_turnover": record.decision_turnover,
        "next_net_return": record.next_net_return,
        "next_gross_return": record.next_gross_return,
        "next_cost_ratio": record.next_cost_ratio,
        "forward_max_drawdown": record.forward_max_drawdown,
    }


def _aggregate_rows(
    policy_records: Mapping[str, Sequence[StepRecord]],
    campaign: Campaign,
    study: RegimeStudy,
) -> tuple[list[Mapping[str, Any]], list[Mapping[str, Any]], Mapping[str, Any]]:
    """Compute bucket, fold, and stable fold-level evidence for all candidates.

    Args:
        policy_records: Complete traces by policy.
        campaign: Bootstrap and era controls.
        study: Bucket and stability controls.

    Returns:
        Bucket rows, fold-effect rows, and nested evidence report.
    """
    bucket_rows: list[Mapping[str, Any]] = []
    effect_rows: list[Mapping[str, Any]] = []
    evidence: dict[str, Any] = {}
    candidates = MARKET_CANDIDATE_NAMES + POLICY_STATE_CANDIDATE_NAMES
    era_order = tuple(era.name for era in campaign.eras)
    for policy, records in policy_records.items():
        evidence[policy] = {}
        for candidate in candidates:
            buckets, effects = compute_fold_effects(records, candidate, study.bucket_count)
            bucket_rows.extend(
                {"policy": policy, "candidate": candidate, **_json_value(item)}
                for item in buckets
            )
            effect_rows.extend(
                {"policy": policy, "candidate": candidate, **_json_value(item)}
                for item in effects
            )
            evidence[policy][candidate] = {
                metric: _json_value(
                    aggregate_fold_effects(
                        effects,
                        metric,
                        campaign.bootstrap_samples,
                        campaign.bootstrap_seed,
                        campaign.moving_block_length,
                        study.minimum_fold_direction_rate,
                        era_order,
                        study.minimum_folds_per_era,
                    )
                )
                for metric in METRIC_NAMES
            }
    return bucket_rows, effect_rows, evidence


def _render_markdown(report: Mapping[str, Any]) -> str:
    """Render the compact human-readable diagnostic summary.

    Args:
        report: Complete JSON report mapping.

    Returns:
        Markdown document.
    """
    lines = [
        f"# {report['study']}",
        "",
        "## Reproducibility checks",
        "",
        "- The baseline-only generic report passed current artifact and provenance validation.",
        "- Every action-mean ensemble was replayed deterministically and matched its sealed metrics.",
        "- The rule was walked independently in the same resolved evaluation environments and timestamps aligned exactly.",
        "",
        "## Stable relationships",
        "",
    ]
    stable = report["stable_relationships"]
    if stable:
        for item in stable:
            lines.append(
                f"- {item['policy']} / {item['candidate']} / {item['metric']}"
            )
    else:
        lines.append("- None under the predeclared fold/era criteria.")
    lines.extend(
        [
            "",
            "## Legacy sanity comparison",
            "",
            "Legacy Issue 1 evidence is reported descriptively only; it is not paired or bootstrapped with current evidence.",
            "",
            "See `sanity_comparison.csv`, `fold_effects.csv`, and `provenance.json` for auditable details.",
        ]
    )
    return "\n".join(lines) + "\n"


def _render_svg(effect_rows: Sequence[Mapping[str, Any]]) -> str:
    """Render a dependency-free SVG of mean agent net-return fold effects.

    Args:
        effect_rows: Flattened fold effects for both policies.

    Returns:
        Complete SVG text.
    """
    candidates = list(MARKET_CANDIDATE_NAMES + POLICY_STATE_CANDIDATE_NAMES)
    means: list[float] = []
    for candidate in candidates:
        values = [
            float(row["high_minus_low_next_net_return"])
            for row in effect_rows
            if row["policy"] == "agent" and row["candidate"] == candidate
        ]
        means.append(float(sum(values) / len(values)))
    scale = max(max(abs(value) for value in means), 1e-12)
    width = 960
    height = 100 + 52 * len(candidates)
    zero = 520
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="20" y="30" font-family="sans-serif" font-size="18">Agent fold-mean high-minus-low next net return</text>',
        f'<line x1="{zero}" y1="50" x2="{zero}" y2="{height - 20}" stroke="#333"/>',
    ]
    for index, (candidate, value) in enumerate(zip(candidates, means)):
        y = 70 + index * 52
        length = value / scale * 360
        x = zero if length >= 0.0 else zero + length
        color = "#2563eb" if value >= 0.0 else "#dc2626"
        lines.append(
            f'<text x="10" y="{y + 16}" font-family="sans-serif" font-size="12">{candidate}</text>'
        )
        lines.append(
            f'<rect x="{x:.3f}" y="{y}" width="{abs(length):.3f}" height="20" fill="{color}"/>'
        )
        lines.append(
            f'<text x="{zero + length + (5 if length >= 0 else -85):.3f}" y="{y + 16}" font-family="monospace" font-size="11">{value:.6g}</text>'
        )
    lines.append("</svg>\n")
    return "\n".join(lines)


def run_regime_study(
    study_path: Path, output_dir: Path
) -> tuple[Path, Mapping[str, Any]]:
    """Validate, replay, aggregate, and atomically publish one regime study.

    Args:
        study_path: Strict regime-study YAML.
        output_dir: New destination directory.

    Returns:
        Published absolute directory and complete report mapping.

    Raises:
        TrainerConfigError: If configuration, artifacts, or replay disagree.
        OSError: If source or destination I/O fails.
        ValueError: If causal candidates or fold evidence are invalid.
    """
    study = load_regime_study(study_path)
    destination = output_dir.resolve()
    if destination.exists():
        raise TrainerConfigError(
            f"Regime study output directory must not already exist: {destination}"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent)
    )
    try:
        _, baseline_report = run_research_report(
            study.baseline_campaign, staging / "baseline_report"
        )
        campaign = load_campaign(study.baseline_campaign)
        configuration = _require_baseline_campaign(campaign, study)
        agent_records: list[StepRecord] = []
        rule_records: list[StepRecord] = []
        replay_metrics: dict[str, Any] = {}
        sources: list[Mapping[str, Any]] = []
        for ensemble_dir in configuration.run_dirs:
            manifest = _load_mapping(
                ensemble_dir / "ensemble.json", "ensemble manifest"
            )
            members = manifest.get("members")
            if not isinstance(members, list) or not members:
                raise TrainerConfigError(
                    f"Ensemble manifest has no members: {ensemble_dir}"
                )
            first_config = _load_mapping(
                Path(str(members[0]["run_dir"])).resolve() / "config_snapshot.yaml",
                "member config snapshot",
            )
            eval_range = first_config.get("eval_range")
            if not isinstance(eval_range, Mapping):
                raise TrainerConfigError(
                    f"Member config lacks eval_range: {ensemble_dir}"
                )
            start = _require_string(
                eval_range.get("start"), f"{ensemble_dir} eval_range.start"
            )
            fold = str(datetime.fromisoformat(start).year)
            era = _era_for_fold(campaign, fold)
            agent, rule, source = _replay_ensemble_fold(
                ensemble_dir, fold, era, study
            )
            agent_records.extend(agent.records)
            rule_records.extend(rule.records)
            replay_metrics[fold] = {
                "agent": agent.metrics,
                "rule": rule.metrics,
            }
            sources.append(source)
        align_agent_rule_records(agent_records, rule_records)
        policy_records: Mapping[str, Sequence[StepRecord]] = {
            "agent": tuple(agent_records),
            "rule": tuple(rule_records),
        }
        bucket_rows, effect_rows, evidence = _aggregate_rows(
            policy_records, campaign, study
        )
        legacy_report = _load_mapping(study.legacy_report, "legacy report")
        legacy_scheme = legacy_report["schemes"][study.legacy_scheme]
        current_configuration = baseline_report["configurations"][
            study.baseline_configuration
        ]
        sanity_rows = compare_legacy_current(legacy_scheme, current_configuration)
        stable_relationships = [
            {"policy": policy, "candidate": candidate, "metric": metric}
            for policy, candidate_values in evidence.items()
            for candidate, metric_values in candidate_values.items()
            for metric, value in metric_values.items()
            if value["is_stable"]
        ]
        report: Mapping[str, Any] = {
            "study": study.name,
            "baseline_configuration": study.baseline_configuration,
            "member_seeds": list(study.member_seeds),
            "rule": asdict(study.rule),
            "forward_drawdown_steps": study.forward_drawdown_steps,
            "baseline_report": baseline_report,
            "sealed_replay_metrics": replay_metrics,
            "evidence": evidence,
            "stable_relationships": stable_relationships,
            "legacy_sanity_comparison": list(sanity_rows),
            "legacy_comparison_interpretation": "descriptive_only_not_paired_evidence",
        }
        trace_rows = [
            _record_row(policy, record)
            for policy, records in policy_records.items()
            for record in records
        ]
        _write_csv(staging / "step_trace.csv", trace_rows)
        _write_csv(staging / "bucket_metrics.csv", bucket_rows)
        _write_csv(staging / "fold_effects.csv", effect_rows)
        _write_csv(staging / "sanity_comparison.csv", sanity_rows)
        _write_json(staging / "report.json", report)
        (staging / "report.md").write_text(
            _render_markdown(report), encoding="utf-8"
        )
        (staging / "regime_effects.svg").write_text(
            _render_svg(effect_rows), encoding="utf-8"
        )
        (staging / "study_snapshot.yaml").write_text(
            study.source_path.read_text(encoding="utf-8"), encoding="utf-8"
        )
        provenance = {
            "diagnostic_git": git_commits(),
            "diagnostic_versions": dependency_versions(),
            "study_path": str(study.source_path),
            "study_sha256": sha256_file(study.source_path),
            "campaign_path": str(study.baseline_campaign),
            "campaign_sha256": sha256_file(study.baseline_campaign),
            "legacy_report_path": str(study.legacy_report),
            "legacy_report_sha256": sha256_file(study.legacy_report),
            "source_ensembles": sources,
            "generated_artifact_sha256": {
                name: sha256_file(staging / name)
                for name in (
                    "report.json",
                    "report.md",
                    "step_trace.csv",
                    "bucket_metrics.csv",
                    "fold_effects.csv",
                    "sanity_comparison.csv",
                    "regime_effects.svg",
                    "study_snapshot.yaml",
                )
            },
        }
        _write_json(staging / "provenance.json", provenance)
        os.replace(staging, destination)
    except Exception:
        shutil.rmtree(staging)
        raise
    return destination, report


def main(argv: list[str] | None = None) -> int:
    """Run the sealed regime-study CLI.

    Args:
        argv: CLI arguments; None delegates parsing to sys.argv.

    Returns:
        Process exit code: zero on success and one on explicit errors.
    """
    parser = argparse.ArgumentParser(
        prog="forex-regime-study",
        description="Replay sealed ensembles and aggregate fold-level regime evidence.",
    )
    parser.add_argument("--study", type=str, required=True, help="Study YAML path.")
    parser.add_argument(
        "--output-dir", type=str, required=True, help="New artifact directory."
    )
    args = parser.parse_args(argv)
    try:
        output, _ = run_regime_study(Path(args.study), Path(args.output_dir))
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
    print(f"regime study: {output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
