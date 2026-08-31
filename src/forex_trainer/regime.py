"""Pure calculations for causal, fold-aware regime diagnostics."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Literal

import numpy as np

from .research_statistics import BootstrapIntervals, bootstrap_mean_intervals

CandidateName = Literal[
    "realized_market_volatility",
    "mean_cross_pair_correlation",
    "cross_sectional_return_dispersion",
    "momentum_dispersion",
    "carry_dispersion",
    "trend_reversal_proxy",
    "decision_gross_exposure",
    "decision_turnover",
]
MetricName = Literal[
    "next_net_return",
    "next_gross_return",
    "next_cost_ratio",
    "next_drawdown_change",
]
Direction = Literal["negative", "neutral", "positive"]

MARKET_CANDIDATE_NAMES: tuple[CandidateName, ...] = (
    "realized_market_volatility",
    "mean_cross_pair_correlation",
    "cross_sectional_return_dispersion",
    "momentum_dispersion",
    "carry_dispersion",
    "trend_reversal_proxy",
)
POLICY_STATE_CANDIDATE_NAMES: tuple[CandidateName, ...] = (
    "decision_gross_exposure",
    "decision_turnover",
)
METRIC_NAMES: tuple[MetricName, ...] = (
    "next_net_return",
    "next_gross_return",
    "next_cost_ratio",
    "next_drawdown_change",
)


@dataclass(frozen=True)
class CandidateRegime:
    """Causal market variables computed at one decision timestamp."""

    realized_market_volatility: float
    mean_cross_pair_correlation: float
    cross_sectional_return_dispersion: float
    momentum_dispersion: float
    carry_dispersion: float
    trend_reversal_proxy: float


@dataclass(frozen=True)
class StepRecord:
    """Decision-time state and the immediately following policy outcomes."""

    fold: str
    era: str
    timestamp: datetime
    regime: CandidateRegime
    decision_gross_exposure: float
    decision_turnover: float
    next_net_return: float
    next_gross_return: float
    next_cost_ratio: float
    next_drawdown_change: float


@dataclass(frozen=True)
class AlignedStepRecord:
    """Agent and rule records at one exactly matched fold and timestamp."""

    agent: StepRecord
    rule: StepRecord


@dataclass(frozen=True)
class BucketAggregate:
    """Outcome means for one within-fold candidate bucket."""

    fold: str
    era: str
    first_timestamp: datetime
    bucket_index: int
    observation_count: int
    minimum_candidate_value: float
    maximum_candidate_value: float
    mean_next_net_return: float
    mean_next_gross_return: float
    mean_next_cost_ratio: float
    mean_next_drawdown_change: float


@dataclass(frozen=True)
class FoldEffect:
    """One fold-level high-minus-low and rank-association observation."""

    fold: str
    era: str
    first_timestamp: datetime
    high_minus_low_next_net_return: float
    high_minus_low_next_gross_return: float
    high_minus_low_next_cost_ratio: float
    high_minus_low_next_drawdown_change: float
    rank_association_next_net_return: float
    rank_association_next_gross_return: float
    rank_association_next_cost_ratio: float
    rank_association_next_drawdown_change: float


@dataclass(frozen=True)
class EraDirection:
    """Direction of a fold-mean high-minus-low effect within one era."""

    era: str
    fold_count: int
    mean_high_minus_low: float
    direction: Direction


@dataclass(frozen=True)
class MetricEvidence:
    """Fold-sampled evidence and explicit directional stability decision."""

    metric_name: MetricName
    fold_count: int
    mean_high_minus_low: float
    fold_direction_rate: float
    intervals: BootstrapIntervals
    era_directions: tuple[EraDirection, ...]
    is_stable: bool
    instability_reasons: tuple[str, ...]


def _require_finite_array(values: np.ndarray, origin: str) -> None:
    """Reject an array containing a missing or non-finite value.

    Args:
        values: Array to validate.
        origin: Input name included in an error.

    Raises:
        ValueError: If any value is missing or non-finite.
    """
    if not np.isfinite(values).all():
        raise ValueError(f"{origin} must contain only finite values.")


def _is_constant(values: np.ndarray) -> bool:
    """Return whether all array values are exactly equal.

    Args:
        values: Non-empty numeric array.

    Returns:
        Whether the array has no observed variation.
    """
    return bool(np.all(values == values.flat[0]))


def compute_candidate_regime(
    market_return_window: np.ndarray,
    carry_values: np.ndarray,
) -> CandidateRegime:
    """Compute causal candidates from the observation available at a decision.

    The input contains no future outcome. Rows are ordered observation times and
    columns are market pairs. The trend/reversal proxy is the lag-one Pearson
    association of the equal-weight market return: positive values indicate
    continuation and negative values indicate reversal.

    Args:
        market_return_window: Observed log returns shaped time by market pair.
        carry_values: Carry readings for the same market-pair columns.

    Returns:
        Six finite decision-time market candidates.

    Raises:
        ValueError: If shape, finiteness, pair alignment, or variation is invalid.
    """
    returns = np.asarray(market_return_window, dtype=np.float64)
    carry = np.asarray(carry_values, dtype=np.float64)
    if returns.ndim != 2:
        raise ValueError("market_return_window must be a two-dimensional array.")
    if returns.shape[0] < 3:
        raise ValueError("market_return_window must contain at least 3 observations.")
    if returns.shape[1] < 2:
        raise ValueError("market_return_window must contain at least 2 market pairs.")
    if carry.ndim != 1 or carry.shape[0] != returns.shape[1]:
        raise ValueError(
            "carry_values must be one-dimensional and aligned with market pairs."
        )
    _require_finite_array(returns, "market_return_window")
    _require_finite_array(carry, "carry_values")
    if _is_constant(carry):
        raise ValueError("constant carry cannot define carry dispersion.")
    constant_columns = [
        index for index in range(returns.shape[1]) if _is_constant(returns[:, index])
    ]
    if constant_columns:
        raise ValueError(
            "constant market return columns cannot define correlation: "
            f"{constant_columns}."
        )

    market_returns = returns.mean(axis=1)
    if _is_constant(market_returns):
        raise ValueError(
            "constant market return cannot define volatility or trend/reversal."
        )
    if _is_constant(market_returns[:-1]) or _is_constant(market_returns[1:]):
        raise ValueError(
            "constant market return lag cannot define trend/reversal association."
        )
    correlation = np.corrcoef(returns, rowvar=False)
    off_diagonal = correlation[np.triu_indices(returns.shape[1], k=1)]
    trend_reversal = float(
        np.corrcoef(market_returns[:-1], market_returns[1:])[0, 1]
    )
    candidates = CandidateRegime(
        realized_market_volatility=float(market_returns.std(ddof=1)),
        mean_cross_pair_correlation=float(off_diagonal.mean()),
        cross_sectional_return_dispersion=float(
            returns.std(axis=1, ddof=1).mean()
        ),
        momentum_dispersion=float(returns.sum(axis=0).std(ddof=1)),
        carry_dispersion=float(carry.std(ddof=1)),
        trend_reversal_proxy=trend_reversal,
    )
    _validate_regime(candidates, "computed candidate regime")
    return candidates


def _regime_values(regime: CandidateRegime) -> tuple[float, ...]:
    """Return candidate values in the declared market-candidate order.

    Args:
        regime: Candidate regime to flatten.

    Returns:
        Ordered market-candidate values.
    """
    return (
        regime.realized_market_volatility,
        regime.mean_cross_pair_correlation,
        regime.cross_sectional_return_dispersion,
        regime.momentum_dispersion,
        regime.carry_dispersion,
        regime.trend_reversal_proxy,
    )


def _validate_regime(regime: CandidateRegime, origin: str) -> None:
    """Validate that every market candidate is finite.

    Args:
        regime: Candidate regime to validate.
        origin: Record origin included in an error.

    Raises:
        ValueError: If a candidate is missing or non-finite.
    """
    values = np.asarray(_regime_values(regime), dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError(f"{origin} contains a missing or non-finite candidate.")


def _validate_record(record: StepRecord, origin: str) -> None:
    """Validate one complete diagnostic step record.

    Args:
        record: Record to validate.
        origin: Policy and position included in an error.

    Raises:
        ValueError: If identity, time, candidates, or responses are invalid.
    """
    if not record.fold:
        raise ValueError(f"{origin} has an empty fold.")
    if not record.era:
        raise ValueError(f"{origin} has an empty era.")
    if record.timestamp.tzinfo is None or record.timestamp.utcoffset() is None:
        raise ValueError(f"{origin} timestamp must include a timezone.")
    _validate_regime(record.regime, origin)
    values = np.asarray(
        (
            record.decision_gross_exposure,
            record.decision_turnover,
            record.next_net_return,
            record.next_gross_return,
            record.next_cost_ratio,
            record.next_drawdown_change,
        ),
        dtype=np.float64,
    )
    if not np.isfinite(values).all():
        raise ValueError(f"{origin} contains a missing or non-finite state/response.")
    if record.decision_gross_exposure < 0.0:
        raise ValueError(f"{origin} decision_gross_exposure must be non-negative.")
    if record.decision_turnover < 0.0:
        raise ValueError(f"{origin} decision_turnover must be non-negative.")
    if record.next_cost_ratio < 0.0:
        raise ValueError(f"{origin} next_cost_ratio must be non-negative.")


def align_agent_rule_records(
    agent_records: Sequence[StepRecord],
    rule_records: Sequence[StepRecord],
) -> tuple[AlignedStepRecord, ...]:
    """Align agent and rule records without dropping unmatched observations.

    Args:
        agent_records: Complete agent trace.
        rule_records: Complete benchmark-rule trace.

    Returns:
        Records sorted by fold and timestamp with exact keys and market regimes.

    Raises:
        ValueError: If either trace is empty, duplicated, invalid, or mismatched.
    """
    if not agent_records:
        raise ValueError("agent_records must not be empty.")
    if not rule_records:
        raise ValueError("rule_records must not be empty.")
    agent_by_key: dict[tuple[str, datetime], StepRecord] = {}
    rule_by_key: dict[tuple[str, datetime], StepRecord] = {}
    for index, record in enumerate(agent_records):
        _validate_record(record, f"agent record {index}")
        key = (record.fold, record.timestamp)
        if key in agent_by_key:
            raise ValueError(f"duplicate agent fold/timestamp: {key!r}.")
        agent_by_key[key] = record
    for index, record in enumerate(rule_records):
        _validate_record(record, f"rule record {index}")
        key = (record.fold, record.timestamp)
        if key in rule_by_key:
            raise ValueError(f"duplicate rule fold/timestamp: {key!r}.")
        rule_by_key[key] = record
    if agent_by_key.keys() != rule_by_key.keys():
        agent_only = sorted(agent_by_key.keys() - rule_by_key.keys())
        rule_only = sorted(rule_by_key.keys() - agent_by_key.keys())
        raise ValueError(
            "agent/rule fold/timestamp mismatch: "
            f"agent_only={agent_only!r}, rule_only={rule_only!r}."
        )

    aligned: list[AlignedStepRecord] = []
    for key in sorted(agent_by_key):
        agent = agent_by_key[key]
        rule = rule_by_key[key]
        if agent.era != rule.era:
            raise ValueError(
                f"agent/rule era mismatch at fold/timestamp {key!r}: "
                f"{agent.era!r} != {rule.era!r}."
            )
        if agent.regime != rule.regime:
            raise ValueError(f"agent/rule regime mismatch at fold/timestamp {key!r}.")
        aligned.append(AlignedStepRecord(agent=agent, rule=rule))
    return tuple(aligned)


def _candidate_value(record: StepRecord, candidate_name: CandidateName) -> float:
    """Select one declared candidate from a record.

    Args:
        record: Diagnostic step record.
        candidate_name: Declared market or policy-state candidate.

    Returns:
        Candidate value at the record's decision timestamp.
    """
    if candidate_name == "realized_market_volatility":
        return record.regime.realized_market_volatility
    if candidate_name == "mean_cross_pair_correlation":
        return record.regime.mean_cross_pair_correlation
    if candidate_name == "cross_sectional_return_dispersion":
        return record.regime.cross_sectional_return_dispersion
    if candidate_name == "momentum_dispersion":
        return record.regime.momentum_dispersion
    if candidate_name == "carry_dispersion":
        return record.regime.carry_dispersion
    if candidate_name == "trend_reversal_proxy":
        return record.regime.trend_reversal_proxy
    if candidate_name == "decision_gross_exposure":
        return record.decision_gross_exposure
    if candidate_name == "decision_turnover":
        return record.decision_turnover
    raise ValueError(f"unknown candidate_name: {candidate_name!r}.")


def _average_ranks(values: np.ndarray) -> np.ndarray:
    """Calculate deterministic one-based average ranks, including ties.

    Args:
        values: Finite non-constant one-dimensional values.

    Returns:
        Average ranks aligned with the input order.
    """
    order = np.argsort(values, kind="stable")
    sorted_values = values[order]
    sorted_ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        stop = start + 1
        while stop < len(values) and sorted_values[stop] == sorted_values[start]:
            stop += 1
        sorted_ranks[start:stop] = (start + 1 + stop) / 2.0
        start = stop
    ranks = np.empty(len(values), dtype=np.float64)
    ranks[order] = sorted_ranks
    return ranks


def _rank_association(candidate: np.ndarray, response: np.ndarray) -> float:
    """Calculate a Spearman rank association without an external dependency.

    Args:
        candidate: Finite non-constant candidate values.
        response: Finite non-constant response values.

    Returns:
        Finite Spearman rank association.

    Raises:
        ValueError: If the calculated association is not finite.
    """
    association = float(
        np.corrcoef(_average_ranks(candidate), _average_ranks(response))[0, 1]
    )
    if not np.isfinite(association):
        raise ValueError("rank association is non-finite.")
    return association


def _response_arrays(records: Sequence[StepRecord], fold: str) -> dict[MetricName, np.ndarray]:
    """Build and validate response arrays for a fold.

    Args:
        records: Validated records from one fold.
        fold: Fold identifier included in errors.

    Returns:
        Response arrays keyed by declared metric.

    Raises:
        ValueError: If any response is constant in the fold.
    """
    responses: dict[MetricName, np.ndarray] = {
        "next_net_return": np.asarray(
            [record.next_net_return for record in records], dtype=np.float64
        ),
        "next_gross_return": np.asarray(
            [record.next_gross_return for record in records], dtype=np.float64
        ),
        "next_cost_ratio": np.asarray(
            [record.next_cost_ratio for record in records], dtype=np.float64
        ),
        "next_drawdown_change": np.asarray(
            [record.next_drawdown_change for record in records], dtype=np.float64
        ),
    }
    for metric_name, values in responses.items():
        if _is_constant(values):
            raise ValueError(f"fold {fold!r} has constant response {metric_name!r}.")
    return responses


def compute_fold_effects(
    records: Sequence[StepRecord],
    candidate_name: CandidateName,
    bucket_count: int,
) -> tuple[tuple[BucketAggregate, ...], tuple[FoldEffect, ...]]:
    """Aggregate fold-local candidate buckets before deriving fold effects.

    Args:
        records: Complete steps for one policy across one or more folds.
        candidate_name: Candidate conditioned on within each fold.
        bucket_count: Number of equal-count rank buckets per fold.

    Returns:
        Bucket aggregates and one high-minus-low/rank effect per fold.

    Raises:
        ValueError: If controls, records, or fold-level variation are invalid.
    """
    if candidate_name not in MARKET_CANDIDATE_NAMES + POLICY_STATE_CANDIDATE_NAMES:
        raise ValueError(f"unknown candidate_name: {candidate_name!r}.")
    if isinstance(bucket_count, bool) or not isinstance(bucket_count, int):
        raise ValueError(f"bucket_count must be an integer, got {bucket_count!r}.")
    if bucket_count < 2:
        raise ValueError(f"bucket_count must be at least 2, got {bucket_count!r}.")
    if not records:
        raise ValueError("records must not be empty.")

    by_fold: dict[str, list[StepRecord]] = defaultdict(list)
    seen_keys: set[tuple[str, datetime]] = set()
    for index, record in enumerate(records):
        _validate_record(record, f"record {index}")
        key = (record.fold, record.timestamp)
        if key in seen_keys:
            raise ValueError(f"duplicate fold/timestamp: {key!r}.")
        seen_keys.add(key)
        by_fold[record.fold].append(record)

    fold_groups = sorted(
        by_fold.items(),
        key=lambda item: min(record.timestamp for record in item[1]),
    )
    buckets: list[BucketAggregate] = []
    effects: list[FoldEffect] = []
    for fold, unordered in fold_groups:
        fold_records = sorted(unordered, key=lambda record: record.timestamp)
        eras = {record.era for record in fold_records}
        if len(eras) != 1:
            raise ValueError(f"fold {fold!r} spans multiple eras: {sorted(eras)!r}.")
        era = fold_records[0].era
        if len(fold_records) < bucket_count:
            raise ValueError(
                f"fold {fold!r} has {len(fold_records)} records for "
                f"{bucket_count} buckets."
            )
        candidate = np.asarray(
            [_candidate_value(record, candidate_name) for record in fold_records],
            dtype=np.float64,
        )
        if _is_constant(candidate):
            raise ValueError(
                f"fold {fold!r} has constant candidate {candidate_name!r}."
            )
        responses = _response_arrays(fold_records, fold)
        unique_values = np.unique(candidate)
        if len(unique_values) < bucket_count:
            raise ValueError(
                f"fold {fold!r} has {len(unique_values)} distinct values for "
                f"{bucket_count} buckets for candidate {candidate_name!r}."
            )
        value_buckets = np.array_split(unique_values, bucket_count)
        bucket_indices = [
            np.flatnonzero(np.isin(candidate, values)) for values in value_buckets
        ]
        fold_buckets: list[BucketAggregate] = []
        for bucket_index, indices in enumerate(bucket_indices):
            aggregate = BucketAggregate(
                fold=fold,
                era=era,
                first_timestamp=fold_records[0].timestamp,
                bucket_index=bucket_index,
                observation_count=len(indices),
                minimum_candidate_value=float(candidate[indices].min()),
                maximum_candidate_value=float(candidate[indices].max()),
                mean_next_net_return=float(responses["next_net_return"][indices].mean()),
                mean_next_gross_return=float(
                    responses["next_gross_return"][indices].mean()
                ),
                mean_next_cost_ratio=float(responses["next_cost_ratio"][indices].mean()),
                mean_next_drawdown_change=float(
                    responses["next_drawdown_change"][indices].mean()
                ),
            )
            fold_buckets.append(aggregate)
            buckets.append(aggregate)
        low = fold_buckets[0]
        high = fold_buckets[-1]
        effects.append(
            FoldEffect(
                fold=fold,
                era=era,
                first_timestamp=fold_records[0].timestamp,
                high_minus_low_next_net_return=(
                    high.mean_next_net_return - low.mean_next_net_return
                ),
                high_minus_low_next_gross_return=(
                    high.mean_next_gross_return - low.mean_next_gross_return
                ),
                high_minus_low_next_cost_ratio=(
                    high.mean_next_cost_ratio - low.mean_next_cost_ratio
                ),
                high_minus_low_next_drawdown_change=(
                    high.mean_next_drawdown_change - low.mean_next_drawdown_change
                ),
                rank_association_next_net_return=_rank_association(
                    candidate, responses["next_net_return"]
                ),
                rank_association_next_gross_return=_rank_association(
                    candidate, responses["next_gross_return"]
                ),
                rank_association_next_cost_ratio=_rank_association(
                    candidate, responses["next_cost_ratio"]
                ),
                rank_association_next_drawdown_change=_rank_association(
                    candidate, responses["next_drawdown_change"]
                ),
            )
        )
    return tuple(buckets), tuple(effects)


def _effect_values(
    effect: FoldEffect,
    metric_name: MetricName,
) -> tuple[float, float]:
    """Select high-minus-low and rank effects for one response metric.

    Args:
        effect: Fold-level effects for every response.
        metric_name: Response metric to select.

    Returns:
        High-minus-low effect and rank association.
    """
    if metric_name == "next_net_return":
        return (
            effect.high_minus_low_next_net_return,
            effect.rank_association_next_net_return,
        )
    if metric_name == "next_gross_return":
        return (
            effect.high_minus_low_next_gross_return,
            effect.rank_association_next_gross_return,
        )
    if metric_name == "next_cost_ratio":
        return (
            effect.high_minus_low_next_cost_ratio,
            effect.rank_association_next_cost_ratio,
        )
    if metric_name == "next_drawdown_change":
        return (
            effect.high_minus_low_next_drawdown_change,
            effect.rank_association_next_drawdown_change,
        )
    raise ValueError(f"unknown metric_name: {metric_name!r}.")


def _direction(value: float, origin: str) -> Direction:
    """Convert a non-zero finite value into an explicit direction.

    Args:
        value: Effect whose sign is classified.
        origin: Calculation origin included in an error.

    Returns:
        Positive, negative, or neutral direction.

    Raises:
        ValueError: If the effect is non-finite.
    """
    if not np.isfinite(value):
        raise ValueError(f"{origin} must be finite, got {value!r}.")
    if value == 0.0:
        return "neutral"
    return "positive" if value > 0.0 else "negative"


def aggregate_fold_effects(
    effects: Sequence[FoldEffect],
    metric_name: MetricName,
    bootstrap_samples: int,
    bootstrap_seed: int,
    block_length: int,
    minimum_fold_direction_rate: float,
    era_order: Sequence[str],
    minimum_folds_per_era: int,
) -> MetricEvidence:
    """Bootstrap ordered fold effects and apply the stability contract.

    Stability requires the declared rate of folds to agree in both bucket and
    rank direction, both 95% intervals to exclude zero, every declared era to
    share the overall direction, and every era to contain enough folds. The
    final condition explicitly prevents an isolated tail year from being called
    stable.

    Args:
        effects: One high-minus-low/rank observation per evaluation fold.
        metric_name: Outcome metric being assessed.
        bootstrap_samples: Number of fold and moving-block resamples.
        bootstrap_seed: Deterministic NumPy bootstrap seed.
        block_length: Circular moving-block length in adjacent folds.
        minimum_fold_direction_rate: Predeclared agreement rate in `(0, 1]`.
        era_order: Complete, unique, chronological declared era names.
        minimum_folds_per_era: Minimum independent folds required in every era.

    Returns:
        Fold-sampled uncertainty, era directions, and stability decision.

    Raises:
        ValueError: If controls, fold effects, or declared eras are invalid.
    """
    if metric_name not in METRIC_NAMES:
        raise ValueError(f"unknown metric_name: {metric_name!r}.")
    if not effects:
        raise ValueError("effects must not be empty.")
    if (
        not np.isfinite(minimum_fold_direction_rate)
        or minimum_fold_direction_rate <= 0.0
        or minimum_fold_direction_rate > 1.0
    ):
        raise ValueError(
            "minimum_fold_direction_rate must be finite and in (0, 1], got "
            f"{minimum_fold_direction_rate!r}."
        )
    if (
        isinstance(minimum_folds_per_era, bool)
        or not isinstance(minimum_folds_per_era, int)
        or minimum_folds_per_era < 1
    ):
        raise ValueError(
            "minimum_folds_per_era must be a positive integer, got "
            f"{minimum_folds_per_era!r}."
        )
    declared_eras = tuple(era_order)
    if not declared_eras or any(not era for era in declared_eras):
        raise ValueError("era_order must contain non-empty declared eras.")
    if len(set(declared_eras)) != len(declared_eras):
        raise ValueError(f"era_order contains duplicate eras: {declared_eras!r}.")

    ordered = sorted(effects, key=lambda effect: effect.first_timestamp)
    folds = [effect.fold for effect in ordered]
    if any(not fold for fold in folds) or len(set(folds)) != len(folds):
        raise ValueError(f"effects must contain one unique non-empty fold: {folds!r}.")
    observed_eras = {effect.era for effect in ordered}
    if observed_eras != set(declared_eras):
        raise ValueError(
            "era_order must exactly match effect eras: "
            f"declared={declared_eras!r}, observed={sorted(observed_eras)!r}."
        )

    selected = [_effect_values(effect, metric_name) for effect in ordered]
    high_minus_low = np.asarray([value[0] for value in selected], dtype=np.float64)
    rank_associations = np.asarray([value[1] for value in selected], dtype=np.float64)
    _require_finite_array(high_minus_low, f"{metric_name} high-minus-low effects")
    _require_finite_array(rank_associations, f"{metric_name} rank associations")
    mean_effect = float(high_minus_low.mean())
    overall_direction = _direction(mean_effect, f"mean {metric_name} high-minus-low")
    if overall_direction == "neutral":
        direction_rate = 0.0
    else:
        expected_sign = 1.0 if overall_direction == "positive" else -1.0
        directional_matches = (high_minus_low * expected_sign > 0.0) & (
            rank_associations * expected_sign > 0.0
        )
        direction_rate = float(directional_matches.mean())
    intervals = bootstrap_mean_intervals(
        high_minus_low,
        bootstrap_samples,
        bootstrap_seed,
        block_length,
    )

    era_directions: list[EraDirection] = []
    instability_reasons: list[str] = []
    if overall_direction == "neutral":
        instability_reasons.append("overall mean direction is neutral")
    if direction_rate < minimum_fold_direction_rate:
        instability_reasons.append(
            f"fold direction rate {direction_rate:.6f} is below required "
            f"{minimum_fold_direction_rate:.6f}"
        )
    if not (intervals.fold_low > 0.0 or intervals.fold_high < 0.0):
        instability_reasons.append("fold bootstrap interval includes zero")
    if not (
        intervals.moving_block_low > 0.0 or intervals.moving_block_high < 0.0
    ):
        instability_reasons.append("moving-block bootstrap interval includes zero")
    for era in declared_eras:
        era_values = np.asarray(
            [
                high_minus_low[index]
                for index, effect in enumerate(ordered)
                if effect.era == era
            ],
            dtype=np.float64,
        )
        era_mean = float(era_values.mean())
        era_direction = _direction(era_mean, f"era {era!r} mean {metric_name}")
        era_directions.append(
            EraDirection(
                era=era,
                fold_count=len(era_values),
                mean_high_minus_low=era_mean,
                direction=era_direction,
            )
        )
        if len(era_values) < minimum_folds_per_era:
            instability_reasons.append(
                f"era {era!r} has {len(era_values)} fold; at least "
                f"{minimum_folds_per_era} are required"
            )
        if overall_direction != "neutral" and era_direction != overall_direction:
            instability_reasons.append(
                f"era {era!r} direction is {era_direction}, expected "
                f"{overall_direction}"
            )
    return MetricEvidence(
        metric_name=metric_name,
        fold_count=len(ordered),
        mean_high_minus_low=mean_effect,
        fold_direction_rate=direction_rate,
        intervals=intervals,
        era_directions=tuple(era_directions),
        is_stable=not instability_reasons,
        instability_reasons=tuple(instability_reasons),
    )
