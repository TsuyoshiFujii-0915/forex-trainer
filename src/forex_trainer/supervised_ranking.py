"""Pure dataset, ridge, and diagnostics logic for supervised pair ranking."""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime
from types import MappingProxyType
from typing import Mapping

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SupervisedDataset:
    """One causal feature window and next relative return per decision and pair."""

    features: np.ndarray
    targets: np.ndarray
    decision_timestamps: tuple[datetime, ...]
    target_timestamps: tuple[datetime, ...]
    symbols: tuple[str, ...]
    feature_names: tuple[str, ...]

    def __post_init__(self) -> None:
        """Validate shapes, timestamps, and finite values at construction."""
        features = np.asarray(self.features, dtype=np.float64)
        targets = np.asarray(self.targets, dtype=np.float64)
        if features.ndim != 3:
            raise ValueError(
                "dataset features must have shape decisions by pairs by features."
            )
        if targets.shape != features.shape[:2]:
            raise ValueError(
                "dataset targets must have shape decisions by pairs matching features."
            )
        if features.shape[0] < 1 or features.shape[1] < 2 or features.shape[2] < 1:
            raise ValueError(
                "dataset requires at least one decision, two pairs, and one feature."
            )
        if not np.isfinite(features).all() or not np.isfinite(targets).all():
            raise ValueError("dataset features and targets must be finite.")
        if len(self.decision_timestamps) != features.shape[0]:
            raise ValueError("decision timestamp count must match dataset decisions.")
        if len(self.target_timestamps) != features.shape[0]:
            raise ValueError("target timestamp count must match dataset decisions.")
        if len(self.symbols) != features.shape[1] or len(set(self.symbols)) != len(
            self.symbols
        ):
            raise ValueError("symbols must be unique and match the dataset pair axis.")
        if len(self.feature_names) != features.shape[2] or len(
            set(self.feature_names)
        ) != len(self.feature_names):
            raise ValueError(
                "feature names must be unique and match the dataset feature axis."
            )
        for decision, target in zip(
            self.decision_timestamps, self.target_timestamps
        ):
            decision_is_aware = (
                decision.tzinfo is not None and decision.utcoffset() is not None
            )
            target_is_aware = target.tzinfo is not None and target.utcoffset() is not None
            if decision_is_aware != target_is_aware:
                raise ValueError(
                    "decision and target timestamps must use the same timezone convention."
                )
            if decision >= target:
                raise ValueError(
                    "each target timestamp must be strictly after its decision timestamp."
                )
        object.__setattr__(self, "features", features)
        object.__setattr__(self, "targets", targets)


@dataclass(frozen=True)
class Standardizer:
    """Training-only feature location and scale."""

    mean: np.ndarray
    scale: np.ndarray


@dataclass(frozen=True)
class RidgeModel:
    """Linear ridge coefficients in the supplied feature coordinate system."""

    coefficients: np.ndarray
    intercept: float
    alpha: float

    def predict(self, features: np.ndarray) -> np.ndarray:
        """Predict pair scores while preserving leading dimensions.

        Args:
            features: Array whose final axis matches the coefficient count.

        Returns:
            Score array with the final feature axis removed.

        Raises:
            ValueError: If features have an invalid shape or non-finite values.
        """
        values = np.asarray(features, dtype=np.float64)
        if values.ndim < 2 or values.shape[-1] != len(self.coefficients):
            raise ValueError(
                "prediction features must end with the model feature dimension."
            )
        if not np.isfinite(values).all():
            raise ValueError("prediction features must be finite.")
        return values @ self.coefficients + self.intercept


@dataclass(frozen=True)
class AlphaSelection:
    """Validation-only ridge regularization selection result."""

    selected_alpha: float
    mean_rank_ic_by_alpha: Mapping[float, float | None]


@dataclass(frozen=True)
class ScoreDiagnostics:
    """Decision-level predictive diagnostics reduced to one fold observation."""

    mean_rank_ic: float | None
    median_rank_ic: float | None
    positive_rank_ic_fraction: float | None
    rank_ic_observation_fraction: float
    mean_tail_spread: float
    tail_ordering_accuracy: float
    mean_top_overlap: float
    mean_bottom_overlap: float
    mean_score_dispersion: float
    mean_rank_churn: float


@dataclass(frozen=True)
class ClassificationInputs:
    """Pre-registered evidence required for the learnability classification."""

    aggregate_mean_rank_ic: float | None
    aggregate_mean_tail_spread: float
    iid_tail_spread_low: float
    moving_block_tail_spread_low: float
    era_tail_spreads: tuple[float, ...]
    leave_one_fold_out_tail_spreads: tuple[float, ...]
    coherent_reversal: bool
    non_degenerate_scores: bool


def _expanded_feature_names(
    feature_names: tuple[str, ...], window_size: int
) -> tuple[str, ...]:
    """Expand base feature names into stable oldest-to-current lag columns.

    Args:
        feature_names: Base environment feature order.
        window_size: Observation window row count.

    Returns:
        Lagged feature names in the same order as a C-order window flatten.
    """
    return tuple(
        f"{name}_lag_{lag}"
        for lag in range(window_size - 1, -1, -1)
        for name in feature_names
    )


def build_aligned_dataset(
    timestamps: pd.DatetimeIndex,
    closes: np.ndarray,
    features: np.ndarray,
    feature_names: tuple[str, ...],
    symbols: tuple[str, ...],
    warmup: int,
    window_size: int,
) -> SupervisedDataset:
    """Align causal observation windows with one-step cross-sectional labels.

    Args:
        timestamps: Raw market row timestamps.
        closes: Close matrix shaped market rows by pairs.
        features: Full feature tensor shaped pairs by market rows by features.
        feature_names: Base feature order in the feature tensor.
        symbols: Stable configured pair order.
        warmup: Leading feature rows unavailable to the environment.
        window_size: Number of feature rows in one ``longf`` observation.

    Returns:
        Dataset shaped decisions by pairs by flattened lagged features.

    Raises:
        ValueError: If inputs are malformed, non-finite, or too short.
    """
    close_values = np.asarray(closes, dtype=np.float64)
    feature_values = np.asarray(features, dtype=np.float64)
    if not isinstance(timestamps, pd.DatetimeIndex):
        raise ValueError("timestamps must be a DatetimeIndex.")
    if not timestamps.is_monotonic_increasing or not timestamps.is_unique:
        raise ValueError("timestamps must be unique and strictly increasing.")
    if close_values.shape != (len(timestamps), len(symbols)):
        raise ValueError("closes must have shape timestamps by configured symbols.")
    if feature_values.ndim != 3 or feature_values.shape[:2] != (
        len(symbols),
        len(timestamps),
    ):
        raise ValueError(
            "features must have shape configured symbols by timestamps by features."
        )
    if feature_values.shape[2] != len(feature_names):
        raise ValueError("feature_names must match the final feature tensor axis.")
    if len(set(symbols)) != len(symbols) or len(set(feature_names)) != len(
        feature_names
    ):
        raise ValueError("symbols and feature names must be unique.")
    if isinstance(warmup, bool) or not isinstance(warmup, int) or warmup < 0:
        raise ValueError("warmup must be a non-negative integer.")
    if (
        isinstance(window_size, bool)
        or not isinstance(window_size, int)
        or window_size < 1
    ):
        raise ValueError("window_size must be a positive integer.")
    first_decision = warmup + window_size - 1
    if first_decision >= len(timestamps) - 1:
        raise ValueError(
            "market data is too short for warmup, observation window, and target."
        )
    if not np.isfinite(close_values).all() or bool(np.any(close_values <= 0.0)):
        raise ValueError("closes must contain finite positive values.")
    usable_features = feature_values[:, warmup:, :]
    if not np.isfinite(usable_features).all():
        raise ValueError("features must be finite after the declared warmup.")

    decision_indices = range(first_decision, len(timestamps) - 1)
    feature_rows: list[np.ndarray] = []
    target_rows: list[np.ndarray] = []
    decision_times: list[datetime] = []
    target_times: list[datetime] = []
    for decision_index in decision_indices:
        window_start = decision_index - window_size + 1
        window = feature_values[:, window_start : decision_index + 1, :]
        feature_rows.append(window.reshape(len(symbols), -1))
        next_log_returns = np.log(
            close_values[decision_index + 1] / close_values[decision_index]
        )
        target_rows.append(next_log_returns - next_log_returns.mean())
        decision_times.append(timestamps[decision_index].to_pydatetime())
        target_times.append(timestamps[decision_index + 1].to_pydatetime())
    dataset = SupervisedDataset(
        features=np.stack(feature_rows),
        targets=np.stack(target_rows),
        decision_timestamps=tuple(decision_times),
        target_timestamps=tuple(target_times),
        symbols=symbols,
        feature_names=_expanded_feature_names(feature_names, window_size),
    )
    if not np.allclose(dataset.targets.mean(axis=1), 0.0, rtol=0.0, atol=1e-14):
        raise ValueError("cross-sectional relative-return targets must average to zero.")
    return dataset


def fit_standardizer(features: np.ndarray) -> Standardizer:
    """Fit feature standardization parameters on training rows only.

    Args:
        features: Training array shaped decisions by pairs by features.

    Returns:
        Population mean and standard deviation for each feature.

    Raises:
        ValueError: If the training matrix is malformed, non-finite, or constant.
    """
    values = np.asarray(features, dtype=np.float64)
    if values.ndim != 3 or min(values.shape) < 1 or not np.isfinite(values).all():
        raise ValueError("training features must be a finite non-empty 3D array.")
    pooled = values.reshape(-1, values.shape[2])
    mean = pooled.mean(axis=0)
    scale = pooled.std(axis=0)
    constant = np.flatnonzero(scale == 0.0)
    if len(constant) > 0:
        raise ValueError(
            f"training feature has zero variance at feature {int(constant[0])}."
        )
    return Standardizer(mean=mean, scale=scale)


def apply_standardizer(
    features: np.ndarray, standardizer: Standardizer
) -> np.ndarray:
    """Apply fixed training standardization parameters.

    Args:
        features: Array shaped decisions by pairs by features.
        standardizer: Parameters fitted from training rows.

    Returns:
        Standardized array with the same shape.

    Raises:
        ValueError: If shapes or values are invalid.
    """
    values = np.asarray(features, dtype=np.float64)
    mean = np.asarray(standardizer.mean, dtype=np.float64)
    scale = np.asarray(standardizer.scale, dtype=np.float64)
    if values.ndim != 3 or mean.ndim != 1 or scale.shape != mean.shape:
        raise ValueError("standardizer inputs have invalid dimensions.")
    if values.shape[2] != len(mean):
        raise ValueError("standardizer feature count does not match the input.")
    if (
        not np.isfinite(values).all()
        or not np.isfinite(mean).all()
        or not np.isfinite(scale).all()
        or bool(np.any(scale <= 0.0))
    ):
        raise ValueError("standardizer inputs must be finite with positive scales.")
    return (values - mean) / scale


def fit_ridge(dataset: SupervisedDataset, alpha: float) -> RidgeModel:
    """Fit a pooled linear ridge model with an unpenalized intercept.

    Args:
        dataset: Training dataset in the desired feature coordinates.
        alpha: Non-negative ridge penalty. Zero uses least squares.

    Returns:
        Deterministic fitted linear model.

    Raises:
        ValueError: If alpha is invalid or the linear solve fails.
    """
    if isinstance(alpha, bool) or not isinstance(alpha, (int, float)):
        raise ValueError("ridge alpha must be a finite non-negative number.")
    penalty = float(alpha)
    if not math.isfinite(penalty) or penalty < 0.0:
        raise ValueError("ridge alpha must be a finite non-negative number.")
    features = dataset.features.reshape(-1, dataset.features.shape[2])
    targets = dataset.targets.reshape(-1)
    feature_mean = features.mean(axis=0)
    target_mean = float(targets.mean())
    centered_features = features - feature_mean
    centered_targets = targets - target_mean
    if penalty == 0.0:
        coefficients, _, _, _ = np.linalg.lstsq(
            centered_features, centered_targets, rcond=None
        )
    else:
        gram = centered_features.T @ centered_features
        right = centered_features.T @ centered_targets
        coefficients = np.linalg.solve(
            gram + penalty * np.eye(gram.shape[0], dtype=np.float64), right
        )
    if not np.isfinite(coefficients).all():
        raise ValueError("ridge fitting produced non-finite coefficients.")
    intercept = target_mean - float(feature_mean @ coefficients)
    return RidgeModel(
        coefficients=np.asarray(coefficients, dtype=np.float64),
        intercept=intercept,
        alpha=penalty,
    )


def _average_ranks(values: np.ndarray) -> np.ndarray:
    """Assign one-based average ranks with stable sorting for ties.

    Args:
        values: Finite one-dimensional values to rank.

    Returns:
        One-based average ranks in the original element order.
    """
    order = np.argsort(values, kind="stable")
    sorted_values = values[order]
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        stop = start + 1
        while stop < len(values) and sorted_values[stop] == sorted_values[start]:
            stop += 1
        ranks[order[start:stop]] = (start + stop - 1) / 2.0 + 1.0
        start = stop
    return ranks


def cross_sectional_spearman(scores: np.ndarray, targets: np.ndarray) -> float:
    """Compute tie-aware Spearman correlation for one decision cross-section.

    Args:
        scores: Predicted pair scores.
        targets: Realized pair targets in the same order.

    Returns:
        Pearson correlation of average ranks.

    Raises:
        ValueError: If values are malformed, non-finite, or constant.
    """
    left = np.asarray(scores, dtype=np.float64)
    right = np.asarray(targets, dtype=np.float64)
    if (
        left.ndim != 1
        or right.shape != left.shape
        or len(left) < 2
        or not np.isfinite(left).all()
        or not np.isfinite(right).all()
    ):
        raise ValueError("Spearman inputs must be equal finite 1D pair arrays.")
    left_ranks = _average_ranks(left)
    right_ranks = _average_ranks(right)
    if float(left_ranks.std()) == 0.0:
        raise ValueError("Spearman is undefined for a constant score.")
    if float(right_ranks.std()) == 0.0:
        raise ValueError("Spearman is undefined for a constant target.")
    return float(np.corrcoef(left_ranks, right_ranks)[0, 1])


def select_validation_alpha(
    training: SupervisedDataset,
    validation: SupervisedDataset,
    alphas: tuple[float, ...],
) -> AlphaSelection:
    """Select ridge alpha using validation mean cross-sectional rank IC only.

    Args:
        training: Standardized training dataset.
        validation: Validation dataset transformed with training parameters.
        alphas: Fixed non-empty candidate grid.

    Returns:
        Selected alpha and every validation score. Exact ties prefer larger alpha.

    Raises:
        ValueError: If the grid is empty, duplicated, or invalid.
    """
    if not alphas or len(set(alphas)) != len(alphas):
        raise ValueError("ridge alpha grid must be non-empty and unique.")
    scores_by_alpha: dict[float, float | None] = {}
    for candidate in alphas:
        model = fit_ridge(training, candidate)
        predictions = model.predict(validation.features)
        rank_ics = [
            cross_sectional_spearman(score_row, target_row)
            for score_row, target_row in zip(predictions, validation.targets)
            if float(target_row.std()) > 0.0
        ]
        scores_by_alpha[model.alpha] = (
            float(np.mean(rank_ics)) if rank_ics else None
        )
    eligible = {
        alpha: value for alpha, value in scores_by_alpha.items() if value is not None
    }
    selected = (
        max(eligible, key=lambda alpha: (eligible[alpha], alpha))
        if eligible
        else max(scores_by_alpha)
    )
    return AlphaSelection(
        selected_alpha=selected,
        mean_rank_ic_by_alpha=MappingProxyType(dict(scores_by_alpha)),
    )


def canonical_reversal_scores(dataset: SupervisedDataset) -> np.ndarray:
    """Extract negative current ``mom24`` as the frozen reversal benchmark.

    Args:
        dataset: Dataset whose expanded columns contain ``mom24_lag_0``.

    Returns:
        Decision-by-pair canonical reversal score matrix.

    Raises:
        ValueError: If the required current momentum column is absent.
    """
    name = "mom24_lag_0"
    if name not in dataset.feature_names:
        raise ValueError(f"dataset lacks canonical reversal feature {name!r}.")
    return -dataset.features[:, :, dataset.feature_names.index(name)]


def compute_score_diagnostics(
    scores: np.ndarray, targets: np.ndarray, top_k: int
) -> ScoreDiagnostics:
    """Reduce decision-level ordering diagnostics to one fold observation.

    Args:
        scores: Decision-by-pair predictive scores.
        targets: Aligned next relative returns.
        top_k: Number of predicted and realized pairs in each rank tail.

    Returns:
        Fold-level predictive diagnostic values.

    Raises:
        ValueError: If arrays or tail size are invalid.
    """
    score_values = np.asarray(scores, dtype=np.float64)
    target_values = np.asarray(targets, dtype=np.float64)
    if (
        score_values.ndim != 2
        or target_values.shape != score_values.shape
        or score_values.shape[0] < 1
        or not np.isfinite(score_values).all()
        or not np.isfinite(target_values).all()
    ):
        raise ValueError("scores and targets must be equal finite decision-pair arrays.")
    pair_count = score_values.shape[1]
    if (
        isinstance(top_k, bool)
        or not isinstance(top_k, int)
        or top_k < 1
        or top_k * 2 > pair_count
    ):
        raise ValueError("top_k must select disjoint non-empty pair tails.")
    rank_ics: list[float] = []
    tail_spreads: list[float] = []
    top_overlaps: list[float] = []
    bottom_overlaps: list[float] = []
    dispersions: list[float] = []
    for score_row, target_row in zip(score_values, target_values):
        if float(target_row.std()) > 0.0 and float(score_row.std()) > 0.0:
            rank_ics.append(cross_sectional_spearman(score_row, target_row))
        predicted_order = np.argsort(score_row, kind="stable")
        realized_order = np.argsort(target_row, kind="stable")
        predicted_bottom = predicted_order[:top_k]
        predicted_top = predicted_order[-top_k:]
        realized_bottom = set(realized_order[:top_k].tolist())
        realized_top = set(realized_order[-top_k:].tolist())
        tail_spreads.append(
            float(target_row[predicted_top].mean() - target_row[predicted_bottom].mean())
        )
        top_overlaps.append(
            len(set(predicted_top.tolist()) & realized_top) / float(top_k)
        )
        bottom_overlaps.append(
            len(set(predicted_bottom.tolist()) & realized_bottom) / float(top_k)
        )
        dispersions.append(float(score_row.std()))
    churn = [
        float(
            np.mean(np.abs(_average_ranks(previous) - _average_ranks(current)))
            / (pair_count - 1)
        )
        for previous, current in zip(score_values[:-1], score_values[1:])
    ]
    rank_array = np.asarray(rank_ics, dtype=np.float64)
    spread_array = np.asarray(tail_spreads, dtype=np.float64)
    return ScoreDiagnostics(
        mean_rank_ic=float(rank_array.mean()) if rank_ics else None,
        median_rank_ic=float(np.median(rank_array)) if rank_ics else None,
        positive_rank_ic_fraction=float(np.mean(rank_array > 0.0)) if rank_ics else None,
        rank_ic_observation_fraction=len(rank_ics) / float(len(score_values)),
        mean_tail_spread=float(spread_array.mean()),
        tail_ordering_accuracy=float(np.mean(spread_array > 0.0)),
        mean_top_overlap=float(np.mean(top_overlaps)),
        mean_bottom_overlap=float(np.mean(bottom_overlaps)),
        mean_score_dispersion=float(np.mean(dispersions)),
        mean_rank_churn=float(np.mean(churn)) if churn else 0.0,
    )


def classify_learnability(inputs: ClassificationInputs) -> str:
    """Apply the pre-registered established/suggestive/not-established rules.

    Args:
        inputs: Aggregate, interval, era, dominance, and coherence evidence.

    Returns:
        Exact Issue #15 classification label.

    Raises:
        ValueError: If numeric evidence is absent or non-finite.
    """
    numeric = (
        inputs.aggregate_mean_tail_spread,
        inputs.iid_tail_spread_low,
        inputs.moving_block_tail_spread_low,
        *inputs.era_tail_spreads,
        *inputs.leave_one_fold_out_tail_spreads,
    )
    if (
        not inputs.era_tail_spreads
        or not inputs.leave_one_fold_out_tail_spreads
        or not all(math.isfinite(value) for value in numeric)
    ):
        raise ValueError("classification requires complete finite fold evidence.")
    point_positive = (
        inputs.aggregate_mean_rank_ic is not None
        and math.isfinite(inputs.aggregate_mean_rank_ic)
        and inputs.aggregate_mean_rank_ic > 0.0
        and inputs.aggregate_mean_tail_spread > 0.0
    )
    coherent = inputs.coherent_reversal and inputs.non_degenerate_scores
    established = (
        point_positive
        and inputs.iid_tail_spread_low > 0.0
        and inputs.moving_block_tail_spread_low > 0.0
        and all(value > 0.0 for value in inputs.era_tail_spreads)
        and all(value > 0.0 for value in inputs.leave_one_fold_out_tail_spreads)
        and coherent
    )
    if established:
        return "established learnable"
    not_one_fold_dominated = all(
        value > 0.0 for value in inputs.leave_one_fold_out_tail_spreads
    )
    if point_positive and coherent and not_one_fold_dominated:
        return "suggestive"
    return "not established"
