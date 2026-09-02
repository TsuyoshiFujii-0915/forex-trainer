"""Behavior tests for the supervised cross-sectional ranking diagnostic."""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

from forex_trainer.supervised_ranking import (
    ClassificationInputs,
    RidgeModel,
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


def _dataset(features: np.ndarray, targets: np.ndarray) -> SupervisedDataset:
    """Build a compact timestamped dataset for model-selection tests."""
    decision_count, pair_count, _ = features.shape
    decisions = tuple(
        datetime(2020, 1, day + 1, tzinfo=timezone.utc)
        for day in range(decision_count)
    )
    targets_at = tuple(
        datetime(2020, 1, day + 2, tzinfo=timezone.utc)
        for day in range(decision_count)
    )
    return SupervisedDataset(
        features=features.astype(np.float64),
        targets=targets.astype(np.float64),
        decision_timestamps=decisions,
        target_timestamps=targets_at,
        symbols=tuple(f"JPY/P{index}" for index in range(pair_count)),
        feature_names=tuple(f"feature_{index}" for index in range(features.shape[2])),
    )


def test_dataset_uses_features_at_t_and_target_only_from_t_to_t_plus_one() -> None:
    """Rows expose time-t features while labels use only the immediately next close."""
    timestamps = pd.date_range("2020-01-01", periods=6, freq="D", tz="UTC")
    symbols = ("JPY/USD", "JPY/EUR", "JPY/GBP")
    closes = np.array(
        [
            [100.0, 200.0, 400.0],
            [101.0, 198.0, 404.0],
            [102.0, 202.0, 400.0],
            [104.0, 204.0, 408.0],
            [103.0, 208.0, 412.0],
            [106.0, 206.0, 420.0],
        ]
    )
    features = np.stack(
        [
            np.column_stack(
                [np.arange(6, dtype=float) + pair, np.full(6, pair, dtype=float)]
            )
            for pair in range(3)
        ]
    )

    dataset = build_aligned_dataset(
        timestamps=timestamps,
        closes=closes,
        features=features,
        feature_names=("time", "pair"),
        symbols=symbols,
        warmup=1,
        window_size=2,
    )

    assert dataset.decision_timestamps == tuple(timestamps[2:-1].to_pydatetime())
    assert dataset.target_timestamps == tuple(timestamps[3:].to_pydatetime())
    np.testing.assert_allclose(
        dataset.features[0], features[:, 1:3, :].reshape(3, 4)
    )
    assert dataset.feature_names == (
        "time_lag_1",
        "pair_lag_1",
        "time_lag_0",
        "pair_lag_0",
    )
    next_returns = np.log(closes[3] / closes[2])
    np.testing.assert_allclose(dataset.targets[0], next_returns - next_returns.mean())
    np.testing.assert_allclose(dataset.targets.mean(axis=1), 0.0, atol=1e-15)


def test_future_close_changes_only_the_label_that_ends_at_that_close() -> None:
    """A future price cannot alter earlier features or labels ending before it."""
    timestamps = pd.date_range("2020-01-01", periods=6, freq="D", tz="UTC")
    closes = np.array(
        [
            [100.0, 100.0],
            [101.0, 99.0],
            [102.0, 98.0],
            [103.0, 97.0],
            [104.0, 96.0],
            [105.0, 95.0],
        ]
    )
    features = np.stack([closes.T, np.square(closes.T)], axis=2)
    original = build_aligned_dataset(
        timestamps, closes, features, ("close", "close_squared"),
        ("JPY/USD", "JPY/EUR"), 1, 2,
    )
    changed_closes = closes.copy()
    changed_closes[-1, 0] *= 2.0
    changed = build_aligned_dataset(
        timestamps, changed_closes, features, ("close", "close_squared"),
        ("JPY/USD", "JPY/EUR"), 1, 2,
    )

    np.testing.assert_array_equal(changed.features, original.features)
    np.testing.assert_allclose(changed.targets[:-1], original.targets[:-1])
    assert not np.allclose(changed.targets[-1], original.targets[-1])


def test_standardization_is_fit_on_training_rows_only() -> None:
    """Validation and evaluation values never change fitted transformation parameters."""
    training = np.array(
        [
            [[1.0, 10.0], [3.0, 14.0]],
            [[5.0, 18.0], [7.0, 22.0]],
        ]
    )
    evaluation = np.array([[[1_000.0, -5_000.0], [2_000.0, -8_000.0]]])

    standardizer = fit_standardizer(training)
    transformed_training = apply_standardizer(training, standardizer)
    transformed_evaluation = apply_standardizer(evaluation, standardizer)

    np.testing.assert_allclose(standardizer.mean, [4.0, 16.0])
    np.testing.assert_allclose(transformed_training.reshape(-1, 2).mean(axis=0), 0.0)
    assert np.abs(transformed_evaluation).min() > 100.0


def test_standardizer_rejects_constant_training_features() -> None:
    """A degenerate train column is an explicit data error, never silently dropped."""
    features = np.array([[[1.0, 2.0], [1.0, 3.0]]])

    with pytest.raises(ValueError, match="zero variance.*feature 0"):
        fit_standardizer(features)


def test_ridge_alpha_is_selected_only_by_validation_ic_with_strong_tie_break() -> None:
    """Equal validation ranks choose stronger regularization regardless of evaluation."""
    train_x = np.array(
        [
            [[-2.0], [-1.0], [1.0], [2.0]],
            [[-3.0], [-0.5], [0.5], [3.0]],
        ]
    )
    train_y = train_x[:, :, 0] * 0.01
    validation_x = np.array(
        [
            [[-4.0], [-2.0], [2.0], [4.0]],
            [[-1.5], [-0.2], [0.2], [1.5]],
        ]
    )
    validation_y = validation_x[:, :, 0] * 0.02
    standardizer = fit_standardizer(train_x)

    selection = select_validation_alpha(
        _dataset(apply_standardizer(train_x, standardizer), train_y),
        _dataset(apply_standardizer(validation_x, standardizer), validation_y),
        (0.0, 0.1, 1.0, 10.0),
    )

    assert selection.selected_alpha == 10.0
    assert set(selection.mean_rank_ic_by_alpha) == {0.0, 0.1, 1.0, 10.0}
    assert all(value == pytest.approx(1.0) for value in selection.mean_rank_ic_by_alpha.values())


def test_ridge_fit_recovers_common_relationship_without_pair_identity() -> None:
    """The pooled model learns shared feature coefficients and no pair fixed effect."""
    features = np.array(
        [
            [[-2.0, 1.0], [-1.0, -1.0], [1.0, 1.0], [2.0, -1.0]],
            [[-3.0, -1.0], [-0.5, 1.0], [0.5, -1.0], [3.0, 1.0]],
        ]
    )
    targets = 2.0 * features[:, :, 0] - 0.5 * features[:, :, 1]

    model = fit_ridge(_dataset(features, targets), alpha=0.0)

    assert isinstance(model, RidgeModel)
    np.testing.assert_allclose(model.coefficients, [2.0, -0.5], atol=1e-12)
    assert model.intercept == pytest.approx(0.0, abs=1e-12)
    assert len(model.coefficients) == features.shape[2]


def test_score_diagnostics_measure_rank_tail_overlap_dispersion_and_churn() -> None:
    """Predictive diagnostics are computed per decision before fold reduction."""
    scores = np.array(
        [
            [-3.0, -2.0, 2.0, 3.0],
            [-2.0, -3.0, 3.0, 2.0],
        ]
    )
    targets = np.array(
        [
            [-0.03, -0.01, 0.01, 0.03],
            [-0.01, -0.03, 0.03, 0.01],
        ]
    )

    diagnostics = compute_score_diagnostics(scores, targets, top_k=1)

    assert diagnostics.mean_rank_ic == pytest.approx(1.0)
    assert diagnostics.median_rank_ic == pytest.approx(1.0)
    assert diagnostics.positive_rank_ic_fraction == pytest.approx(1.0)
    assert diagnostics.mean_tail_spread == pytest.approx(0.06)
    assert diagnostics.tail_ordering_accuracy == pytest.approx(1.0)
    assert diagnostics.mean_top_overlap == pytest.approx(1.0)
    assert diagnostics.mean_bottom_overlap == pytest.approx(1.0)
    assert diagnostics.mean_score_dispersion > 0.0
    assert 0.0 < diagnostics.mean_rank_churn < 1.0


def test_spearman_uses_average_ranks_for_ties_and_rejects_constant_scores() -> None:
    """Tie handling follows Spearman's average-rank definition without NaN fallback."""
    assert cross_sectional_spearman(
        np.array([1.0, 1.0, 3.0]), np.array([1.0, 2.0, 3.0])
    ) == pytest.approx(np.sqrt(3.0) / 2.0)
    with pytest.raises(ValueError, match="constant score"):
        cross_sectional_spearman(
            np.array([1.0, 1.0, 1.0]), np.array([1.0, 2.0, 3.0])
        )


def test_ranking_diagnostics_are_invariant_to_common_score_shift() -> None:
    """A common score offset cannot change ordering-based diagnostics."""
    scores = np.array([[-2.0, -1.0, 1.0, 2.0], [-1.0, -2.0, 2.0, 1.0]])
    targets = scores * 0.01

    original = compute_score_diagnostics(scores, targets, top_k=1)
    shifted = compute_score_diagnostics(scores + 100.0, targets, top_k=1)

    assert shifted == original


def test_canonical_reversal_score_is_negative_current_mom24() -> None:
    """Low momentum receives the highest predictive score in stable pair order."""
    dataset = _dataset(
        np.array([[[-0.2, 1.0], [0.1, 2.0], [0.4, 3.0]]]),
        np.array([[0.02, 0.0, -0.02]]),
    )
    dataset = SupervisedDataset(
        features=dataset.features,
        targets=dataset.targets,
        decision_timestamps=dataset.decision_timestamps,
        target_timestamps=dataset.target_timestamps,
        symbols=dataset.symbols,
        feature_names=("mom24_lag_0", "other_lag_0"),
    )

    np.testing.assert_allclose(canonical_reversal_scores(dataset), [[0.2, -0.1, -0.4]])


@pytest.mark.parametrize(
    ("inputs", "expected"),
    [
        (
            ClassificationInputs(
                aggregate_mean_rank_ic=0.03,
                aggregate_mean_tail_spread=0.002,
                iid_tail_spread_low=0.0002,
                moving_block_tail_spread_low=0.0001,
                era_tail_spreads=(0.001, 0.003),
                leave_one_fold_out_tail_spreads=(0.001, 0.002),
                coherent_reversal=True,
                non_degenerate_scores=True,
            ),
            "established learnable",
        ),
        (
            ClassificationInputs(
                aggregate_mean_rank_ic=0.03,
                aggregate_mean_tail_spread=0.002,
                iid_tail_spread_low=-0.0002,
                moving_block_tail_spread_low=0.0001,
                era_tail_spreads=(0.001, 0.003),
                leave_one_fold_out_tail_spreads=(0.001, 0.002),
                coherent_reversal=True,
                non_degenerate_scores=True,
            ),
            "suggestive",
        ),
        (
            ClassificationInputs(
                aggregate_mean_rank_ic=-0.01,
                aggregate_mean_tail_spread=-0.002,
                iid_tail_spread_low=-0.003,
                moving_block_tail_spread_low=-0.004,
                era_tail_spreads=(-0.001, 0.001),
                leave_one_fold_out_tail_spreads=(-0.002, 0.001),
                coherent_reversal=False,
                non_degenerate_scores=True,
            ),
            "not established",
        ),
    ],
)
def test_classification_follows_preregistered_acceptance_criteria(
    inputs: ClassificationInputs, expected: str
) -> None:
    """The strongest label is unavailable when any mandatory condition fails."""
    assert classify_learnability(inputs) == expected
