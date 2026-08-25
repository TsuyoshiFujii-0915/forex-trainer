"""Paired scaling-difference uncertainty tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from forex_trainer.data_scaling import ScalingStudy, _paired_differences


def test_paired_differences_resample_market_folds_not_seed_rows() -> None:
    """Seed variation within a fold must not become extra market histories."""
    study = ScalingStudy(
        name="paired",
        fold_configs=(Path("fold-a"), Path("fold-b")),
        audit_fold_configs=(Path("fold-a"), Path("fold-b")),
        seeds=(1, 2),
        history_years=(2,),
        device="cpu",
        workers=1,
        bootstrap_samples=1_000,
        bootstrap_seed=17,
        source_path=Path("study.yaml"),
    )
    rows = []
    for fold in ("fold-a", "fold-b"):
        rows.extend(
            [
                {
                    "fold": fold,
                    "condition": "2y",
                    "result_kind": "seed",
                    "seed": 1,
                    "cumulative_log_return": 0.0,
                },
                {
                    "fold": fold,
                    "condition": "2y",
                    "result_kind": "seed",
                    "seed": 2,
                    "cumulative_log_return": 0.0,
                },
                {
                    "fold": fold,
                    "condition": "expanding",
                    "result_kind": "seed",
                    "seed": 1,
                    "cumulative_log_return": -1.0,
                },
                {
                    "fold": fold,
                    "condition": "expanding",
                    "result_kind": "seed",
                    "seed": 2,
                    "cumulative_log_return": 1.0,
                },
            ]
        )

    paired = _paired_differences(rows, study)[0]

    assert paired["folds"] == 2
    assert paired["seed_fold_pairs"] == 4
    assert paired["fold_bootstrap_95_low"] == pytest.approx(0.0)
    assert paired["fold_bootstrap_95_high"] == pytest.approx(0.0)
    assert paired["moving_block_95_low"] == pytest.approx(0.0)
    assert paired["moving_block_95_high"] == pytest.approx(0.0)
