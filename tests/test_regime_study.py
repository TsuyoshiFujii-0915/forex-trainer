"""Artifact I/O and CLI behavior tests for the regime study."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from forex_trainer.config import TrainerConfigError
from forex_trainer.regime_study import (
    RuleSpec,
    _sealed_evaluation_device,
    build_momentum_reversal_action,
    compare_legacy_current,
    load_regime_study,
    main,
)


def _write_valid_study(tmp_path: Path) -> Path:
    """Write the smallest strict study manifest and its referenced files.

    Args:
        tmp_path: Test-scoped directory.

    Returns:
        Study YAML path.
    """
    campaign_path = tmp_path / "campaign.yaml"
    campaign_path.write_text("name: baseline\n", encoding="utf-8")
    legacy_path = tmp_path / "legacy.json"
    legacy_path.write_text(
        json.dumps(
            {
                "schemes": {
                    "validation_best": {
                        "overall": {
                            "mean_annualized_net_return": 0.1,
                            "mean_annualized_gross_return": 0.2,
                            "mean_max_drawdown": 0.3,
                            "worst_max_drawdown": 0.4,
                        },
                        "eras": {},
                        "folds": {},
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    study_path = tmp_path / "study.yaml"
    study_path.write_text(
        yaml.safe_dump(
            {
                "name": "regime-contract",
                "baseline_campaign": campaign_path.name,
                "baseline_configuration": "direct_longf_ens3",
                "legacy_report": legacy_path.name,
                "legacy_scheme": "validation_best",
                "member_seeds": [42, 43, 44],
                "rule": {"feature": "mom24", "top_k": 1, "base_size": 0.5},
                "bucket_count": 3,
                "forward_drawdown_steps": 5,
                "minimum_fold_direction_rate": 0.75,
                "minimum_folds_per_era": 2,
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return study_path


def test_study_yaml_is_strict_and_resolves_referenced_artifacts(
    tmp_path: Path,
) -> None:
    """Every declared field is mandatory and unknown fields are rejected."""
    study_path = _write_valid_study(tmp_path)

    study = load_regime_study(study_path)

    assert study.name == "regime-contract"
    assert study.baseline_campaign == (tmp_path / "campaign.yaml").resolve()
    assert study.legacy_report == (tmp_path / "legacy.json").resolve()
    assert study.member_seeds == (42, 43, 44)
    assert study.rule == RuleSpec(feature="mom24", top_k=1, base_size=0.5)

    raw = yaml.safe_load(study_path.read_text(encoding="utf-8"))
    del raw["bucket_count"]
    raw["undeclared_fallback"] = True
    study_path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(TrainerConfigError, match=r"missing=.*bucket_count.*unknown=.*undeclared_fallback"):
        load_regime_study(study_path)


def test_study_rejects_legacy_report_without_requested_study_scheme(
    tmp_path: Path,
) -> None:
    """A generic or malformed report cannot silently replace Issue 1 evidence."""
    study_path = _write_valid_study(tmp_path)
    (tmp_path / "legacy.json").write_text(
        json.dumps({"campaign": "current-generic-report", "configurations": {}}),
        encoding="utf-8",
    )

    with pytest.raises(TrainerConfigError, match="legacy.*validation_best"):
        load_regime_study(study_path)


def test_rule_is_direct_weight_cross_sectional_momentum_reversal() -> None:
    """Low momentum is long and high momentum is short at the declared size."""
    market = np.zeros((4, 3, 3), dtype=np.float64)
    market[:, -1, 1] = np.array([0.4, -0.2, 0.1, 0.8])
    observation = {"market": market}

    action = build_momentum_reversal_action(
        observation,
        ("log_return", "mom24", "carry_annual"),
        RuleSpec(feature="mom24", top_k=1, base_size=0.6),
    )

    np.testing.assert_allclose(action[:, 0], [0.0, 0.6, 0.0, -0.6])
    assert action.dtype == np.float32


def test_legacy_comparison_is_descriptive_and_fold_explicit() -> None:
    """Sanity comparison reports levels and deltas without paired inference."""
    legacy = {
        "overall": {
            "mean_annualized_net_return": 0.10,
            "mean_annualized_gross_return": 0.14,
            "mean_max_drawdown": 0.20,
            "worst_max_drawdown": 0.35,
        },
        "eras": {
            "early": {
                "mean_annualized_net_return": 0.12,
                "mean_annualized_gross_return": 0.16,
                "mean_max_drawdown": 0.22,
                "worst_max_drawdown": 0.35,
            }
        },
        "folds": {
            "2015": {
                "annualized_net_return": -0.30,
                "annualized_gross_return": -0.25,
                "max_drawdown": 0.35,
            }
        },
    }
    current = {
        "overall": {
            "annualized_net_return": 0.08,
            "annualized_gross_return": 0.12,
            "mean_max_drawdown": 0.18,
            "worst_max_drawdown": 0.30,
        },
        "eras": {
            "early": {
                "annualized_net_return": 0.09,
                "annualized_gross_return": 0.13,
                "mean_max_drawdown": 0.19,
                "worst_max_drawdown": 0.30,
            }
        },
        "folds": {
            "2015": {
                "annualized_net_return": -0.20,
                "annualized_gross_return": -0.17,
                "mean_max_drawdown": 0.30,
            }
        },
    }

    rows = compare_legacy_current(legacy, current)

    net = next(
        row
        for row in rows
        if row["scope"] == "overall" and row["metric"] == "annualized_net_return"
    )
    assert net["legacy"] == pytest.approx(0.10)
    assert net["current"] == pytest.approx(0.08)
    assert net["current_minus_legacy"] == pytest.approx(-0.02)
    assert {row["scope"] for row in rows} == {"overall", "era:early", "fold:2015"}


def test_cli_validation_failure_leaves_no_partial_output(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Validation errors are explicit and output publication is transactional."""
    study_path = _write_valid_study(tmp_path)
    raw = yaml.safe_load(study_path.read_text(encoding="utf-8"))
    raw["member_seeds"] = [42, 42]
    study_path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    output_dir = tmp_path / "output"

    assert main(["--study", str(study_path), "--output-dir", str(output_dir)]) == 1
    assert "duplicate" in capsys.readouterr().err
    assert not output_dir.exists()


def test_replay_device_comes_from_the_sealed_ensemble_manifest(
    tmp_path: Path,
) -> None:
    """Replay rejects unresolved or absent devices instead of resolving run.device again."""
    manifest_path = tmp_path / "ensemble.json"

    assert (
        _sealed_evaluation_device(
            {"evaluation": {"resolved_device": "cpu"}}, manifest_path
        )
        == "cpu"
    )
    with pytest.raises(TrainerConfigError, match="sealed resolved_device"):
        _sealed_evaluation_device(
            {"evaluation": {"resolved_device": "auto"}}, manifest_path
        )
    with pytest.raises(TrainerConfigError, match="sealed resolved_device"):
        _sealed_evaluation_device({"evaluation": {}}, manifest_path)
