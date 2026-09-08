"""Study configuration, environment alignment, and CLI behavior tests."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
import pytest
import yaml
from forex_env import ForexEnv, parse_config

from forex_trainer.config import (
    TrainerConfigError,
    parse_experiment_config,
    resolve_env_raw,
)
from forex_trainer.features import CROSS_FEATURE_REGISTRY, FEATURE_REGISTRY
from forex_trainer.supervised_ranking_study import (
    PpoScoreTrace,
    build_dataset_from_env_raw,
    load_supervised_study,
    main,
    require_score_alignment,
)
from helpers import make_experiment_raw


def _write_valid_study(tmp_path: Path) -> Path:
    """Write a minimal path-valid fixed-protocol 17-fold manifest."""
    folds: dict[str, dict[str, str]] = {}
    for year in range(2009, 2026):
        config = tmp_path / f"fold-{year}.yaml"
        config.write_text("experiment: placeholder\n", encoding="utf-8")
        ensemble = tmp_path / f"ensemble-{year}"
        ensemble.mkdir()
        folds[str(year)] = {
            "config": config.name,
            "ppo_ensemble": ensemble.name,
        }
    study = tmp_path / "study.yaml"
    study.write_text(
        yaml.safe_dump(
            {
                "name": "issue15-test",
                "folds": folds,
                "alpha_grid": [0.0, 0.1, 1.0, 10.0],
                "top_k": 2,
                "member_seeds": [42, 43, 44],
                "eras": {
                    "2009-2018": {"start": 2009, "end": 2018},
                    "2019-2025": {"start": 2019, "end": 2025},
                },
                "bootstrap_samples": 10_000,
                "bootstrap_seed": 15,
                "moving_block_length": 3,
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return study


def test_study_manifest_requires_fixed_grid_controls_and_exact_17_folds(
    tmp_path: Path,
) -> None:
    """The preregistered protocol cannot be changed through study YAML."""
    path = _write_valid_study(tmp_path)

    study = load_supervised_study(path)

    assert tuple(study.folds) == tuple(str(year) for year in range(2009, 2026))
    assert study.alpha_grid == (0.0, 0.1, 1.0, 10.0)
    assert study.top_k == 2
    assert study.member_seeds == (42, 43, 44)
    assert study.bootstrap_samples == 10_000
    assert study.moving_block_length == 3

    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    del raw["folds"]["2014"]
    raw["implicit_fallback"] = True
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(
        TrainerConfigError,
        match=r"unknown=.*implicit_fallback.*folds.*2009.*2025|folds.*2009.*2025",
    ):
        load_supervised_study(path)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("alpha_grid", [0.0, 1.0]),
        ("top_k", 1),
        ("member_seeds", [42, 43]),
        ("bootstrap_samples", 9999),
        ("moving_block_length", 2),
    ],
)
def test_study_manifest_rejects_protocol_search_controls(
    tmp_path: Path, field: str, value: object
) -> None:
    """Fixed research controls are validated rather than treated as switches."""
    path = _write_valid_study(tmp_path)
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    raw[field] = value
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")

    with pytest.raises(TrainerConfigError, match=field):
        load_supervised_study(path)


def test_dataset_row_exactly_matches_longf_market_observation() -> None:
    """Dataset rows retain the full environment market window in stable order."""
    raw = make_experiment_raw()
    pairs = ["JPY/USD", "JPY/EUR", "JPY/GBP"]
    raw["env"]["environment"]["currency_pairs"] = pairs
    raw["env"]["environment"]["window_size"] = 8
    raw["env"]["features"]["volatility_window"] = 8
    raw["env"]["features"]["normalize"] = False
    raw["env"]["features"]["selected"] = [
        "log_return",
        "volatility",
        "sma20_ratio",
        "mom24",
        "xz_mom24",
        "xr_mom24",
    ]
    raw["env"]["transaction_costs"]["spreads"] = {
        pair: 0.0001 for pair in pairs
    }
    config = parse_experiment_config(raw)
    env_raw = resolve_env_raw(config.env, config.train_range, for_eval=False)

    dataset = build_dataset_from_env_raw(
        env_raw,
        config.custom_feature_names,
        config.custom_cross_feature_names,
    )
    env = ForexEnv(
        parse_config(env_raw),
        custom_features={
            name: FEATURE_REGISTRY[name] for name in config.custom_feature_names
        },
        custom_cross_features={
            name: CROSS_FEATURE_REGISTRY[name]
            for name in config.custom_cross_feature_names
        },
    )
    observation, info = env.reset(seed=0)
    env.close()

    assert dataset.decision_timestamps[0] == datetime.fromisoformat(info["timestamp"])
    np.testing.assert_allclose(
        dataset.features[0], observation["market"].reshape(len(pairs), -1)
    )
    assert dataset.symbols == tuple(pairs)
    assert dataset.feature_names[0] == "log_return_lag_7"
    assert dataset.feature_names[-1] == "xr_mom24_lag_0"


def test_score_alignment_rejects_timestamp_or_pair_intersection_fallback() -> None:
    """All benchmarks must cover the exact same ordered decisions and pairs."""
    raw = make_experiment_raw()
    raw["env"]["environment"]["currency_pairs"] = ["JPY/USD", "JPY/EUR"]
    raw["env"]["transaction_costs"]["spreads"] = {
        "JPY/USD": 0.0001,
        "JPY/EUR": 0.0001,
    }
    config = parse_experiment_config(raw)
    env_raw = resolve_env_raw(config.env, config.eval_range, for_eval=True)
    dataset = build_dataset_from_env_raw(
        env_raw,
        config.custom_feature_names,
        config.custom_cross_feature_names,
    )
    aligned = PpoScoreTrace(
        scores=np.ones(dataset.targets.shape),
        decision_timestamps=dataset.decision_timestamps,
        symbols=dataset.symbols,
    )
    require_score_alignment(dataset, aligned)

    missing = PpoScoreTrace(
        scores=np.ones((len(dataset.decision_timestamps) - 1, len(dataset.symbols))),
        decision_timestamps=dataset.decision_timestamps[:-1],
        symbols=dataset.symbols,
    )
    with pytest.raises(ValueError, match="decision timestamps"):
        require_score_alignment(dataset, missing)
    reordered = PpoScoreTrace(
        scores=np.ones(dataset.targets.shape),
        decision_timestamps=dataset.decision_timestamps,
        symbols=tuple(reversed(dataset.symbols)),
    )
    with pytest.raises(ValueError, match="pair order"):
        require_score_alignment(dataset, reordered)


def test_cli_validation_failure_leaves_no_partial_output(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A malformed study fails before transactional output publication."""
    path = _write_valid_study(tmp_path)
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    raw["member_seeds"] = [42, 42, 44]
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    output = tmp_path / "result"

    assert main(["--study", str(path), "--output-dir", str(output)]) == 1
    assert "member_seeds" in capsys.readouterr().err
    assert not output.exists()
