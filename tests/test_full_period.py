"""Behavioral contracts for history-separated evaluation (Issue #28)."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from forex_env.data.file_provider import save_ohlcv_parquet
from stable_baselines3 import PPO

from forex_trainer.artifact_provenance import dependency_versions, git_commits, sha256_file
from forex_trainer.config import RangeConfig, resolve_env_raw
from forex_trainer.full_period import (
    FEATURES, MEASUREMENT_ID, FrozenRidge, build_period_env, evaluate_policy,
    prepare_period, require_comparable, run_period_campaign,
)
from helpers import REPO_ROOT


def write_json(path: Path, value: Any) -> dict[str, str]:
    """Write fixture input and return its explicit identity."""
    path.write_text(json.dumps(value), encoding="utf-8")
    return {"path": str(path), "sha256": sha256_file(path)}


def market_calendar(start: str, periods: int) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Create explicit sessions spanning holidays, leap days, and DST."""
    times = pd.date_range(start, periods=periods, freq="B", tz="America/New_York") + pd.Timedelta(hours=17)
    labels = times.tz_localize(None).normalize()
    symbols = tuple(config_raw()["env"]["environment"]["currency_pairs"])
    columns: dict[tuple[str, str], np.ndarray] = {}
    for i, symbol in enumerate(symbols):
        close = 0.01 * np.exp((i - 4) * np.arange(periods) * 0.0001)
        for field in ("Open", "High", "Low", "Close"):
            columns[symbol, field] = close.copy()
        columns[symbol, "Volume"] = np.ones(periods)
        columns[symbol, "CarryAnnual"] = np.full(periods, i * 0.001)
    calendar = {
        "version": "fixture-v1", "timezone": "America/New_York",
        "symbols": list(symbols), "holidays": [],
        "label_rule": "Date label maps to assumed New York 17:00 close; delivery unknown.",
        "sessions": [{"bar_label": label.isoformat(),
                      "session_open": (time - pd.Timedelta(hours=23)).isoformat(),
                      "session_close": time.isoformat(), "available_at": None}
                     for label, time in zip(labels, times)],
    }
    return pd.DataFrame(columns, index=labels), calendar


def config_raw() -> dict[str, Any]:
    """Use the real current longf configuration geometry."""
    import yaml
    return yaml.safe_load((REPO_ROOT / "configs/wf_r16_long_full/wf2025_r16_longf.yaml").read_text())


def prepare_fixture(tmp_path: Path, count: int) -> tuple[Any, dict[str, Any], dict[str, Any]]:
    """Prepare a 63-bar prefix and a short deterministic evaluation."""
    market, calendar = market_calendar("2023-11-30", count)
    start = calendar["sessions"][63]["session_close"]
    end = (pd.Timestamp(calendar["sessions"][-1]["session_close"]) + pd.Timedelta(days=1)).isoformat()
    raw = config_raw()["env"]
    plan = prepare_period(market, calendar, start, end, tuple(calendar["symbols"]), tmp_path / "market.parquet")
    return plan, raw, calendar


def constant_policy(observation: dict[str, np.ndarray], window: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Hold a fixed one-pair allocation; no history trades."""
    weights = np.zeros((9, 1), dtype=np.float32)
    weights[0, 0] = 0.8
    return weights[:, 0].astype(float), weights


def test_63_history_reset_first_entry_and_final_mark(tmp_path: Path) -> None:
    plan, raw, calendar = prepare_fixture(tmp_path, 67)
    env, windows = build_period_env(raw, plan, tmp_path / "slice.parquet")
    observation, info = env.reset(seed=0)
    assert info["timestamp"] == pd.Timestamp(calendar["sessions"][63]["session_close"]).tz_convert("UTC").isoformat()
    assert info["equity_jpy"] == 1_000_000
    assert set(info["exposures_jpy"].values()) == {0.0}
    assert np.count_nonzero(observation["assets"]) == 0
    assert "costs_jpy" not in info
    np.testing.assert_array_equal(observation["market"], windows[0].astype(np.float32))
    result = evaluate_policy(env, windows, plan, constant_policy)
    assert len(result["trace"]) == 3
    first = result["trace"][0]
    assert first["equity_before"] == 1_000_000
    assert first["spread_jpy"] == pytest.approx(float(np.float32(0.8)) * 1_000_000 * 1.5e-5)
    assert result["metrics"]["steps"] == 3
    assert result["trace"][-1]["target_timestamp"] == plan.timestamps[-1]
    assert result["initial_assets"] == np.zeros((9, 3)).tolist()
    second = evaluate_policy(env, windows, plan, constant_policy)
    assert second == result
    env.close()


def test_62_history_fails_before_inference(tmp_path: Path) -> None:
    market, calendar = market_calendar("2023-11-30", 67)
    with pytest.raises(ValueError, match="63.*62"):
        prepare_period(market, calendar, calendar["sessions"][62]["session_close"], "2024-04-01T00:00:00Z", tuple(calendar["symbols"]), tmp_path / "raw.parquet")


@pytest.mark.parametrize("end_delta", [0, -1])
def test_target_at_or_after_end_excluded(tmp_path: Path, end_delta: int) -> None:
    market, calendar = market_calendar("2023-11-30", 68)
    end = pd.Timestamp(calendar["sessions"][66]["session_close"]) + pd.Timedelta(seconds=end_delta)
    plan = prepare_period(market, calendar, calendar["sessions"][63]["session_close"], end.isoformat(), tuple(calendar["symbols"]), tmp_path / "raw.parquet")
    assert len(plan.timestamps) == 3
    assert pd.Timestamp(plan.timestamps[-1]) < end


@pytest.mark.parametrize("bad", ["missing", "pair", "extra", "naive", "late"])
def test_calendar_errors_identify_origin(tmp_path: Path, bad: str) -> None:
    market, calendar = market_calendar("2023-11-30", 68)
    start = calendar["sessions"][63]["session_close"]
    if bad == "missing":
        market = market.drop(market.index[64])
    elif bad == "pair":
        calendar["symbols"] = list(reversed(calendar["symbols"]))
    elif bad == "extra":
        calendar["sessions"].pop(64)
    elif bad == "naive":
        start = "2024-02-27"
    else:
        calendar["sessions"][63]["available_at"] = "2025-01-01T00:00:00Z"
    with pytest.raises(ValueError, match="raw.parquet"):
        prepare_period(market, calendar, start, "2024-04-01T00:00:00Z", tuple(config_raw()["env"]["environment"]["currency_pairs"]), tmp_path / "raw.parquet")


def test_holiday_leap_day_dst_and_actual_annualization(tmp_path: Path) -> None:
    market, calendar = market_calendar("2023-11-20", 90)
    holiday = "2024-02-19T00:00:00"
    calendar["holidays"] = ["2024-02-19"]
    calendar["sessions"] = [row for row in calendar["sessions"] if row["bar_label"] != holiday]
    market = market.drop(pd.Timestamp(holiday))
    plan = prepare_period(market, calendar, "2024-02-19T00:00:00Z", "2024-03-23T00:00:00Z", tuple(calendar["symbols"]), tmp_path / "raw.parquet")
    assert pd.Timestamp(plan.timestamps[0]).day == 20
    assert any("2024-02-29" in time for time in plan.timestamps)
    assert (pd.Timestamp("2024-03-11T21:00:00Z") - pd.Timestamp("2024-03-08T22:00:00Z")).total_seconds() == 71 * 3600
    env, windows = build_period_env(config_raw()["env"], plan, tmp_path / "slice.parquet")
    result = evaluate_policy(env, windows, plan, constant_policy)
    elapsed = (pd.Timestamp(plan.timestamps[-1]) - pd.Timestamp(plan.timestamps[0])).total_seconds()
    assert result["coverage"]["elapsed_seconds"] == elapsed
    assert result["coverage"]["initial_flat_seconds"] > 0
    expected = np.expm1(result["metrics"]["cumulative_log_return"] / (elapsed / (365.25 * 86400)))
    assert result["metrics"]["annualized_net_return"] == pytest.approx(expected)
    env.close()


def ridge_record() -> dict[str, Any]:
    """Construct explicit frozen parameters, without fitting."""
    return {"feature_names": [f"{name}_lag_{lag}" for lag in range(31, -1, -1) for name in FEATURES],
            "train_mean": [0.0] * 256, "train_scale": [1.0] * 256,
            "coefficients": np.linspace(-0.01, 0.01, 256).tolist(),
            "intercept": 0.001, "selected_alpha": 10.0}


def test_future_change_preserves_past_observation_score_and_parameters(tmp_path: Path) -> None:
    market, calendar = market_calendar("2023-11-30", 70)
    raw = config_raw()["env"]
    record = ridge_record()
    before = json.dumps(record)
    ridge = FrozenRidge(record, tmp_path / "models.json")
    observations: list[np.ndarray] = []
    scores: list[np.ndarray] = []
    for variant in range(2):
        plan = prepare_period(market, calendar, calendar["sessions"][63]["session_close"], "2024-04-01T00:00:00Z", tuple(calendar["symbols"]), tmp_path / "raw.parquet")
        env, windows = build_period_env(raw, plan, tmp_path / f"slice{variant}.parquet")
        observation, _ = env.reset(seed=0)
        observations.append(observation["market"])
        scores.append(ridge.predict(windows[0]))
        env.close()
        market.iloc[65:, :] *= 1.01
    np.testing.assert_array_equal(observations[0], observations[1])
    np.testing.assert_array_equal(scores[0], scores[1])
    assert json.dumps(record) == before
    assert ridge.parameter_sha256 == FrozenRidge(record, tmp_path / "models.json").parameter_sha256


def campaign_fixture(tmp_path: Path) -> Path:
    """Seal real untrained PPO fixture files and an explicit calendar."""
    import yaml
    market, calendar = market_calendar("2023-10-02", 350)
    cache = tmp_path / "raw.parquet"
    save_ohlcv_parquet(market, "1d", "2003-01-01", "2026-01-01", cache)
    raw = config_raw()
    raw["env"]["data"]["path"] = str(cache)
    raw["train_range"] = {"start": "2023-10-02", "end": "2023-12-01"}
    raw["val_range"] = {"start": "2023-12-01", "end": "2024-01-01"}
    raw["eval_range"] = {"start": "2024-01-01", "end": "2025-01-01"}
    data_identity = {"provider": "file", "path": str(cache), "sha256": sha256_file(cache)}
    calendar_ref = write_json(tmp_path / "calendar.json", calendar)
    plan = prepare_period(market, calendar, "2024-01-01T00:00:00Z", "2025-01-01T00:00:00Z", tuple(calendar["symbols"]), cache)
    env, _ = build_period_env(raw["env"], plan, tmp_path / "slice.parquet")
    members: list[dict[str, Any]] = []
    eval_raw = resolve_env_raw(raw["env"], RangeConfig("2024-01-01", "2025-01-01"), for_eval=True)
    for seed in (42, 43, 44):
        member = tmp_path / str(seed)
        member.mkdir()
        member_raw = copy.deepcopy(raw)
        member_raw["run"]["seed"] = seed
        (member / "config_snapshot.yaml").write_text(yaml.safe_dump(member_raw))
        (member / "env_eval.yaml").write_text(yaml.safe_dump(eval_raw))
        write_json(member / "meta.json", {"experiment": raw["experiment"], "seed": seed,
                   "requested_device": "cpu", "device": "cpu", "algorithm": "ppo", "network": "mlp",
                   "git": git_commits(), "versions": dependency_versions(), "data_identity": data_identity})
        model = PPO("MultiInputPolicy", env, seed=seed, device="cpu", n_steps=2, batch_size=2, policy_kwargs={"net_arch": [8]})
        model.save(member / "model_final.zip")
        members.append({"run_dir": str(member), "seed": seed, "experiment": raw["experiment"], "model_path": "model_final.zip",
                        "model_sha256": sha256_file(member / "model_final.zip"),
                        "config_snapshot_sha256": sha256_file(member / "config_snapshot.yaml"),
                        "meta_sha256": sha256_file(member / "meta.json")})
    env.close()
    ensemble = tmp_path / "ensemble"
    ensemble.mkdir()
    (ensemble / "env_eval.yaml").write_text(yaml.safe_dump(eval_raw))
    write_json(ensemble / "metrics.json", {"fixture": True})
    evaluation = {"resolved_device": "cpu", "git": git_commits(), "versions": dependency_versions(),
                  "data_identity": data_identity, "metrics_sha256": sha256_file(ensemble / "metrics.json"),
                  "env_eval_sha256": sha256_file(ensemble / "env_eval.yaml")}
    write_json(ensemble / "ensemble.json", {"manifest_version": 2, "policy": "action_mean", "model_selection": "validation_best", "decision_interval": 1, "members": members, "evaluation": evaluation})
    config = tmp_path / "config.yaml"
    config.write_text(yaml.safe_dump(raw))
    models_ref = write_json(tmp_path / "models.json", {"2024": ridge_record()})
    source_ref = write_json(tmp_path / "provenance.json", {
        "artifact_version": 1, "generated_artifact_sha256": {"models.json": models_ref["sha256"]},
        "fold_sources": {"2024": {"config_path": str(config), "config_sha256": sha256_file(config), "data_identity": data_identity,
            "ppo": {"ensemble_dir": str(ensemble), "ensemble_manifest_sha256": sha256_file(ensemble / "ensemble.json"),
                    "member_model_sha256": [m["model_sha256"] for m in members], "sealed_device": "cpu",
                    "data_identity": data_identity, "evaluation_git": evaluation["git"], "evaluation_versions": evaluation["versions"],
                    "metrics_sha256": evaluation["metrics_sha256"], "eval_env_sha256": evaluation["env_eval_sha256"]}}}})
    lineage_ref = write_json(tmp_path / "lineage.json", {"raw": [{"path": str(cache), "sha256": sha256_file(cache)}],
        "clean": [{"path": str(cache), "sha256": sha256_file(cache)}], "carry": [{"path": str(cache), "sha256": sha256_file(cache)}],
        "transformations": "Deterministic synthetic fixture; no market evidence.",
        "rate_vintage": "synthetic", "available_at_evidence": "unknown; historical assumptions only"})
    path = tmp_path / "campaign.json"
    write_json(path, {"mode": "full_period", "measurement_id": MEASUREMENT_ID,
        "evidence_class": "development_historical", "source": source_ref, "lineage": lineage_ref,
        "runtime": {"forex_env_sha": git_commits()["forex_env"], "versions": dependency_versions(), "device": "cpu"},
        "folds": [{"fold": "2024", "measurement_start": "2024-01-01T00:00:00Z", "measurement_end": "2025-01-01T00:00:00Z", "calendar": calendar_ref}]})
    return path


@pytest.fixture(scope="module")
def sealed_fixture(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Build deterministic serialized fixture models once."""
    return campaign_fixture(tmp_path_factory.mktemp("full-period"))


def test_three_frozen_policies_full_year_end_to_end(sealed_fixture: Path, tmp_path: Path) -> None:
    before = {path: path.read_bytes() for path in sealed_fixture.parent.rglob("*") if path.is_file()}
    result = run_period_campaign(sealed_fixture, tmp_path / "result")
    assert result["measurement_id"] == MEASUREMENT_ID
    assert result["status"] == "complete"
    assert set(result["folds"]["2024"]) == {"ridge", "canonical", "ppo_ens3"}
    results = list(result["folds"]["2024"].values())
    for row in results:
        assert row["metrics"]["steps"] == 261
        assert row["coverage"]["first_decision"] == "2024-01-01T22:00:00+00:00"
        assert row["initial_assets"] == np.zeros((9, 3)).tolist()
    require_comparable(results)
    assert (tmp_path / "result" / "manifest.json").is_file()
    assert (tmp_path / "result" / "steps.csv").is_file()
    assert (tmp_path / "result" / "report.md").is_file()
    assert all(path.read_bytes() == content for path, content in before.items())
    with pytest.raises(ValueError, match="exist"):
        run_period_campaign(sealed_fixture, tmp_path / "result")
    mixed = copy.deepcopy(results)
    mixed[0]["measurement_id"] = "legacy"
    with pytest.raises(ValueError, match="measurement"):
        require_comparable(mixed)
    for field in ("runtime", "symbols", "timestamps", "costs"):
        mixed = copy.deepcopy(results)
        mixed[0][field] = "different"
        with pytest.raises(ValueError, match=field):
            require_comparable(mixed)


@pytest.mark.parametrize("failure", ["hash", "missing", "runtime", "mode"])
def test_preflight_rejects_bad_source_before_output(sealed_fixture: Path, tmp_path: Path, failure: str) -> None:
    campaign = json.loads(sealed_fixture.read_text())
    if failure == "hash":
        campaign["source"]["sha256"] = "0" * 64
    elif failure == "missing":
        campaign["source"]["path"] = str(tmp_path / "missing-provenance.json")
    elif failure == "runtime":
        campaign["runtime"]["versions"]["numpy"] = "0.0"
    else:
        campaign["measurement_id"] = "legacy"
    path = tmp_path / "bad.json"
    write_json(path, campaign)
    with pytest.raises(ValueError, match="hash|missing|runtime|measurement"):
        run_period_campaign(path, tmp_path / "result")
    assert not (tmp_path / "result").exists()


def test_legacy_mode_keeps_existing_metric_bytes(sealed_fixture: Path, tmp_path: Path) -> None:
    from forex_trainer.evaluate import _run_evaluation
    from forex_trainer.env_factory import GateEvaluationMode
    source = sealed_fixture.parent / "42"
    reference = tmp_path / "reference"
    reference.mkdir()
    _run_evaluation(source, reference, GateEvaluationMode.LEARNED)
    before = {path: path.read_bytes() for path in source.iterdir() if path.is_file()}
    campaign = tmp_path / "legacy.json"
    write_json(campaign, {"mode": "legacy", "run_dir": str(source)})
    run_period_campaign(campaign, tmp_path / "legacy")
    for name in ("metrics.json", "equity_curve.csv", "evaluation.json"):
        assert (reference / name).read_bytes() == (tmp_path / "legacy" / name).read_bytes()
    assert all(path.read_bytes() == content for path, content in before.items())


def test_margin_call_preserves_terminal_and_blocks_comparison(tmp_path: Path) -> None:
    plan, raw, _ = prepare_fixture(tmp_path, 70)
    raw["environment"]["margin_call_threshold"] = 0.99999
    env, windows = build_period_env(raw, plan, tmp_path / "slice.parquet")
    result = evaluate_policy(env, windows, plan, constant_policy)
    assert result["status"] == "incomplete_margin_call"
    assert result["metrics"]["terminated_by_margin_call"] is True
    assert len(result["trace"]) == 1
    assert result["coverage"]["last_mark"] < result["coverage"]["planned_last_mark"]
    with pytest.raises(ValueError, match="incomplete"):
        require_comparable([result, result])
    env.close()


def test_ridge_rejects_invalid_standardizer(tmp_path: Path) -> None:
    record = ridge_record()
    record["train_scale"][4] = 0.0
    with pytest.raises(ValueError, match="scale.*models.json"):
        FrozenRidge(record, tmp_path / "models.json")
