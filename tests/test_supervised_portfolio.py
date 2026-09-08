"""Behavioral tests for frozen-score portfolio translation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from forex_trainer.artifact_provenance import sha256_file
from forex_trainer.config import parse_experiment_config, resolve_env_raw
from forex_trainer.env_factory import GateEvaluationMode, build_single_env
from forex_trainer.supervised_ranking_study import build_dataset_from_env_raw
from forex_trainer.supervised_portfolio import (
    classify_portfolio,
    fixed_portfolio_weights,
    load_frozen_source,
    paired_evidence,
    prediction_matrices,
    replay_weights,
)
from helpers import make_experiment_raw


def test_fixed_mapping_preserves_score_direction_and_geometry() -> None:
    scores = np.array([[4, 0, 8, 1, 6, 2, 5, 7, 3]], dtype=float)
    weights = fixed_portfolio_weights(scores)
    np.testing.assert_allclose(weights, [[0, -.8, .8, -.8, 0, 0, 0, .8, 0]])
    assert np.abs(weights).sum() == pytest.approx(3.2)
    np.testing.assert_array_equal(weights, fixed_portfolio_weights(scores * 100 + 3))


def test_fixed_mapping_has_deterministic_pair_order_ties() -> None:
    np.testing.assert_allclose(
        fixed_portfolio_weights(np.zeros((1, 9))),
        [[-.8, -.8, 0, 0, 0, 0, 0, .8, .8]],
    )


@pytest.mark.parametrize("scores", [np.zeros((2, 8)), np.zeros(9), np.full((1, 9), np.nan), np.empty((0, 9))])
def test_invalid_scores_fail_before_trading(scores: np.ndarray) -> None:
    with pytest.raises(ValueError, match="scores"):
        fixed_portfolio_weights(scores)


def _source(tmp_path: Path, classification: str) -> tuple[Path, str]:
    names = ("predictions.csv", "models.json", "report.json", "study_snapshot.yaml",
             "fold_metrics.csv", "coefficients.csv", "coefficient_summary.csv", "report.md")
    for name in names:
        (tmp_path / name).write_text("{}", encoding="utf-8")
    (tmp_path / "report.json").write_text(json.dumps({"classification": classification}))
    provenance = {"artifact_version": 1, "generated_artifact_sha256": {
        name: sha256_file(tmp_path / name) for name in names}}
    path = tmp_path / "provenance.json"
    path.write_text(json.dumps(provenance), encoding="utf-8")
    return tmp_path, sha256_file(path)


def test_source_seal_detects_prediction_changes(tmp_path: Path) -> None:
    source, digest = _source(tmp_path, "established learnable")
    load_frozen_source(source, digest)
    (source / "predictions.csv").write_text("altered", encoding="utf-8")
    with pytest.raises(ValueError, match="predictions.csv"):
        load_frozen_source(source, digest)


def test_source_manifest_cannot_be_resealed_silently(tmp_path: Path) -> None:
    source, digest = _source(tmp_path, "established learnable")
    (source / "provenance.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="provenance"):
        load_frozen_source(source, digest)


@pytest.mark.parametrize("classification", ["not established", "suggestive"])
def test_unapproved_predictive_classification_blocks_portfolio(tmp_path: Path, classification: str) -> None:
    source, digest = _source(tmp_path, classification)
    with pytest.raises(ValueError, match="established learnable"):
        load_frozen_source(source, digest)


def _market() -> tuple[Any, Any, dict[str, Any]]:
    raw = make_experiment_raw()
    pairs = ["JPY/USD", "JPY/EUR", "JPY/GBP", "JPY/AUD", "JPY/CHF", "JPY/CAD", "JPY/NZD", "JPY/NOK", "JPY/SEK"]
    raw["env"]["environment"]["currency_pairs"] = pairs
    raw["env"]["features"]["normalize"] = False
    raw["env"]["features"]["selected"] = ["log_return", "volatility", "sma20_ratio", "mom24"]
    raw["env"]["transaction_costs"]["spreads"] = {pair: .0001 for pair in pairs}
    config = parse_experiment_config(raw)
    env_raw = resolve_env_raw(config.env, config.eval_range, for_eval=True)
    dataset = build_dataset_from_env_raw(env_raw, config.custom_feature_names, config.custom_cross_feature_names)
    return config, dataset, env_raw


def _predictions(dataset: Any) -> pd.DataFrame:
    return pd.DataFrame([
        {"fold": "2020", "pair": pair, "decision_timestamp": decision.isoformat(),
         "target_timestamp": target.isoformat(), "target_relative_log_return": dataset.targets[t, p],
         "supervised_score": float(p), "reversal_score": -dataset.features[t, p, dataset.feature_names.index("mom24_lag_0")],
         "ppo_score": .1}
        for t, (decision, target) in enumerate(zip(dataset.decision_timestamps, dataset.target_timestamps))
        for p, pair in enumerate(dataset.symbols)
    ])


def test_predictions_require_exact_rows_pairs_targets_and_finite_scores() -> None:
    _, dataset, _ = _market()
    frame = _predictions(dataset)
    assert prediction_matrices(frame, dataset)["supervised"].shape == dataset.targets.shape
    altered_frames = [frame.iloc[1:], frame.iloc[::-1], pd.concat([frame, frame.iloc[:1]])]
    wrong_target = frame.copy()
    wrong_target.loc[0, "target_relative_log_return"] += .001
    altered_frames.append(wrong_target)
    bad_score = frame.copy()
    bad_score.loc[0, "supervised_score"] = np.inf
    altered_frames.append(bad_score)
    for altered in altered_frames:
        with pytest.raises(ValueError, match="prediction"):
            prediction_matrices(altered, dataset)


def test_real_environment_replay_preserves_weights_costs_and_gross_attribution() -> None:
    config, dataset, env_raw = _market()
    scores = np.tile(np.arange(9, dtype=float), (len(dataset.targets), 1))
    weights = fixed_portfolio_weights(scores)
    env = build_single_env(env_raw, config.custom_feature_names, config.custom_cross_feature_names,
                           seed=0, decision_interval=1, residual=None, rank_allocation=None,
                           apply_hold_gate=None, gate_evaluation_mode=GateEvaluationMode.LEARNED)
    try:
        metrics, steps, pairs = replay_weights(env, dataset, weights, scores, np.zeros_like(scores))
    finally:
        env.close()
    assert metrics["steps"] == len(dataset.targets)
    assert metrics["total_weight_turnover"] == pytest.approx(3.2)
    assert metrics["total_cost_ratio"] > 0
    assert metrics["annualized_gross_return"] > metrics["annualized_net_return"]
    assert len(pairs) == len(steps) * 9
    for index, step in enumerate(steps):
        contributions = pairs[index * 9:(index + 1) * 9]
        assert sum(row["gross_simple_contribution"] for row in contributions) == pytest.approx(np.expm1(step["gross_log_return"]), abs=1e-12)
        assert sum(abs(row["target_weight"]) for row in contributions) == pytest.approx(3.2)


def test_replay_rejects_truncated_decision_range() -> None:
    config, dataset, env_raw = _market()
    env_raw["environment"]["episode_max_steps"] = 2
    env = build_single_env(env_raw, config.custom_feature_names, config.custom_cross_feature_names,
                           seed=0, decision_interval=1, residual=None, rank_allocation=None,
                           apply_hold_gate=None, gate_evaluation_mode=GateEvaluationMode.LEARNED)
    scores = np.ones_like(dataset.targets)
    try:
        with pytest.raises(ValueError, match="decision|range"):
            replay_weights(env, dataset, fixed_portfolio_weights(scores), scores, np.zeros_like(scores))
    finally:
        env.close()


def test_paired_evidence_retains_fold_order_and_exposes_tail_dependency() -> None:
    evidence = paired_evidence(np.array([10.] + [-.1] * 16), np.zeros(17))
    assert evidence["mean_difference"] > 0
    assert evidence["minimum_leave_one_fold_out_mean"] < 0
    assert evidence["largest_absolute_fold"] == "2009"
    assert evidence["intervals"]["fold_low"] < 0
    with pytest.raises(ValueError, match="17|shape"):
        paired_evidence(np.ones(16), np.ones(17))


@pytest.mark.parametrize(("gross", "net", "rule", "leverage", "expected"), [
    ([.1] * 17, [.06] * 17, [.05] * 17, 3.2, "learned and tradable"),
    ([.1] * 17, [-.02] * 17, [.05] * 17, 3.2, "learned but cost-limited"),
    ([.1] * 17, [.02] * 17, [.08] * 17, 3.2, "learned but cost-limited"),
    ([1.] + [-.01] * 16, [.5] + [-.02] * 16, [.05] * 17, 3.2, "not successfully translated to portfolio alpha"),
    ([.1] * 17, [.06] * 17, [.05] * 17, 0., "not successfully translated to portfolio alpha"),
])
def test_classification_requires_stable_non_degenerate_economic_evidence(
    gross: list[float], net: list[float], rule: list[float], leverage: float, expected: str,
) -> None:
    result = classify_portfolio(np.array(gross), np.array(net), np.array(rule), np.full(17, leverage))
    assert result["classification"] == expected


def test_campaign_rejects_portfolio_search_and_source_substitution(tmp_path: Path) -> None:
    from forex_trainer.supervised_portfolio_study import load_campaign

    source, digest = _source(tmp_path, "established learnable")
    study = tmp_path / "study.yaml"
    study.write_text("name: explicit", encoding="utf-8")
    campaign = tmp_path / "campaign.json"
    raw = {"source_dir": str(source), "source_provenance_sha256": digest,
           "source_study": str(study), "source_study_sha256": sha256_file(study)}
    campaign.write_text(json.dumps(raw), encoding="utf-8")
    assert load_campaign(campaign)[0] == source
    raw["top_k"] = 3
    campaign.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="keys|top_k"):
        load_campaign(campaign)
    del raw["top_k"]
    campaign.write_text(json.dumps(raw), encoding="utf-8")
    study.write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="study"):
        load_campaign(campaign)


def test_campaign_failure_publishes_no_partial_results(tmp_path: Path) -> None:
    from forex_trainer.supervised_portfolio_study import run_portfolio_study

    campaign = tmp_path / "campaign.json"
    campaign.write_text("{}", encoding="utf-8")
    output = tmp_path / "output"
    with pytest.raises(ValueError):
        run_portfolio_study(campaign, output)
    assert not output.exists()
    output.mkdir()
    (output / "existing.txt").write_text("keep", encoding="utf-8")
    with pytest.raises(FileExistsError):
        run_portfolio_study(campaign, output)
    assert (output / "existing.txt").read_text(encoding="utf-8") == "keep"


def test_signed_carry_uses_environment_file_provider_and_reconciles_pnl(tmp_path: Path) -> None:
    from forex_env.data.file_provider import save_ohlcv_parquet
    from forex_env.data.synthetic import SyntheticDataProvider
    from forex_trainer.supervised_portfolio_study import load_decision_carry

    config, dataset, env_raw = _market()
    market = SyntheticDataProvider(seed=11).get_data(dataset.symbols, "2020-02-15", "2020-03-01", "1h")
    for index, pair in enumerate(dataset.symbols):
        market[(pair, "CarryAnnual")] = (index - 4) * .02
    cache = tmp_path / "market.parquet"
    save_ohlcv_parquet(market, "1h", "2020-02-15", "2020-03-01", cache)
    env_raw["data"]["provider"] = "file"
    env_raw["data"]["path"] = str(cache)
    env_raw["transaction_costs"]["carry_mode"] = "signed"
    dataset = build_dataset_from_env_raw(env_raw, config.custom_feature_names, config.custom_cross_feature_names)
    carry = load_decision_carry(env_raw, dataset)
    np.testing.assert_allclose(carry[0], np.arange(-4, 5) * .02)
    scores = np.tile(np.arange(9, dtype=float), (len(dataset.targets), 1))
    env = build_single_env(env_raw, config.custom_feature_names, config.custom_cross_feature_names,
                           seed=0, decision_interval=1, residual=None, rank_allocation=None,
                           apply_hold_gate=None, gate_evaluation_mode=GateEvaluationMode.LEARNED)
    try:
        _, steps, pairs = replay_weights(env, dataset, fixed_portfolio_weights(scores), scores, carry)
    finally:
        env.close()
    assert sum(step["financing_jpy"] for step in steps) < 0
    assert sum(pair["carry_simple_contribution"] for pair in pairs) < 0


def test_direct_ppo_replay_retains_environment_gross_cap() -> None:
    config, dataset, env_raw = _market()
    weights = np.ones_like(dataset.targets)
    env = build_single_env(env_raw, config.custom_feature_names, config.custom_cross_feature_names,
                           seed=0, decision_interval=1, residual=None, rank_allocation=None,
                           apply_hold_gate=None, gate_evaluation_mode=GateEvaluationMode.LEARNED)
    try:
        _, _, pairs = replay_weights(env, dataset, weights, weights, np.zeros_like(weights))
    finally:
        env.close()
    assert sum(row["target_weight"] for row in pairs[:9]) == pytest.approx(5.)


def test_concentration_reports_three_favorable_folds_without_hiding_losses() -> None:
    values = np.array([1., 2., 3., 4.] + [-.1] * 13)
    evidence = paired_evidence(values, np.zeros(17))
    assert evidence["positive_fold_count"] == 4
    assert evidence["largest_three_positive_contribution_fraction"] == pytest.approx(.9)
    assert evidence["mean_without_three_largest_folds"] < 0
    zero = paired_evidence(np.zeros(17), np.zeros(17))
    assert zero["largest_three_positive_contribution_fraction"] is None
