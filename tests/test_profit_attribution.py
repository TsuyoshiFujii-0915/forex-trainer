"""Behavioral contracts for the preregistered profit-source diagnostic."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from forex_env.data.file_provider import save_ohlcv_parquet

from forex_trainer.cost_campaign import account_trace, fold_metrics
from forex_trainer.full_period import build_period_env, evaluate_policy
from forex_trainer.profit_attribution import (
    COMPONENTS, POLICIES, ConstantAllocation, CommonProjection, attribute_account,
    summarize_attribution, train_allocation, validate_config, verify_reproduction,
)
from test_full_period import config_raw, constant_policy, market_calendar, prepare_fixture


def feedback_policy(observation: dict[str, np.ndarray], window: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Provide a deterministic state-dependent policy for real account walks.

    Args:
        observation: Current account observation.
        window: Causal market window.

    Returns:
        Scores and bounded direct allocation proposal.
    """
    weights = np.linspace(-.8, .9, 9) + observation["assets"][:, 0] * .1
    action = np.clip(weights, -1, 1).astype(np.float32).reshape(9, 1)
    return action[:, 0].astype(float), action


def evaluated(tmp_path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Evaluate a real capped account with both signed financing and fees.

    Args:
        tmp_path: Private fixture directory.

    Returns:
        Reconciled step and pair account traces.
    """
    plan, raw, _ = prepare_fixture(tmp_path, 70)
    raw["environment"]["max_leverage"] = .4
    raw["transaction_costs"]["commission_rate"] = .001
    env, windows = build_period_env(raw, plan, tmp_path / "slice.parquet")
    try:
        result = evaluate_policy(env, windows, plan, constant_policy)
    finally:
        env.close()
    result.update({"fold": "2024", "policy": "ppo_ens3", "scenario": "F0"})
    return result, account_trace(result, plan)


def test_effective_exposure_and_common_relative_log_accounting(tmp_path: Path) -> None:
    result, pairs = evaluated(tmp_path)
    steps, fold = attribute_account(result, pairs)
    assert result["trace"][0]["action"][0] > result["trace"][0]["target_weights"][0]
    for step, original in zip(steps, result["trace"]):
        selected = [r for r in pairs if r["decision_timestamp"] == step["decision_timestamp"]]
        returns = np.array([r["price_relative"] - 1 for r in selected])
        weights = np.array(original["target_weights"])
        assert step["common_simple"] == pytest.approx(weights.sum() * returns.mean())
        assert step["relative_simple"] == pytest.approx(weights @ (returns - returns.mean()))
        assert step["price_simple"] == pytest.approx(weights @ returns)
        assert step["net_simple"] == pytest.approx(sum(step[c + "_simple"] for c in COMPONENTS))
        assert step["net_log"] == pytest.approx(sum(step[c + "_log"] for c in COMPONENTS))
        assert step["net_log"] == pytest.approx(np.log(original["equity_jpy"] / original["equity_before"]))
        assert step["spread_simple"] < 0 and step["commission_simple"] < 0 and step["overnight_simple"] < 0
    assert fold["net_cumulative_log"] == pytest.approx(result["metrics"]["cumulative_log_return"])
    assert sum(fold[c + "_annual_log"] for c in COMPONENTS) == pytest.approx(fold["net_annual_log"])
    assert sum(fold[c + "_pnl_jpy"] for c in COMPONENTS) == pytest.approx(result["trace"][-1]["equity_jpy"] - 1_000_000)


@pytest.mark.parametrize("change", ["pair_order", "time", "weight", "price", "carry", "spread", "commission", "overnight", "equity", "boundary", "nonfinite", "stress"])
def test_tampered_account_is_rejected_at_its_origin(tmp_path: Path, change: str) -> None:
    result, pairs = evaluated(tmp_path)
    if change == "pair_order":
        pairs[0], pairs[1] = pairs[1], pairs[0]
    elif change == "time":
        pairs[0]["target_timestamp"] = pairs[0]["decision_timestamp"]
    elif change == "boundary":
        result["coverage"]["measurement_end"] = result["trace"][-1]["target_timestamp"]
    elif change == "equity":
        result["trace"][1]["equity_before"] += 100
    elif change == "stress":
        result["scenario"] = "F1"
    else:
        field = {"weight": "target_weight", "price": "price_relative", "carry": "signed_carry_jpy", "spread": "spread_jpy", "commission": "commission_jpy", "overnight": "overnight_jpy", "nonfinite": "target_weight"}[change]
        pairs[1][field] += float("nan") if change == "nonfinite" else 1
    with pytest.raises(ValueError):
        attribute_account(result, pairs)


def test_neutral_allocation_has_zero_common_and_signed_carry(tmp_path: Path) -> None:
    plan, raw, _ = prepare_fixture(tmp_path, 70)
    policy = ConstantAllocation(np.array([.8, -.8, 0, 0, 0, 0, 0, 0, 0]), 5.)
    env, windows = build_period_env(raw, plan, tmp_path / "slice.parquet")
    try:
        result = evaluate_policy(env, windows, plan, policy.action)
    finally:
        env.close()
    result.update({"fold": "2024", "policy": "canonical", "scenario": "F0"})
    pairs = account_trace(result, plan)
    steps, _ = attribute_account(result, pairs)
    assert all(s["common_simple"] == 0 for s in steps)
    assert all(s["financing_simple"] > 0 for s in steps)


def test_projection_reinfers_from_its_own_assets_and_caps_before_projection(tmp_path: Path) -> None:
    plan, raw, _ = prepare_fixture(tmp_path, 70)
    raw["environment"]["max_leverage"] = .5
    projection = CommonProjection(feedback_policy, .5)
    env, windows = build_period_env(raw, plan, tmp_path / "projected.parquet")
    try:
        result = evaluate_policy(env, windows, plan, projection.action)
    finally:
        env.close()
    for row, window in zip(result["trace"], windows):
        observation = {"market": window.astype(np.float32), "assets": np.array(row["assets_before"], dtype=np.float32)}
        scores, proposed = feedback_policy(observation, window)
        effective = proposed[:, 0].astype(float)
        effective *= min(1., .5 / np.abs(effective).sum())
        np.testing.assert_array_equal(row["scores"], scores)
        np.testing.assert_allclose(row["target_weights"], np.full(9, effective.mean()), atol=1e-9)
        assert sum(abs(w) for w in row["target_weights"]) <= .5 + 1e-7
    assert result["trace"][0]["assets_before"] == np.zeros((9, 3)).tolist()
    env, windows = build_period_env(raw, plan, tmp_path / "direct.parquet")
    try:
        direct = evaluate_policy(env, windows, plan, feedback_policy)
    finally:
        env.close()
    assert result["trace"][1]["assets_before"] != direct["trace"][1]["assets_before"]
    assert result["trace"][1]["scores"] != direct["trace"][1]["scores"]


def test_constant_is_train_only_and_uses_effective_weights(tmp_path: Path) -> None:
    market, _ = market_calendar("2023-01-02", 180)
    raw = config_raw()
    raw["train_range"] = {"start": "2023-01-02", "end": "2023-06-01"}
    raw["val_range"] = {"start": "2023-06-01", "end": "2023-07-01"}
    raw["eval_range"] = {"start": "2023-07-01", "end": "2024-01-01"}
    raw["env"]["environment"]["max_leverage"] = .4
    cache = tmp_path / "train.parquet"
    raw["env"]["data"]["path"] = str(cache)
    save_ohlcv_parquet(market, "1d", "2023-01-02", "2024-01-01", cache)
    first = train_allocation(raw, feedback_policy)
    assert first["status"] == "complete"
    assert all(pd.Timestamp(r["target_timestamp"]) < pd.Timestamp("2023-06-01") for r in first["trace"])
    expected = np.mean([r["target_weights"] for r in first["trace"]], axis=0)
    np.testing.assert_array_equal(first["weights"], expected)
    assert np.abs(expected).sum() <= .4 + 1e-7
    market.loc[market.index >= "2023-06-01", :] *= 10
    save_ohlcv_parquet(market, "1d", "2023-01-02", "2024-01-01", cache)
    second = train_allocation(raw, feedback_policy)
    assert first == second
    constant = ConstantAllocation(expected, .4)
    scores, action = constant.action({"assets": np.ones((9, 3))}, np.ones((9, 32, 8)))
    np.testing.assert_array_equal(scores, expected)
    np.testing.assert_array_equal(action[:, 0], expected.astype(np.float32))


def test_reproduction_requires_identical_path_not_just_equal_net(tmp_path: Path) -> None:
    result, _ = evaluated(tmp_path)
    reproduced = copy.deepcopy(result)
    reproduced["runtime"]["git"]["forex_trainer"] = "new additive diagnostic commit"
    verify_reproduction(result, reproduced)
    reproduced["trace"][1]["assets_before"][0][0] += .1
    with pytest.raises(ValueError, match="trace"):
        verify_reproduction(result, reproduced)


def test_registered_panel_reports_all_folds_and_withholds_partial_inference(tmp_path: Path) -> None:
    result, pairs = evaluated(tmp_path)
    _, attribution = attribute_account(result, pairs)
    metrics = fold_metrics(result)
    cells = [{"fold": str(y), "policy": p, "status": "complete", "metrics": metrics, "attribution": attribution} for y in range(2009, 2026) for p in POLICIES]
    report = summarize_attribution(cells)
    evidence = report["component_evidence"]["ppo_ens3"]["common_annual_log"]
    assert len(evidence["leave_one_fold_out_means"]) == 17
    assert report["summaries"]["ppo_ens3"]["2009-2018"]["fold_count"] == 10
    assert report["summaries"]["ppo_ens3"]["2019-2025"]["fold_count"] == 7
    assert report["paired_evidence"]["common_projected_minus_ppo_ens3"]["annualized_net_return"]["mean_difference"] == 0
    cells[0] = {"fold": "2009", "policy": POLICIES[0], "status": "blocked_input", "error": "missing bar"}
    partial = summarize_attribution(cells)
    assert partial["status"] == "incomplete"
    assert partial["summaries"] is None and partial["component_evidence"] is None and partial["paired_evidence"] is None
    assert partial["conclusion"] == "uncertain_not_identified"
    with pytest.raises(ValueError, match="85|fold|panel"):
        summarize_attribution(cells[:-1])


def test_manifest_freezes_two_controls_f0_and_all_folds() -> None:
    import json
    config = json.loads(Path("configs/research/issue30_profit_attribution.json").read_text())
    validate_config(config)
    for key, value in (("scenario", "F1"), ("folds", ["2024"]), ("controls", ["relative_only"])):
        changed = {**config, key: value}
        with pytest.raises(ValueError):
            validate_config(changed)


def test_flat_allocation_uses_continuous_zero_net_log_limit(tmp_path: Path) -> None:
    plan, raw, _ = prepare_fixture(tmp_path, 70)
    env, windows = build_period_env(raw, plan, tmp_path / "flat.parquet")
    try:
        result = evaluate_policy(env, windows, plan, ConstantAllocation(np.zeros(9), 5.).action)
    finally:
        env.close()
    result.update({"fold": "2024", "policy": "train_constant", "scenario": "F0"})
    steps, fold = attribute_account(result, account_trace(result, plan))
    assert all(s["log_allocation_factor"] == 1 for s in steps)
    assert fold["net_cumulative_log"] == 0


def test_bankruptcy_preserves_simple_attribution_and_marks_log_undefined(tmp_path: Path) -> None:
    plan, raw, _ = prepare_fixture(tmp_path, 70)
    for symbol in plan.symbols:
        for field in ("Open", "High", "Low", "Close"):
            plan.market.loc[plan.market.index[64]:, (symbol, field)] *= .01
    env, windows = build_period_env(raw, plan, tmp_path / "bankrupt.parquet")
    try:
        result = evaluate_policy(env, windows, plan, ConstantAllocation(np.full(9, .5), 5.).action)
    finally:
        env.close()
    result.update({"fold": "2024", "policy": "train_constant", "scenario": "F0"})
    steps, fold = attribute_account(result, account_trace(result, plan))
    assert result["status"] == "incomplete_margin_call"
    assert steps[-1]["net_simple"] <= -1
    assert steps[-1]["net_log"] is None
    assert steps[-1]["log_undefined_reason"] == "non_positive_terminal_equity"
    assert fold["net_annual_log"] is None
