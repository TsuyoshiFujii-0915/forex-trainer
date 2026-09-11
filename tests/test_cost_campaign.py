"""Behavioral contracts for the fixed full-period cost experiment."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from forex_trainer.cost_campaign import (
    SCENARIOS, account_trace, compare_sensitivity, run_fold_scenarios,
    scenario_environment, summarize_campaign, validate_campaign,
)
from forex_trainer.full_period import build_period_env, evaluate_policy, require_comparable
from forex_trainer.full_period_sources import load_campaign_sources
from forex_trainer.full_period import read_json
from test_full_period import constant_policy, prepare_fixture, sealed_fixture


@pytest.mark.parametrize("scenario,spread,overnight", [("F0", 1, 1), ("F1", 2, 1), ("F2", 1, 2)])
def test_only_registered_cost_axis_changes(tmp_path: Path, scenario: str, spread: int, overnight: int) -> None:
    _, raw, _ = prepare_fixture(tmp_path, 67)
    before = copy.deepcopy(raw)
    changed = scenario_environment(raw, scenario)
    expected = copy.deepcopy(raw)
    expected["transaction_costs"]["spreads"] = {p: v * spread for p, v in raw["transaction_costs"]["spreads"].items()}
    expected["transaction_costs"]["overnight_rate"] *= overnight
    assert changed == expected
    assert raw == before
    with pytest.raises(ValueError, match="scenario"):
        scenario_environment(raw, "F3")


def test_actual_notional_includes_marked_position_drift_and_costs_reconcile(tmp_path: Path) -> None:
    plan, raw, _ = prepare_fixture(tmp_path, 68)
    env, windows = build_period_env(raw, plan, tmp_path / "slice.parquet")
    try:
        result = evaluate_policy(env, windows, plan, constant_policy)
    finally:
        env.close()
    pairs = account_trace(result, plan)
    assert len(pairs) == 9 * 4
    first, second = result["trace"][:2]
    assert first["actual_traded_notional_jpy"] == pytest.approx(float(np.float32(.8)) * 1_000_000)
    assert second["weight_turnover"] == 0
    assert second["actual_traded_notional_jpy"] > 0
    for row in result["trace"]:
        assert row["equity_jpy"] == pytest.approx(row["equity_before"] + row["price_pnl_jpy"] + row["financing_jpy"] - row["spread_jpy"] - row["commission_jpy"] - row["overnight_jpy"])
        selected = [p for p in pairs if p["decision_timestamp"] == row["decision_timestamp"]]
        assert sum(p["actual_traded_notional_jpy"] for p in selected) == pytest.approx(row["actual_traded_notional_jpy"])
        assert sum(p["signed_carry_jpy"] for p in selected) == pytest.approx(row["financing_jpy"])
    broken = copy.deepcopy(result)
    broken["trace"][1]["spread_jpy"] += 1
    with pytest.raises(ValueError, match="spread"):
        account_trace(broken, plan)


def test_frozen_models_reinfer_from_each_cost_scenarios_account(sealed_fixture: Path, tmp_path: Path) -> None:
    sources = load_campaign_sources(read_json(sealed_fixture), sealed_fixture)
    results = run_fold_scenarios(sources[0], tmp_path)
    assert len(results) == 9
    by_key = {(r["scenario"], r["policy"]): r for r in results}
    for scenario in SCENARIOS:
        require_comparable([by_key[scenario, p] for p in ("canonical", "ridge", "ppo_ens3")])
    f0, f1 = by_key["F0", "ppo_ens3"], by_key["F1", "ppo_ens3"]
    assert f0["trace"][0]["action"] == f1["trace"][0]["action"]
    assert f0["trace"][0]["equity_jpy"] != f1["trace"][0]["equity_jpy"]
    assert f0["trace"][1]["assets_before"] != f1["trace"][1]["assets_before"]
    sensitivity = compare_sensitivity(f0, f1)
    assert sensitivity["changed_action_decisions"] > 0
    with pytest.raises(ValueError, match="costs"):
        require_comparable([f0, f1])
    for field in ("timestamps", "symbols", "source_policy_sha256", "runtime", "market_sha256"):
        bad = copy.deepcopy(f1)
        bad[field] = "different"
        with pytest.raises(ValueError, match=field):
            compare_sensitivity(f0, bad)
    bad = copy.deepcopy(f1)
    bad["costs"]["overnight_rate"] *= 2
    with pytest.raises(ValueError, match="cost"):
        compare_sensitivity(f0, bad)


def test_margin_call_keeps_terminal_account_and_blocks_sensitivity(tmp_path: Path) -> None:
    plan, raw, _ = prepare_fixture(tmp_path, 70)
    raw["environment"]["margin_call_threshold"] = .99999
    env, windows = build_period_env(raw, plan, tmp_path / "slice.parquet")
    try:
        result = evaluate_policy(env, windows, plan, constant_policy)
    finally:
        env.close()
    account_trace(result, plan)
    assert result["status"] == "incomplete_margin_call"
    assert len(result["trace"]) == 1
    with pytest.raises(ValueError, match="incomplete"):
        compare_sensitivity(result, result)


def test_campaign_rejects_budget_changes_and_legacy_measurement() -> None:
    config = read_json(Path("configs/research/issue29_full_period_cost.json"))
    validate_campaign(config)
    for change in ("scenario", "folds", "measurement"):
        bad = copy.deepcopy(config)
        if change == "scenario":
            bad["scenarios"]["F1"]["spread_multiplier"] = 1.5
        elif change == "folds":
            bad["evaluation"]["folds"].pop()
        else:
            bad["evaluation"]["measurement_id"] = "legacy"
        with pytest.raises(ValueError, match="scenario|fold|measurement"):
            validate_campaign(bad)


def test_partial_campaign_does_not_publish_selected_fold_statistics() -> None:
    cells: list[dict[str, Any]] = [
        {"scenario": s, "fold": str(y), "policy": p, "status": "blocked_input", "error": "missing source"}
        for s in SCENARIOS for y in range(2009, 2026) for p in ("canonical", "ridge", "ppo_ens3")
    ]
    summary = summarize_campaign(cells)
    assert summary["status"] == "incomplete"
    assert summary["completed_policy_folds"] == 0
    assert len(summary["cells"]) == 153
    assert summary["paired_evidence"] is None
    assert summary["sensitivity_evidence"] is None
    assert all(v["classification"] == "measurement_or_data_insufficient" for v in summary["policy_assessments"].values())
    with pytest.raises(ValueError, match="153|duplicate|cells"):
        summarize_campaign(cells[:-1])


def test_missing_source_is_saved_for_all_153_cells_and_artifact_tampering_fails(tmp_path: Path) -> None:
    import json
    from forex_trainer.cost_campaign import run_campaign, verify_campaign
    config = read_json(Path("configs/research/issue29_full_period_cost.json"))
    config["evaluation"]["source"]["path"] = str(tmp_path / "missing-source.json")
    campaign = tmp_path / "campaign.json"
    campaign.write_text(json.dumps(config))
    output = tmp_path / "result"
    report = run_campaign(campaign, output)
    assert len(report["cells"]) == 153
    assert {c["status"] for c in report["cells"]} == {"blocked_input"}
    assert all("missing-source.json" in c["error"] for c in report["cells"])
    assert verify_campaign(output)["completed_policy_folds"] == 0
    with pytest.raises(ValueError, match="exists"):
        run_campaign(campaign, output)
    (output / "status.csv").write_text("tampered")
    with pytest.raises(ValueError, match="hash mismatch"):
        verify_campaign(output)


def test_complete_panel_reuses_registered_fold_statistics() -> None:
    from forex_trainer.supervised_portfolio import paired_evidence
    cells = []
    for scenario_index, scenario in enumerate(SCENARIOS):
        for year in range(2009, 2026):
            for policy_index, policy in enumerate(("canonical", "ridge", "ppo_ens3")):
                net = (year - 2015) * .01 + policy_index * .02 - scenario_index * .03
                metrics = {"annualized_net_return": net, "annualized_gross_return": net + .05,
                           "sharpe_annualized": .5, "max_drawdown": .2, "mean_gross_leverage": 3.,
                           "mean_target_gross_exposure": 3., "mean_target_net_exposure": 0.,
                           "total_weight_turnover": 10., "total_cost_ratio": .05,
                           "annualized_gross_minus_net": .05, "total_price_pnl_jpy": 100.,
                           "total_financing_jpy": 10., "total_spread_jpy": 5.,
                           "total_commission_jpy": 0., "total_overnight_jpy": 2.,
                           "total_actual_traded_notional_jpy": 1000.}
                cells.append({"scenario": scenario, "fold": str(year), "policy": policy, "status": "complete", "metrics": metrics})
    report = summarize_campaign(cells)
    assert report["completed_policy_folds"] == 153
    expected = paired_evidence(np.array([(y - 2015) * .01 for y in range(2009, 2026)]), np.array([(y - 2015) * .01 + .02 for y in range(2009, 2026)]))
    assert report["paired_evidence"]["F0"]["canonical_minus_ridge"]["annualized_net_return"] == expected
    assert len(expected["leave_one_fold_out_means"]) == 17
    assert report["summaries"]["F0"]["canonical"]["2009-2018"]["fold_count"] == 10
    assert report["summaries"]["F0"]["canonical"]["2019-2025"]["fold_count"] == 7


def test_one_policy_execution_failure_preserves_other_scenarios(sealed_fixture: Path, tmp_path: Path) -> None:
    source = load_campaign_sources(read_json(sealed_fixture), sealed_fixture)[0]
    source.ridge.model.coefficients[:] = np.nan
    results = run_fold_scenarios(source, tmp_path)
    assert len(results) == 9
    assert sum(r["status"] == "complete" for r in results) == 6
    errors = [r for r in results if r["status"] == "execution_error"]
    assert len(errors) == 3
    assert {r["policy"] for r in errors} == {"ridge"}
    assert all("Malformed frozen policy output" in r["error"] for r in errors)


def test_year_start_contribution_is_separate_from_unpaired_period_difference(tmp_path: Path) -> None:
    from forex_trainer.cost_campaign import describe_period_change
    plan, raw, _ = prepare_fixture(tmp_path, 68)
    env, windows = build_period_env(raw, plan, tmp_path / "slice.parquet")
    try:
        result = evaluate_policy(env, windows, plan, constant_policy)
    finally:
        env.close()
    account_trace(result, plan)
    legacy = {**result["metrics"], "eval_start": result["trace"][1]["decision_timestamp"]}
    description = describe_period_change(result, legacy)
    assert description["comparison_kind"] == "descriptive_different_measurements_no_paired_test"
    assert description["new_year_start_steps"] == 1
    assert description["new_year_start_net_pnl_jpy"] == pytest.approx(result["trace"][0]["equity_jpy"] - 1_000_000)
    assert description["new_year_start_price_pnl_jpy"] == result["trace"][0]["price_pnl_jpy"]
