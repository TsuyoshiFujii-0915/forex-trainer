"""Bounded frozen-policy full-period cost experiment for Issue #29."""

from __future__ import annotations

import argparse
import copy
import gzip
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from forex_env.errors import DataError

from .artifact_provenance import sha256_file
from .full_period import (
    MEASUREMENT_ID, PeriodPlan, build_period_env, canonical_action, checked_path,
    evaluate_policy, json_hash, read_json, require_comparable, runtime_identity,
)
from .full_period_sources import FoldSources, load_campaign_sources
from .spread_decomposition import _match
from .supervised_portfolio import FOLDS, paired_evidence

SCENARIOS = {
    "F0": {"spread_multiplier": 1, "overnight_multiplier": 1},
    "F1": {"spread_multiplier": 2, "overnight_multiplier": 1},
    "F2": {"spread_multiplier": 1, "overnight_multiplier": 2},
}
POLICIES = ("canonical", "ridge", "ppo_ens3")
CAMPAIGN_ID = "issue29-full-period-cost-v1"


def scenario_environment(raw: dict[str, Any], scenario: str) -> dict[str, Any]:
    """Apply exactly one registered treatment to a copied source environment.

    Args:
        raw: Frozen baseline environment.
        scenario: F0, F1, or F2.

    Returns:
        Independent environment configuration with only registered costs changed.
    """
    if scenario not in SCENARIOS:
        raise ValueError(f"Unregistered cost scenario: {scenario}")
    result = copy.deepcopy(raw)
    costs = result["transaction_costs"]
    costs["spreads"] = {p: v * SCENARIOS[scenario]["spread_multiplier"] for p, v in costs["spreads"].items()}
    costs["overnight_rate"] *= SCENARIOS[scenario]["overnight_multiplier"]
    return result


def account_trace(result: dict[str, Any], plan: PeriodPlan) -> list[dict[str, Any]]:
    """Reconcile the actual account and add step/pair monetary attribution.

    Args:
        result: Evaluated policy, enriched in place after successful reconciliation.
        plan: Exact market observations underlying the evaluation.

    Returns:
        Ordered pair records including marked positions and actual traded notional.
    """
    symbols = plan.symbols
    if result["symbols"] != list(symbols):
        raise ValueError(f"Accounting symbols differ: {plan.origin}")
    costs = result["costs"]
    if costs["carry_mode"] != "signed":
        raise ValueError(f"Accounting requires signed carry: {plan.origin}")
    previous = np.zeros(9)
    previous_weights = np.zeros(9)
    equity = 1_000_000.0
    pairs: list[dict[str, Any]] = []
    for i, row in enumerate(result["trace"]):
        decision, target = row["decision_timestamp"], row["target_timestamp"]
        context = f"{plan.origin} {decision}"
        if (decision, target) != plan.timestamps[i:i + 2]:
            raise ValueError(f"Accounting timestamps differ: {context}")
        before = row["equity_before"]
        _match(before, equity, context + " equity continuity", 1e-7)
        weights = np.asarray(row["target_weights"])
        exposure = weights * before
        relatives = np.array([plan.market.loc[pd.Timestamp(target), (s, "Close")] / plan.market.loc[pd.Timestamp(decision), (s, "Close")] for s in symbols])
        rates = np.array([plan.market.loc[pd.Timestamp(decision), (s, "CarryAnnual")] for s in symbols])
        days = (pd.Timestamp(target) - pd.Timestamp(decision)).total_seconds() / 86400
        notional = np.abs(exposure - previous)
        marked = exposure * relatives
        price = exposure * (relatives - 1)
        carry = -marked * rates * days / 365
        spread = notional * np.array([costs["spreads"][s] for s in symbols])
        commission = notional * costs["commission_rate"]
        overnight = np.abs(exposure) * costs["overnight_rate"] * days
        for field, value in (("spread_jpy", spread.sum()), ("commission_jpy", commission.sum()), ("overnight_jpy", overnight.sum()), ("financing_jpy", carry.sum()), ("cost_jpy", (spread + commission + overnight).sum()), ("weight_turnover", np.abs(weights - previous_weights).sum())):
            _match(row[field], float(value), context + " " + field, 1e-7)
        _match(np.array([row["exposures_jpy"][s] for s in symbols]), marked, context + " marked exposure", 1e-7)
        equity = before + price.sum() + carry.sum() - spread.sum() - commission.sum() - overnight.sum()
        _match(row["equity_jpy"], float(equity), context + " equity", 1e-7)
        row.update({"price_pnl_jpy": float(price.sum()), "actual_traded_notional_jpy": float(notional.sum()), "target_gross_exposure": float(np.abs(weights).sum()), "target_net_exposure": float(weights.sum()),
                    "market_transition_sha256": json_hash({"symbols": list(symbols), "decision": decision, "target": target,
                                                           "price_relatives": relatives.tolist(), "carry_annual": rates.tolist()})})
        for j, symbol in enumerate(symbols):
            pairs.append({"decision_timestamp": decision, "target_timestamp": target, "pair": symbol,
                          "score": row["scores"][j], "action": row["action"][j], "target_weight": float(weights[j]),
                          "previous_marked_exposure_jpy": float(previous[j]), "target_exposure_jpy": float(exposure[j]),
                          "marked_exposure_jpy": float(marked[j]), "price_relative": float(relatives[j]),
                          "carry_annual": float(rates[j]), "elapsed_days": days,
                          "price_pnl_jpy": float(price[j]), "signed_carry_jpy": float(carry[j]),
                          "spread_jpy": float(spread[j]), "commission_jpy": float(commission[j]),
                          "overnight_jpy": float(overnight[j]), "actual_traded_notional_jpy": float(notional[j])})
        previous, previous_weights = marked, weights
    return pairs


def compare_sensitivity(baseline: dict[str, Any], stressed: dict[str, Any]) -> dict[str, Any]:
    """Compare one frozen policy with only its registered cost treatment changed.

    Args:
        baseline: Complete F0 account.
        stressed: Complete F1 or F2 account for the same frozen policy and fold.

    Returns:
        Descriptive net/gross differences and observed action changes.
    """
    if baseline["status"] != "complete" or stressed["status"] != "complete":
        raise ValueError("incomplete account cannot enter cost sensitivity")
    for field in ("measurement_id", "fold", "policy", "source_policy_sha256", "timestamps", "symbols", "runtime", "market_sha256", "environment", "coverage"):
        if baseline[field] != stressed[field]:
            raise ValueError(f"Cost sensitivity differs in {field}")
    if baseline["measurement_id"] != MEASUREMENT_ID:
        raise ValueError("Cost sensitivity rejects legacy measurement")
    if baseline["scenario"] != "F0" or stressed["scenario"] not in ("F1", "F2"):
        raise ValueError("Cost sensitivity requires F0 versus F1/F2")
    expected = scenario_environment({"transaction_costs": baseline["costs"]}, stressed["scenario"])["transaction_costs"]
    if stressed["costs"] != expected:
        raise ValueError("Cost sensitivity differs outside registered costs")
    actions = np.array([r["action"] for r in stressed["trace"]]) - np.array([r["action"] for r in baseline["trace"]])
    return {"scenario": stressed["scenario"], "fold": baseline["fold"], "policy": baseline["policy"],
            "changed_action_decisions": int(np.any(actions != 0, axis=1).sum()),
            "maximum_absolute_action_change": float(np.abs(actions).max()),
            "mean_absolute_action_change": float(np.abs(actions).mean()),
            **{name + "_difference": stressed["metrics"][name] - baseline["metrics"][name] for name in ("annualized_net_return", "annualized_gross_return")}}


def run_fold_scenarios(source: FoldSources, output: Path) -> list[dict[str, Any]]:
    """Evaluate nine independent accounts using the existing frozen adapters.

    Args:
        source: Validated frozen fold.
        output: Private directory for new environment caches.

    Returns:
        Nine results with reconciled pair and step traces, including terminals.
    """
    results: list[dict[str, Any]] = []
    adapters = {"canonical": canonical_action, "ridge": source.ridge.action, "ppo_ens3": source.ppo.action}
    for scenario in SCENARIOS:
        for policy in POLICIES:
            identity = {"scenario": scenario, "fold": source.fold, "policy": policy}
            cache = output / f"{source.fold}-{scenario}-{policy}.parquet"
            try:
                if policy == "ridge" and not np.isfinite(source.ridge.model.coefficients).all():
                    raise ValueError("Malformed frozen policy output: nonfinite ridge coefficients")
                env, windows = build_period_env(scenario_environment(source.env_raw, scenario), source.plan, cache)
                try:
                    result = evaluate_policy(env, windows, source.plan, adapters[policy])
                finally:
                    env.close()
                result.update({**identity,
                               "source_policy_sha256": json_hash({"policy": policy, "source": source.identity, "ridge_parameters": source.ridge.parameter_sha256})})
                result["pairs"] = account_trace(result, source.plan)
            except (ValueError, TypeError, KeyError, OSError, RuntimeError, DataError) as exc:
                result = {**identity, "status": "execution_error", "error": f"{type(exc).__name__}: {exc}"}
            finally:
                if cache.exists():
                    cache.unlink()
            stem = f"{scenario}-{policy}"
            payload = {k: v for k, v in result.items() if k != "pairs"}
            with (output / f"{stem}.json.gz").open("xb") as stream:
                stream.write(gzip.compress(json.dumps(payload, separators=(",", ":"), allow_nan=False).encode(), mtime=0))
            if "pairs" in result:
                pd.DataFrame(result["pairs"]).to_csv(output / f"{stem}-pairs.csv.gz", index=False, compression={"method": "gzip", "mtime": 0})
            results.append(result)
        current = results[-3:]
        if all(r["status"] == "complete" for r in current):
            require_comparable(current)
    return results


def validate_campaign(config: dict[str, Any]) -> None:
    """Reject budget changes or mixed measurements before touching artifacts.

    Args:
        config: Issue #29 campaign configuration.
    """
    required = {"campaign_id", "parent_contract", "parent_portfolio", "lock", "scenarios", "evaluation"}
    if set(config) != required or config["campaign_id"] != CAMPAIGN_ID:
        raise ValueError("Expected explicit Issue #29 campaign fields/ID")
    if config["scenarios"] != SCENARIOS:
        raise ValueError("Only the registered F0/F1/F2 scenarios are permitted")
    evaluation = config["evaluation"]
    if evaluation["measurement_id"] != MEASUREMENT_ID or evaluation["mode"] != "full_period":
        raise ValueError("Campaign requires full-period measurement")
    if [f["fold"] for f in evaluation["folds"]] != list(FOLDS):
        raise ValueError("Campaign requires ordered 2009-2025 folds")
    for fold in evaluation["folds"]:
        year = int(fold["fold"])
        if pd.Timestamp(fold["measurement_start"]) != pd.Timestamp(f"{year}-01-01T00:00:00Z") or pd.Timestamp(fold["measurement_end"]) != pd.Timestamp(f"{year + 1}-01-01T00:00:00Z"):
            raise ValueError(f"Campaign requires full UTC calendar year: fold {year}")


def fold_metrics(result: dict[str, Any]) -> dict[str, Any]:
    """Reduce a reconciled account without labelling a terminal as annual evidence.

    Args:
        result: One complete or terminal account.

    Returns:
        Standard metrics and explicit monetary/exposure totals.
    """
    trace, metric = result["trace"], result["metrics"]
    return {**metric,
            "annualized_gross_minus_net": metric["annualized_gross_return"] - metric["annualized_net_return"] if result["status"] == "complete" else None,
            **{"total_" + name: float(sum(r[name] for r in trace)) for name in ("price_pnl_jpy", "financing_jpy", "spread_jpy", "commission_jpy", "overnight_jpy", "actual_traded_notional_jpy")},
            "mean_target_gross_exposure": float(np.mean([r["target_gross_exposure"] for r in trace])),
            "mean_target_net_exposure": float(np.mean([r["target_net_exposure"] for r in trace]))}


def describe_period_change(result: dict[str, Any], legacy: dict[str, Any]) -> dict[str, Any]:
    """Separate new year-start PnL from a descriptive change of measurement.

    Args:
        result: Reconciled complete F0 account.
        legacy: Sealed Issue #16 row for the same source policy and fold.

    Returns:
        Explicitly unpaired period differences and the new account's early PnL.
    """
    if result["status"] != "complete":
        raise ValueError("Incomplete account cannot describe a full-year period change")
    early = [r for r in result["trace"] if pd.Timestamp(r["decision_timestamp"]) < pd.Timestamp(legacy["eval_start"])]
    return {"comparison_kind": "descriptive_different_measurements_no_paired_test",
            "legacy_first_decision": legacy["eval_start"], "legacy_last_mark": legacy["eval_end"],
            "full_period_first_decision": result["coverage"]["first_decision"], "full_period_last_mark": result["coverage"]["last_mark"],
            "new_year_start_steps": len(early),
            "new_year_start_net_pnl_jpy": float(sum(r["equity_jpy"] - r["equity_before"] for r in early)),
            **{"new_year_start_" + k: float(sum(r[k] for r in early)) for k in ("price_pnl_jpy", "financing_jpy", "spread_jpy", "commission_jpy", "overnight_jpy")},
            **{k + "_descriptive_difference": result["metrics"][k] - legacy[k] for k in ("annualized_net_return", "annualized_gross_return")}}


def summarize_campaign(cells: list[dict[str, Any]]) -> dict[str, Any]:
    """Report all cells and withhold inference when the registered panel is partial.

    Args:
        cells: Exactly 153 labelled policy-fold observations or explicit failures.

    Returns:
        Fold/era/all summaries and registered paired evidence, or explicit nulls.
    """
    expected = {(s, f, p) for s in SCENARIOS for f in FOLDS for p in POLICIES}
    keys = [(r["scenario"], r["fold"], r["policy"]) for r in cells]
    if len(keys) != 153 or set(keys) != expected:
        raise ValueError("Campaign requires 153 unique registered cells; duplicate/missing cells")
    complete = sum(r["status"] == "complete" for r in cells)
    report: dict[str, Any] = {"campaign_id": CAMPAIGN_ID, "measurement_id": MEASUREMENT_ID,
        "evidence_class": "development_historical", "status": "complete" if complete == 153 else "incomplete",
        "completed_policy_folds": complete, "cells": cells, "summaries": None,
        "paired_evidence": None, "sensitivity_evidence": None,
        "policy_assessments": {p: {"classification": "measurement_or_data_insufficient", "reason": "All 17 registered folds are required; no selected-fold inference."} for p in POLICIES}}
    if complete != 153:
        return report
    indexed = {(r["scenario"], r["fold"], r["policy"]): r["metrics"] for r in cells}
    aggregates: dict[str, Any] = {}
    paired: dict[str, Any] = {}
    sensitivity: dict[str, Any] = {}
    for scenario in SCENARIOS:
        aggregates[scenario], paired[scenario] = {}, {}
        for policy in POLICIES:
            aggregates[scenario][policy] = {}
            for era, folds in (("all", FOLDS), ("2009-2018", FOLDS[:10]), ("2019-2025", FOLDS[10:])):
                rows = [indexed[scenario, f, policy] for f in folds]
                names = ("annualized_net_return", "annualized_gross_return", "sharpe_annualized", "max_drawdown", "mean_gross_leverage", "mean_target_gross_exposure", "mean_target_net_exposure", "total_weight_turnover", "total_cost_ratio", "annualized_gross_minus_net", "total_price_pnl_jpy", "total_financing_jpy", "total_spread_jpy", "total_commission_jpy", "total_overnight_jpy", "total_actual_traded_notional_jpy")
                aggregates[scenario][policy][era] = {"fold_count": len(rows), "winning_folds": sum(r["annualized_net_return"] > 0 for r in rows), "worst_max_drawdown": max(r["max_drawdown"] for r in rows), **{"mean_" + k: float(np.mean([r[k] for r in rows])) for k in names}}
        for candidate, baseline in combinations(POLICIES, 2):
            paired[scenario][f"{candidate}_minus_{baseline}"] = {m: paired_evidence(np.array([indexed[scenario, f, candidate][m] for f in FOLDS]), np.array([indexed[scenario, f, baseline][m] for f in FOLDS])) for m in ("annualized_net_return", "annualized_gross_return")}
        if scenario != "F0":
            sensitivity[scenario] = {p: {m: paired_evidence(np.array([indexed[scenario, f, p][m] for f in FOLDS]), np.array([indexed["F0", f, p][m] for f in FOLDS])) for m in ("annualized_net_return", "annualized_gross_return")} for p in POLICIES}
    report.update({"summaries": aggregates, "paired_evidence": paired, "sensitivity_evidence": sensitivity})
    for policy in POLICIES:
        net = {s: aggregates[s][policy]["all"]["mean_annualized_net_return"] for s in SCENARIOS}
        report["policy_assessments"][policy] = {"classification": "cost_fragile" if net["F0"] > 0 and min(net.values()) <= 0 else "worth_development_under_current_assumptions" if min(net.values()) > 0 else "no_positive_net_baseline", "scenario_mean_net": net, "reason": "Descriptive development triage only; no independent profitability recognition or change to Issue #16."}
    return report


def write_json(path: Path, value: Any) -> None:
    """Write finite JSON to a new artifact path.

    Args:
        path: Destination path.
        value: JSON-compatible content.
    """
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def _render(report: dict[str, Any]) -> str:
    """Render a compact status table without silently dropping failed folds.

    Args:
        report: Machine-readable experiment report.

    Returns:
        Markdown report text.
    """
    lines = ["# Issue #29 full-period cost experiment", "", f"Status: **{report['status']}**; {report['completed_policy_folds']}/153 policy-folds complete.",
             "", "Evidence: development_historical. Costs are sensitivity assumptions, not broker estimates.",
             "Incomplete panels have no era/all-fold means, paired tests, or profitability classification.",
             "Legacy Issue #16 is a separate descriptive reference; no mixed-measurement paired tests.", "",
             "| Scenario | Fold | Policy | Status | Net annualized | Gross annualized | MDD |", "|---|---|---|---|---:|---:|---:|"]
    for cell in report["cells"]:
        values = [f"{cell['metrics'][k]:.6f}" if cell["status"] == "complete" else "unavailable" for k in ("annualized_net_return", "annualized_gross_return", "max_drawdown")]
        lines.append(f"| {cell['scenario']} | {cell['fold']} | {cell['policy']} | {cell['status']} | " + " | ".join(values) + " |")
    lines.extend(["", "## Input and execution failures", ""])
    errors = sorted({(r["fold"], r["error"]) for r in report["cells"] if "error" in r})
    lines.extend(f"- {fold}: {error}" for fold, error in errors)
    lines.extend(["", "Full metrics, sensitivities, source identities, and pair traces are linked by report.json and manifest.json."])
    return "\n".join(lines) + "\n"


def run_campaign(config_path: Path, output: Path) -> dict[str, Any]:
    """Run the registered panel, preserving explicit input and strategy failures.

    Args:
        config_path: Fixed Issue #29 campaign JSON.
        output: New persistent artifact directory, including partial attempts.

    Returns:
        Complete or explicitly incomplete research report.
    """
    config = read_json(config_path)
    validate_campaign(config)
    if output.exists():
        raise ValueError(f"Output already exists: {output}")
    for field in ("parent_contract", "parent_portfolio", "lock"):
        checked_path(config[field])
    parent = read_json(Path(config["parent_portfolio"]["path"]))
    parent_dir = Path(config["parent_portfolio"]["path"]).parent
    for name, digest in parent["generated_artifact_sha256"].items():
        checked_path({"path": str(parent_dir / name), "sha256": digest})
    if parent["source_provenance_sha256"] != config["evaluation"]["source"]["sha256"]:
        raise ValueError("Issue #15/#16 parent source seals differ")
    output.mkdir(parents=True)
    write_json(output / "campaign_snapshot.json", config)
    runtime = runtime_identity()
    manifest: dict[str, Any] = {"campaign_id": CAMPAIGN_ID, "config_sha256": sha256_file(config_path),
        "runtime": runtime, "python": platform.python_version(), "started_at": datetime.now(timezone.utc).isoformat(),
        "git_status": {"forex_trainer": subprocess.check_output(["git", "status", "--porcelain"], text=True), "forex_env": subprocess.check_output(["git", "-C", "../forex-env-v3", "status", "--porcelain"], text=True)},
        "command": ["forex-cost-campaign", "--config", str(config_path), "--output", str(output)], "folds": {}}
    write_json(output / "started.json", manifest)
    cells: list[dict[str, Any]] = []
    sensitivities: list[dict[str, Any]] = []
    period_changes: list[dict[str, Any]] = []
    legacy = pd.read_csv(parent_dir / "fold_metrics.csv")
    legacy.to_csv(output / "legacy_reference.csv", index=False)
    for fold_config in config["evaluation"]["folds"]:
        fold = fold_config["fold"]
        selected = copy.deepcopy(config["evaluation"])
        selected["folds"] = [fold_config]
        try:
            source = load_campaign_sources(selected, config_path)[0]
            baseline = parent["fold_sources"][fold]
            if source.identity != {**baseline["source"], "lineage": source.identity["lineage"], "calendar_source": source.identity["calendar_source"]}:
                raise ValueError(f"Fold {fold}: Issue #15/#16 source identities differ")
            if source.env_raw["transaction_costs"] != baseline["resolved_eval_env"]["transaction_costs"]:
                raise ValueError(f"Fold {fold}: F0 costs differ from sealed Issue #16")
        except (ValueError, TypeError, KeyError, OSError, DataError) as exc:
            cells.extend({"scenario": s, "fold": fold, "policy": p, "status": "blocked_input", "error": f"{type(exc).__name__}: {exc}"} for s in SCENARIOS for p in POLICIES)
            print(f"{fold}: blocked_input: {exc}", flush=True)
            continue
        manifest["folds"][fold] = {"source": source.identity, "ridge_parameter_sha256": source.ridge.parameter_sha256,
            "history_labels": source.plan.history_labels, "history_instants": [t.isoformat() for t in source.plan.market.index[:63]],
            "effective_timestamps": source.plan.timestamps, "effective_timestamps_sha256": json_hash(source.plan.timestamps)}
        work = output / f"fold-{fold}"
        work.mkdir()
        results = run_fold_scenarios(source, work)
        for result in results:
            stem = f"{result['scenario']}-{result['policy']}"
            cell = {k: result[k] for k in ("scenario", "fold", "policy", "status")}
            cell["result_path"] = f"fold-{fold}/{stem}.json.gz"
            if result["status"] == "execution_error":
                cell["error"] = result["error"]
            else:
                cell.update({"coverage": result["coverage"], "metrics": fold_metrics(result), "pairs_path": f"fold-{fold}/{stem}-pairs.csv.gz"})
            cells.append(cell)
        indexed = {(r["scenario"], r["policy"]): r for r in results}
        for policy, legacy_policy in (("canonical", "reversal"), ("ridge", "supervised"), ("ppo_ens3", "ppo")):
            result = indexed["F0", policy]
            if result["status"] == "complete":
                rows = legacy[(legacy["fold"].astype(str) == fold) & (legacy["policy"] == legacy_policy)]
                if len(rows) != 1:
                    raise ValueError(f"Legacy reference requires one sealed row: {fold}/{policy}")
                period_changes.append({"fold": fold, "policy": policy, **describe_period_change(result, rows.iloc[0].to_dict())})
        for scenario in ("F1", "F2"):
            for policy in POLICIES:
                f0, stress = indexed["F0", policy], indexed[scenario, policy]
                if f0["status"] == stress["status"] == "complete":
                    sensitivities.append(compare_sensitivity(f0, stress))
        print(f"{fold}: {sum(r['status'] == 'complete' for r in results)}/9 complete", flush=True)
        for cache in work.glob("*.parquet"):
            cache.unlink()
    if runtime_identity() != runtime:
        raise ValueError(f"Evaluation runtime changed during campaign; inspect partial attempt {output}")
    report = summarize_campaign(cells)
    report["fold_cost_sensitivity"] = sensitivities
    report["descriptive_period_changes"] = period_changes
    report["legacy_reference"] = {"path": "legacy_reference.csv", "measurement": "legacy_in_year_warmup", "usage": "Descriptive reference only; no paired test or pooled inference with full-period results."}
    write_json(output / "report.json", report)
    (output / "report.md").write_text(_render(report), encoding="utf-8")
    pd.json_normalize(cells).to_csv(output / "status.csv", index=False)
    manifest["finished_at"] = datetime.now(timezone.utc).isoformat()
    manifest["generated_artifact_sha256"] = {str(p.relative_to(output)): sha256_file(p) for p in sorted(output.rglob("*")) if p.is_file()}
    write_json(output / "manifest.json", manifest)
    return report


def verify_campaign(output: Path) -> dict[str, Any]:
    """Verify persisted artifact hashes and scenario/time/pair comparison contracts.

    Args:
        output: Completed or explicitly incomplete campaign directory.

    Returns:
        Verified report; altered or missing artifacts raise explicit exceptions.
    """
    manifest = read_json(output / "manifest.json")
    hashes = manifest["generated_artifact_sha256"]
    actual_files = {str(p.relative_to(output)) for p in output.rglob("*") if p.is_file() and p != output / "manifest.json"}
    if actual_files != set(hashes):
        raise ValueError(f"Campaign artifact inventory differs: {output}")
    for name, digest in hashes.items():
        checked_path({"path": str(output / name), "sha256": digest})
    report = read_json(output / "report.json")
    summary = summarize_campaign(report["cells"])
    for name in summary:
        if summary[name] != report[name]:
            raise ValueError(f"Campaign report summary differs: {name}: {output}")
    results: dict[tuple[str, str, str], dict[str, Any]] = {}
    for cell in report["cells"]:
        if cell["status"] not in ("complete", "incomplete_margin_call"):
            continue
        with gzip.open(output / cell["result_path"], "rt", encoding="utf-8") as stream:
            result = json.load(stream)
        if result["runtime"] != manifest["runtime"] or fold_metrics(result) != cell["metrics"]:
            raise ValueError(f"Campaign cell runtime/metrics differ: {cell['result_path']}")
        pairs = pd.read_csv(output / cell["pairs_path"])
        expected = [(r["decision_timestamp"], r["target_timestamp"], s) for r in result["trace"] for s in result["symbols"]]
        if list(pairs[["decision_timestamp", "target_timestamp", "pair"]].itertuples(index=False, name=None)) != expected:
            raise ValueError(f"Campaign pair trace time/order mismatch: {cell['pairs_path']}")
        results[cell["scenario"], cell["fold"], cell["policy"]] = result
    sensitivities = []
    for fold in FOLDS:
        for scenario in SCENARIOS:
            same = [results[scenario, fold, p] for p in POLICIES if (scenario, fold, p) in results]
            if len(same) == 3 and all(r["status"] == "complete" for r in same):
                require_comparable(same)
        for scenario in ("F1", "F2"):
            for policy in POLICIES:
                keys = [(s, fold, policy) for s in ("F0", scenario)]
                if all(k in results and results[k]["status"] == "complete" for k in keys):
                    sensitivities.append(compare_sensitivity(results[keys[0]], results[keys[1]]))
    if sensitivities != report["fold_cost_sensitivity"]:
        raise ValueError(f"Campaign sensitivity evidence differs: {output}")
    return report


def main() -> int:
    """Run the bounded campaign CLI.

    Returns:
        Zero for all 153 complete, two for incomplete, one for configuration errors.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    try:
        report = run_campaign(args.config, args.output)
    except (ValueError, TypeError, KeyError, OSError, DataError) as exc:
        print(f"cost campaign failed: {exc}", file=sys.stderr)
        return 1
    print(f"{report['status']}: {report['completed_policy_folds']}/153; {args.output}")
    return 0 if report["status"] == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
