"""Frozen F0 profit attribution and two preregistered descriptive controls."""

from __future__ import annotations

import argparse
import copy
import gzip
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from forex_env.errors import DataError

from .artifact_provenance import sha256_file
from .config import RangeConfig, parse_experiment_config, resolve_env_raw
from .cost_campaign import account_trace, fold_metrics, verify_campaign, write_json
from .env_factory import GateEvaluationMode, build_single_env
from .features import CROSS_FEATURE_REGISTRY, FEATURE_REGISTRY
from .full_period import (
    MEASUREMENT_ID, Policy, build_period_env, canonical_action, checked_path,
    evaluate_policy, json_hash, read_json, require_comparable, runtime_identity,
)
from .full_period_sources import load_campaign_sources
from .spread_decomposition import _match
from .supervised_portfolio import FOLDS, paired_evidence

CAMPAIGN_ID = "issue30-profit-attribution-v1"
BASE_POLICIES = ("canonical", "ridge", "ppo_ens3")
CONTROLS = ("train_constant", "common_projected")
POLICIES = BASE_POLICIES + CONTROLS
COMPONENTS = ("common", "relative", "financing", "spread", "commission", "overnight")


def effective_proposal(action: np.ndarray, cap: float) -> np.ndarray:
    """Apply the pinned-leverage environment's pair clip and global cap.

    Args:
        action: Nine-pair direct action.
        cap: Explicit positive source gross limit.

    Returns:
        Float64 effective proposal before common projection.
    """
    if action.shape != (9, 1) or not np.isfinite(action).all():
        raise ValueError("Common projection requires finite (9, 1) proposal")
    if not np.isfinite(cap) or cap <= 0:
        raise ValueError(f"Invalid source leverage cap: {cap}")
    weights = np.clip(action[:, 0].astype(np.float64), -1, 1)
    gross = float(np.abs(weights).sum())
    if gross > cap:
        weights *= cap / gross
    return weights


class CommonProjection:
    """Infer the frozen policy on the projected account and retain its mean."""

    def __init__(self, predict: Policy, cap: float) -> None:
        """Bind a frozen policy and its original risk limit.

        Args:
            predict: Closed-loop frozen policy inference.
            cap: Source gross exposure limit.
        """
        effective_proposal(np.zeros((9, 1)), cap)
        self.predict = predict
        self.cap = cap

    def action(self, observation: dict[str, np.ndarray], window: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Project the current account's newly inferred capped proposal.

        Args:
            observation: Projected account state, including its own assets.
            window: Current causal feature window.

        Returns:
            Original proposal scores and common-only float32 action.
        """
        scores, proposal = self.predict(observation, window)
        mean = float(effective_proposal(proposal, self.cap).mean())
        return scores, np.full((9, 1), mean, dtype=np.float32)


class ConstantAllocation:
    """A train-derived fixed allocation with no evaluation-time fitting."""

    def __init__(self, weights: np.ndarray, cap: float) -> None:
        """Freeze finite feasible train means.

        Args:
            weights: Mean effective train weights in sealed pair order.
            cap: Unchanged source gross limit.
        """
        effective_proposal(np.zeros((9, 1)), cap)
        if weights.shape != (9,) or not np.isfinite(weights).all():
            raise ValueError("Constant allocation requires nine finite train weights")
        if np.abs(weights).max() > 1 + 1e-12 or np.abs(weights).sum() > cap + 1e-12:
            raise ValueError("Train mean allocation exceeds original risk limits")
        self.weights = weights.astype(np.float64).copy()
        self.weights.setflags(write=False)

    def action(self, observation: dict[str, np.ndarray], window: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return the frozen train allocation regardless of evaluation outcomes.

        Args:
            observation: Evaluation state; intentionally unused.
            window: Evaluation features; intentionally unused.

        Returns:
            Train mean and its direct float32 execution proposal.
        """
        return self.weights.copy(), self.weights.astype(np.float32).reshape(9, 1)


def train_allocation(raw: dict[str, Any], predict: Policy) -> dict[str, Any]:
    """Replay only the frozen training range, retaining effective weights.

    Args:
        raw: Original hash-verified experiment config.
        predict: Frozen PPO inference; no training or parameter selection.

    Returns:
        Complete train mean or explicit terminal failure with the full trace.
    """
    config = parse_experiment_config(raw)
    if config.run.decision_interval != 1 or config.env["environment"]["allow_action_leverage"]:
        raise ValueError("Train constant requires daily pinned-leverage direct policy")
    start, end = pd.Timestamp(config.train_range.start), pd.Timestamp(config.train_range.end)
    stop = (end - pd.Timedelta(days=1)).date().isoformat()
    resolved = resolve_env_raw(config.env, RangeConfig(config.train_range.start, stop), for_eval=True)
    market = pd.read_parquet(config.env["data"]["path"])
    dates = market.index.tz_localize(None) if market.index.tz is not None else market.index
    selected = dates[(dates >= start) & (dates < end)]
    if len(selected) < 65:
        raise ValueError(f"Train range {start}/{end} requires at least 65 bars; got {len(selected)}")
    names = tuple(config.env["features"]["selected"])
    env = build_single_env(resolved, tuple(n for n in names if n in FEATURE_REGISTRY),
                           tuple(n for n in names if n in CROSS_FEATURE_REGISTRY),
                           seed=0, decision_interval=1, residual=None, rank_allocation=None,
                           apply_hold_gate=None, gate_evaluation_mode=GateEvaluationMode.LEARNED)
    trace: list[dict[str, Any]] = []
    try:
        observation, info = env.reset(seed=0)
        if np.any(observation["assets"]) or info["equity_jpy"] != 1_000_000:
            raise ValueError("Train replay must reset to flat 1000000 JPY")
        for index in range(len(selected) - 64):
            before = info
            assets = observation["assets"].tolist()
            decision = pd.Timestamp(before["timestamp"])
            decision_label = decision.tz_localize(None) if decision.tz is not None else decision
            if decision_label != selected[index + 63]:
                raise ValueError(f"Train decision alignment differs at {index}: {decision}")
            scores, action = predict(observation, observation["market"].astype(np.float64))
            effective_proposal(action, float(config.env["environment"]["max_leverage"]))
            observation, reward, terminated, truncated, info = env.step(action)
            target = pd.Timestamp(info["timestamp"])
            target_label = target.tz_localize(None) if target.tz is not None else target
            if target_label != selected[index + 64] or not start <= decision_label < target_label < end:
                raise ValueError(f"Train-only boundary/alignment violation: {decision}/{target}")
            trace.append({"decision_timestamp": decision.isoformat(), "target_timestamp": target.isoformat(),
                          "assets_before": assets, "scores": scores.tolist(), "action": action[:, 0].tolist(),
                          "target_weights": info["target_weights"], "equity_before": before["equity_jpy"],
                          "equity_jpy": info["equity_jpy"], "reward": float(reward),
                          "terminated": bool(terminated), "truncated": bool(truncated)})
            if terminated:
                return {"status": "incomplete_margin_call", "weights": None, "train_range": raw["train_range"],
                        "symbols": config.env["environment"]["currency_pairs"], "trace": trace}
            if truncated != (index == len(selected) - 65):
                raise ValueError(f"Unexpected train truncation at {target}")
    finally:
        env.close()
    weights = np.mean([r["target_weights"] for r in trace], axis=0, dtype=np.float64)
    ConstantAllocation(weights, float(config.env["environment"]["max_leverage"]))
    return {"status": "complete", "weights": weights.tolist(), "train_range": raw["train_range"],
            "symbols": config.env["environment"]["currency_pairs"], "trace": trace}


def attribute_account(result: dict[str, Any], pairs: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Reconcile a sealed F0 account and split its realized price return.

    Args:
        result: Source or independently replayed F0 step trace.
        pairs: Exact ordered pair accounting records for those transitions.

    Returns:
        Additive step attribution and fold-level realized-path contributions.
    """
    origin = f"{result['fold']}/{result['policy']}"
    if result["measurement_id"] != MEASUREMENT_ID or result["scenario"] != "F0":
        raise ValueError(f"Attribution requires full-period F0: {origin}")
    symbols, trace = result["symbols"], result["trace"]
    if len(symbols) != 9 or len(set(symbols)) != 9 or not trace:
        raise ValueError(f"Attribution requires nine unique pairs and nonempty trace: {origin}")
    expected = [(r["decision_timestamp"], r["target_timestamp"], p) for r in trace for p in symbols]
    actual = [(r["decision_timestamp"], r["target_timestamp"], r["pair"]) for r in pairs]
    if actual != expected:
        raise ValueError(f"Pair order/time/count mismatch: {origin}")
    times = [trace[0]["decision_timestamp"], *[r["target_timestamp"] for r in trace]]
    if times != result["timestamps"]:
        raise ValueError(f"Step timestamps mismatch: {origin}")
    costs = result["costs"]
    if costs["carry_mode"] != "signed":
        raise ValueError(f"Expected signed financing: {origin}")
    equity, previous = 1_000_000., np.zeros(9)
    steps: list[dict[str, Any]] = []
    start, end = (pd.Timestamp(result["coverage"][k]) for k in ("measurement_start", "measurement_end"))
    for index, row in enumerate(trace):
        context = f"{origin} {row['decision_timestamp']}"
        decision, target = pd.Timestamp(row["decision_timestamp"]), pd.Timestamp(row["target_timestamp"])
        if not start <= decision < target < end or row["decision_timestamp"] != times[index]:
            raise ValueError(f"Fold boundary/continuity violation: {context}")
        before = float(row["equity_before"])
        if before <= 0 or not np.isfinite(before):
            raise ValueError(f"Non-positive/nonfinite starting equity: {context}")
        _match(before / equity, 1., context + " equity continuity", 1e-12)
        selected = pairs[index * 9:(index + 1) * 9]
        values = {key: np.array([p[key] for p in selected], dtype=np.float64) for key in selected[0] if key not in ("decision_timestamp", "target_timestamp", "pair")}
        for key, value in values.items():
            if not np.isfinite(value).all():
                raise ValueError(f"Nonfinite pair {key}: {context}")
        market_digest = json_hash({"symbols": symbols, "decision": row["decision_timestamp"], "target": row["target_timestamp"],
                                   "price_relatives": values["price_relative"].tolist(), "carry_annual": values["carry_annual"].tolist()})
        if market_digest != row["market_transition_sha256"]:
            raise ValueError(f"Source market transition differs, including unheld pairs: {context}")
        weights = values["target_exposure_jpy"] / before
        _match(weights, values["target_weight"], context + " effective weight", 1e-12)
        _match(weights, np.asarray(row["target_weights"]), context + " step weight", 1e-12)
        _match(values["action"], np.asarray(row["action"]), context + " action", 1e-12)
        _match(values["score"], np.asarray(row["scores"]), context + " scores", 1e-12)
        _match(weights, effective_proposal(np.asarray(row["action"]).reshape(9, 1), result["environment"]["max_leverage"]), context + " capped action", 1e-12)
        returns = values["price_relative"] - 1
        days = (target - decision).total_seconds() / 86400
        _match(values["elapsed_days"], np.full(9, days), context + " elapsed days", 1e-12)
        _match(values["previous_marked_exposure_jpy"] / before, previous / before, context + " previous marked", 1e-12)
        marked = weights * values["price_relative"]
        notional = np.abs(weights - previous / before)
        price = weights * returns
        financing = -marked * values["carry_annual"] * days / 365
        spread = notional * np.array([costs["spreads"][p] for p in symbols])
        commission = notional * costs["commission_rate"]
        overnight = np.abs(weights) * costs["overnight_rate"] * days
        for key, computed in (("marked_exposure_jpy", marked), ("actual_traded_notional_jpy", notional),
                              ("price_pnl_jpy", price), ("signed_carry_jpy", financing),
                              ("spread_jpy", spread), ("commission_jpy", commission), ("overnight_jpy", overnight)):
            _match(values[key] / before, computed, context + " pair " + key, 1e-12)
        _match(np.array([row["exposures_jpy"][p] for p in symbols]) / before, marked, context + " marked exposures", 1e-12)
        for key, computed in (("price_pnl_jpy", price), ("financing_jpy", financing), ("spread_jpy", spread),
                              ("commission_jpy", commission), ("overnight_jpy", overnight),
                              ("cost_jpy", spread + commission + overnight), ("actual_traded_notional_jpy", notional)):
            _match(row[key] / before, float(computed.sum()), context + " step " + key, 1e-12)
        common = float(weights.sum() * returns.mean())
        relative = float(weights @ (returns - returns.mean()))
        _match(common + relative, float(price.sum()), context + " price identity", 1e-12)
        components = dict(zip(COMPONENTS, (common, relative, float(financing.sum()), -float(spread.sum()), -float(commission.sum()), -float(overnight.sum()))))
        net = row["equity_jpy"] / before - 1
        _match(float(sum(components.values())), net, context + " net identity", 1e-12)
        undefined = net <= -1
        if undefined and not (row["terminated"] and index == len(trace) - 1 and result["status"] == "incomplete_margin_call"):
            raise ValueError(f"Non-positive equity without terminal margin call: {context}")
        net_log = None if undefined else float(np.log1p(net))
        factor = None if undefined else 1. if net == 0 else net_log / net
        step = {"fold": result["fold"], "policy": result["policy"], "decision_timestamp": row["decision_timestamp"],
                "target_timestamp": row["target_timestamp"], "equity_before": before, "equity_jpy": row["equity_jpy"],
                "price_simple": float(price.sum()), "net_simple": net, "net_log": net_log,
                "log_allocation_factor": factor, "log_undefined_reason": "non_positive_terminal_equity" if undefined else None,
                "gross_exposure": float(np.abs(weights).sum()), "net_exposure": float(weights.sum()),
                "actual_traded_notional_jpy": row["actual_traded_notional_jpy"], "weight_turnover": row["weight_turnover"]}
        for component, value in components.items():
            step.update({component + "_simple": value, component + "_log": None if undefined else value * factor,
                         component + "_pnl_jpy": value * before})
        if not undefined:
            _match(sum(step[c + "_log"] for c in COMPONENTS), net_log, context + " log identity", 1e-12)
        steps.append(step)
        equity, previous = row["equity_jpy"], marked * before
    years = (pd.Timestamp(times[-1]) - pd.Timestamp(times[0])).total_seconds() / (365.25 * 86400)
    log_defined = all(s["net_log"] is not None for s in steps)
    annual_defined = log_defined and result["status"] == "complete"
    fold: dict[str, Any] = {"elapsed_years": years, "steps": len(steps), "log_undefined_reason": None if log_defined else "non_positive_terminal_equity"}
    for name in (*COMPONENTS, "net"):
        total = float(sum(s[name + "_log"] for s in steps)) if log_defined else None
        fold[name + "_cumulative_log"] = total
        fold[name + "_annual_log"] = total / years if annual_defined else None
        if name != "net":
            fold[name + "_pnl_jpy"] = float(sum(s[name + "_pnl_jpy"] for s in steps))
    if log_defined:
        _match(fold["net_cumulative_log"], result["metrics"]["cumulative_log_return"], origin + " cumulative net log", len(steps) * 1e-12)
    _match(sum(fold[c + "_pnl_jpy"] for c in COMPONENTS) / 1_000_000, (equity - 1_000_000) / 1_000_000, origin + " cumulative JPY", len(steps) * 1e-12)
    fold["annualized_net_log_volatility"] = float(np.std([s["net_log"] for s in steps], ddof=1) * np.sqrt(len(steps) / years)) if annual_defined and len(steps) > 1 else None
    return steps, fold


def verify_reproduction(sealed: dict[str, Any], replayed: dict[str, Any]) -> None:
    """Bridge an additive diagnostic runtime only with identical account paths.

    Args:
        sealed: Original F0 result, retaining its runtime identity.
        replayed: Same frozen policy evaluated under the new runtime.
    """
    fields = ("measurement_id", "fold", "policy", "scenario", "status", "symbols", "timestamps", "initial_assets",
              "market_sha256", "environment", "costs", "coverage", "metrics", "trace")
    for field in fields:
        if sealed[field] != replayed[field]:
            raise ValueError(f"F0 reproduction differs in {field}: {sealed['fold']}/{sealed['policy']}")
    for field in ("device", "versions"):
        if sealed["runtime"][field] != replayed["runtime"][field]:
            raise ValueError(f"F0 reproduction runtime {field} differs")
    if sealed["runtime"]["git"]["forex_env"] != replayed["runtime"]["git"]["forex_env"] or sealed["runtime"]["source_sha256"]["forex_env"] != replayed["runtime"]["source_sha256"]["forex_env"]:
        raise ValueError("F0 reproduction environment implementation differs")


def summarize_attribution(cells: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate only the complete registered five-policy, seventeen-fold panel.

    Args:
        cells: All 85 explicit results or input/terminal/execution failures.

    Returns:
        Fold evidence, era summaries and two paired controls, or explicit nulls.
    """
    expected = {(f, p) for f in FOLDS for p in POLICIES}
    if len(cells) != 85 or {(c["fold"], c["policy"]) for c in cells} != expected:
        raise ValueError("Attribution panel requires 85 unique registered policy-fold cells")
    allowed = {"complete", "blocked_input", "execution_error", "incomplete_margin_call", "blocked_train_terminal"}
    if any(c["status"] not in allowed for c in cells):
        raise ValueError("Unknown attribution cell status")
    count = sum(c["status"] == "complete" for c in cells)
    report: dict[str, Any] = {"campaign_id": CAMPAIGN_ID, "measurement_id": MEASUREMENT_ID,
        "evidence_class": "development_historical", "status": "complete" if count == 85 else "incomplete",
        "completed_policy_folds": count, "cells": cells, "summaries": None, "component_evidence": None,
        "paired_evidence": None, "conclusion": "uncertain_not_identified", "next_learning_problem": "insufficient_information",
        "interpretation": "Realized-path attribution, not standalone component CAGR or causal/risk-adjusted superiority. No automatic post-hoc dominance thresholds."}
    if count != 85:
        return report
    indexed = {(c["fold"], c["policy"]): c for c in cells}
    summaries, component_evidence, paired = {}, {}, {}
    for policy in POLICIES:
        rows = [indexed[f, policy] for f in FOLDS]
        component_evidence[policy] = {name + "_annual_log": paired_evidence(np.array([r["attribution"][name + "_annual_log"] for r in rows]), np.zeros(17)) for name in (*COMPONENTS, "net")}
        summaries[policy] = {}
        for era, folds in (("all", FOLDS), ("2009-2018", FOLDS[:10]), ("2019-2025", FOLDS[10:])):
            selected = [indexed[f, policy] for f in folds]
            summaries[policy][era] = {"fold_count": len(selected),
                "mean_metrics": {k: float(np.mean([r["metrics"][k] for r in selected])) for k in ("annualized_net_return", "annualized_gross_return", "max_drawdown", "mean_target_gross_exposure", "mean_target_net_exposure", "total_weight_turnover", "total_actual_traded_notional_jpy")},
                "worst_max_drawdown": max(r["metrics"]["max_drawdown"] for r in selected),
                "mean_attribution": {k: float(np.mean([r["attribution"][k] for r in selected])) for k in rows[0]["attribution"] if k not in ("log_undefined_reason", "steps")}}
    for policy in CONTROLS:
        paired[policy + "_minus_ppo_ens3"] = {metric: paired_evidence(np.array([indexed[f, policy]["metrics"][metric] for f in FOLDS]), np.array([indexed[f, "ppo_ens3"]["metrics"][metric] for f in FOLDS])) for metric in ("annualized_net_return", "annualized_gross_return")}
    report.update({"summaries": summaries, "component_evidence": component_evidence, "paired_evidence": paired,
                   "interpretation": report["interpretation"] + " Complete evidence requires a reasoned research-note interpretation; uncertain remains a valid conclusion."})
    return report


def validate_config(config: dict[str, Any]) -> None:
    """Reject changes to the registered budget and source requirements.

    Args:
        config: Explicit diagnostic configuration.
    """
    if set(config) != {"campaign_id", "scenario", "folds", "controls", "parent_manifest", "registration", "parent_contract", "lock"}:
        raise ValueError("Expected exact Issue #30 config fields")
    if config["campaign_id"] != CAMPAIGN_ID or config["scenario"] != "F0" or config["folds"] != list(FOLDS) or config["controls"] != list(CONTROLS):
        raise ValueError("Issue #30 requires F0, all 17 folds, and exactly two registered controls")


def write_compressed(path: Path, value: Any) -> None:
    """Write a deterministic new compressed JSON artifact.

    Args:
        path: New output file.
        value: Finite JSON-compatible payload.
    """
    with path.open("xb") as stream:
        stream.write(gzip.compress(json.dumps(value, separators=(",", ":"), allow_nan=False).encode(), mtime=0))


def read_compressed(path: Path) -> dict[str, Any]:
    """Read an explicitly identified compressed account.

    Args:
        path: Verified artifact path.

    Returns:
        Account mapping.
    """
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        return json.load(stream)


def run_attribution(config_path: Path, output: Path) -> dict[str, Any]:
    """Verify F0, reproduce frozen paths and evaluate registered controls.

    Args:
        config_path: Preregistered manifest path.
        output: New directory, never an overwrite target.

    Returns:
        Complete or explicitly incomplete diagnostic report.
    """
    config = read_json(config_path)
    validate_config(config)
    for field in ("parent_manifest", "registration", "parent_contract", "lock"):
        checked_path(config[field])
    parent_dir = Path(config["parent_manifest"]["path"]).parent
    parent = verify_campaign(parent_dir)
    parent_manifest = read_json(parent_dir / "manifest.json")
    parent_config = read_json(parent_dir / "campaign_snapshot.json")
    if parent_config["parent_contract"] != config["parent_contract"] or parent_config["lock"] != config["lock"]:
        raise ValueError("Parent research/lock contract differs")
    if output.exists():
        raise ValueError(f"Output already exists: {output}")
    output.mkdir(parents=True)
    runtime = runtime_identity()
    manifest = {"campaign_id": CAMPAIGN_ID, "config_sha256": sha256_file(config_path), "runtime": runtime,
                "parent_manifest": config["parent_manifest"], "parent_runtime": parent_manifest["runtime"],
                "started_at": datetime.now(timezone.utc).isoformat(), "fold_sources": {},
                "command": ["forex-profit-attribution", "--config", str(config_path), "--output", str(output)]}
    write_json(output / "campaign_snapshot.json", config)
    write_json(output / "started.json", manifest)
    cells: list[dict[str, Any]] = []
    all_steps: list[dict[str, Any]] = []
    parent_cells = {(c["fold"], c["policy"]): c for c in parent["cells"] if c["scenario"] == "F0"}
    for fold_config in parent_config["evaluation"]["folds"]:
        fold = fold_config["fold"]
        originals: dict[str, dict[str, Any]] = {}
        work = output / f"fold-{fold}"
        source = None
        if any(parent_cells[fold, p]["status"] in ("complete", "incomplete_margin_call") for p in BASE_POLICIES):
            selected = copy.deepcopy(parent_config["evaluation"])
            selected["folds"] = [fold_config]
            source = load_campaign_sources(selected, config_path)[0]
            if source.identity != parent_manifest["folds"][fold]["source"]:
                raise ValueError(f"Source identity differs from parent F0: {fold}")
            manifest["fold_sources"][fold] = parent_manifest["folds"][fold]
        for policy in BASE_POLICIES:
            cell = parent_cells[fold, policy]
            if cell["status"] not in ("complete", "incomplete_margin_call"):
                cells.append({"fold": fold, "policy": policy, "status": cell["status"], "error": cell["error"]})
                continue
            sealed = read_compressed(parent_dir / cell["result_path"])
            if source is None:
                raise ValueError(f"Missing verified market source for F0 attribution: {fold}")
            reconstructed = copy.deepcopy(sealed)
            account_trace(reconstructed, source.plan)
            for original_row, reconstructed_row in zip(sealed["trace"], reconstructed["trace"]):
                for field in ("price_pnl_jpy", "actual_traded_notional_jpy", "target_gross_exposure", "target_net_exposure"):
                    _match(original_row[field], reconstructed_row[field], f"Parent F0 {fold}/{policy} {original_row['decision_timestamp']} {field}", 1e-12)
            sealed = reconstructed
            pairs = pd.read_csv(parent_dir / cell["pairs_path"], float_precision="round_trip").to_dict("records")
            steps, attribution = attribute_account(sealed, pairs)
            identity = {"source_result_path": str(parent_dir / cell["result_path"]),
                        "source_result_sha256": parent_manifest["generated_artifact_sha256"][cell["result_path"]],
                        "source_pairs_sha256": parent_manifest["generated_artifact_sha256"][cell["pairs_path"]]}
            all_steps.extend({**s, **identity} for s in steps)
            cells.append({"fold": fold, "policy": policy, "status": sealed["status"], "metrics": fold_metrics(sealed), "attribution": attribution, **identity})
            originals[policy] = sealed
        if len(originals) != 3 or any(r["status"] != "complete" for r in originals.values()):
            reasons = [{"policy": p, "status": parent_cells[fold, p]["status"],
                        "detail": parent_cells[fold, p]} for p in BASE_POLICIES if parent_cells[fold, p]["status"] != "complete"]
            cells.extend({"fold": fold, "policy": p, "status": "blocked_input", "error": "Parent F0 panel incomplete", "parent_failures": reasons} for p in CONTROLS)
            print(f"{fold}: parent F0 incomplete; controls blocked", flush=True)
            continue
        if source is None:
            raise ValueError(f"Missing verified replay source: {fold}")
        work.mkdir()
        reproduced: list[dict[str, Any]] = []
        for policy, predict in (("canonical", canonical_action), ("ridge", source.ridge.action), ("ppo_ens3", source.ppo.action)):
            cache = work / f"{policy}.parquet"
            env, windows = build_period_env(source.env_raw, source.plan, cache)
            try:
                replayed = evaluate_policy(env, windows, source.plan, predict)
            finally:
                env.close()
                cache.unlink()
            replayed.update({"fold": fold, "policy": policy, "scenario": "F0"})
            account_trace(replayed, source.plan)
            verify_reproduction(originals[policy], replayed)
            write_compressed(work / f"reproduced-{policy}.json.gz", replayed)
            reproduced.append(replayed)
        raw = yaml.safe_load(checked_path({"path": source.identity["config_path"], "sha256": source.identity["config_sha256"]}).read_text())
        training = train_allocation(raw, source.ppo.action)
        write_compressed(work / "train-replay.json.gz", training)
        cap = float(source.env_raw["environment"]["max_leverage"])
        adapters: dict[str, Policy] = {"common_projected": CommonProjection(source.ppo.action, cap).action}
        if training["status"] == "complete":
            adapters["train_constant"] = ConstantAllocation(np.asarray(training["weights"]), cap).action
        else:
            cells.append({"fold": fold, "policy": "train_constant", "status": "blocked_train_terminal", "error": "Train PPO terminated; no partial-train mean used", "train_trace_path": f"fold-{fold}/train-replay.json.gz"})
        for policy in CONTROLS:
            if policy not in adapters:
                continue
            cache = work / f"{policy}.parquet"
            env, windows = build_period_env(source.env_raw, source.plan, cache)
            try:
                result = evaluate_policy(env, windows, source.plan, adapters[policy])
            finally:
                env.close()
                cache.unlink()
            result.update({"fold": fold, "policy": policy, "scenario": "F0"})
            pairs = account_trace(result, source.plan)
            steps, attribution = attribute_account(result, pairs)
            path = work / f"{policy}.json.gz"
            write_compressed(path, result)
            pair_path = work / f"{policy}-pairs.csv.gz"
            pd.DataFrame(pairs).to_csv(pair_path, index=False, compression={"method": "gzip", "mtime": 0})
            identity = {"source_result_path": str(path), "source_result_sha256": sha256_file(path), "source_pairs_sha256": sha256_file(pair_path)}
            all_steps.extend({**s, **identity} for s in steps)
            cells.append({"fold": fold, "policy": policy, "status": result["status"], "metrics": fold_metrics(result),
                          "attribution": attribution, "result_path": str(path.relative_to(output)),
                          "pairs_path": str(pair_path.relative_to(output)), **identity})
            if result["status"] == "complete":
                require_comparable([*reproduced, result])
        print(f"{fold}: attributed F0; train={training['status']}; controls replayed", flush=True)
    if runtime_identity() != runtime:
        raise ValueError(f"Runtime changed during attribution: {output}")
    cells.sort(key=lambda c: (c["fold"], POLICIES.index(c["policy"])))
    report = summarize_attribution(cells)
    write_json(output / "report.json", report)
    pd.json_normalize(cells).to_csv(output / "folds.csv", index=False)
    pd.DataFrame(all_steps).to_csv(output / "steps.csv.gz", index=False, compression={"method": "gzip", "mtime": 0})
    lines = ["# Issue #30 profit attribution", "", f"Status: **{report['status']}**; {report['completed_policy_folds']}/85 policy-folds complete.",
             "", "Evidence: development_historical. Common denotes the nine-pair coordinate mean, not an identified JPY factor.",
             "Contributions follow the realized equity path; they are not component strategy CAGRs.",
             "Controls may have different exposure and risk. Net differences do not establish causal superiority.",
             "Partial panels withhold era/all-fold evidence, bootstrap and LOO. Next learning problem: insufficient information.",
             "", "| Fold | Policy | Status | Common annual log | Relative annual log | Net annual log |", "|---|---|---|---:|---:|---:|"]
    for cell in cells:
        values = [str(cell["attribution"][k + "_annual_log"]) if "attribution" in cell else "unavailable" for k in ("common", "relative", "net")]
        lines.append(f"| {cell['fold']} | {cell['policy']} | {cell['status']} | " + " | ".join(values) + " |")
    (output / "report.md").write_text("\n".join(lines) + "\n")
    manifest["finished_at"] = datetime.now(timezone.utc).isoformat()
    manifest["generated_artifact_sha256"] = {str(p.relative_to(output)): sha256_file(p) for p in sorted(output.rglob("*")) if p.is_file()}
    write_json(output / "manifest.json", manifest)
    return report


def verify_attribution(output: Path) -> dict[str, Any]:
    """Verify the diagnostic inventory, parent seal and complete-panel summary.

    Args:
        output: Explicit existing diagnostic output directory.

    Returns:
        Hash-verified report; changed/missing artifacts raise errors.
    """
    manifest = read_json(output / "manifest.json")
    checked_path(manifest["parent_manifest"])
    hashes = manifest["generated_artifact_sha256"]
    actual = {str(p.relative_to(output)) for p in output.rglob("*") if p.is_file() and p != output / "manifest.json"}
    if actual != set(hashes):
        raise ValueError(f"Attribution artifact inventory differs: {output}")
    for name, digest in hashes.items():
        checked_path({"path": str(output / name), "sha256": digest})
    report = read_json(output / "report.json")
    if summarize_attribution(report["cells"]) != report:
        raise ValueError(f"Attribution report summary differs: {output}")
    return report


def main() -> int:
    """Run the deterministic registered diagnostic command.

    Returns:
        Zero for complete, two for incomplete evidence, one for explicit errors.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    try:
        report = run_attribution(args.config, args.output)
    except (ValueError, TypeError, KeyError, OSError, RuntimeError, DataError) as exc:
        print(f"profit attribution failed: {exc}", file=sys.stderr)
        return 1
    print(f"{report['status']}: {report['completed_policy_folds']}/85; {args.output}")
    return 0 if report["status"] == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
