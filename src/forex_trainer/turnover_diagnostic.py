"""Frozen score persistence and actual trading cost attribution (Issue #21)."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

import forex_env.accounting as environment_accounting
import numpy as np
import pandas as pd
from forex_env import parse_config
from forex_env.accounting import PortfolioAccount
from forex_env.config import TransactionCostsConfig

from .artifact_provenance import git_commits, sha256_file
from .research_statistics import _bootstrap_indices
from .spread_decomposition import _match, load_seal, run_decomposition
from .supervised_portfolio import FOLDS
from .supervised_ranking_study import _study_versions

STATES: tuple[str, ...] = (
    "initial_entry",
    "entry",
    "exit",
    "reversal",
    "retained",
    "inactive",
)
ACCOUNTING_ARTIFACTS: set[str] = {
    "fold_metrics.csv",
    "steps.csv",
    "report.json",
    "report.md",
    "bootstrap_indices.npz",
    "campaign_snapshot.json",
}


def _correlation(left: np.ndarray, right: np.ndarray) -> tuple[float, str]:
    """Compute a correlation with explicit undefined status.

    Args:
        left: Finite observations.
        right: Aligned finite observations.

    Returns:
        Pearson correlation and its status; NaN only for a constant input.
    """
    if np.ptp(left) == 0 or np.ptp(right) == 0:
        return float("nan"), "constant_input"
    return float(np.corrcoef(left, right)[0, 1]), "defined"


def diagnose_fold(
    predictions: pd.DataFrame,
    pairs: pd.DataFrame,
    steps: pd.DataFrame,
    accounting: pd.DataFrame,
    symbols: tuple[str, ...],
    policy: str,
    costs: TransactionCostsConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Attribute actual trades without conflating target changes and drift.

    Args:
        predictions: Sealed scores for one complete fold in original pair order.
        pairs: Sealed effective weights and price/carry contributions.
        steps: Sealed equity and transaction cost trace.
        accounting: Independently validated Issue #19 next-decision accounting.
        symbols: Configured ordered currency pairs.
        policy: Supervised or canonical reversal score column.
        costs: Same signed-carry transaction cost contract as the source.

    Returns:
        Pair transitions, decision diagnostics, and right-censored holding spells.
    """
    context = f"{policy} turnover"
    if (
        policy not in ("supervised", "reversal")
        or len(symbols) != 9
        or len(set(symbols)) != 9
    ):
        raise ValueError(
            f"{context}: requires a fixed-map policy and nine unique pairs."
        )
    if costs.carry_mode != "signed":
        raise ValueError(f"{context}: requires signed carry.")
    times = list(steps["decision_timestamp"])
    targets = list(steps["target_timestamp"])
    if len(times) < 2 or len(set(times)) != len(times):
        raise ValueError(f"{context}: requires unique consecutive decisions.")
    dt = pd.to_datetime(times, utc=True)
    tt = pd.to_datetime(targets, utc=True)
    if (
        not dt.is_monotonic_increasing
        or not (tt > dt).all()
        or targets[:-1] != times[1:]
    ):
        raise ValueError(f"{context}: decision/target timestamps are not consecutive.")
    expected = [(d, t, p) for d, t in zip(times, targets) for p in symbols]
    if (
        list(
            predictions[["decision_timestamp", "target_timestamp", "pair"]].itertuples(
                index=False, name=None
            )
        )
        != expected
    ):
        raise ValueError(f"{context}: prediction timestamps/pair order mismatch.")
    if list(
        pairs[["decision_timestamp", "pair"]].itertuples(index=False, name=None)
    ) != [(d, p) for d in times for p in symbols]:
        raise ValueError(f"{context}: pair trace timestamps/pair order mismatch.")
    if list(
        accounting[["decision_timestamp", "target_timestamp"]].itertuples(
            index=False, name=None
        )
    ) != list(zip(times, targets)):
        raise ValueError(f"{context}: accounting timestamps mismatch.")
    shape = (len(times), len(symbols))
    scores = predictions[f"{policy}_score"].to_numpy(float).reshape(shape)
    weights = pairs["target_weight"].to_numpy(float).reshape(shape)
    price = pairs["price_simple_contribution"].to_numpy(float).reshape(shape)
    carry = pairs["carry_simple_contribution"].to_numpy(float).reshape(shape)
    gross = pairs["gross_simple_contribution"].to_numpy(float).reshape(shape)
    before, after = (
        steps["equity_before"].to_numpy(float),
        steps["equity_after"].to_numpy(float),
    )
    for name, values in (
        ("scores", scores),
        ("weights", weights),
        ("price", price),
        ("carry", carry),
        ("equity before", before),
        ("equity after", after),
    ):
        if not np.isfinite(values).all():
            raise ValueError(f"{context}: nonfinite {name}.")
    if (before <= 0).any() or (after <= 0).any():
        raise ValueError(f"{context}: nonpositive equity.")
    magnitude = float(np.float32(0.8))
    if (
        not np.isin(weights, [-magnitude, 0.0, magnitude]).all()
        or not ((weights > 0).sum(axis=1) == 2).all()
        or not ((weights < 0).sum(axis=1) == 2).all()
    ):
        raise ValueError(f"{context}: effective fixed-map geometry differs.")
    _match(before[1:], after[:-1], context + " equity continuity", 1e-12)
    _match(gross, price + carry, context + " pair gross", 1e-12)
    for name, values in (
        ("price_simple_return", price.sum(axis=1)),
        ("carry_simple_return", carry.sum(axis=1)),
        ("net_simple_return", after / before - 1),
        ("cost_ratio", steps["cost_jpy"].to_numpy(float) / before),
    ):
        _match(values, accounting[name].to_numpy(float), context + " " + name, 1e-12)
    _match(
        accounting["net_simple_return"].to_numpy(float),
        gross.sum(axis=1) - accounting["cost_ratio"].to_numpy(float),
        context + " net",
        1e-12,
    )
    previous = np.vstack([np.zeros(9), weights[:-1]])
    marked = np.vstack([np.zeros(9), before[:-1, None] * (weights[:-1] + price[:-1])])
    target_exposure = before[:, None] * weights
    trade = target_exposure - marked
    notional = np.abs(trade)
    target_change = before[:, None] * (weights - previous)
    drift = before[:, None] * previous - marked
    spreads = np.array([costs.spreads[p] for p in symbols])
    pair_spread, pair_commission = notional * spreads, notional * costs.commission_rate
    pair_trading = pair_spread + pair_commission
    elapsed_days = np.asarray((tt - dt).total_seconds(), dtype=float) / 86400
    pair_overnight = (
        np.abs(target_exposure) * costs.overnight_rate * elapsed_days[:, None]
    )
    pair_cost = pair_trading + pair_overnight
    for t in range(len(times)):
        account = PortfolioAccount(float(before[t]), symbols, costs)
        account.exposures_jpy = marked[t].copy()
        spread, commission = account.rebalance(target_exposure[t])
        # The environment charges overnight fees on pre-price target exposures.
        _, overnight = account.mark_to_market(np.ones(9), float(elapsed_days[t]))
        _match(
            pair_overnight[t].sum() / before[t],
            overnight / before[t],
            f"{context} {times[t]} environment overnight",
            1e-12,
        )
        _match(
            np.array([pair_spread[t].sum(), pair_commission[t].sum()]) / before[t],
            np.array([spread, commission]) / before[t],
            f"{context} {times[t]} environment cost",
            1e-12,
        )
    residual = pair_cost.sum(axis=1) / before - accounting["cost_ratio"].to_numpy(float)
    _match(residual, np.zeros(len(times)), context + " sealed cost", 1e-12)
    turnover = np.abs(weights - previous).sum(axis=1)
    _match(
        turnover,
        steps["weight_turnover"].to_numpy(float),
        context + " target turnover",
        1e-12,
    )
    ranks = pd.DataFrame(scores).rank(axis=1, method="average").to_numpy()
    ordinal = (
        np.argsort(np.argsort(scores, axis=1, kind="stable"), axis=1, kind="stable") + 1
    )
    _match(
        ordinal,
        pairs["predicted_rank"].to_numpy(float).reshape(shape),
        context + " ordinal ranks",
        0.0,
    )
    membership, old = np.sign(weights), np.sign(previous)
    pair_rows: list[dict[str, Any]] = []
    decisions: list[dict[str, Any]] = []
    for t in range(len(times)):
        entries = (old[t] == 0) & (membership[t] != 0)
        exits = (old[t] != 0) & (membership[t] == 0)
        flips = old[t] * membership[t] == -1
        exited = exits | flips
        ordered = np.sort(scores[t], kind="stable")
        score_corr, score_status = (
            (float("nan"), "initial_decision")
            if t == 0
            else _correlation(scores[t - 1], scores[t])
        )
        rank_corr, rank_status = (
            (float("nan"), "initial_decision")
            if t == 0
            else _correlation(ranks[t - 1], ranks[t])
        )
        decisions.append(
            {
                "decision_index": t,
                "decision_timestamp": times[t],
                "target_timestamp": targets[t],
                "state": "initial_entry"
                if t == 0
                else "membership_changed"
                if exited.any()
                else "membership_retained",
                "score_correlation": score_corr,
                "score_correlation_status": score_status,
                "rank_correlation": rank_corr,
                "rank_correlation_status": rank_status,
                "score_changed_pairs": float("nan")
                if t == 0
                else int((scores[t] != scores[t - 1]).sum()),
                "original_rank_churn": float("nan")
                if t == 0
                else float(np.abs(ranks[t] - ranks[t - 1]).mean() / 8),
                "rank_churn": float("nan")
                if t == 0
                else float(np.abs(ordinal[t] - ordinal[t - 1]).mean() / 8),
                "membership_exit_rate": float("nan")
                if t == 0
                else float(exited.sum() / 4),
                "entries": int(entries.sum()),
                "exits": int(exits.sum()),
                "reversals": int(flips.sum()),
                "bottom_boundary_gap": float(ordered[2] - ordered[1]),
                "top_boundary_gap": float(ordered[-2] - ordered[-3]),
                "weight_turnover": float(turnover[t]),
                "traded_notional_jpy": float(notional[t].sum()),
                "traded_notional_over_equity": float(notional[t].sum() / before[t]),
                "spread_jpy": float(pair_spread[t].sum()),
                "commission_jpy": float(pair_commission[t].sum()),
                "trading_cost_jpy": float(pair_trading[t].sum()),
                "overnight_jpy": float(pair_overnight[t].sum()),
                "cost_jpy": float(pair_cost[t].sum()),
                "cost_ratio": float(pair_cost[t].sum() / before[t]),
                "cost_residual_ratio": float(residual[t]),
                "price_simple_return": float(price[t].sum()),
                "carry_simple_return": float(carry[t].sum()),
                "gross_simple_return": float(gross[t].sum()),
                "net_simple_return": float(accounting.iloc[t]["net_simple_return"]),
            }
        )
        for p, symbol in enumerate(symbols):
            if entries[p]:
                state = "initial_entry" if t == 0 else "entry"
            elif exits[p]:
                state = "exit"
            elif flips[p]:
                state = "reversal"
            else:
                state = "retained" if membership[t, p] else "inactive"
            pair_rows.append(
                {
                    "decision_index": t,
                    "decision_timestamp": times[t],
                    "target_timestamp": targets[t],
                    "pair": symbol,
                    "state": state,
                    "score": float(scores[t, p]),
                    "score_changed": None
                    if t == 0
                    else bool(scores[t, p] != scores[t - 1, p]),
                    "rank": int(ordinal[t, p]),
                    "target_weight": float(weights[t, p]),
                    "previous_weight": float(previous[t, p]),
                    "marked_exposure_jpy": float(marked[t, p]),
                    "signed_trade_jpy": float(trade[t, p]),
                    "signed_target_change_jpy": float(target_change[t, p]),
                    "signed_drift_jpy": float(drift[t, p]),
                    "traded_notional_jpy": float(notional[t, p]),
                    "spread_jpy": float(pair_spread[t, p]),
                    "commission_jpy": float(pair_commission[t, p]),
                    "trading_cost_jpy": float(pair_trading[t, p]),
                    "overnight_jpy": float(pair_overnight[t, p]),
                    "cost_jpy": float(pair_cost[t, p]),
                    "cost_ratio": float(pair_cost[t, p] / before[t]),
                    "price_simple_contribution": float(price[t, p]),
                    "carry_simple_contribution": float(carry[t, p]),
                    "gross_simple_contribution": float(gross[t, p]),
                    "net_simple_contribution": float(
                        gross[t, p] - pair_cost[t, p] / before[t]
                    ),
                }
            )
    spells: list[dict[str, Any]] = []
    for p, symbol in enumerate(symbols):
        start = 0
        while start < len(times):
            side = membership[start, p]
            end = start + 1
            while end < len(times) and membership[end, p] == side:
                end += 1
            if side:
                spells.append(
                    {
                        "pair": symbol,
                        "side": "long" if side > 0 else "short",
                        "start_timestamp": times[start],
                        "last_decision_timestamp": times[end - 1],
                        "end_timestamp": targets[end - 1],
                        "duration_decisions": end - start,
                        "right_censored": end == len(times),
                    }
                )
            start = end
    return pd.DataFrame(pair_rows), pd.DataFrame(decisions), pd.DataFrame(spells)


def _read_frame(path: Path) -> pd.DataFrame:
    """Read sealed numeric values without precision loss.

    Args:
        path: Explicit CSV path.

    Returns:
        Original ordered rows with string fold identifiers.
    """
    return pd.read_csv(path, dtype={"fold": str}, float_precision="round_trip")


def _write_json(path: Path, value: dict[str, Any]) -> None:
    """Write strict JSON without nonfinite fallback values.

    Args:
        path: Destination file.
        value: JSON object with explicit null for undefined observations.
    """
    path.write_text(
        json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


def _evidence(values: np.ndarray, iid: np.ndarray, block: np.ndarray) -> dict[str, Any]:
    """Summarize the complete 17-fold vector with shared bootstrap indices.

    Args:
        values: One finite value per chronological fold.
        iid: Shared IID index draws.
        block: Shared circular moving-block draws.

    Returns:
        Equal-fold mean, both eras, intervals, and leave-one-fold-out means.
    """
    if values.shape != (17,) or not np.isfinite(values).all():
        raise ValueError("fold evidence requires exactly 17 finite observations.")
    return {
        "mean": float(values.mean()),
        "eras": {
            "2009-2018": float(values[:10].mean()),
            "2019-2025": float(values[10:].mean()),
        },
        "iid": np.quantile(values[iid].mean(axis=1), [0.025, 0.975]).tolist(),
        "moving_block": np.quantile(
            values[block].mean(axis=1), [0.025, 0.975]
        ).tolist(),
        "leave_one_fold_out": {
            fold: float(np.delete(values, i).mean()) for i, fold in enumerate(FOLDS)
        },
        "minimum_fold": float(values.min()),
        "maximum_fold": float(values.max()),
    }


def _state_summary(frame: pd.DataFrame) -> dict[str, Any]:
    """Aggregate exploratory state returns with explicit observed-fold counts.

    Args:
        frame: All fold/state rows, including zero-count states.

    Returns:
        Equal observed-fold conditional means for each state and era.
    """
    result: dict[str, Any] = {}
    for policy in ("supervised", "reversal"):
        result[policy] = {}
        for state in ("initial_entry", "membership_changed", "membership_retained"):
            result[policy][state] = {}
            for era in ("2009-2018", "2019-2025"):
                rows = frame.loc[
                    (frame.policy == policy)
                    & (frame.state == state)
                    & (frame.era == era)
                ]
                observed = rows.loc[rows.decisions > 0]
                result[policy][state][era] = {
                    "total_folds": len(rows),
                    "observed_folds": len(observed),
                    "decisions": int(rows.decisions.sum()),
                    **{
                        column: float(observed[column].mean())
                        if len(observed)
                        else None
                        for column in (
                            "mean_price_simple_return",
                            "mean_carry_simple_return",
                            "mean_gross_simple_return",
                            "mean_net_simple_return",
                            "mean_cost_ratio",
                        )
                    },
                }
    return result


def _render(report: dict[str, Any]) -> str:
    """Render the diagnostic and its interpretive limits.

    Args:
        report: Completed fold-level report.

    Returns:
        Markdown report with primary and descriptive cause summaries.
    """
    lines = [
        "# Frozen turnover/cost diagnostic — Issue #21",
        "",
        "Classification: **" + report["classification"] + "** (unchanged).",
        "Decision: **unresolved**. Gross uncertainty remains; turnover reduction does not establish profitability.",
        "",
        "Primary metrics use equal fold weights. Intervals: 10,000 IID / 3-fold circular moving-block draws, seed 16.",
        "Score/rank persistence excludes the initial decision; membership exit rate excludes initial entry.",
        "",
        "| Policy / metric | Mean | 2009–2018 | 2019–2025 | IID 95% | Moving-block 95% |",
        "|---|---:|---:|---:|---|---|",
    ]
    for policy, metrics in report["primary"].items():
        for name, evidence in metrics.items():
            if evidence["status"] != "defined":
                lines.append(
                    f"| {policy} / {name} | undefined ({evidence['observed_folds']}/17 folds) | — | — | — | — |"
                )
                continue
            lines.append(
                f"| {policy} / {name} | {evidence['mean']:.8g} | {evidence['eras']['2009-2018']:.8g} | {evidence['eras']['2019-2025']:.8g} | {evidence['iid']} | {evidence['moving_block']} |"
            )
    lines += [
        "",
        "## Cause attribution (descriptive)",
        "",
        "Cost shares are means of within-fold shares, not pooled JPY amounts. Total cost = spread + commission + overnight; signed carry is a separate contribution. JSON and CSV split trading/overnight costs.",
        "Retained-target trades include price drift and equity changes due to price, carry and costs.",
        "",
        "| Policy | Cause | Mean total cost share | Share of trading cost | 2009–2018 total | 2019–2025 total |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for policy, states in report["causes"].items():
        for state, summary in states.items():
            lines.append(
                f"| {policy} | {state} | {summary['mean_cost_share']:.6%} | {summary['mean_share_of_trading_cost']:.6%} | {summary['era_cost_shares']['2009-2018']:.6%} | {summary['era_cost_shares']['2019-2025']:.6%} |"
            )
    lines += [
        "",
        f"Maximum absolute sealed-cost residual / starting equity: {report['max_abs_cost_residual_ratio']:.4g}.",
        "",
        "## Outputs and limits",
        "",
        "- pair_transitions.csv: signed target change, drift, actual trade, state and next-decision price/carry/gross/net simple contributions.",
        "- decisions.csv: persistence, boundary gaps, membership events, actual notional/equity and accounting residuals.",
        "- holding_spells.csv / holding_duration_counts.csv: long/short observed durations with right-censoring at every fold end; no cross-fold stitching.",
        "- fold_metrics.csv / cause_metrics.csv: primary fold metrics, exhaustive cost attribution and descriptive next-period state contributions.",
        "- exploratory_states.csv: membership-maintained/changed next-decision return associations, including zero-count states.",
        "- exploratory_boundary.csv: contemporaneous boundary-gap/cost and boundary-gap/exit-rate correlations, with undefined counts/status.",
        "- report.json: both eras, shared fold intervals, all LOO means, undefined counts and original Issue #19 economics (including PPO).",
        "",
        "State comparisons and boundary correlations are descriptive, not causal execution effects. No horizon, top-k, threshold, allocation or policy was selected.",
        "Completed-only holding durations do not estimate unconditional persistence; censored observations remain explicit.",
        "No unexplained cost beyond the recorded residual tolerance. Real-world slippage/market impact is absent from this environment; it is not estimated here.",
        "A gate, bandit or low-turnover policy is not justified by these associations. Any next experiment requires a separate preregistration under Issue #20 / ADR-0031.",
        "",
    ]
    return "\n".join(lines)


def run_diagnostic(
    campaign_path: Path, output_dir: Path
) -> tuple[Path, dict[str, Any]]:
    """Verify the frozen accounting chain and publish turnover diagnostics.

    Args:
        campaign_path: Manifest pinning the Issue #19 campaign and output seal.
        output_dir: New destination directory, never overwritten.

    Returns:
        Published directory and report.
    """
    destination = output_dir.resolve()
    if destination.exists():
        raise FileExistsError(f"output directory already exists: {destination}")
    campaign_path = campaign_path.resolve()
    campaign = json.loads(campaign_path.read_text(encoding="utf-8"))
    if not isinstance(campaign, dict) or set(campaign) != {
        "accounting_campaign",
        "accounting_source",
    }:
        raise ValueError(
            f"campaign requires explicit accounting_campaign and accounting_source: {campaign_path}"
        )
    for key, fields in (
        ("accounting_campaign", {"path", "sha256"}),
        ("accounting_source", {"directory", "provenance_sha256"}),
    ):
        if (
            not isinstance(campaign[key], dict)
            or set(campaign[key]) != fields
            or any(not isinstance(v, str) or not v for v in campaign[key].values())
        ):
            raise ValueError(
                f"campaign {key} requires {sorted(fields)}: {campaign_path}"
            )
    accounting_campaign = (
        campaign_path.parent / campaign["accounting_campaign"]["path"]
    ).resolve()
    if sha256_file(accounting_campaign) != campaign["accounting_campaign"]["sha256"]:
        raise ValueError(f"accounting campaign hash mismatch: {accounting_campaign}")
    source_dir = (
        campaign_path.parent / campaign["accounting_source"]["directory"]
    ).resolve()
    seal = load_seal(
        source_dir,
        campaign["accounting_source"]["provenance_sha256"],
        ACCOUNTING_ARTIFACTS,
    )
    if seal["campaign_sha256"] != sha256_file(accounting_campaign):
        raise ValueError("accounting campaign differs from selected Issue #19 source.")
    source_campaign = json.loads(accounting_campaign.read_text(encoding="utf-8"))
    source_paths: dict[str, Path] = {}
    for name in ("ranking", "portfolio"):
        selected = source_campaign[name]
        source_paths[name] = (
            accounting_campaign.parent / selected["directory"]
        ).resolve()
        if selected["provenance_sha256"] != seal["sources"][name]["provenance_sha256"]:
            raise ValueError(f"Issue #19 {name} seal differs from campaign.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}-", dir=destination.parent)
    )
    try:
        verified_dir, accounting_report = run_decomposition(
            accounting_campaign, staging / "accounting_verification"
        )
        for name in ("steps.csv", "fold_metrics.csv"):
            original, current = (
                _read_frame(source_dir / name),
                _read_frame(verified_dir / name),
            )
            try:
                pd.testing.assert_frame_equal(
                    original, current, check_exact=False, rtol=1e-10, atol=1e-12
                )
            except AssertionError as error:
                raise ValueError(
                    f"Issue #19 regenerated {name} differs: {error}"
                ) from error
        verified_seal = json.loads((verified_dir / "provenance.json").read_text())
        if verified_seal["source_fold_provenance"] != seal["source_fold_provenance"]:
            raise ValueError("Issue #19 original model/config/data provenance differs.")
        predictions = _read_frame(source_paths["ranking"] / "predictions.csv")
        pairs = _read_frame(source_paths["portfolio"] / "pair_contributions.csv")
        steps = _read_frame(source_paths["portfolio"] / "steps.csv")
        originals = _read_frame(source_paths["portfolio"] / "fold_metrics.csv")
        ranking_metrics = _read_frame(source_paths["ranking"] / "fold_metrics.csv")
        accounting = _read_frame(verified_dir / "steps.csv")
        all_pairs: list[pd.DataFrame] = []
        all_decisions: list[pd.DataFrame] = []
        all_spells: list[pd.DataFrame] = []
        fold_rows: list[dict[str, Any]] = []
        cause_rows: list[dict[str, Any]] = []
        state_rows: list[dict[str, Any]] = []
        boundary_rows: list[dict[str, Any]] = []
        for fold in FOLDS:
            print(
                f"turnover fold {fold}: attributing marked exposures and costs",
                flush=True,
            )
            env = parse_config(
                seal["source_fold_provenance"]["portfolio"][fold]["resolved_eval_env"]
            )
            era = "2009-2018" if int(fold) <= 2018 else "2019-2025"
            for policy in ("supervised", "reversal"):
                selected = (steps.fold == fold) & (steps.policy == policy)
                fold_steps = steps.loc[selected]
                try:
                    pair_trace, decisions, spells = diagnose_fold(
                        predictions.loc[predictions.fold == fold],
                        pairs.loc[(pairs.fold == fold) & (pairs.policy == policy)],
                        fold_steps,
                        accounting.loc[
                            (accounting.fold == fold) & (accounting.policy == policy)
                        ],
                        env.environment.currency_pairs,
                        policy,
                        env.transaction_costs,
                    )
                except ValueError as error:
                    raise ValueError(f"fold {fold}: {error}") from error
                identity = {"fold": fold, "era": era, "policy": policy}
                for frame, collection in (
                    (pair_trace, all_pairs),
                    (decisions, all_decisions),
                    (spells, all_spells),
                ):
                    collection.append(frame.assign(**identity))
                total_cost = float(decisions.cost_jpy.sum())
                total_notional = float(decisions.traded_notional_jpy.sum())
                if total_cost <= 0 or total_notional <= 0:
                    raise ValueError(
                        f"fold {fold} {policy}: cost/notional shares require positive totals."
                    )
                metric: dict[str, Any] = {
                    **identity,
                    "decisions": len(decisions),
                    "total_cost_jpy": total_cost,
                    "total_traded_notional_jpy": total_notional,
                    "trading_cost_share": float(
                        decisions.trading_cost_jpy.sum() / total_cost
                    ),
                    "overnight_cost_share": float(
                        decisions.overnight_jpy.sum() / total_cost
                    ),
                    "total_cost_ratio": total_cost
                    / float(fold_steps.iloc[0].equity_before),
                    "mean_weight_turnover": float(decisions.weight_turnover.mean()),
                    "mean_membership_turnover": float(
                        decisions.membership_exit_rate.iloc[1:].mean()
                    ),
                    "mean_rank_churn": float(
                        decisions.original_rank_churn.iloc[1:].mean()
                    ),
                    "mean_actual_notional_over_equity": float(
                        decisions.traded_notional_over_equity.mean()
                    ),
                    "mean_reversals": float(decisions.reversals.iloc[1:].mean()),
                    "mean_bottom_boundary_gap": float(
                        decisions.bottom_boundary_gap.mean()
                    ),
                    "mean_top_boundary_gap": float(decisions.top_boundary_gap.mean()),
                    "mean_score_changed_pairs": float(
                        decisions.score_changed_pairs.iloc[1:].mean()
                    ),
                    "completed_spells": int((~spells.right_censored).sum()),
                    "censored_spells": int(spells.right_censored.sum()),
                }
                for correlation in ("score_correlation", "rank_correlation"):
                    valid = decisions[f"{correlation}_status"] == "defined"
                    metric[f"{correlation}_observations"] = int(valid.sum())
                    metric[f"{correlation}_undefined"] = int(
                        (decisions[f"{correlation}_status"] == "constant_input").sum()
                    )
                    metric[f"mean_{correlation}"] = (
                        float(decisions.loc[valid, correlation].mean())
                        if valid.any()
                        else None
                    )
                original = originals.loc[
                    (originals.fold == fold) & (originals.policy == policy)
                ].iloc[0]
                reproduced = ["total_cost_ratio", "mean_weight_turnover"]
                if policy == "supervised":
                    reproduced.extend(["mean_membership_turnover", "mean_rank_churn"])
                elif (
                    not original[["mean_membership_turnover", "mean_rank_churn"]]
                    .isna()
                    .all()
                ):
                    raise ValueError(
                        f"fold {fold}: unexpected canonical Issue #16 diagnostic schema."
                    )
                ranking_metric = ranking_metrics.loc[
                    (ranking_metrics.fold == fold) & (ranking_metrics.score == policy)
                ].iloc[0]
                _match(
                    metric["mean_rank_churn"],
                    float(ranking_metric.mean_rank_churn),
                    f"fold {fold} {policy} Issue #15 rank churn",
                    1e-12,
                )
                for key in reproduced:
                    _match(
                        metric[key],
                        float(original[key]),
                        f"fold {fold} {policy} original {key}",
                        1e-12,
                    )
                fold_rows.append(metric)
                for state in STATES:
                    rows = pair_trace.loc[pair_trace.state == state]
                    cause: dict[str, Any] = {
                        **identity,
                        "state": state,
                        "pair_observations": len(rows),
                    }
                    for column in (
                        "traded_notional_jpy",
                        "spread_jpy",
                        "commission_jpy",
                        "trading_cost_jpy",
                        "overnight_jpy",
                        "cost_jpy",
                    ):
                        cause[column] = float(rows[column].sum())
                    cause["cost_share"] = cause["cost_jpy"] / total_cost
                    cause["trading_cost_share_of_total"] = (
                        cause["trading_cost_jpy"] / total_cost
                    )
                    cause["overnight_share_of_total"] = (
                        cause["overnight_jpy"] / total_cost
                    )
                    cause["share_of_trading_cost"] = cause["trading_cost_jpy"] / float(
                        decisions.trading_cost_jpy.sum()
                    )
                    cause["notional_share"] = (
                        cause["traded_notional_jpy"] / total_notional
                    )
                    for column in (
                        "price_simple_contribution",
                        "carry_simple_contribution",
                        "gross_simple_contribution",
                        "net_simple_contribution",
                    ):
                        cause[f"mean_{column}"] = (
                            float(rows[column].mean()) if len(rows) else None
                        )
                    cause_rows.append(cause)
                for state in (
                    "initial_entry",
                    "membership_changed",
                    "membership_retained",
                ):
                    rows = decisions.loc[decisions.state == state]
                    state_rows.append(
                        {
                            **identity,
                            "state": state,
                            "decisions": len(rows),
                            **{
                                f"mean_{column}": float(rows[column].mean())
                                if len(rows)
                                else None
                                for column in (
                                    "price_simple_return",
                                    "carry_simple_return",
                                    "gross_simple_return",
                                    "net_simple_return",
                                    "cost_ratio",
                                )
                            },
                        }
                    )
                for gap in ("bottom_boundary_gap", "top_boundary_gap"):
                    for response in ("membership_exit_rate", "cost_ratio"):
                        correlation, status = _correlation(
                            decisions[gap].to_numpy()[1:],
                            decisions[response].to_numpy()[1:],
                        )
                        boundary_rows.append(
                            {
                                **identity,
                                "gap": gap,
                                "response": response,
                                "correlation": correlation,
                                "status": status,
                                "decisions": len(decisions) - 1,
                            }
                        )
        folds = pd.DataFrame(fold_rows)
        causes = pd.DataFrame(cause_rows)
        decision_frame = pd.concat(all_decisions, ignore_index=True)
        spell_frame = pd.concat(all_spells, ignore_index=True)
        iid, block = _bootstrap_indices(17, 10000, 16, 3)
        primary: dict[str, Any] = {}
        cause_summary: dict[str, Any] = {}
        for policy in ("supervised", "reversal"):
            selected_folds = folds.loc[folds.policy == policy]
            primary[policy] = {}
            for metric in (
                "mean_weight_turnover",
                "mean_membership_turnover",
                "mean_actual_notional_over_equity",
                "total_cost_ratio",
                "trading_cost_share",
                "overnight_cost_share",
                "mean_score_correlation",
                "mean_rank_correlation",
                "mean_reversals",
                "mean_bottom_boundary_gap",
                "mean_top_boundary_gap",
                "mean_rank_churn",
                "mean_score_changed_pairs",
            ):
                values = selected_folds[metric].to_numpy(float)
                observed = int(np.isfinite(values).sum())
                primary[policy][metric] = {
                    "status": "defined" if observed == 17 else "undefined_fold_values",
                    "observed_folds": observed,
                }
                if observed == 17:
                    primary[policy][metric].update(_evidence(values, iid, block))
            cause_summary[policy] = {}
            for state in STATES:
                rows = causes.loc[(causes.policy == policy) & (causes.state == state)]
                cause_summary[policy][state] = {
                    "mean_cost_share": float(rows.cost_share.mean()),
                    "mean_notional_share": float(rows.notional_share.mean()),
                    "mean_trading_cost_share_of_total": float(
                        rows.trading_cost_share_of_total.mean()
                    ),
                    "mean_overnight_share_of_total": float(
                        rows.overnight_share_of_total.mean()
                    ),
                    "mean_share_of_trading_cost": float(
                        rows.share_of_trading_cost.mean()
                    ),
                    "observed_folds": int((rows.pair_observations > 0).sum()),
                    "era_trading_cost_shares": {
                        era: float(
                            rows.loc[rows.era == era, "share_of_trading_cost"].mean()
                        )
                        for era in ("2009-2018", "2019-2025")
                    },
                    "era_cost_shares": {
                        era: float(rows.loc[rows.era == era, "cost_share"].mean())
                        for era in ("2009-2018", "2019-2025")
                    },
                }
        report: dict[str, Any] = {
            "classification": accounting_report["classification"],
            "decision": "unresolved",
            "reason": "Gross uncertainty remains under Issue #19; descriptive turnover associations do not justify a gate/bandit or establish profitability.",
            "protocol": {
                "adr": "0033 (supersedes 0032)",
                "horizon": "original next decision",
                "fold_weight": "equal",
                "folds": list(FOLDS),
                "bootstrap_samples": 10000,
                "seed": 16,
                "moving_block_length": 3,
                "causes": "descriptive exhaustive pair states",
                "exploratory": [
                    "membership-state next-period outcomes",
                    "boundary-gap correlations",
                ],
                "spells": "right censored at fold end; starts from initial flat account",
                "rank_churn": "ordinal rank_churn and original tied-average original_rank_churn are separately recorded",
            },
            "primary": primary,
            "causes": cause_summary,
            "exploratory_states": _state_summary(pd.DataFrame(state_rows)),
            "max_abs_cost_residual_ratio": float(
                decision_frame.cost_residual_ratio.abs().max()
            ),
            "undefined_correlations": {
                policy: {
                    name: int(folds.loc[folds.policy == policy, name].sum())
                    for name in (
                        "score_correlation_undefined",
                        "rank_correlation_undefined",
                    )
                }
                for policy in ("supervised", "reversal")
            },
            "original_metrics": accounting_report["original_metrics"],
            "original_reproduction": {
                "both_policies": [
                    "Issue #16 total_cost_ratio",
                    "Issue #16 mean_weight_turnover",
                    "Issue #15 mean_rank_churn",
                ],
                "supervised_only": [
                    "Issue #16 mean_membership_turnover",
                    "Issue #16 mean_rank_churn",
                ],
                "source_absence": "Issue #16 did not store canonical membership/rank diagnostic columns; canonical membership is newly measured with the same definition.",
            },
            "limitations": [
                "No independent positive gross established",
                "State associations are not causal execution effects",
                "Slippage/market impact absent from this environment",
                "Observed duration includes right censoring; no completed-only unconditional mean",
            ],
        }
        frames = {
            "pair_transitions.csv": pd.concat(all_pairs, ignore_index=True),
            "decisions.csv": decision_frame,
            "holding_spells.csv": spell_frame,
            "holding_duration_counts.csv": spell_frame.groupby(
                [
                    "fold",
                    "era",
                    "policy",
                    "side",
                    "duration_decisions",
                    "right_censored",
                ],
                sort=False,
            )
            .size()
            .rename("spells")
            .reset_index(),
            "fold_metrics.csv": folds,
            "cause_metrics.csv": causes,
            "exploratory_states.csv": pd.DataFrame(state_rows),
            "exploratory_boundary.csv": pd.DataFrame(boundary_rows),
        }
        for name, frame in frames.items():
            frame.to_csv(staging / name, index=False)
        np.savez_compressed(
            staging / "bootstrap_indices.npz", iid=iid, moving_block=block
        )
        _write_json(staging / "report.json", report)
        (staging / "report.md").write_text(_render(report), encoding="utf-8")
        shutil.copyfile(campaign_path, staging / "campaign_snapshot.json")
        shutil.rmtree(verified_dir)
        provenance = {
            "artifact_version": 1,
            "campaign_sha256": sha256_file(campaign_path),
            "source_accounting_provenance_sha256": campaign["accounting_source"][
                "provenance_sha256"
            ],
            "source_accounting_artifact_sha256": seal["generated_artifact_sha256"],
            "sources": seal["sources"],
            "source_fold_provenance": seal["source_fold_provenance"],
            "accounting_verification": verified_seal,
            "diagnostic_git": git_commits(),
            "diagnostic_versions": dict(_study_versions()),
            "implementation_sha256": sha256_file(Path(__file__)),
            "environment_accounting_sha256": sha256_file(
                Path(environment_accounting.__file__)
            ),
            "generated_artifact_sha256": {
                p.name: sha256_file(p) for p in staging.iterdir()
            },
        }
        _write_json(staging / "provenance.json", provenance)
        os.replace(staging, destination)
    except Exception:
        shutil.rmtree(staging)
        raise
    return destination, report


def main() -> int:
    """Run the frozen diagnostic CLI.

    Returns:
        Zero after successful atomic publication.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output, _ = run_diagnostic(args.campaign, args.output_dir)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
