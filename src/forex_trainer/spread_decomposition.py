"""Sealed, fold-level spread-to-net accounting diagnostic (Issue #19)."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
from collections.abc import Mapping
from dataclasses import asdict
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from forex_env import parse_config as parse_env_config
from forex_env.data import build_provider

from .artifact_provenance import data_identity_from_config, git_commits, sha256_file
from .config import load_experiment_config, resolve_env_raw
from .research_statistics import _bootstrap_indices, bootstrap_mean_intervals
from .supervised_portfolio import (
    FOLDS,
    SOURCE_ARTIFACTS,
    fixed_portfolio_weights,
    paired_evidence,
    prediction_matrices,
)
from .supervised_portfolio_study import _canonical_weights
from .supervised_ranking_study import _study_versions, build_dataset_from_env_raw

POLICIES: tuple[str, ...] = ("supervised", "reversal", "ppo")
STAGES: tuple[str, ...] = (
    "scaled_tail_spread",
    "weighted_price_log_return",
    "price_log_return",
    "gross_log_return",
    "net_log_return",
)
PORTFOLIO_ARTIFACTS: set[str] = {
    "campaign_snapshot.json",
    "fold_metrics.csv",
    "steps.csv",
    "pair_contributions.csv",
    "rank_contributions.csv",
    "report.json",
    "report.md",
}


def _match(
    actual: np.ndarray | float, expected: np.ndarray | float, context: str, atol: float
) -> None:
    """Require finite matching accounting quantities.

    Args:
        actual: Reconstructed values.
        expected: Independently sealed or calculated values.
        context: Origin and quantity for an actionable error.
        atol: Absolute error bound in the quantity's units.
    """
    a, b = np.asarray(actual), np.asarray(expected)
    if (
        a.shape != b.shape
        or not np.isfinite(a).all()
        or not np.isfinite(b).all()
        or not np.allclose(a, b, rtol=1e-10, atol=atol)
    ):
        if a.shape != b.shape:
            detail = f"shape {a.shape} != {b.shape}"
        else:
            bad = (
                ~np.isclose(a, b, rtol=1e-10, atol=atol)
                | ~np.isfinite(a)
                | ~np.isfinite(b)
            )
            index = tuple(np.argwhere(bad)[0])
            detail = f"index {index}: actual={a[index]!r}, expected={b[index]!r}"
        raise ValueError(
            f"{context}: accounting/value mismatch; {detail} (rtol=1e-10, atol={atol})."
        )


def load_seal(directory: Path, digest: str, required: set[str]) -> dict[str, Any]:
    """Verify an explicit source seal and every named artifact.

    Args:
        directory: Explicit source directory.
        digest: Externally pinned SHA-256 of provenance.json.
        required: Exact artifact names required by this source version.

    Returns:
        Verified provenance, retaining the original model/config/data chain.
    """
    path = directory / "provenance.json"
    if sha256_file(path) != digest:
        raise ValueError(f"source provenance hash mismatch: {path}")
    seal = json.loads(path.read_text(encoding="utf-8"))
    if (
        seal["artifact_version"] != 1
        or set(seal["generated_artifact_sha256"]) != required
    ):
        raise ValueError(f"source artifact schema mismatch: {path}")
    for name, expected in seal["generated_artifact_sha256"].items():
        if sha256_file(directory / name) != expected:
            raise ValueError(f"source artifact hash mismatch: {directory / name}")
    return seal


def decompose_fold(
    predictions: pd.DataFrame,
    steps: pd.DataFrame,
    pairs: pd.DataFrame,
    symbols: tuple[str, ...],
    price_relatives: np.ndarray,
    carry_rates: np.ndarray,
    initial_equity: float,
    policy: str,
    expected_weights: np.ndarray,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Reconcile one complete policy/fold against independent market values.

    Args:
        predictions: Ordered source predictions for the complete fold.
        steps: Ordered sealed policy steps.
        pairs: Ordered sealed policy pair contributions.
        symbols: Canonical configured pair order.
        price_relatives: Market close at target divided by decision close.
        carry_rates: Raw signed annual carry at each decision.
        initial_equity: Configured starting equity.
        policy: Explicit supervised, reversal, or PPO source score name.
        expected_weights: Independently reconstructed effective portfolio weights.

    Returns:
        Per-step accounting and fold metrics in explicit original/annual-log units.
    """
    context = f"{policy} fold"
    n = len(steps)
    shape = (n, len(symbols))
    if policy not in POLICIES or n == 0 or len(symbols) != 9 or len(set(symbols)) != 9:
        raise ValueError(f"{context}: invalid policy/decision/pair count.")
    for name, matrix in (
        ("price", price_relatives),
        ("carry", carry_rates),
        ("weight", expected_weights),
    ):
        if matrix.shape != shape or not np.isfinite(matrix).all():
            raise ValueError(f"{context}: invalid {name} shape/values.")
    if (
        np.any(price_relatives <= 0)
        or not np.isfinite(initial_equity)
        or initial_equity <= 0
    ):
        raise ValueError(f"{context}: prices and initial equity must be positive.")
    decisions = steps["decision_timestamp"].tolist()
    targets = steps["target_timestamp"].tolist()
    d = pd.to_datetime(decisions, utc=True)
    t = pd.to_datetime(targets, utc=True)
    if (
        d.hasnans
        or t.hasnans
        or not d.is_monotonic_increasing
        or d.has_duplicates
        or not (t > d).all()
        or decisions[1:] != targets[:-1]
    ):
        raise ValueError(
            f"{context}: discontinuous or unordered decision/target timestamps."
        )
    expected_keys = [
        (decision, target, symbol)
        for decision, target in zip(decisions, targets)
        for symbol in symbols
    ]
    if (
        list(
            predictions[["decision_timestamp", "target_timestamp", "pair"]].itertuples(
                index=False, name=None
            )
        )
        != expected_keys
    ):
        raise ValueError(
            f"{context}: prediction timestamps/pair order differ from steps."
        )
    if list(
        pairs[["decision_timestamp", "pair"]].itertuples(index=False, name=None)
    ) != [(a, c) for a, _, c in expected_keys]:
        raise ValueError(
            f"{context}: contribution timestamps/pair order differ from steps."
        )
    scores = predictions[f"{policy}_score"].to_numpy(dtype=float).reshape(shape)
    if not np.isfinite(scores).all():
        raise ValueError(f"{context}: nonfinite scores.")
    log_prices = np.log(price_relatives)
    relative_targets = (
        predictions["target_relative_log_return"].to_numpy(dtype=float).reshape(shape)
    )
    _match(
        relative_targets,
        log_prices - log_prices.mean(axis=1, keepdims=True),
        f"{context} relative targets",
        1e-12,
    )
    order = np.argsort(scores, axis=1, kind="stable")
    ranks = np.empty(shape, dtype=int)
    np.put_along_axis(ranks, order, np.broadcast_to(np.arange(1, 10), shape), axis=1)
    _match(
        pairs["predicted_rank"].to_numpy().reshape(shape),
        ranks,
        f"{context} predicted ranks",
        0.0,
    )
    weights = pairs["target_weight"].to_numpy(dtype=float).reshape(shape)
    _match(weights, expected_weights, f"{context} effective weights", 0.0)
    if policy == "supervised":
        _match(
            weights,
            fixed_portfolio_weights(scores).astype(float),
            f"{context} fixed map",
            0.0,
        )
    elapsed_days = (t - d).total_seconds().to_numpy() / 86400
    price_pair = weights * (price_relatives - 1)
    carry_pair = -weights * price_relatives * carry_rates * elapsed_days[:, None] / 365
    price = price_pair.sum(axis=1)
    carry = carry_pair.sum(axis=1)
    gross = price + carry
    if np.any(price <= -1) or np.any(gross <= -1):
        raise ValueError(
            f"{context}: nonpositive price/gross factor prevents log decomposition."
        )
    gross_log = np.log1p(gross)
    scale = np.ones(n)
    np.divide(gross_log, gross, out=scale, where=gross != 0)
    for column, expected in (
        ("price_simple_contribution", price_pair),
        ("carry_simple_contribution", carry_pair),
        ("gross_simple_contribution", price_pair + carry_pair),
        ("gross_log_contribution", (price_pair + carry_pair) * scale[:, None]),
    ):
        _match(
            pairs[column].to_numpy(dtype=float).reshape(shape),
            expected,
            f"{context} {column}",
            1e-12,
        )
    before = steps["equity_before"].to_numpy(dtype=float)
    after = steps["equity_after"].to_numpy(dtype=float)
    costs = steps["cost_jpy"].to_numpy(dtype=float)
    if (
        not np.isfinite(before).all()
        or not np.isfinite(after).all()
        or np.any(before <= 0)
        or np.any(after <= 0)
        or not np.isfinite(costs).all()
        or np.any(costs < 0)
    ):
        raise ValueError(f"{context}: invalid equity/cost values.")
    _match(
        before / initial_equity,
        np.r_[initial_equity, after[:-1]] / initial_equity,
        f"{context} equity continuity",
        1e-12,
    )
    cost_ratio = costs / before
    net = after / before - 1
    _match(
        steps["financing_jpy"].to_numpy(dtype=float) / before,
        carry,
        f"{context} signed financing",
        1e-12,
    )
    _match(net, gross - cost_ratio, f"{context} price+carry-cost=net", 1e-12)
    _match((after + costs) / before - 1, gross, f"{context} gross equity", 1e-12)
    _match(
        steps["gross_log_return"].to_numpy(dtype=float),
        gross_log,
        f"{context} gross log",
        1e-12,
    )
    net_log = np.log(after / before)
    _match(
        steps["net_log_return"].to_numpy(dtype=float),
        net_log,
        f"{context} net log",
        1e-12,
    )
    ranked_targets = np.take_along_axis(relative_targets, order, axis=1)
    tail = ranked_targets[:, -2:].mean(axis=1) - ranked_targets[:, :2].mean(axis=1)
    weighted_log = (weights * log_prices).sum(axis=1)
    if policy == "supervised":
        _match(
            weighted_log,
            2 * float(np.float32(0.8)) * tail,
            f"{context} weighted tail identity",
            1e-12,
        )
    years = float((t[-1] - d[0]).total_seconds() / (365.25 * 86400))
    rows = pd.DataFrame(
        {
            "decision_timestamp": decisions,
            "target_timestamp": targets,
            "scaled_tail_spread": 1.6 * tail,
            "weighted_price_log_return": weighted_log,
            "price_simple_return": price,
            "carry_simple_return": carry,
            "cost_ratio": cost_ratio,
            "net_simple_return": net,
            "price_log_return": np.log1p(price),
            "gross_log_return": gross_log,
            "net_log_return": net_log,
            "accounting_residual": net - (price + carry - cost_ratio),
        }
    )
    metrics = {
        "years": years,
        "decisions": n,
        "mean_tail_spread": float(tail.mean()),
        "annualized_gross_return": float(np.expm1(gross_log.sum() / years)),
        "annualized_net_return": float(np.expm1(net_log.sum() / years)),
        "gross_cumulative_log_return": float(gross_log.sum()),
        "cumulative_log_return": float(net_log.sum()),
        "total_cost_ratio": float(costs.sum() / initial_equity),
        "max_abs_accounting_residual": float(np.abs(rows["accounting_residual"]).max()),
    }
    metrics.update({stage: float(rows[stage].sum() / years) for stage in STAGES})
    return rows, metrics


def _json(path: Path) -> dict[str, Any]:
    """Read an explicit JSON object.

    Args:
        path: Source path.

    Returns:
        Parsed object.
    """
    return json.loads(path.read_text(encoding="utf-8"))


def _verify_file(path: Path, digest: str) -> None:
    """Reject a changed provenance dependency.

    Args:
        path: Explicit dependency path.
        digest: Sealed SHA-256.
    """
    if sha256_file(path) != digest:
        raise ValueError(f"source dependency hash mismatch: {path}")


def _verify_dependencies(source: Mapping[str, Any]) -> None:
    """Verify config/data/ensemble/member bytes without model inference.

    Args:
        source: One fold of the original source provenance.
    """
    _verify_file(Path(source["config_path"]), source["config_sha256"])
    data = source["data_identity"]
    _verify_file(Path(data["path"]), data["sha256"])
    ppo = source["ppo"]
    if ppo["data_identity"] != data:
        raise ValueError(f"source PPO data identity differs: {source['config_path']}")
    ensemble = Path(ppo["ensemble_dir"])
    _verify_file(ensemble / "ensemble.json", ppo["ensemble_manifest_sha256"])
    _verify_file(ensemble / "metrics.json", ppo["metrics_sha256"])
    members = _json(ensemble / "ensemble.json")["members"]
    if [member["model_sha256"] for member in members] != ppo["member_model_sha256"]:
        raise ValueError(f"source PPO member order/hash mismatch: {ensemble}")
    for member in members:
        run = Path(member["run_dir"])
        _verify_file(run / member["model_path"], member["model_sha256"])
        _verify_file(run / "config_snapshot.yaml", member["config_snapshot_sha256"])
        _verify_file(run / "meta.json", member["meta_sha256"])


def _render(report: Mapping[str, Any]) -> str:
    """Render one comparison table covering stages, differences, and controls.

    Args:
        report: Complete diagnostic evidence.

    Returns:
        Markdown with explicit units and source digests.
    """
    lines = [
        "# Frozen ranking spread-to-net diagnostic",
        "",
        f"Classification retained: **{report['classification']}**",
        "",
        "Stages and adjacent differences: annual log return, equal-weight folds. Original tail: log return/decision; original gross/net: annual simple return. Never compare CI bounds across different units.",
        "",
        "| Series (unit) | Mean | 2009–2018 | 2019–2025 | IID 95% | 3-fold block 95% | LOO min / max | Largest abs fold (share) | Without top 3 |",
        "|---|---:|---:|---:|---|---|---|---|---:|",
    ]
    evidence = {
        **{f"{key} (annual log)": value for key, value in report["stages"].items()},
        **{
            f"{key} (annual log)": value
            for key, value in report["stage_differences"].items()
        },
    }
    for policy, metrics in report["original_metrics"].items():
        for key, value in metrics.items():
            unit = (
                "log/decision"
                if key == "mean_tail_spread"
                else "initial equity ratio"
                if key == "total_cost_ratio"
                else "annual simple"
            )
            evidence[f"{policy}: {key} ({unit})"] = value
    for label, e in evidence.items():
        ci = e["intervals"]
        loo = e["leave_one_fold_out_means"].values()
        share = e["largest_absolute_contribution_fraction"]
        share_text = (
            "undefined (zero absolute sum)" if share is None else f"{share:.2%}"
        )
        lines.append(
            f"| {label} | {e['mean_difference']:.8f} | {e['eras']['2009-2018']:.8f} | {e['eras']['2019-2025']:.8f} | [{ci['fold_low']:.8f}, {ci['fold_high']:.8f}] | [{ci['moving_block_low']:.8f}, {ci['moving_block_high']:.8f}] | {min(loo):.8f} / {max(loo):.8f} | {e['largest_absolute_fold']} ({share_text}) | {e['mean_without_three_largest_folds']:.8f} |"
        )
    lines += [
        "",
        "All table intervals use shared seed-16 draws. The original Issue #15 seed-15 intervals are independently reproduced below and in original_reported_intervals; the ranking metric definition is unchanged.",
        "",
        "| Original source interval | Seed | IID 95% | Moving-block 95% |",
        "|---|---:|---|---|",
    ]
    for policy, metrics in report["original_reported_intervals"].items():
        for key, item in metrics.items():
            ci = item["intervals"]
            lines.append(
                f"| {policy}: {key} | {item['seed']} | [{ci['fold_low']:.8f}, {ci['fold_high']:.8f}] | [{ci['moving_block_low']:.8f}, {ci['moving_block_high']:.8f}] |"
            )
    lines += [
        "",
        "Price-only compounds the realized-path price contributions; it is not an independently operated cost-free account.",
        "Annual gross minus net is a difference of expm1 values, not total cost / initial equity.",
        "All individual fold and leave-one-fold-out values are in report.json and fold_metrics.csv; exact shared resampling indices are in bootstrap_indices.npz.",
        f"The weighted-price minus scaled-tail mean difference ({report['stage_differences']['weighted_price_log_return_minus_scaled_tail_spread']['mean_difference']:.3e} annual log units) is float32 weight precision, not economic evidence; unrounded values are retained in JSON/CSV.",
        "",
        f"Maximum per-step accounting residual: {report['max_abs_accounting_residual']:.3e} (simple return).",
        "",
        "## Source provenance SHA-256",
        "",
    ]
    lines.extend(
        f"- {name}: `{value['provenance_sha256']}`"
        for name, value in report["sources"].items()
    )
    return "\n".join(lines) + "\n"


def run_decomposition(
    campaign_path: Path, output_dir: Path
) -> tuple[Path, dict[str, Any]]:
    """Validate explicit inputs and atomically publish the frozen diagnostic.

    Args:
        campaign_path: JSON pinning both source directories and provenance hashes.
        output_dir: New output directory.

    Returns:
        Published directory and complete report.
    """
    destination = output_dir.resolve()
    if destination.exists():
        raise FileExistsError(f"output directory already exists: {destination}")
    campaign_path = campaign_path.resolve()
    campaign = _json(campaign_path)
    if set(campaign) != {"ranking", "portfolio"}:
        raise ValueError(
            f"campaign must explicitly select ranking and portfolio: {campaign_path}"
        )
    directories: dict[str, Path] = {}
    seals: dict[str, dict[str, Any]] = {}
    for name, required in (
        ("ranking", set(SOURCE_ARTIFACTS)),
        ("portfolio", PORTFOLIO_ARTIFACTS),
    ):
        selected = campaign[name]
        if (
            not isinstance(selected, dict)
            or set(selected) != {"directory", "provenance_sha256"}
            or any(
                not isinstance(value, str) or not value for value in selected.values()
            )
        ):
            raise ValueError(
                f"campaign {name} requires explicit directory and provenance_sha256."
            )
        directories[name] = (campaign_path.parent / selected["directory"]).resolve()
        seals[name] = load_seal(
            directories[name], selected["provenance_sha256"], required
        )
    ranking, portfolio = seals["ranking"], seals["portfolio"]
    if (
        portfolio["source_provenance_sha256"]
        != campaign["ranking"]["provenance_sha256"]
        or portfolio["source_artifact_sha256"] != ranking["generated_artifact_sha256"]
    ):
        raise ValueError("portfolio source seal differs from selected ranking source.")
    for name, seal in seals.items():
        if set(seal["fold_sources"]) != set(FOLDS):
            raise ValueError(f"{name} source requires exactly 17 folds.")
    _verify_file(
        directories["ranking"] / "study_snapshot.yaml", ranking["study_sha256"]
    )
    _verify_file(
        directories["portfolio"] / "campaign_snapshot.json",
        portfolio["campaign_sha256"],
    )
    source_reports = {
        name: _json(directory / "report.json")
        for name, directory in directories.items()
    }
    if source_reports["ranking"]["classification"] != "established learnable":
        raise ValueError("selected ranking source must be established learnable.")
    protocol = source_reports["portfolio"]["protocol"]
    for key, expected in {
        "top_k": 2,
        "weight_magnitude": 0.8,
        "gross_target": 3.2,
        "decision_interval": 1,
        "bootstrap_samples": 10000,
        "bootstrap_seed": 16,
        "moving_block_length": 3,
    }.items():
        if protocol[key] != expected:
            raise ValueError(f"portfolio protocol mismatch: {key}")
    frames: dict[str, pd.DataFrame] = {}
    for name, source, file in (
        ("predictions", "ranking", "predictions.csv"),
        ("ranking_metrics", "ranking", "fold_metrics.csv"),
        ("steps", "portfolio", "steps.csv"),
        ("pairs", "portfolio", "pair_contributions.csv"),
        ("portfolio_metrics", "portfolio", "fold_metrics.csv"),
    ):
        frame = pd.read_csv(
            directories[source] / file,
            dtype={"fold": str},
            float_precision="round_trip",
        )
        if (
            tuple(frame["fold"].drop_duplicates()) != FOLDS
            or frame["fold"].isna().any()
        ):
            raise ValueError(
                f"{name}: folds must be ordered exactly 2009 through 2025."
            )
        if name != "predictions":
            column = "score" if name == "ranking_metrics" else "policy"
            expected_keys = [(fold, policy) for fold in FOLDS for policy in POLICIES]
            if (
                list(
                    frame[["fold", column]]
                    .drop_duplicates()
                    .itertuples(index=False, name=None)
                )
                != expected_keys
                or frame[column].isna().any()
            ):
                raise ValueError(
                    f"{name}: complete ordered fold/policy groups required."
                )
        frames[name] = frame
    all_rows: list[pd.DataFrame] = []
    fold_rows: list[dict[str, Any]] = []
    models = _json(directories["ranking"] / "models.json")
    if set(models) != set(FOLDS):
        raise ValueError("ranking models must contain exactly 17 folds.")
    for fold in FOLDS:
        print(
            f"decomposition fold {fold}: verifying sources and accounting", flush=True
        )
        source = ranking["fold_sources"][fold]
        frozen = portfolio["fold_sources"][fold]
        if (
            frozen["source"] != source
            or frozen["current_replay_matches_sealed_ppo"] is not True
            or frozen["selected_alpha"] != models[fold]["selected_alpha"]
        ):
            raise ValueError(
                f"fold {fold}: model/config/data provenance chain differs."
            )
        _verify_dependencies(source)
        config_path = Path(source["config_path"])
        config, raw = load_experiment_config(config_path)
        if data_identity_from_config(raw, config_path) != source["data_identity"]:
            raise ValueError(f"fold {fold}: current data identity differs.")
        env = resolve_env_raw(config.env, config.eval_range, for_eval=True)
        if (
            env != frozen["resolved_eval_env"]
            or env["transaction_costs"]["carry_mode"] != "signed"
            or config.run.decision_interval != 1
        ):
            raise ValueError(f"fold {fold}: evaluation environment differs.")
        dataset = build_dataset_from_env_raw(
            env, config.custom_feature_names, config.custom_cross_feature_names
        )
        if models[fold]["evaluation_decisions"] != len(dataset.targets) or models[fold][
            "feature_names"
        ] != list(dataset.feature_names):
            raise ValueError(f"fold {fold}: frozen model dataset differs.")
        predictions = frames["predictions"].loc[frames["predictions"]["fold"] == fold]
        matrices = prediction_matrices(predictions, dataset)
        parsed = parse_env_config(env)
        market = build_provider(parsed.data, seed=parsed.environment.seed).get_data(
            dataset.symbols,
            parsed.data.start_date,
            parsed.data.end_date,
            parsed.data.timeframe,
        )
        closes = market.loc[:, [(pair, "Close") for pair in dataset.symbols]]
        relatives = closes.loc[list(dataset.target_timestamps)].to_numpy(
            dtype=float
        ) / closes.loc[list(dataset.decision_timestamps)].to_numpy(dtype=float)
        carry = market.loc[
            list(dataset.decision_timestamps),
            [(pair, "CarryAnnual") for pair in dataset.symbols],
        ].to_numpy(dtype=float)
        ppo = matrices["ppo"].astype(np.float32).astype(float)
        # Direct actions are clipped before the environment's gross leverage cap.
        ppo = np.clip(ppo, -1.0, 1.0)
        leverage = np.abs(ppo).sum(axis=1)
        cap = float(env["environment"]["max_leverage"])
        scale = np.ones(len(ppo))
        np.divide(cap, leverage, out=scale, where=leverage > cap)
        weights = {
            "supervised": fixed_portfolio_weights(matrices["supervised"]).astype(float),
            "reversal": _canonical_weights(dataset).astype(float),
            "ppo": ppo * scale[:, None],
        }
        era = "2009-2018" if int(fold) <= 2018 else "2019-2025"
        for policy in POLICIES:
            subsets = {
                name: frames[name].loc[
                    (frames[name]["fold"] == fold) & (frames[name]["policy"] == policy)
                ]
                for name in ("steps", "pairs", "portfolio_metrics")
            }
            try:
                rows, metrics = decompose_fold(
                    predictions,
                    subsets["steps"],
                    subsets["pairs"],
                    dataset.symbols,
                    relatives,
                    carry,
                    float(env["environment"]["initial_balance_jpy"]),
                    policy,
                    weights[policy],
                )
                original = subsets["portfolio_metrics"]
                predictive = frames["ranking_metrics"].loc[
                    (frames["ranking_metrics"]["fold"] == fold)
                    & (frames["ranking_metrics"]["score"] == policy)
                ]
                if len(original) != 1 or len(predictive) != 1:
                    raise ValueError(
                        "source metrics require exactly one fold/policy row."
                    )
                _match(
                    metrics["mean_tail_spread"],
                    float(predictive.iloc[0]["mean_tail_spread"]),
                    "original tail spread",
                    1e-12,
                )
                for key in (
                    "annualized_gross_return",
                    "annualized_net_return",
                    "gross_cumulative_log_return",
                    "cumulative_log_return",
                    "total_cost_ratio",
                ):
                    _match(
                        metrics[key],
                        float(original.iloc[0][key]),
                        f"original {key}",
                        len(rows) * 1e-12 / min(1.0, metrics["years"]),
                    )
                if (
                    int(original.iloc[0]["steps"]) != len(rows)
                    or original.iloc[0]["eval_start"]
                    != rows.iloc[0]["decision_timestamp"]
                    or original.iloc[0]["eval_end"] != rows.iloc[-1]["target_timestamp"]
                ):
                    raise ValueError("original evaluation range differs.")
            except ValueError as exc:
                raise ValueError(f"fold {fold} {policy}: {exc}") from exc
            for table in (subsets["steps"], subsets["pairs"], original, predictive):
                if not (table["era"] == era).all():
                    raise ValueError(f"fold {fold} {policy}: era mismatch.")
            rows.insert(0, "policy", policy)
            rows.insert(0, "fold", fold)
            all_rows.append(rows)
            fold_rows.append({"fold": fold, "era": era, "policy": policy, **metrics})
    folds = pd.DataFrame(fold_rows)
    stages = {
        stage: paired_evidence(
            folds.loc[folds["policy"] == "supervised", stage].to_numpy(), np.zeros(17)
        )
        for stage in STAGES
    }
    differences = {
        f"{later}_minus_{earlier}": paired_evidence(
            folds.loc[folds["policy"] == "supervised", later].to_numpy(),
            folds.loc[folds["policy"] == "supervised", earlier].to_numpy(),
        )
        for earlier, later in pairwise(STAGES)
    }
    originals = {
        policy: {
            key: paired_evidence(
                folds.loc[folds["policy"] == policy, key].to_numpy(), np.zeros(17)
            )
            for key in (
                "mean_tail_spread",
                "annualized_gross_return",
                "annualized_net_return",
                "total_cost_ratio",
            )
        }
        for policy in POLICIES
    }
    for policy, metrics in originals.items():
        for key, evidence in metrics.items():
            source = source_reports[
                "ranking" if key == "mean_tail_spread" else "portfolio"
            ]["aggregates"][policy]
            _match(
                evidence["mean_difference"],
                source["overall"][key],
                f"original aggregate {policy} {key}",
                1e-9,
            )
            for era in ("2009-2018", "2019-2025"):
                _match(
                    evidence["eras"][era],
                    source["eras"][era][key],
                    f"original era {policy} {key}",
                    1e-9,
                )
    reported_intervals: dict[str, dict[str, Any]] = {}
    for policy in POLICIES:
        reported_intervals[policy] = {}
        for key in (
            "mean_tail_spread",
            "annualized_gross_return",
            "annualized_net_return",
        ):
            seed = 15 if key == "mean_tail_spread" else 16
            expected_ci = (
                source_reports["ranking"]["aggregates"][policy]["bootstrap_intervals"][
                    key
                ]
                if key == "mean_tail_spread"
                else source_reports["portfolio"]["absolute_evidence"][policy][key][
                    "intervals"
                ]
            )
            reproduced_ci = asdict(
                bootstrap_mean_intervals(
                    folds.loc[folds["policy"] == policy, key].to_numpy(), 10000, seed, 3
                )
            )
            for bound, value in reproduced_ci.items():
                _match(
                    value,
                    expected_ci[bound],
                    f"original CI {policy} {key} {bound}",
                    1e-9,
                )
            reported_intervals[policy][key] = {"seed": seed, "intervals": reproduced_ci}
    sources = {
        name: {
            "directory": str(directories[name]),
            "provenance_sha256": campaign[name]["provenance_sha256"],
            "generated_artifact_sha256": seals[name]["generated_artifact_sha256"],
        }
        for name in directories
    }
    report = {
        "artifact_version": 1,
        "classification": source_reports["portfolio"]["classification"],
        "classification_changed": False,
        "sources": sources,
        "original_reported_intervals": reported_intervals,
        "stages": stages,
        "stage_differences": differences,
        "original_metrics": originals,
        "max_abs_accounting_residual": float(
            folds["max_abs_accounting_residual"].max()
        ),
        "protocol": {
            "adr": "0030",
            "folds": list(FOLDS),
            "bootstrap_samples": 10000,
            "seed": 16,
            "moving_block_length": 3,
            "stage_unit": "annual log return",
            "fold_weight": "equal",
            "years": "actual elapsed seconds / (365.25 * 86400)",
            "rtol": 1e-10,
            "step_atol": 1e-12,
            "price_only": "realized equity path contribution; not an independent cost-free account",
            "classification": "inherited unchanged from Issue #16",
        },
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}-", dir=destination.parent)
    )
    try:
        folds.to_csv(staging / "fold_metrics.csv", index=False)
        pd.concat(all_rows, ignore_index=True).to_csv(
            staging / "steps.csv", index=False
        )
        iid, block = _bootstrap_indices(17, 10000, 16, 3)
        np.savez_compressed(
            staging / "bootstrap_indices.npz", iid=iid, moving_block=block
        )
        (staging / "report.json").write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        (staging / "report.md").write_text(_render(report), encoding="utf-8")
        shutil.copyfile(campaign_path, staging / "campaign_snapshot.json")
        provenance = {
            "artifact_version": 1,
            "sources": sources,
            "source_fold_provenance": {
                name: seal["fold_sources"] for name, seal in seals.items()
            },
            "diagnostic_git": git_commits(),
            "diagnostic_versions": dict(_study_versions()),
            "implementation_sha256": sha256_file(Path(__file__)),
            "campaign_sha256": sha256_file(campaign_path),
            "generated_artifact_sha256": {
                path.name: sha256_file(path) for path in staging.iterdir()
            },
        }
        (staging / "provenance.json").write_text(
            json.dumps(provenance, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        os.replace(staging, destination)
    except Exception:
        shutil.rmtree(staging)
        raise
    return destination, report


def main() -> int:
    """Run the explicit diagnostic CLI.

    Returns:
        Zero on success, one on an explicit validation failure.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    try:
        output, report = run_decomposition(Path(args.campaign), Path(args.output_dir))
    except (ValueError, OSError, KeyError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"diagnostic: {output}\nclassification retained: {report['classification']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
