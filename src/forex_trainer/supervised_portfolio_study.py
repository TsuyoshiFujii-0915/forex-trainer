"""Sealed fold-aware portfolio translation campaign for Issue #16."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from forex_env import parse_config as parse_env_config
from forex_env.data import build_provider

from .artifact_provenance import data_identity_from_config, git_commits, sha256_file
from .config import TrainerConfigError, load_experiment_config, resolve_env_raw
from .env_factory import GateEvaluationMode, build_single_env
from .regime_study import RuleSpec, build_momentum_reversal_action
from .supervised_portfolio import (
    FOLDS,
    _positive,
    classify_portfolio,
    fixed_portfolio_weights,
    load_frozen_source,
    paired_evidence,
    prediction_matrices,
    replay_weights,
)
from .supervised_ranking import SupervisedDataset, compute_score_diagnostics
from .supervised_ranking_study import (
    _metrics_match,
    _replay_ppo_scores,
    _study_versions,
    _validate_longf_config,
    _write_csv,
    _write_json,
    build_dataset_from_env_raw,
    load_supervised_study,
    require_score_alignment,
)

_POLICIES: tuple[str, ...] = ("supervised", "reversal", "ppo")
_METRICS: tuple[str, ...] = (
    "annualized_net_return",
    "annualized_gross_return",
    "sharpe_annualized",
    "max_drawdown",
    "mean_gross_leverage",
    "mean_weight_turnover",
    "total_weight_turnover",
    "total_cost_ratio",
    "cumulative_log_return",
    "gross_cumulative_log_return",
)


def load_decision_carry(
    env_raw: Mapping[str, Any], dataset: SupervisedDataset
) -> np.ndarray:
    """Load raw carry through the same validated data provider as ForexEnv.

    Args:
        env_raw: Resolved evaluation environment with signed carry.
        dataset: Complete ordered decision timestamps and pair identity.

    Returns:
        Raw decimal annual carry differentials, aligned to decisions and pairs.
    """
    parsed = parse_env_config(env_raw)
    provider = build_provider(parsed.data, seed=parsed.environment.seed)
    market = provider.get_data(
        dataset.symbols,
        parsed.data.start_date,
        parsed.data.end_date,
        parsed.data.timeframe,
    )
    return market.loc[
        list(dataset.decision_timestamps),
        [(pair, "CarryAnnual") for pair in dataset.symbols],
    ].to_numpy(dtype=float)


def load_campaign(path: Path) -> tuple[Path, str, Path]:
    """Resolve only explicitly pinned sources; portfolio controls are not tunable.

    Args:
        path: Campaign JSON containing source and study paths and digests.

    Returns:
        Source directory, pinned provenance digest, and original study manifest.
    """
    raw = json.loads(path.read_text(encoding="utf-8"))
    keys = {
        "source_dir",
        "source_provenance_sha256",
        "source_study",
        "source_study_sha256",
    }
    if not isinstance(raw, dict) or set(raw) != keys:
        raise ValueError(f"campaign keys must be exactly {sorted(keys)}: {path}")
    if any(not isinstance(value, str) or not value for value in raw.values()):
        raise ValueError(f"campaign values must be nonempty strings: {path}")
    source = (path.parent / raw["source_dir"]).resolve()
    study = (path.parent / raw["source_study"]).resolve()
    if sha256_file(study) != raw["source_study_sha256"]:
        raise ValueError(f"source study hash mismatch: {study}")
    return source, raw["source_provenance_sha256"], study


def _canonical_weights(dataset: SupervisedDataset) -> np.ndarray:
    """Reproduce the established rule including its ascending-momentum tie order.

    Args:
        dataset: Canonical 32-by-8 unnormalized market observation windows.

    Returns:
        Direct rule weights built by the existing canonical evaluator.
    """
    features = (
        "log_return",
        "volatility",
        "sma20_ratio",
        "mom24",
        "xz_mom24",
        "xr_mom24",
        "carry_annual",
        "xz_carry",
    )
    rule = RuleSpec(feature="mom24", top_k=2, base_size=0.8)
    return np.stack(
        [
            build_momentum_reversal_action(
                {"market": row.reshape(9, 32, 8).astype(np.float32)}, features, rule
            )[:, 0]
            for row in dataset.features
        ]
    )


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Reduce aligned fold metrics with equal fold weight and explicit totals.

    Args:
        rows: Nonempty policy observations for the complete period or one era.

    Returns:
        Fold means, winning-fold count, worst drawdown, and turnover/cost sums.
    """
    return {
        **{key: float(np.mean([row[key] for row in rows])) for key in _METRICS},
        "fold_count": len(rows),
        "winning_folds": sum(row["annualized_net_return"] > 0 for row in rows),
        "worst_max_drawdown": max(row["max_drawdown"] for row in rows),
        "sum_total_weight_turnover": sum(row["total_weight_turnover"] for row in rows),
        "sum_total_cost_ratio": sum(row["total_cost_ratio"] for row in rows),
    }


def _diagnostics(
    scores: np.ndarray,
    weights: np.ndarray,
    rule: np.ndarray,
    dataset: SupervisedDataset,
) -> dict[str, float]:
    """Reduce score churn and signed membership changes per fold.

    Args:
        scores: Frozen supervised score matrix.
        weights: Fixed supervised target portfolios.
        rule: Aligned canonical target portfolios.
        dataset: Evaluation dataset supplying predictive diagnostic targets.

    Returns:
        Mean rank churn, membership exit rate, and rule disagreement fraction.
    """
    membership = np.sign(weights)
    exited = (membership[:-1] != 0) & (membership[:-1] != membership[1:])
    if len(scores) < 2:
        raise ValueError("membership turnover requires at least two decisions.")
    predictive = compute_score_diagnostics(scores, dataset.targets, 2)
    return {
        "mean_rank_churn": predictive.mean_rank_churn,
        "mean_membership_turnover": float(exited.sum(axis=1).mean() / 4),
        "membership_differs_from_rule_fraction": float(
            np.any(weights != rule, axis=1).mean()
        ),
    }


def _render_report(report: Mapping[str, Any]) -> str:
    """Render a compact report with full evidence in adjacent JSON/CSV.

    Args:
        report: Complete portfolio comparison report.

    Returns:
        Markdown tables, classification, and the next research decision.
    """
    lines = [
        "# Frozen supervised portfolio translation",
        "",
        f"Classification: **{report['classification']}**",
        "",
        "Equal-weight fold means; daily decisions are not independent bootstrap samples.",
        "",
        "| Policy | Net annualized | Gross annualized | Sharpe | Mean / worst MDD | Winning folds | Leverage | Mean turnover | Mean total cost ratio |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for policy, summaries in report["aggregates"].items():
        row = summaries["overall"]
        lines.append(
            f"| {policy} | {row['annualized_net_return']:.2%} | {row['annualized_gross_return']:.2%} | {row['sharpe_annualized']:.3f} | {row['max_drawdown']:.2%} / {row['worst_max_drawdown']:.2%} | {row['winning_folds']}/17 | {row['mean_gross_leverage']:.3f} | {row['mean_weight_turnover']:.3f} | {row['total_cost_ratio']:.2%} |"
        )
    lines += [
        "",
        "## Paired annualized return differences",
        "",
        "| Control | Return | Mean | IID 95% | Moving block 95% |",
        "|---|---|---:|---:|---:|",
    ]
    for control, metrics in report["paired"].items():
        for metric, evidence in metrics.items():
            ci = evidence["intervals"]
            lines.append(
                f"| {control} | {metric} | {evidence['mean_difference']:.2%} | [{ci['fold_low']:.2%}, {ci['fold_high']:.2%}] | [{ci['moving_block_low']:.2%}, {ci['moving_block_high']:.2%}] |"
            )
    lines += [
        "",
        f"Decision: {report['next_decision']}",
        "",
        "See report.json for eras, classification inputs, all leave-one-fold-out means, rank contributions, and concentration diagnostics.",
        "See fold_metrics.csv, steps.csv, and pair_contributions.csv for aligned observations.",
        "",
    ]
    return "\n".join(lines)


def run_portfolio_study(
    campaign_path: Path, output_dir: Path
) -> tuple[Path, Mapping[str, Any]]:
    """Verify frozen sources, evaluate all policies, and atomically seal results.

    Args:
        campaign_path: Explicit preregistered source manifest.
        output_dir: New destination directory, never an existing artifact.

    Returns:
        Published artifact directory and complete comparative report.
    """
    destination = output_dir.resolve()
    if destination.exists():
        raise FileExistsError(f"output directory already exists: {destination}")
    source_dir, source_digest, study_path = load_campaign(campaign_path.resolve())
    provenance = load_frozen_source(source_dir, source_digest)
    if sha256_file(study_path) != provenance["study_sha256"]:
        raise ValueError("campaign study differs from the frozen source study.")
    study = load_supervised_study(study_path)
    predictions = pd.read_csv(
        source_dir / "predictions.csv",
        dtype={"fold": str},
        float_precision="round_trip",
    )
    if tuple(predictions["fold"].drop_duplicates()) != FOLDS:
        raise ValueError("prediction folds must be ordered exactly 2009 through 2025.")
    if set(provenance["fold_sources"]) != set(FOLDS):
        raise ValueError("source provenance must contain exactly 17 folds.")
    models = json.loads((source_dir / "models.json").read_text(encoding="utf-8"))
    if set(models) != set(FOLDS):
        raise ValueError("source models must contain exactly 17 folds.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}-", dir=destination.parent)
    )
    try:
        fold_rows: list[dict[str, Any]] = []
        all_steps: list[dict[str, Any]] = []
        all_pairs: list[dict[str, Any]] = []
        rank_rows: list[dict[str, Any]] = []
        sources: dict[str, Any] = {}
        runtime = {"git": git_commits(), "versions": dict(_study_versions())}
        for fold, source in study.folds.items():
            print(
                f"portfolio fold {fold}: verifying frozen sources and replaying three policies",
                flush=True,
            )
            sealed = provenance["fold_sources"][fold]
            if sha256_file(source.config) != sealed["config_sha256"]:
                raise ValueError(
                    f"fold {fold} source config hash mismatch: {source.config}"
                )
            config, raw = load_experiment_config(source.config)
            _validate_longf_config(config, fold)
            identity = data_identity_from_config(raw, source.config)
            if identity != sealed["data_identity"]:
                raise ValueError(f"fold {fold} source data identity mismatch.")
            env_raw = resolve_env_raw(config.env, config.eval_range, for_eval=True)
            if env_raw["transaction_costs"]["carry_mode"] != "signed":
                raise ValueError(f"fold {fold} requires signed carry.")
            dataset = build_dataset_from_env_raw(
                env_raw, config.custom_feature_names, config.custom_cross_feature_names
            )
            if models[fold]["evaluation_decisions"] != len(dataset.targets) or models[
                fold
            ]["feature_names"] != list(dataset.feature_names):
                raise ValueError(f"fold {fold} frozen model dataset contract mismatch.")
            matrices = prediction_matrices(
                predictions.loc[predictions["fold"] == fold], dataset
            )
            ensemble_manifest = source.ppo_ensemble / "ensemble.json"
            if (
                sha256_file(ensemble_manifest)
                != sealed["ppo"]["ensemble_manifest_sha256"]
            ):
                raise ValueError(f"fold {fold} source PPO manifest hash mismatch.")
            ppo, ppo_source = _replay_ppo_scores(
                source.ppo_ensemble, env_raw, dataset.symbols, study.member_seeds
            )
            require_score_alignment(dataset, ppo)
            if ppo_source != sealed["ppo"] or not np.array_equal(
                ppo.scores, matrices["ppo"]
            ):
                raise ValueError(
                    f"fold {fold} PPO replay differs from frozen Issue #15 source."
                )
            # Read the raw carry column: the learned feature may transform units.
            carry = load_decision_carry(env_raw, dataset)
            weights = {
                "supervised": fixed_portfolio_weights(matrices["supervised"]),
                "reversal": _canonical_weights(dataset),
                "ppo": matrices["ppo"].astype(np.float32),
            }
            diagnostics = _diagnostics(
                matrices["supervised"],
                weights["supervised"],
                weights["reversal"],
                dataset,
            )
            era = "2009-2018" if int(fold) <= 2018 else "2019-2025"
            for policy in _POLICIES:
                env = build_single_env(
                    env_raw,
                    config.custom_feature_names,
                    config.custom_cross_feature_names,
                    seed=0,
                    decision_interval=1,
                    residual=None,
                    rank_allocation=None,
                    apply_hold_gate=None,
                    gate_evaluation_mode=GateEvaluationMode.LEARNED,
                )
                try:
                    metrics, steps, pairs = replay_weights(
                        env, dataset, weights[policy], matrices[policy], carry
                    )
                finally:
                    env.close()
                if policy == "ppo":
                    metrics_path = source.ppo_ensemble / "metrics.json"
                    _metrics_match(
                        metrics,
                        json.loads(metrics_path.read_text(encoding="utf-8")),
                        metrics_path,
                    )
                else:
                    effective = np.array(
                        [pair["target_weight"] for pair in pairs]
                    ).reshape(dataset.targets.shape)
                    if not np.array_equal(effective, weights[policy].astype(float)):
                        raise ValueError(
                            f"fold {fold} {policy} fixed portfolio was scaled by the environment."
                        )
                row = {"fold": fold, "era": era, "policy": policy, **metrics}
                if policy == "supervised":
                    row.update(diagnostics)
                else:
                    row.update({key: None for key in diagnostics})
                gross_log = metrics["gross_cumulative_log_return"]
                drag = gross_log - metrics["cumulative_log_return"]
                row["cost_drag_log_return"] = drag
                row["cost_drag_fraction_of_gross_log_return"] = (
                    drag / gross_log if gross_log != 0 else None
                )
                fold_rows.append(row)
                all_steps.extend(
                    {"fold": fold, "era": era, "policy": policy, **step}
                    for step in steps
                )
                all_pairs.extend(
                    {"fold": fold, "era": era, "policy": policy, **pair}
                    for pair in pairs
                )
                if policy == "supervised":
                    years = (
                        dataset.target_timestamps[-1] - dataset.decision_timestamps[0]
                    ).total_seconds() / (365.25 * 86400)
                    for rank in range(1, 10):
                        selected = [
                            pair for pair in pairs if pair["predicted_rank"] == rank
                        ]
                        rank_rows.append(
                            {
                                "fold": fold,
                                "era": era,
                                "predicted_rank": rank,
                                "annualized_gross_log_contribution": sum(
                                    pair["gross_log_contribution"] for pair in selected
                                )
                                / years,
                                "sum_price_simple_contribution": sum(
                                    pair["price_simple_contribution"]
                                    for pair in selected
                                ),
                                "sum_carry_simple_contribution": sum(
                                    pair["carry_simple_contribution"]
                                    for pair in selected
                                ),
                            }
                        )
            sources[fold] = {
                "source": sealed,
                "resolved_eval_env": env_raw,
                "selected_alpha": models[fold]["selected_alpha"],
                "current_replay_matches_sealed_ppo": True,
            }
        policy_rows = {
            policy: [row for row in fold_rows if row["policy"] == policy]
            for policy in _POLICIES
        }
        aggregates = {
            policy: {
                "overall": _summary(rows),
                "eras": {
                    era: _summary([row for row in rows if row["era"] == era])
                    for era in ("2009-2018", "2019-2025")
                },
            }
            for policy, rows in policy_rows.items()
        }
        values = {
            policy: {
                metric: np.array([row[metric] for row in rows])
                for metric in (
                    "annualized_net_return",
                    "annualized_gross_return",
                    "mean_gross_leverage",
                )
            }
            for policy, rows in policy_rows.items()
        }
        paired = {
            control: {
                metric: paired_evidence(
                    values["supervised"][metric], values[control][metric]
                )
                for metric in ("annualized_net_return", "annualized_gross_return")
            }
            for control in ("reversal", "ppo")
        }
        classification = classify_portfolio(
            values["supervised"]["annualized_gross_return"],
            values["supervised"]["annualized_net_return"],
            values["reversal"]["annualized_net_return"],
            values["supervised"]["mean_gross_leverage"],
        )
        promote = classification["classification"] == "learned and tradable" and all(
            _positive(item) and min(item["eras"].values()) > 0
            for item in paired["reversal"].values()
        )
        decisions = {
            "learned and tradable": "Adopt supervised fixed-map baseline."
            if promote
            else "Retain supervised tradability evidence; keep canonical reversal as trading benchmark.",
            "learned but cost-limited": "Keep canonical benchmark; a subsequent preregistered low-capacity cost-aware/contextual-bandit study is justified. Do not reopen generic PPO search.",
            "not successfully translated to portfolio alpha": "Stop before bandit work and reconcile predictive objectives with portfolio economics.",
        }
        report = {
            "artifact_version": 1,
            "classification": classification["classification"],
            "classification_inputs": classification,
            "promote_supervised_baseline": promote,
            "next_decision": decisions[classification["classification"]],
            "protocol": {
                "top_k": 2,
                "weight_magnitude": 0.8,
                "gross_target": 3.2,
                "decision_interval": 1,
                "score_direction": "high long, low short",
                "tie_break": "stable pair order; canonical preserves existing momentum sort",
                "bootstrap_samples": 10000,
                "bootstrap_seed": 16,
                "moving_block_length": 3,
                "aggregate": "equal-weight arithmetic mean of fold annualized returns",
                "gross": "transaction costs added back; signed carry retained",
                "rank_log_attribution": "price plus signed carry simple contributions scaled by log1p(total)/total",
                "cost_drag_fraction": "(gross log return - net log return) / gross log return; null only at zero gross",
            },
            "aggregates": aggregates,
            "paired": paired,
            "absolute_evidence": {
                policy: {
                    metric: paired_evidence(values[policy][metric], np.zeros(17))
                    for metric in ("annualized_net_return", "annualized_gross_return")
                }
                for policy in _POLICIES
            },
            "supervised_diagnostics": {
                scope: {
                    key: float(
                        np.mean(
                            [
                                row[key]
                                for row in policy_rows["supervised"]
                                if scope == "overall" or row["era"] == scope
                            ]
                        )
                    )
                    for key in (*diagnostics, "cost_drag_log_return")
                }
                for scope in ("overall", "2009-2018", "2019-2025")
            },
            "rank_contributions": {
                scope: {
                    str(rank): float(
                        np.mean(
                            [
                                row["annualized_gross_log_contribution"]
                                for row in rank_rows
                                if row["predicted_rank"] == rank
                                and (scope == "overall" or row["era"] == scope)
                            ]
                        )
                    )
                    for rank in range(1, 10)
                }
                for scope in ("overall", "2009-2018", "2019-2025")
            },
        }
        for name, rows in (
            ("fold_metrics.csv", fold_rows),
            ("steps.csv", all_steps),
            ("pair_contributions.csv", all_pairs),
            ("rank_contributions.csv", rank_rows),
        ):
            _write_csv(staging / name, rows)
        _write_json(staging / "report.json", report)
        (staging / "report.md").write_text(_render_report(report), encoding="utf-8")
        shutil.copyfile(campaign_path, staging / "campaign_snapshot.json")
        artifacts = {path.name: sha256_file(path) for path in staging.iterdir()}
        _write_json(
            staging / "provenance.json",
            {
                "artifact_version": 1,
                "evaluation": runtime,
                "source_dir": str(source_dir),
                "source_provenance_sha256": source_digest,
                "source_artifact_sha256": provenance["generated_artifact_sha256"],
                "campaign_path": str(campaign_path.resolve()),
                "campaign_sha256": sha256_file(campaign_path),
                "fold_sources": sources,
                "generated_artifact_sha256": artifacts,
            },
        )
        os.replace(staging, destination)
    except Exception:
        shutil.rmtree(staging)
        raise
    return destination, report


def main() -> int:
    """Run the explicit campaign CLI and report validation failures.

    Returns:
        Zero on success, one on an explicit input or artifact validation error.
    """
    parser = argparse.ArgumentParser(
        description="Evaluate frozen Issue #15 scores through the fixed Issue #16 map."
    )
    parser.add_argument("--campaign", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    try:
        output, report = run_portfolio_study(Path(args.campaign), Path(args.output_dir))
    except (ValueError, OSError, KeyError, TrainerConfigError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"portfolio study: {output}\nclassification: {report['classification']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
