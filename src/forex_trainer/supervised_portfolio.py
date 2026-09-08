"""Deterministic fixed-map portfolio economics for sealed Issue #15 scores."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .artifact_provenance import sha256_file
from .evaluate import compute_metrics
from .research_statistics import bootstrap_mean_intervals
from .supervised_ranking import SupervisedDataset, canonical_reversal_scores

FOLDS: tuple[str, ...] = tuple(str(year) for year in range(2009, 2026))
SOURCE_ARTIFACTS: tuple[str, ...] = (
    "predictions.csv",
    "models.json",
    "report.json",
    "study_snapshot.yaml",
    "fold_metrics.csv",
    "coefficients.csv",
    "coefficient_summary.csv",
    "report.md",
)


def fixed_portfolio_weights(scores: np.ndarray) -> np.ndarray:
    """Map nine finite pair scores to the preregistered four-position portfolio.

    Args:
        scores: Nonempty decision-by-nine score matrix, high scores long.

    Returns:
        Float32 direct weights, with stable configured-pair tie breaking.
    """
    values = np.asarray(scores, dtype=np.float64)
    if (
        values.ndim != 2
        or values.shape[1] != 9
        or len(values) == 0
        or not np.isfinite(values).all()
    ):
        raise ValueError("scores must be a nonempty finite decision-by-9 matrix.")
    order = np.argsort(values, axis=1, kind="stable")
    weights = np.zeros(values.shape, dtype=np.float32)
    np.put_along_axis(weights, order[:, :2], -0.8, axis=1)
    np.put_along_axis(weights, order[:, -2:], 0.8, axis=1)
    return weights


def load_frozen_source(source: Path, provenance_sha256: str) -> Mapping[str, Any]:
    """Validate the externally pinned source seal and predictive eligibility.

    Args:
        source: Explicit Issue #15 artifact directory.
        provenance_sha256: Preregistered digest of its provenance manifest.

    Returns:
        Verified source provenance; no prediction is refitted.
    """
    manifest = source / "provenance.json"
    if sha256_file(manifest) != provenance_sha256:
        raise ValueError(f"source provenance hash mismatch: {manifest}")
    provenance = json.loads(manifest.read_text(encoding="utf-8"))
    if provenance["artifact_version"] != 1:
        raise ValueError(f"unsupported source artifact version: {manifest}")
    hashes = provenance["generated_artifact_sha256"]
    if set(hashes) != set(SOURCE_ARTIFACTS):
        raise ValueError(f"incomplete source artifact hashes: {manifest}")
    for name in SOURCE_ARTIFACTS:
        if sha256_file(source / name) != hashes[name]:
            raise ValueError(f"source artifact hash mismatch: {source / name}")
    report = json.loads((source / "report.json").read_text(encoding="utf-8"))
    if report["classification"] != "established learnable":
        raise ValueError(
            "Issue #16 requires established learnable; suggestive requires a separate explicit confirmation decision."
        )
    return provenance


def prediction_matrices(
    frame: pd.DataFrame, dataset: SupervisedDataset
) -> dict[str, np.ndarray]:
    """Require sealed rows to match the complete ordered evaluation dataset.

    Args:
        frame: One fold of the original prediction CSV, without sorting/filtering.
        dataset: Independently reconstructed current evaluation dataset.

    Returns:
        Exact sealed supervised, reversal, and PPO score matrices.
    """
    expected = [
        (d.isoformat(), t.isoformat(), pair)
        for d, t in zip(dataset.decision_timestamps, dataset.target_timestamps)
        for pair in dataset.symbols
    ]
    actual = list(
        frame[["decision_timestamp", "target_timestamp", "pair"]].itertuples(
            index=False, name=None
        )
    )
    if actual != expected:
        raise ValueError(
            "prediction decision/target timestamps or pair order differ from evaluation dataset."
        )
    matrices: dict[str, np.ndarray] = {}
    for name in ("supervised", "reversal", "ppo"):
        values = (
            frame[f"{name}_score"].to_numpy(dtype=float).reshape(dataset.targets.shape)
        )
        if not np.isfinite(values).all():
            raise ValueError(f"prediction {name} scores contain non-finite values.")
        matrices[name] = values
    targets = (
        frame["target_relative_log_return"]
        .to_numpy(dtype=float)
        .reshape(dataset.targets.shape)
    )
    if not np.array_equal(targets, dataset.targets):
        raise ValueError("prediction relative-return targets differ from current data.")
    if not np.array_equal(matrices["reversal"], canonical_reversal_scores(dataset)):
        raise ValueError(
            "prediction canonical reversal scores differ from current data."
        )
    return matrices


def replay_weights(
    env: Any,
    dataset: SupervisedDataset,
    weights: np.ndarray,
    scores: np.ndarray,
    carry_rates: np.ndarray,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    """Replay exact direct weights with timestamp and gross-PnL reconciliation.

    Args:
        env: Fresh direct-action environment using the common evaluation contract.
        dataset: Ordered evaluation decisions, pairs, and targets.
        weights: Explicit direct target weights for every decision and pair.
        scores: Score ordering used only to attribute gross returns by rank.
        carry_rates: Raw annual carry differentials at each decision, or explicit
            zeros for an environment configured without signed carry.

    Returns:
        Standard evaluator metrics, step trace, and pair/rank attribution trace.
    """
    for name, matrix in (
        ("weights", weights),
        ("scores", scores),
        ("carry_rates", carry_rates),
    ):
        if matrix.shape != dataset.targets.shape or not np.isfinite(matrix).all():
            raise ValueError(
                f"replay {name} shape or finite values differ from dataset."
            )
    _, info = env.reset(seed=0)
    equities = [float(info["equity_jpy"])]
    timestamps = [str(info["timestamp"])]
    rewards: list[float] = []
    costs: list[float] = []
    leverages: list[float] = []
    turnovers: list[float] = []
    steps: list[dict[str, Any]] = []
    pairs: list[dict[str, Any]] = []
    previous = np.zeros(len(dataset.symbols))
    for index, (decision, target_time) in enumerate(
        zip(dataset.decision_timestamps, dataset.target_timestamps)
    ):
        if str(info["timestamp"]) != decision.isoformat():
            raise ValueError(f"replay decision timestamp mismatch at {decision}.")
        prices = np.array([info["prices"][pair] for pair in dataset.symbols])
        before = equities[-1]
        _, reward, terminated, truncated, info = env.step(
            weights[index].astype(np.float32).reshape(-1, 1)
        )
        if str(info["timestamp"]) != target_time.isoformat():
            raise ValueError(f"replay target timestamp mismatch at {decision}.")
        target = np.asarray(info["target_weights"], dtype=float)
        if target.shape != (len(dataset.symbols),) or not np.isfinite(target).all():
            raise ValueError(
                f"replay effective target weights are invalid at {decision}."
            )
        after = float(info["equity_jpy"])
        cost = float(info["costs_jpy"]["total"])
        if before <= 0 or after <= 0:
            raise ValueError(
                f"non-positive equity prevents aligned return measurement at {decision}."
            )
        relatives = (
            np.array([info["prices"][pair] for pair in dataset.symbols]) / prices
        )
        elapsed_days = (target_time - decision).total_seconds() / 86400
        price_contribution = target * (relatives - 1)
        carry_contribution = (
            -target * relatives * carry_rates[index] * elapsed_days / 365
        )
        gross_contribution = price_contribution + carry_contribution
        gross_simple = (after + cost) / before - 1
        if not np.isclose(
            carry_contribution.sum(),
            float(info["financing_jpy"]) / before,
            rtol=1e-10,
            atol=1e-12,
        ):
            raise ValueError(
                f"signed carry attribution differs from environment at {decision}."
            )
        if not np.isclose(
            gross_contribution.sum(), gross_simple, rtol=1e-10, atol=1e-12
        ):
            raise ValueError(
                f"gross rank attribution differs from environment at {decision}."
            )
        gross_log = float(np.log1p(gross_simple))
        # The analytic limit log1p(x)/x is one at zero.
        log_scale = 1.0 if gross_simple == 0 else gross_log / gross_simple
        order = np.argsort(scores[index], kind="stable")
        ranks = np.empty(len(order), dtype=int)
        ranks[order] = np.arange(1, len(order) + 1)
        turnover = float(np.abs(target - previous).sum())
        for p, pair in enumerate(dataset.symbols):
            pairs.append(
                {
                    "decision_timestamp": decision.isoformat(),
                    "pair": pair,
                    "predicted_rank": int(ranks[p]),
                    "target_weight": float(target[p]),
                    "price_simple_contribution": float(price_contribution[p]),
                    "carry_simple_contribution": float(carry_contribution[p]),
                    "gross_simple_contribution": float(gross_contribution[p]),
                    "gross_log_contribution": float(gross_contribution[p] * log_scale),
                }
            )
        steps.append(
            {
                "decision_timestamp": decision.isoformat(),
                "target_timestamp": target_time.isoformat(),
                "equity_before": before,
                "equity_after": after,
                "net_log_return": float(reward),
                "gross_log_return": gross_log,
                "cost_jpy": cost,
                "financing_jpy": float(info["financing_jpy"]),
                "weight_turnover": turnover,
                "gross_leverage": float(info["gross_leverage"]),
            }
        )
        previous = target
        equities.append(after)
        timestamps.append(str(info["timestamp"]))
        rewards.append(float(reward))
        costs.append(cost)
        leverages.append(float(info["gross_leverage"]))
        turnovers.append(turnover)
        last = index == len(dataset.targets) - 1
        if (terminated or truncated) != last:
            raise ValueError(f"replay decision range ends differently at {decision}.")
    return (
        compute_metrics(
            rewards, equities, timestamps, costs, leverages, turnovers, terminated
        ),
        steps,
        pairs,
    )


def paired_evidence(candidate: np.ndarray, baseline: np.ndarray) -> dict[str, Any]:
    """Compute paired fold evidence without discarding or reordering observations.

    Args:
        candidate: Exactly 17 finite ordered fold observations.
        baseline: Exactly aligned 17 control observations; zeros for absolute evidence.

    Returns:
        Mean, two bootstrap intervals, and individual-fold influence diagnostics.
    """
    if (
        candidate.shape != (17,)
        or baseline.shape != (17,)
        or not np.isfinite(candidate).all()
        or not np.isfinite(baseline).all()
    ):
        raise ValueError(
            "paired evidence requires finite shape (17,) for both policies."
        )
    delta = candidate - baseline
    absolute_sum = float(np.abs(delta).sum())
    largest = int(np.argmax(np.abs(delta)))
    positive_sum = float(delta[delta > 0].sum())
    descending = np.argsort(-delta, kind="stable")
    leave_out = {
        fold: float(np.delete(delta, i).mean()) for i, fold in enumerate(FOLDS)
    }
    return {
        "mean_difference": float(delta.mean()),
        "intervals": asdict(bootstrap_mean_intervals(delta, 10000, 16, 3)),
        "fold_differences": dict(zip(FOLDS, delta.tolist())),
        "largest_absolute_fold": FOLDS[largest],
        "largest_fold_difference": float(delta[largest]),
        "positive_fold_count": int(np.count_nonzero(delta > 0)),
        "largest_three_positive_contribution_fraction": float(
            np.maximum(delta[descending[:3]], 0).sum() / positive_sum
        )
        if positive_sum > 0
        else None,
        "mean_without_three_largest_folds": float(delta[descending[3:]].mean()),
        "largest_absolute_contribution_fraction": abs(float(delta[largest]))
        / absolute_sum
        if absolute_sum > 0
        else None,
        "leave_one_fold_out_means": leave_out,
        "minimum_leave_one_fold_out_mean": min(leave_out.values()),
        "eras": {
            "2009-2018": float(delta[:10].mean()),
            "2019-2025": float(delta[10:].mean()),
        },
    }


def _positive(evidence: Mapping[str, Any]) -> bool:
    """Require persistent positive fold evidence.

    Args:
        evidence: Paired or absolute evidence returned by paired_evidence.

    Returns:
        Whether both lower bounds and every leave-one-out mean are positive.
    """
    return (
        evidence["mean_difference"] > 0
        and evidence["intervals"]["fold_low"] > 0
        and evidence["intervals"]["moving_block_low"] > 0
        and evidence["minimum_leave_one_fold_out_mean"] > 0
    )


def classify_portfolio(
    gross: np.ndarray, net: np.ndarray, rule_net: np.ndarray, leverage: np.ndarray
) -> dict[str, Any]:
    """Apply the economic criteria fixed before evaluating portfolio results.

    Args:
        gross: Ordered supervised annualized gross returns for 17 folds.
        net: Ordered supervised annualized net returns.
        rule_net: Aligned canonical annualized net returns.
        leverage: Supervised mean realized gross leverage per fold.

    Returns:
        Classification and all gate inputs used for the decision.
    """
    if leverage.shape != (17,) or not np.isfinite(leverage).all():
        raise ValueError("classification leverage must contain 17 finite folds.")
    gross_evidence = paired_evidence(gross, np.zeros(17))
    net_evidence = paired_evidence(net, np.zeros(17))
    relative = paired_evidence(net, rule_net)
    stable_gross = (
        _positive(gross_evidence)
        and min(gross_evidence["eras"].values()) > 0
        and bool(np.all(leverage >= 1))
    )
    inferior = (
        relative["intervals"]["fold_high"] < 0
        and relative["intervals"]["moving_block_high"] < 0
    )
    if stable_gross and (net.mean() <= 0 or inferior):
        classification = "learned but cost-limited"
    elif stable_gross and _positive(net_evidence):
        classification = "learned and tradable"
    else:
        classification = "not successfully translated to portfolio alpha"
    return {
        "classification": classification,
        "stable_positive_gross": stable_gross,
        "persistent_positive_net": _positive(net_evidence),
        "clearly_inferior_net_to_rule": bool(inferior),
        "gross_evidence": gross_evidence,
        "net_evidence": net_evidence,
        "net_vs_rule": relative,
        "minimum_fold_mean_gross_leverage": float(leverage.min()),
    }
