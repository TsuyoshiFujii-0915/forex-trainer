"""Behavioral contracts for the sealed spread-to-net diagnostic."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from forex_trainer.artifact_provenance import sha256_file
from forex_trainer.spread_decomposition import decompose_fold, load_seal, run_decomposition
from forex_trainer.supervised_portfolio import fixed_portfolio_weights


def _case() -> dict[str, Any]:
    symbols = tuple(f"pair{i}" for i in range(9))
    times = ["2020-01-01T00:00:00+00:00", "2020-01-02T00:00:00+00:00", "2020-01-04T00:00:00+00:00"]
    scores = np.tile(np.arange(9, dtype=float), (2, 1))
    relatives = np.array([[1 + (i - 4) * .001 for i in range(9)]] * 2)
    carry = np.tile(np.arange(9) * .01, (2, 1))
    weights = fixed_portfolio_weights(scores).astype(float)
    logs = np.log(relatives)
    targets = logs - logs.mean(axis=1, keepdims=True)
    predictions: list[dict[str, Any]] = []
    pairs: list[dict[str, Any]] = []
    steps: list[dict[str, Any]] = []
    before = 1000.
    for t in range(2):
        price = weights[t] * (relatives[t] - 1)
        financing = -weights[t] * relatives[t] * carry[t] * (t + 1) / 365
        gross = float((price + financing).sum())
        after = before * (1 + gross) - 1.
        for p, symbol in enumerate(symbols):
            predictions.append({"decision_timestamp": times[t], "target_timestamp": times[t+1], "pair": symbol,
                                "supervised_score": scores[t, p], "target_relative_log_return": targets[t, p]})
            pairs.append({"decision_timestamp": times[t], "pair": symbol, "predicted_rank": p+1,
                          "target_weight": weights[t, p], "price_simple_contribution": price[p],
                          "carry_simple_contribution": financing[p], "gross_simple_contribution": price[p]+financing[p],
                          "gross_log_contribution": (price[p]+financing[p])*np.log1p(gross)/gross})
        steps.append({"decision_timestamp": times[t], "target_timestamp": times[t+1], "equity_before": before,
                      "equity_after": after, "cost_jpy": 1., "financing_jpy": before*financing.sum(),
                      "gross_log_return": np.log1p(gross), "net_log_return": np.log(after/before)})
        before = after
    return {"predictions": pd.DataFrame(predictions), "steps": pd.DataFrame(steps), "pairs": pd.DataFrame(pairs),
            "symbols": symbols, "price_relatives": relatives, "carry_rates": carry, "initial_equity": 1000.,
            "policy": "supervised", "expected_weights": weights}


def test_realized_price_carry_cost_accounting_and_units() -> None:
    case = _case()
    rows, metrics = decompose_fold(**case)
    weights = case["expected_weights"]
    price = (weights * (case["price_relatives"] - 1)).sum(axis=1)
    np.testing.assert_allclose(rows["price_log_return"], np.log1p(price), atol=1e-12)
    np.testing.assert_allclose(rows["weighted_price_log_return"], rows["scaled_tail_spread"], rtol=2e-8)
    assert not np.allclose(rows["price_log_return"], rows["weighted_price_log_return"], rtol=1e-6)
    np.testing.assert_allclose(rows["net_simple_return"], rows["price_simple_return"] + rows["carry_simple_return"] - rows["cost_ratio"])
    assert metrics["years"] == pytest.approx(3 / 365.25)
    assert metrics["annualized_gross_return"] == pytest.approx(np.expm1(case["steps"]["gross_log_return"].sum() / (3 / 365.25)))
    assert metrics["total_cost_ratio"] == pytest.approx(.002)
    assert metrics["mean_tail_spread"] == pytest.approx(rows["scaled_tail_spread"].mean() / 1.6)


@pytest.mark.parametrize("table,column", [("steps", "financing_jpy"), ("steps", "cost_jpy"),
    ("steps", "equity_before"), ("steps", "gross_log_return"), ("steps", "net_log_return"),
    ("pairs", "price_simple_contribution"), ("pairs", "carry_simple_contribution"),
    ("pairs", "gross_simple_contribution"), ("pairs", "gross_log_contribution"),
    ("pairs", "target_weight"), ("pairs", "predicted_rank"), ("predictions", "target_relative_log_return")])
def test_accounting_corruption_is_rejected(table: str, column: str) -> None:
    case = _case()
    case[table].loc[0, column] += 1
    with pytest.raises(ValueError, match="supervised"):
        decompose_fold(**case)


@pytest.mark.parametrize("table", ["steps", "pairs", "predictions"])
def test_missing_duplicate_or_reordered_rows_are_not_intersected(table: str) -> None:
    for operation in ("missing", "duplicate", "reorder"):
        case = _case()
        frame = case[table]
        if operation == "missing":
            case[table] = frame.iloc[1:]
        elif operation == "duplicate":
            case[table] = pd.concat([frame, frame.iloc[:1]])
        else:
            case[table] = frame.iloc[::-1]
        with pytest.raises(ValueError, match="supervised"):
            decompose_fold(**case)


def test_target_time_mutation_fails() -> None:
    case = _case()
    case["predictions"].loc[0, "target_timestamp"] = "2020-01-03T00:00:00+00:00"
    with pytest.raises(ValueError, match="supervised"):
        decompose_fold(**case)


@pytest.mark.parametrize("column", ["cost_jpy", "equity_after"])
def test_nonfinite_accounting_fails(column: str) -> None:
    case = _case()
    case["steps"].loc[0, column] = np.nan
    with pytest.raises(ValueError, match="supervised"):
        decompose_fold(**case)


def test_seal_detects_source_change_and_manifest_resealing(tmp_path: Path) -> None:
    artifact = tmp_path / "steps.csv"
    artifact.write_text("original", encoding="utf-8")
    manifest = tmp_path / "provenance.json"
    manifest.write_text(json.dumps({"artifact_version": 1, "generated_artifact_sha256": {"steps.csv": sha256_file(artifact)}}))
    digest = sha256_file(manifest)
    assert load_seal(tmp_path, digest, {"steps.csv"})["artifact_version"] == 1
    artifact.write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="steps.csv"):
        load_seal(tmp_path, digest, {"steps.csv"})
    manifest.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="provenance"):
        load_seal(tmp_path, digest, {"steps.csv"})


def test_invalid_campaign_publishes_nothing_and_preserves_existing_output(tmp_path: Path) -> None:
    campaign = tmp_path / "campaign.json"
    campaign.write_text("{}", encoding="utf-8")
    output = tmp_path / "out"
    with pytest.raises(ValueError, match="campaign"):
        run_decomposition(campaign, output)
    assert not output.exists()
    output.mkdir()
    with pytest.raises(FileExistsError):
        run_decomposition(campaign, output)


def test_sealed_campaign_reproduces_original_metrics_and_paired_evidence(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    if not (root / "data/jpy_9pairs_1d_2003_carry.parquet").exists():
        pytest.skip("Integration requires the explicitly pinned local market and model artifacts.")
    output, report = run_decomposition(root / "configs/research/issue19_spread_decomposition.json", tmp_path / "diagnostic")
    assert report["classification"] == "not successfully translated to portfolio alpha"
    assert report["original_metrics"]["supervised"]["annualized_gross_return"]["mean_difference"] == pytest.approx(.0459, abs=.00005)
    assert report["original_metrics"]["supervised"]["annualized_net_return"]["mean_difference"] == pytest.approx(-.0264, abs=.00005)
    assert report["original_metrics"]["supervised"]["mean_tail_spread"]["mean_difference"] == pytest.approx(.00011991, abs=5e-9)
    folds = pd.read_csv(output / "fold_metrics.csv", dtype={"fold": str})
    assert len(folds) == 51
    for stage in ("scaled_tail_spread", "weighted_price_log_return", "price_log_return", "gross_log_return", "net_log_return"):
        assert len(report["stages"][stage]["leave_one_fold_out_means"]) == 17
    delta = report["stage_differences"]["gross_log_return_minus_price_log_return"]
    gross = report["stages"]["gross_log_return"]
    price = report["stages"]["price_log_return"]
    assert delta["mean_difference"] == pytest.approx(gross["mean_difference"] - price["mean_difference"])
    for fold in delta["fold_differences"]:
        assert delta["fold_differences"][fold] == pytest.approx(gross["fold_differences"][fold] - price["fold_differences"][fold])
    provenance = json.loads((output / "provenance.json").read_text())
    for name, digest in provenance["generated_artifact_sha256"].items():
        assert sha256_file(output / name) == digest
    indices = np.load(output / "bootstrap_indices.npz")
    values = np.array(list(delta["fold_differences"].values()))
    for key, prefix in (("iid", "fold"), ("moving_block", "moving_block")):
        draws = values[indices[key]].mean(axis=1)
        assert delta["intervals"][f"{prefix}_low"] == pytest.approx(np.quantile(draws, .025))
        assert delta["intervals"][f"{prefix}_high"] == pytest.approx(np.quantile(draws, .975))
