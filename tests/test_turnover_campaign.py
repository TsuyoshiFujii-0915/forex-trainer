"""Publication and fold aggregation contracts for Issue #21."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from forex_trainer.artifact_provenance import sha256_file
from forex_trainer.turnover_diagnostic import run_diagnostic


def test_invalid_manifest_does_not_publish_or_replace_output(tmp_path: Path) -> None:
    campaign = tmp_path / "campaign.json"
    campaign.write_text("{}", encoding="utf-8")
    output = tmp_path / "result"
    with pytest.raises(ValueError, match="campaign"):
        run_diagnostic(campaign, output)
    assert not output.exists()
    output.mkdir()
    with pytest.raises(FileExistsError):
        run_diagnostic(campaign, output)


def test_campaign_reproduces_costs_churn_and_fold_uncertainty(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    if not (root / "data/jpy_9pairs_1d_2003_carry.parquet").exists():
        pytest.skip("Requires the explicitly pinned local model and data artifacts.")
    output, report = run_diagnostic(root / "configs/research/issue21_turnover_diagnostic.json", tmp_path / "result")
    assert report["classification"] == "not successfully translated to portfolio alpha"
    assert report["decision"] == "unresolved"
    supervised = report["primary"]["supervised"]
    assert supervised["mean_weight_turnover"]["mean"] == pytest.approx(4.982, abs=.001)
    assert supervised["mean_membership_turnover"]["mean"] == pytest.approx(.7799, abs=.0001)
    assert supervised["total_cost_ratio"]["mean"] == pytest.approx(.05344, abs=.00001)
    folds = pd.read_csv(output / "fold_metrics.csv", dtype={"fold": str})
    assert len(folds) == 34
    values = folds.loc[folds.policy == "supervised", "mean_weight_turnover"].to_numpy()
    assert supervised["mean_weight_turnover"]["mean"] == pytest.approx(values.mean())
    with np.load(output / "bootstrap_indices.npz") as indices:
        for key in ("iid", "moving_block"):
            assert supervised["mean_weight_turnover"][key] == pytest.approx(np.quantile(values[indices[key]].mean(axis=1), [.025, .975]))
    causes = pd.read_csv(output / "cause_metrics.csv", dtype={"fold": str})
    for fold, rows in causes.groupby(["fold", "policy"]):
        metric = folds.loc[(folds.fold == fold[0]) & (folds.policy == fold[1])].iloc[0]
        assert rows.cost_jpy.sum() == pytest.approx(metric.total_cost_jpy)
        assert rows.traded_notional_jpy.sum() == pytest.approx(metric.total_traded_notional_jpy)
    provenance = json.loads((output / "provenance.json").read_text())
    for name, digest in provenance["generated_artifact_sha256"].items():
        assert sha256_file(output / name) == digest
    assert report["max_abs_cost_residual_ratio"] < 1e-12
