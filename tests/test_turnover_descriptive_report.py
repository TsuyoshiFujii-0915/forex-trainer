"""Report absent descriptive states without inventing supporting evidence."""

from __future__ import annotations

from pathlib import Path

import pytest

from forex_trainer.turnover_diagnostic import run_diagnostic


def test_unobserved_era_and_sparse_hold_state_have_explicit_denominators(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    if not (root / "data/jpy_9pairs_1d_2003_carry.parquet").exists():
        pytest.skip("Requires the explicitly pinned local model and data artifacts.")
    _, report = run_diagnostic(root / "configs/research/issue21_turnover_diagnostic.json", tmp_path / "result")
    hold = report["exploratory_states"]["supervised"]["membership_retained"]
    early = hold["2009-2018"]
    assert early["observed_folds"] == 0
    assert early["decisions"] == 0
    assert early["mean_net_simple_return"] is None
    late = hold["2019-2025"]
    assert late["decisions"] == 11
    assert 0 < late["observed_folds"] <= 7
    assert late["mean_net_simple_return"] is not None
    assert report["causes"]["supervised"]["retained"]["mean_share_of_trading_cost"] < .002
    assert report["primary"]["supervised"]["overnight_cost_share"]["mean"] == pytest.approx(.3273, abs=.0001)
