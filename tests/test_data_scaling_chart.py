"""Committed scaling-curve artifact tests."""

from __future__ import annotations

from pathlib import Path

from forex_trainer.data_scaling import write_scaling_curve_svg


def test_scaling_curve_uses_unique_history_years_on_x_axis(tmp_path: Path) -> None:
    """The curve is indexed by unique market history, not optimizer steps."""
    report = {
        "conditions": ["2y", "expanding"],
        "summary": {
            "2y": {
                "annualized_net_return": -0.02,
                "annualized_net_return_bootstrap_95_low": -0.06,
                "annualized_net_return_bootstrap_95_high": 0.02,
                "median_fold_seed_standard_deviation": 0.08,
            },
            "expanding": {
                "annualized_net_return": 0.01,
                "annualized_net_return_bootstrap_95_low": -0.03,
                "annualized_net_return_bootstrap_95_high": 0.05,
                "median_fold_seed_standard_deviation": 0.04,
            },
        },
    }
    audits = [
        {"condition": "2y", "years": 1.99, "bars": 520},
        {"condition": "expanding", "years": 15.0, "bars": 3_900},
    ]
    output = tmp_path / "scaling.svg"

    write_scaling_curve_svg(report, audits, output)

    svg = output.read_text(encoding="utf-8")
    assert "Unique training history (years)" in svg
    assert "Annualized OOS net return" in svg
    assert "2y" in svg and "expanding" in svg
