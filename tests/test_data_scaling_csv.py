"""Portable machine-readable scaling artifact tests."""

from __future__ import annotations

from pathlib import Path

from forex_trainer.data_scaling import _write_csv


def test_scaling_csv_uses_repository_lf_line_endings(tmp_path: Path) -> None:
    """Generated CSV files must not create all-line Git whitespace warnings."""
    output = tmp_path / "result.csv"

    _write_csv(output, [{"condition": "2y", "bars": 522}])

    assert b"\r\n" not in output.read_bytes()
    assert output.read_bytes().endswith(b"\n")
