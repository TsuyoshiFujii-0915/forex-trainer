"""Reproducible walk-forward comparison of checkpoint-selection schemes."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import yaml

from .config import TrainerConfigError, load_experiment_config
from .model_selection import (
    ModelArtifact,
    evaluate_model_artifacts,
    load_late_checkpoint_artifacts,
)
from .run_dir import create_run_dir
from .train import run_training

SCHEMES: tuple[str, ...] = (
    "validation_best",
    "last",
    "late_checkpoint_ensemble",
)
BASELINE_SCHEME = "validation_best"
ERA_RANGES: dict[str, tuple[int, int]] = {
    "2009-2018": (2009, 2018),
    "2019-2025": (2019, 2025),
}
_REPORT_METRICS: tuple[str, ...] = (
    "annualized_net_return",
    "annualized_gross_return",
    "sharpe_annualized",
    "max_drawdown",
)


@dataclass(frozen=True)
class StudyConfig:
    """Validated checkpoint-selection study definition."""

    name: str
    fold_configs: tuple[Path, ...]
    seeds: tuple[int, ...]


@dataclass(frozen=True)
class FoldResult:
    """Metrics for one fold and one selection scheme."""

    fold: str
    scheme: str
    seeds: tuple[int, ...]
    metrics: Mapping[str, object]


def load_study_config(study_path: Path) -> StudyConfig:
    """Load and validate a checkpoint-selection study YAML.

    Args:
        study_path: YAML path. Fold paths are relative to its directory.

    Returns:
        Validated study configuration.

    Raises:
        TrainerConfigError: If the study or any fold config is invalid.
    """
    if not study_path.is_file():
        raise TrainerConfigError(f"Study config file not found: {study_path}")
    try:
        raw = yaml.safe_load(study_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise TrainerConfigError(
            f"Failed to parse study config {study_path}: {exc}"
        ) from exc
    if not isinstance(raw, dict) or set(raw) != {"name", "fold_configs", "seeds"}:
        raise TrainerConfigError(
            "Study config must contain exactly name, fold_configs, and seeds."
        )
    name = raw["name"]
    fold_values = raw["fold_configs"]
    seed_values = raw["seeds"]
    if not isinstance(name, str) or not name:
        raise TrainerConfigError("Study name must be a non-empty string.")
    if not isinstance(fold_values, list) or not fold_values:
        raise TrainerConfigError("Study fold_configs must be a non-empty list.")
    if not all(isinstance(value, str) and value for value in fold_values):
        raise TrainerConfigError("Every study fold config must be a non-empty path.")
    if not isinstance(seed_values, list) or len(seed_values) < 2:
        raise TrainerConfigError("Study seeds must contain at least two integers.")
    if not all(
        isinstance(value, int) and not isinstance(value, bool) for value in seed_values
    ):
        raise TrainerConfigError("Every study seed must be an integer.")
    if len(set(seed_values)) != len(seed_values):
        raise TrainerConfigError(f"Study seeds contain duplicates: {seed_values}")

    fold_paths = tuple((study_path.parent / value).resolve() for value in fold_values)
    fold_years: dict[str, Path] = {}
    reference_protocol: str | None = None
    for fold_path in fold_paths:
        config, fold_raw = load_experiment_config(fold_path)
        fold = config.eval_range.start[:4]
        if fold in fold_years:
            raise TrainerConfigError(
                f"Study contains duplicate evaluation fold {fold}: "
                f"{fold_years[fold]} and {fold_path}."
            )
        fold_years[fold] = fold_path
        normalized = dict(fold_raw)
        normalized["experiment"] = "<fold>"
        normalized["train_range"] = "<fold>"
        normalized["val_range"] = "<fold>"
        normalized["eval_range"] = "<fold>"
        normalized["run"] = dict(normalized["run"])
        normalized["run"]["seed"] = "<seed>"
        protocol = json.dumps(normalized, sort_keys=True, separators=(",", ":"))
        if reference_protocol is None:
            reference_protocol = protocol
        elif protocol != reference_protocol:
            raise TrainerConfigError(
                f"Fold config {fold_path} differs from the study protocol in more "
                "than experiment, ranges, or run.seed."
            )
    return StudyConfig(name=name, fold_configs=fold_paths, seeds=tuple(seed_values))


def _mean(values: list[float]) -> float:
    """Return the arithmetic mean of a non-empty list.

    Args:
        values: Numeric observations.

    Returns:
        Arithmetic mean.
    """
    return float(statistics.fmean(values))


def _metric(result: FoldResult, name: str) -> float:
    """Read one required finite numeric report metric.

    Args:
        result: Fold result containing the metric.
        name: Required metric name.

    Returns:
        Metric as a float.

    Raises:
        TrainerConfigError: If the metric is absent or non-numeric.
    """
    if name not in result.metrics:
        raise TrainerConfigError(
            f"Fold {result.fold} scheme {result.scheme} lacks metric {name}."
        )
    value = result.metrics[name]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TrainerConfigError(
            f"Fold {result.fold} scheme {result.scheme} lacks numeric metric {name}."
        )
    number = float(value)
    if not math.isfinite(number):
        raise TrainerConfigError(
            f"Fold {result.fold} scheme {result.scheme} has non-finite metric "
            f"{name}={number}."
        )
    return number


def _summary(results: list[FoldResult]) -> dict[str, object]:
    """Summarize a non-empty set of fold results.

    Args:
        results: Results for one scheme and period.

    Returns:
        Fold count, winning count, and mean metrics.
    """
    if not results:
        return {
            "folds": 0,
            "winning_folds": 0,
            **{f"mean_{name}": None for name in _REPORT_METRICS},
        }
    summary: dict[str, object] = {
        "folds": len(results),
        "winning_folds": sum(
            _metric(result, "annualized_net_return") > 0.0 for result in results
        ),
    }
    for name in _REPORT_METRICS:
        summary[f"mean_{name}"] = _mean([_metric(result, name) for result in results])
    summary["worst_max_drawdown"] = max(
        _metric(result, "max_drawdown") for result in results
    )
    return summary


def build_study_report(results: list[FoldResult]) -> dict[str, object]:
    """Build overall, era, winning-fold, and paired-difference results.

    Args:
        results: Exactly one result per fold and selection scheme.

    Returns:
        JSON-serializable report mapping.

    Raises:
        TrainerConfigError: If the fold/scheme matrix is incomplete or seeds
            differ across any comparison cell.
    """
    if not results:
        raise TrainerConfigError("Selection study produced no fold results.")
    cells: dict[tuple[str, str], FoldResult] = {}
    for result in results:
        if result.scheme not in SCHEMES:
            raise TrainerConfigError(f"Unknown selection scheme: {result.scheme}")
        key = (result.fold, result.scheme)
        if key in cells:
            raise TrainerConfigError(f"Duplicate study result for {key}.")
        cells[key] = result
    folds = sorted({result.fold for result in results})
    for fold in folds:
        missing = [scheme for scheme in SCHEMES if (fold, scheme) not in cells]
        if missing:
            raise TrainerConfigError(f"Fold {fold} is missing schemes {missing}.")
        seed_sets = {cells[(fold, scheme)].seeds for scheme in SCHEMES}
        if len(seed_sets) != 1:
            raise TrainerConfigError(
                f"Fold {fold} does not use identical seeds across schemes: {seed_sets}."
            )
    all_seed_sets = {result.seeds for result in results}
    if len(all_seed_sets) != 1:
        raise TrainerConfigError(
            f"Study folds do not use one identical seed set: {all_seed_sets}."
        )

    scheme_report: dict[str, object] = {}
    for scheme in SCHEMES:
        scheme_results = [cells[(fold, scheme)] for fold in folds]
        eras: dict[str, object] = {}
        for label, (start, end) in ERA_RANGES.items():
            era_results = [
                result for result in scheme_results if start <= int(result.fold) <= end
            ]
            eras[label] = _summary(era_results)
        scheme_report[scheme] = {
            "overall": _summary(scheme_results),
            "eras": eras,
            "folds": {
                result.fold: {name: _metric(result, name) for name in _REPORT_METRICS}
                for result in scheme_results
            },
        }

    paired: dict[str, object] = {}
    for scheme in SCHEMES:
        if scheme == BASELINE_SCHEME:
            continue
        fold_differences: dict[str, dict[str, float]] = {}
        for fold in folds:
            candidate = cells[(fold, scheme)]
            baseline = cells[(fold, BASELINE_SCHEME)]
            fold_differences[fold] = {
                name: _metric(candidate, name) - _metric(baseline, name)
                for name in _REPORT_METRICS
            }
        paired[scheme] = {
            "folds": fold_differences,
            "mean": {
                name: _mean([values[name] for values in fold_differences.values()])
                for name in _REPORT_METRICS
            },
            "median": {
                name: float(
                    statistics.median(
                        [values[name] for values in fold_differences.values()]
                    )
                )
                for name in _REPORT_METRICS
            },
            "improved_folds": {
                name: sum(
                    (values[name] > 0.0)
                    if name != "max_drawdown"
                    else (values[name] < 0.0)
                    for values in fold_differences.values()
                )
                for name in _REPORT_METRICS
            },
        }
    seeds = list(next(iter(all_seed_sets)))
    return {
        "folds": folds,
        "seeds": seeds,
        "schemes": scheme_report,
        "paired_differences_vs_validation_best": paired,
    }


def _data_identity(config_path: Path) -> dict[str, str]:
    """Return an explicit data identity for one fold config.

    Args:
        config_path: Experiment config path.

    Returns:
        Provider plus either a file SHA-256 or a synthetic-config digest.

    Raises:
        TrainerConfigError: If a file-backed cache is absent.
    """
    config, _ = load_experiment_config(config_path)
    data = config.env["data"]
    provider = str(data["provider"])
    if provider == "file":
        path = Path(str(data["path"]))
        if not path.is_file():
            raise TrainerConfigError(
                f"Study data cache is missing for {config_path}: {path}"
            )
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        return {"provider": provider, "path": str(path), "sha256": digest.hexdigest()}
    encoded = json.dumps(dict(data), sort_keys=True, separators=(",", ":")).encode()
    return {
        "provider": provider,
        "config_sha256": hashlib.sha256(encoded).hexdigest(),
    }


def _require_data_identity(
    config_path: Path, expected: Mapping[str, str]
) -> None:
    """Require a fold's data identity to remain unchanged.

    Args:
        config_path: Fold config whose data is re-hashed.
        expected: Identity captured before the study began.

    Raises:
        TrainerConfigError: If the data changed during the study.
    """
    actual = _data_identity(config_path)
    if actual != expected:
        raise TrainerConfigError(
            f"Study data changed while running {config_path}: "
            f"expected={dict(expected)}, actual={actual}."
        )


def _write_json(path: Path, value: object) -> None:
    """Write JSON through a sibling temporary file before publication.

    Args:
        path: Destination path.
        value: JSON-serializable value.
    """
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary.replace(path)


def _write_report_markdown(study_dir: Path, report: Mapping[str, object]) -> None:
    """Write a compact human-readable study report.

    Args:
        study_dir: Study artifact directory.
        report: Report produced by build_study_report.
    """
    schemes = report["schemes"]
    assert isinstance(schemes, dict)
    lines = [
        "# Checkpoint selection study",
        "",
        (
            "| Scheme | Mean fold net/year | Mean fold gross/year | "
            "Mean fold Sharpe | Worst fold drawdown | Winning folds |"
        ),
        "|---|---:|---:|---:|---:|---:|",
    ]
    for scheme in SCHEMES:
        entry = schemes[scheme]
        assert isinstance(entry, dict)
        overall = entry["overall"]
        assert isinstance(overall, dict)
        lines.append(
            f"| {scheme} | {overall['mean_annualized_net_return']:.4%} | "
            f"{overall['mean_annualized_gross_return']:.4%} | "
            f"{overall['mean_sharpe_annualized']:.3f} | "
            f"{overall['worst_max_drawdown']:.4%} | "
            f"{overall['winning_folds']}/{overall['folds']} |"
        )
    lines.extend(
        [
            "",
            (
                "The late-checkpoint scheme averages the final five validation-time "
                "checkpoints from every seed (15 equal-weight members for three seeds)."
            ),
            "Paired fold differences are recorded in `report.json`.",
        ]
    )
    (study_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_selection_study(
    study_path: Path, runs_root: Path
) -> tuple[Path, dict[str, object]]:
    """Train and compare all checkpoint schemes on one fold/seed matrix.

    Args:
        study_path: Committed study YAML.
        runs_root: Root for training and study artifacts.

    Returns:
        Study artifact directory and report.

    Raises:
        TrainerConfigError: On incompatible folds, seeds, or data identities.
    """
    study = load_study_config(study_path)
    identities = [_data_identity(path) for path in study.fold_configs]
    if len({json.dumps(item, sort_keys=True) for item in identities}) != 1:
        raise TrainerConfigError(
            "Study fold configs do not identify one identical data source: "
            f"{identities}"
        )
    study_dir = create_run_dir(runs_root, study.name)
    (study_dir / "study_snapshot.yaml").write_text(
        study_path.read_text(encoding="utf-8"), encoding="utf-8"
    )
    expected_identity = identities[0]
    _write_json(study_dir / "data_identity.json", expected_identity)

    results: list[FoldResult] = []
    serialized_rows: list[dict[str, object]] = []
    source_runs: dict[str, dict[str, str]] = {}
    _write_json(study_dir / "source_runs.json", source_runs)
    _write_json(study_dir / "fold_results.json", serialized_rows)
    for config_path in study.fold_configs:
        config, _ = load_experiment_config(config_path)
        fold = config.eval_range.start[:4]
        fold_runs: dict[int, Path] = {}
        for seed in study.seeds:
            _require_data_identity(config_path, expected_identity)
            fold_runs[seed] = run_training(config_path, runs_root, seed_override=seed)
            _require_data_identity(config_path, expected_identity)
            if fold not in source_runs:
                source_runs[fold] = {}
            source_runs[fold][str(seed)] = str(fold_runs[seed])
            _write_json(study_dir / "source_runs.json", source_runs)
        schemes: dict[str, list[ModelArtifact]] = {
            "validation_best": [
                ModelArtifact.from_run_model(fold_runs[seed], "model_final.zip")
                for seed in study.seeds
            ],
            "last": [
                ModelArtifact.from_run_model(fold_runs[seed], "model_last.zip")
                for seed in study.seeds
            ],
            "late_checkpoint_ensemble": [
                artifact
                for seed in study.seeds
                for artifact in load_late_checkpoint_artifacts(fold_runs[seed])
            ],
        }
        for scheme in SCHEMES:
            _require_data_identity(config_path, expected_identity)
            relative_dir = Path("folds") / fold / scheme
            metrics = evaluate_model_artifacts(
                schemes[scheme], study_dir / relative_dir, scheme
            )
            result = FoldResult(fold, scheme, study.seeds, metrics)
            results.append(result)
            serialized_rows.append(
                {
                    "fold": fold,
                    "scheme": scheme,
                    "seeds": list(study.seeds),
                    "artifact_dir": str(relative_dir),
                    "metrics": metrics,
                }
            )
            _write_json(study_dir / "fold_results.json", serialized_rows)

    _require_data_identity(study.fold_configs[-1], expected_identity)
    report = build_study_report(results)
    with (study_dir / "fold_results.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("fold", "scheme", "seeds", *_REPORT_METRICS),
        )
        writer.writeheader()
        for result in results:
            writer.writerow(
                {
                    "fold": result.fold,
                    "scheme": result.scheme,
                    "seeds": ",".join(str(seed) for seed in result.seeds),
                    **{name: _metric(result, name) for name in _REPORT_METRICS},
                }
            )
    _write_json(study_dir / "report.json", report)
    _write_report_markdown(study_dir, report)
    return study_dir, report


def main(argv: list[str] | None = None) -> int:
    """Run the checkpoint-selection study CLI.

    Args:
        argv: CLI arguments; None lets argparse read sys.argv.

    Returns:
        Process exit code: 0 on success, 1 on study errors.
    """
    parser = argparse.ArgumentParser(
        prog="forex-selection-study",
        description="Train and compare validation-best, last, and late checkpoints.",
    )
    parser.add_argument("--study", type=str, required=True, help="Study YAML path.")
    parser.add_argument(
        "--runs-root", type=str, required=True, help="Root directory for artifacts."
    )
    args = parser.parse_args(argv)
    try:
        study_dir, report = run_selection_study(
            Path(args.study), Path(args.runs_root)
        )
    except (TrainerConfigError, OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"selection study: {study_dir}")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
