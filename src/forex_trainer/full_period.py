"""Explicit legacy and history-separated historical evaluation (ADR-0037)."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
import tempfile
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from importlib.metadata import version
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from forex_env import parse_config as parse_env_config
from forex_env.data.base import validate_ohlcv
from forex_env.data.file_provider import save_ohlcv_parquet
from forex_env.errors import DataError
from forex_env.features import FeaturePipeline
from stable_baselines3.common.monitor import Monitor

from .artifact_provenance import dependency_versions, git_commits, sha256_file
from .env_factory import GateEvaluationMode, build_single_env
from .evaluate import _run_evaluation, compute_metrics
from .features import CROSS_FEATURE_REGISTRY, FEATURE_REGISTRY
from .regime_study import RuleSpec, build_momentum_reversal_action
from .supervised_portfolio import fixed_portfolio_weights
from .supervised_ranking import RidgeModel, Standardizer, apply_standardizer

MEASUREMENT_ID = "full-period-history-v1"
FEATURES = (
    "log_return",
    "volatility",
    "sma20_ratio",
    "mom24",
    "xz_mom24",
    "xr_mom24",
    "carry_annual",
    "xz_carry",
)
FEATURE_NAMES = tuple(
    f"{name}_lag_{lag}" for lag in range(31, -1, -1) for name in FEATURES
)
Policy = Callable[[dict[str, np.ndarray], np.ndarray], tuple[np.ndarray, np.ndarray]]


def json_hash(value: Any) -> str:
    """Hash a JSON value with deterministic finite encoding.

    Args:
        value: JSON-serializable identity.

    Returns:
        SHA-256 hex digest.
    """
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    """Read a required JSON mapping with its origin in any error.

    Args:
        path: Input path.

    Returns:
        Parsed mapping.
    """
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError(f"Cannot read JSON source {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise TypeError(f"Expected JSON mapping: {path}")
    return value


def checked_path(reference: Mapping[str, str]) -> Path:
    """Verify a mandatory explicit source path and hash.

    Args:
        reference: Path and SHA-256 mapping.

    Returns:
        Verified absolute file path.
    """
    if set(reference) != {"path", "sha256"}:
        raise ValueError(f"Source identity requires path and sha256: {reference}")
    path = Path(reference["path"]).resolve()
    expected = reference["sha256"]
    if not path.is_file():
        raise ValueError(f"missing source {path}; expected sha256={expected}")
    actual = sha256_file(path)
    if actual != expected:
        raise ValueError(
            f"source hash mismatch {path}: expected={expected}, actual={actual}"
        )
    return path


def utc_instant(value: str, origin: str) -> pd.Timestamp:
    """Require an aware instant and convert it to UTC without guessing.

    Args:
        value: ISO timestamp with timezone.
        origin: Error context.

    Returns:
        UTC timestamp.
    """
    try:
        time = pd.Timestamp(value)
    except (ValueError, TypeError) as exc:
        raise ValueError(f"Invalid instant {value!r}: {origin}") from exc
    if pd.isna(time) or time.tzinfo is None:
        raise ValueError(f"Timezone-aware instant required for {value!r}: {origin}")
    return time.tz_convert("UTC")


@dataclass(frozen=True)
class PeriodPlan:
    """Validated exact input prefix and measured bar set."""

    market: pd.DataFrame
    timestamps: tuple[str, ...]
    symbols: tuple[str, ...]
    measurement_start: str
    measurement_end: str
    history_labels: tuple[str, ...]
    calendar: dict[str, Any]
    origin: str


def prepare_period(
    market: pd.DataFrame,
    calendar: Mapping[str, Any],
    start: str,
    end: str,
    symbols: tuple[str, ...],
    origin: Path,
) -> PeriodPlan:
    """Validate a registered calendar and separate 63 history bars from PnL.

    Args:
        market: Source data with original bar labels, including history.
        calendar: Explicit pair order, session mapping, and holiday declaration.
        start: Inclusive measurement instant.
        end: Exclusive measurement instant.
        symbols: Canonical ordered pair axis.
        origin: Source path for actionable errors.

    Returns:
        Exact 63-bar prefix plus all registered bars in the half-open interval.
    """
    context = str(origin)
    begin, stop = utc_instant(start, context), utc_instant(end, context)
    if begin >= stop:
        raise ValueError(f"measurement_start must precede measurement_end: {origin}")
    required = {"version", "timezone", "symbols", "holidays", "label_rule", "sessions"}
    if set(calendar) != required:
        raise ValueError(f"calendar fields must equal {sorted(required)}: {origin}")
    if tuple(calendar["symbols"]) != symbols or len(set(symbols)) != 9:
        raise ValueError(f"calendar pair order mismatch, expected {symbols}: {origin}")
    for field in ("version", "timezone", "label_rule"):
        if not isinstance(calendar[field], str) or not calendar[field].strip():
            raise ValueError(f"calendar.{field} must be explicit: {origin}")
    try:
        ZoneInfo(calendar["timezone"])
    except (ValueError, KeyError) as exc:
        raise ValueError(f"Invalid calendar timezone: {origin}") from exc
    if not isinstance(calendar["holidays"], list):
        raise TypeError(f"calendar.holidays must be a list: {origin}")
    sessions = calendar["sessions"]
    if not isinstance(sessions, list) or len(sessions) < 2:
        raise ValueError(f"calendar requires explicit sessions: {origin}")
    closes: list[pd.Timestamp] = []
    labels: list[pd.Timestamp] = []
    for row in sessions:
        if set(row) != {"bar_label", "session_open", "session_close", "available_at"}:
            raise ValueError(f"Malformed calendar session: {origin}: {row}")
        label = pd.Timestamp(row["bar_label"])
        close = utc_instant(row["session_close"], context)
        opened = utc_instant(row["session_open"], context)
        if opened >= close:
            raise ValueError(f"session_open must precede close: {origin}: {row}")
        available = row["available_at"]
        if available is not None and utc_instant(available, context) > close:
            raise ValueError(
                f"Input available_at after same-close decision: {origin}: {row}"
            )
        if str(label.date()) in calendar["holidays"]:
            raise ValueError(f"Session contradicts registered holiday: {origin}: {row}")
        labels.append(label)
        closes.append(close)
    aware_labels = [label.tzinfo is not None for label in labels]
    if any(aware_labels) and not all(aware_labels):
        raise ValueError(f"Mixed aware and naive calendar bar labels: {origin}")
    label_index = pd.DatetimeIndex(
        [label.tz_convert("UTC") for label in labels] if all(aware_labels) else labels
    )
    close_index = pd.DatetimeIndex(closes)
    if (
        label_index.hasnans
        or not label_index.is_unique
        or not label_index.is_monotonic_increasing
        or not close_index.is_unique
        or not close_index.is_monotonic_increasing
    ):
        raise ValueError(f"calendar labels and closes must strictly increase: {origin}")
    selected = np.flatnonzero((close_index >= begin) & (close_index < stop))
    if len(selected) < 2:
        raise ValueError(
            f"measurement requires a decision and later mark before E: {origin}"
        )
    first, last = int(selected[0]), int(selected[-1])
    if first < 63:
        raise ValueError(
            f"63 history bars required; only {first} before first decision: {origin}"
        )
    expected = label_index[first - 63 : last + 1]
    if not isinstance(market.index, pd.DatetimeIndex) or (market.index.tz is None) != (
        label_index.tz is None
    ):
        raise ValueError(f"source/calendar bar label timezone mismatch: {origin}")
    if market.index.tz is not None:
        market = market.copy()
        market.index = market.index.tz_convert("UTC")
    missing = expected.difference(market.index)
    if len(missing):
        raise ValueError(
            f"Missing expected bar for pairs {symbols} at {missing[0]}: {origin}"
        )
    actual = market.loc[expected[0] : expected[-1]]
    if not actual.index.equals(expected):
        raise ValueError(
            f"Unexpected or duplicate bar in registered calendar range: {origin}"
        )
    actual_symbols = tuple(dict.fromkeys(symbol for symbol, _ in actual.columns))
    if actual_symbols != symbols:
        raise ValueError(
            f"source pair order mismatch {actual_symbols}, expected {symbols}: {origin}"
        )
    try:
        validate_ohlcv(actual, symbols)
        for symbol in symbols:
            if (symbol, "CarryAnnual") not in actual.columns:
                raise ValueError(f"Missing CarryAnnual for {symbol}")
            if not np.isfinite(
                actual[(symbol, "CarryAnnual")].to_numpy(dtype=float)
            ).all():
                raise ValueError(f"Non-finite CarryAnnual for {symbol}")
    except (ValueError, DataError) as exc:
        raise ValueError(f"Invalid market data at {origin}: {exc}") from exc
    mapped = actual.copy()
    mapped.index = close_index[first - 63 : last + 1]
    return PeriodPlan(
        market=mapped,
        timestamps=tuple(time.isoformat() for time in close_index[first : last + 1]),
        symbols=symbols,
        measurement_start=begin.isoformat(),
        measurement_end=stop.isoformat(),
        history_labels=tuple(row["bar_label"] for row in sessions[first - 63 : first]),
        calendar=copy.deepcopy(dict(calendar)),
        origin=context,
    )


def build_period_env(
    env_raw: Mapping[str, Any],
    plan: PeriodPlan,
    cache_path: Path,
) -> tuple[Monitor, np.ndarray]:
    """Build the unchanged env from a validated temporary input cache.

    Args:
        env_raw: Original environment settings, including source cost contract.
        plan: Validated history and measurement plan.
        cache_path: New private cache path.

    Returns:
        Fresh environment and float64 ridge windows, one per decision.
    """
    raw = copy.deepcopy(dict(env_raw))
    environment, features = raw["environment"], raw["features"]
    if (
        environment["window_size"] != 32
        or features["volatility_window"] != 32
        or tuple(features["selected"]) != FEATURES
        or features["normalize"] is not False
        or environment["initial_balance_jpy"] != 1_000_000
        or environment["allow_action_leverage"] is not False
        or tuple(environment["currency_pairs"]) != plan.symbols
    ):
        raise ValueError(
            f"full-period requires current longf geometry and flat 1000000 JPY reset: {plan.origin}"
        )
    if raw["data"]["timeframe"] != "1d":
        raise ValueError(f"full-period requires daily source data: {plan.origin}")
    if cache_path.exists():
        raise ValueError(f"staged cache already exists: {cache_path}")
    begin = str(plan.market.index[0].date())
    stop = str(plan.market.index[-1].date())
    save_ohlcv_parquet(plan.market, "1d", begin, stop, cache_path)
    raw["data"] = {
        "provider": "file",
        "path": str(cache_path),
        "timeframe": "1d",
        "start_date": begin,
        "end_date": stop,
    }
    environment["random_start"] = False
    environment["episode_max_steps"] = len(plan.timestamps)
    per_pair = tuple(name for name in FEATURES if name in FEATURE_REGISTRY)
    cross = tuple(name for name in FEATURES if name in CROSS_FEATURE_REGISTRY)
    pipeline = FeaturePipeline(
        parse_env_config(raw).features,
        {name: FEATURE_REGISTRY[name] for name in per_pair},
        custom_cross_features={name: CROSS_FEATURE_REGISTRY[name] for name in cross},
    )
    tensor = pipeline.compute(plan.market, plan.symbols)
    windows = np.stack(
        [
            tensor[:, index - 31 : index + 1, :]
            for index in range(63, len(plan.market) - 1)
        ]
    )
    env = build_single_env(
        raw,
        per_pair,
        cross,
        seed=0,
        decision_interval=1,
        residual=None,
        rank_allocation=None,
        apply_hold_gate=None,
        gate_evaluation_mode=GateEvaluationMode.LEARNED,
    )
    return env, windows


class FrozenRidge:
    """Frozen source standardizer, selected alpha, and float64 coefficients."""

    def __init__(self, record: Mapping[str, Any], origin: Path) -> None:
        """Load parameters without fitting or changing their feature coordinates.

        Args:
            record: One fold from the sealed models.json.
            origin: Source model artifact.
        """
        if tuple(record["feature_names"]) != FEATURE_NAMES:
            raise ValueError(f"ridge feature order mismatch: {origin}")
        arrays: dict[str, np.ndarray] = {}
        for name in ("train_mean", "train_scale", "coefficients"):
            array = np.array(record[name], dtype=np.float64, copy=True)
            if array.shape != (256,) or not np.isfinite(array).all():
                raise ValueError(f"Invalid ridge {name}: {origin}")
            array.setflags(write=False)
            arrays[name] = array
        if np.any(arrays["train_scale"] <= 0):
            raise ValueError(f"ridge train_scale must be positive: {origin}")
        intercept, alpha = float(record["intercept"]), float(record["selected_alpha"])
        if not np.isfinite([intercept, alpha]).all() or alpha < 0:
            raise ValueError(f"Invalid ridge intercept/alpha: {origin}")
        self.standardizer = Standardizer(arrays["train_mean"], arrays["train_scale"])
        self.model = RidgeModel(arrays["coefficients"], intercept, alpha)
        self.parameter_sha256 = json_hash(
            {
                name: record[name]
                for name in (
                    "feature_names",
                    "train_mean",
                    "train_scale",
                    "coefficients",
                    "intercept",
                    "selected_alpha",
                )
            }
        )

    def predict(self, window: np.ndarray) -> np.ndarray:
        """Predict fresh scores from a causal float64 window.

        Args:
            window: Nine pairs by 32 observations by eight features.

        Returns:
            Nine frozen-model scores.
        """
        if window.shape != (9, 32, 8):
            raise ValueError(f"ridge window must be (9, 32, 8), got {window.shape}")
        return self.model.predict(
            apply_standardizer(window.reshape(1, 9, 256), self.standardizer)[0]
        )

    def action(
        self, observation: dict[str, np.ndarray], window: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Apply the sealed fixed map to float64 model scores.

        Args:
            observation: Shared env observation; ridge has no account input.
            window: Causal float64 feature window.

        Returns:
            Scores and float32 direct allocation.
        """
        scores = self.predict(window)
        return scores, fixed_portfolio_weights(scores[None, :])[0, :, None]


def canonical_action(
    observation: dict[str, np.ndarray], window: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Infer canonical reversal with the existing float32 tie contract.

    Args:
        observation: Actual current observation.
        window: Ridge precision window, unused by the established rule.

    Returns:
        Negative-momentum diagnostics and canonical allocations.
    """
    action = build_momentum_reversal_action(
        observation, FEATURES, RuleSpec(feature="mom24", top_k=2, base_size=0.8)
    )
    return -observation["market"][:, -1, FEATURES.index("mom24")].astype(float), action


def evaluate_policy(
    env: Monitor,
    windows: np.ndarray,
    plan: PeriodPlan,
    predict: Policy,
) -> dict[str, Any]:
    """Walk a fresh account and preserve terminal failures and exact times.

    Args:
        env: Environment constructed from the plan.
        windows: Float64 windows aligned with planned decisions.
        plan: Calendar and measurement contract.
        predict: Frozen policy adapter using this account's current assets.

    Returns:
        Metrics, accounting trace, coverage, and comparison identity.
    """
    observation, info = env.reset(seed=0)
    if (
        info["timestamp"] != plan.timestamps[0]
        or info["equity_jpy"] != 1_000_000
        or any(info["exposures_jpy"].values())
        or np.any(observation["assets"])
    ):
        raise ValueError(
            f"Incorrect initial portfolio or first decision: {plan.origin}"
        )
    initial_assets = observation["assets"].tolist()
    equities, times = [float(info["equity_jpy"])], [str(info["timestamp"])]
    rewards: list[float] = []
    costs: list[float] = []
    leverages: list[float] = []
    turnovers: list[float] = []
    trace: list[dict[str, Any]] = []
    previous = np.zeros(9)
    terminated = False
    for index, window in enumerate(windows):
        if str(info["timestamp"]) != plan.timestamps[index]:
            raise ValueError(f"decision time mismatch at {index}: {plan.origin}")
        if not np.array_equal(observation["market"], window.astype(np.float32)):
            raise ValueError(
                f"Ridge and env observation windows differ at {info['timestamp']}: {plan.origin}"
            )
        assets_before = observation["assets"].tolist()
        scores, action = predict(observation, window)
        if (
            scores.shape != (9,)
            or not np.isfinite(scores).all()
            or action.shape != (9, 1)
        ):
            raise ValueError(
                f"Malformed frozen policy output at {info['timestamp']}: {plan.origin}"
            )
        equity_before = float(info["equity_jpy"])
        decision = str(info["timestamp"])
        observation, reward, terminated, truncated, info = env.step(action)
        if str(info["timestamp"]) != plan.timestamps[index + 1]:
            raise ValueError(f"target time mismatch at {index}: {plan.origin}")
        weights = np.asarray(info["target_weights"], dtype=float)
        turnover = float(np.abs(weights - previous).sum())
        previous = weights
        trace.append(
            {
                "decision_timestamp": decision,
                "target_timestamp": str(info["timestamp"]),
                "equity_before": equity_before,
                "equity_jpy": float(info["equity_jpy"]),
                "scores": scores.tolist(),
                "action": action[:, 0].tolist(),
                "target_weights": weights.tolist(),
                "assets_before": assets_before,
                "exposures_jpy": info["exposures_jpy"],
                "reward": float(reward),
                "spread_jpy": float(info["costs_jpy"]["spread"]),
                "commission_jpy": float(info["costs_jpy"]["commission"]),
                "overnight_jpy": float(info["costs_jpy"]["overnight"]),
                "financing_jpy": float(info["financing_jpy"]),
                "cost_jpy": float(info["costs_jpy"]["total"]),
                "gross_leverage": float(info["gross_leverage"]),
                "weight_turnover": turnover,
                "terminated": bool(terminated),
                "truncated": bool(truncated),
            }
        )
        rewards.append(float(reward))
        costs.append(float(info["costs_jpy"]["total"]))
        leverages.append(float(info["gross_leverage"]))
        turnovers.append(turnover)
        equities.append(float(info["equity_jpy"]))
        times.append(str(info["timestamp"]))
        if terminated:
            break
        if truncated != (index == len(windows) - 1):
            raise ValueError(
                f"Unexpected truncation at {info['timestamp']}: {plan.origin}"
            )
    first, last = pd.Timestamp(times[0]), pd.Timestamp(times[-1])
    coverage = {
        "measurement_start": plan.measurement_start,
        "measurement_end": plan.measurement_end,
        "first_decision": times[0],
        "last_mark": times[-1],
        "planned_last_mark": plan.timestamps[-1],
        "initial_flat_seconds": (
            first - pd.Timestamp(plan.measurement_start)
        ).total_seconds(),
        "trailing_coverage_gap_seconds": (
            pd.Timestamp(plan.measurement_end) - last
        ).total_seconds(),
        "elapsed_seconds": (last - first).total_seconds(),
    }
    if equities[-1] <= 0:
        if not terminated:
            raise ValueError(
                f"Non-positive equity without terminal margin call: {plan.origin}"
            )
        metrics = {
            "steps": len(rewards),
            "terminated_by_margin_call": True,
            "undefined_reason": "non_positive_terminal_equity",
            "annualized_net_return": None,
            "annualized_gross_return": None,
            "cumulative_log_return": None,
            "gross_cumulative_log_return": None,
            "sharpe_annualized": None,
            "environment_cumulative_reward": float(sum(rewards)),
            "final_equity_ratio": equities[-1] / equities[0],
            "max_drawdown": float(
                np.max(1 - np.array(equities) / np.maximum.accumulate(equities))
            ),
            "total_cost_ratio": sum(costs) / equities[0],
            "mean_gross_leverage": float(np.mean(leverages)),
            "mean_weight_turnover": float(np.mean(turnovers)),
            "total_weight_turnover": sum(turnovers),
            "eval_start": times[0],
            "eval_end": times[-1],
        }
    else:
        metrics = compute_metrics(
            rewards, equities, times, costs, leverages, turnovers, terminated
        )
    return {
        "measurement_id": MEASUREMENT_ID,
        "status": "incomplete_margin_call" if terminated else "complete",
        "symbols": list(plan.symbols),
        "timestamps": times,
        "initial_assets": initial_assets,
        "runtime": runtime_identity(),
        "market_sha256": hashlib.sha256(
            plan.market.to_numpy(dtype=np.float64).tobytes()
        ).hexdigest(),
        "environment": {
            "max_leverage": env.unwrapped._max_leverage,
            "margin_call_threshold": env.unwrapped._margin_call_threshold,
            "features": list(FEATURES),
            "normalize": False,
            "window_size": 32,
            "decision_interval": 1,
            "initial_balance_jpy": 1_000_000,
            "execution_model": "historical_same_close",
        },
        "costs": asdict(env.unwrapped._costs_config),
        "coverage": coverage,
        "metrics": metrics,
        "trace": trace,
    }


def runtime_identity() -> dict[str, Any]:
    """Record numerical dependencies and actual source bytes as well as HEAD.

    Returns:
        Complete evaluation runtime provenance, including uncommitted source.
    """
    import forex_env

    roots = {
        "forex_trainer": Path(__file__).parent,
        "forex_env": Path(forex_env.__file__).parent,
    }
    return {
        "git": git_commits(),
        "device": "cpu",
        "versions": {
            **dependency_versions(),
            **{
                name: version(name) for name in ("numpy", "pandas", "pyarrow", "PyYAML")
            },
        },
        "source_sha256": {
            name: {
                str(path.relative_to(root)): sha256_file(path)
                for path in sorted(root.rglob("*.py"))
            }
            for name, root in roots.items()
        },
    }


def require_comparable(results: list[dict[str, Any]]) -> None:
    """Reject partial or mixed measurement conditions before any paired report.

    Args:
        results: Policy observations from the same fold.
    """
    if not results:
        raise ValueError("No results to compare")
    for result in results:
        if result["status"] != "complete":
            raise ValueError(
                "incomplete margin-call result cannot enter paired comparison"
            )
        for field in (
            "measurement_id",
            "runtime",
            "symbols",
            "timestamps",
            "costs",
            "coverage",
            "market_sha256",
            "environment",
        ):
            if result[field] != results[0][field]:
                raise ValueError(
                    f"Different {field}; re-evaluate all policies under one measurement contract"
                )
        if result["measurement_id"] != MEASUREMENT_ID:
            raise ValueError(
                "Legacy measurement cannot enter full-period paired comparison"
            )


def _write_json(path: Path, value: Any) -> None:
    """Write finite JSON artifacts.

    Args:
        path: New output file.
        value: Artifact content.
    """
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def run_period_campaign(config_path: Path, output: Path) -> dict[str, Any]:
    """Validate sources, evaluate independent folds, and publish a new artifact.

    Args:
        config_path: Explicit legacy/full_period JSON configuration.
        output: New output directory; existing destinations are rejected.

    Returns:
        Complete or terminal-incomplete campaign result.
    """
    from .full_period_sources import load_campaign_sources

    if output.exists():
        raise ValueError(f"Output already exists: {output}")
    config = read_json(config_path)
    if config["mode"] == "legacy":
        if set(config) != {"mode", "run_dir"}:
            raise ValueError(f"legacy requires only mode and run_dir: {config_path}")
        output.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(
            dir=output.parent, prefix=".legacy-"
        ) as temporary:
            staged = Path(temporary) / "result"
            staged.mkdir()
            metrics = _run_evaluation(
                Path(config["run_dir"]), staged, GateEvaluationMode.LEARNED
            )
            _write_json(
                staged / "period-mode.json",
                {"measurement_id": "legacy", "source": config},
            )
            staged.rename(output)
        return {"measurement_id": "legacy", "metrics": metrics}
    sources = load_campaign_sources(config, config_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {
        "measurement_id": MEASUREMENT_ID,
        "evidence_class": "development_historical",
        "status": "complete",
        "folds": {},
    }
    manifest: dict[str, Any] = {
        "measurement_id": MEASUREMENT_ID,
        "config": config,
        "config_sha256": sha256_file(config_path),
        "command": [
            "forex-period-eval",
            "--config",
            str(config_path.resolve()),
            "--output",
            str(output.resolve()),
        ],
        "runtime": runtime_identity(),
        "folds": {},
    }
    with tempfile.TemporaryDirectory(
        dir=output.parent, prefix=".full-period-"
    ) as temporary:
        root = Path(temporary)
        staged = root / "result"
        staged.mkdir()
        for source in sources:
            fold, plan = source.fold, source.plan
            results: dict[str, Any] = {}
            for name, adapter in (
                ("ridge", source.ridge.action),
                ("canonical", canonical_action),
                ("ppo_ens3", source.ppo.action),
            ):
                env, windows = build_period_env(
                    source.env_raw, plan, root / f"{fold}-{name}.parquet"
                )
                try:
                    results[name] = evaluate_policy(env, windows, plan, adapter)
                finally:
                    env.close()
            report["folds"][fold] = results
            manifest["folds"][fold] = {
                "source": source.identity,
                "calendar": plan.calendar,
                "history_labels": list(plan.history_labels),
                "history_instants": [
                    time.isoformat() for time in plan.market.index[:63]
                ],
                "measurement_start": plan.measurement_start,
                "measurement_end": plan.measurement_end,
                "effective_timestamps": list(plan.timestamps),
                "effective_timestamps_sha256": json_hash(plan.timestamps),
                "ridge_parameter_sha256": source.ridge.parameter_sha256,
                "initial_equity_jpy": 1_000_000,
                "initial_exposures": [0.0] * 9,
                "features": list(FEATURES),
                "normalize": False,
            }
            if any(result["status"] != "complete" for result in results.values()):
                report["status"] = "incomplete_margin_call"
            else:
                require_comparable(list(results.values()))
        all_rows: list[dict[str, Any]] = []
        lines = [
            "# History-separated historical evaluation",
            "",
            f"Measurement: `{MEASUREMENT_ID}`; evidence: `development_historical`.",
            "",
            f"Status: `{report['status']}`. Metrics annualize actual first-decision to last-mark elapsed time.",
            "Independent folds reset to JPY 1,000,000; no continuous-operation CAGR is reported.",
            "",
            "| Fold | Policy | Status | First decision | Last mark | Steps | Net annualized return |",
            "|---|---|---|---|---|---:|---:|",
        ]
        for fold, results in report["folds"].items():
            for policy, result in results.items():
                metric, coverage = result["metrics"], result["coverage"]
                net = metric["annualized_net_return"]
                displayed_net = "undefined (bankrupt)" if net is None else f"{net:.8f}"
                lines.append(
                    f"| {fold} | {policy} | {result['status']} | {coverage['first_decision']} | {coverage['last_mark']} | {metric['steps']} | {displayed_net} |"
                )
                for row in result["trace"]:
                    all_rows.append(
                        {
                            "measurement_id": MEASUREMENT_ID,
                            "fold": fold,
                            "policy": policy,
                            **{
                                key: json.dumps(value, separators=(",", ":"))
                                if isinstance(value, (list, dict))
                                else value
                                for key, value in row.items()
                            },
                        }
                    )
        if report["status"] != "complete":
            lines.extend(
                [
                    "",
                    "Paired comparison stopped: terminal margin-call observations are preserved, not removed.",
                ]
            )
        _write_json(staged / "report.json", report)
        (staged / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        pd.DataFrame(all_rows).to_csv(staged / "steps.csv", index=False)
        manifest["generated_artifact_sha256"] = {
            name: sha256_file(staged / name)
            for name in ("report.json", "report.md", "steps.csv")
        }
        _write_json(staged / "manifest.json", manifest)
        staged.rename(output)
    return report


def main() -> int:
    """Run the explicitly configured evaluation CLI.

    Returns:
        Zero for a complete result, two for terminal strategy failure.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    try:
        result = run_period_campaign(args.config, args.output)
    except (ValueError, TypeError, OSError, KeyError, DataError) as exc:
        print(
            f"full-period evaluation failed for {args.config}: {exc}", file=sys.stderr
        )
        return 1
    print(f"Wrote {result['measurement_id']} evaluation to {args.output}")
    if result["measurement_id"] == "legacy":
        return 0
    return 2 if result["status"] == "incomplete_margin_call" else 0


if __name__ == "__main__":
    raise SystemExit(main())
