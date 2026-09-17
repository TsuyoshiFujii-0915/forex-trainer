"""One-cycle development shadow recording; never an independent confirmation run."""

from __future__ import annotations

import argparse
import copy
import json
import os
import uuid
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from jsonschema import Draft202012Validator, FormatChecker, ValidationError
from zoneinfo import ZoneInfo
from forex_env.config import FeaturesConfig
from forex_env.errors import DataError, FeatureError
from forex_env.data.base import validate_ohlcv
from forex_env.features import FeaturePipeline

from .config import parse_experiment_config
from .features import FEATURE_REGISTRY, CROSS_FEATURE_REGISTRY
from .full_period import (
    FEATURES,
    FEATURE_NAMES,
    Policy,
    checked_path,
    read_json,
    runtime_identity,
    json_hash,
)
from .quote_replay import (
    execute_quote,
    initial_state,
    instant,
    validate as validate_execution,
)
from .quote_replay import SYMBOLS, SCHEMA as EXECUTION_SCHEMA
from .quote_replay_policies import fixture_policies, frozen_policies
from .shadow_store import (
    EventLog,
    artifact_path,
    digest,
    encode,
    now,
    publish,
    read_events,
    sync_directory,
)

POLICIES = {
    "canonical": "canonical_reversal",
    "ridge": "supervised_fixed_map",
    "ppo_ens3": "direct_longf_ens3",
}
FIELDS = ["Open", "High", "Low", "Close", "Volume", "CarryAnnual"]
IDENTITIES = (
    Path(__file__).resolve().parents[2]
    / "docs/research/protocols/issue20/artifact-identities.json"
)


def require_keys(value: dict[str, Any], keys: set[str], origin: str) -> None:
    """Reject missing or undeclared fields at a configuration boundary."""
    if not isinstance(value, dict) or set(value) != keys:
        raise ValueError(f"{origin}: fields must equal {sorted(keys)}")


def utc(value: str) -> str:
    """Normalize an explicitly aware timestamp to the event schema's UTC form."""
    return instant(value).isoformat().replace("+00:00", "Z")


def validate_manifest(manifest: dict[str, Any]) -> None:
    """Require the bounded registration, source, schedule and storage contract."""
    require_keys(
        manifest,
        {
            "manifest_version",
            "evidence_class",
            "mode",
            "trial_id",
            "registered_at",
            "measurement_start",
            "measurement_end",
            "source",
            "initial_state",
            "policy_mode",
            "policy_sources",
            "feature_config",
            "execution",
            "expected_costs",
            "schedule",
            "storage",
            "failure_storage",
            "stop_policy",
            "holdout_eligible",
        },
        "manifest",
    )
    encode(manifest)
    if (
        manifest["manifest_version"] != 1
        or manifest["evidence_class"] != "development_shadow"
        or manifest["holdout_eligible"] is not False
        or manifest["stop_policy"] != "halt_all_preserve_records_no_backfill"
    ):
        raise ValueError(
            "Only the registered non-holdout development shadow contract is supported"
        )
    if manifest["mode"] not in {"fixture", "observed_file"}:
        raise ValueError("Explicit fixture or observed_file mode required")
    if not isinstance(manifest["trial_id"], str) or not manifest["trial_id"].strip():
        raise ValueError("Explicit trial_id required")
    require_keys(
        manifest["source"],
        {"id", "access", "timestamp_meaning", "carry_lineage"},
        "source",
    )
    if manifest["source"]["access"] != "read_only_file" or any(
        not isinstance(v, str) or not v.strip() for v in manifest["source"].values()
    ):
        raise ValueError(
            "Explicit read-only source, timestamp meaning and carry lineage required"
        )
    if manifest["initial_state"] != {
        "equity_jpy": 1_000_000.0,
        "quantity_base": [0.0] * 9,
        "assets": [[0.0] * 3 for _ in range(9)],
    }:
        raise ValueError("Initial account must be flat 1000000 JPY with zero assets")
    begin, end = (
        instant(manifest["measurement_start"]),
        instant(manifest["measurement_end"]),
    )
    if not instant(manifest["registered_at"]) < begin < end:
        raise ValueError("Registration must precede the half-open measurement range")
    store, failure = (
        Path(manifest["storage"]).resolve(),
        Path(manifest["failure_storage"]).resolve(),
    )
    if store == failure or store in failure.parents or failure in store.parents:
        raise ValueError("Separate non-nested failure_storage is required")
    costs = manifest["expected_costs"]
    require_keys(
        costs, {"transaction_cost_jpy", "signed_carry_jpy", "quality"}, "expected_costs"
    )
    if costs["quality"] == "unavailable":
        if (
            costs["transaction_cost_jpy"] is not None
            or costs["signed_carry_jpy"] is not None
        ):
            raise ValueError(
                "Unavailable cost estimates must be null, never fabricated zero"
            )
    elif costs["quality"] == "assumption":
        if (
            not all(
                isinstance(costs[k], (int, float))
                for k in ("transaction_cost_jpy", "signed_carry_jpy")
            )
            or costs["transaction_cost_jpy"] < 0
        ):
            raise ValueError(
                "Explicit finite assumed cost and signed carry estimates required"
            )
    else:
        raise ValueError("Unknown expected cost quality")
    steps = manifest["schedule"]
    if not steps:
        raise ValueError("Nonempty explicit cycle schedule required")
    ids: set[str] = set()
    previous_target: str | None = None
    for step in steps:
        require_keys(
            step,
            {
                "cycle_id",
                "session_close",
                "decision_not_before",
                "decision_deadline",
                "target_at",
                "raw_path",
                "settlement_path",
                "history_event_times",
            },
            "schedule",
        )
        close, target = instant(step["session_close"]), instant(step["target_at"])
        if (
            not begin
            <= close
            <= instant(step["decision_not_before"])
            <= instant(step["decision_deadline"])
            < target
            < end
        ):
            raise ValueError(
                f"Invalid decision deadline/target/range: {step['cycle_id']}"
            )
        if step["cycle_id"] in ids or not step["cycle_id"]:
            raise ValueError("duplicate or empty cycle ID")
        ids.add(step["cycle_id"])
        if previous_target is not None and instant(previous_target) != close:
            raise ValueError("Each next decision must equal the preceding mark")
        previous_target = step["target_at"]
        history = [instant(t) for t in step["history_event_times"]]
        if (
            len(history) != 64
            or history[-1] != close
            or any(a >= b for a, b in zip(history, history[1:]))
        ):
            raise ValueError(
                "Exactly 63 registered history rows plus the decision row required"
            )
    config = parse_experiment_config(
        yaml.safe_load(checked_path(manifest["feature_config"]).read_text())
    )
    env = config.env
    if (
        tuple(env["environment"]["currency_pairs"]) != SYMBOLS
        or env["environment"]["window_size"] != 32
        or tuple(env["features"]["selected"]) != FEATURES
        or env["features"]["volatility_window"] != 32
        or env["features"]["normalize"] is not False
    ):
        raise ValueError("Pinned current longf feature geometry required")
    require_keys(
        manifest["execution"],
        {
            "execution_id",
            "measurement_id",
            "broker",
            "account",
            "product_specification",
            "instruments",
            "calendar",
            "costs",
        },
        "execution",
    )
    execution = manifest["execution"]
    schema = read_json(EXECUTION_SCHEMA)
    contract_schema = {
        "type": "object",
        "properties": {k: schema["properties"][k] for k in execution},
        "required": list(execution),
        "additionalProperties": False,
        "$defs": schema["$defs"],
    }
    try:
        Draft202012Validator(contract_schema, format_checker=FormatChecker()).validate(
            execution
        )
    except ValidationError as exc:
        raise ValueError(f"execution schema: {exc.message}") from exc
    if tuple(item["model_pair"] for item in execution["instruments"]) != SYMBOLS:
        raise ValueError("execution instruments require the registered nine-pair order")
    for item in execution["instruments"]:
        if item["model_pair"] != f"JPY/{item['base']}" or item["base"] == "JPY":
            raise ValueError("execution base/quote mapping is unsupported")
    if execution["costs"]["swap_includes_markup"] and any(
        execution["costs"]["markup_per_base_day"]
    ):
        raise ValueError("execution costs double-count swap markup")
    if (execution["broker"] is None) != (execution["account"] is None):
        raise ValueError(
            "execution broker and account must both be explicit or unresolved"
        )
    if execution["broker"] is not None and execution["product_specification"] is None:
        raise ValueError(
            "execution named broker requires official dated product specifications"
        )
    ZoneInfo(execution["calendar"]["timezone"])
    previous_end = None
    for session in execution["calendar"]["sessions"]:
        opened, closed = instant(session["open"]), instant(session["close"])
        if opened >= closed or (previous_end is not None and opened < previous_end):
            raise ValueError("execution calendar has unordered or overlapping sessions")
        previous_end = closed
    if manifest["policy_mode"] == "synthetic_fixture":
        if manifest["mode"] != "fixture" or manifest["policy_sources"] is not None:
            raise ValueError(
                "Synthetic policies are fixture-only; observed registration requires frozen_2025"
            )
    elif manifest["policy_mode"] == "frozen_2025":
        sources = read_json(checked_path(manifest["policy_sources"]))
        identities = read_json(IDENTITIES)
        if sources["fold"] != "2025" or sources["source"] != identities["artifacts"][0]:
            raise ValueError(
                "Policies must pin the sealed last chronological 2025 fold"
            )
        if (
            manifest["feature_config"]["sha256"]
            != identities["forward_selection"]["configuration"]["sha256"]
        ):
            raise ValueError("Frozen 2025 config hash mismatch")
    else:
        raise ValueError("Explicit frozen_2025 or synthetic_fixture policies required")


def load_policies(manifest: dict[str, Any]) -> tuple[dict[str, Policy], dict[str, Any]]:
    """Load explicit fixed policies and collect all externally verifiable identities."""
    if manifest["policy_mode"] == "synthetic_fixture":
        policies, identity = fixture_policies()
        return policies, {
            "mode": "synthetic_fixture",
            "identity": identity,
            "artifacts": [],
        }
    policies, identity = frozen_policies(checked_path(manifest["policy_sources"]))
    expected = read_json(IDENTITIES)["forward_selection"]
    if identity["fold_sources"]["ppo"] != expected["ppo"]:
        raise ValueError("PPO identity differs from Issue 20 frozen selection")
    ppo = identity["fold_sources"]["ppo"]
    directory = Path(ppo["ensemble_dir"])
    ensemble = read_json(directory / "ensemble.json")
    refs = [
        manifest["policy_sources"],
        manifest["feature_config"],
        identity["parent"],
        {
            "path": str(Path(identity["parent"]["path"]).parent / "models.json"),
            "sha256": identity["ridge_file_sha256"],
        },
        {
            "path": str(directory / "ensemble.json"),
            "sha256": ppo["ensemble_manifest_sha256"],
        },
    ]
    for member in ensemble["members"]:
        for name, field in [
            ("model_final.zip", "model_sha256"),
            ("config_snapshot.yaml", "config_snapshot_sha256"),
            ("meta.json", "meta_sha256"),
        ]:
            refs.append(
                {"path": str(Path(member["run_dir"]) / name), "sha256": member[field]}
            )
    for ref in refs:
        ref["path"] = str(checked_path(ref))
    return policies, {
        "mode": "frozen_2025",
        "identity": identity,
        "artifacts": refs,
        "limitations": "Old 2025 bundles; quote-account assets differ from their training coordinate; no retraining or selection",
    }


def event(
    manifest: dict[str, Any],
    event_id: str,
    kind: str,
    cycle_id: str | None,
    at: str,
    payload: dict[str, Any],
) -> dict[str, Any]:
    """Construct a versioned event body before store sequence/hash assignment."""
    return {
        "event_id": event_id,
        "event_type": kind,
        "cycle_id": cycle_id,
        "recorded_at": utc(at),
        "wall_recorded_at": now(),
        "clock_basis": "fixture_virtual"
        if manifest["mode"] == "fixture"
        else "wall_clock",
        "payload": payload,
    }


def fail_external(manifest: dict[str, Any], origin: str, exc: Exception) -> None:
    """Record storage/audit failure independently and surface secondary failure too."""
    directory = Path(manifest["failure_storage"])
    try:
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{uuid.uuid4().hex}.json"
        content = (
            encode(
                {
                    "trial_id": manifest["trial_id"],
                    "storage": manifest["storage"],
                    "origin": origin,
                    "detected_at": now(),
                    "type": type(exc).__name__,
                    "message": str(exc),
                    "response": "stop; explicit recovery required",
                }
            )
            + b"\n"
        )
        with path.open("xb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        sync_directory(directory)
    except OSError as secondary:
        raise OSError(
            f"{origin}: {exc}; independent failure recording also failed at {directory}: {secondary}"
        ) from exc


def initialize(path: Path) -> dict[str, Any]:
    """Register a new immutable trial before its first decision."""
    manifest = read_json(path)
    validate_manifest(manifest)
    if manifest["mode"] == "observed_file" and not instant(
        manifest["registered_at"]
    ) <= instant(now()) < instant(manifest["measurement_start"]):
        raise ValueError(
            "Retroactive registration forbidden; observed start must be in the future"
        )
    for field in ("storage", "failure_storage"):
        manifest[field] = str(Path(manifest[field]).resolve())
    for field in ("feature_config", "policy_sources"):
        if manifest[field] is not None:
            manifest[field] = {
                **manifest[field],
                "path": str(checked_path(manifest[field])),
            }
    for step in manifest["schedule"]:
        for field in ("raw_path", "settlement_path"):
            step[field] = str(Path(step[field]).resolve())
    _, identities = load_policies(manifest)
    root = Path(manifest["storage"])
    root.mkdir(parents=True, exist_ok=False)
    try:
        for name in ("events", "artifacts", "staging", "orphans", "checkpoints"):
            (root / name).mkdir()
        publish(root / "manifest.json", encode(manifest) + b"\n", root / "staging")
        with EventLog(root) as log:
            policy_ref = log.artifact(encode(identities))
            runtime_ref = log.artifact(encode(runtime_identity()))
            log.append(
                [
                    event(
                        manifest,
                        "registration",
                        "registration",
                        None,
                        manifest["registered_at"]
                        if manifest["mode"] == "fixture"
                        else now(),
                        {
                            "manifest": {
                                "path": "manifest.json",
                                "sha256": digest((root / "manifest.json").read_bytes()),
                            },
                            "runtime": runtime_ref,
                            "policies": policy_ref,
                            "holdout_eligible": False,
                        },
                    )
                ]
            )
        sync_directory(root.parent)
    except Exception as exc:
        fail_external(manifest, "initialize", exc)
        raise
    return {
        "status": "registered",
        "storage": str(root),
        "mode": manifest["mode"],
        "holdout_eligible": False,
    }


def build_window(
    manifest: dict[str, Any], step: dict[str, Any], raw: dict[str, Any], cutoff: str
) -> dict[str, Any]:
    """Build the existing 32x8 feature window from 63 history rows and today's row.

    Args:
        manifest: Registered source and feature geometry.
        step: Explicit decision schedule and exact history event times.
        raw: Timestamped nine-pair OHLCV/carry snapshot.
        cutoff: Actual inference information cutoff (virtual only for fixture).

    Returns:
        Auditable float64 window, feature/lag order and input timing metadata.
    """
    require_keys(raw, {"source", "quality", "pairs", "fields", "rows"}, "raw snapshot")
    if (
        raw["source"] != manifest["source"]["id"]
        or raw["pairs"] != list(SYMBOLS)
        or raw["fields"] != FIELDS
    ):
        raise ValueError("Raw source/pair/field order mismatch")
    quality = "synthetic" if manifest["mode"] == "fixture" else "observed"
    if raw["quality"] != quality:
        raise ValueError("Raw quality does not match the registered observation mode")
    rows = raw["rows"]
    if [row["event_time"] for row in rows] != step["history_event_times"]:
        raise ValueError(
            "Missing, future or unordered raw history; exact registered 64 rows required"
        )
    available, retrieved = [], []
    for index, row in enumerate(rows):
        require_keys(
            row,
            {
                "event_time",
                "available_at",
                "retrieved_at",
                "carry_available_at",
                "carry_retrieved_at",
                "carry_vintage",
                "values",
            },
            f"raw/rows/{index}",
        )
        if (
            not instant(row["event_time"])
            <= instant(row["available_at"])
            <= instant(row["retrieved_at"])
            <= instant(cutoff)
        ):
            raise ValueError(f"Future/unavailable raw input: row {index}")
        if (
            not instant(row["carry_available_at"])
            <= instant(row["carry_retrieved_at"])
            <= instant(cutoff)
            or not row["carry_vintage"]
        ):
            raise ValueError(f"Unavailable carry or missing vintage: row {index}")
        available.extend([row["available_at"], row["carry_available_at"]])
        retrieved.extend([row["retrieved_at"], row["carry_retrieved_at"]])
    values = np.asarray([row["values"] for row in rows], dtype=np.float64)
    if values.shape != (64, 9, 6) or not np.isfinite(values).all():
        raise ValueError("Raw snapshot requires finite 64 x 9 x 6 values")
    frame = pd.DataFrame(
        values.reshape(64, 54),
        index=pd.DatetimeIndex([instant(row["event_time"]) for row in rows]),
        columns=pd.MultiIndex.from_product([SYMBOLS, FIELDS]),
    )
    validate_ohlcv(frame, SYMBOLS)
    config = parse_experiment_config(
        yaml.safe_load(checked_path(manifest["feature_config"]).read_text())
    )
    pipeline = FeaturePipeline(
        FeaturesConfig(
            volatility_window=config.env["features"]["volatility_window"],
            normalize=config.env["features"]["normalize"],
            selected=tuple(config.env["features"]["selected"]),
        ),
        {name: FEATURE_REGISTRY[name] for name in FEATURES if name in FEATURE_REGISTRY},
        custom_cross_features={
            name: CROSS_FEATURE_REGISTRY[name]
            for name in FEATURES
            if name in CROSS_FEATURE_REGISTRY
        },
    )
    window = pipeline.compute(frame, SYMBOLS)[:, 32:, :]
    if window.shape != (9, 32, 8) or not np.isfinite(window).all():
        raise ValueError("Non-finite or malformed feature window")
    return {
        "window": window.tolist(),
        "feature_names": list(FEATURE_NAMES),
        "pairs": list(SYMBOLS),
        "history_count": 63,
        "inputs_available_at": utc(max(available, key=instant)),
        "inputs_retrieved_at": utc(max(retrieved, key=instant)),
        "raw_timing": [{k: v for k, v in row.items() if k != "values"} for row in rows],
        "quality": raw["quality"],
        "initial_mid": (1 / values[-1, :, 3]).tolist(),
    }


def incident(
    log: EventLog,
    manifest: dict[str, Any],
    step: dict[str, Any],
    code: str,
    details: str,
    response: str,
) -> None:
    """Append an operational incident without inventing a decision or outcome."""
    at = step["decision_not_before"] if manifest["mode"] == "fixture" else now()
    log.append(
        [
            event(
                manifest,
                f"incident-{uuid.uuid4().hex}",
                "incident",
                step["cycle_id"],
                at,
                {
                    "decision_event_id": None,
                    "scheduled_decision_at": utc(step["decision_not_before"]),
                    "code": code,
                    "origin": f"cycle/{step['cycle_id']}",
                    "details": details,
                    "response": response,
                },
            )
        ]
    )


def saved_json(root: Path, reference: dict[str, str]) -> dict[str, Any]:
    """Read an artifact only after verifying its stored bytes."""
    return read_json(artifact_path(root, reference))


def cycle_events(
    rows: list[dict[str, Any]], cycle_id: str, kind: str
) -> list[dict[str, Any]]:
    """Select a cycle's append-only stage records."""
    return [
        row for row in rows if row["cycle_id"] == cycle_id and row["event_type"] == kind
    ]


def clipped(action: np.ndarray) -> np.ndarray:
    """Apply the unchanged registered pair clip and gross cap to a proposal."""
    weights = np.clip(action[:, 0].astype(np.float32), -1, 1).astype(float)
    gross = float(np.abs(weights).sum())
    if gross > 5:
        weights *= 5 / gross
    return weights


def settlement_result(
    manifest: dict[str, Any],
    step: dict[str, Any],
    decision: dict[str, Any],
    barrier: dict[str, Any],
    inputs: dict[str, Any],
    settlement: dict[str, Any],
) -> dict[str, Any]:
    """Use the single shared quote engine on a copy of the saved account state."""
    require_keys(settlement, {"quotes", "mark", "funding"}, "settlement")
    if instant(settlement["mark"]["timestamp"]) != instant(step["target_at"]):
        raise ValueError("Mark must equal the registered target")
    quality = "synthetic" if manifest["mode"] == "fixture" else "observed"
    for item in [*settlement["quotes"], settlement["mark"], *settlement["funding"]]:
        if item["source"] != manifest["source"]["id"]:
            raise ValueError("Unregistered settlement source")
    if any(
        item["quality"] != quality
        for item in [*settlement["quotes"], settlement["mark"]]
    ):
        raise ValueError("Settlement quality differs from observation mode")
    data = decision["payload"]
    replay_step = {
        "bar_label": instant(step["session_close"]).date().isoformat(),
        "session_close": step["session_close"],
        "decision_generated_at": data["decision_at"],
        "decision_recorded_at": barrier["payload"]["durable_at"],
        "window": inputs["window"],
        "inputs": [
            {
                "name": "market_and_carry_window",
                "available_at": inputs["inputs_available_at"],
                "retrieved_at": inputs["inputs_retrieved_at"],
                "payload_sha256": json_hash(inputs["window"]),
                "source": manifest["source"]["id"],
                "quality": quality,
            }
        ],
        "quotes": settlement["quotes"],
        "mark": settlement["mark"],
    }
    state = copy.deepcopy(data["state_before"])
    initial_mark = {
        "timestamp": state["timestamp"],
        "available_at": state["timestamp"],
        "retrieved_at": state["timestamp"],
        "mid": state["mid"],
        "source": manifest["source"]["id"],
        "quality": quality,
    }
    raw = {
        **manifest["execution"],
        "evidence_kind": "fixture" if manifest["mode"] == "fixture" else "shadow_quote",
        "funding": settlement["funding"],
        "initial_mark": initial_mark,
        "steps": [replay_step],
    }
    validate_execution(raw)
    fills: list[dict[str, Any]] = []

    def retain_fill(value: dict[str, Any]) -> None:
        """Retain staged quote evidence; account commit occurs with the outcome batch."""
        fills.append(copy.deepcopy(value))

    result = execute_quote(
        raw, replay_step, state, np.asarray(data["effective_action"]), retain_fill
    )
    if result["trace"] is None:
        raise ValueError(f"Account terminal before complete outcome: {result}")
    return {
        "trace": result["trace"],
        "state_after": state,
        "fills": fills,
        "status": result["status"],
    }


def audit(root: Path, rows: list[dict[str, Any]], head: str | None) -> dict[str, Any]:
    """Check references, artifact bytes, causal times and full shared-engine accounting."""
    manifest = read_json(root / "manifest.json")
    validate_manifest(manifest)
    if (
        not rows
        or rows[0]["event_type"] != "registration"
        or rows[0]["event_id"] != "registration"
    ):
        raise ValueError("Missing initial registration event")
    registration = rows[0]["payload"]
    for ref in (
        registration["manifest"],
        registration["runtime"],
        registration["policies"],
    ):
        artifact_path(root, ref)
    if saved_json(root, registration["manifest"]) != manifest:
        raise ValueError("Registered manifest mismatch")
    identities = saved_json(root, registration["policies"])
    for ref in identities["artifacts"]:
        artifact_path(root, ref)
    states: dict[str, Any] = {}
    seen: dict[str, dict[str, Any]] = {}
    completed: list[str] = []
    schedules = {s["cycle_id"]: s for s in manifest["schedule"]}
    outcomes_seen: set[str] = set()
    for row in rows:
        kind, payload = row["event_type"], row["payload"]
        cycle_id = row["cycle_id"]
        if row["clock_basis"] != (
            "fixture_virtual" if manifest["mode"] == "fixture" else "wall_clock"
        ):
            raise ValueError("Event clock basis differs from manifest")
        if cycle_id is not None and cycle_id not in schedules:
            raise ValueError(f"Unknown cycle reference {cycle_id}")
        for value in payload.values():
            if isinstance(value, dict) and set(value) == {"path", "sha256"}:
                artifact_path(root, value)
        if kind == "registration" and seen:
            raise ValueError("duplicate registration")
        if kind == "decision":
            step = schedules[cycle_id]
            policy_id = payload["policy_id"]
            if any(
                e["event_type"] == "decision"
                and e["cycle_id"] == cycle_id
                and e["payload"]["policy_id"] == policy_id
                for e in seen.values()
            ):
                raise ValueError("duplicate policy decision")
            position = list(schedules).index(cycle_id)
            if completed != list(schedules)[:position]:
                raise ValueError("Decision precedes completion of the preceding cycle")
            if (
                not instant(step["decision_not_before"])
                <= instant(payload["decision_at"])
                <= instant(row["recorded_at"])
                <= instant(step["decision_deadline"])
            ):
                raise ValueError(
                    "Decision generation/recording violates registered deadline"
                )
            if payload["target_at"] != utc(step["target_at"]) or not instant(
                payload["decision_at"]
            ) < instant(payload["target_at"]):
                raise ValueError("Decision target/time mismatch")
            if (
                payload["policy"] != registration["policies"]
                or payload["config"] != registration["manifest"]
            ):
                raise ValueError("Policy/config artifact reference mismatch")
            raw = saved_json(root, payload["data"])
            inputs = saved_json(root, payload["inputs"])
            expected = build_window(manifest, step, raw, payload["decision_at"])
            if (
                inputs != expected
                or payload["inputs_available_at"] != inputs["inputs_available_at"]
            ):
                raise ValueError("Input feature/timing artifact mismatch")
            if not any(
                e["event_type"] == "raw"
                and e["cycle_id"] == cycle_id
                and e["payload"]["kind"] == "input"
                and e["payload"]["artifact"] == payload["data"]
                for e in seen.values()
            ):
                raise ValueError("Decision missing prior raw snapshot reference")
            before = payload["state_before"]
            if policy_id in states:
                if before != states[policy_id]:
                    raise ValueError(
                        "Account/equity continuity mismatch before decision"
                    )
            else:
                expected_state = initial_state()
                expected_state.update(
                    timestamp=utc(step["session_close"]), mid=inputs["initial_mid"]
                )
                if before != expected_state:
                    raise ValueError(
                        "Initial account must be flat with zero history trades"
                    )
            if instant(before["timestamp"]) != instant(step["session_close"]):
                raise ValueError("Account timestamp differs from decision session")
            action = np.asarray(payload["proposed_action"]).reshape(9, 1)
            if not np.array_equal(clipped(action), payload["effective_action"]):
                raise ValueError(
                    "Recorded effective action differs from clip/gross contract"
                )
            costs = manifest["expected_costs"]
            if (
                payload["expected_transaction_cost_jpy"]
                != costs["transaction_cost_jpy"]
                or payload["expected_signed_carry_jpy"] != costs["signed_carry_jpy"]
                or payload["cost_quality"] != costs["quality"]
                or payload["cost_model_sha256"]
                != json_hash(manifest["execution"]["costs"])
            ):
                raise ValueError("Expected costs do not match registered assumptions")
        elif kind == "decision_barrier":
            step = schedules[cycle_id]
            decisions = cycle_events(list(seen.values()), cycle_id, "decision")
            if len(decisions) != 3 or {
                d["payload"]["policy_id"] for d in decisions
            } != set(POLICIES.values()):
                raise ValueError(
                    "All three decisions must be durable before barrier/apply"
                )
            if payload["decision_event_ids"] != [d["event_id"] for d in decisions]:
                raise ValueError("Decision barrier reference mismatch")
            at = instant(payload["durable_at"])
            if (
                not max(instant(d["recorded_at"]) for d in decisions)
                <= at
                <= instant(step["decision_deadline"])
            ):
                raise ValueError("Durable decision deadline exceeded")
            if instant(row["recorded_at"]) < at or cycle_events(
                list(seen.values()), cycle_id, "decision_barrier"
            ):
                raise ValueError("Invalid or duplicate decision barrier")
        elif kind == "outcome":
            decision_id = payload["decision_event_id"]
            if (
                decision_id not in seen
                or seen[decision_id]["event_type"] != "decision"
                or decision_id in outcomes_seen
            ):
                raise ValueError("Unknown or duplicate outcome decision reference")
            decision = seen[decision_id]
            if (
                decision["cycle_id"] != cycle_id
                or decision["payload"]["policy_id"] != payload["policy_id"]
            ):
                raise ValueError("Outcome policy/cycle reference mismatch")
            barriers = cycle_events(list(seen.values()), cycle_id, "decision_barrier")
            if len(barriers) != 1:
                raise ValueError("Outcome before durable three-decision barrier")
            if instant(payload["observed_at"]) < instant(
                decision["payload"]["target_at"]
            ) or instant(row["recorded_at"]) < instant(payload["observed_at"]):
                raise ValueError("Outcome recorded before its mark was observed")
            evidence = saved_json(root, payload["evidence"])
            settlement = saved_json(root, evidence["settlement"])
            if not any(
                e["event_type"] == "raw"
                and e["cycle_id"] == cycle_id
                and e["payload"]["kind"] == "settlement"
                and e["payload"]["artifact"] == evidence["settlement"]
                for e in seen.values()
            ):
                raise ValueError("Outcome requires a prior settlement snapshot")
            expected = settlement_result(
                manifest,
                schedules[cycle_id],
                decision,
                barriers[0],
                saved_json(root, decision["payload"]["inputs"]),
                settlement,
            )
            if evidence["result"] != expected:
                raise ValueError(
                    "Outcome account/quote evidence differs from shared execution engine"
                )
            trace = expected["trace"]
            transaction = trace["spread_jpy"] + trace["commission_jpy"]
            carry = (
                trace["gap_financing_jpy"]
                + trace["holding_financing_jpy"]
                - trace["gap_markup_jpy"]
                - trace["holding_markup_jpy"]
            )
            price = trace["gap_pnl_jpy"] + trace["holding_pnl_jpy"]
            if (
                payload["state_after"] != expected["state_after"]
                or payload["equity_before_jpy"] != trace["equity_before"]
                or payload["equity_after_jpy"] != trace["equity_jpy"]
                or payload["observed_transaction_cost_jpy"] != transaction
                or payload["observed_signed_carry_jpy"] != carry
                or payload["price_pnl_jpy"] != price
            ):
                raise ValueError("Outcome equity/accounting mismatch")
            if not np.isclose(
                payload["equity_before_jpy"] + price + carry - transaction,
                payload["equity_after_jpy"],
                rtol=1e-12,
                atol=1e-7,
            ):
                raise ValueError("Equity accounting identity mismatch")
            states[payload["policy_id"]] = payload["state_after"]
            outcomes_seen.add(decision_id)
            if len(cycle_events([*seen.values(), row], cycle_id, "outcome")) == 3:
                completed.append(cycle_id)
        elif kind == "correction":
            if payload["corrects_event_id"] not in seen:
                raise ValueError("Unknown correction reference")
        elif kind == "incident" and payload["decision_event_id"] is not None:
            if (
                payload["decision_event_id"] not in seen
                or seen[payload["decision_event_id"]]["event_type"] != "decision"
            ):
                raise ValueError("Unknown incident decision reference")
        seen[row["event_id"]] = row
    for cycle_id in schedules:
        for kind in ("decision", "outcome"):
            if len(cycle_events(rows, cycle_id, kind)) not in (0, 3):
                raise ValueError(
                    f"Incomplete atomic three-policy {kind} batch: {cycle_id}"
                )
    for checkpoint in (root / "checkpoints").glob("*.json"):
        value = read_json(checkpoint)
        count = value["event_count"]
        if (
            count < 1
            or count > len(rows)
            or digest(encode(rows[count - 1])) != value["head_sha256"]
        ):
            raise ValueError(f"Checkpoint head hash mismatch: {checkpoint}")
        expected_states: dict[str, Any] = {}
        for row in rows[:count]:
            if row["event_type"] == "outcome":
                expected_states[row["payload"]["policy_id"]] = row["payload"][
                    "state_after"
                ]
        if value["accounts"] != expected_states:
            raise ValueError(f"Checkpoint account mismatch: {checkpoint}")
    return {
        "events": rows,
        "head_sha256": head,
        "event_count": len(rows),
        "accounts": states,
        "completed_cycles": completed,
        "holdout_eligible": False,
    }


def verify_log(root: Path) -> dict[str, Any]:
    """Verify the full committed log independently of its derived checkpoint."""
    rows, head = read_events(root)
    return audit(root, rows, head)


def checkpoint(log: EventLog, result: dict[str, Any]) -> None:
    """Save a new immutable head-bound account checkpoint after outcome commit."""
    path = log.root / "checkpoints" / f"{len(log.events):012d}.json"
    value = {
        "event_count": len(log.events),
        "head_sha256": log.head,
        "accounts": result["accounts"],
    }
    if path.exists():
        if read_json(path) != value:
            raise ValueError(f"Checkpoint mismatch: {path}")
    else:
        publish(path, encode(value) + b"\n", log.root / "staging")


def pending_failures(
    manifest: dict[str, Any], rows: list[dict[str, Any]]
) -> list[Path]:
    """Find this trial's independent failures not yet acknowledged in its log."""
    ids = {row["event_id"] for row in rows}
    result: list[Path] = []
    for path in sorted(Path(manifest["failure_storage"]).glob("*.json")):
        record = read_json(path)
        if (
            record["trial_id"] == manifest["trial_id"]
            and record["storage"] == manifest["storage"]
            and f"failure/{path.name}" not in ids
        ):
            result.append(path)
    return result


def run_cycle(root: Path, cycle_id: str) -> dict[str, Any]:
    """Record or resume exactly one scheduled cycle without redoing saved inference."""
    manifest = read_json(root / "manifest.json")
    try:
        with EventLog(root) as log:
            if list((root / "staging").iterdir()):
                raise ValueError(
                    "Interrupted staging writes require explicit recover before continuation"
                )
            if pending_failures(manifest, log.events):
                raise ValueError("Independent failure records require explicit recover")
            current = audit(root, log.events, log.head)
            matching = [s for s in manifest["schedule"] if s["cycle_id"] == cycle_id]
            if len(matching) != 1:
                raise ValueError(f"Unknown cycle {cycle_id}")
            step = matching[0]
            if cycle_id in current["completed_cycles"]:
                return {"status": "already_complete", "cycle_id": cycle_id}
            if any(
                e["event_type"] == "incident" and e["payload"]["response"] == "halt"
                for e in log.events
            ):
                raise ValueError(
                    "Trial halted/incomplete; no backfilled decisions allowed"
                )
            position = manifest["schedule"].index(step)
            if current["completed_cycles"] != [
                s["cycle_id"] for s in manifest["schedule"][:position]
            ]:
                raise ValueError(
                    "Previous cycle must complete before the next decision"
                )
            try:
                registration = log.events[0]["payload"]
                saved_runtime = saved_json(root, registration["runtime"])
                actual_runtime = runtime_identity()
                if any(
                    saved_runtime[k] != actual_runtime[k]
                    for k in ("versions", "source_sha256", "device")
                ):
                    raise ValueError(
                        "Runtime/code hash changed since registration; new trial required"
                    )
                decisions = cycle_events(log.events, cycle_id, "decision")
                cutoff = (
                    step["decision_not_before"]
                    if manifest["mode"] == "fixture"
                    else now()
                )
                if not decisions:
                    if (
                        not instant(step["decision_not_before"])
                        <= instant(cutoff)
                        <= instant(step["decision_deadline"])
                    ):
                        raise ValueError(
                            "Decision deadline exceeded or execution before scheduled time; no backfill"
                        )
                    previous_raw = [
                        e
                        for e in cycle_events(log.events, cycle_id, "raw")
                        if e["payload"]["kind"] == "input"
                    ]
                    if previous_raw:
                        raise ValueError(
                            "Raw saved without decisions after interruption; no retrospective inference"
                        )
                    raw_path = Path(step["raw_path"])
                    if not raw_path.is_file():
                        raise ValueError(f"missing raw input: {raw_path}")
                    raw_ref = log.artifact(raw_path.read_bytes())
                    log.append(
                        [
                            event(
                                manifest,
                                f"{cycle_id}/raw",
                                "raw",
                                cycle_id,
                                cutoff,
                                {
                                    "kind": "input",
                                    "artifact": raw_ref,
                                    "source_path": str(raw_path),
                                },
                            )
                        ]
                    )
                    inputs = build_window(
                        manifest, step, saved_json(root, raw_ref), cutoff
                    )
                    inputs_ref = log.artifact(encode(inputs))
                    policies, identity = load_policies(manifest)
                    if identity != saved_json(root, registration["policies"]):
                        raise ValueError(
                            "Pinned policy identity changed since registration"
                        )
                    records = []
                    for name, policy in policies.items():
                        policy_id = POLICIES[name]
                        if policy_id in current["accounts"]:
                            state = copy.deepcopy(current["accounts"][policy_id])
                        else:
                            state = initial_state()
                            state.update(
                                timestamp=utc(step["session_close"]),
                                mid=inputs["initial_mid"],
                            )
                        window = np.array(inputs["window"], dtype=np.float64)
                        scores, action = policy(
                            {
                                "market": window.astype(np.float32),
                                "assets": np.array(state["assets"], dtype=np.float32),
                            },
                            window.copy(),
                        )
                        if (
                            scores.shape != (9,)
                            or action.shape != (9, 1)
                            or not np.isfinite(scores).all()
                            or not np.isfinite(action).all()
                        ):
                            raise ValueError(f"Malformed policy output: {policy_id}")
                        generated = cutoff if manifest["mode"] == "fixture" else now()
                        if instant(generated) > instant(step["decision_deadline"]):
                            raise ValueError("Inference exceeded decision deadline")
                        payload = {
                            "policy_id": policy_id,
                            "decision_at": utc(generated),
                            "target_at": utc(step["target_at"]),
                            "inputs_available_at": inputs["inputs_available_at"],
                            "policy": registration["policies"],
                            "config": registration["manifest"],
                            "data": raw_ref,
                            "inputs": inputs_ref,
                            "pairs": list(SYMBOLS),
                            "scores": scores.tolist(),
                            "proposed_action": action[:, 0].tolist(),
                            "effective_action": clipped(action).tolist(),
                            "expected_transaction_cost_jpy": manifest["expected_costs"][
                                "transaction_cost_jpy"
                            ],
                            "expected_signed_carry_jpy": manifest["expected_costs"][
                                "signed_carry_jpy"
                            ],
                            "cost_quality": manifest["expected_costs"]["quality"],
                            "cost_model_sha256": json_hash(
                                manifest["execution"]["costs"]
                            ),
                            "state_before": state,
                        }
                        records.append(
                            event(
                                manifest,
                                f"{cycle_id}/decision/{policy_id}",
                                "decision",
                                cycle_id,
                                generated,
                                payload,
                            )
                        )
                    decisions = log.append(records)
                barriers = cycle_events(log.events, cycle_id, "decision_barrier")
                if not barriers:
                    durable = (
                        utc(instant(step["decision_not_before"]).isoformat())
                        if manifest["mode"] == "fixture"
                        else now()
                    )
                    if instant(durable) > instant(step["decision_deadline"]):
                        raise ValueError(
                            "Durable decision recording exceeded deadline; stop without applying"
                        )
                    barriers = log.append(
                        [
                            event(
                                manifest,
                                f"{cycle_id}/barrier",
                                "decision_barrier",
                                cycle_id,
                                durable,
                                {
                                    "decision_event_ids": [
                                        d["event_id"] for d in decisions
                                    ],
                                    "durable_at": durable,
                                },
                            )
                        ]
                    )
                settlement_path = Path(step["settlement_path"])
                if not settlement_path.exists():
                    existing = cycle_events(log.events, cycle_id, "incident")
                    if not any(
                        e["payload"]["code"] == "missing_settlement" for e in existing
                    ):
                        incident(
                            log,
                            manifest,
                            step,
                            "missing_settlement",
                            f"Quote/mark/financing not received: {settlement_path}",
                            "awaiting_settlement",
                        )
                    return {"status": "awaiting_settlement", "cycle_id": cycle_id}
                settlements = [
                    e
                    for e in cycle_events(log.events, cycle_id, "raw")
                    if e["payload"]["kind"] == "settlement"
                ]
                observed = step["target_at"] if manifest["mode"] == "fixture" else now()
                if settlements:
                    settlement_ref = settlements[-1]["payload"]["artifact"]
                else:
                    settlement_ref = log.artifact(settlement_path.read_bytes())
                    log.append(
                        [
                            event(
                                manifest,
                                f"{cycle_id}/settlement",
                                "raw",
                                cycle_id,
                                observed,
                                {
                                    "kind": "settlement",
                                    "artifact": settlement_ref,
                                    "source_path": str(settlement_path),
                                },
                            )
                        ]
                    )
                settlement = saved_json(root, settlement_ref)
                if manifest["mode"] == "observed_file":
                    for item in [
                        *settlement["quotes"],
                        settlement["mark"],
                        *settlement["funding"],
                    ]:
                        if instant(item["retrieved_at"]) > instant(observed):
                            raise ValueError(
                                "Future settlement input unavailable at current wall clock"
                            )
                outcomes = []
                terminal = False
                for decision in decisions:
                    inputs = saved_json(root, decision["payload"]["inputs"])
                    result = settlement_result(
                        manifest, step, decision, barriers[0], inputs, settlement
                    )
                    terminal |= result["status"] == "terminal_margin"
                    trace = result["trace"]
                    evidence_ref = log.artifact(
                        encode({"settlement": settlement_ref, "result": result})
                    )
                    payload = {
                        "decision_event_id": decision["event_id"],
                        "policy_id": decision["payload"]["policy_id"],
                        "observed_at": utc(observed),
                        "execution_basis": "shadow_quote",
                        "observed_transaction_cost_jpy": trace["spread_jpy"]
                        + trace["commission_jpy"],
                        "observed_signed_carry_jpy": trace["gap_financing_jpy"]
                        + trace["holding_financing_jpy"]
                        - trace["gap_markup_jpy"]
                        - trace["holding_markup_jpy"],
                        "price_pnl_jpy": trace["gap_pnl_jpy"]
                        + trace["holding_pnl_jpy"],
                        "equity_before_jpy": trace["equity_before"],
                        "equity_after_jpy": trace["equity_jpy"],
                        "evidence": evidence_ref,
                        "state_after": result["state_after"],
                    }
                    outcomes.append(
                        event(
                            manifest,
                            decision["event_id"] + "/outcome",
                            "outcome",
                            cycle_id,
                            observed,
                            payload,
                        )
                    )
                log.append(outcomes)
                result = audit(root, log.events, log.head)
                checkpoint(log, result)
                if terminal:
                    incident(
                        log,
                        manifest,
                        step,
                        "terminal_margin",
                        "At least one account reached registered margin stop",
                        "halt",
                    )
                return {
                    "status": "terminal_margin" if terminal else "complete",
                    "cycle_id": cycle_id,
                    "head_sha256": log.head,
                }
            except (
                ValueError,
                KeyError,
                TypeError,
                ArithmeticError,
                DataError,
                FeatureError,
            ) as exc:
                incident(log, manifest, step, type(exc).__name__, str(exc), "halt")
                raise ValueError(f"cycle/{cycle_id}: {exc}") from exc
    except Exception as exc:
        fail_external(manifest, f"cycle/{cycle_id}", exc)
        raise


def recover(root: Path) -> dict[str, Any]:
    """Preserve orphan bytes and explicitly record recovery without backfilled inference."""
    manifest = read_json(root / "manifest.json")
    try:
        with EventLog(root) as log:
            audit(root, log.events, log.head)
            paths = {p.name: p for p in (root / "orphans").iterdir()}
            paths.update({p.name: p for p in (root / "staging").iterdir()})
            for name, path in sorted(paths.items()):
                event_id = f"recovery/{name}"
                if any(e["event_id"] == event_id for e in log.events):
                    if path.parent == root / "staging":
                        path.unlink()
                        sync_directory(root / "staging")
                    continue
                orphan = root / "orphans" / name
                if path.parent == root / "staging":
                    if orphan.exists():
                        if orphan.read_bytes() != path.read_bytes():
                            raise ValueError(f"Orphan name collision: {orphan}")
                    else:
                        os.link(path, orphan)
                        sync_directory(orphan.parent)
                interrupted_raw = any(
                    any(
                        e["payload"]["kind"] == "input"
                        for e in cycle_events(log.events, s["cycle_id"], "raw")
                    )
                    and not cycle_events(log.events, s["cycle_id"], "decision")
                    for s in manifest["schedule"]
                )
                resumable = not interrupted_raw and any(
                    len(cycle_events(log.events, s["cycle_id"], "decision")) == 3
                    for s in manifest["schedule"]
                )
                # Re-sync committed batches in EventLog before permitting any continuation.
                log.append(
                    [
                        event(
                            manifest,
                            event_id,
                            "incident",
                            None,
                            log.events[-1]["recorded_at"]
                            if manifest["mode"] == "fixture"
                            else now(),
                            {
                                "decision_event_id": None,
                                "scheduled_decision_at": utc(
                                    manifest["schedule"][0]["decision_not_before"]
                                ),
                                "code": "recovered_staging",
                                "origin": str(path),
                                "details": json.dumps(
                                    {
                                        "preserved_path": str(orphan),
                                        "sha256": digest(orphan.read_bytes()),
                                    }
                                ),
                                "response": "resume_saved_decisions"
                                if resumable
                                else "halt",
                            },
                        )
                    ]
                )
                if path.parent == root / "staging":
                    path.unlink()
                    sync_directory(root / "staging")
            for failure in pending_failures(manifest, log.events):
                proof = log.artifact(failure.read_bytes())
                log.append(
                    [
                        event(
                            manifest,
                            f"failure/{failure.name}",
                            "incident",
                            None,
                            log.events[-1]["recorded_at"]
                            if manifest["mode"] == "fixture"
                            else now(),
                            {
                                "decision_event_id": None,
                                "scheduled_decision_at": utc(
                                    manifest["schedule"][0]["decision_not_before"]
                                ),
                                "code": "recovered_failure",
                                "origin": str(failure),
                                "details": json.dumps(
                                    {
                                        "failure_evidence": proof,
                                        "record": read_json(failure),
                                    }
                                ),
                                "response": "acknowledged_storage_failure",
                            },
                        )
                    ]
                )
            result = audit(root, log.events, log.head)
            checkpoint(log, result)
            return {"status": "recovered", "head_sha256": log.head}
    except Exception as exc:
        fail_external(manifest, "recover", exc)
        raise


def append_correction(
    root: Path, event_id: str, target: str, reason: str, evidence: Path
) -> None:
    """Append correction evidence without changing the original event or account."""
    manifest = read_json(root / "manifest.json")
    try:
        with EventLog(root) as log:
            audit(root, log.events, log.head)
            if target not in {e["event_id"] for e in log.events}:
                raise ValueError(f"Unknown correction reference: {target}")
            if event_id in {e["event_id"] for e in log.events}:
                raise ValueError(f"duplicate event ID: {event_id}")
            ref = log.artifact(evidence.read_bytes())
            log.append(
                [
                    event(
                        manifest,
                        event_id,
                        "correction",
                        None,
                        now(),
                        {
                            "corrects_event_id": target,
                            "reason": reason,
                            "evidence": ref,
                        },
                    )
                ]
            )
    except OSError as exc:
        fail_external(manifest, "append_correction", exc)
        raise


def record_trial(root: Path, event_id: str, kind: str, reason: str) -> None:
    """Record performance access or a design change in this non-holdout trial ledger."""
    manifest = read_json(root / "manifest.json")
    try:
        with EventLog(root) as log:
            audit(root, log.events, log.head)
            log.append(
                [
                    event(
                        manifest,
                        event_id,
                        "trial_entry",
                        None,
                        now(),
                        {"kind": kind, "reason": reason, "holdout_eligible": False},
                    )
                ]
            )
    except OSError as exc:
        fail_external(manifest, "record_trial", exc)
        raise


def export_head(root: Path, destination: Path) -> None:
    """Export a verified head to a new file for explicit independent retention."""
    result = verify_log(root)
    value = {
        "head_sha256": result["head_sha256"],
        "event_count": result["event_count"],
        "manifest_sha256": digest((root / "manifest.json").read_bytes()),
        "exported_at": now(),
        "guarantee": "Local integrity only; independent retention and access separation are operator responsibilities",
    }
    from .quote_replay import write_json

    write_json(destination, value)


def main() -> None:
    """Run explicit registration, cycle/resume, validation and audit operations."""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    register = commands.add_parser("init")
    register.add_argument("--manifest", type=Path, required=True)
    for name in (
        "cycle",
        "resume",
        "verify",
        "recover",
        "head",
        "correct",
        "trial",
        "show",
    ):
        sub = commands.add_parser(name)
        sub.add_argument("--store", type=Path, required=True)
        if name in ("cycle", "resume"):
            sub.add_argument("--cycle", required=True)
        if name == "head":
            sub.add_argument("--output", type=Path, required=True)
        if name in ("correct", "trial", "show"):
            sub.add_argument("--event-id", required=True)
            sub.add_argument("--reason", required=True)
        if name == "correct":
            sub.add_argument("--target", required=True)
            sub.add_argument("--evidence", type=Path, required=True)
        if name == "trial":
            sub.add_argument(
                "--kind", choices=("performance_view", "design_change"), required=True
            )
    args = parser.parse_args()
    try:
        if args.command == "init":
            result = initialize(args.manifest)
        elif args.command in ("cycle", "resume"):
            result = run_cycle(args.store, args.cycle)
        elif args.command == "recover":
            result = recover(args.store)
        elif args.command == "verify":
            checked = verify_log(args.store)
            result = {
                k: checked[k]
                for k in (
                    "head_sha256",
                    "event_count",
                    "completed_cycles",
                    "holdout_eligible",
                )
            }
        elif args.command == "head":
            export_head(args.store, args.output)
            result = {"exported_head": str(args.output)}
        elif args.command == "correct":
            append_correction(
                args.store, args.event_id, args.target, args.reason, args.evidence
            )
            result = {"appended_correction": args.event_id}
        elif args.command == "trial":
            record_trial(args.store, args.event_id, args.kind, args.reason)
            result = {"appended_trial_entry": args.event_id}
        else:
            record_trial(args.store, args.event_id, "performance_view", args.reason)
            result = {
                "accounts": verify_log(args.store)["accounts"],
                "holdout_eligible": False,
            }
    except (ValueError, OSError, KeyError, TypeError) as exc:
        parser.exit(2, f"{type(exc).__name__}: {exc}\n")
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
