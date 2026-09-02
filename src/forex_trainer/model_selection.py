"""Explicit model-artifact loading and action-mean evaluation."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from stable_baselines3.common.callbacks import BaseCallback

from .algorithms import ALGO_REGISTRY, resolve_device
from .config import TrainerConfigError, parse_experiment_config
from .env_factory import GateEvaluationMode, build_single_env
from .evaluate import compute_metrics, walk_eval_range

LATE_CHECKPOINT_RULE = "final_five_validation_walks"
_VALIDATION_WALKS = 20
_LATE_VALIDATION_WALKS: tuple[int, ...] = (16, 17, 18, 19, 20)


def late_checkpoint_timesteps(total_timesteps: int, n_envs: int) -> tuple[int, ...]:
    """Return the five nominal late-checkpoint timesteps.

    The points are 80%, 85%, 90%, 95%, and 100% of the nominal training
    budget, rounded up to a vector-environment step boundary. This is exactly
    240k/255k/270k/285k/300k for the longf protocol.

    Args:
        total_timesteps: Nominal training budget.
        n_envs: Number of vectorized training environments.

    Returns:
        Five non-decreasing agent-timestep thresholds. Very small budgets can
        map multiple named checkpoints to one vector-environment step.
    """
    if total_timesteps < 1 or n_envs < 1:
        raise TrainerConfigError(
            "Late-checkpoint scheduling requires positive total_timesteps and n_envs."
        )
    thresholds = tuple(
        min(
            ((total_timesteps * walk + _VALIDATION_WALKS * n_envs - 1)
             // (_VALIDATION_WALKS * n_envs))
            * n_envs,
            ((total_timesteps + n_envs - 1) // n_envs) * n_envs,
        )
        for walk in _LATE_VALIDATION_WALKS
    )
    return thresholds


class LateCheckpointCallback(BaseCallback):
    """Save the declared late checkpoints after validation evaluations."""

    def __init__(self, run_dir: Path, total_timesteps: int, n_envs: int) -> None:
        """Initialize a late-checkpoint callback.

        Args:
            run_dir: Training run artifact directory.
            total_timesteps: Nominal training budget.
            n_envs: Number of vectorized training environments.
        """
        super().__init__(verbose=0)
        self._run_dir = run_dir
        self._checkpoint_dir = run_dir / "late_checkpoints"
        self._checkpoint_dir.mkdir(parents=True, exist_ok=False)
        self._nominal_timesteps = late_checkpoint_timesteps(total_timesteps, n_envs)
        self._next_index = 0
        self._saved: list[dict[str, int | str]] = []
        self._write_manifest()

    def _write_manifest(self) -> None:
        """Persist the current checkpoint inventory for audit."""
        manifest = {
            "rule": LATE_CHECKPOINT_RULE,
            "nominal_timesteps": list(self._nominal_timesteps),
            "checkpoints": list(self._saved),
        }
        (self._run_dir / "late_checkpoints.json").write_text(
            json.dumps(manifest, indent=2), encoding="utf-8"
        )

    def _save_reached(self, actual_timestep: int) -> None:
        """Save every threshold reached at an explicit training timestep.

        Args:
            actual_timestep: Current agent timestep from SB3.
        """
        while (
            self._next_index < len(self._nominal_timesteps)
            and actual_timestep >= self._nominal_timesteps[self._next_index]
        ):
            validation_walk = _LATE_VALIDATION_WALKS[self._next_index]
            nominal = self._nominal_timesteps[self._next_index]
            filename = f"model_walk_{validation_walk}_{nominal}_steps.zip"
            self.model.save(self._checkpoint_dir / filename)
            self._saved.append(
                {
                    "nominal_timestep": nominal,
                    "actual_timestep": actual_timestep,
                    "path": filename,
                }
            )
            self._next_index += 1
            self._write_manifest()

    def _on_step(self) -> bool:
        """Save every threshold reached by the current validation event."""
        self._save_reached(self.num_timesteps)
        return True

    def finalize(self) -> None:
        """Save thresholds not reached by a non-divisible evaluation cadence.

        The longf protocol reaches all five thresholds during validation. This
        explicit training-end step preserves the artifact contract for other
        valid budgets whose final nominal timestep is not an evaluation event.

        Raises:
            RuntimeError: If training ended before a declared threshold.
        """
        actual_timestep = self.model.num_timesteps
        self._save_reached(actual_timestep)
        if self._next_index != len(self._nominal_timesteps):
            raise RuntimeError(
                f"Training ended at {actual_timestep} before late-checkpoint "
                f"threshold {self._nominal_timesteps[self._next_index]}."
            )


def _read_run_meta(run_dir: Path) -> dict[str, Any]:
    """Read the required seed and git provenance from a run.

    Args:
        run_dir: Training run directory.

    Returns:
        Parsed metadata mapping.

    Raises:
        TrainerConfigError: If metadata is missing or malformed.
    """
    meta_path = run_dir / "meta.json"
    if not meta_path.is_file():
        raise TrainerConfigError(f"Run metadata is missing: {meta_path}")
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise TrainerConfigError(f"Run metadata is malformed: {meta_path}") from exc
    if not isinstance(meta, dict):
        raise TrainerConfigError(f"Run metadata must be a mapping: {meta_path}")
    if "seed" not in meta:
        raise TrainerConfigError(f"Run metadata lacks seed: {meta_path}")
    seed = meta["seed"]
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TrainerConfigError(f"Run metadata has an invalid seed: {meta_path}")
    if "git" not in meta or not isinstance(meta["git"], dict):
        raise TrainerConfigError(f"Run metadata lacks git provenance: {meta_path}")
    return meta


@dataclass(frozen=True)
class ModelArtifact:
    """One explicitly selected model and its source run."""

    source_run: Path
    model_path: Path
    seed: int
    nominal_timestep: int | None

    @classmethod
    def from_run_model(cls, run_dir: Path, filename: str) -> ModelArtifact:
        """Create an artifact for one named model in a run directory.

        Args:
            run_dir: Training run directory.
            filename: Exact model filename, such as model_final.zip.

        Returns:
            Validated model artifact.

        Raises:
            TrainerConfigError: If model or run metadata is missing.
        """
        model_path = run_dir / filename
        if not model_path.is_file():
            raise TrainerConfigError(f"Model artifact is missing: {model_path}")
        meta = _read_run_meta(run_dir)
        return cls(run_dir, model_path, meta["seed"], None)


def load_late_checkpoint_artifacts(run_dir: Path) -> list[ModelArtifact]:
    """Load the exact five checkpoints declared by a training run.

    Args:
        run_dir: Training run directory.

    Returns:
        Five checkpoint artifacts in nominal-timestep order.

    Raises:
        TrainerConfigError: If the manifest is malformed or incomplete.
    """
    manifest_path = run_dir / "late_checkpoints.json"
    if not manifest_path.is_file():
        raise TrainerConfigError(
            f"Late-checkpoint manifest is missing: {manifest_path}"
        )
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise TrainerConfigError(
            f"Late-checkpoint manifest is malformed: {manifest_path}: {exc}"
        ) from exc
    expected_keys = {"rule", "nominal_timesteps", "checkpoints"}
    if not isinstance(manifest, dict) or set(manifest) != expected_keys:
        raise TrainerConfigError(
            f"Late-checkpoint manifest has invalid keys: {manifest_path}"
        )
    if manifest["rule"] != LATE_CHECKPOINT_RULE:
        raise TrainerConfigError(
            f"Late-checkpoint manifest has unknown rule at {manifest_path}: "
            f"{manifest['rule']!r}"
        )
    checkpoints = manifest["checkpoints"]
    nominal_timesteps = manifest["nominal_timesteps"]
    if not isinstance(checkpoints, list) or len(checkpoints) != 5:
        raise TrainerConfigError(
            "Late-checkpoint manifest must declare exactly five models: "
            f"{manifest_path}"
        )
    if not isinstance(nominal_timesteps, list) or len(nominal_timesteps) != 5:
        raise TrainerConfigError(
            f"Late-checkpoint manifest must declare five timesteps: {manifest_path}"
        )
    snapshot_path = run_dir / "config_snapshot.yaml"
    if not snapshot_path.is_file():
        raise TrainerConfigError(f"Run artifact is missing: {snapshot_path}")
    snapshot = yaml.safe_load(snapshot_path.read_text(encoding="utf-8"))
    config = parse_experiment_config(snapshot)
    expected_timesteps = list(
        late_checkpoint_timesteps(
            config.run.total_timesteps,
            config.run.n_envs,
        )
    )
    if nominal_timesteps != expected_timesteps:
        raise TrainerConfigError(
            f"Late-checkpoint manifest timesteps at {manifest_path} are "
            f"{nominal_timesteps}, expected {expected_timesteps}."
        )
    meta = _read_run_meta(run_dir)
    artifacts: list[ModelArtifact] = []
    observed: list[int] = []
    actual_timesteps: list[int] = []
    eval_freq = max(
        1,
        config.run.total_timesteps // (_VALIDATION_WALKS * config.run.n_envs),
    )
    evaluation_interval = eval_freq * config.run.n_envs
    for index, item in enumerate(checkpoints):
        if not isinstance(item, dict) or set(item) != {
            "nominal_timestep",
            "actual_timestep",
            "path",
        }:
            raise TrainerConfigError(
                f"Late-checkpoint entry is malformed: {manifest_path}"
            )
        nominal = item["nominal_timestep"]
        actual = item["actual_timestep"]
        filename = item["path"]
        if isinstance(nominal, bool) or not isinstance(nominal, int):
            raise TrainerConfigError(
                f"Late-checkpoint timestep is invalid: {manifest_path}"
            )
        if (
            isinstance(actual, bool)
            or not isinstance(actual, int)
            or actual < nominal
        ):
            raise TrainerConfigError(
                f"Late-checkpoint actual timestep is invalid: {manifest_path}"
            )
        next_evaluation = (
            (nominal + evaluation_interval - 1) // evaluation_interval
        ) * evaluation_interval
        if actual > next_evaluation:
            raise TrainerConfigError(
                f"Late-checkpoint actual timestep {actual} exceeds the next "
                f"validation event {next_evaluation}: {manifest_path}"
            )
        validation_walk = _LATE_VALIDATION_WALKS[index]
        expected_filename = f"model_walk_{validation_walk}_{nominal}_steps.zip"
        if filename != expected_filename or Path(filename).name != filename:
            raise TrainerConfigError(
                f"Late-checkpoint path is invalid: {manifest_path}: {filename!r}"
            )
        model_path = run_dir / "late_checkpoints" / filename
        if not model_path.is_file():
            raise TrainerConfigError(f"Model artifact is missing: {model_path}")
        observed.append(nominal)
        actual_timesteps.append(actual)
        artifacts.append(
            ModelArtifact(run_dir, model_path, meta["seed"], nominal_timestep=nominal)
        )
    if observed != nominal_timesteps or observed != sorted(observed):
        raise TrainerConfigError(
            f"Late-checkpoint timesteps are inconsistent: {manifest_path}"
        )
    if actual_timesteps != sorted(actual_timesteps):
        raise TrainerConfigError(
            f"Late-checkpoint actual timesteps are not monotonic: {manifest_path}"
        )
    declared_files = {artifact.model_path.name for artifact in artifacts}
    actual_files = {
        path.name for path in (run_dir / "late_checkpoints").glob("*.zip")
    }
    if actual_files != declared_files:
        raise TrainerConfigError(
            f"Late-checkpoint directory inventory differs from {manifest_path}: "
            f"declared={sorted(declared_files)}, actual={sorted(actual_files)}."
        )
    return artifacts


def _sha256(path: Path) -> str:
    """Return the SHA-256 digest of a file.

    Args:
        path: File to hash.

    Returns:
        Lowercase hexadecimal digest.
    """
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def evaluate_model_artifacts(
    artifacts: list[ModelArtifact], output_dir: Path, scheme: str
) -> dict[str, Any]:
    """Evaluate the action mean of explicitly selected model artifacts.

    Args:
        artifacts: Non-empty model list from compatible training runs.
        output_dir: Fresh directory for evaluation artifacts.
        scheme: Selection-scheme label recorded in the manifest.

    Returns:
        Evaluation metrics.

    Raises:
        TrainerConfigError: If members are absent, missing, or incompatible.
    """
    if not artifacts:
        raise TrainerConfigError("Model evaluation requires at least one artifact.")
    if not scheme:
        raise TrainerConfigError("Model-selection scheme must be non-empty.")
    preflight: list[tuple[ModelArtifact, Any, dict[str, Any], dict[str, Any]]] = []
    member_manifests: list[dict[str, object]] = []
    for artifact in artifacts:
        if not artifact.model_path.is_file():
            raise TrainerConfigError(
                f"Model artifact is missing: {artifact.model_path}"
            )
        snapshot_path = artifact.source_run / "config_snapshot.yaml"
        eval_env_path = artifact.source_run / "env_eval.yaml"
        for required in (snapshot_path, eval_env_path):
            if not required.is_file():
                raise TrainerConfigError(f"Run artifact is missing: {required}")
        try:
            raw = yaml.safe_load(snapshot_path.read_text(encoding="utf-8"))
            resolved_eval = yaml.safe_load(eval_env_path.read_text(encoding="utf-8"))
        except yaml.YAMLError as exc:
            raise TrainerConfigError(
                f"Run YAML artifact is malformed for {artifact.source_run}: {exc}"
            ) from exc
        config = parse_experiment_config(raw)
        if artifact.seed != config.run.seed:
            raise TrainerConfigError(
                f"Model {artifact.model_path} records seed {artifact.seed}, but "
                f"{snapshot_path} records {config.run.seed}."
            )
        meta = _read_run_meta(artifact.source_run)
        member_manifests.append(
            {
                "source_run": str(artifact.source_run),
                "model_path": str(artifact.model_path),
                "model_sha256": _sha256(artifact.model_path),
                "seed": artifact.seed,
                "nominal_timestep": artifact.nominal_timestep,
                "git": meta["git"],
            }
        )
        preflight.append((artifact, config, resolved_eval, raw))

    reference_artifact, reference_config, reference_eval, reference_raw = preflight[0]
    for artifact, config, resolved_eval, raw in preflight[1:]:
        if resolved_eval != reference_eval:
            raise TrainerConfigError(
                f"Model {artifact.model_path} has a different resolved eval env than "
                f"{reference_artifact.model_path}."
            )
        normalized = dict(raw)
        normalized["run"] = dict(normalized["run"])
        normalized["run"]["seed"] = reference_raw["run"]["seed"]
        if normalized != reference_raw:
            raise TrainerConfigError(
                f"Model {artifact.model_path} differs from the reference config in "
                "more than run.seed."
            )
        if config.run.decision_interval != reference_config.run.decision_interval:
            raise TrainerConfigError(
                f"Model {artifact.model_path} has a different decision_interval."
            )

    loaded: list[tuple[ModelArtifact, Any, Any]] = []
    for artifact, config, _, _ in preflight:
        model = ALGO_REGISTRY[config.algorithm.name].algo_class.load(
            artifact.model_path, device=resolve_device(config.run.device)
        )
        loaded.append((artifact, config, model))

    output_dir.mkdir(parents=True, exist_ok=False)
    env = build_single_env(
        reference_eval,
        reference_config.custom_feature_names,
        reference_config.custom_cross_feature_names,
        seed=0,
        decision_interval=reference_config.run.decision_interval,
        residual=reference_config.run.residual,
        rank_allocation=reference_config.run.rank_allocation,
        apply_hold_gate=reference_config.run.apply_hold_gate,
        gate_evaluation_mode=GateEvaluationMode.LEARNED,
    )
    states: list[Any] = [None] * len(loaded)

    def predict(observation: dict[str, Any], episode_start: np.ndarray) -> np.ndarray:
        actions: list[np.ndarray] = []
        for index, (_, _, model) in enumerate(loaded):
            action, states[index] = model.predict(
                observation,
                state=states[index],
                episode_start=episode_start,
                deterministic=True,
            )
            actions.append(np.asarray(action, dtype=np.float64))
        return np.mean(np.stack(actions, axis=0), axis=0).astype(np.float32)

    try:
        walk = walk_eval_range(env, predict)
    finally:
        env.close()
    metrics = compute_metrics(*walk)
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    pd.DataFrame({"timestamp": walk[2], "equity_jpy": walk[1]}).to_csv(
        output_dir / "equity_curve.csv", index=False
    )
    (output_dir / "model_selection.json").write_text(
        json.dumps({"scheme": scheme, "members": member_manifests}, indent=2),
        encoding="utf-8",
    )
    return metrics
