"""Algorithm registry and model construction (ADR-0002)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from sb3_contrib import TQC, RecurrentPPO
from stable_baselines3 import PPO, SAC, TD3
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.vec_env import VecEnv

from .networks import NETWORK_REGISTRY

if TYPE_CHECKING:
    from .config import ExperimentConfig


@dataclass(frozen=True)
class AlgoSpec:
    """Pairing of an SB3 algorithm class with its Dict-observation policy."""

    algo_class: type[BaseAlgorithm]
    policy: str


ALGO_REGISTRY: dict[str, AlgoSpec] = {
    "ppo": AlgoSpec(PPO, "MultiInputPolicy"),
    "recurrent_ppo": AlgoSpec(RecurrentPPO, "MultiInputLstmPolicy"),
    "sac": AlgoSpec(SAC, "MultiInputPolicy"),
    "td3": AlgoSpec(TD3, "MultiInputPolicy"),
    "tqc": AlgoSpec(TQC, "MultiInputPolicy"),
}


def resolve_device(device: str) -> str:
    """Resolve the configured device, expanding "auto".

    Args:
        device: One of "auto", "cpu", "cuda", "mps".

    Returns:
        Concrete device string. "auto" prefers cuda, then mps, then cpu.
    """
    if device != "auto":
        return device
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def build_model(
    config: "ExperimentConfig",
    vec_env: VecEnv,
    device: str,
    tensorboard_dir: Path,
) -> BaseAlgorithm:
    """Instantiate the configured algorithm with the configured network.

    Args:
        config: Experiment configuration.
        vec_env: Vectorized training environment.
        device: Concrete torch device string.
        tensorboard_dir: Directory for TensorBoard event files.

    Returns:
        Untrained SB3 model.
    """
    spec = ALGO_REGISTRY[config.algorithm.name]
    policy_kwargs = {
        "features_extractor_class": NETWORK_REGISTRY[config.network.name],
        "features_extractor_kwargs": dict(config.network.kwargs),
    }
    return spec.algo_class(
        spec.policy,
        vec_env,
        policy_kwargs=policy_kwargs,
        seed=config.run.seed,
        device=device,
        tensorboard_log=str(tensorboard_dir),
        verbose=0,
        **dict(config.algorithm.hyperparams),
    )
