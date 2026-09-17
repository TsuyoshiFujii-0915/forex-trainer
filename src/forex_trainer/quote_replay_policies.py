"""Frozen canonical/ridge/PPO adapters for the quote execution contract."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import yaml
from gymnasium import spaces
from stable_baselines3 import PPO

from .artifact_provenance import sha256_file
from .config import parse_experiment_config
from .full_period import FEATURES, FEATURE_NAMES, FrozenRidge, Policy, canonical_action, checked_path, read_json, runtime_identity
from .full_period_sources import FrozenEnsemble, load_frozen_ensemble


class FixtureObservationSpace(gym.Env):
    """Inference-only geometry for untrained fixture PPOs; stepping is unsupported."""

    def __init__(self) -> None:
        """Declare the registered market, assets, and direct-action spaces."""
        self.observation_space = spaces.Dict({
            'market': spaces.Box(-np.inf, np.inf, shape=(9, 32, 8), dtype=np.float32),
            'assets': spaces.Box(-np.inf, np.inf, shape=(9, 3), dtype=np.float32),
        })
        self.action_space = spaces.Box(-1, 1, shape=(9, 1), dtype=np.float32)


def fixture_policies() -> tuple[dict[str, Policy], dict[str, Any]]:
    """Build deterministic real models solely to validate inference and accounting.

    Returns:
        Three policy callables and explicit untrained parameter identities.
    """
    coefficients = np.zeros(256)
    coefficients[-5] = 1.
    record = {'feature_names': list(FEATURE_NAMES), 'train_mean': [0.] * 256,
              'train_scale': [1.] * 256, 'coefficients': coefficients.tolist(),
              'intercept': 0., 'selected_alpha': 1.}
    ridge = FrozenRidge(record, Path('synthetic-fixture-ridge'))
    models, hashes = [], []
    for seed in (42, 43, 44):
        model = PPO('MultiInputPolicy', FixtureObservationSpace(), seed=seed, device='cpu',
                    n_steps=2, batch_size=2, policy_kwargs={'net_arch': [8]})
        digest = hashlib.sha256()
        for name, tensor in sorted(model.policy.state_dict().items()):
            digest.update(name.encode())
            digest.update(tensor.detach().cpu().numpy().tobytes())
        hashes.append(digest.hexdigest())
        models.append(model)
    ensemble = FrozenEnsemble(tuple(models))
    return {'canonical': canonical_action, 'ridge': ridge.action, 'ppo_ens3': ensemble.action}, {
        'ppo_kind': 'untrained_fixture_only', 'seeds': [42, 43, 44], 'ppo_parameter_sha256': hashes,
        'ridge_kind': 'synthetic_fixed_coefficients', 'ridge_parameter_sha256': ridge.parameter_sha256,
    }


def frozen_policies(path: Path) -> tuple[dict[str, Policy], dict[str, Any]]:
    """Load one explicitly pinned research fold without training or quote fallback.

    Args:
        path: Manifest containing source, fold and current runtime requirements.

    Returns:
        Registered policies and verified parent/config/model identities.
    """
    manifest = read_json(path)
    if set(manifest) != {'source', 'fold', 'runtime'}:
        raise ValueError(f'Frozen quote sources require source, fold, runtime: {path}')
    runtime = runtime_identity()
    expected = {'forex_env_sha': runtime['git']['forex_env'], 'versions': runtime['versions'], 'device': 'cpu'}
    if manifest['runtime'] != expected:
        raise ValueError(f'Frozen quote runtime mismatch: {path}; actual={expected}')
    parent_path = checked_path(manifest['source'])
    parent = read_json(parent_path)
    if parent['artifact_version'] != 1:
        raise ValueError(f'Unsupported parent seal version: {parent_path}')
    models_path = checked_path({'path': str(parent_path.parent / 'models.json'),
                                'sha256': parent['generated_artifact_sha256']['models.json']})
    fold = manifest['fold']
    models = read_json(models_path)
    if fold not in parent['fold_sources'] or fold not in models:
        raise ValueError(f'Missing frozen fold {fold}: {parent_path}')
    identity = parent['fold_sources'][fold]
    config_path = checked_path({'path': identity['config_path'], 'sha256': identity['config_sha256']})
    raw = yaml.safe_load(config_path.read_text())
    parsed = parse_experiment_config(raw)
    from .quote_replay import SYMBOLS
    if (tuple(parsed.env['environment']['currency_pairs']) != SYMBOLS
            or tuple(parsed.env['features']['selected']) != FEATURES
            or parsed.env['features']['normalize']
            or parsed.env['environment']['window_size'] != 32):
        raise ValueError(f'Frozen observation coordinate mismatch: {config_path}')
    ridge = FrozenRidge(models[fold], models_path)
    ensemble = load_frozen_ensemble(identity['ppo'], raw)
    return {'canonical': canonical_action, 'ridge': ridge.action, 'ppo_ens3': ensemble.action}, {
        'manifest_path': str(path), 'manifest_sha256': sha256_file(path), 'parent': manifest['source'],
        'fold': fold, 'fold_sources': identity, 'ridge_parameter_sha256': ridge.parameter_sha256,
        'ridge_file_sha256': sha256_file(models_path), 'training_and_evaluation_config': raw,
    }
