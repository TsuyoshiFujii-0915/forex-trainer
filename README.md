# forex-trainer

RL agent training and experiment harness for [forex-env-v3](../forex-env-v3). One YAML file defines one experiment over three axes — **algorithm × network × features** — and every run leaves a fully reproducible artifact directory.

- Algorithms (Stable-Baselines3 + sb3-contrib): `ppo`, `recurrent_ppo`, `sac`, `td3`, `tqc`
- Networks (feature extractors over the market window): `mlp`, `cnn1d`, `lstm`, `attention`
- Features: base (`log_return`, `volatility`) + trainer registry (`sma20_ratio`, `rsi14`, `atr14_ratio`, `macd_ratio`) — every registered feature is automatically lookahead-tested
- Tracking: local only — TensorBoard + per-run `metrics.json` / `equity_curve.csv`

## Layout requirement

The two repositories must sit side by side (ADR-0001):

```
01_Projects/
├── forex-env-v3/
└── forex-trainer/
```

## Setup

```bash
uv sync          # Mac: MPS-enabled torch from PyPI; Linux x86_64: CUDA-enabled torch
```

## Workflow

```bash
# 0. (real data) materialize the parquet cache once
uv run forex-env-fetch --config configs/fetch_1d.yaml --output data/jpy_usd_eur_1d.parquet

# 1. define an experiment (copy + edit a YAML; commit it)
cp configs/ppo_mlp_daily.yaml configs/ppo_cnn1d_daily.yaml

# 2. train — creates runs/<experiment>/<timestamp>/ automatically
uv run forex-train --config configs/ppo_cnn1d_daily.yaml

# 3. monitor
uv run tensorboard --logdir runs/

# 4. evaluate on the held-out range (deterministic policy)
uv run forex-eval --run runs/ppo_cnn1d_daily/<timestamp>

# 5. compare all evaluated runs
uv run forex-compare runs
```

Each run directory contains the config snapshot, resolved env configs for train/eval, `meta.json` (git SHAs of both repos, versions, seed, device), TensorBoard logs, `model_final.zip`, and after evaluation `metrics.json` + `equity_curve.csv`. `configs/` is committed; `runs/` is gitignored.

## Experiment YAML

```yaml
experiment: ppo_mlp_daily
env:                       # forex-env-v3 sections, except data dates
  environment: { seed: 42, initial_balance_jpy: 1000000.0, episode_max_steps: 128,
                 window_size: 32, max_leverage: 5.0, margin_call_threshold: 0.2,
                 allow_action_leverage: false, random_start: true,
                 currency_pairs: ["JPY/USD", "JPY/EUR"] }
  data: { provider: file, timeframe: "1d", path: data/jpy_usd_eur_1d.parquet }
  features: { volatility_window: 32, normalize: true,
              selected: [log_return, volatility, rsi14, sma20_ratio] }
  transaction_costs: { commission_rate: 0.0002, overnight_rate: 0.0001,
                       spreads: { JPY/USD: 0.0002, JPY/EUR: 0.0002 } }
train_range: { start: "2023-01-01", end: "2025-06-30" }
eval_range:  { start: "2025-07-01", end: "2025-12-31" }   # must start after train ends
algorithm: { name: ppo, hyperparams: { n_steps: 256, batch_size: 256, learning_rate: 3.0e-4 } }
network:   { name: mlp, kwargs: { features_dim: 128 } }
run: { total_timesteps: 20000, seed: 42, device: auto, n_envs: 4, vec_env: dummy }
```

Rules enforced at load time (fail fast): every key required, unknown keys rejected, `eval_range` must start at or after `train_range.end`, algorithm/network/feature names must exist in their registries, and `env.data` must not contain dates (the ranges inject them).

## Adding to the axes

- **Feature**: write one function in `src/forex_trainer/features.py` (per-pair OHLCV DataFrame → Series, causal, finite from the first rows) and register it in `FEATURE_REGISTRY`. The test suite automatically checks every registered feature for lookahead and for finiteness inside the env.
- **Network**: subclass `BaseFeaturesExtractor` in `src/forex_trainer/networks.py` and register it in `NETWORK_REGISTRY`. The smoke tests train every registered network.
- **Algorithm**: add an `AlgoSpec` to `ALGO_REGISTRY` in `src/forex_trainer/algorithms.py`.

## Docker (NVIDIA GPU)

Mac Docker has no CUDA — on Mac run via `uv` directly (MPS). On a GPU machine:

```bash
# from the parent directory layout above
docker compose build
docker compose run train uv run forex-train --config configs/ppo_mlp_daily.yaml
```

`compose.yaml` uses the parent directory as build context (the env repo is a path dependency), mounts `runs/`, `configs/`, and `data/`, and requests all GPUs. The same `uv.lock` pins versions across uv-direct and Docker; on Linux x86_64 the PyPI torch wheel ships CUDA support. To pin a specific CUDA build instead, add a `[tool.uv.sources]` torch entry pointing at the matching `download.pytorch.org/whl/cuXXX` index.

## Evaluation metrics

`forex-eval` walks the entire eval range once with the deterministic policy (`random_start` off, episode cap lifted) and reports: cumulative log return, final equity ratio, annualized Sharpe (from per-step log returns and actual bar spacing), max drawdown, total cost ratio, and mean gross leverage.

## License

Proprietary. See `pyproject.toml`.
