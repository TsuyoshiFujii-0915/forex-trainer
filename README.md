# forex-trainer

RL agent training and experiment harness for [forex-env-v3](../forex-env-v3). One YAML file defines one experiment over three axes — **algorithm × network × features** — and every run leaves a fully reproducible artifact directory.

- Algorithms (Stable-Baselines3 + sb3-contrib): `ppo`, `recurrent_ppo`, `sac`, `td3`, `tqc`
- Networks (feature extractors over the market window): `mlp`, `cnn1d`, `lstm`, `attention`
- Features: base (`log_return`, `volatility`) + trainer registry (`sma20_ratio`, `rsi14`, `atr14_ratio`, `macd_ratio`, `mom24`, `mom72`, `mom168`) — every registered feature is automatically lookahead-tested
- Decision interval (ADR-0004): `run.decision_interval` holds each target allocation for k bars, structurally capping turnover
- Rank allocation (ADR-0010/0011): optional learned pair scores are converted into a fixed-gross sparse top-k/bottom-k portfolio without embedding a trading signal or clipping finite score order
- Model selection (ADR-0005): training periodically walks `val_range`; the best validation model becomes `model_final.zip` (`model_last.zip` keeps the end-of-training model)
- Selection audit: every run also saves five late checkpoints (80/85/90/95/100% of the nominal budget, aligned to validation events for the longf protocol) so validation-best, last, and checkpoint averaging can be compared without retraining three times
- Tracking: local only — TensorBoard + per-run `metrics.json` / `equity_curve.csv`

## Research log

実験の経緯・確立した方法論・現在の結論は [docs/research/](docs/research/README.md) を参照。

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
# or long-history hourly data (2015+) from the Dukascopy public datafeed (ADR-0006)
uv run forex-fetch-dukascopy --pairs "JPY/USD,JPY/EUR" --start 2015-01-01 --end 2026-06-30 \
  --output data/jpy_usd_eur_1h_long.parquet

# Rebuild the clean nine-pair daily longf cache used by Issue #7.
# The expected repair count is a data-lineage assertion: stop if yfinance changes.
uv run forex-env-fetch --config configs/fetch_1d9p_2003.yaml \
  --output data/jpy_9pairs_1d_2003.parquet
uv run forex-clean-spikes --input data/jpy_9pairs_1d_2003.parquet \
  --output data/jpy_9pairs_1d_2003_clean.parquet \
  --residual-threshold 0.08 --reversal-tolerance 0.04 --expected-repairs 14
uv run forex-add-carry --input data/jpy_9pairs_1d_2003_clean.parquet \
  --output data/jpy_9pairs_1d_2003_carry.parquet --lag-days 60

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

# 6. generate the fold/seed aggregate evidence used for research decisions
uv run forex-report \
  --campaign configs/studies/<campaign>.yaml \
  --output-dir runs/reports/<campaign>

# Reproduce issue #1 across the 17 longf folds and seeds 42/43/44.
# This requires the canonical longf parquet cache named by the fold configs.
uv run forex-selection-study \
  --study configs/studies/issue1_longf_checkpoint_selection.yaml \
  --runs-root runs
```

Each run directory contains the config snapshot, resolved env configs for train/val/eval, `meta.json` (git SHAs of both repos, versions, seed, requested/resolved device, and training-time data identity), TensorBoard logs, `model_final.zip` (validation-selected), `model_last.zip`, `late_checkpoints.json` plus five models under `late_checkpoints/`, the validation history `evaluations.npz`, and after evaluation `metrics.json` + `equity_curve.csv` + `evaluation.json`. Evaluation manifests are versioned and seal the selected model, metrics, config snapshot, resolved eval env, actual evaluation device, evaluation-time Git SHAs and dependency versions, and data identity with SHA-256 digests. `configs/` is committed; `runs/` is gitignored.

`forex-selection-study` trains each fold/seed exactly once. For each fold it
compares three action-mean policies from the same source matrix: three
validation-best models, three last models, and all 15 late checkpoints (five
per seed). Its timestamped study directory records the source runs, exact model
paths and SHA-256 digests, data identity, fold metrics, era summaries, winning
folds, and paired differences against validation-best as JSON, CSV, and
Markdown.

`forex-report` consumes an explicit campaign manifest and never chooses a
"latest" run implicitly. Paths in each configuration's `runs` list are
resolved relative to the campaign YAML. File-backed `env.data.path` values in
run snapshots follow the repository convention and are resolved from the
working directory. A minimal campaign has this shape:

```yaml
name: candidate_vs_baseline
configurations:
  baseline:
    model_selection: validation_best
    result_kind: ensemble
    range_policy: { kind: rolling, train_years: 2 }
    runs:
      - ../../runs/wf2019_baseline_ens3/<timestamp>
      - ../../runs/wf2020_baseline_ens3/<timestamp>
  candidate:
    model_selection: validation_best
    result_kind: ensemble
    range_policy: { kind: expanding, train_start: "2003-06-01" }
    runs:
      - ../../runs/wf2019_candidate_ens3/<timestamp>
      - ../../runs/wf2020_candidate_ens3/<timestamp>
comparisons:
  - { baseline: baseline, candidate: candidate }
eras:
  pre_2019: { start: 2009, end: 2018 }
  recent: { start: 2019, end: 2025 }
bootstrap_samples: 10000
bootstrap_seed: 7
moving_block_length: 2
trial_count: 12
```

Use `result_kind: seed` and list every fold/seed run when the policy is one
validation-selected model. Use `result_kind: ensemble` and list one
`forex-ensemble-eval` directory per fold when the policy is the action mean of
multiple seeds. Ensemble metrics are consumed directly as one observation per
fold; seed metrics are never averaged as a proxy for ensemble behavior.
`model_selection` is an expected value that must match the sealed artifact and
is not a replacement for provenance.

Every configuration declares its training-range treatment. Rolling windows
accept exactly 2, 4, or 8 calendar years; expanding windows require an explicit
fixed `train_start`. The report checks that training ends at the six-month
validation window and that evaluation spans the following calendar year. It
keeps the normalized range policy in protocol identity while also recording
every fold's absolute train/validation/evaluation dates.

The command rejects incomplete or duplicate matrices, misaligned effective
evaluation periods, changed training data or sealed artifacts, and mixed data,
training/evaluation-device, or model-selection conditions. Training and evaluation Git,
dependency, requested/resolved-device, range-policy, and protocol differences
between baseline and candidate remain visible treatment provenance. The command
writes canonical observations, fold/seed or fold/ensemble aggregates, paired
candidate-minus-baseline differences, fold and moving-block bootstrap intervals,
provenance, and the selection-bias limitation as JSON, CSV, and Markdown.

Checkpoint-selection composites keep using `forex-selection-study`, whose
existing report contract remains unchanged.

Runs and ensembles created before the version-2 evaluation provenance contracts
remain reproducible through their existing study-specific commands and
committed outputs, but are rejected by `forex-report` instead of receiving
unverifiable fallback provenance. In particular, the existing `longf ens3`
baseline must be regenerated with the current `forex-eval` and
`forex-ensemble-eval` before it can be admitted to a generic campaign report.

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
train_range: { start: "2023-01-01", end: "2025-03-31" }
val_range:   { start: "2025-03-31", end: "2025-06-30" }   # model selection only
eval_range:  { start: "2025-07-01", end: "2025-12-31" }   # final holdout
algorithm: { name: ppo, hyperparams: { n_steps: 256, batch_size: 256, learning_rate: 3.0e-4 } }
network:   { name: mlp, kwargs: { features_dim: 128 } }
run: { total_timesteps: 20000, seed: 42, device: auto, n_envs: 4, vec_env: dummy,
       decision_interval: 1, residual: none, rank_allocation: none }
```

Rules enforced at load time (fail fast): every key required, unknown keys rejected, ranges must satisfy `train_range.end <= val_range.start < val_range.end <= eval_range.start`, `run.decision_interval >= 1`, algorithm/network/feature names must exist in their registries, and `env.data` must not contain dates (the ranges inject them). `run.total_timesteps` counts agent decisions; with `decision_interval: k` one decision spans k bars.

To expose sparse portfolio geometry while leaving the score signal entirely to
the policy, replace `rank_allocation: none` with:

```yaml
rank_allocation: { top_k: 2, gross_exposure: 2.0 }
```

The highest scores are long, the lowest are short, and all other pairs are
flat. The gross exposure is split equally between both sides. Rank allocation
accepts the full finite float32 score domain so ranking is preserved, requires
pinned leverage, and cannot be combined with residual actions.

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

`forex-eval` walks the entire eval range once with the deterministic policy (`random_start` off, episode cap lifted) and reports: cumulative and annualized return (net and gross of transaction costs), final equity ratio, annualized Sharpe (from per-step log returns and actual bar spacing), max drawdown, total cost ratio, mean gross leverage, and mean/total target-weight turnover.

## License

Proprietary. See `pyproject.toml`.
