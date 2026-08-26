# Issue #1 checkpoint-selection experiment

## Protocol and provenance

- Execution date: 2026-08-25
- Folds: 2009–2025 (17 annual development folds)
- Seeds: 42, 43, 44 (51 training runs)
- Selection schemes: `validation_best`, `last`, `late_checkpoint_ensemble`
- Training device: CPU, as fixed by the committed fold configs
- forex-trainer commit: `3e2f33669b9a3aa9904ccd66168f0f960670c405`
- forex-env-v3 commit: `6024b91c0f3592611849bc231922ab60e6090aed`
- Data SHA-256: `723db7c935dcc27c147007728358d098243eae96998eae146db4c7df02bd081e`
- Study config SHA-256: `5f60d85ee9a445838fa060474657f2c3daaed9ce5fe8ef358c340dd51ec7aa99`

The 51 training runs completed successfully. Each fold uses the same three
training trajectories for all schemes: three validation-best models, three
last models, and 15 equal-weight late checkpoints (five per seed). A
"positive-return fold" below means annualized net return is greater than zero;
it does not mean that the scheme ranked first among the three schemes.

## Overall results

| Scheme | Mean fold net/year | Mean fold gross/year | Mean fold Sharpe | Mean fold DD | Worst fold DD | Positive-return folds |
|---|---:|---:|---:|---:|---:|---:|
| validation_best | 10.31% | 13.06% | 0.222 | 16.52% | 36.95% | 8/17 |
| last | 20.54% | 24.46% | 0.301 | 19.93% | 37.32% | 7/17 |
| late_checkpoint_ensemble | 14.16% | 17.52% | 0.241 | 20.16% | 45.44% | 7/17 |

## Era results

| Era | Scheme | Mean net/year | Mean gross/year | Mean Sharpe | Mean DD | Worst DD | Positive-return folds |
|---|---|---:|---:|---:|---:|---:|---:|
| 2009–2018 | validation_best | 17.24% | 20.43% | 0.493 | 18.76% | 36.95% | 6/10 |
| 2009–2018 | last | 36.03% | 40.79% | 0.654 | 22.18% | 37.32% | 5/10 |
| 2009–2018 | late_checkpoint_ensemble | 25.00% | 29.00% | 0.561 | 22.85% | 45.44% | 5/10 |
| 2019–2025 | validation_best | 0.41% | 2.55% | -0.167 | 13.32% | 22.05% | 2/7 |
| 2019–2025 | last | -1.60% | 1.14% | -0.203 | 16.72% | 23.29% | 2/7 |
| 2019–2025 | late_checkpoint_ensemble | -1.33% | 1.13% | -0.216 | 16.31% | 22.58% | 2/7 |

## Paired differences versus validation-best

Positive return and Sharpe differences favor the candidate. Negative drawdown
differences favor the candidate.

| Candidate | Mean net diff | Median net diff | Net improved | Mean Sharpe diff | Median Sharpe diff | Sharpe improved | Mean DD diff | Median DD diff | DD improved |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| last | +10.23 pt | -0.02 pt | 8/17 | +0.080 | +0.085 | 10/17 | +3.41 pt | +3.66 pt | 3/17 |
| late_checkpoint_ensemble | +3.85 pt | -0.84 pt | 7/17 | +0.019 | +0.049 | 9/17 | +3.64 pt | +2.47 pt | 4/17 |

## Decision

Retain ADR-0005 and keep `validation_best` as the default.

`last` has the highest arithmetic mean return and Sharpe, but its mean return
advantage is concentrated in the 2011 and 2012 folds. Its paired median net
difference is effectively zero, it improves net return in only 8 of 17 folds,
and it worsens drawdown in 14 of 17 folds. In the more recent 2019–2025 era,
validation-best has the best mean net return, Sharpe, mean drawdown, and worst
drawdown.

The late-checkpoint ensemble is not a robust replacement either: its paired
median net difference is negative, it improves net return in 7 of 17 folds,
and its overall worst drawdown is 45.44%, compared with 36.95% for
validation-best. The experiment therefore does not provide sufficiently
consistent evidence to supersede ADR-0005.

Machine-readable results are in [report.json](report.json), and all 51
fold/scheme result rows are in [fold_results.csv](fold_results.csv).
