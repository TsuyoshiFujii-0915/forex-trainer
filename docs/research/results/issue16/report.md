# Frozen supervised portfolio translation

Classification: **not successfully translated to portfolio alpha**

Equal-weight fold means; daily decisions are not independent bootstrap samples.

| Policy | Net annualized | Gross annualized | Sharpe | Mean / worst MDD | Winning folds | Leverage | Mean turnover | Mean total cost ratio |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| supervised | -2.64% | 4.59% | -0.339 | 10.57% / 18.21% | 7/17 | 3.200 | 4.982 | 5.34% |
| reversal | 4.98% | 8.81% | 0.240 | 9.84% / 14.94% | 9/17 | 3.199 | 1.243 | 2.76% |
| ppo | 10.31% | 13.06% | 0.222 | 16.52% / 36.95% | 8/17 | 2.173 | 0.861 | 1.91% |

## Paired annualized return differences

| Control | Return | Mean | IID 95% | Moving block 95% |
|---|---|---:|---:|---:|
| reversal | annualized_net_return | -7.62% | [-13.57%, -1.57%] | [-12.28%, -3.18%] |
| reversal | annualized_gross_return | -4.22% | [-10.51%, 2.16%] | [-9.19%, 0.51%] |
| ppo | annualized_net_return | -12.95% | [-28.13%, 0.83%] | [-28.80%, 0.05%] |
| ppo | annualized_gross_return | -8.48% | [-24.23%, 5.69%] | [-24.87%, 4.98%] |

Decision: Stop before bandit work and reconcile predictive objectives with portfolio economics.

See report.json for eras, classification inputs, all leave-one-fold-out means, rank contributions, and concentration diagnostics.
See fold_metrics.csv, steps.csv, and pair_contributions.csv for aligned observations.
