# issue5_current_longf_baseline

## Aggregate evidence

| Configuration | Net/year | Gross/year | Sharpe | Mean / worst drawdown | Winning folds |
|---|---:|---:|---:|---:|---:|
| direct_longf_ens3 | 10.3098% | 13.0648% | 0.222 | 16.5187% / 36.9549% | 8/17 |

## Paired comparisons


## Statistical assumptions and provenance

- Sampling unit: evaluation_fold. Seed-policy observations are averaged within each fold; action-mean ensemble metrics are observed directly once per fold.
- Bootstrap draws: 10000; deterministic seed: 5.
- Moving-block length: 3 adjacent folds with circular wrapping.
- Campaign trials: 9; reported configurations: 1.
- Selection-bias statistic: not_estimated. Probabilistic/deflated Sharpe is not reported because fold-level aggregate artifacts do not justify the return-distribution, autocorrelation, and effective-independent-trial assumptions.
- Full run, config, data, device, software, and model-selection provenance is in `provenance.json`.
