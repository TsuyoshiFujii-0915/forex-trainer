# issue4_apply_hold_gate

## Aggregate evidence

| Configuration | Net/year | Gross/year | Sharpe | Mean / worst drawdown | Cost ratio | Gross leverage | Turnover | Winning folds |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| direct_longf_ens3 | 10.3098% | 13.0648% | 0.222 | 16.5187% / 36.9549% | 1.9138% | 2.173 | 0.861 | 8/17 |
| gated_longf_ens3 | 2.9208% | 4.6874% | 0.087 | 14.6834% / 46.9661% | 1.2872% | 1.785 | 0.396 | 9/17 |
  - Gate mode `learned`: apply 59.64%, hold 40.36%, mean hold run 8.19, avoided turnover 122.336.
| gated_forced_apply_ens3 | 4.1186% | 6.0628% | 0.226 | 13.0513% / 46.2635% | 1.4208% | 1.757 | 0.583 | 9/17 |
  - Gate mode `forced_apply`: apply 59.67%, hold 40.33%, mean hold run 8.39, avoided turnover 0.000.

## Paired comparisons

- gated_longf_ens3 − direct_longf_ens3: mean net/year difference -7.3890%; fold-bootstrap 95% CI [-19.1543%, 3.6497%]; moving-block 95% CI [-17.7303%, 2.5865%]. Gross/year difference -8.3774%; cost-ratio difference -0.6266%; positive folds 9/17; largest absolute fold 2013 (18.14% of total absolute fold effect).
- gated_longf_ens3 − gated_forced_apply_ens3: mean net/year difference -1.1978%; fold-bootstrap 95% CI [-6.3571%, 4.9454%]; moving-block 95% CI [-5.5523%, 4.0075%]. Gross/year difference -1.3754%; cost-ratio difference -0.1336%; positive folds 7/17; largest absolute fold 2014 (33.99% of total absolute fold effect).
- gated_forced_apply_ens3 − direct_longf_ens3: mean net/year difference -6.1912%; fold-bootstrap 95% CI [-17.0221%, 4.1992%]; moving-block 95% CI [-15.5572%, 2.9259%]. Gross/year difference -7.0020%; cost-ratio difference -0.4930%; positive folds 8/17; largest absolute fold 2013 (18.00% of total absolute fold effect).

## Statistical assumptions and provenance

- Sampling unit: evaluation_fold. Seed-policy observations are averaged within each fold; action-mean ensemble metrics are observed directly once per fold.
- Bootstrap draws: 10000; deterministic seed: 5.
- Moving-block length: 3 adjacent folds with circular wrapping.
- Campaign trials: 10; reported configurations: 3.
- Selection-bias statistic: not_estimated. Probabilistic/deflated Sharpe is not reported because fold-level aggregate artifacts do not justify the return-distribution, autocorrelation, and effective-independent-trial assumptions.
- Full run, config, data, device, software, and model-selection provenance is in `provenance.json`.
