# issue15_supervised_cross_sectional_ranking

Classification: **established learnable**

| score | mean rank IC | mean top2-bottom2 spread | positive IC fraction |
|---|---:|---:|---:|
| supervised | 0.0109211 | 0.000119908 | 0.508173 |
| reversal | 0.0182019 | 0.000205344 | 0.518972 |
| ppo | -0.00338181 | -4.08685e-05 | 0.488095 |

The primary tail-spread 95% intervals use evaluation folds as the sampling unit.

- IID fold: [1.44418e-05, 0.000232737]
- Circular moving block (3 folds): [5.56285e-06, 0.000237289]
- Scores are predictive diagnostics only; no transaction-cost or portfolio optimization is performed.
