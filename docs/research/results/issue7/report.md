# issue7_longf_data_scaling

![Scaling curve](scaling_curve.svg)

## Net-return scaling summary

| Condition | Observations | Mean net log return | Annualized net return | Fold bootstrap 95% CI | 2-fold block 95% CI | Median fold seed SD | ens3 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 2y | 21 | -0.037869 | -0.048850 | [-0.169136, 0.105607] | [-0.182109, 0.106353] | 0.051679 | -0.03654385974811071 |
| 4y | 21 | -0.022055 | -0.028748 | [-0.146910, 0.108815] | [-0.157426, 0.090394] | 0.136891 | -0.0033155629045531153 |
| 8y | 21 | 0.044337 | 0.060391 | [-0.012432, 0.156931] | [-0.016933, 0.144969] | 0.176551 | 0.07938575921714901 |
| expanding | 21 | 0.008112 | 0.010786 | [-0.092169, 0.126474] | [-0.097943, 0.132099] | 0.119086 | 0.03623408708254659 |

## Adjacent paired differences

- 2y → 4y: mean Δ log return 0.015814 (7 folds / 21 seed-fold pairs, fold 95% CI [-0.040371, 0.079049], 2-fold block 95% CI [-0.035243, 0.080065])
- 4y → 8y: mean Δ log return 0.066392 (7 folds / 21 seed-fold pairs, fold 95% CI [-0.007369, 0.129924], 2-fold block 95% CI [0.008017, 0.129016])
- 8y → expanding: mean Δ log return -0.036225 (7 folds / 21 seed-fold pairs, fold 95% CI [-0.081236, 0.009637], 2-fold block 95% CI [-0.078875, 0.005058])

## Reproducibility

- Study SHA-256: `43bbac00baf50b6a4ffa1ed67e5f572e1b08081ad727cc482c2b443c2c9419ee`
- Completed source runs: 84
- Completed ensembles: 28
- Audit rows: 38
