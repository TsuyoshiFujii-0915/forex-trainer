# Frozen ranking spread-to-net diagnostic

Classification retained: **not successfully translated to portfolio alpha**

Stages and adjacent differences: annual log return, equal-weight folds. Original tail: log return/decision; original gross/net: annual simple return. Never compare CI bounds across different units.

| Series (unit) | Mean | 2009–2018 | 2019–2025 | IID 95% | 3-fold block 95% | LOO min / max | Largest abs fold (share) | Without top 3 |
|---|---:|---:|---:|---|---|---|---|---:|
| scaled_tail_spread (annual log) | 0.04985078 | 0.05331457 | 0.04490250 | [0.00608758, 0.09584151] | [0.00360260, 0.09828937] | 0.03591043 / 0.06000359 | 2025 (19.75%) | 0.01939972 |
| weighted_price_log_return (annual log) | 0.04985078 | 0.05331457 | 0.04490250 | [0.00608758, 0.09584151] | [0.00360260, 0.09828937] | 0.03591043 / 0.06000359 | 2025 (19.75%) | 0.01939972 |
| price_log_return (annual log) | 0.04311886 | 0.04562792 | 0.03953448 | [0.00010438, 0.08895075] | [-0.00231394, 0.09047322] | 0.02909261 / 0.05313169 | 2025 (19.96%) | 0.01292618 |
| gross_log_return (annual log) | 0.04046785 | 0.04294057 | 0.03693539 | [-0.00256160, 0.08597685] | [-0.00460504, 0.08724196] | 0.02655111 / 0.05047226 | 2025 (19.87%) | 0.01077696 |
| net_log_return (annual log) | -0.03104851 | -0.02983195 | -0.03278645 | [-0.07368007, 0.01392407] | [-0.07514846, 0.01486773] | -0.04494984 / -0.02103947 | 2025 (14.20%) | -0.06021209 |
| weighted_price_log_return_minus_scaled_tail_spread (annual log) | 0.00000000 | 0.00000000 | 0.00000000 | [0.00000000, 0.00000000] | [0.00000000, 0.00000000] | 0.00000000 / 0.00000000 | 2025 (19.75%) | 0.00000000 |
| price_log_return_minus_weighted_price_log_return (annual log) | -0.00673192 | -0.00768664 | -0.00536802 | [-0.00820159, -0.00536245] | [-0.00839437, -0.00520613] | -0.00699748 / -0.00637883 | 2010 (10.82%) | -0.00741630 |
| gross_log_return_minus_price_log_return (annual log) | -0.00265101 | -0.00268735 | -0.00259909 | [-0.00394252, -0.00134441] | [-0.00392571, -0.00149566] | -0.00303830 / -0.00233344 | 2011 (14.57%) | -0.00347647 |
| net_log_return_minus_gross_log_return (annual log) | -0.07151636 | -0.07277253 | -0.06972184 | [-0.07270634, -0.07032362] | [-0.07331463, -0.06982071] | -0.07183241 / -0.07114726 | 2014 (6.37%) | -0.07222771 |
| supervised: mean_tail_spread (log/decision) | 0.00011991 | 0.00012776 | 0.00010868 | [0.00001469, 0.00023105] | [0.00000862, 0.00023665] | 0.00008596 / 0.00014421 | 2025 (19.97%) | 0.00004658 |
| supervised: annualized_gross_return (annual simple) | 0.04586437 | 0.04686468 | 0.04443537 | [0.00086498, 0.09506699] | [-0.00161808, 0.09619203] | 0.02991820 / 0.05577634 | 2025 (21.64%) | 0.01321751 |
| supervised: annualized_net_return (annual simple) | -0.02638032 | -0.02670316 | -0.02591913 | [-0.06827282, 0.01903751] | [-0.06976895, 0.01993953] | -0.04121102 / -0.01715240 | 2025 (15.99%) | -0.05622017 |
| supervised: total_cost_ratio (initial equity ratio) | 0.05344046 | 0.05429162 | 0.05222453 | [0.05190283, 0.05499979] | [0.05126689, 0.05556285] | 0.05300927 / 0.05378109 | 2025 (6.64%) | 0.05241196 |
| reversal: mean_tail_spread (log/decision) | 0.00020534 | 0.00012461 | 0.00032067 | [0.00006437, 0.00035287] | [0.00005548, 0.00034576] | 0.00016928 / 0.00023893 | 2025 (15.96%) | 0.00010711 |
| reversal: annualized_gross_return (annual simple) | 0.08808772 | 0.04822342 | 0.14503671 | [0.02457620, 0.15576290] | [0.02266756, 0.15212219] | 0.06961174 / 0.10200768 | 2025 (18.05%) | 0.04093698 |
| reversal: annualized_net_return (annual simple) | 0.04983122 | 0.01178247 | 0.10418659 | [-0.01065583, 0.11451203] | [-0.01268340, 0.11091216] | 0.03222335 / 0.06315401 | 2025 (16.86%) | 0.00480084 |
| reversal: total_cost_ratio (initial equity ratio) | 0.02758238 | 0.02691515 | 0.02853557 | [0.02641285, 0.02881453] | [0.02629314, 0.02880092] | 0.02723074 / 0.02779856 | 2025 (7.08%) | 0.02677531 |
| ppo: mean_tail_spread (log/decision) | -0.00004087 | -0.00007625 | 0.00000968 | [-0.00013916, 0.00006368] | [-0.00012638, 0.00003145] | -0.00007267 / -0.00002015 | 2025 (16.05%) | -0.00010594 |
| ppo: annualized_gross_return (annual simple) | 0.13064795 | 0.20427617 | 0.02546478 | [-0.01507054, 0.29313454] | [-0.01797915, 0.30565171] | 0.07962013 / 0.16039642 | 2013 (22.99%) | 0.01604032 |
| ppo: annualized_net_return (annual simple) | 0.10309791 | 0.17238432 | 0.00411732 | [-0.03872975, 0.26092087] | [-0.04075664, 0.27147433] | 0.05416571 / 0.13239950 | 2013 (22.00%) | -0.00656253 |
| ppo: total_cost_ratio (initial equity ratio) | 0.01913760 | 0.02139388 | 0.01591433 | [0.01668093, 0.02181462] | [0.01665433, 0.02247704] | 0.01827557 / 0.01959105 | 2013 (10.12%) | 0.01708538 |

All table intervals use shared seed-16 draws. The original Issue #15 seed-15 intervals are independently reproduced below and in original_reported_intervals; the ranking metric definition is unchanged.

| Original source interval | Seed | IID 95% | Moving-block 95% |
|---|---:|---|---|
| supervised: mean_tail_spread | 15 | [0.00001444, 0.00023274] | [0.00000556, 0.00023729] |
| supervised: annualized_gross_return | 16 | [0.00086498, 0.09506699] | [-0.00161808, 0.09619203] |
| supervised: annualized_net_return | 16 | [-0.06827282, 0.01903751] | [-0.06976895, 0.01993953] |
| reversal: mean_tail_spread | 15 | [0.00006199, 0.00034918] | [0.00005160, 0.00034410] |
| reversal: annualized_gross_return | 16 | [0.02457620, 0.15576290] | [0.02266756, 0.15212219] |
| reversal: annualized_net_return | 16 | [-0.01065583, 0.11451203] | [-0.01268340, 0.11091216] |
| ppo: mean_tail_spread | 15 | [-0.00013877, 0.00006351] | [-0.00012681, 0.00003116] |
| ppo: annualized_gross_return | 16 | [-0.01507054, 0.29313454] | [-0.01797915, 0.30565171] |
| ppo: annualized_net_return | 16 | [-0.03872975, 0.26092087] | [-0.04075664, 0.27147433] |

Price-only compounds the realized-path price contributions; it is not an independently operated cost-free account.
Annual gross minus net is a difference of expm1 values, not total cost / initial equity.
All individual fold and leave-one-fold-out values are in report.json and fold_metrics.csv; exact shared resampling indices are in bootstrap_indices.npz.
The weighted-price minus scaled-tail mean difference (7.428e-10 annual log units) is float32 weight precision, not economic evidence; unrounded values are retained in JSON/CSV.

Maximum per-step accounting residual: 3.461e-16 (simple return).

## Source provenance SHA-256

- ranking: `a398ee18a01de364e21deb1d5f0ac5a66c1c1e445e60dc21c7c713215cebfb61`
- portfolio: `d2db580d5f11ec7ca341af04549c3a09e7ddb6777c43562b1117a24ae8a57db3`
