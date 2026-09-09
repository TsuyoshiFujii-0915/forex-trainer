# Frozen turnover/cost diagnostic — Issue #21

Classification: **not successfully translated to portfolio alpha** (unchanged).
Decision: **unresolved**. Gross uncertainty remains; turnover reduction does not establish profitability.

Primary metrics use equal fold weights. Intervals: 10,000 IID / 3-fold circular moving-block draws, seed 16.
Score/rank persistence excludes the initial decision; membership exit rate excludes initial entry.

| Policy / metric | Mean | 2009–2018 | 2019–2025 | IID 95% | Moving-block 95% |
|---|---:|---:|---:|---|---|
| supervised / mean_weight_turnover | 4.9822473 | 5.1253458 | 4.7778208 | [4.862316578451794, 5.095237314252681] | [4.80797608815777, 5.149939928217842] |
| supervised / mean_membership_turnover | 0.77989987 | 0.80237319 | 0.74779512 | [0.7610677106484851, 0.7976454730032851] | [0.7525296046172554, 0.8062347421956688] |
| supervised / mean_actual_notional_over_equity | 4.9870756 | 5.1302242 | 4.7825776 | [4.86768952543417, 5.099756451313133] | [4.813129544065739, 5.154277939346934] |
| supervised / total_cost_ratio | 0.05344046 | 0.054291615 | 0.052224525 | [0.051902829686637474, 0.05499978847852503] | [0.051266888713244314, 0.05556284807160475] |
| supervised / trading_cost_share | 0.67273412 | 0.67842303 | 0.66460712 | [0.667264042653752, 0.6780690629559691] | [0.6649848220676041, 0.6807608754354868] |
| supervised / overnight_cost_share | 0.32726588 | 0.32157697 | 0.33539288 | [0.3219309370440309, 0.33273595734624795] | [0.3192391245645133, 0.335015177932396] |
| supervised / mean_score_correlation | -0.036565059 | -0.10491406 | 0.061076369 | [-0.08589876393411038, 0.013928585593554858] | [-0.11518672681684518, 0.03761021409856734] |
| supervised / mean_rank_correlation | -0.034671692 | -0.094698519 | 0.051080917 | [-0.07877991185920875, 0.009831922438822959] | [-0.10424212072262765, 0.03181708828001926] |
| supervised / mean_reversals | 1.0004703 | 1.0937906 | 0.86715559 | [0.9219546001818127, 1.083821557650864] | [0.8913464138961519, 1.1281866204014706] |
| supervised / mean_bottom_boundary_gap | 0.00017363972 | 0.00020449531 | 0.00012956032 | [0.00013759171646754957, 0.00022082909232469666] | [0.0001315261416092061, 0.00022878396955940545] |
| supervised / mean_top_boundary_gap | 0.00017535297 | 0.0002091616 | 0.00012705493 | [0.00013857275486263006, 0.00022283045947072121] | [0.00013201839536854934, 0.0002315176462369993] |
| supervised / mean_rank_churn | 0.3744005 | 0.3880991 | 0.35483106 | [0.36421449975083936, 0.3842780601689197] | [0.3589891602414363, 0.390084344489811] |
| supervised / mean_score_changed_pairs | 9 | 9 | 9 | [9.0, 9.0] | [9.0, 9.0] |
| reversal / mean_weight_turnover | 1.2432631 | 1.2046163 | 1.2984729 | [1.1848715852842233, 1.3055210021873749] | [1.1771943020974585, 1.30589508404973] |
| reversal / mean_membership_turnover | 0.19269731 | 0.18662738 | 0.20136864 | [0.18353071313779892, 0.20247222988687205] | [0.1823209932655977, 0.2025314909742339] |
| reversal / mean_actual_notional_over_equity | 1.2631366 | 1.2263071 | 1.3157501 | [1.2048888440934762, 1.3245776395553817] | [1.196561069028049, 1.3258654980509335] |
| reversal / total_cost_ratio | 0.027582384 | 0.026915152 | 0.028535572 | [0.026412853748338364, 0.028814530247115458] | [0.026293139228849096, 0.028800921911220728] |
| reversal / trading_cost_share | 0.3438964 | 0.3371033 | 0.35360082 | [0.33141812790218517, 0.3566989947258282] | [0.32932752384013486, 0.35710718377288764] |
| reversal / overnight_cost_share | 0.6561036 | 0.6628967 | 0.64639918 | [0.6433010052741718, 0.6685818720978148] | [0.6428928162271123, 0.6706724761598651] |
| reversal / mean_score_correlation | 0.9390861 | 0.94255451 | 0.93413123 | [0.9345328394822215, 0.9431217479788452] | [0.9342643382326835, 0.9434335942257183] |
| reversal / mean_rank_correlation | 0.8988969 | 0.90245948 | 0.89380751 | [0.891829816074528, 0.9056618778554438] | [0.892505564616438, 0.9058383821770951] |
| reversal / mean_reversals | 0.0045145401 | 0.0035924675 | 0.0058317867 | [0.0018022906155824297, 0.007520184582394259] | [0.001793066095843044, 0.007508571136820463] |
| reversal / mean_bottom_boundary_gap | 0.0063200519 | 0.0069768624 | 0.0053817512 | [0.005471005624861382, 0.00713899893197392] | [0.005457554135971047, 0.007173731971782508] |
| reversal / mean_top_boundary_gap | 0.0061315359 | 0.0067935719 | 0.0051857703 | [0.005360788812315122, 0.006940197577073089] | [0.005344081249368605, 0.007003706225276139] |
| reversal / mean_rank_churn | 0.088109954 | 0.086543723 | 0.090347428 | [0.08459699121373769, 0.09178354142885814] | [0.08450588089212807, 0.09155277128253807] |
| reversal / mean_score_changed_pairs | 9 | 9 | 9 | [9.0, 9.0] | [9.0, 9.0] |

## Cause attribution (descriptive)

Cost shares are means of within-fold shares, not pooled JPY amounts. Total cost = spread + commission + overnight; signed carry is a separate contribution. JSON and CSV split trading/overnight costs.
Retained-target trades include price drift and equity changes due to price, carry and costs.

| Policy | Cause | Mean total cost share | Share of trading cost | 2009–2018 total | 2019–2025 total |
|---|---|---:|---:|---:|---:|
| supervised | initial_entry | 0.372608% | 0.333621% | 0.370265% | 0.375955% |
| supervised | entry | 39.902274% | 33.688495% | 39.200304% | 40.905088% |
| supervised | exit | 22.646328% | 33.692333% | 22.171578% | 23.324542% |
| supervised | reversal | 29.922981% | 32.174189% | 32.037933% | 26.901621% |
| supervised | retained | 7.155809% | 0.111362% | 6.219920% | 8.492794% |
| supervised | inactive | 0.000000% | 0.000000% | 0.000000% | 0.000000% |
| reversal | initial_entry | 0.710477% | 1.239327% | 0.694543% | 0.733239% |
| reversal | entry | 29.446599% | 48.361368% | 28.832103% | 30.324450% |
| reversal | exit | 16.587107% | 48.235338% | 16.247025% | 17.072939% |
| reversal | reversal | 0.259837% | 0.517194% | 0.183444% | 0.368970% |
| reversal | retained | 52.995980% | 1.646773% | 54.042885% | 51.500401% |
| reversal | inactive | 0.000000% | 0.000000% | 0.000000% | 0.000000% |

Maximum absolute sealed-cost residual / starting equity: 1.626e-19.

## Outputs and limits

- pair_transitions.csv: signed target change, drift, actual trade, state and next-decision price/carry/gross/net simple contributions.
- decisions.csv: persistence, boundary gaps, membership events, actual notional/equity and accounting residuals.
- holding_spells.csv / holding_duration_counts.csv: long/short observed durations with right-censoring at every fold end; no cross-fold stitching.
- fold_metrics.csv / cause_metrics.csv: primary fold metrics, exhaustive cost attribution and descriptive next-period state contributions.
- exploratory_states.csv: membership-maintained/changed next-decision return associations, including zero-count states.
- exploratory_boundary.csv: contemporaneous boundary-gap/cost and boundary-gap/exit-rate correlations, with undefined counts/status.
- report.json: both eras, shared fold intervals, all LOO means, undefined counts and original Issue #19 economics (including PPO).

State comparisons and boundary correlations are descriptive, not causal execution effects. No horizon, top-k, threshold, allocation or policy was selected.
Completed-only holding durations do not estimate unconditional persistence; censored observations remain explicit.
No unexplained cost beyond the recorded residual tolerance. Real-world slippage/market impact is absent from this environment; it is not estimated here.
A gate, bandit or low-turnover policy is not justified by these associations. Any next experiment requires a separate preregistration under Issue #20 / ADR-0031.
