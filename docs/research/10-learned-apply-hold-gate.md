# 10. learned apply/hold execution gate（Issue #4）

期間: 2026-09-01 / 対象: direct `longf ens3`、評価年2009〜2025

## 結論

learned apply/hold gateは**非採用 / unproven**とする。17 foldsのpaired年率net returnはcurrent direct baseline比
**−7.39ポイント**、same-model forced apply比**−1.20ポイント**だった。gateはdecisionの40.4%で
holdし、turnoverとcostを下げたが、gross returnの低下がcost削減を上回った。したがって改善をgate
behaviorへ帰属できず、current generic PPO `longf` formulationへgateを採用しない。

Issue #4で事前に定めたとおり、zero threshold以外を同じevaluation foldsで探索せず、これを
current formulationに対する最後のexecution-wrapper実験とする。次の研究branchでは
lower-capacityまたは別定式化の学習問題をdirect PPO baselineと比較する。

## 事前登録プロトコル

唯一のtraining treatmentは、direct pair-weight proposalへ1個のbounded gate scalarを追加すること
である。`gate < 0`は直前のeffective target allocationを再送し、`gate >= 0`は新しいproposalを
適用する。thresholdはzeroに固定し、Issue #5のregime feature、residual、rank allocation、learned
gross sizing、architecture/HPO変更を加えていない。

- folds: 2009〜2025の17 walk-forward folds
- seeds: 42 / 43 / 44
- training budget: 各run 300,000 requested steps（実績303,104 steps）
- policy: PPO + MLP `longf`、validation-best
- ensemble: 3 memberのproposalとgate signalを平均してからgate判定を1回だけ実行
- attribution: 同じ3 modelを`learned`と`forced_apply`で評価
- uncertainty: 10,000回のIID fold bootstrapと3-fold circular moving-block bootstrap
- trial count: Issue #5までの9 trialsに事前登録gate treatment 1件を加えた累計10

Issue #5 controlは再学習していない。封印済み51 source modelを現行evaluatorで再評価し、17個の
control ens3を新規生成した。全foldのmetricsはIssue #5 controlと一致した。gated treatmentは
17 folds × 3 seedsの全51 runを新規学習し、各modelとens3をlearned / forced applyの両modeで
評価した。

## Provenance

- data: `data/jpy_9pairs_1d_2003_carry.parquet`
- data SHA-256: `723db7c935dcc27c147007728358d098243eae96998eae146db4c7df02bd081e`
- gated training / all evaluation trainer SHA: `0771b0d06d145cf68dbd9def5c37dac3d07743c8`
- reused control training trainer SHA: `49c04492aa4a0902ea97f81edc5d9217b0f2bf75`
- forex-env SHA: `6024b91c0f3592611849bc231922ab60e6090aed`
- requested / resolved training and evaluation device: CPU / CPU
- versions: stable-baselines3 2.8.0、sb3-contrib 2.8.0、torch 2.12.0、Gymnasium 1.2.3
- member seeds: 42 / 43 / 44
- model selection: validation-best

`forex-report`はdata identity、member seeds、device、model selection、evaluation Git/dependencies、
eval intervalを一致させた。learnedとforced applyは各foldでmodel/config/meta SHAが完全一致し、
same-model attribution contractを満たした。

## 3構成の結果

値は17 foldの算術平均。MDDはfold平均で、worst MDDは全fold中の最大値である。

| Policy | Net/年 | Gross/年 | Sharpe | Mean MDD | Worst MDD | Cost ratio | Mean gross | Mean turnover | 勝ちfold |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| direct control | +10.31% | +13.06% | 0.222 | 16.52% | 36.95% | 1.91% | 2.173 | 0.861 | 8/17 |
| gated learned | +2.92% | +4.69% | 0.087 | 14.68% | 46.97% | 1.29% | 1.785 | 0.396 | 9/17 |
| gated forced apply | +4.12% | +6.06% | 0.226 | 13.05% | 46.26% | 1.42% | 1.757 | 0.583 | 9/17 |

learned gateはdirect比でcost ratioを0.63ポイント、mean turnoverを0.465下げた一方、gross returnを
8.38ポイント失った。net差は−7.39ポイントであり、cost削減だけでは失ったgross alphaを補えない。
forced apply比でもcost ratioは0.13ポイント低いがgross returnは1.38ポイント低く、net差は
−1.20ポイントだった。

mean gross leverageはdirectの2.173からlearnedの1.785へ17.9%低下した。near-flatへのcollapseでは
ないが、forced applyも1.757であるため、directとの差の大部分はgate decisionだけでなく、追加action
headを含むtraining treatmentで学習されたproposal headの変化を含む。MDD平均の改善もworst MDDの
悪化を伴い、net generalizationの代用にはならない。

## Paired attributionと不確実性

| Comparison | Net差/年 | Gross差/年 | Cost ratio差 | 改善fold | IID fold 95% CI | Moving-block 95% CI | 最大fold寄与 |
|---|---:|---:|---:|---:|---:|---:|---:|
| learned − direct | −7.39pt | −8.38pt | −0.63pt | 9/17 | [−19.15pt, +3.65pt] | [−17.73pt, +2.59pt] | 2013 / 18.1% |
| learned − forced apply | −1.20pt | −1.38pt | −0.13pt | 7/17 | [−6.36pt, +4.95pt] | [−5.55pt, +4.01pt] | 2014 / 34.0% |
| forced apply − direct | −6.19pt | −7.00pt | −0.49pt | 8/17 | [−17.02pt, +4.20pt] | [−15.56pt, +2.93pt] | 2013 / 18.0% |

両primary comparisonでpoint estimateが負であり、IID・moving-blockの両区間がゼロを跨ぐ。
uncertainty以前にnet improvementとgate attributionの必要条件を満たさない。

## Eraとfoldの一貫性

| Era | Direct net/年 | Learned net/年 | Forced net/年 | Learned−direct | Learned−forced | Apply |
|---|---:|---:|---:|---:|---:|---:|
| 2009〜2018 | +17.24% | +5.25% | +6.54% | −11.99pt | −1.29pt | 63.2% |
| 2019〜2025 | +0.41% | −0.41% | +0.66% | −0.82pt | −1.07pt | 54.5% |

learned gateは両eraでdirectとforced applyの双方を下回った。単一tail yearだけに依存した非採用判断では
ない。fold-level結果は次のとおり。

| Fold | Direct net | Learned net | Forced net | Learned−direct | Learned−forced | Apply |
|---:|---:|---:|---:|---:|---:|---:|
| 2009 | −23.20% | −40.45% | −37.73% | −17.25pt | −2.72pt | 91.8% |
| 2010 | +13.77% | −8.58% | −9.22% | −22.35pt | +0.64pt | 68.5% |
| 2011 | +42.20% | +3.51% | +3.51% | −38.69pt | +0.01pt | 63.1% |
| 2012 | +44.23% | +61.48% | +80.31% | +17.26pt | −18.82pt | 77.7% |
| 2013 | +88.60% | +26.12% | +30.53% | −62.48pt | −4.41pt | 37.8% |
| 2014 | −4.38% | +25.02% | −14.81% | +29.39pt | +39.83pt | 66.2% |
| 2015 | −36.57% | −28.87% | −31.34% | +7.71pt | +2.47pt | 85.4% |
| 2016 | +0.19% | −7.11% | +13.24% | −7.30pt | −20.35pt | 56.9% |
| 2017 | −4.08% | +6.25% | +7.90% | +10.33pt | −1.65pt | 48.7% |
| 2018 | +51.62% | +15.12% | +22.98% | −36.50pt | −7.86pt | 36.4% |
| 2019 | −17.95% | −1.46% | −1.01% | +16.49pt | −0.45pt | 68.5% |
| 2020 | −16.83% | −5.38% | −8.36% | +11.45pt | +2.98pt | 54.8% |
| 2021 | −5.67% | −5.57% | −6.55% | +0.10pt | +0.99pt | 45.7% |
| 2022 | −2.25% | +8.91% | +7.42% | +11.17pt | +1.50pt | 54.6% |
| 2023 | −2.81% | +2.68% | +4.41% | +5.49pt | −1.73pt | 58.9% |
| 2024 | +27.13% | +11.90% | +13.01% | −15.23pt | −1.11pt | 54.5% |
| 2025 | +21.26% | −13.94% | −4.27% | −35.20pt | −9.67pt | 44.6% |

## Gate behavior

learned ens3は平均59.6%をapply、40.4%をholdし、fold別apply率は36.4%〜91.8%だった。平均hold
run長は8.19 decisions、fold内max hold run長の平均は27.65 decisionsであり、gateは実質always
apply/holdのtrivial solutionではない。fold平均でtarget-weight turnover 122.34、即時取引cost
4,544 JPYをholdにより回避した。mean turnoverはforced apply比32.1%、direct比54.0%低下した。

一方、learnedのmean proposed gross exposure 1.763に対してmean applied gross exposureは1.785、
proposalと保持allocationのmean gross exposure driftは0.279だった。gateは単純なzero blendではなく
既存allocationを保持しているが、そのactive behaviorはnet改善へつながらなかった。

`hold`はliteral no-tradeまたはabsolute JPY exposure固定ではない。直前のeffective target allocationを
再送するため、市場価格変動後はそのtargetへ戻す小さなrebalanceとcostが発生し得る。本結果の
avoided turnover/costは、この既存allocation意味論に対するproposal適用とのcounterfactualである。

## Acceptance criteria

1. **Net improvement: fail** — learned − directは−7.39ポイントで、両eraとも負。
2. **Gate attribution: fail** — learned − same-model forced applyは−1.20ポイント。
3. **Uncertainty: unproven** — 両primary comparisonのIID / moving-block区間がゼロを跨ぎ、
   Issue #4の事前基準どおりeffectをゼロと区別できない。
4. **No degenerate risk shrinkage: descriptive pass** — mean grossはbaselineの82.1%でnear-flatではないが、
   gross alphaが大きく低下したため採用根拠にならない。
5. **Behaviorally active gate: pass** — apply 59.6% / hold 40.4%で非自明だが、net evidenceがない。
6. **Isolation: pass** — data、features、PPO+MLP、budget、seeds、device、model selection、evaluatorを
   固定し、learned / forcedは同一model identityで評価した。

必要条件1〜3を満たさないため、総合判定は**not adopted / unproven**である。point estimateは両比較・
両eraで負だが、bootstrap uncertaintyがゼロを跨ぐため`rejected`とは分類しない。

## 成果物と再現

- gated fold configs: `configs/wf_r20_gate_full/`
- campaign: `configs/research/issue4_apply_hold_gate.yaml`
- report: `docs/research/results/issue4/report/`
- canonical observations: `docs/research/results/issue4/report/observations.csv`
- complete provenance and member identities: `docs/research/results/issue4/report/provenance.json`
- machine-readable result: `docs/research/results/issue4/report/report.json`
- human-readable generated summary: `docs/research/results/issue4/report/report.md`

再集計コマンド:

```console
uv run forex-report \
  --campaign configs/research/issue4_apply_hold_gate.yaml \
  --output-dir docs/research/results/issue4/report
```

run artifactは`runs/issue4_gate/`、learned/forced ensembleは
`runs/issue4_gate_ensembles/`、現行evaluatorで再生成したcontrolは
`runs/issue4_control_ensembles/`にある。campaignは各timestamped artifactを明示し、暗黙のlatest
選択を行わない。
