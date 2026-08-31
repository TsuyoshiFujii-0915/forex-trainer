# 09. tail lossのregime診断（Issue #5）

期間: 2026-08-31 / 対象: direct `longf ens3`、評価年2009〜2025

## 結論

agent featureへ昇格できるregime変数はなかった。realized volatility、pair間相関、return
dispersion、momentum dispersion、carry dispersion、trend/reversal proxyのいずれも、agentの
次期net returnに対するhigh-minus-low効果がfold bootstrapとmoving-block bootstrapの両方で
ゼロを跨がず、かつ両era・70%以上のfoldで同方向、という事前基準を満たさなかった。

rule benchmarkではmomentum dispersionだけが安定した正の関係を示したが、agentでは同じ関係が
不安定だった。これは「cross-sectional momentumのばらつきが大きい局面ほどreversal ruleが
機能する」というmarket/rule構造であり、direct agentへ同変数を追加すればtail lossを抑えられる
証拠ではない。したがって本issueからfeature追加issueは作らない。

## current-contract baselineの再構築

ADR-0018に従い、legacy source metadataへ現在値をbackfillせず、
`configs/wf_r16_long_full/`の17 folds × seeds 42/43/44を全51 run再学習した。その後、全runを
現行`forex-eval`で評価し、foldごとに同じ3 seedsを使うversion 2 action-mean ens3を生成した。

- data: `data/jpy_9pairs_1d_2003_carry.parquet`
- data SHA-256: `723db7c935dcc27c147007728358d098243eae96998eae146db4c7df02bd081e`
- member seeds: 42 / 43 / 44
- requested / resolved training device: CPU / CPU
- training・evaluation trainer SHA: `49c04492aa4a0902ea97f81edc5d9217b0f2bf75`
- diagnostic trainer SHA: `e43f11a3c95f00ed37282fd9d2b092a829c681cf`
- forex-env SHA: `6024b91c0f3592611849bc231922ab60e6090aed`
- model selection: validation-best
- campaign range policy: expanding、declared start `2003-06-01`
- effective eval folds: 2009〜2025、各年約193〜199 decisions
- campaign trial count: 9

`forex-report`のcurrent-provenance集約は次のとおり。95%区間がゼロを跨ぐため、全体の正の平均を
統計的に確立した期待値とは扱わない。

| Scope | Net/年 | Gross/年 | Sharpe | Mean MDD | 勝ちfold |
|---|---:|---:|---:|---:|---:|
| 全17 folds | +10.31% | +13.06% | +0.222 | 16.52% | 8/17 |
| 2009〜2018 | +17.24% | +20.43% | +0.493 | 18.76% | 6/10 |
| 2019〜2025 | +0.41% | +2.55% | −0.167 | 13.32% | 2/7 |

全体net returnの95%区間はfold bootstrapで **−3.62%〜+25.63%**、circular moving-blockで
**−4.23%〜+27.28%**。worst MDDは36.95%だった。

## legacy sanity comparison

historical inputはIssue #1のstudy-specific `validation_best` evidenceを使い、generic reportへ
混入させていない。currentとlegacyはaggregate、両era、全17 foldのnet/gross return・MDDで
数値が一致し、material changeはなかった。したがって旧研究のtail observationsは今回も再現した。

| Fold | Net/年 | MDD | 判定 |
|---:|---:|---:|---|
| 2009 | −23.20% | 33.26% | currentでもtail loss |
| 2015 | −36.57% | 36.95% | currentでもworst tail loss |

この一致はlegacy provenanceを現在値で補完した結果ではない。独立に再学習・評価したcurrent成果物と
historical集計を記述的に照合した結果であり、世代を跨ぐpaired bootstrapは行っていない。

## 診断方法

current ens3を封印済みmember model・同じresolved eval env・CPUで決定論的に再生し、再生metricsが
各`metrics.json`と一致することを全17 foldsで確認した。比較ruleは同じenvを独立に歩く
`mom24` cross-sectional reversal（top 2 / bottom 2、weight 0.8、gross 3.2）。agentとruleはfoldと
decision timestampの完全一致を要求した。

各decision直前の32-bar observationだけから候補値を計算し、その後のnet/gross return、cost、
20-decision forward maximum drawdownへ対応させた。各fold内で候補のdistinct valueを順序付き3
bucketsへ分け、high-minus-lowを1 fold 1観測へ縮約した。同値候補はbucket境界で分割しないため、
tiesが多いpolicy-state候補ではbucketの観測数は均等にならない。不確実性は日次barでなくfoldを
標本とする10,000回のIID fold bootstrapと3-fold circular moving-block bootstrapで評価した。

## agentのmarket-regime結果

表は高regime bucket − 低regime bucketの次期net log return。`Direction rate`はbucket効果と
rank associationの両方が全体方向に一致したfold比率である。

| Candidate | Mean H−L | Fold 95% CI | Block 95% CI | Direction rate | Era方向 | Stable |
|---|---:|---:|---:|---:|---|---|
| realized market volatility | +0.012% | [−0.079%, +0.107%] | [−0.063%, +0.094%] | 24% | + / + | no |
| mean cross-pair correlation | +0.019% | [−0.074%, +0.108%] | [−0.065%, +0.122%] | 35% | + / − | no |
| cross-sectional return dispersion | −0.060% | [−0.184%, +0.054%] | [−0.228%, +0.073%] | 59% | − / + | no |
| momentum dispersion | +0.092% | [−0.001%, +0.199%] | [+0.003%, +0.174%] | 41% | + / + | no |
| carry dispersion | −0.019% | [−0.097%, +0.063%] | [−0.097%, +0.056%] | 53% | − / + | no |
| trend/reversal proxy | −0.021% | [−0.108%, +0.075%] | [−0.104%, +0.057%] | 47% | − / + | no |

momentum dispersionはmoving-block区間だけが正だったが、fold区間がゼロを跨ぎ、方向一致も41%に
留まった。単一の見栄えの良い区間を選んで採用しない。

## rule benchmarkとの切り分け

ruleは全17 foldsでnet +4.98%/年、gross +8.81%/年、平均MDD 9.84%、9/17 folds勝ちだった。
2009は+8.63%、2015は+6.98%であり、direct agentの両tail lossを共有しなかった。一方でruleも
2009〜2018は+1.18%/年、2019〜2025は+10.42%/年とera差が大きい。

ruleのmomentum dispersion効果はH−L **+0.135%/decision**、fold 95% CI
[+0.071%, +0.212%]、moving-block 95% CI [+0.074%, +0.203%]、方向一致76%、両era正でstableだった。
他のmarket候補はruleでも事前基準を満たさなかった。よってmomentum dispersionはreversal ruleの
機会集合を記述するが、agent固有のfailure driverではない。

本studyは6 market candidatesと2 policy-state candidatesを4 responses・2 policiesへ広くscreenして
おり、多重screening補正や独立holdoutを持たない。したがってruleのmomentum-dispersionを含むpositive
findingは探索的証拠であり、独立期間で再確認するまで確立した効果や新featureの採用根拠としない。

agentのgross exposureとturnoverは次期costとは安定して正に関係した。これは会計上期待される
診断である。一方、次期net returnとの関係はgross exposure・turnoverともbootstrap区間がゼロを
跨ぎ、era/fold安定性を満たさなかった。単純なexposure/turnover gatingをtail対策として正当化する
結果ではない。

Issue #2のrank allocatorはdrawdownを半減したがgross/net alphaを失ったため、今回もsecondary contrast
に留める。tail年を避けるためにdirect baselineをrankへ置換する判断は変更しない。

## 成果物と再現

- campaign: `configs/research/issue5_longf_baseline.yaml`
- study: `configs/research/issue5_regime_study.yaml`
- generic report: `docs/research/results/issue5/baseline_report/`
- full diagnostic: `docs/research/results/issue5/report.json`
- traces/tables: `step_trace.csv`、`bucket_metrics.csv`、`fold_effects.csv`、`sanity_comparison.csv`
- plot: `regime_effects.svg`
- complete source/artifact hashes: `provenance.json`

再実行コマンド:

```console
forex-regime-study \
  --study configs/research/issue5_regime_study.yaml \
  --output-dir docs/research/results/issue5
```
