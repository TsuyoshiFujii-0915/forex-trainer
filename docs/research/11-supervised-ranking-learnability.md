# 教師ありcross-sectional ranking学習可能性診断

## 目的

Issue #15では、逐次信用割当、portfolio sizing、transaction cost最適化を分離し、既存`longf`
market情報だけから次decisionのpair相対return順序をout-of-sampleで学習できるかを検証する。
これは新しい取引戦略の探索ではなく、PPOの不安定性が問題設定に由来するか、feature/labelの
mapping自体に由来するかを切り分けるlearnability diagnosticである。

## 事前登録プロトコル

- canonical 9 JPY pairs、日足、expanding 17 folds（evaluation year 2009--2025）
- train開始は2003-06-01、validationは各evaluation直前6か月
- 1 rowは`(decision timestamp, pair)`。入力はpairごとの32-lag×8 `longf` market windowを
  oldest-to-currentで展開した256列
- account stateとpair identityは入力しない
- featureはdecision時刻`t`まで、labelは`t -> t+1`のpair log returnから同時刻の9-pair平均を
  引いたrelative return
- 各rangeは既存envと同じ32-row feature warmupと32-row observation windowを独立適用する。
  したがって2009 evaluationの最初のdecisionは2009-04-03であり、年初から暗黙に行を補わない
- train-only standardization後のpooled linear ridge。alphaは`0.0, 0.1, 1.0, 10.0`固定
- validation mean cross-sectional Spearman rank ICでalphaを選び、同値は強い正則化を選択
- evaluationはalpha、feature、target、model familyの選択に使用しない
- frozen benchmarkはraw `-mom24` reversal scoreとsealed current-provenance `longf ens3` action mean
- primary tail diagnosticはpredicted top-2のnext relative return平均からbottom-2平均を引く
- 全pair targetが同値のdecisionはrank ICだけから除外し、そのobservation fractionを報告する。
  tail spreadなどreturn-level診断には含め、未定義ICを0へ置換しない
- decisionをfold内で集約後、foldをsampling unitとして10,000 IID bootstrapと3-fold circular
  moving-block bootstrapを行う

時刻・label・input契約はADR-0026、model selectionはADR-0027、fold evidenceと分類は
ADR-0028に固定した。

## 実行方法と成果物

```bash
uv run forex-supervised-ranking \
  --study configs/research/issue15_supervised_ranking.yaml \
  --output-dir docs/research/results/issue15
```

出力は`predictions.csv`、`coefficients.csv`、`models.json`、`fold_metrics.csv`、`report.json`、
`report.md`、`provenance.json`である。source config/data/ensemble/model、生成物、Git、NumPyを含む
依存versionをhashで封印し、既存出力へ上書きしない。

## 結果

判定は **established learnable** となった。

| score | mean rank IC | median rank IC | positive IC fraction | top2-bottom2 next-relative-return spread |
|---|---:|---:|---:|---:|
| supervised ridge | 0.01092 | 0.01667 | 0.5082 | 0.0001199 |
| canonical `-mom24` | 0.01820 | 0.03333 | 0.5190 | 0.0002053 |
| sealed PPO `longf ens3` | -0.00338 | 0.00000 | 0.4881 | -0.0000409 |

supervised tail spreadの95%区間はIID fold bootstrapで
`[0.0000144, 0.0002327]`、3-fold circular moving-block bootstrapで
`[0.00000556, 0.0002373]`となり、いずれもpositive directionを支持した。era別のpoint estimateも
2009--2018が`0.0001278`、2019--2025が`0.0001087`でともに正だった。各evaluation foldを1つずつ
除外した17通りのtail-spread平均もすべて正で、最小は`0.00008596`だったため、単一year支配ではない。

economic coherenceは、supervised scoreとfrozen `-mom24` scoreのdecision-level rank associationを
fold内平均して判定した。17 foldすべてで正（range `0.0851`--`0.3056`）、fold平均`0.1978`、era別も
`0.2017`と`0.1921`で正だった。pair identityを入力しておらず、score dispersionも全foldで非zeroで
あるため、memorizationまたはconstant scoreによる成立ではない。

alphaはvalidationだけで選択され、`10.0`が7 fold、`0.0`が5 fold、`0.1`が3 fold、`1.0`が2 fold
だった。単一feature係数は強く相関するraw/cross-sectional momentum lag間の多重共線性に敏感で、
たとえばcurrent `mom24` standardized coefficientのfold平均は正だった。このためcoherence判定には
単一係数を使わず、実際のscore orderingをcanonical reversalと比較した。全256列のfold別係数と
era/aggregateのsign・magnitudeは`coefficients.csv`と`coefficient_summary.csv`に保存した。

## 解釈と限界

既存feature setには、次decisionのrelative-return orderingを低capacityの共通mappingとして
out-of-sampleで回収できる情報がある。canonical reversalより弱いものの、sealed PPO orderingが
negativeだった同じdecision集合でridgeはpositive orderingを得たため、PPOの不安定性には少なくとも
formulationまたはcredit assignmentが寄与している。

一方、mean ICは小さく、後半eraのmean ICは`0.00397`、tail ordering accuracyは`0.5077`に留まる。
個別foldではtail spreadが負の年もあり、alpha選択もfold間で安定していない。これはpredictive
learnabilityの証拠であって、transaction cost控除後のtradability、portfolio sizing、risk controlを
示さない。今回の結論を使ってfeature、horizon、model familyをevaluation上で追加探索してはならない。

Issue #15のdecision ruleに従い、次は凍結したsupervised scoreから固定portfolio mapへのtranslationを
評価するblocked issueへ進める。ただし本studyのscore/model/alpha protocolを固定し、portfolio側で
このevaluation結果を使ったHPOを行わない。
