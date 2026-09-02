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

実行結果をここへ記録する。
