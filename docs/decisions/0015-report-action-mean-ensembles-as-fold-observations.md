# 0015. action-mean ensembleをfold単位の観測として集約する

## Status

accepted

## Context

各seedのreturn平均は、各decisionでactionを平均して一度だけ環境を歩くensemble方策のreturnと
一致しない。seed runだけを入力するgeneric reportでは、既定基準である`longf ens3`を正しく表現
できず、別の統計対象を代用する危険がある。旧`ensemble.json`には、後からmember modelや評価環境を
厳密に照合するための情報も不足していた。

## Decision

`forex-ensemble-eval`はversion 2の`ensemble.json`を出力する。policy、model-selection、
decision interval、各memberのrun、experiment、seed、model pathとhash、config snapshot hash、
meta hash、および評価時provenanceとmetrics/eval env hashを必須とする。`forex-report`のcampaignは
`result_kind: ensemble`を明示し、manifestと全member artifactを照合したうえで、ensemble自身の
metricsを1 foldにつき1観測として集約する。比較とbootstrapの標本単位はfoldとする。

## Consequences

- action-mean方策の実測成績をseed平均で代用せずに比較できる
- member modelまたは評価成果物の変更をreport生成時に検出できる
- 旧`longf ens3`成果物は暗黙変換せず、version 2契約で再生成してからgeneric reportへ登録する
