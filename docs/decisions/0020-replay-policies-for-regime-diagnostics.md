# 0020. regime診断は封印済み方策を同一評価器で再生する

## Status

accepted

## Context

Issue #5では、次期return、cost、drawdownを決定時点の市場regimeと方策状態で条件付ける必要が
ある。既存のversion 2評価成果物は集約metricsとequity curveを保存するが、決定時点の観測、
target weight、turnoverをprovenance付きtraceとして保存していない。旧成果物へ現在値を追記する
backfillや、封印されていないCSVから不足項目を推測すると、ADR-0013とADR-0018に反する。

## Decision

regime診断は、現行training provenanceで再学習しversion 2の`forex-ensemble-eval`で封印した
action-mean ensembleだけを入力とする。`forex-report`によるartifact照合を先に通し、同じmember
model、resolved eval env、device、評価実装でensemble方策を決定論的に再生する。再生した集約
metricsが封印済みmetricsと一致しなければ診断を失敗させる。

比較するrule benchmarkは、診断campaignにfeature、top-k、weight magnitudeを必須宣言し、同じ
resolved eval envを独立に一度歩く。agentとruleのtraceはfoldと決定timestampの完全一致を要求し、
intersectionによる暗黙の欠損除外を行わない。診断成果物にはmember seeds、rule定義、data identity、
training/evaluation/diagnostic Gitと依存version、source artifact hash、生成trace hashを記録する。

## Consequences

- version 2評価契約と既存generic reportの振る舞いを変更せず、必要な決定時点traceを再現できる
- model、env、data、metricsの改変は診断前または再生照合時に検出される
- 診断には全member modelの再推論コストがかかるが、再学習は不要である
- ruleはtraining artifactを装わず、historical/current RL provenanceと混同されない
