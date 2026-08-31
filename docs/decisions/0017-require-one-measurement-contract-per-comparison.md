# 0017. paired comparisonでは評価実装を一致必須とする

## Status

accepted

## Context

training実装の差は比較対象となるtreatmentだが、評価実装の差は測定尺度そのものを変え得る。
実際にSharpe annualizationの修正履歴があり、異なるevaluator commitまたは依存versionで生成した
Sharpeやmax drawdownを直接差し引くと、policy差とmetric定義差を分離できない。差分をprovenanceへ
記録するだけでは標準研究証拠として不十分である。

## Decision

`forex-report`のpaired comparisonは、baselineとcandidateでevaluation-timeの両repository Git SHAと
主要依存versionが完全一致することを必須とする。不一致時は比較を生成せず、同一evaluator環境で
再評価を要求する。training-time Gitと依存versionの差はtreatment provenanceとして許可し、
比較結果へ明示する。

## Consequences

- Sharpe、drawdownを含むpaired metricが同一measurement contractから生成される
- evaluator更新をまたぐ比較には、両群の再評価が必要になる
- training implementationの比較可能性は維持される
