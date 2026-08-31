# 0019. ensemble比較ではmember seed集合を一致させる

## Status

accepted

## Context

action-mean ensembleはmemberのseedによって学習済みpolicyと評価結果が変わる。foldだけを対応付けて、
baselineをseeds 42/43/44、candidateをseeds 45/46/47として比較すると、method差にtraining
stochasticityのdraw差が交絡する。standard seed runではfold×seedを完全一致させており、ensemble
だけ異なる条件を許可する根拠はない。

## Decision

`forex-report`で`result_kind: ensemble`同士をpaired comparisonする場合、baselineとcandidateの
正規化済み`member_seeds`を完全一致必須とする。member数だけでなく、重複を拒否して昇順化した
seed tuple全体を比較する。不一致時はfold deltaやbootstrapを生成しない。

## Consequences

- ensemble method差とmember seed draw差を分離した対応比較になる
- candidateごとにbaselineと同じseed集合のsource runが必要になる
- 異なるseed集合のensembleは個別集計できるが、相互のpaired comparisonには使用できない
