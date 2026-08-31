# 0023. 同値のregime観測は同じbucketに保持する

## Status

accepted

## Context

fold内の候補値を単純な観測数分位で分割すると、同じ値がlow/high双方へ入り得る。特にturnoverが
ゼロの日のような同値が多いpolicy-state候補では、stable sortの二次的な時刻順と将来returnの関係を
regime効果として誤認する危険がある。一方、同値を保持するとbucketの観測数は均等にならない。

## Decision

各foldでcandidateのdistinct valueを昇順に並べ、宣言bucket数へ分割する。同じcandidate valueを
複数bucketへ分割しない。したがってbucketはequal-count quantileではなく、ordered
distinct-value bucketと呼ぶ。distinct value数がbucket数未満なら、そのfoldを暗黙に減らさず診断を
失敗させる。

## Consequences

- 同値候補の時刻順がhigh-minus-low効果へ混入しない
- tiesが多い候補ではbucketごとの観測数が大きく異なり得る
- bucket表の観測数と値域を確認してpolicy-state結果を解釈する必要がある
