# 0021. regime変数の採否はfold単位の方向安定性で判断する

## Status

accepted

## Context

日次decisionを独立標本として扱うと、同じ市場履歴内の自己相関とregime持続を無視して不確実性を
過小評価する。また、2009年や2015年のような単一tail yearを説明する変数は、別時代で再現しなければ
agent featureへ昇格する根拠にならない。ADR-0012は研究判断の標本単位をfoldと定めている。

## Decision

各candidate regime値はdecision時点までの観測窓だけから計算し、その直後の1 decisionにおける
agent/ruleのnet return、gross return、cost ratio、drawdown changeへ対応させる。regime bucketは
各fold内で作り、bucket平均からhigh-minus-low効果を1 fold 1値へ縮約する。方向関係もfold内の
rank associationを1 fold 1値として計算する。

集約は評価年順のfoldを標本とし、IID fold bootstrapとcircular moving-block bootstrapの95%区間を
併記する。変数をstableと呼ぶには、campaignで事前宣言した最低fold方向一致率を満たし、全期間の
両bootstrap区間がゼロを跨がず、宣言した全eraで同じ方向を持つことを必須とする。constant、欠損、
非finiteなcandidateやresponse、fold/timestamp不一致は早期エラーとする。

legacy baselineとの比較は異なるprovenance/evaluator世代を跨ぐため、aggregate return、drawdown、
era、tail foldの記述的sanity comparisonに限定し、paired evidenceやbootstrap差として扱わない。

## Consequences

- 一つのtail yearだけを説明する変数はfeature候補として採用されない
- 日次bar数を独立な市場標本数と誤認しない不確実性評価になる
- 厳しい採否基準により候補が一つも残らない場合も、明示的で再現可能な研究結論になる
- bucket境界はfoldごとに異なるため、水準比較ではなく各fold内の条件付き方向を検証する
