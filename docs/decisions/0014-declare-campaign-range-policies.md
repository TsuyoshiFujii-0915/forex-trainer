# 0014. 学習期間の扱いをcampaignの明示的なtreatmentとする

## Status

accepted

## Context

foldごとの絶対日付をprotocol identityから除くだけでは、rolling 2年、4年、8年とexpandingを
区別できない。逆に絶対日付をそのままhashすると、同じrolling方針でもfoldが違うだけで別protocolに
なる。学習履歴量は研究上のtreatmentであり、暗黙推定や欠損時のfallbackでは比較の意味が変わる。

## Decision

各campaign configurationは`range_policy`を必須とする。rollingは`train_years`を2、4、8のいずれかで
宣言し、expandingは固定の`train_start`をISO日付で宣言する。reportは各runについて、評価を1月1日
から翌年1月1日、validationを直前の7月1日から評価開始日、training終了をvalidation開始日として
厳密に検証する。rollingの開始日は宣言年数から計算し、expandingの開始日は宣言値と一致させる。
protocol identityには絶対日付ではなく正規化したrange policyを含め、provenanceには両方を保存する。

## Consequences

- foldをまたいだ同一方針と、履歴量が異なるtreatmentを機械的に区別できる
- 不正な混在やcampaign宣言との不一致は集計前に失敗する
- 現在の研究標準以外の期間形状を使う場合は、契約と判断根拠を明示的に更新する必要がある
