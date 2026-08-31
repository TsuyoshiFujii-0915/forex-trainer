# 0024. regime診断のdrawdown応答は固定期間forward maximum drawdownとする

## Status

accepted

## Context

ADR-0021はdrawdown responseを直後1 decisionのdrawdown changeと記述したが、Issue #5のtail risk診断
では一日変化より、そのregimeから固定期間内に生じる最大下落の方が目的に合う。実装はstudyで必須
宣言した20 decisionsを使ってcurrent equityから将来最小equityまでの下落率を測っており、旧field名
`next_drawdown_change`は意味を正しく表していなかった。

## Decision

drawdown responseを`forward_max_drawdown`とし、decision時点のequityに対する、宣言した完全な
forward horizon内の最小equityまでの最大下落率を記録する。完全なhorizonを持たない末尾decisionは
診断traceから除外し、短い期間へfallbackしない。この決定はADR-0021のdrawdown responseに関する
決定を置き換える。

## Consequences

- field名、研究ノート、実装が同じtail-risk量を表す
- 最後の`forward_drawdown_steps - 1` decisionsは条件分析の標本にならない
- horizonの異なるstudyを比較する場合はcampaign定義の差として扱う必要がある
