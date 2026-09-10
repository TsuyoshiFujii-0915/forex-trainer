# Title

0034. turnover診断ではovernight保有費用を売買costから分離する

## Status

accepted

## Context

ADR-0033の実装テストで、#16の実際のsealed契約はcarry_mode=signedであっても
**overnight_rate=0.00002/day**を含むと確認した。envはsigned financingとovernightを
別々に計上する。#19のcostもspread+commission+overnightであり、overnight=0という
ADR-0033の入力前提は誤りだった。最終集計前のsource照合で失敗として検出した。

## Decision

ADR-0033を置換し、その明示入力、exposure復元、所属状態、持続性、打切り、fold統計、
探索的sliceと分類維持の定義を継承する。ただしovernight=0の制限と
「spread+commissionだけで元total costを照合する」という記述は適用しない。

実売買costはspread+commission。overnightは各decisionの
`abs(target_weight) × equity_before × overnight_rate × elapsed_days`でpairへ配賦する。
これはenvがmark-to-market時に**価格適用前のtarget exposure**に課す保有費用であり、
売買量の原因別説明へ混同しない。環境のPortfolioAccount.mark_to_marketでも費用を照合する。
#19 cost = trading cost + overnightを照合し、signed carryは別の収益寄与として維持する。
原因別CSV/JSONにはtotal cost、trading cost、overnightを別列・別比率で保存する。

rank churnは#16の平均同順位rank版を元指標の再現に使い、ADR-0033のordinal版も
別列に保存する。同点順位のためのstable pair orderを変更しない。

## Consequences

- total costのうち低回転化と直接対応しない保有費用を定量化できる。
- 入力前提誤りを隠さず、accepted ADRの本文を改変せず訂正過程を残す。
- policy、評価会計、既存成果物の定義は一切変更しない。
