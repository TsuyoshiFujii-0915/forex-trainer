# Title

0049. 封印済み短期357口座を期間logと完全な状態台帳で比較する

## Status

accepted

## Context

Issue #43 は ADR-0046/0047/0048 の17区間・2予測・数量会計を接続する。
旧対照の年率統計や成功foldだけの集計は、新契約の期間logの採否に使えない。
途中の入力・実装障害と戦略terminalを区別し、未実行を含む全357セルを残す必要がある。

## Decision

専用CLI `forex-common-direction` で preflight / seal / run / verify を提供する。
全7方策×17区間×F0/F1/F2を実行前に登録し、#41予測と旧対照sourceを同一runtimeで
#42数量会計へ接続する。各口座は実行直前に固定出力先の台帳へ予約する。通常枠の再取得・
自動retryは認めない。実装commitと実行結果commitを分離する。

比較の一次量は17区間等重みの期間net log。既存fold bootstrapを再利用し、登録済みの
risk・stress・era・LOO・寄与集中・tie条件を固定する。必要な17区間が未完なら比較は保留、
戦略terminalの候補は非採用。候補間の選択は全候補の測定完了まで保留する。
記録可能な例外はincident、全セルreport、manifestを保存して再送する。verifyは再推論せず
元市場・予測identity・会計・統計・台帳を照合し、未完了も検証可能にする。

## Consequences

- 口座経路の完了、開発候補、配分の追加価値、独立収益性を別々に表示する。
- #46は選定hの予測基準と逐次仮説を別途確認する。#49には適格な単純候補を渡せる。
  中期gateや実quote未準備は短期の測定・判断を妨げない。RL学習は開始しない。
- hash/会計違反を無視せず、未実行を成功と呼ばない。旧measurementとpaired比較しない。
- 同じcapはrisk一致の証明ではなく、部分年・既知開発期間・未補正CIの限界を保持する。
