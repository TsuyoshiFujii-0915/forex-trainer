# Title

0048. 共通方向配分に数量保持を表現する opt-in 会計を採用する

## Status

accepted

## Context

Issue #42 の literal hold は、前回 target weight の再送と異なる。既存の target API・
報酬を変更すると旧成果物の意味が変わる。ADR-0045/0046/0047 の予測・予算を消費し、
ADR-0037 の履歴と測定区間の分離を新会計にも保持する必要がある。

## Decision

`issue40-common-quantity-same-close-v1` を専用 adapter で実装する。quantity は
既存逆数価格座標で `exposure_jpy / price` と定義する研究上の数量であり、実商品の base
数量へ読み替えない。既存 `PortfolioAccount` の実 notional rebalance、price PnL、開始
notional の日次 markup、終了 marked notional の signed financing を再利用する。
費用・価格変動の二重計上を避け、旧 env API・reward・旧 measurement は変更しない。

各 decision の推論前に drift 超過数量を比例縮小する。提案後も売買費用控除後の equity に
対する gross 5・pair 1 を検査し、必要な最小比例縮小を明示する。縮小率は自身の売買費用を
含む区分一次制約から解き、不能・非有限・残差超過は例外とする。最終 mark は評価のみで、
仮想決済や追加 decision を作らない。margin は旧契約と同じ初期 equity の 0.2、非正 equity
も独立の strategy terminal として保存する。

配分予測の financing/markup は現在 marked notional と当日既知 rate を一定とする近似。
実現 financing は旧会計どおり終了 marked notional を使う。この差を trace に残し、未来価格を
utility に入れない。h=5 は登録 calendar の UTC 経過時間で毎営業日再計画し、末尾も短縮しない。

## Consequences

- 新旧方策は独立した同一会計口座へ接続できるが、共通 cap を risk-matched と呼ばない。
- 同 close と逆数座標は歴史研究仮定であり、実執行は別契約のまま。
- 通常合成口座は最大16、実行前予約と clean runtime seal を必須にし、失敗も消費する。
  保存済み trace の検証は新規口座実行を伴わない。retry は共有枠の別登録が必要。
- #43 は本番357口座の独立した seal が必要。本Issueに市場口座実行CLIは追加しない。
