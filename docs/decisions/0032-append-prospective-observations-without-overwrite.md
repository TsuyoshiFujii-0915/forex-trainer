# Title

0032. 前向き観測をdecision時点と事後結果に分けて追記保存する

## Status

accepted

## Context

後日再取得したmarket dataや、結果を知った後に作成したactionだけでは、当時利用可能だった
情報と提案の証拠にならない。障害行の削除、cost欠損の0補完、事後訂正による上書きも監査を壊す。

## Decision

[記録schemaと運用手順](../research/protocols/issue20/forward-operations.md)を最小交換契約とする。
decisionには利用可能時刻、入力snapshotとmodel/config/data identity、score、提案・実効action、
想定costを記録する。観測costと損益は別のoutcome、欠損・障害はincident、訂正は元eventを参照する
correctionとして追記する。各eventはprotocol hash、連番、前eventのraw bytes SHA-256を持つ。
結果閲覧と契約違反も記録し、開発利用された確認期間を区別する。

保存側は上書き拒否・連鎖照合・外部保管先への定期head固定を開始前に用意する。
JSON Schemaは1行の構造を検証する契約であり、時系列比較、hashの実体照合、append-only保存を
実装したものとは扱わない。本Issueではschema・例・手順を納品し、継続収集の実装と実行は別作業とする。

## Consequences

- 後から到着した情報や訂正を、当時のdecisionへ混入させずに追跡できる。
- hash chain単体で改ざん耐性が保証されるとは主張しない。書込み権限と独立したhead保管が必要となる。
- 記録不能時は停止し、同一期間を欠損除外して成功した確認として報告しない。
