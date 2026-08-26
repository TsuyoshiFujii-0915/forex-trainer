# 08. 疎なrank allocation行動空間（Issue #2、Round 18〜19）

期間: 2026-08-26〜 / 対象: `longf` PPO + MLP

## 仮説

直接ウェイト行動は多数ペアへ露出を分散しやすい。方策にペア別scoreだけを出力させ、上位と
下位だけへ固定grossを配ると、特定シグナルを埋め込まずに疎性を課せるため、コスト規律と
時代間の汎化が改善する可能性がある。

## 事前登録プロトコル

- direct基準: `configs/wf_r16_long_full/` のvalidation-best ens3
- 共通学習: PPO + MLP、`n_steps=batch_size=1024`、300k decisions、seed 42/43/44
- 共通データ: 9 JPYペア日足、signed financing、同一特徴量・fold境界
- Round 18 screen: 2010/2013/2016/2019/2022/2025の6 foldsで、gross 2.0を固定し
  `top_k=1` と `top_k=2` を比較
- Round 19 confirm: Round 18の平均年率net returnが高い方を、2009〜2025の17 foldsへ拡張
- tie: 通貨ペア設定順を二次キーにしたstable sort
- ensemble: 3モデルのscoreを平均してからrank allocationを一度だけ適用

Round 18の選択基準が同値の場合は、より疎な`top_k=1`を採る。報告指標は年率net/gross
return、Sharpe、max drawdown、target-weight turnover、cost ratio、mean gross leverage、
勝ちfold数、2009〜2018 / 2019〜2025のera別平均とする。各foldの年率値はIssue #1と同じ
実効評価期間ベースで計算する。

## 結果

実験実行後に記録する。
