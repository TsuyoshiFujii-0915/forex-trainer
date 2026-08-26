# 0010. 学習スコアを疎なクロスセクション配分へ変換する

## Status

accepted

## Context

直接ウェイト行動では、方策が多数の通貨ペアへ小さなポジションを分散しやすく、レバレッジと
取引コストが増える。これまでの研究ではクロスセクション順位の端に成績が集中しているが、
`mom24` など特定のルールを行動変換へ埋め込むと、方策がシグナルを学習したかを検証できない。
また、既存の residual action (ADR-0009) は明示したルールをベースにするため、今回の
「ポートフォリオ形状だけを制約する」目的とは異なる。

## Decision

`run.rank_allocation` を必須設定とし、`"none"` または次のマッピングを受け付ける。

- `top_k`: long と short の各側で選ぶペア数
- `gross_exposure`: 全ペアの絶対ウェイト合計

有効時、方策の行動空間は通貨ペア設定順の score `(N, 1)`、範囲 `[-1, 1]` とする。
score を範囲内へclipしてstable sortし、上位 `top_k` をlong、下位 `top_k` をshort、
残りをflatにする。選択された各ペアの絶対ウェイトは
`gross_exposure / (2 * top_k)` とし、long/shortへgrossを半分ずつ配る。

同点時の二次キーは `env.environment.currency_pairs` の設定順とする。したがって全scoreが
同じ場合、先頭側の `top_k` がshort、末尾側の `top_k` がlongになる。wrapperは観測や特徴量を
一切参照しない。アンサンブルではADR-0007と同様にagent-facing actionを平均するため、
rank allocationではscore平均後に一度だけ配分へ変換する。

固定grossを暗黙に崩さないため、設定読込時に `2 * top_k <= N`、各ウェイト `<= 1`、
`gross_exposure <= max_leverage`、`allow_action_leverage: false` を検証する。residual actionとの
同時有効化は、疎性とgrossを壊し変換順にも意味がないため拒否する。

## Consequences

- 方策は任意の観測からscoreを学習でき、特定の反転・モメンタムシグナルはハードコードされない
- 選択ペア数、market neutrality、総grossが全決定で固定され、直接ウェイトとの比較条件が明確になる
- 同点を含む変換が通貨ペア順に対して完全に再現可能になる
- learned exposure sizing と trade/no-trade gating は表現できず、別の実験課題として残る
- 既存configにはdirect-weight動作を明示する `rank_allocation: none` の追加が必要になる
