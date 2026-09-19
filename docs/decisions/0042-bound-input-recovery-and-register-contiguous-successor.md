# Title

0042. 入力回復を有限で終了し、連続区間の後継研究契約を登録する

## Status

accepted

## Context

Issue #39の有限調査では元pair別response、join前系列、取得ログを回収できなかった。
Yahooと代替Dukascopyの各1requestは429。歴史的休場と配信欠損を分離できず、
現在の取得で旧vintageを復元することもできない。旧#29/#30の完了条件は保持する。

## Decision

出口Bとして `issue39-contiguous-snapshot-v1` を登録する。取得可能でhashを照合できる
既存Yahoo raw/clean/carry snapshotを唯一の価格sourceに選ぶ。新providerの採用ではなく、
同じ値から有限の評価区間と派生cacheを再生成する新しいdata/calendar版である。
新しい市場データを取得したとは称さない。損益による選択・補間・forward-fillは行わない。

London全平日、休日除外0、label=仮定close、available_at=nullを保持する。
全不足日をunknownとして残し、各fold内の連続した期待bar区間を列挙する。
欠損後は63本の連続履歴を再度必要とし、2本以上の測定barを持つ最大の区間を各foldで
一つだけ選ぶ。同数なら早い区間を選ぶ。この規則を推論前に固定する。
各区間はflatから独立開始、最終barはmarkのみ。区間を連結した口座・年次CAGRを作らない。
週末とDSTをまたぐ観測間の資金調達経過時間は既存envの実UTC経過時間を保持する。
欠損をまたぐ持越しや仮想清算は導入しない。

新契約のmanifestに旧source hash、派生cache hash、calendar hash、元train/val/eval範囲、
全候補区間、除外範囲、欠損×pair、変換lineage、実行codeを固定する。
元modelの訓練lineageは変更しない。特徴量・pair順・逆数価格座標・費用・報酬を保持する。
既存full-period評価器で新範囲をpreflightできるが、旧#29/#30の153/85セルの完了とは別に数える。
#30のtrain_constantは旧train replayの欠損横断仮定を継承するため、その不足も監査して明記する。
後継範囲の本費用campaign・帰属診断への採用と統計設計は#29/#30の明示判断へ渡す。
新sourceや再学習を要する拡張は別の有限予算とする。

## Consequences

- 外部取得の無期限待ちを終了し、3凍結方策を同じ入力で検査できる。
- 元17foldを回復したとは主張しない。全不足と履歴不足を機械可読で残す。
- 先の価格を用いた14行の価格修復とFRED改訂vintage不明は残り、因果的入力ではない。
- ADR-0037/0038/0039と旧sealed成果物を変更せず、後継範囲の判断材料だけ追加する。
