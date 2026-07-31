# 0008. FRED 金利による CarryAnnual 付与(forex-add-carry)

## Status

superseded

Superseded by ADR-0010.

## Context

forex-env ADR-0009 は符号付き資金調達(キャリー)を導入し、データの各シンボルに
`CarryAnnual`(年率金利差 counter − JPY、小数)を要求する。金利データの取得・整形・
因果ラグの適用はデータ作成側(trainer)の責務である。FRED(fredgraph.csv、認証不要)に
主要9通貨 + JPY の政策金利/3ヶ月インターバンク金利(月次)が揃うことを確認した
(SGD は欠落、対象外)。

## Decision

`forex-add-carry` CLI(`src/forex_trainer/carry.py`)を追加する。

- 入力: 既存の parquet キャッシュ。出力: `CarryAnnual` 列を各シンボルに付与した parquet
  (メタデータは入力から引き継ぐ)
- 金利は FRED 月次系列を日次に forward-fill し、`--lag-days`(必須)だけ遅らせてから
  キャッシュのタイムスタンプに整列する(発表ラグによるルックアヘッド防止)
- ペア→FRED系列の対応はモジュール内レジストリで管理し、対応の無いシンボルを含む
  キャッシュは明示的にエラーとする
- ネットワーク境界(系列取得)は注入可能とし、テストは合成系列で行う
- 特徴量レジストリに `carry_annual`(ペア別、そのまま観測へ)と `xz_carry`
  (ペア横断 z-score)を追加する

## Consequences

- signed モードの実験が「キャッシュ構築 → forex-add-carry → 実験」の3手で再現可能になる
- 金利の意味は選んだ系列(政策金利/3M interbank)に依存する近似であり、実際のスワップ
  レートとは乖離しうる(overnight_rate のマークアップで保守側に倒す)
- FRED の系列改廃リスクはレジストリの更新で吸収する
