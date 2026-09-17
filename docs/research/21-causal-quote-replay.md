# 公開後quoteを使う因果的約定replay（Issue #31）

**実装検証済み／執行の経済的影響は未検証。** 対象broker・口座・商品は未指定。
仮定付き研究replayを実装した。実口座収益性、同closeの実行可能性、17年の独立収益性は認定しない。
#28/#29/#30のhistorical開発と既存結果は変更していない。

[ADR-0040](../decisions/0040-replay-first-eligible-post-publication-quote.md)に
execution `first-eligible-quote-v1`／measurement `post-publication-quote-jpy-v1`を固定した。
同close研究の`full-period-history-v1`とは別の契約である。

## 実行と成果物

```console
uv sync --group dev
uv run forex-quote-replay --scenario fixture \
  --input tests/fixtures/issue31/scenario.json \
  --output runs/issue31-fixture
uv run pytest tests/test_quote_replay.py tests/test_quote_replay_edges.py -q
```

開発専用コマンドのため、schema検証には既存のdev依存（jsonschema / rfc3339-validator）を使う。
出力先は新規directory必須。scenarioは省略不可。`fixture`は合成された9商品・2 decision、
未学習の実PPO seed42/43/44、合成ridgeと既存canonicalの実装検証専用で、新規学習はしない。
fixtureの成績を学習済み方策の成績として扱わない。3方策は別口座のassetsから毎回再推論する。

| 出力 | 内容 |
|---|---|
| `input.json` | window、各情報のavailable_at/retrieved_at/source/quality、quote、mark、calendar、費用の全snapshot |
| `coverage-plan.json` | replay前に固定した共通範囲・quote数・3方策・入力hash |
| `<policy>/decision-NNNN.json` | 実口座状態からのscores/action、観測hash、仮想生成/保存時刻。書込みとfsync後にfill処理へ進む |
| `<policy>/fill-NNNN.json` | quote、売買数量、bid/ask約定価格、spread/commission、fill後口座、decision hash、実際のfsync完了UTC時刻 |
| `<policy>/result.json` | 全stepのgap/保有PnL、資金調達/markup、notional、数量、JPY換算、時刻、最終口座。incident時も保存 |
| `summary.json` / `provenance.json` | 全方策の完走状態、入力/schema/model/code/runtimeと出力のSHA-256。経済的検証はunverified |
| `incident.json` | schemaを満たさないCLI入力のpreflight例外と初期flat口座 |

quote欠損、deadline超過、資金調達のcoverage不足、数量不足等は原因とoriginを記録し、
CLIは終了code 2。最後の妥当な状態がfillまで進んでいれば、その数量とequityを保存する。
margin breachはデータ例外と区別して`terminal_margin`を残し、未計測範囲を保持する。
全方策完走時にも、fixtureの「実装完走」を経済的検証へ読み替えない。

検証結果: `uv run --locked pytest -q`は**418 passed**。新規replay回帰テスト42件を含む。
明示fixture CLIでも3方策の2 decisionが完走した。テストは時刻・会計・数量・費用・
incident保存・実PPOのassets再推論・封印済みmodel改変拒否を検証し、既存テストは変更していない。

## 入力と会計の境界

[閉じたJSON Schema](../../src/forex_trainer/quote_replay.schema.json)と
[実行できるfixture](../../tests/fixtures/issue31/scenario.json)を参照。
全timestampはoffset付き必須。bar_labelはUTC session-close日を識別し、公開時刻の代わりにはしない。
windowは固定pair順のfloat64 `[9,32,8]`（既存FEATURES順）。PPO/canonicalへはfloat32を渡す。
`market_and_carry_window`のpayload SHA-256はsort_keys・compact separators・有限JSONで照合する。
各入力には公開/取得時刻を付け、window構成に使った資料のsource/qualityを保持する。
実データでは、そのwindowが当時公開された入力から作られた証拠を別途確保する必要がある。

判断生成以前に全情報が取得済みで、判断保存後に発生した最初の適格quoteを使う。
quoteの受信まで待ってfillし、quote発生→受信のageは1秒以内、判断保存→受信は60秒以内。
発生・受信順の逆転、crossed市場は拒否する。発生から受信まで同じ明示session内で、両時点が休日外、
tradableであることが必要。適格quoteのdepth不足はその場で停止し、後のquoteを選ばない。

シナリオ上の生成・保存時刻は**仮想replay clock**。実ファイルの永続化時刻は別に記録する。
過去の実時間内に推論・保存・発注できたことの証明にはならない。
入力は9商品の同期basket。全量basketは研究仮定であり、実市場での原子的約定を保証しない。
部分約定、非同期pair合成、翌open/barによる補完は未対応として拒否する。

対応商品はXXX/JPYのbase単位で保有するJPY cash-settled線形契約に限定する。
例えばmodel `JPY/USD`の+0.5はUSD/JPY商品のshort。
fill直前equityをE、midをPとして`q = trunc((-w E / P) / lot) × lot`。
商品PnLは`q × (P_end − P_start)` JPY、quote通貨JPYからの換算は1。
逆数returnを符号反転して商品PnLと同一視しない。JPY base、非JPY quote、現物多通貨cashは拒否する。

前建玉のdecision→fillのgap損益とfinancingを反映したEで新数量を決める。
buy deltaはask、sell deltaはbid。spreadはmidとの差を1回計上し、固定spread debitは0のみ受け付ける。
commissionは実fill価格の売買notionalに課す。lotへ丸めた後の非ゼロdeltaがminimum未満なら拒否する。
最終markでは損益を計上し、架空の清算はしない。

financingは商品long/short別のsigned JPY/base/dayを、明示有効区間と実経過秒/86400で積分する。
別markupもbase/day単位で記録し、swap内包時の追加markupは拒否する。
全有効区間の情報がdecision時点で取得済みでなければ停止する。
これは線形の暦日資金調達仮定であり、実brokerの離散rolloverや請求swapを再現したものではない。

assetsの3列は、実現数量のmodel向きfill notional/E、leverage=1、前区間の商品price PnL/decision equity。
初期のみ全0で、price PnLにはgapと保有区間を含む。新数量会計のassetsへ既存PPOを適用すること自体が
感度仮定であり、旧inverse-return環境と同一分布を保証しない。pair clip ±1、gross cap5、
初期100万円、equity20万円以下の停止と明示margin率を使う。intrabar強制清算は未対応。

## 観測quoteを用意できた場合

`shadow_quote`でも結果は仮定付き研究replay。入力のmarket/quote/mark qualityは`observed`が必須。
`actual_fill`はCLI/schemaとも受け付けない。sourceがない場合にfixtureへ切り替えない。

```console
uv run forex-quote-replay --scenario shadow_quote \
  --input path/to/registered-observed-input.json \
  --sources path/to/frozen-sources.json --output runs/issue31-registered-shadow
```

sourcesには次の3キーを必須とする。pathは既存リポジトリ規約どおり実行cwdから解決する。

```json
{
  "source": {"path": "docs/research/results/issue15/provenance.json", "sha256": "<事前に固定した親seal hash>"},
  "fold": "<事前に指定したfold>",
  "runtime": {"forex_env_sha": "<固定env SHA>", "versions": {"<全runtime依存名>": "<固定version>"}, "device": "cpu"}
}
```

`runtime_identity()`のversions集合をすべて固定する。親models.json、fold config、
validation-best ensembleとseed42/43/44の各model/config/metaを既存loaderでhash検証し、
最新runの探索・legacyへの代用・再学習はしない。
broker/口座を記入するなら、商品仕様の公式URL・取得日時・有効日時・保存資料hashも必須とする。
現時点ではすべて未決で、特定brokerの仕様や費用を採用していない。

## 実データcoverageと引渡し

[損益閲覧前のcoverage台帳](protocols/issue31/coverage.json)に検査時刻とファイルhashを保存した。
ローカル`data/*.parquet`のschema・metadata、および`data/`と`runs/`のquote/tick/jsonl候補名を棚卸しした。
外部に取得可能なquoteが存在しないと結論したわけではない。大規模収集は開始していない。

| 保存入力 | 範囲・件数 | 公開後quoteに必要な不足 |
|---|---|---|
| 日足raw/clean/carryの3 parquet | 各5,292行、9pair、2005-07-19〜2025-12-31、Europe/London label | available_at/retrieved_atなし、bid/ask/depthなし。labelは公開証拠にならない |
| quote/tick/jsonl候補 | 上記ローカル範囲では0 | 9商品の時刻付き取引可能quoteとquantity証拠が未取得 |
| 共通replay可能範囲 | **なし、17foldすべて適格decision 0** | 同closeとの実データ対応比較は開始していない |

次の入力を揃え、損益を見る前に3方策の共通期間・source・calendar・費用を固定する必要がある。

- 対象broker/口座/商品と、lot・minimum・margin・netting/決済・financingの公式一次資料、取得日・有効日。
- 当時のOHLC・修復・carry vintageを含む入力snapshotと、公開・取得記録。
- 記録後の9商品のbid/ask、quote発生・受信時刻、数量対応、depth、holiday/session記録。
- 同一入力・範囲・初期口座を使う同close研究対照と公開後quoteの別campaign。

実データが揃っても、provider・商品・価格座標・数量丸め・資金調達まで異なれば全差をlatency効果とは呼ばない。
短期quote観測やfixtureを17foldの長期収益性へ外挿しない。#31の実装と不足sourceの引渡しは完了し、
執行の経済的影響は未検証として次のデータ契約へ渡す。

実装時にContext7 MCPで参照した一次資料（2026-09-12取得）:
[SB3 deterministic inference](https://github.com/DLR-RM/stable-baselines3/blob/master/docs/modules/ppo.md)、
[Gymnasium spaces](https://gymnasium.farama.org/api/spaces/)、
[jsonschema validation](https://python-jsonschema.readthedocs.io/en/stable/validate/)。
