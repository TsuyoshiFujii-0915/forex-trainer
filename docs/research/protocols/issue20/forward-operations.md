# 前向き記録の開始・終了・開封契約

契約ID `issue20-forward-v1`。現在は **blocked / 未開始**。本書は記録schemaと運用手順であり、
収集サービス、発注機能、broker接続を実装したという意味ではない。
運用実装と観測の実行は別作業とし、将来観測の完了をIssue #20の完了条件に含めない。

## 候補日程と開始条件

すべてUTC、範囲は開始を含み終了を含まない。候補は2027年のshadow観測1 campaignとする。
実口座は対象外。shadowでも外部価格・carry・costの供給元を推測しない。

| 項目 | 契約 |
|---|---|
| 方策・protocol固定期限 | 2026-12-15T00:00:00Z。期限までに下記activation manifestを封入できなければ未開始として停止 |
| market data取得対象 | 2027-01-01T00:00:00Z〜2028-01-01T00:00:00Z。各時点までに公開された9-pair日足とcarryを逐次snapshot化。未来dataを先に取得しない |
| 観測用履歴 | 同じ2027年range内の先頭32 feature-warmup rowsと32-row observation window。既存envの先頭decision index63を維持。別年の履歴を暗黙に継ぎ足さない |
| 損益計測開始 | 先頭decision index63の時刻。**正確なUTC日時は未確定のため開始不可**。供給元のbar calendarと公開遅延を固定し、予定日時と実行deadlineをactivation manifestに実行前登録する |
| 損益計測終了 | 2028-01-01T00:00:00Zより前にtargetが入る最後のtransitionまで。年境界を越すtargetで延長しない |
| decision記録開始 | 登録済みの最初の実効decision。warmup中は取得・欠損のoperational記録のみ |
| 終了条件 | 上記暦日による固定終了。正の成績になるまで延長する方式や「有効N件に達するまで」延長する方式は使わない |
| 開封 | 2028-02-01T00:00:00Z以降に1回。終了後31日間は確定待ちと照合のみ。未到着dataは欠損として報告し、開封を好都合な結果まで遅らせない |

ここでの年内warmup契約はhistorical評価と同じであり、「2027年全日を取引した」成績とは呼ばない。
全暦年損益のために前年履歴を与える変更、評価会計の変更は別の現実性監査・実装Issueとする。
年内holiday等で予定index63の日時が実到着calendarと一致しなければincidentを残し、確認は中断する。
欠損を詰めて開始日を後ろにずらさない。

固定期限までに次の全項目を持つ **activation manifest** を新規ファイルとして作成し、
担当者のレビュー記録とcommit、raw bytes SHA-256を独立保管する。未確定をnullのまま開始してはならない。
必要情報が揃わないまま期限を過ぎたらこの候補は失効する。別の将来日程を事前登録し、遡及開始しない。

| 必須項目 | 必要な確定内容・現在の不足 |
|---|---|
| データ | provider名・取得手段・raw schema・9-pair順・日足timestampの意味・timezone・休日calendar・OHLC確定/配信遅延・改訂の扱い。現時点では未選定 |
| carry / cost | 利用可能時刻を持つcarry供給元とlag、spread/commissionの想定値の根拠、観測値の取得手段、JPY換算、cost model hash。旧FRED歴史系列の60日lagだけでpoint-in-time性を保証しない |
| 時刻 | 正確なfirst decision UTC、各barからのdecision deadline、target/mark時刻。公開済み入力で実行可能である証拠。bar labelと公開時刻を混同しない |
| 方策 | [identity manifest](artifact-identities.json)の2025 ridgeモデルとPPO3 member、canonical実装・同点処理、config、data、依存・両Git SHA、CPU、初期shadow残高1,000,000 JPYを含むbundleのpath/hash |
| 測定 | 3方策を同じ入力/時刻/会計で動かすadapterの実装・検証証跡、raw equity・price/carry/cost分解、turnover、MDD定義。価格未到着や記録失敗時の停止経路 |
| 保存 | append-only保存先、書込/閲覧権限、独立head保管先、責任者、バックアップ復元の検証、毎日00:00Zのhead固定手段。現在未構築 |
| 閲覧分離 | operatorは到着・連鎖・遅延・障害のみ監視、研究担当は損益・累積score統計に開封までアクセス不可。役割と実際のアクセス制御を確定 |
| 登録証跡 | protocol・schema・artifact manifestのhash、開始終了開封日時、探索0/学習0、登録commit、担当者とレビュー日時、trial ledgerのplanned行 |

shadowの「観測cost」は実際に到着したquote等に固定cost modelを適用した値であり、約定costではない。
outcomeの`execution_basis`を`shadow_quote`とし、実口座成績として説明しない。
開始条件を満たしても売買発注や継続運用が本Issueから自動実行されるわけではない。

## 1行の記録schema

[forward-record.schema.json](forward-record.schema.json)はJSON Schema Draft 2020-12。
UTF-8 JSON Linesで各行に1 event。全eventはversion、固有ID、連番、記録UTC、protocol hash、前行hashを持つ。
未知fieldは禁止。NaN/InfinityはJSONではなく拒否する。9数値vectorは以下のcanonical順に固定する:
`JPY/USD, JPY/EUR, JPY/GBP, JPY/AUD, JPY/CHF, JPY/CAD, JPY/NZD, JPY/NOK, JPY/SEK`。

| event_type | payloadの意味 |
|---|---|
| decision | policy ID、decision/target時刻、入力の最終available_at、policy/config/data/input snapshot各path+SHA-256、9 score、提案weight、実効weight、想定transaction cost JPY、想定signed carry JPY、cost model hash |
| outcome | decision event ID、観測時刻、shadow quote由来の観測cost JPYとsigned carry、price PnL、開始/終了equity、根拠snapshot path+hash。decisionには後から書き足さない |
| incident | 対象decision ID（作成前ならnull）、予定decision時刻、error code、原因・origin、検出/停止内容。input不足、配信遅延、hash不一致、記録失敗、閲覧、開封、中断を残す |
| correction | 元event ID、訂正理由、訂正証拠snapshot path+hash。元bytesを変更せず、派生reportで訂正の扱いを明示する |

policy artifactはPPOなら全member hashを持つmanifest、ridgeなら係数・standardizer・alphaのbundle、
canonicalならrule実装とparametersのmanifest。data artifactはその時点のraw data snapshot。
inputs artifactは実際に渡したfeature tensor、各入力のevent time / available_at / retrieved_at、
pair・feature・lag順、carry publication/vintage、欠損状態を含む。
`inputs_available_at`はその全入力のavailable_at最大値。mutable URLやパスだけでは証拠にならない。

[架空の例](forward-record.examples.jsonl)はdecision→outcome→incident→correctionの順。
`example://`参照と架空hashはschemaの説明用であり、実データ・確認開始の証跡ではない。

## 書込み時の運用検証（収集実装の必須要件）

1. 起動時にactivation manifestと全bundleのhash、依存、pair順を照合する。不一致は原因とpathを
   含む例外で停止する。新しいrunや異なるmodelへfallbackしない。
2. 各decision前に、全入力の`available_at <= decision_at < target_at`と実際のdecision deadlineを照合する。
   `recorded_at`がdeadlineを過ぎたdecisionは有効な事前提案として扱わずincidentにする。
   遅延・欠損・同値degenerate score等で提案が作れない場合はゼロactionを捏造しない。
3. JSON Schema + FormatCheckerで検証する。schema単体は行間参照、時刻の大小、hash実体、
   scoreからactionへの意味的整合を検証しないので、保存側が追加で照合する。
4. 全3方策についてdecisionと想定costをdurable保存してからshadow適用する。
   appendを単一writerで直列化し、event IDの再利用・連番欠落・二重適用を拒否する。
   前行の改行LFを除く実UTF-8 bytesをSHA-256にし、次行に保存する。genesisのみsequence=0、前hash=null。
5. outcomeは対応decisionが存在し、policy/targetが一致し、同じdecisionの二重outcomeではないことを照合する。
   `equity_after = equity_before + price_pnl_jpy + observed_signed_carry_jpy - observed_transaction_cost_jpy`
   とequity連続性を検証する。schemaのnumber許可だけで会計が通ったとしない。
6. 当日head hash・行数を毎日00:00Zに研究担当が変更できない別保管先へ固定する。
   原本の上書き/削除を拒否し、atomic appendと永続化を確認する。chainのみでは全履歴の書換えを防げない。
7. 記録できなければ以降のdecisionを止め、独立した障害記録先へoriginと失敗時刻を保存する。
   復旧後はincidentを追記する。過去のdecisionを再推論して「当時の提案」として補完しない。
8. 訂正はcorrectionを追記する。修正後の表だけでなく元記録・訂正理由・両者の差も開封reportに残す。

schema検証例（repository root、開発依存をsync後）:

```console
uv run pytest tests/test_forward_record_schema.py -q
```

このテストは交換schemaの正常例・拒否例を検証する。上記の永続化、権限、時刻・会計照合が実装済みで
あることは証明しない。収集実装の別作業ではそれらの振る舞いテストを実装前に作る。

## 途中閲覧・中断・開封

途中に許可するのは収集の稼働、到着数、hash整合、遅延・障害の確認のみ。
PnL、勝率、Sharpe、CI、方策間の成績差を研究担当へ定期表示しない。operatorによる内容アクセスは
理由・範囲・時刻をincidentに残す。research担当による途中成績閲覧が起きた場合は、理由を問わず
当該campaignを`opened_early`とし確認資格を失わせる。継続観測はdevelopmentとして分離する。
成績を使った設計変更は必ず新trialであり、残りの同一年を再封印しない。

hash/会計/時刻の不一致、入力欠損、decision期限超過、記録不能、equity≤0または既存margin-callで
**全3方策のcampaignを中断**する。実行可能な方策だけ残す・欠損日を除く・都合のよい終了日に切ることはしない。
損益が良い/悪い、CIが0を跨いだ等の成績を理由に早期成功宣言や延長をしない。
停止した結果は`incomplete`とし、既知の原本・障害とともに終了日程どおり開封する。

開封担当者は予定時刻以降に`scheduled_opening` incidentを追記し、凍結manifest、head、全行、
時刻・pair・会計照合、欠損/閲覧/訂正台帳を検証してから一度だけ報告する。
3方策の実効開始終了、decision数、price/carry/gross/net、mean/worst MDD、exposure/turnover、
想定/観測cost差と障害を同じ表に残す。結果が不完全でも除外して合格にしない。

**1年のshadow観測は運用再現性の確認であり、統計的収益性の確立ではない。**
1つの将来foldからIID/3-fold moving-block区間やLOOを捏造せず、記述的に報告する。
既存17開発foldへ2027を足して独立確認の標本数を18と呼ばない。
数週間〜数か月の良好な成績、あるいは1年のnet正でも既存分類を昇格しない。
統計的な独立確認には、開封前に独立標本・期間・検出力・停止規則を固定した別契約が必要となる。
