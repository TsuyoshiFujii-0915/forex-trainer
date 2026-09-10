# 通年評価・執行確認の具体的仕様案

状態: **proposed / 未実装**。親決定は[ADR-0035](../../../decisions/0035-separate-observation-history-from-measurement-range.md)。
この文書は次のtrainer/env実装をレビューするための受入仕様であり、Issue #22で測定契約を変更しない。
[現行監査](../../16-execution-and-measurement-audit.md)と[出典](sources.md)を前提にする。

## 範囲と時刻を固定する

新契約`full-period-history-v1`では以下を必須入力としてmanifestに保存する。

| 入力 | 定義 |
|---|---|
| measurement_start / end | timezone付きUTC instant、半開区間`[S,E)`。year labelだけから推測しない |
| provider calendar | 9 pairの期待bar集合、timezone、session開始終了、休日、bar label→close/available_atの規則、version/hash |
| observation history | first decision直前の有効raw 63本と、そのfirst decision以降の測定対象bar。取得開始label・raw hash・available_atも固定 |
| execution model | 同close仮定を残すhistorical研究用モデルと、公開後quoteで動くshadowモデルを別IDにする。双方を混在させない |
| portfolio boundary | foldごとに初期equity=1,000,000 JPY、全pair exposure=0、PPO assets=0、initial entry costを含む |
| termination | Eより前の最後の対象markで評価終了。endまでに存在しないquoteを補完しない。early margin callはincompleteとして全対照の判定を止める |

first decisionは**事前calendar上でS以降の最初の実行可能なdecision**とする。
Sが休日ならその時刻までは建玉ゼロで、運用上の待機期間と報告する。休日と配信欠損を同じ扱いにしない。
予定first decisionをactivation時にUTCで確定し、当日の履歴不足・未公開・bar欠損は例外/incidentとする。
データがある最初の日へ黙ってスライドしない。

historical同close診断では、raw日足labelに対する登録済み時刻写像を使い、
`S <= decision < target < E`を満たす連続bar transitionだけを対象にする。
終端targetをEへ延長せず、target=Eも除外する。最後のbarはmarkにのみ使用し、次targetがなければ
新しいrebalanceをしない。終端で仮想清算・清算costを追加しない（既存の時価評価契約を維持）。
この最終markからEまでの未測定時間をcoverage gapとして報告し、年末まで売買できたと説明しない。

年初・年末に24時間市場が閉じていても「通年」は指定範囲の全**取引可能**transitionの意味とする。
常にS、E、first decision、last mark、初期flat待機時間、末尾coverage gapを別々に出す。
主年率化とSharpeは既存通りfirst decision→last markの実経過秒/365.25日を使う。
宣言した365/366日で割った別値へ主指標を差し替えない。もしS→Eの暦年資産経路を測る必要があれば、
Eでの評価quoteとboundary mark会計を別途規定した新契約が必要である。

shadowではlabelを実時間だと仮定せず、全入力のavailable_at以後に推論・記録し、その後の
取引可能bid/ask quoteを使う。historical close→closeと同じscore target/損益区間でなくなる場合は
別測定IDで3対照を揃える。翌openへの変更はclose→closeの単なる時刻置換ではない。
前回建玉のgap PnL、約定時equity、価格基準、保有時間、翌markを含めた仕様を別ADRで採用するまで
その評価を実装・実行しない。

## 履歴・特徴量・fold境界

現行longfは8特徴量（log_return、volatility32、sma20_ratio、mom24、xz_mom24、xr_mom24、
carry_annual、xz_carry）、normalize=false、window32である。最初の観測windowはdecisionを含む
32行、その最古のfeature rowまでに32本のraw warmupが必要なので、decision直前の63本を要求する。
「64本の過去履歴」にしない。first decisionそのものを含む入力は64本になる。
週末・休日・欠損で日数は変わるため、60/90暦日の固定fetchでは十分性を判定しない。

履歴は同じprovider・pair順・修復/vintage契約で揃える。9-pair intersectionで欠けた日時を黙って消さず、
期待calendarとの差を例外にする。prefixが不足するfoldは評価不能として止め、他foldだけの採否にしない。
特徴量は観測時点までのデータのみで計算し、未来の値を変更しても過去windowが変わらないことを検証する。
trainでfitしたridge standardizer/coefficients/alpha、PPOモデルは凍結し、履歴やevaluationで再fitしない。
normalize=trueの別構成ならenv既存の観測window内z-scoreを維持する。新しいglobal正規化を導入しない。
現行にないEMA等へ拡張せず、特徴量を変える要求があれば別契約で必要履歴を定義する。

前年末の価格・carry履歴は読み込むが、前年のportfolio equity/exposure/assetsを次foldへ引き継がない。
独立foldの初期資産と連続運用の資産継承は別問題である。fold間の連結損益・継続保有を捏造しない。
これまで年内warmupにより観測されなかった1〜3月は新しい予測が必要だが、2009〜2025は
#20のdevelopment期間のままであり、新しい独立確認期間へ昇格しない。

## 全3対照の再評価・provenance

1. #20の[identity manifest](../issue20/artifact-identities.json)と#15/#16 sealを親とし、17 foldそれぞれの
   frozen ridge model（standardizer含む）、canonical mom24 rule、current PPO ensemble seed42/43/44を固定する。
   policy family・seed・checkpoint・top/bottom各2・±0.8・k=1・同点契約を変更しない。
2. 拡張した観測履歴から全対象decisionのscore/actionを新規生成する。#15/#16の旧CSVを継ぎ足して
   新しいsealと呼ばない。欠けたpredictionをゼロ・canonical・latestモデルで補わない。
3. 全方策を同じdata、観測範囲、実効decision/target、cost、execution ID、両Git SHA、依存version、
   CPUで評価する（ADR-0016/0017/0029）。履歴拡張前後のaction差も説明できるtraceを保存する。
4. 新manifestに親seal、各model/config/raw/clean/carry hash、変換lineage、calendar、各inputのavailable_at、
   history開始終了・63本の一覧、S/Eと実効時刻集合hash、boundary/initial portfolio、特徴量/正規化、
   rate vintage、source/評価Gitと依存、commandと新出力先を保存する。replayの時刻・pair・会計照合も保存する。
5. 新measurement IDの配下へCSV/JSON/reportを作成し、既存directoryへの出力は例外とする。
   旧fold metricとのpaired比較を拒否する。新17 fold内で3対照を比較し、旧結果は別表の記述的参考に留める。
   10,000 IID/3-fold moving-block、seed16、era・LOO・採否は#20既存基盤に従う。

## trainer/envの責務と受入テスト

テストを先に固定し、実装中に期待値を変更しない。以下は**後続実装の受入条件**であり、今回の
13件の台帳テストだけで実装済みとはしない。

| 責務・担当repository | 再現する利用者の振る舞い | 受入条件 |
|---|---|---|
| trainer: config/manifest、env: resetとmeasurement mask | 年初に十分な前年履歴を指定して評価 | first decisionが登録時刻と一致。warmupのequity変化・costは0。initial assets/exposure=0、初回entry costは一度だけ |
| trainer: calendar/data検証 | Sが週末/祝日、途中に配信欠損 | 登録済み休日は予定どおり次session。期待bar欠落はpair・時刻・originを含む例外。範囲縮小なし |
| trainer: history loader | 63本と62本のprefixをそれぞれ供給 | 63本で同じfirst window、62本では例外。推論結果を生成せず停止 |
| env: feature/window、trainer: standardizer | first decision以後の未来値を改変 | 当時の観測・score/actionは不変。fit済み標準化hashは不変、window内normalizeの既存挙動を維持 |
| trainer/env: time boundary | DST、閏年、target=E、target>Eの入力 | UTC実経過秒で課金・年率化。E以上は対象外。年境界の重複transitionなし、終了後rebalanceなし |
| trainer: fold orchestration | 前foldを利益/損失で終え次foldへ進む | 次foldは常に初期残高・建玉ゼロ。観測用市場履歴だけを継承 |
| trainer: model adapters | 旧CSVにない年初decisionを要求 | 明示凍結modelで推論、3対照の時刻/pair完全一致。model不足/改変は停止。再学習なし |
| trainer: provenance/report | 旧contractの対照、新cost、異なる依存、既存出力先を混ぜる | 比較または書込前に原因付き例外。旧成果物bytes不変 |
| env: execution/financing（別ADR） | 同じtargetでも価格変動、long/short、週末、rolloverを通す | 実notional、price、spread/commission、markup、signed carryの会計恒等式。#19/#21関数を再利用し二重課金を拒否 |
| trainer: shadow adapter | 入力公開前、quote未到着、partial fill、記録失敗 | 有効なdecision/outcomeを捏造せずincident・全対照中断。fill未取得をhistorical closeで埋めない |

## #20への引継ぎとpaper開始前の未決項目

`issue20-forward-v1`の2027候補は依然として年内warmup・未開始である。本提案採用時は新revisionを
登録し、2026-12-15固定期限、2027の計測候補、2028-02-01開封の整合を実行前に再確認する。
前年履歴取得のためだけに2026を「未使用」に戻さない。既存運用契約・schemaはこのPRでは変更しない。

| 未決項目 | 確定する入力・計測と保存場所 | 開始の条件 |
|---|---|---|
| 対象broker/法域/口座/商品 | spot/CFD等、基軸/決済通貨、min lot・step・margin・netting、営業日、公式契約URL/発効日/取得日/hash | 利用者が対象を指定し、公式仕様を照合。推測選定しない |
| 日足の意味・入力遅延 | 各pairのsession open/close、label、available_at、retrieved_at、quote timestamp、calendar hashをinput snapshotへ | 当時取得可能なOHLCとcarryでdeadline内に3方策を推論できる |
| 実売買と費用 | bid/ask、quote量、order/ack/fill時刻、fill量・価格、actual notional、commission、JPY換算rateをevidence snapshotへ | shadow quote・paper fill・live fillを区別。今回発注はしない |
| 資金調達 | long/short rate、indicative/final、as-of/publication時刻、rollover基準・倍率、holiday/value date、markupをsnapshotへ | finalの事後情報をdecision入力に使わず、broker請求を再照合可能 |
| 異常値・改訂 | raw bytesと旧/新値、検出時刻、利用可能時刻、変換code/config hash、修正理由を訂正recordへ | future-neighbor補間を当時の入力として使わない。拒否・中断経路を先に検証 |
| cost eventの区別 | trading=spread+commission、holding=markup、signed carry、各JPY額、想定/観測を区別 | #20 outcomeの`observed_transaction_cost_jpy`は恒等式上のtotal debitとしてmarkupも含め、snapshotで内訳を持つ。フィールド名を狭義売買費用と解釈してmarkupを落とさない |
| 記録schemaと保存 | #20 inputs/evidence snapshotに上記補助明細を追加。decision→outcome→incident/correctionの参照・hash・時系列を照合 | schemaの未知top-level field禁止を守る。実行基準を増やす場合は別schema versionを先に登録 |
| stress・測定ID | [監査のstress案](../../16-execution-and-measurement-audit.md#基準条件と保守的stressの設計案)の実測/仮定、全3対照の共通manifest | 未取得quoteや費用仕様があれば未開始。恣意的cost引下げなし |

paper接続・継続運用は別作業。全条件が揃わなければ未開始/中断と記録するが、
何が未決かを特定した本監査の完了を妨げない。短いpaper期間を収益性の統計的確立としない。
