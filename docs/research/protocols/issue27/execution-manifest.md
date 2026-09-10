# 短期開発の実行manifest

親契約: [issue27-development-v1](../../17-bounded-development-protocol.md)。登録日: 2026-09-10。
これは実行順・予算・引渡しの固定一覧であり、実行可能CLI configや収集開始のactivation manifestではない。
状態は全件**計画登録、実行未開始**。本Issueで実験結果を新たに生成していない。
実行前に埋める必須項目が未確定なら、その作業は開始不可。未確定を既定値や現在値で埋めない。

## 実行順・並行作業・入力と出力

主経路は **#27 → #28 → #29 → #30最終解析**。
#30の既存trace診断、#31のinterface/fixture、#32の記録実装を並行して進める。

| Issue / 登録campaign ID | 依存と先行可能な作業 | 入力 | 出力・引渡し条件 | 有限予算 |
|---|---|---|---|---|
| [#27](https://github.com/TsuyoshiFujii-0915/forex-trainer/issues/27) / `issue27-development-v1` | 最初に契約を確定。子Issue完了待ち不要 | #20/#22契約、既存seal、#28〜#32 scope | 本契約・判断表・本manifest・README | 契約1、新規学習0、市場評価0 |
| [#28](https://github.com/TsuyoshiFujii-0915/forex-trainer/issues/28) / `issue28-full-period-implementation-v1` | #27。設計/テスト準備は先行可 | ADR-0035案、#22 full-period仕様、3方策source | 採用仕様、legacy/full-period明示CLI、measurement ID、必要source一覧、fixture E2E、利用可能artifact smoke、env関連PR/SHAを#29へ | 測定実装1、研究campaign0、新規学習0。smokeは新たな市場仮説選択に使用しない |
| [#29](https://github.com/TsuyoshiFujii-0915/forex-trainer/issues/29) / `issue29-full-period-cost-v1` | #27/#28。source所在確認・manifest準備は先行可 | #28 evaluator、以下の凍結source、F0/F1/F2 | 153セルの完了状態、sealed step/pair/equity/cost trace、fold/era/対応差report、再現コマンド。F0全3方策の17foldを#30/#31へ | 1 campaign、3条件×3方策×17fold=153 policy-fold、新規学習0 |
| [#30](https://github.com/TsuyoshiFujii-0915/forex-trainer/issues/30) / `issue30-ppo-attribution-v1` | #27。#16/#19 legacy traceで実装・恒等式・先行記述解析可。最終は#29 F0必須 | 同一measurementの3方策trace、凍結PPOとtrain範囲 | common/relative/carry/costのstep/fold帰属、2対照とPPOの対応差、risk、era/LOO、次期仮説。legacyと通年は別ID/別表 | 最終解析1、先行legacy診断最大1。追加対照2×17=34評価、constant作成のtrain replay17。元3方策のF0を再利用、新規fit0 |
| [#31](https://github.com/TsuyoshiFujii-0915/forex-trainer/issues/31) / `issue31-post-publication-quote-v1` | #27。interface・fixture・会計実装は#28/#29と並行。通年統合は#28、基準線は#29 | 時刻付きinput/quote/mark、数量・費用仕様、凍結3方策 | 約定ADR/measurement ID、共通input interface、独立口座replay、coverage、実データ比較または必要source/未検証一覧を#32/次期campaignへ | 執行方式1、実データ感度campaign最大1。同close/公開後quoteの2条件×3方策、登録範囲は最大17fold。latency/session/fill探索0 |
| [#32](https://github.com/TsuyoshiFujii-0915/forex-trainer/issues/32) / `issue32-development-shadow-v1` | #27。#31 interfaceと整合して並行実装、#29/#30結果待ち不要。history統合に#28、outcome会計に#31を利用 | 2025凍結bundle、timestamped fixture、登録済read-only source | 1サイクル/再開/検証/head export CLI、append-only記録、fixture E2E。実装検証と実収集状態を分けて次期へ | adapter1、3固定方策、実sourceの短いsmoke最大1区間。開始/終了と最大decision数を初回前に固定。長期常駐・新規学習0 |

#30の予算は2対照を増減して選ぶ許可ではない。train-derived constantのtrain replayでは
凍結PPOを使い、parameter更新0。先行legacy診断は既存traceによる会計診断に限定し、
追加対照の市場評価はF0の34セルだけで行う。train replayの期間・初期状態・capと投影の厳密な仕様を
#30で結果前に固定する。common-onlyとconstantは元PPOと別口座経路を持つ。

#31の実比較範囲は損益を開く前のcoverage調査でUTC開始/終了・予定時刻集合・全体の不足範囲を
明示登録する。実データが不足すれば実装・fixtureの検証と必要sourceを納品し、経済的検証は未完了とする。
bar-based proxyの実験予算は本バッチでは0。同closeへの暗黙fallbackとして追加しない。
#32の実source/日程未指定ならsmoke実行0で「実装検証済み／実収集未開始」を引き渡せる。
データ不足を理由に期間を延長して成功件数を揃えない。

## 凍結するsource identityとruntime

参照snapshot（契約策定時のコード）:
trainer `66841be014a2a8b42210f8c8d918f408f229f90a`（PR #26）、
env `6024b91c0f3592611849bc231922ab60e6090aed`。
新評価器の実行SHAはまだ存在しないため、このsnapshotを将来の評価SHAとして流用しない。
#28採用後、各campaign開始前に実際の両repo SHA・dirty状態・lock hash・依存version・CPUを固定する。
同一比較中はコードを変更しない。env変更時は関連PRと採用SHAを登録する。

| source | 固定参照・照合内容 |
|---|---|
| 親identity | [#20 artifact-identities.json](../issue20/artifact-identities.json)。`artifacts`各pathのSHA-256を照合。現行sourceコードを旧hashへ補完せず、旧revisionと新evaluatorを区別 |
| 全17foldのPPO | [#15 provenance](../../results/issue15/provenance.json)の`fold_sources[year].ppo`。SHA-256 `a398ee18a01de364e21deb1d5f0ac5a66c1c1e445e60dc21c7c713215cebfb61`。各foldの明示ensemble path、17 manifestと51 member model hash、seed42/43/44、validation-bestを継承 |
| ridge | [#15 models.json](../../results/issue15/models.json)。SHA-256 `af0449183bee5c7d48528b567087b21b60e36d85f6ec200a8c620ef64a2f2b8c`。各foldのstandardizer/coefficients/alphaを保持。#15 predictionsの既存CSVを年初の推論へ継ぎ足さない |
| canonical/map/費用 | [#16 provenance](../../results/issue16/provenance.json)。SHA-256 `d2db580d5f11ec7ca341af04549c3a09e7ddb6777c43562b1117a24ae8a57db3`。#16 sealed configと共通会計・既存tie処理を継承 |
| historical data | `data/jpy_9pairs_1d_2003_carry.parquet`、SHA-256 `723db7c935dcc27c147007728358d098243eae96998eae146db4c7df02bd081e`。raw/clean/carryのlineageも照合。不足historyは具体的path/hashを記録し、別cacheへ切替不可 |
| shadow bundle | #20 identityの`forward_selection`。時間順で最後の2025 foldのridgeとPPO全3member、canonical。成績でfoldを選ばず、旧bundleの制約を記録 |

依存の出発点は#16 provenance `evaluation.versions`:
forex-env-v3 0.1.0、gymnasium 1.2.3、stable-baselines3/sb3-contrib 2.8.0、torch 2.12.0、
numpy 2.4.6、pandas 3.0.3、pyarrow 24.0.0、PyYAML 6.0.3。
これは過去評価の記録。新campaignの実環境を実測・封印し、差分があれば結果前に理由付きで登録する。
旧training時のSHA/依存を現在値で置換しない。全対照のevaluation-time両SHA・依存は一致必須。
source不足時にlegacy longfのmeta backfillや再学習を行う予算は0。

pair順は `JPY/USD, JPY/EUR, JPY/GBP, JPY/AUD, JPY/CHF, JPY/CAD, JPY/NZD, JPY/NOK, JPY/SEK`。
F0のspreadはこの順で
`0.000015, 0.000025, 0.000030, 0.000030, 0.000035, 0.000035, 0.000040, 0.000060, 0.000060`、
commission_rate=0、overnight_rate=0.00002/day、carry_mode=signed。
F1はspread列のみ2倍、F2はovernight_rate=0.00004/dayのみへ変更する。
数値はsealed #16/longf設定との一致を実行前に検証し、不一致なら停止する。

## campaign開始・引渡しで必須の記録

各子Issueは既存の明示manifest/reportを再利用し、少なくとも以下を固定・引渡す。
本書を汎用experiment managerの実装要求とは扱わない。

| 記録 | 必須内容 |
|---|---|
| 登録 | campaign/trial ID、親契約ID/hash、evidence_class、measurement/execution ID、登録日時/commit、具体仮説と変更軸、予算、planned状態 |
| 入力 | source所在と取得元、親seal、model/config/data/raw変換hash、全fold/seed、pair/feature/lag順、train/validation固定規則 |
| runtime | command、CPU、両repo SHAとdirty状態、Python/主要依存version、lock hash。training provenanceとevaluation provenanceを分離 |
| 時間と口座 | UTC S/E、calendar hash、history一覧、予定/実効decision/target集合、available_atの実測/仮定、初期状態、cap/margin、終了とcoverage gap |
| 出力 | 既存成果物と別の明示保存先、trace/report/statusとSHA-256、時刻/pair/会計照合、全セルのcomplete/terminal/incompleteと理由 |
| 判断 | net/gross対応差、fold/era/CI/全LOO、risk/cost、閲覧日時、分岐と反証/不足証拠、次期に引き継ぐ仮定 |

試行・attemptの数え方は[#20台帳](../issue20/period-and-trial-ledgers.md)を継承し、
**新しい子campaign台帳**へplanned→実行→閲覧→終了を追記する。旧凍結台帳を編集しない。
`trial_id / parent_trial_id / scope / hypothesis / changed_variable / frozen_values / budget /
folds / seeds / validation_rule / registered_at / registration_commit / opened_at / status /
attempts / artifacts_and_hashes / decision / unknowns`を記録する。
同じbytes/同じ条件の障害再実行は同trialのattempt、仮説や測定条件が変われば別revision/trialとして
実行前登録する。失敗・中断・閲覧後の修正も削除しない。再実行で探索予算を増やさない。
153セル、34対照評価、日次行やseedを独立trial/市場標本に数えず、全研究の累積独立trial数は不明のまま保持する。

## 外部情報の未決台帳と停止範囲

未決は明示的な開始条件不足であり、既定broker/providerの選定を意味しない。
historical cacheのlabel/session写像は研究仮定として#28で固定できるが、実際のavailable_atや
point-in-time vintageを証明したことにはならない。その未検証事項を次期campaignへ引き継ぐ。

| ID / 現在状態 | 必要な確定情報・担当 | 止める作業 / 続行可能な作業 |
|---|---|---|
| U1 broker/法域/口座/商品: 未指定 | 利用者指定後#31で公式一次資料URL、取得日・発効日/hash、base/quote、long/short、JPY換算、lot/step、margin/nettingを照合 | broker実行可能性の認定不可。#28/#29/#30とfixture実装は続行可 |
| U2 日足/quote provider: 未指定 | #31/#32で利用者指定read-only source、9pair、session/休日/DST、label/close/available_at/retrieved_at、bid/ask、quote量/qualityを登録 | 実quote比較・外部収集未開始。#28の仮定付きhistoricalとfile fixtureは続行可 |
| U3 執行時刻・数量: 未固定 | #31で入力公開≤判断≤durable記録<適格quote、staleness/deadline、fill/mark、gap PnL、数量/rounding、未対応partial fillの拒否を結果前に固定 | 実約定感度は未実行。単一方式のinterface/テスト準備可、latency探索不可 |
| U4 financing/費用: 実仕様未指定 | #31/#32でlong/short rate、publication/as-of、indicative/final、rollover/休日倍率、commission、markup内包とspread二重課金回避を登録 | 実cost/outcomeの認定不可。F0/F1/F2の固定仮定は実行可。shadowはraw/有効decisionまでの部分稼働を明記、outcome捏造不可 |
| U5 開発shadow日程・保存先: 未指定 | #32で開始/終了・最大decision数、deadline、2025 bundle、初期100万円、single-writer/durable append、restart、障害先/head export、停止運用を初回前に固定 | 実smoke未開始。fixture検証可。遡及開始や旧#20日程流用不可 |
| U6 独立確認運用: 未整備 | 別確認契約で未使用性、事前統計設計、閲覧分離、独立head保管と責任者を確定 | 独立確認は未開始。development shadowに確認運用の保証を付与しない |

実sourceは指定済みと推測しない。口座作成、有料契約、認証情報取得、売買発注、
大規模高頻度収集、長期常駐運用はこの予算に含まない。#31/#32の未決や長期観測待ちを理由に
historical開発全体を無期限停止しない。終了時は解消済み/未解消、必要source、影響範囲を引き渡す。
