# 共通方向の短期再開契約と段階別gate（Issue #55）

**`issue40-common-direction-development-v3`：全34外側ケースのcalendar適格性を確認した。**
2009は24か月、2018は12か月、残り15foldは6か月のvalidationを採用する。
短期契約は確定、中期RLと実執行はblocked。本PRのレビュー・マージおよび各Issueの実装・入力・
runtime sealが必要であり、現在の本学習・市場評価・外部取得の有効予算は全て0。

[新登録・試行台帳](protocols/issue40/revisions/v3/registration.json)、
[全候補136ケース](protocols/issue40/revisions/v3/preflight/candidates.csv)、
[採用34ケース](protocols/issue40/revisions/v3/preflight/counts.csv)、
[判断・境界・hash](protocols/issue40/revisions/v3/preflight/report.json)を正本とする。
[ADR-0046](../decisions/0046-separate-short-campaign-and-rl-data-gates.md)に基づく。

## 改訂範囲と保持する履歴

親は[Issue #40 v2 manifest](26-common-direction-learning-campaign.md)と
[v2登録](protocols/issue40/registration.json)。親の「現在」はv2登録時点を指し、現在の入口は本v3。
v1/v2の本文・登録台帳・preflight、#39/#29後継/#30後継のsealed成果物は元bytesのまま保持する。
変更は短期train/validation境界と段階別開始条件のみ。仮説common、ridge、h=1/5、5特徴量、
alpha、数量配分2×2、費用F0/F1/F2、17評価区間、risk、0.005期間net log、比較、tie-breakは
v2の第1〜6節を継承し変更しない。中期モデル構成も探索しない。

#40は停止判断まで歴史上完了。旧#29/#30の通年受入条件は未達（81/45不足）のまま、
復旧タスクを現行経路から取り下げる。後継153/85口座を診断の引渡しとして使い、実績を上書きしない。
将来元通年契約を回復する必要が生じた場合だけ別途判断する。部分年9区間や既知development期間を
独立確認へ昇格しない。学習改善・配分価値・canonical置換・独立収益性も別判断のまま。

## Calendarによる一つの分割規則

baselineはv2の[442ケース／97不足](protocols/issue40/preflight/report.json)。今回再計算するのは
短期外側だけで、中期374 cross-fit、17 PPO validation、17 RL trainを救済・再診断しない。
入力はsealed #39 coverageの要求London日付・欠損日一覧とv2登録のみ。価格、特徴量実値、
予測誤差、取引損益、model推論は読まず、外部取得もしない。

1. 外側fold Yの情報cutoffはLondon local `Y-01-01 00:00`。評価区間は#39の17区間を維持。
2. validation lookbackは6、12、18、24か月の4つだけ。各期間でh=1/5を両方検査し、
   両方がtrain≥252・validation≥60になる最短期間を一つ選ぶ。4期間の結果を全て保存する。
3. trainは`[2003-06-01, validation開始)`、validationは`[validation開始, cutoff)`。
   実在開始2005-07-19を要求開始の代用にせず不足を保持する。horizonごとの境界変更は禁止。
4. 連続63本の**先行**historyとh営業日のlabel終点が同じ期待平日ブロック内に必要。
   過去feature historyの持込みは許すが、欠損を横断せず、終点≥range終端のlabelをpurgeする。
   London平日・休日除外0を保持する。UTCへはEurope/LondonのDSTを含めて変換する。
5. 全候補/採用ケースにpresent、history除外、gap-label除外、境界purge、有効数、最長区間を保存。
   `present = history除外 + gap-label除外 + boundary purge + 有効数`。
   CSVのfirst_decision/last_markは既存仕様どおり最長ブロックの情報であり、全有効行の端点ではない。

| fold | 採用月数 | train終端＝validation開始（London） | train h1 / h5 | validation h1 / h5 |
|---|---:|---|---:|---:|
| 2009 | 24 | 2007-01-01 | 268 / 264 | 189 / 181 |
| 2010〜2017 | 6 | 各Y-1年07-01 | 全て252以上 | 全て60以上 |
| 2018 | 12 | 2017-01-01 | 2229 / 2205 | 162 / 154 |
| 2019〜2025 | 6 | 各Y-1年07-01 | 全て252以上 | 全て60以上 |

2009の18か月はvalidation59/51で不適格。24か月のh5 train264は最低252に近く、後続の
実値検査で減れば停止する。日付上の適格性は価格/feature/targetの有限性や学習成功の認定ではない。

4候補が不成立の場合も34行を残し、未採用の境界を実行に渡さない。
`lookback_months=null`、`boundary_role=last_candidate_diagnostic_only`の行は24か月候補の
診断件数であって採用splitではない。学習は0のまま、変更する不変条件を「既存data bytesのみ」に
絞り、**別data identityで有限の入力回復を登録する**ことを単一推奨案とする。
17区間・history・最低件数を保持し、取得元・試行数・予算は別レビューで確定する。ここでは取得0。
元responseの無期限探索、追加lookback、損益によるfold削除には戻らない。今回はこの分岐は不要。

## 段階別gateと実行許可

| gate | 現在 | 登録する必要条件 | 許可対象（各実装・実行seal後） |
|---|---|---|---|
| short_data_ready | ready | 全34 calendar、以下の実値検査契約、固定17範囲、予算・比較・採否seal | #41 fit、#42配分/会計、#43評価 |
| rl_data_ready | blocked | #43最大1h/逐次仮説、#46過去のみの日程とtrain/validation/forecast範囲、連続PPO validation、根拠付きRL訓練量、予算seal | #46生成→#47 smoke→#48本学習/評価を前段完了後 |
| execution_ready | blocked | #44 source/権限/時刻/有限収集契約と#45独立執行実験契約 | 実収集・執行感度だけ |

`short_data_ready=ready`は契約上の入力gate。レビュー・マージ済み、実装検証済み、本実行可能を
同じ状態にしない。登録台帳はmerge未確認、#41/#42/#43実装/実値検査/runtime未sealを明示する。
#43は中期gateを待たず採否を判断し、候補なしなら#46〜#48はnot_planned。
#44/#45や#49連続入力回復の完了も短期gateへ加えない。

#46は選定1hについて、過去だけの更新日、inner train/validation、forecast適用範囲、外側RL train、
PPOの連続60decision以上のvalidation、訓練量（独立日付数・ブロック長・episode開始候補数と最低量）
を初回fit前に固定する。v2の2008年開始・187更新・561fitは**旧計画値**で、短期の必要条件ではない。
新しいU×3の実fit数は#46で確定し、まず561以内。上限変更は明示改訂・レビューが必要。
生成前のcalendar/日程/予算gateと、生成後の実forecast finite/as-of検証を分ける。
生成済みbundleを#46開始の必要条件にして循環依存を作らない。#47/#48には生成後検証も必要。

## #41/#42/#43への実装前検査とinterface

データは元carry cache SHA-256
`723db7c935dcc27c147007728358d098243eae96998eae146db4c7df02bd081e`と
#39派生cache/calendar。v2 source sealから追跡し、現在取得値へ差し替えない。
逆数価格、14行修復、FRED vintage/公開時刻不明、60日lagの限界を引き継ぐ。
以下を実装テスト・初回fit前検査として#41で満たす。

- 全hash、9pair順、timestamp一意性/単調性、London calendar、採用split、63先行history、gapと
  label終点を照合する。価格は正かつ有限、feature/targetは有限でなければ原因・時刻・pairを保存。
  行の除外は理由を明示して実有効数を再計数し、252/60未達・未登録の入力差異は例外として停止する。
  異常値を0、補間、既定分散へ置換しない。eval予測の欠損による都合のよいintersectionは禁止。
- 入力は9pair平均の`log_return, volatility, sma20_ratio, mom24, carry_annual`固定順。
  h=1/5のtarget、train-only標準化・共分散、定数列で停止、mean-loss ridge、3alpha・tieはv2第3/4節。
  future perturbation、gap、境界purge、標準化、alpha選択をテストしてからfitする。
- #42へ渡す共分散は採用train範囲のみからv2の縮小式で作り、その行集合・範囲・hashを保存する。
  特徴量、目的変数、価格座標、horizon、費用、評価範囲を同時に探索しない。

予測bundleは新ID `issue40-short-forecast-v3`。同一の封印予測をfixed/costが消費する。
行ごとのキーは`(fold, horizon_business_days, decision_at_utc)`で、重複・不足は例外。

| 欄 | 契約 |
|---|---|
| campaign_id / forecast_bundle_id | v3と上記interface ID、bundle content SHA-256 |
| fold / horizon_business_days | 2009〜2025、1または5。h間で同じsplit |
| decision_at_utc / information_cutoff_utc | London labelをUTC化。後者は外側Y年初、decision以前 |
| predicted_basket_simple_return | 有限scalar、v2 basketのh営業日price simple return。carry/costを含まない |
| diagnostic_target_end_utc / diagnostic_target_status | h先の期待日時とavailable/unavailableの理由。末尾label不足でも予測・口座decisionは省かない |
| model_sha256 / parameter_as_of | 係数、alpha、standardizer、共分散のartifact/hashと最大使用label終点。各train/validation境界より前 |
| provenance | train/validation範囲・有効行hash・purge件数、pair/feature順、data/calendar/config/code/runtime hash |

#41は全34採用modelと両hの全予定予測、3alphaのfit台帳を#43へ渡す。予測単位の不一致、
未来label、非有限、欠落を例外とし、古いmodel/ゼロで埋めない。#42はquantity会計、literal hold、
solver・stress・terminalを合成fixtureで検証し測定IDをsealする。#43は新4構成と凍結3対照を
同じmeasurement/runtimeで全17×F0/F1/F2に再評価する。経済的採否はv2第5/6節をそのまま適用する。

## 有限予算・試行台帳

分割を変えても#41は2h×3alpha×17=**102 fit**、採用34、train+val再fit0。
#43は4新構成204＋3対照再評価153=**357口座**。対照再評価153は357の内数で追加枠ではない。
合成fixtureは#41=6 ridge fit、#42=16口座。#46の6 ridge fit、#47の4096step smoke/12合成口座は
中期用に予約し、短期へ移さない。fixture市場評価は0。

障害retryはv2を継承してcampaign共有ridge12 fit、PPO3学習、口座24（validation含む）、
同一bytes/条件のwork itemにつき1回まで。低成績retry0。Issueごとに複製せず、失敗attemptも消費し、
実行前予約→実消費→閲覧→判断を別eventとして追記する。通常fit、fixture、retry、対照再評価を
別カウンタにする。未使用枠の転用0。v2全体上限（ridge687、PPO55/14,159,872step等）は維持するが、
旧cross-fit数は実行日程や消費実績に転記しない。中期・執行・連続口座予算の自動有効化は0。

7 trialの旧行とv2停止eventは元台帳に保持し、新台帳に親hash・v3の登録commit・変更理由・
新分割・新しいgate・未実行eventを追加する。trialを追加モデルfamilyとして数えず、独立trial総数不明を保持。
登録commitは本manifest/実装/calendar成果物のcommit SHAであり、後続実行のruntime SHAではない。
初回fit/評価前にCLI/config、両repoのclean SHA、lock hash、Python/依存version、CPU、
全model/data/forecast/calendar hashを各Issueで封印する。実行中のsource変更0。

## #41〜#50への引渡し表

| Issue | 開始条件 | 具体的な引渡しと完了状態 |
|---|---|---|
| #41 | v3のレビュー・マージ＋short gate＋実装/実値検査/runtime seal | v3全17 split・元data hash・上記予測interface→102fit以内/全34model。実装・本fit未完了 |
| #42 | v3と予測interface、合成fixtureは#41と並行可 | fixed/cost/quantity会計と測定seal。実装未完了 |
| #43 | short gate＋#41/#42＋全対照同runtime、評価seal | 357口座とv2数値採否→最大1h/単純候補維持/逐次仮説/停止。中期待ちなし |
| #44 | source準備は独立、実収集前に個別有限契約 | source/権限/時刻/保存先/開始終了/予算。実収集blocked |
| #45 | #44 quote契約または別ID proxyの独立契約＋予算seal | v2次期待close proxyを含む感度。proxyも自動開始不可 |
| #46 | #43で候補＋逐次仮説。まず中期契約登録 | 選定hの日程・実fit数・PPO連続区間・訓練必要量を確定→rl gate→cross-fit。現在blocked |
| #47 | rl gate＋#46生成後検証＋#42 | 4行動PPO/greedy、fixtureと4096step smoke。現在blocked |
| #48 | rl gate＋#46/#47完了＋本実行seal | 同一forecast PPO/allocator/greedy比較。現在blocked |
| #49 | #43単純候補、RLなら#48後。連続入力/切替/予算の別seal | 一候補のみ、連続最大9＋年度reset最大153。入力不足は短期を止めない |
| #50 | 候補と#44/#45/#49の証拠 | 独立確認の設計のみ。実収集・学習・発注の自動開始0 |

実行順は`#55 → #41 + #42 → #43`。単純候補は#49へ、逐次仮説を採用した場合だけ
`#46 → #47 → #48 → #49`。別経路`#44 → #45`と候補/#49の証拠を#50へ渡す。
GitHub Issue本文の2026-09-27再編と同期した引渡しであり、旧台帳の完了/未完了を改変しない。

## 再現と状態

```bash
python3 -m docs.research.protocols.issue40.short_preflight \
  --registration docs/research/protocols/issue40/registration.json \
  --coverage docs/research/results/issue39/snapshot/coverage.json \
  --output /tmp/issue55-calendar-reproduction
python3 -m unittest discover -s tests -p 'test_issue*preflight.py'
```

出力先は未作成ディレクトリ必須、入力seal不一致は例外、calendar不足はexit 2、全fold適格はexit 0。
標準ライブラリだけで既存preflightのcalendar_blocks/audit_range/rowを再利用する最小拡張。
候補追加・train/val overlap・価格参照・自動実行機能はない。全成果物は決定論的に再現できる。

| 状態軸 | 現在 |
|---|---|
| 本Issueのcalendar/gate実装 | 実装・回帰検証済み |
| 短期分割と有限契約 | v3として確定、レビュー・マージが必要 |
| #41/#42/#43の実装・実値検査・runtime | 後続Issueで未完了 |
| 本学習/市場評価/外部取得 | 本Issue実行0、有効予算0 |
| 中期RL/実執行 | blocked、別段階で登録 |
