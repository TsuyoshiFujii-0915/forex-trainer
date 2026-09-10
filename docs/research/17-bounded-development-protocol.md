# 有限な開発実験と独立確認の分離（Issue #27）

契約ID: `issue27-development-v1`。登録日: 2026-09-10。
状態: **短期開発の契約登録。実験・外部収集は本Issueでは実行しない**。
本PRのmerge commitを登録revisionとする。子campaignは結果生成・閲覧前に本書、
[実行manifest](protocols/issue27/execution-manifest.md)、各具体仕様のraw bytes SHA-256と
登録commitを保存する。既知の#15〜#22結果を踏まえた開発であり、未使用データへの事前登録とは呼ばない。
決定は[ADR-0036](../decisions/0036-register-bounded-development-separately-from-confirmation.md)。

## 目的と証拠区分

今回答えるのは、年初履歴を分離した通年成績、PPOの共通方向/相対配分の利益源泉、
費用・約定仮定を変えた場合の検討余地、将来到着する入力・判断・quoteを当時記録できるか、の4点。
現行方策の昇格ではなく、次の低容量予測・コスト対応実験の測定と仮説選別を目的とする。
#15の`established learnable`、#16の`not successfully translated to portfolio alpha`を維持する。
#19/#21のgross・費用の不確実性も保持し、cost-limited、tradable、canonical置換、RL追加価値を新たに認定しない。

| evidence_class | 利用・閲覧 | 言えることと制限 |
|---|---|---|
| `development_historical` | 使用済み履歴で固定仮説を評価し、結果を次の設計に使用可 | #28/#29/#30と#31のhistorical replay。仮定付きの開発証拠。新しいmeasurementでも独立確認にならない |
| `development_shadow` | #32で到着時の入力・判断・quoteを保存。損益閲覧可、閲覧と設計変更を台帳へ追記 | 遅延・欠損・quote費用を調べる。開始時から開発区分であり、結果未閲覧でも後から独立確認へ昇格しない |
| `independent_confirmation` | 設計固定後の未使用期間、独立した事前統計設計、開始・終了・開封と閲覧分離が必要 | 別契約の対象範囲だけで認定判断。#20は未開始であり、短いshadowや2027候補の記述的結果だけではこの資格を満たさない |

[既存期間台帳](protocols/issue20/period-and-trial-ledgers.md)を保持する。
2009〜2025の17foldはdevelopment、2007〜2008と2026H1は過去確認使用済み。
年初warmupで従来測らなかった部分もdevelopmentのまま。2026H2は未使用と確認できておらず、
暦上新しいことだけで確認候補に戻さない。学習・validation履歴も未使用ではない。
fixtureは実装検証用であり、historical市場成績や実前向き観測の代用にしない。

## 旧#20契約との適用範囲

旧[issue20-forward-v1](14-research-and-forward-protocol.md)と
[運用契約](protocols/issue20/forward-operations.md)の本文・hash・日時は変更しない。
旧ノートの「今後」「次期」はその契約範囲を指す。今回の開発に適用する差分は以下に限定する。

| 対象 | 保持する旧契約 | 新開発契約での扱い |
|---|---|---|
| #20 B1〜B4の入口 | 凍結replicationと独立確認後の分岐、そこでの探索予算0 | #28〜#32の開発をB1の独立確認待ちにしない。次の学習も別campaignの事前固定が必要 |
| gross・net・置換・RL採否表 | 旧確認計画の厳格な条件と過去分類を保持 | 開発着手に独立grossの95%区間下限>0を要求しない。下記は仮説の次手を選ぶ表であり収益性の合格表ではない |
| gross対応差とrisk条件 | #20のgross対応差>0、exposure下限/比率、MDD基準は同契約内で保持 | 将来の費用改善にはgross減少を上回る費用削減によるnet改善を認め得る。共通リスク制約とtrain-only調整を別campaignで固定し、旧条件を黙って緩めて旧方策を採用しない |
| 期間・証拠・identity | 使用済み期間と既存sealed成果物、current/legacyの区別 | 完全に継承。新測定は新出力・親hashを持ち、再封印・上書きしない |
| 評価期間 | #20は年内warmup、decision index63 | #28でADR-0035案をレビュー・採用した通年測定へ。約定変更は#31の別ADR/ID |
| 日程 | 固定期限2026-12-15T00:00:00Z、2027-01-01〜2028-01-01取得候補、開封2028-02-01T00:00:00Z以降。正確なfirst decisionは未確定 | #32は独自の開始/終了を初回decision前に登録。旧確認の開始・延長・遡及開始・開封変更をしない |
| 記録 | ADR-0032のdecision/outcome/incident/correction、追記・hash照合 | 再利用。必要なschema拡張は新version。開発の閲覧権限を独立確認の権限保証と呼ばない |

## 短期予算と測定条件

予算はこの一巡に対する上限。新規モデルfit、architecture、feature、horizon、map、
seed/checkpoint選抜、HPOはいずれも0。低容量予測モデル2候補、決定論的コスト対応方策、
中期の逐次RLは本バッチでは実装しない。各子Issueの実行上限と入力/出力は
[実行manifest](protocols/issue27/execution-manifest.md)で固定する。

既存3方策はcanonical mom24 reversal（昇順先頭2に+0.8、末尾2に−0.8）、
凍結ridge固定map（score昇順末尾2に+0.8、先頭2に−0.8）、
current-provenance direct PPO longf ens3（validation-best、全seed42/43/44）。
固定mapはgross3.2、k=1、既存pair順・tie処理を維持する。legacy longfへ切り替えない。
親identityとsource所在はmanifestの参照を用い、モデル不足はpath/hash付きの停止理由にする。

| #29 scenario | 基準から変える値 | policy-fold上限 |
|---|---|---|
| F0 | #28通年・同close研究基準、#16と同じspread/commission/signed carry/overnight markup | 17×3 = 51 |
| F1 | F0の全pair spreadのみ2倍 | 51 |
| F2 | F0のovernight markupのみ2倍 | 51 |

合計153評価。commissionとsigned carryは不変。F1+F2合成、倍率追加・引下げ、最良scenario選択は禁止。
2倍は感度仮定であり、実broker費用の推定・上限ではない。各scenarioを別口座経路で実行し、
PPOはそのequity/assetsで再推論する。旧netから費用差を引くだけで代用しない。

#30だけにtrain-derived constant allocationとcommon-only projected policyの2記述的対照を許可する。
前者は各foldの凍結PPOをtrain区間にreplayした実効weightの時間平均を固定し、
evalから符号・サイズを決めない。後者は同じPPO提案をpair平均weightへ投影する。
独立した口座経路でPPOのassetsも再推論し、共通cap・投影/初期状態は結果前に固定する。
relative-onlyや最適ブレンドなど第三の対照を追加しない。これは帰属診断であり新規学習/HPOではない。

## 比較・会計・不確実性の共通契約

- 2009〜2025の17fold、era 2009〜2018 / 2019〜2025、CPU、9 JPY pairを固定。
  方策間の一次比較は同一scenario/measurement内。data、calendar、全decision/target、pair順、
  評価時の両repo SHA・依存を一致させ、source training provenanceは別に保持する。
- #28の通年契約はUTC半開区間`[S,E)`、`S <= decision < target < E`、first decision直前の有効raw63本。
  履歴PnL/費用0、foldごと初期100万円・建玉/資産観測0、初回entry費用1回、最終barはmarkのみ。
  予定範囲、first decision、last mark、初期待機・末尾coverage gapを併記する。
- 主指標はfoldごとの`expm1(net累積log return / 実経過年数)`の算術平均。
  年数はfirst decision→last markの実秒/365.25日。grossはsigned carryを含む既存定義。
  fold平均を連続運用CAGRと呼ばず、宣言した1年で割る値へ切り替えない。
- net/grossのcandidate−comparator対応差をfold内で計算し、17fold等重み平均を報告。
  10,000 IID fold bootstrapと3-fold circular moving-block、95%区間、seed16、
  paired比較は同じ再標本化index。全17 LOO、両era、最大絶対寄与foldと寄与率を併記する。
  日次行やensemble member seedを独立市場標本に加算しない。全探索の多重性補正済みとは主張しない。
- net/gross、price/signed carry/spread/commission/markupのJPY額と寄与、Sharpe、勝ちfold数、
  mean/worst MDD、gross/net exposure、volatility、初回entry込みtarget turnover、actual traded notionalを保存。
  annualized gross−net差とinitial-equity基準cost ratioを区別し、#19/#21の会計照合を再利用する。
  `forex-report`と既存の会計・統計を使い、汎用基盤を作り直さない。
- F1/F2−F0は同じsource/時刻で費用だけを変えた専用の感度比較。generic reportの測定一致検証は緩めない。
  旧#16との新旧期間差は別表の記述的参考であり、混在したpaired採否を作らない。
  #31ではcoverageを損益閲覧前に棚卸しして比較範囲を固定し、全体の不足範囲も出す。
  時刻/quote不一致を事後intersectionで落とさない。17fold未満のquote観測は限定的感度検証とする。

### 失敗の保存と停止

data/history/model不足、hash・pair・時刻・会計不一致は原因とorigin付き例外/incidentで停止する。
不明costの0補完、latest、再学習、欠損fold/seed除外、終了日の後ずらしは禁止。
影響しない作業は継続できるが、不足を無視した採否は作らない。

margin call等はデータ不備と区別した**戦略のterminal結果**として、発生時刻・原因・最終equity・
損失・exposure・費用・完走状態・未計測範囲を保存する。失敗foldを除く採否は禁止。
通年未完走値を通年値に偽装せず、log/年率が未定義なら理由を明示し、NaNや0で補わない。
元の153セルの状態一覧を残し、全期間対応統計が定義できなければ「判定不能」とする。
残るfoldの診断は記述的に報告できるが、成功foldだけのprimaryを作らない。
#22仕様案のearly margin callによる判定停止はこの意味で継承し、#28で受入仕様に明記する。

## 採否・停止・終了時の分岐

ここで「採用」するのは**次期campaignで検証する仮説**。既存方策の独立収益性・実口座導入の採用ではない。
下表は点推定だけで合格を出す閾値ではなく、対応差・CI・era・LOO・riskと不足証拠を揃えた判断記録を要求する。
複数の説明が残る場合は併記し、結果に合わせた新閾値で強制分類しない。

| 条件・証拠 | 判断と次の作業 | 停止/非採用の扱い |
|---|---|---|
| source/時刻/会計照合が不成立、またはterminalにより主統計が未定義 | 原因・全不足範囲を列挙し「測定/データ不足で未判断」または「戦略失敗」を区別 | 当該比較の採否停止。修復で条件が変われば新revision、旧attemptは保持 |
| F0に検討余地があり、F1/F2で悪化 | 「費用に脆弱」と根拠付きで整理。price/carry/売買/保有費用のどれが障害かを記録 | costだけが障害とはgrossの不確実性を残したまま断定しない。倍率調整で救済しない |
| #30で相対配分の寄与に支持があり、費用・riskを含め次に検証すべき点が特定できる | **横断的な予測/売買の改善へ**。低容量予測とcost-aware配分の小規模campaignを別起票 | 候補数・指標・risk・停止条件の登録までは新規学習0 |
| #30で共通方向の寄与が主要と説明され、constant/projectedとの比較で検証課題が特定できる | **共通要因のポジション制御へ**。少数の共通方向への露出制御を別campaignとして設計 | pair平均成分を純粋JPY因子と断定しない。net差だけで因果的優越性としない |
| #31でavailable_at/quote/数量対応が不足、または仮定変更の影響を分離できない | **約定/データ不足の解決へ**。必要sourceと実装済み/経済的未検証を引き渡す | 同closeの実行可能性は未認定。#29/#30のhistorical次手まで一律停止しない |
| CIが広い、eraで異なる、寄与が混在し識別不能、経済的余地の支持がない | 「判断不能」または当該仮説の非採用を正式な結果にし、不足証拠・反証条件を記録 | 自動的に新候補を追加しない。独立確認待ちだけを唯一の次手にしない |
| 全登録作業が終了、または所定source不足で実行不能範囲が確定 | 予算消化・attempt・全結果と未検証事項を閉じ、次の契約へ渡す | 同campaignの期間延長・seed追加・条件探索なし |

将来の執行改善は、grossが少し下がっても売買/保有費用削減がそれを上回りnetが改善する仮説を
検討できる。gross対応差>0を一律必須にせず、gross、carry、spread/commission、markupを分解する。
ただし共通の事前リスク制約を適用し、必要なリスク調整の推定・サイズ決定はtrain側だけで固定する。
評価成績に合わせたレバレッジ調整は禁止。現バッチは既存max leverage5・margin threshold0.2を維持し、
2記述的対照にも同じ上限制約を使うが、異なる実効riskを等riskの比較と呼ばない。
常時flatによる損失回避だけを経済的成功とせず、一時的flatは期間・頻度・exposure・turnoverと
制約上の理由を報告して区別する。全foldに一律exposure下限を新設して合理的な待機を排除しない。

次期の低容量予測・コスト対応方策は候補数、正確な候補、選択指標、train/validation規則、
risk設定、seed/fit上限、停止条件、実行仮定と未検証事項を**結果閲覧前に別campaignで固定**する。
本契約は無制限探索や実口座導入を許可しない。

## 本Issueの完了境界

新契約、旧契約対応表、予算・分岐表、子Issue実行manifest、未決情報、研究READMEの導線を納品する。
評価器・帰属分析・収集adapterは子Issueの成果物。153評価、#30最終解析、#31実データ検証、
#32実収集、長期独立確認の完了をIssue #27の完了条件に含めない。
