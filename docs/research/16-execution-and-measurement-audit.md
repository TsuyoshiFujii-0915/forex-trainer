# 実効評価期間・約定・取引費用と実運用条件の監査（Issue #22）

監査日: **2026-09-10**。監査・仕様化は完了、通年評価器・broker接続・paper運用は未実装/未開始。
現行の17 fold・3方策は同じ実効decision集合で比較されているが、年初からの全期間運用ではない。
日足closeを見て同じcloseで約定する仮定と、実際にその価格で発注できることは別である。
費用・金利・修復データには実口座へ移す前に追加確認が必要な点がある。
#19/#21の収益分解・回転原因分析は参照に留め、#16の`not successfully translated to portfolio alpha`を維持する。

成果物:

- [fold別期間台帳](results/issue22/fold_periods.csv)、[decision前に除外される全1,071 bar](results/issue22/excluded_bars.csv)、[provenance](results/issue22/provenance.json)
- [価格修復差分33セル](results/issue22/price_repairs.csv)、[data lineage監査](results/issue22/data_lineage.json)
- [固定コード・公開一次資料の出典台帳](protocols/issue22/sources.md)（以下のT/E/A/W IDを参照）
- [通年評価・前向き計測・受入仕様](protocols/issue22/full-period-spec.md)、[ADR-0035案](../decisions/0035-separate-observation-history-from-measurement-range.md)

## 対象と再現手順

#16の明示sourceを読み、全artifact SHA-256、17 fold集合、carry parquet SHA-256を検証した。
raw cacheから現行file providerのinclusive範囲を再取得し、3方策すべてのordered decision/target列を
全件照合した。各foldのstep数・metrics開始終了・net/gross年率化も照合した（51 policy-fold）。
推論、学習、会計replay、bootstrapを新たに実行していない。model/configは親sealのidentityを継承し、
今回全51 modelのbytesを再検証したとは主張しない。会計・modelの照合はA19/A21を参照する。

```console
uv run python -m forex_trainer.execution_audit \
  --source-dir docs/research/results/issue16 \
  --source-provenance-sha256 d2db580d5f11ec7ca341af04549c3a09e7ddb6777c43562b1117a24ae8a57db3 \
  --data data/jpy_9pairs_1d_2003_carry.parquet \
  --output-dir runs/issue22_period_audit_reproduction
uv run pytest tests/test_execution_audit.py -q
```

出力先は新規directoryを指定する。暗黙latest、欠損intersection、既存成果物の上書きは拒否する。
再生成した2 CSVは公開済み台帳とbyte比較できる。監査provenanceは実行Gitとmodule hashを記録するので
別revisionでの再実行では異なる。元評価Git/依存と監査Gitを区別する。
CSVの数は方策当たりであり、3方策や9 pairを独立な市場標本として加算しない。

## 17 foldの実効期間

宣言範囲は各年1月1日〜翌年1月1日。**file providerは終了日を含む**（E3）。
raw先頭32本がfeature warmup、次の31本がdecision前の観測履歴で、raw index63のdecision自身を
含めて32-row windowになる。したがって64本を除外するのではなく、decision前は63本。
全fold共通で年率化の分母は`(last_target − first_decision).total_seconds() / (365.25 × 86400)`。
net/grossは累積log returnをこの年数で割ってexpm1、Sharpe倍率は`√(decision数 / 経過年数)`（T1）。
曜日数×252、宣言した1年、最初のtargetからの経過時間で代用しない。

以下は読みやすさのため日付のみ。正確なEurope/London offset、秒、年数、観測window開始、
raw開始終了、各除外barはCSVに保存した。DSTによる1時間差も計算に含む。

| Fold | Raw本数 | Decision開始→終了 | Target開始→終了 | Decision数 | Equity観測数 | 経過日数 |
|---|---:|---|---|---:|---:|---:|
| 2009 | 259 | 2009-04-03 → 2009-12-31 | 2009-04-06 → 2010-01-01 | 195 | 196 | 273.041667 |
| 2010 | 261 | 2010-03-31 → 2010-12-30 | 2010-04-01 → 2010-12-31 | 197 | 198 | 275.041667 |
| 2011 | 259 | 2011-03-31 → 2011-12-29 | 2011-04-01 → 2011-12-30 | 195 | 196 | 274.041667 |
| 2012 | 261 | 2012-03-29 → 2012-12-31 | 2012-03-30 → 2013-01-01 | 197 | 198 | 278.041667 |
| 2013 | 260 | 2013-04-01 → 2013-12-31 | 2013-04-02 → 2014-01-01 | 196 | 197 | 275.041667 |
| 2014 | 262 | 2014-03-31 → 2014-12-31 | 2014-04-01 → 2015-01-01 | 198 | 199 | 276.041667 |
| 2015 | 262 | 2015-03-31 → 2015-12-31 | 2015-04-01 → 2016-01-01 | 198 | 199 | 276.041667 |
| 2016 | 261 | 2016-03-30 → 2016-12-29 | 2016-03-31 → 2016-12-30 | 197 | 198 | 275.041667 |
| 2017 | 259 | 2017-03-30 → 2017-12-29 | 2017-03-31 → 2018-01-01 | 195 | 196 | 277.041667 |
| 2018 | 262 | 2018-03-29 → 2018-12-31 | 2018-03-30 → 2019-01-01 | 198 | 199 | 278.041667 |
| 2019 | 261 | 2019-03-29 → 2019-12-31 | 2019-04-01 → 2020-01-01 | 197 | 198 | 278.000000 |
| 2020 | 263 | 2020-03-30 → 2020-12-31 | 2020-03-31 → 2021-01-01 | 199 | 200 | 277.041667 |
| 2021 | 261 | 2021-03-31 → 2021-12-30 | 2021-04-01 → 2021-12-31 | 197 | 198 | 275.041667 |
| 2022 | 260 | 2022-03-31 → 2022-12-29 | 2022-04-01 → 2022-12-30 | 196 | 197 | 274.041667 |
| 2023 | 261 | 2023-03-30 → 2023-12-29 | 2023-03-31 → 2024-01-01 | 197 | 198 | 277.041667 |
| 2024 | 262 | 2024-03-28 → 2024-12-30 | 2024-03-29 → 2024-12-31 | 198 | 199 | 278.000000 |
| 2025 | 257 | 2025-04-01 → 2025-12-30 | 2025-04-02 → 2025-12-31 | 193 | 194 | 274.041667 |

各方策の合計は3,343 decision。raw最終行はtarget用でdecisionではない。
10 foldに翌年1月1日のtargetが1本ずつある。翌foldは初期portfolioをresetして年内warmupを行うので、
同じ損益transitionを重複計上しているわけではないが、暦年の半開区間とは違う。
元データに1月1日やholiday labelがあっても、その日に対象口座で取引可能だった証明にはならない。
また2024 fold末尾に2025-01-01は存在しない。欠損と正式休場の内訳は元レスポンスがなく未確認である。

## 観測→decision→約定→損益の時刻対応

`b[t]`はcacheのbar label、`p[t]`はそのClose、`E[t]`はrebalance前equity、`e−[t]`は前回から
mark済みの保有exposure。下表はE1/E2/T2/T5から確認した実装であり、wall-clock可用性の保証ではない。

| 段階 | 利用する時刻・値 | 現行の意味と実行上の条件 |
|---|---|---|
| Raw OHLCV | bar tのOpen/High/Low/Close/Volume、CarryAnnual[t] | providerが全列を返す。日足OHLC完成時刻・配信時刻はcacheにない |
| Market observation | t−31〜tの8特徴量。log return・volatility32・SMA20比・mom24はClose[t]まで、xz/xrは同時刻9 pair、carryとxz_carryはtの値 | 生のOHLC全列を直接観測する構成ではない。現在の8特徴量は価格部分にCloseを使う。normalize=false。特徴量計算自体は因果的だが、修復/vintageの因果性は別 |
| Assets observation | 前stepのtarget weights/leverageとprice PnL比 | reset時はゼロ。mark済みexposureそのものとは異なる |
| Decision | observationの最終行tに基づくscore/action、k=1 | code上の時刻はb[t]。予測対象は次barのcross-sectional relative log return |
| Rebalance / fill仮定 | target `e[t]=w[t]×leverage[t]×E[t]`、相対的にp[t]で全量調整 | OHLCのClose[t]を見終わった後、同じClose[t]に遅延・slippageなしで約定する仮定。具体的なorder book/約定イベントはない |
| 価格PnL | `Σ e[t]×(p[t+1]/p[t]−1)`、exposureは`e[t]×relative`にmark | t→t+1のclose-to-close。bar内部のhigh/low、margin breach、gap中の約定は追わない |
| 保有markup | `Σabs(e[t])×0.00002×elapsed_days` | **価格適用前**target exposure、elapsed_daysはlabel間の実経過秒/86400 |
| Signed financing | `Σ (e[t]×relative)×(−CarryAnnual[t])×elapsed_days/365` | **価格適用後**exposure、rateはdecision t。JPY/XXX longはcounter金利が高いと支払う |
| Reward / target | b[t+1]のequityとcostを記録、次観測へ進む | net=`log(E[t+1]/E[t])`、gross=`log((E[t+1]+cost[t])/E[t])`。grossにsigned carryを含む |

2009の最初の例は観測最終label `2009-04-03T00:00:00+01:00`、同closeでrebalance、
最初のtarget `2009-04-06T00:00:00+01:00`、3日分のmarkup/carry。
labelがLondon 00:00であることから、OHLCがその00:00に取得可能だったとは断定できない。
公開後の次quoteでのshadow記録・執行を検証するまで、同close結果を実運用可能な損益とは呼ばない。

## 現行モデルと実運用条件の監査表

「確認」はcode/config/保存値、「例示」は別broker公式仕様、「未決」は対象仕様・観測証拠不足。

| 項目 | 現行モデル | 実運用条件・不足情報 | 差の影響 | 根拠・状態 |
|---|---|---|---|---|
| 評価期間 | 年内63本のdecision前履歴、終了日inclusive | 指定開始以前の観測履歴、営業calendar、開始/終了quote | 年初約3か月を含まない。年を跨ぐtargetもある | T1/T2/E1/E3、期間CSV。確認 |
| 約定 | same-close、即時全量、直接target allocation | 公開/推論/通信/受付/fill時刻、bid/ask深さ、最小lot、部分約定 | 有利不利の幅は日足だけでは測れない。執行可能性未確認 | E1/E2/E4。未決 |
| Commission | 実売買notional×0.0 | 対象口座のcommission、最低手数料、数量tier | 0はconfig仮定。実口座の無手数料を証明しない | A16 resolved config/E2。確認/対象未決 |
| Spread | 下表の固定比率×実notional | 時刻・数量・direction別bid/ask、口座方式、pips→JPY変換 | 固定rateは動的spread・流動性を表現しない。quoted spreadのhalf/fullとの対応も要確認 | A16/E2。確認/未決 |
| 実notional | `abs(e_target − e_current_marked)`、JPY建て | instrument units、JPY換算quote、netting/hedging契約 | target-weight turnoverやmembershipと同じではない。同targetでも価格/equity変動後はrebalanceする | E2/A21。確認、#21を参照 |
| Signed carry | FRED counter−JPY差、long/short対称の符号反転、mark後exposure、/365 | brokerのlong/short各swap、tom-next、発効/請求時刻、換算 | 3M・月平均とovernight実swapの満期/頻度/信用・業者調整が異なる。乖離額は未測定 | T3/E2/W1/W2。近似と確認 |
| Markup | 0.00002/day×mark前gross×経過日数 | 口座のswap markup、別建commissionとの重複 | gross exposure当たり年単純換算0.73%。signed carryとは別費用。全costを売買回転に帰属させない | E2/ADR-0034/A21。確認 |
| 週末/休日 | bar間の暦時間に線形課金、DSTなら端数日 | rollover cutoff、value date、複数日課金、休日calendar | 週末3日分を金曜→月曜に配るモデルと、水曜等の保有への課金はポジションが変われば一致しない | E1/W2。例示、対象未決 |
| 日足価格 | JPY/XXXへ逆数変換、High/Low入替、London label | session定義、mid/bid/ask、OHLC確定、provider改訂履歴 | Close labelとavailable_at不明。逆quoteでのallocationを実際の商品数量へ写像する必要 | E4/W3、data lineage。確認/未決 |
| 異常値修復 | raw→cleanの14 symbol-date / 33 OHLCセル、前後近傍を使う | 当時の訂正配信・元quoteとの照合 | hindsight修復が特徴量・学習に影響し得る。損益影響の因果量は未計測。前向き入力に適用不可 | T4、price_repairs.csv。確認 |
| 金利可用性 | documented lag60日、現在取得系列のffill/shift | source release/vintage、当時のavailability、staleness基準 | lagだけでは改訂由来lookaheadを排除した証明にならない | T3/W1、下節。証拠不足 |
| Margin/終了 | max gross5、初期100万円、equityが初期20%以下でstep末margin call、終端清算なし | 口座margin、強制決済・intrabar評価、清算spread | 現実の強制決済/損失制限と一致するとは限らない | A16/E1。確認/対象未決 |
| Slippage/latency等 | 未モデル化 | quote→fill差、通信断、拒否、最低数量、rounding、価格impact | 経済性・exposure・turnoverを変え得る。方向/大きさ未確定 | E1/E2、口座観測なし |

固定spreadの単位は**取引notionalに対する片道課金比率**。bid/ask価格を生成しているわけではなく、
無変動でentry→exitすると同じnotionalに2回課金される。半spreadだと推測して半減しない。

| Pair | Rate | notional比bps |
|---|---:|---:|
| JPY/USD | 0.000015 | 0.15 |
| JPY/EUR | 0.000025 | 0.25 |
| JPY/GBP | 0.000030 | 0.30 |
| JPY/AUD | 0.000030 | 0.30 |
| JPY/CHF | 0.000035 | 0.35 |
| JPY/CAD | 0.000035 | 0.35 |
| JPY/NZD | 0.000040 | 0.40 |
| JPY/NOK | 0.000060 | 0.60 |
| JPY/SEK | 0.000060 | 0.60 |

## FRED lag・日足timestamp・修復lineage

T3は`fredgraph.csv?id=...`の現在レスポンスを取得し、数値化不能/欠損をdrop、日次calendarへ
ffill、60日shift、cache indexのtimezoneを外したlocal dateへ整列する（60日はREADMEの再現手順）。
残った欠損は例外になるが、値の長期stalenessやvintageは保存・検証しない。
USDは月平均effective rate、EURは日次deposit facility、残りとJPYは月次3M interbankである。
各定義は[公式系列出典](protocols/issue22/sources.md#fred系列の公式定義)を参照。

parquetにはtimeframe/宣言start/endとpandas metadataがあるが、FRED rawレスポンス・取得日時・
lag引数・release時刻・vintageはない。したがって既存carryのhash同一性は確認できても、元の
FRED取得を完全再現できない。現在のFREDを取り直して異なる値を見つけても、過去の誤りとは
断定できない。逆に60日lagだけで当時公表済みだったとも証明できない（W1）。
point-in-time検証には当時のvintage/公表時刻を取得・比較し、取得不能箇所は明示して判断を止める。

対象は**yfinance由来の9-pair日足**であり、Dukascopy時間足キャッシュ（ADR-0006）ではない。
3つのparquetは全て5,292行、実範囲2005-07-19〜2025-12-31、timezone=Europe/London。
宣言metadataは2003-01-01〜2026-01-01でも2003年の実データがあるわけではない。
providerは欠損OHLC行を除去して全pairをinner joinしindexを保持する（E4）。
元のpair別レスポンスがないため、失われたbarと休日の内訳や配信時刻は再構築できない。

raw/cleanのindexと列は同一。cleanの全OHLCVはcarry追加後も値まで同一。
既存修復コマンドのthreshold=0.08、reversal tolerance=0.04、expected repairs=14でrawから
一時出力を作り、clean parquetとDataFrame全値一致を確認した。price_repairs.csvは両者のセル差分。
検出の反転判定と近傍median、補間は未来側価格も参照するため、これは事後の研究データ修復である。
修復自体が間違いと断定せず、元の価格feedでの裏付けと修復値の利用可能時刻が未確認とする。

修復再確認例（既存入力は上書きしない）:

```console
uv run forex-clean-spikes --input data/jpy_9pairs_1d_2003.parquet \
  --output runs/issue22_clean_reproduction.parquet \
  --residual-threshold 0.08 --reversal-tolerance 0.04 --expected-repairs 14
uv run python -c 'import pandas as pd; pd.testing.assert_frame_equal(pd.read_parquet("runs/issue22_clean_reproduction.parquet"), pd.read_parquet("data/jpy_9pairs_1d_2003_clean.parquet"), check_exact=True)'
```

この確認は保存raw→cleanの再現であり、Yahoo/FREDの過去取得、当時の実価格やvintageの証明ではない。

## 基準条件と保守的stressの設計案

**未実行の感度分析案**。評価結果を見て採用するcost値を選ばない。基準を引き下げて利益を作らない。
実測＝保存quote/請求値、仮定＝研究config/登録値、取得不能＝根拠がない項目として別列に残す。
どのscenarioも3方策のmodelは固定する。口座経路が変わるためcost差を旧netから引くだけで済ませず、
同じscenarioのenv経路を再評価する。PPO assets依存も含め、適用actionを再記録する。

| Scenario | 固定する条件案 | 根拠・状態 | 必要な実装/観測 |
|---|---|---|---|
| H0 歴史基準 | A16のcommission0、spread上表、markup2e-5/day、signed carry、same close、旧期間 | 保存config。**仮定**であり実口座費用ではない | #16既存結果を参照。本Issueでは再実行しない |
| H1 売買費用stress | H0の各spreadを2倍、他は同一 | 動的spread未観測に対する単一の感度仮定。2倍を実測上限/95%点とは呼ばない | 別campaignで全3対照をreplay。H0より安くしない |
| H2 保有費用stress | H0のmarkupを4e-5/dayへ2倍、signed FRED差はそのまま | 売買費用と保有費用を分離するA21に基づく単一仮定。broker quoteの代用ではない | H1と別の一変数scenarioで比較、grossとnetを分解 |
| H3 約定stress | bar tの公開済み入力でscore、次sessionで入力公開・記録完了後の最初の取引可能bid/askで約定 | same-close仮定を外す時間感度。**遅らせれば必ず収益が下がるとは限らない** | 日足labelだけでは未取得。前建玉のgap PnLと新建玉を分ける別execution ADRが先決 |
| B0 対象口座基準 | 指定口座の公表commissionと観測bid/ask、long/short swap・rollover・数量/換算 | **対象未指定、未決**。H0を口座実測と呼ばない | 法域/口座/商品を確定、source URL/有効日/hashを保存 |
| B1 対象口座stress | 固定した事前calibration窓のpair/session別95% adverse quote-to-fill差・spread、保有方向別の不利なfinancing、95% latencyを適用 | 95%は事前登録する保守的設計仮定。値は未取得、tail上限保証ではない | calibration期間・最低件数・分位法・片道換算を評価開封前に固定。件数不足は取得不能として停止 |

H1/H2の2倍は「適正broker費用」の推定ではない。H1/H2/H3を合成したscenarioを後から選ばない。
B1はB0よりcommission/spread/debitを小さくせず、creditを大きくしない。
bid/askのfill価格でspreadを表現する場合はH0型spread debitを重ねない。broker swapにmarkupが
内包される場合も同様に二重課金を避け、公式仕様から会計対応を決めてから実装する。
観測slippageが負のケースもraw記録を保持し、都合のよい側だけで基準を推定しない。

## 次の作業と完了範囲

通年評価は[具体的仕様](protocols/issue22/full-period-spec.md)に従う別実装とし、
観測用履歴・休日・履歴不足・初期portfolio・fold境界・正規化と特徴量・全3対照の再評価・
新しいprovenance・利用者の振る舞いテストを必須にする。accepted ADR、過去の成果物は変更しない。
#20には同文書の未決項目表と入力snapshot/費用内訳/時刻の計測要件を渡す。

既知のbroker情報不足に加え、bar可用時刻、vintage、異常値修復lineageの当時性、
実notional→商品数量の対応、rollover仕様、quote/fill記録、shadow adapterの検証がpaper開始前の条件。
これらは未解決と明示した監査結果であり、運用を開始できたという結論ではない。
#20の年内warmup契約を自動更新せず、通年契約を採用する場合は開封前に別revisionを登録する。
