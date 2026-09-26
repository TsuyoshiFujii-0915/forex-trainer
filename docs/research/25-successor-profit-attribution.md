# 固定連続区間でのPPO利益源泉と2対照（Issue #30後継）

**後継85/85口座が完走。実現利益の平均寄与は共通方向が主要であり、次期の優先仮説を少数共通要因へのポジション制御とする。**
PPOの区間平均年率log寄与はcommon +5.378、relative −0.767、financing +0.328、費用 −2.415ポイント。
共通投影対照の年率netは+6.40%、元PPOは+4.97%で、対応差+1.42pp。
ただしcommon自体の区間推定はゼロをまたぎ、投影対照にも最大MDD43.33%が残る。
この結論は固定された後継区間内の会計記述と次期仮説であり、学習alpha・独立収益性・対照の因果的優越性の認定ではない。

[事前登録](protocols/issue30-successor/registration.md)、[ADR-0044](../decisions/0044-attribute-profit-on-the-frozen-successor-panel.md)、
[config](../../configs/research/issue30_successor_attribution.json)、[report](results/issue30-successor/report.json)、
[85セルCSV](results/issue30-successor/folds.csv)、[step帰属](results/issue30-successor/steps.csv.gz)、
[seal](results/issue30-successor/manifest.json)、[検証・判断記録](protocols/issue30-successor/postrun-assessment.json)。

## 範囲と実行

campaign IDは`issue30-contiguous-attribution-v1`、証拠は`development_historical`。
#39の出口Bで損益を使わずに固定した17区間（8全期待bar範囲＋9部分年）を、2026-09-22の評価継続指示に基づいて採用した。
親は[#29後継のF0全51口座](24-successor-cost-baselines.md)。F1/F2や旧legacyを混合していない。
旧[#30通年attempt](20-ppo-profit-attribution.md)の40/85完走・45入力不足は保持する。
**今回の85口座完走を旧17通年診断の完了へ読み替えない。**

実行trainer SHA: `adb596ab6dd9217de4fbd24572c78ab90845ce18`、env SHA: `6024b91c0f3592611849bc231922ab60e6090aed`。
登録・実装をcommitした後、2026-09-22 13:07:08〜13:08:13 UTC（22:07〜22:08 JST）にCPUでcampaignを1回実行。
85評価口座で17,845 step、17 train replayで46,863 step。新規学習・再選択・追加対照・データ取得0。
全85評価・全17 train replayでmargin call・実行例外0。実行中のソース編集0。

全51元方策のtrace・metrics・時刻・市場値を親F0と完全一致で再現した。
その同runtimeの3方策と2対照に既存`require_comparable`を適用した。
保存後、193 artifactのhash、全口座のpair/step会計・帰属・coverage・runtime、train平均と期間境界、
投影/固定action、report集計、step CSVを再検証した。全486テスト通過（新規8件）。
旧attemptの8 train traceと16対照口座のtrace/metrics/時刻/coverageも完全一致。
これを独立な再現実験や新たな市場標本として数えない。

## 会計上の主分解

開始equity単位の実効post-rebalance weightで、price = common + relative、
net = price + signed financing − spread − commission − overnight markupを毎step照合した。
log寄与は全成分に同じlog1p(net)/net係数を配賦する。成分別の単独運用CAGRではない。
以下は17区間等重みの**年率log寄与×100**。commissionは全方策0。

| 方策 | common | relative | financing | spread | markup | net log |
|---|---:|---:|---:|---:|---:|---:|
| canonical | 0.000 | +10.567 | +0.018 | −1.256 | −2.338 | +6.991 |
| ridge | 0.000 | +3.890 | −0.303 | −4.835 | −2.338 | −3.587 |
| PPO ens3 | +5.378 | −0.767 | +0.328 | −0.858 | −1.557 | +2.524 |
| train constant | −0.375 | +0.089 | +0.212 | −0.005 | −0.282 | −0.361 |
| common projected | +5.603 | ≈0 | +0.330 | −0.669 | −1.188 | +4.076 |

canonical/ridgeのcommon=0はnet exposure=0の固定mapによる構造確認。
PPOと同リスクの比較を意味しない。commonはこの9pair座標系の平均であり、純粋JPY factorや独立市場因子とは断定しない。
PPOの正の平均price寄与は共通方向が担い、相対配分は平均では減益。carryは正だが主源泉ではない。
一方、relativeが常に有害、commonが安定したalpha、という推論は支持されない。

PPOの実現JPY寄与の区間平均はcommon +43,255円、relative −7,691円、financing +3,728円、
spread −6,897円、markup −12,594円、net +19,801円。これらは初期100万円の別口座・不等期間の金額平均。
累積logと各区間の経過時間もCSV/JSONに保存している。
年率simple returnの平均+4.97%と、年率log寄与の合計+2.524は異なる集約であり、相互に置き換えない。

## era、区間推定、寄与集中

| PPOのera | common | relative | financing | spread | markup | net log |
|---|---:|---:|---:|---:|---:|---:|
| 2009–2018 | +4.973 | −0.245 | −0.069 | −0.941 | −1.724 | +1.994 |
| 2019–2025 | +5.957 | −1.513 | +0.895 | −0.741 | −1.318 | +3.281 |

commonの平均寄与は両eraで正。近年はrelativeの負寄与が大きく、carryは正寄与へ変わる。
単位は上表と同じ年率log×100。

| PPO成分 | 平均 | IID 95%区間 | moving-block 95%区間 | 全1区間除外平均の範囲 |
|---|---:|---:|---:|---:|
| common | +5.378 | [−4.091, +15.247] | [−2.183, +12.156] | [+3.166, +7.274] |
| relative | −0.767 | [−1.854, +0.294] | [−1.771, +0.241] | [−1.020, −0.509] |
| financing | +0.328 | [−0.259, +0.983] | [−0.268, +1.076] | 全LOOはreport参照 |
| net log | +2.524 | [−7.589, +12.843] | [−5.166, +9.635] | [+0.302, +4.700] |

10,000 IID fold bootstrap / 3-fold circular moving-block、seed16、全17 LOO。
日次step・ensemble seed・過去attemptを独立標本に加算していない。
coverage選択の不確実性はbootstrapで解消しない。区間長は約140〜365日で、短い部分年の年率化は数値を大きくし得る。

commonの最大絶対寄与foldは2013（+40.767、絶対寄与総和の14.52%）。
正の上位3区間を除く既存診断ではcommon平均−1.480、net log平均−4.419となり、利益集中は残る。
2015のcommonは−24.962、2009は−23.207であり、共通方向は損失にも大きく寄与する。
relativeの最大絶対寄与foldは2020（−4.890）。正のfoldも8/17あり、一律に無価値とは言えない。
全成分・全方策の最大寄与foldと個別LOOはreportの`component_evidence`を正本とする。

## 2対照とリスク

constantは各foldの**元train [start,end)**の既存warmup後decisionで実効weightを平均した。
元cacheの観測済み最初の日付は2005-07-19で、要求開始2003-06-01より遅い。
[coverage](results/issue30-successor/coverage.json)に訓練履歴不足と不足平日を保存した。
旧train replayのgap横断・実UTC経過時間の資金調達を明示的に継承しており、欠損のない訓練データとは称さない。
後継評価の連続区間規則を過去訓練に遡及適用せず、補間・欠損削除・再学習もしていない。
validation/evalから配分やサイズを推定していない。この対照は訓練由来の記述であり独立alpha発見ではない。

projectedは自身のassetsを使って凍結PPOを毎decision再推論し、clip/cap後weightをpair平均に投影した。
全17区間で、初回以外の全decisionのassetsと元提案がdirect PPOと異なることをtraceで確認。
元PPOの固定action列を再生したものではない。全口座は100万円・flat、cap=5、F0費用、k=1から独立開始。
費用はそれぞれの口座の実売買から計算し、投影後のgross再拡大やリスク合わせはしていない。

以下は17区間平均。net/gross・期間netは%、MDD/volatilityも%。grossはsigned financingを含む既存定義。

| 方策 | 年率net | 年率gross | 期間net | gross / net exposure | MDD平均 / 最悪 | volatility | turnover |
|---|---:|---:|---:|---:|---:|---:|---:|
| canonical | +8.15 | +12.13 | +4.42 | 3.200 / 0 | 9.95 / 22.22 | 12.92 | 262.49 |
| ridge | −3.05 | +4.17 | −2.97 | 3.200 / 0 | 11.86 / 23.77 | 11.75 | 1,042.54 |
| PPO ens3 | +4.97 | +7.54 | +1.98 | 2.132 / −0.065 | 17.87 / 43.47 | 20.80 | 183.16 |
| train constant | −0.32 | −0.03 | −0.42 | 0.386 / −0.170 | 2.69 / 6.19 | 3.06 | 0.386 |
| common projected | +6.40 | +8.39 | +3.33 | 1.630 / −0.069 | 16.83 / 43.33 | 19.76 | 146.07 |

volatilityは#29後継metricsの母標準偏差(ddof=0)による年率net log-return。
帰属側に保存された標本標準偏差(ddof=1)とは区別する。turnoverはtarget-weight差の期間総和。
actual notionalの区間平均はPPO 183.11百万円、constant 0.876百万円、projected 146.56百万円。
constantも時価建玉のdrift再調整で実売買が発生する。符号付き平均net exposureが小さくても、時点ごとの共通方向リスクが小さいとは限らない。

| 対照−PPO | 年率net差 pp | IID 95%区間 | moving-block 95%区間 | 年率gross差 pp | IID / moving-block 95%区間 |
|---|---:|---:|---:|---:|---|
| train constant | −5.30 | [−16.60, +5.30] | [−13.13, +2.82] | −7.58 | [−19.18, +3.27] / [−15.69, +0.81] |
| common projected | +1.42 | [+0.21, +2.67] | [+0.04, +2.76] | +0.84 | [−0.42, +2.14] / [−0.52, +2.18] |

projectedのnet差は11/17区間で正、全LOO平均は+1.16〜+1.70pp、era差は+0.76 / +2.37pp。
gross差は9/17区間で正だが、両95%区間がゼロをまたぐ。
net年率平均差+1.423ppはgross差+0.844ppとgross−net dragの減少0.579ppに分かれる。
これは別々の実現口座の記述で、費用の旧成績からの比例配賦や独立した因果効果ではない。
constantとの差の区間はゼロをまたぎ、gross/volatilityの大差もあるため、PPOの学習タイミング価値を確立したとは言えない。

## 判断と次期への引渡し

**会計上の主要寄与は共通方向。独立した学習価値・将来の主要源泉は未識別。**
平均、両era、projected対照は、PPOの次期研究をpair ranking改善に固定しない根拠になる。
relativeの全体区間はゼロをまたぎ、共通投影のgross優位も未確定なので、相対配分の全面廃止を採用判断にはしない。
数値reportの`conclusion=uncertain_not_identified` / `next_learning_problem=insufficient_information`は
既存集計器の自動判断保留であり、上記の記述的解釈と次期仮説は別の[判断記録](protocols/issue30-successor/postrun-assessment.json)に根拠付きで保存した。

次期の**優先候補は少数共通要因へのポジション制御**。共通方向の符号・サイズ・回転と損失抑制を学習問題として検討する。
特にPPOとprojectedのworst MDDがともに約43%であるため、投影だけでtail riskを解決したとは言えない。
ここでいう共通要因は研究仮説であり、純粋JPY factorの識別、新featureの効果、最適因子数を得たという意味ではない。
次期学習には別の有限予算と実行前契約が必要で、本作業では学習・特徴量追加・サイズ探索を実施していない。

横断予測＋cost-aware配分は別の候補として残る。canonicalのrelative寄与は正で、ridgeは回転と費用の課題が大きい。
ただし、それらを根拠にPPOの利益源泉を横断選別と読み替えない。canonicalを基準として維持し、置換や旧#16判断変更は行わない。

残る未識別事項は、共通方向の持続的alpha、同リスクでの学習追加価値、欠損期間を含む17通年結果、
同close約定仮定、公開時刻とFRED vintage、事後的OHLC修復の情報時点である。
連続区間の独立口座を接続して運用CAGRにしたり、部分年を旧通年の完了にしたりしない。

## 再現・検証

追加推論を行わない保存結果の検証:

```bash
uv run forex-successor-attribution verify --output docs/research/results/issue30-successor
```

同じ凍結sourceで別attemptとして再実行する場合（今回の市場campaignは1回）:

```bash
uv run forex-successor-attribution run \
  --config configs/research/issue30_successor_attribution.json \
  --output runs/issue30-successor-reproduction
```

出力先が存在すれば拒否する。同一hashのlocal data/modelが必要。exit 0は後継85セル完了、exit 2は未完了。
旧通年の完了状態とは独立したstatusである。
