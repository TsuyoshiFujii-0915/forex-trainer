# 固定連続区間での3方策・費用stress評価（Issue #29後継）

**後継パネル153/153口座が完走した。F0の区間平均年率netはcanonical +8.15%、ridge −3.05%、PPO ens3 +4.97%。**
#39の出口Bで固定した17区間（8全期待bar範囲＋9部分年）を採用した結果である。
旧17通年の81不足セルは未解消のまま保全し、今回の完了を旧attemptの完了とはしない。
モデル・特徴量・範囲の再選択、新規学習、追加データ取得は0。

[事前登録](protocols/issue29-successor/registration.md)、[ADR-0043](../decisions/0043-evaluate-costs-on-the-frozen-successor-panel.md)、
[config](../../configs/research/issue29_successor_cost.json)、[全153セルCSV](results/issue29-successor/status.csv)、
[機械可読report](results/issue29-successor/report.json)、[seal](results/issue29-successor/manifest.json)。
元入力契約は[#39研究記録](23-bounded-input-recovery.md)、比較の旧attemptは[#29通年部分結果](19-full-period-cost-baselines.md)。

## 範囲・実行・検証

新campaign IDは`issue29-contiguous-cost-v1`、evidenceは`development_historical`。
利用者の2026-09-22の評価継続指示に基づき、#39で損益を使わず選んだ各fold最大coverage区間をそのまま採用した。
新しい欠損除去やcalendar短縮、結果を見た区間変更はしていない。
first decision→last markの期間は約140〜365日であり、短い区間の年率化は値を大きくし得る。
年率net/grossの17区間等重み平均は、選択済みcoverageに条件付けた記述であり、17通年の平均や連続運用CAGRではない。
[coverage](results/issue29-successor/coverage.json)に元範囲、全候補、採用範囲、除外時刻とtrain/validation不足を保存した。

実行trainer SHAは`f18a8928fd4232b5ca609a5a94d73ae3e240b0e4`、env SHAは`6024b91c0f3592611849bc231922ab60e6090aed`。
実行は2026-09-22 12:39:27〜12:40:17 UTC（同日21:39〜21:40 JST）、CPU、依存・lock・入力hashはmanifestに固定した。
登録・実装を先にcommitし、市場campaignは1回、合計32,121口座step。
margin call、実行例外、入力blockは今回いずれも0。
各方策/scenarioを100万円・flatから独立開始し、凍結PPOは各口座の観測から毎step再推論した。
末尾barはmarkのみで、仮想清算や異なる区間の口座連結は行っていない。

既存`run_fold_scenarios`、`account_trace`、`compare_sensitivity`、`summarize_campaign`、`paired_evidence`を再利用した。
旧通年validatorやgeneric comparison、約定・報酬・費用会計は変更していない。
保存後に315 artifactのhash、全step会計とpair値、同scenario内の3方策の全時刻・pair・runtime一致、
全102件の費用感度、reportの集計・区間・LOOを再検証した。
全テスト478件通過。実モデルを初期化した合成fixtureで153口座を検証する新規テストを含む。
[検証記録](protocols/issue29-successor/postrun-verification.json)を保存した。

旧attemptで完走した72口座は、新実行でも既存traceの全数値・状態、metrics、時刻、coverage、market hash、費用が完全一致した。
新traceにのみある`market_transition_sha256`は、以前merge済みの帰属処理が追加した監査fieldであり、実市場値から再照合した。
新旧のsource/calendar/runtime identityは別に保持している。旧sealed成果物は上書きしていない。

## 主結果と費用感度

各値は17区間の年率returnの等重み平均、単位は%。grossはsigned carryを含む既存定義。
F1はspreadのみ2倍、F2はovernight markupのみ2倍。commission・signed carryは同じ。
2倍は固定感度仮定であり、実broker費用の推定ではない。

| 方策 | F0 net | F0 gross | F1 net | F2 net | F0区間実収益の平均 | F0勝ち区間 |
|---|---:|---:|---:|---:|---:|---:|
| canonical | +8.15 | +12.13 | +6.79 | +5.66 | +4.42 | 11/17 |
| ridge | −3.05 | +4.17 | −7.63 | −5.29 | −2.97 | 8/17 |
| PPO ens3 | +4.97 | +7.54 | +4.07 | +3.36 | +1.98 | 7/17 |

年率gross−netの平均差はcanonical 3.97、ridge 7.22、PPO 2.57 percentage points。
初期100万円に対する区間累積費用率の平均はそれぞれ2.95%、5.65%、1.95%であり、上記年率差とは別指標。

| 方策 | F1−F0 net（pp） | F2−F0 net（pp） | target turnover平均 | actual notional平均（百万円/区間） |
|---|---:|---:|---:|---:|
| canonical | −1.37 | −2.50 | 262.49 | 275.69 |
| ridge | −4.59 | −2.24 | 1,042.54 | 1,025.81 |
| PPO ens3 | −0.90 | −1.61 | 183.16 | 183.11 |

ridgeのtarget turnoverはcanonicalの約4倍で、spread stressによる低下も大きい。
一方、F0 gross自体もcanonicalより低く、「費用だけを直せば優位になる」とは判断できない。
PPOのF1/F2対F0のaction変更は全decisionで0。現行assetsは残高額そのものを含まず、費用差が
action差を必ず生む構造ではない。独立reset・再推論を実行しており、旧netから費用差を引く代用ではない。
建玉額、actual notional、累積費用、最終equityは口座ごとに変化している。

F0のprice PnL / signed carry / spread / markupの区間平均JPYは、canonicalが
73,854 / −117 / 10,235 / 19,290、ridgeが29,163 / −2,370 / 38,057 / 18,465、PPOが
35,565 / +3,728 / 6,897 / 12,594。commissionは0。これらは不等期間の口座別金額の平均であり年率寄与ではない。

## 不確実性・era・弱い区間

10,000 IID fold bootstrap、3-fold circular moving-block、seed16、95%区間、全17 LOOを固定した。
日次行やensemble seedを独立市場標本として数えていない。欠損によるcoverage選択の不確実性はこのbootstrapでは解消されない。

| 方策 | F0 net平均 | IID 95%区間 | moving-block 95%区間 | 全1区間除外平均の最小値 |
|---|---:|---:|---:|---:|
| canonical | +8.15% | [+1.79%, +15.77%] | [+2.64%, +13.92%] | +5.38% |
| ridge | −3.05% | [−7.55%, +1.18%] | [−9.04%, +2.01%] | −3.78% |
| PPO ens3 | +4.97% | [−5.55%, +16.19%] | [−3.28%, +12.68%] | +2.39% |

canonicalのF2平均は正だが、IID区間は[−0.56%, +13.09%]とゼロをまたぐ。
PPOはgrossでもIID [−3.32%, +19.10%] / moving-block [−0.97%, +15.47%]でゼロをまたぐ。
単一fold除外でPPO平均が正でも、既存の上位3fold除外診断では平均−2.94%となり、利益集中は残る。
canonicalでは同じ上位3fold除外診断の平均は+3.00%。最大絶対寄与は2011で27.72%。
その2011区間は実収益+21.60%に対して年率+52.58%であり、部分年の年率化とcoverageの限界を併せて読む必要がある。

F0のcanonical−ridgeの年率net対応差は+11.20pp、IID [+5.25,+17.85]、moving-block [+7.55,+14.74]。
gross対応差も+7.96pp、IID [+1.71,+14.90]で正だが、これは固定後継範囲内の記述である。
canonical−PPOのnet差は+3.18pp、IID [−7.80,+12.92]、moving-block [−4.28,+10.63]で、優劣を確定できない。
ridge−PPOのnet差は−8.02ppで、IIDはゼロをまたぎ、moving-blockでは負。
全比較・era差・最大fold寄与・LOOはreportを正本とする。

| 方策 | 2009–2018 net / gross平均 | 2019–2025 net / gross平均 | mean / worst MDD | net volatility平均 |
|---|---:|---:|---:|---:|
| canonical | +7.52 / +11.44% | +9.06 / +13.12% | 9.95 / 22.22% | 12.92% |
| ridge | −3.09 / +4.25% | −2.98 / +4.06% | 11.86 / 23.77% | 11.75% |
| PPO ens3 | +5.35 / +8.19% | +4.43 / +6.62% | 17.87 / 43.47% | 20.80% |

volatilityは各区間の口座net log-returnの母標準偏差を実測step頻度で年率化した値。
F0のmean gross / net target exposureはcanonical 3.20 / 0、ridge 3.20 / 0、PPO 2.13 / −0.065。
リスクの差を事後的にサイズ調整して比較を有利にしていない。

弱い区間としてcanonical 2018は年率net−14.49%、ridge 2017は−21.41%、ridge 2016は−21.25%。
PPOは2009 −27.60%、2015 −25.34%、2017 −21.33%。PPO最大MDD43.47%は2012区間で、
同区間の年率net−9.94%だけでは見えない大きな途中損失がある。
canonicalのworst MDDは2020の22.22%、ridgeは2016の23.77%。

## 年初追加と旧measurement

#16のlegacy全51行は[別CSV](results/issue29-successor/legacy_reference.csv)へ保存した。
新F0のうちlegacy first decisionより前の観測は各方策で合計784step。
`descriptive_period_changes`にそのnet/price/carry/費用と、新旧年率差を別々に記録した。
後継開始が遅い区間では新規prefixが0でも、元年初が回復した意味にはならない。
期間変更・初期flat reset・末尾mark・年率化分母が絡むため、新旧差を年初追加の因果効果と解釈しない。
旧#16や旧#29の部分結果と新パネルを混ぜたpaired検定・採否は行っていない。

## 次の設計への判断と#30への引渡し

- **canonical:** 固定した開発仮定下の参照基準として維持する。平均netはF0/F1/F2とも正だが、
  coverage選択、事後的価格修復、vintage不明、同close仮定、F2の不確実性は残る。独立収益性を認定しない。
- **ridge:** 現在の固定mapには正のnet基準線がない。高い回転・spread負担とgross自体の劣後を分けて扱う。
  次期学習では予測精度だけでなく費用を考慮した配分・取引抑制を検討する根拠になるが、新モデルや閾値を今回は探索しない。
- **PPO:** 事前の平均値ベースの分類は「現行仮定下で検討余地あり」だが、net/grossの不確実性とtail riskが大きい。
  canonicalを置き換える根拠や、独立した学習の追加価値は未確定。#30で共通方向・相対配分・carryへの帰属を確認する。

[帰属引渡し](results/issue29-successor/attribution-handoff.json)に新しい親snapshotと51件のF0口座を明示した。
各foldの`F0-{canonical,ridge,ppo_ens3}.json.gz`がstep trace、対応する`-pairs.csv.gz`がpair会計。
#30はこの親sealと固定後継範囲を明示した新しい契約で進める必要がある。
旧#30の85セル契約へ新結果を継ぎ足さない。2帰属対照の評価は本作業では行っていない。
元#16の判定と旧#29の81入力不足は保持する。ここで完了したのは新しい153口座の後継実験である。

## 再現・検証

追加推論を行わない保存結果の検証:

```bash
uv run forex-successor-cost-campaign verify --output docs/research/results/issue29-successor
```

同じ凍結sourceを使い別attemptとして再実行する場合のコマンド（今回実行した市場パネルは1回）:

```bash
uv run forex-successor-cost-campaign run \
  --config configs/research/issue29_successor_cost.json \
  --output runs/issue29-successor-reproduction
```

出力先が存在すれば拒否する。元raw/modelは保存された同一hashのlocal data/runsが必要で、公開repoに追加していない。
exit 0は後継153セル完了、exit 2は後継パネル未完了。旧通年attemptの完了とは独立したstatusである。
