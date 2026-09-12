# PPO利益源泉の会計帰属と2対照（Issue #30）

**実装・部分診断済み。40/85 policy-fold完走、45/85は入力不足。17foldの最終診断は未完了。**
元F0の24口座（8fold×3方策）を帰属し、全traceとmetricsの再現を確認した。
8件のtrain replayで固定配分を作り、16対照口座を独立評価した。全train/対照ともmargin call・実行例外は0。
不足foldは2009/2011/2012/2013/2014/2017/2018/2019/2025。利用可能foldだけの平均・era集約・採否は行わない。

[実行前登録](protocols/issue30/registration.md)、[ADR-0039](../decisions/0039-attribute-profit-and-replay-two-frozen-controls.md)、
[config](../../configs/research/issue30_profit_attribution.json)、[report](results/issue30/report.json)、
[全85セルCSV](results/issue30/folds.csv)、[step帰属CSV](results/issue30/steps.csv.gz)、[seal](results/issue30/manifest.json)。

## 実行・入力の固定

実行trainer SHA: `5e2444cbf26a6938f92ce54ef2b2984d7e762fbf`。env SHA: `6024b91c0f3592611849bc231922ab60e6090aed`。
CPU、実行開始 `2026-09-11T09:29:06.162315+00:00`、終了 `2026-09-11T09:29:45.749538+00:00`。
依存version・source bytes・元config/data/model/calendarはmanifestから親#29/#15/#16へ追跡できる。
契約と実装をcommitした後に実行し、実行中のソース変更・再学習・再選択・cache補修は0。
F0だけを主入力とし、F1/F2・legacyの結果は混合していない。新旧runtimeは両方保存し、
元方策の全step（assets/action/weight/建玉/equity/費用/時刻）とmetricsが一致することを確認してから、
新runtime内の同一measurementで対照を比較した。汎用measurement検証は変更していない。

## 会計の定義と照合

weightは開始equityに対するcap適用後の実効exposure。common = sum(w) × mean(r)、
relative = sum(w × (r−mean(r)))。price = common + relativeを毎step照合する。
signed financingを足し、spread/commission/overnight markupをそれぞれ控除するとnetとなる。
全9pairの価格・carryを元市場からhash照合し、未保有pairも検証する。元sealedファイルは上書きしていない。
全成分に共通のlog1p(net)/net係数を掛け、成分和をnet logに一致させる（net=0では係数1）。
非正terminal equityはsimple帰属を残し、logはnullと原因を保存する。terminalを通年成績へ含めない。
CSV/JSONにはsimple・累積log・年率log・実現JPY寄与を区別して保存する。成分単独運用のCAGRではない。
commonはこの9pair座標系の平均成分であり、純粋JPY factorや独立市場因子と断定しない。

以下はPPOの個別fold記述値。単位は年率log寄与×100。commissionは全8foldで0。

| Fold | common | relative | financing | spread | markup | net |
|---|---:|---:|---:|---:|---:|---:|
| 2010 | +17.423 | -2.508 | +0.341 | -0.892 | -1.399 | +12.965 |
| 2015 | -24.962 | -2.762 | +1.494 | -0.929 | -2.067 | -29.226 |
| 2016 | +1.063 | +0.233 | +0.343 | -0.589 | -1.150 | -0.100 |
| 2020 | -7.429 | -4.890 | -0.152 | -0.736 | -1.374 | -14.582 |
| 2021 | -5.842 | -0.750 | -0.162 | -0.868 | -1.334 | -8.956 |
| 2022 | +7.987 | +1.101 | +0.106 | -0.843 | -1.575 | +6.776 |
| 2023 | -0.529 | -2.735 | +1.398 | -0.554 | -1.092 | -3.512 |
| 2024 | +19.598 | -2.207 | +3.753 | -0.769 | -1.468 | +18.907 |

canonical/ridgeのcommonは全利用可能foldで0（固定mapのnet exposureが0）。
これは構造確認であり、PPOに対する公正なリスク調整比較の代用ではない。両方策の全成分はCSV/JSONに併記した。

## 固定した2対照とリスク

constantは各foldの元train [start,end)内のwarmup後decisionで実効weightを平均した。
validation/evalの平均・符号・サイズは使わない。学習区間由来の記述的対照であり独立alpha発見ではない。
projectedは各decisionで自身のassetsをPPOへ渡し、元提案をclip/capしてからpair平均へ投影した。
全口座を100万円・flatから独立resetし、共通cap=5、k=1、F0費用を維持した。
投影後にgrossを元PPOへ合わせる再拡大は行わず、費用は各経路の実notionalから算出した。
元PPOの固定action列再生は行っていない。各foldのtrain traceと再推論前assets/元score/投影actionをsealした。

各セルは年率net / 年率gross（%）。個別foldの対応差はこれらから読めるが、17foldのpaired CI/LOOは未出力。

| Fold | PPO | train constant | common projected |
|---|---:|---:|---:|
| 2010 | +13.843 / +16.481 | -3.750 / -3.484 | +19.262 / +21.580 |
| 2015 | -25.342 / -23.069 | -3.431 / -3.176 | -23.294 / -21.330 |
| 2016 | -0.100 / +1.652 | -1.750 / -1.591 | -0.126 / +1.141 |
| 2020 | -13.568 / -11.725 | +0.428 / +0.714 | -7.935 / -6.624 |
| 2021 | -8.567 / -6.531 | -1.137 / -0.925 | -6.419 / -5.028 |
| 2022 | +7.011 / +9.631 | +0.484 / +0.698 | +7.467 / +9.197 |
| 2023 | -3.451 / -1.849 | -4.959 / -4.707 | -0.427 / +0.512 |
| 2024 | +20.813 / +23.544 | +0.497 / +0.626 | +24.498 / +26.316 |

各対照−PPOの個別fold対応差（年率net / 年率gross、percentage points）。全体検定ではない。

| Fold | constant−PPO | projected−PPO |
|---|---:|---:|
| 2010 | -17.593 / -19.964 | +5.419 / +5.100 |
| 2015 | +21.911 / +19.894 | +2.048 / +1.739 |
| 2016 | -1.650 / -3.243 | -0.025 / -0.512 |
| 2020 | +13.996 / +12.439 | +5.633 / +5.100 |
| 2021 | +7.430 / +5.606 | +2.147 / +1.503 |
| 2022 | -6.527 / -8.933 | +0.457 / -0.434 |
| 2023 | -1.508 / -2.859 | +3.023 / +2.360 |
| 2024 | -20.316 / -22.918 | +3.685 / +2.772 |

以下はリスクと回転の個別値。volatilityは年率net log-return標準偏差（%）、turnoverはtarget-weight差絶対値の期間総和。
実売買JPY notionalは別指標としてCSV/JSONに保存した。constantも時価建玉のdrift再調整で実売買が発生する。

| Fold | Policy | gross exposure平均 | net exposure平均 | MDD % | volatility % | turnover |
|---|---|---:|---:|---:|---:|---:|
| 2010 | ppo_ens3 | 1.907 | -0.046 | 22.486 | 31.765 | 237.141 |
| 2010 | train_constant | 0.373 | -0.356 | 5.583 | 5.206 | 0.373 |
| 2010 | common_projected | 1.556 | -0.018 | 21.921 | 30.092 | 216.629 |
| 2015 | ppo_ens3 | 2.812 | -2.062 | 36.955 | 28.947 | 241.767 |
| 2015 | train_constant | 0.357 | -0.313 | 3.646 | 3.130 | 0.357 |
| 2015 | common_projected | 2.461 | -2.086 | 36.440 | 27.043 | 196.490 |
| 2016 | ppo_ens3 | 1.574 | -0.778 | 14.248 | 16.804 | 157.296 |
| 2016 | train_constant | 0.219 | -0.180 | 3.532 | 2.195 | 0.219 |
| 2016 | common_projected | 1.141 | -0.800 | 15.254 | 16.940 | 116.469 |
| 2020 | ppo_ens3 | 1.855 | +0.579 | 17.810 | 13.661 | 194.731 |
| 2020 | train_constant | 0.385 | -0.080 | 1.036 | 1.202 | 0.385 |
| 2020 | common_projected | 1.177 | +0.571 | 14.833 | 11.881 | 142.138 |
| 2021 | ppo_ens3 | 1.845 | +0.351 | 16.110 | 10.961 | 226.288 |
| 2021 | train_constant | 0.291 | +0.197 | 1.584 | 1.212 | 0.291 |
| 2021 | common_projected | 1.184 | +0.376 | 15.853 | 9.728 | 170.140 |
| 2022 | ppo_ens3 | 2.162 | -0.031 | 22.048 | 23.105 | 221.848 |
| 2022 | train_constant | 0.288 | -0.049 | 0.761 | 0.799 | 0.288 |
| 2022 | common_projected | 1.388 | -0.051 | 21.041 | 21.730 | 160.268 |
| 2023 | ppo_ens3 | 1.494 | -0.246 | 11.943 | 9.346 | 142.973 |
| 2023 | train_constant | 0.359 | +0.359 | 6.191 | 3.299 | 0.359 |
| 2023 | common_projected | 0.797 | -0.246 | 8.617 | 8.415 | 97.283 |
| 2024 | ppo_ens3 | 2.031 | -0.972 | 9.039 | 14.008 | 209.461 |
| 2024 | train_constant | 0.173 | -0.047 | 0.423 | 0.502 | 0.173 |
| 2024 | common_projected | 1.329 | -0.954 | 7.775 | 13.894 | 138.351 |

## 判断と未識別事項

全体結論は **不確実で識別できない**、次期学習問題は **情報不足** とする。
例えば2010ではcommonが+17.423に対しrelativeは−2.508、2015ではcommonが−24.962、
2023ではrelativeが−2.735とcommonの−0.529より大きい損失寄与を持つ。
共通方向は正負両方の損益に関係しているが、欠けた9foldを含む主要源泉・era差は判定できない。
projectedのrelativeほぼ0は設計通りであり、そのnet差だけで学習の価値や因果的優越性を認定しない。
constant/projectedはgross・volatilityが元PPOと異なり、eval成績に合わせたサイズ調整もしていない。

次の作業は#29の不足barの由来とcalendar根拠の確保である。必要source/path/hashは
[#29入力監査](protocols/issue29/input-audit.json)と[不足一覧](19-full-period-cost-baselines.md)を参照する。
元providerのpair別responseと休場/欠損の根拠が必要であり、不足日の削除や補間で通年完了にしない。
新data/calendarを使う場合は新しい親sealと事前登録が必要。現attemptは保持する。

全17foldが揃った段階で、横断予測＋cost-aware配分と少数共通要因へのポジション制御を、
寄与・リスク・2対照・fold対応証拠に基づいて選び直す。現時点でpair ranking改善へ固定せず、
共通要因の新featureや追加学習もこの部分診断から採用しない。#16の旧判定・独立収益性は変更しない。

## 再現・検証

```bash
uv run forex-profit-attribution \
  --config configs/research/issue30_profit_attribution.json \
  --output runs/issue30-reproduction

uv run python -c 'from pathlib import Path; from forex_trainer.profit_attribution import verify_attribution; print(verify_attribution(Path("docs/research/results/issue30"))["status"])'
```

出力は新規directoryのみ。exit 2は登録85セルの未完了、exit 1は明示的な入力/実行エラー。
`uv run pytest -q`：376件通過。新規21件は実効cap、common/relativeの恒等式、signed carry、
費用分離、neutral、pair順/時刻/未保有価格の改変、fold境界、train-only、独立口座投影、
ゼロnetのlog極限、破産のlog未定義、全17fold統計と部分panelの集約停止を検証した。
保存後に親#29の全artifactと16対照口座を再照合した。
同一commit・同一sourceで別directoryへ全campaignを再実行し、64個のfold artifact（train trace、
元方策再現、対照step/pair）がbyte単位で一致し、40件のfold metrics/帰属も完全一致した。
保存traceの監査では、全8foldのprojected口座で初回以外の全decisionのassetsと元PPO提案が
対応するdirect PPO口座と異なっていた。独立口座の状態を使う対照であることを実経路でも確認した。
