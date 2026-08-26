# 08. 疎なrank allocation行動空間（Issue #2、Round 18〜19）

期間: 2026-08-26〜 / 対象: `longf` PPO + MLP

## 仮説

直接ウェイト行動は多数ペアへ露出を分散しやすい。方策にペア別scoreだけを出力させ、上位と
下位だけへ固定grossを配ると、特定シグナルを埋め込まずに疎性を課せるため、コスト規律と
時代間の汎化が改善する可能性がある。

## 事前登録プロトコル

- direct基準: `configs/wf_r16_long_full/` のvalidation-best ens3
- 共通学習: PPO + MLP、`n_steps=batch_size=1024`、300k decisions、seed 42/43/44
- 共通データ: 9 JPYペア日足、signed financing、同一特徴量・fold境界
- Round 18 screen: 2010/2013/2016/2019/2022/2025の6 foldsで、gross 2.0を固定し
  `top_k=1` と `top_k=2` を比較
- Round 19 confirm: Round 18の平均年率net returnが高い方を、2009〜2025の17 foldsへ拡張
- tie: 通貨ペア設定順を二次キーにしたstable sort
- ensemble: 3モデルのscoreを平均してからrank allocationを一度だけ適用

Round 18の選択基準が同値の場合は、より疎な`top_k=1`を採る。報告指標は年率net/gross
return、Sharpe、max drawdown、target-weight turnover、cost ratio、mean gross leverage、
勝ちfold数、2009〜2018 / 2019〜2025のera別平均とする。各foldの年率値はIssue #1と同じ
実効評価期間ベースで計算する。

## 結果

### 実行同一性

- canonical cache SHA-256:
  `723db7c935dcc27c147007728358d098243eae96998eae146db4c7df02bd081e`
- forex-env SHA: `6024b91`
- device: NVIDIA RTX A6000 (`cuda`)
- direct基準はIssue #1のvalidation-best source runsを現行評価器で再走査した。年率指標は
  Issue #1 reportと一致し、今回追加したtarget-weight turnoverだけを新たに計測した

小型MLPではCPU単独52秒に対してCUDA単独47秒だった。CPU 6並列はPyTorch threadの
過剰競合で大幅に遅く、Round 18はCUDA逐次、Round 19はGPU利用率を上げるCUDA 4並列で
実行した。学習条件・fold・seedはdevice以外すべてdirect基準と同じである。

### Round 18: sparsity screen

表のturnoverは1決定あたり `sum(abs(target_weight_t - target_weight_t-1))` の平均、MDDは
6 foldの平均である。Worst MDDだけはfold最大値を示す。

| Action | Net/年 | Gross/年 | Sharpe | Mean MDD | Worst MDD | Turnover | Cost ratio | Mean gross | 勝ちfold |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| direct | +17.27% | +20.08% | +0.296 | 15.94% | 22.05% | 0.804 | 1.84% | 2.001 | 4/6 |
| rank top1 | −7.46% | −4.91% | −0.955 | 11.23% | 20.53% | 1.322 | 1.99% | 2.000 | 2/6 |
| rank top2 | −3.08% | −0.74% | −0.586 | 8.12% | 12.59% | 0.967 | 1.78% | 2.000 | 3/6 |

両rank条件とも負だったが、事前登録した選択規則に従って平均netが高い`top_k=2`を
Round 19へ進めた。top2はtop1よりturnover、cost、drawdownのすべてが小さく、固定gross下では
極端な2ペアだけを毎回入れ替えるより4ペアへ分散する方がcost disciplineに優れた。

### Round 19: 17-fold confirmation

| Action | Net/年 | Gross/年 | Sharpe | Mean MDD | Worst MDD | Turnover | Cost ratio | Mean gross | 勝ちfold |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| direct | **+10.31%** | **+13.06%** | **+0.222** | 16.52% | 36.95% | **0.861** | 1.91% | 2.173 | **8/17** |
| rank top2 | **−3.86%** | **−1.54%** | **−0.818** | 8.42% | 14.41% | **0.974** | 1.77% | 2.000 | **5/17** |

rank top2のfold対応net差はdirect比 **−14.17 percentage points/年**。rankがdirectを
上回ったのは7/17 foldだが、そのうち複数は両方が負の年で、rank自身のプラス年は5/17に
留まった。

| Era | Action | Net/年 | Gross/年 | Sharpe | Mean MDD | Worst MDD | Turnover | Cost ratio | Mean gross | 勝ちfold |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2009–2018 | direct | +17.24% | +20.43% | +0.493 | 18.76% | 36.95% | 0.950 | 2.14% | 2.371 | 6/10 |
| 2009–2018 | rank top2 | −1.33% | +1.03% | −0.375 | 8.36% | 12.59% | 0.951 | 1.77% | 2.000 | 3/10 |
| 2019–2025 | direct | +0.41% | +2.55% | −0.167 | 13.32% | 22.05% | 0.735 | 1.59% | 1.890 | 2/7 |
| 2019–2025 | rank top2 | −7.48% | −5.21% | −1.451 | 8.50% | 14.41% | 1.005 | 1.78% | 2.000 | 2/7 |

fold別netは次のとおり。

| 評価年 | Direct | Rank top2 | Rank − direct |
|---:|---:|---:|---:|
| 2009 | −23.20% | +4.61% | +27.81pt |
| 2010 | +13.77% | −15.69% | −29.46pt |
| 2011 | +42.20% | +14.78% | −27.42pt |
| 2012 | +44.23% | −2.39% | −46.62pt |
| 2013 | +88.60% | −11.21% | −99.81pt |
| 2014 | −4.38% | −1.09% | +3.29pt |
| 2015 | −36.57% | −2.26% | +34.31pt |
| 2016 | +0.19% | +14.59% | +14.40pt |
| 2017 | −4.08% | −14.32% | −10.23pt |
| 2018 | +51.62% | −0.33% | −51.96pt |
| 2019 | −17.95% | −11.57% | +6.37pt |
| 2020 | −16.83% | −14.96% | +1.87pt |
| 2021 | −5.67% | −10.28% | −4.61pt |
| 2022 | −2.25% | +1.06% | +3.32pt |
| 2023 | −2.81% | −18.02% | −15.21pt |
| 2024 | +27.13% | −2.95% | −30.08pt |
| 2025 | +21.26% | +4.33% | −16.93pt |

## 判定

**structured rank allocationは不採用**。固定grossとmarket neutralityによりdrawdownは半減し、
全期間のcost ratioもわずかに下がった。しかしturnover自体は減らず、後半eraではdirectより
turnover・costとも増えた。grossでも負なので、失敗原因はコストだけではなく、方策scoreの
順位シグナルがout-of-periodで一般化しなかったことにある。

したがって「tailだけを取る」という正しいポートフォリオ形状を課しても、それだけでは
direct weightsを上回らない。learned sizingやtrade/no-trade gatingは非ゴールのまま残し、
既定は`rank_allocation: none`のdirect modeを維持する。実装は今後の独立した構造比較に使えるが、
今回の成績を理由に既定動作を変更しない。
