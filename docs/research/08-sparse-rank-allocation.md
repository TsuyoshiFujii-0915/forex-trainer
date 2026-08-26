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
- Round 19 expanded evaluation: Round 18の平均年率net returnが高い方を、
  screen 6 foldsを含む2009〜2025の17 foldsへ拡張し、未使用11 foldsも別集計
- score表現: float32で表現可能な全有限域をaction spaceとし、clipせず生scoreを順位化
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
- device: direct、rankともに`cpu`
- rank score/order修正 SHA: `bb4a92a`、Round 19選択確定 SHA: `48245fb`
- direct基準はIssue #1のvalidation-best source runsを現行評価器で再走査した。年率指標は
  Issue #1 reportと一致し、今回追加したtarget-weight turnoverだけを新たに計測した

ADR-0011に従い、旧`[-1, 1]` score clipを廃止した。Stable-Baselines3が有限Box境界を
要求するため、境界は`±np.finfo(np.float32).max`とした。PPOの全有限float32出力に対して
SB3の境界clipは恒等写像であり、wrapperも生scoreをstable sortする。`5 > 2 > 1`が
同点にならない回帰テストと、実際のPPO学習を通すpipeline testで確認した。

direct source 51 runsとrank 69 runsはすべてCPUで学習した。旧clip/CUDA成果物との混在を
避けるためrank再学習は専用`runs/issue2_review_cpu`で逐次実行し、全metadataについて
seed 42/43/44、`device: cpu`、該当git SHAを検証してからensemble評価した。

### Round 18: sparsity screen

表のturnoverは1決定あたり `sum(abs(target_weight_t - target_weight_t-1))` の平均、MDDは
6 foldの平均である。Worst MDDだけはfold最大値を示す。

| Action | Net/年 | Gross/年 | Sharpe | Mean MDD | Worst MDD | Turnover | Cost ratio | Mean gross | 勝ちfold |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| direct | +17.27% | +20.08% | +0.296 | 15.94% | 22.05% | 0.804 | 1.84% | 2.001 | 4/6 |
| rank top1 | +1.14% | +3.93% | +0.113 | 7.58% | 12.60% | 1.284 | 2.07% | 2.000 | 3/6 |
| rank top2 | −1.19% | +1.19% | −0.345 | 7.33% | 9.51% | 0.974 | 1.79% | 2.000 | 3/6 |

事前登録した選択規則に従って平均netが高い`top_k=1`をRound 19へ進めた。top2はtop1より
turnover、cost、drawdownが小さい一方、net選択基準ではtop1が2.33pt/年上回った。

### Round 19: 17-fold expanded evaluation

Round 19の17 foldsにはRound 18のscreen 6 foldsを含むため、独立confirmationではない。
選択後に新たに評価した未使用11 foldsは次節で分離する。

| Action | Net/年 | Gross/年 | Sharpe | Mean MDD | Worst MDD | Turnover | Cost ratio | Mean gross | 勝ちfold |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| direct | **+10.31%** | **+13.06%** | **+0.222** | 16.52% | 36.95% | **0.861** | 1.91% | 2.173 | **8/17** |
| rank top1 | **−2.78%** | **−0.09%** | **−0.314** | 8.71% | 17.24% | **1.301** | 2.03% | 2.000 | **7/17** |

rank top1のfold対応net差はdirect比 **−13.09 percentage points/年**。rankがdirectを
上回ったのは7/17 fold、rank自身のプラス年も7/17だった。

### 選択に未使用の11-fold結果

未使用集合は2009/2011/2012/2014/2015/2017/2018/2020/2021/2023/2024である。

| Action | Net/年 | Gross/年 | Sharpe | Mean MDD | Worst MDD | Turnover | Cost ratio | Mean gross | 勝ちfold |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| direct | +6.51% | +9.24% | +0.181 | 16.83% | 36.95% | 0.893 | 1.95% | 2.267 | 4/11 |
| rank top1 | −4.92% | −2.28% | −0.547 | 9.33% | 17.24% | 1.309 | 2.01% | 2.000 | 4/11 |

未使用11 foldsだけでもfold対応net差はdirect比 **−11.44 percentage points/年**で、
rankがdirectを上回ったのは4/11 foldsだった。screen 6 foldsを除いても結論は同じである。

| Era | Action | Net/年 | Gross/年 | Sharpe | Mean MDD | Worst MDD | Turnover | Cost ratio | Mean gross | 勝ちfold |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2009–2018 | direct | +17.24% | +20.43% | +0.493 | 18.76% | 36.95% | 0.950 | 2.14% | 2.371 | 6/10 |
| 2009–2018 | rank top1 | −3.87% | −1.27% | −0.441 | 9.59% | 17.24% | 1.256 | 1.97% | 2.000 | 3/10 |
| 2019–2025 | direct | +0.41% | +2.55% | −0.167 | 13.32% | 22.05% | 0.735 | 1.59% | 1.890 | 2/7 |
| 2019–2025 | rank top1 | −1.23% | +1.59% | −0.132 | 7.46% | 12.66% | 1.364 | 2.12% | 2.000 | 4/7 |

fold別netは次のとおり。

| 評価年 | Direct | Rank top1 | Rank − direct |
|---:|---:|---:|---:|
| 2009 | −23.20% | −16.63% | +6.57pt |
| 2010 | +13.77% | −3.04% | −16.81pt |
| 2011 | +42.20% | +3.33% | −38.88pt |
| 2012 | +44.23% | +8.67% | −35.56pt |
| 2013 | +88.60% | −3.87% | −92.48pt |
| 2014 | −4.38% | −5.52% | −1.14pt |
| 2015 | −36.57% | −15.70% | +20.88pt |
| 2016 | +0.19% | +11.13% | +10.94pt |
| 2017 | −4.08% | −8.47% | −4.38pt |
| 2018 | +51.62% | −8.64% | −60.27pt |
| 2019 | −17.95% | +0.30% | +18.25pt |
| 2020 | −16.83% | −2.18% | +14.64pt |
| 2021 | −5.67% | +5.86% | +11.53pt |
| 2022 | −2.25% | +4.20% | +6.46pt |
| 2023 | −2.81% | −15.18% | −12.37pt |
| 2024 | +27.13% | +0.30% | −26.83pt |
| 2025 | +21.26% | −1.90% | −23.15pt |

## 判定

**structured rank allocationは不採用**。score順位を保持し、deviceをCPUに揃え、選択未使用の
11 foldsを分離してもdirectを下回った。固定grossとmarket neutralityにより全17 foldsの
drawdownは約半減したが、turnoverとcost ratioはdirectより増え、gross returnもほぼゼロだった。
失敗原因はコストだけではなく、方策scoreの順位シグナルがout-of-periodで一般化しなかった
ことにある。

したがって「tailだけを取る」という正しいポートフォリオ形状を課しても、それだけでは
direct weightsを上回らない。learned sizingやtrade/no-trade gatingは非ゴールのまま残し、
既定は`rank_allocation: none`のdirect modeを維持する。実装は今後の独立した構造比較に使えるが、
今回の成績を理由に既定動作を変更しない。
