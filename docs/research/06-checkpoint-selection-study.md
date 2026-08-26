# 06. チェックポイント選択方式の比較（Issue #1）

期間: 2026-08-25〜 / 対象: `longf` development protocol

## 仮説

ADR-0005 は学習中に約20回検証し、検証リターン最大のモデルを選ぶ。この反復選択は
固定された6か月検証区間のノイズへ適合し、次の評価年への汎化を悪化させる可能性がある。
そこで同じ学習軌跡から次の3方式を比較する。

1. `validation_best`: 現行どおり、各seedの検証リターン最大モデル
2. `last`: 各seedの最終rollout更新後モデル
3. `late_checkpoint_ensemble`: 検証時点のうち名目学習budgetの
   80/85/90/95/100%に最初に到達した5モデルをseed横断で等ウェイト行動平均

`longf` では3番の時点は240k/255k/270k/285k/300kである。PPOはrollout単位で
学習を終了するため、`model_last.zip` は303,104 stepsの更新後であり、300k checkpointは
その更新前である。これは両者を区別する意図した比較条件である。

## 固定プロトコル

- folds: 2009〜2025の17暦年（2009〜2018 / 2019〜2025も別集計）
- seeds: 42 / 43 / 44
- 学習: `configs/wf_r16_long_full/` を無変更で使用し、各fold/seedを1回だけ学習
- fold policy: bestとlastは3モデル、lateは15モデルを等ウェイト行動平均
- 指標: 年率net/gross return、Sharpe、max drawdown、勝ちfold、era別結果、
  validation-bestに対するfold対応差
- 再現定義: `configs/studies/issue1_longf_checkpoint_selection.yaml`

実行コマンド:

```bash
uv run forex-selection-study \
  --study configs/studies/issue1_longf_checkpoint_selection.yaml \
  --runs-root runs
```

study成果物は入力parquetのSHA-256、source run、seed、実モデルpathとSHA-256、git SHAを
記録する。不完全なcheckpoint集合やfold/seedの不一致はエラーとし、残存メンバーだけでの
暗黙評価は行わない。

各foldの年率returnは、実効評価区間の経過年数を `Y`、累積log returnを `R` として
`exp(R / Y) - 1` で計算する。特徴量warmup後の実効期間を使うため、暦年foldのsimple
returnを平均した既存研究ノートの `longf +4.7%/年` とは定義が異なり、数値を直接比較
しない。overall/era欄はfold年率returnとfold Sharpeの算術平均、drawdownは対象fold中の
worst値である。walk-forward equityを連結したportfolio指標ではない。fold対応差も同じ
fold指標同士で算出する。

## 実行状況とADR-0005の判定

2026-08-25に17 folds × 3 seedsの51学習と、各foldの3方式評価を完走した。
実行時の識別情報は次のとおり。

- forex-trainer: `3e2f33669b9a3aa9904ccd66168f0f960670c405`
- forex-env-v3: `6024b91c0f3592611849bc231922ab60e6090aed`
- data SHA-256: `723db7c935dcc27c147007728358d098243eae96998eae146db4c7df02bd081e`
- study config SHA-256: `5f60d85ee9a445838fa060474657f2c3daaed9ce5fe8ef358c340dd51ec7aa99`
- device: committed configどおりCPU

実行前に、週末を含む日足データのSharpeをmedian bar間隔（1日）で年率化し、実際の
年間bar数より過大評価する不具合を修正した。実験では評価期間全体の経過年数と実観測bar数を
使って年率化している。

### 全期間（2009〜2025）

| 方式 | fold平均net/年 | fold平均gross/年 | fold平均Sharpe | fold平均DD | 最悪DD | netプラスfold |
|---|---:|---:|---:|---:|---:|---:|
| validation-best | 10.31% | 13.06% | 0.222 | 16.52% | 36.95% | 8/17 |
| last | 20.54% | 24.46% | 0.301 | 19.93% | 37.32% | 7/17 |
| late ensemble | 14.16% | 17.52% | 0.241 | 20.16% | 45.44% | 7/17 |

### Era別

| Era | 方式 | fold平均net/年 | fold平均Sharpe | fold平均DD | 最悪DD |
|---|---|---:|---:|---:|---:|
| 2009〜2018 | validation-best | 17.24% | 0.493 | 18.76% | 36.95% |
| 2009〜2018 | last | 36.03% | 0.654 | 22.18% | 37.32% |
| 2009〜2018 | late ensemble | 25.00% | 0.561 | 22.85% | 45.44% |
| 2019〜2025 | validation-best | 0.41% | -0.167 | 13.32% | 22.05% |
| 2019〜2025 | last | -1.60% | -0.203 | 16.72% | 23.29% |
| 2019〜2025 | late ensemble | -1.33% | -0.216 | 16.31% | 22.58% |

### validation-bestとのfold対応差

lastはnet年率の平均差が+10.23ポイントだが、中央値は-0.02ポイントで、改善は8/17 fold
だった。平均差は主に2011・2012 foldの大幅上振れによる。一方、DDは平均+3.41ポイント、
中央値+3.66ポイント悪化し、改善は3/17 foldに留まった。

late ensembleはnet年率の平均差+3.85ポイントに対して中央値-0.84ポイント、改善7/17 fold
であり、DDは平均+3.64ポイント悪化した。最悪DDも45.44%で、validation-bestの36.95%を
上回った。

以上から **ADR-0005を維持（retain）** し、既定の`validation_best`を変更しない。lastの
全期間平均は高いがfold横断で一貫せず、直近eraではvalidation-bestがnet、Sharpe、平均DD、
最悪DDのすべてで最良だった。late ensembleにも既定方式を置換する頑健な証拠はない。

完全な集計は [report.md](results/issue1/report.md)、機械可読値は
[report.json](results/issue1/report.json)、全fold行は
[fold_results.csv](results/issue1/fold_results.csv) に保存した。
