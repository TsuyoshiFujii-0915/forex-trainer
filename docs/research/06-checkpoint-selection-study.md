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

このcheckoutには既存研究で使用したcanonical cache
`data/jpy_9pairs_1d_2003_carry.parquet` が存在せず、その元となる旧yfinance不良print修復済み
cacheの完全な再生成手順もcommitされていない。異なる再取得データで「identical data」の
条件を破ることは避け、17 folds × 3 seedsの本比較は未実行とした。

したがって、現時点の判定は **ADR-0005を維持（retain）** とする。これはvalidation-bestの
優位性を確認したという意味ではなく、比較証拠が得られるまで既定動作を変更しないという
保守的な判断である。canonical cacheを配置して上記コマンドを完走後、生成された
`report.json`のfold対応差に基づき、この節を結果で更新する。新しい既定方式を採用する場合は
acceptedなADR-0005を直接書き換えず、後継ADRを追加してADR-0005をsupersededにする。
