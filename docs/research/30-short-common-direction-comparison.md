# 短期2×2比較（Issue #43）

[ADR-0049](../decisions/0049-evaluate-the-sealed-short-matrix-with-period-log.md)に基づき、
[v3契約](27-short-common-direction-revision.md)、[#41予測](28-common-basket-forecaster.md)、
[#42数量会計](29-common-quantity-allocation.md)を接続する。

## 実行契約

`configs/research/issue43_common_direction.json` は4新構成204口座とcanonical/ridge/direct PPO ens3
153口座を固定する。17区間、F0/F1/F2、初期100万円・flat、pair順・時刻・risk capは共通。
同hのfixed/costは同じmodel・forecast・train共分散を使う。各scenarioの口座を新規に推論し、
旧metricsの流用・再fit・追加fixture口座・PPO学習・自動retryは0。

実行前に全357セル、元モデル訓練lineage、data/calendar/forecast/model、lock・依存・両repoの
clean SHAをsealする。#58を含むマージ済みdependencyが必要。中期RLや実quoteのgateは要求しない。
通常予算は固定run directoryとコミット済み結果の双方で一度だけ予約し、各口座の開始前に
fsyncした台帳へ予約を追記する。失敗も消費に含める。

```bash
uv run forex-common-direction preflight --config configs/research/issue43_common_direction.json --output runs/issue43-preflight
uv run forex-common-direction seal --config configs/research/issue43_common_direction.json --output runs/issue43-execution-seal
uv run forex-common-direction run --config configs/research/issue43_common_direction.json --activation runs/issue43-execution-seal/activation.json
uv run forex-common-direction verify --directory runs/issue43-common-direction-v3
```

preflight/verifyは口座を開かず、fit・推論を行わない。実装とテストをcommitしたclean状態で
seal/runを行い、結果は別commitにする。通常実行のやり直しは禁止。

## 統計と採否

一次量は `log(E_last/1_000_000)` の17区間等重み対応差。cost−fixedの2主比較、同配分内h差、
各3対照差、交互作用の計17比較を各scenarioに表示する。既存の10,000 IID / 長3 circular block、
seed16・95% CI・両era・全LOO・最大絶対fold寄与を使う。寄与分母0はnullと理由を保存する。
経過時間・年率log/simpleは別欄。終端前停止や非正equityを年率化しない。

grossはstepのspread・commission・markupをend equityに戻してlogを集計し、signed financingを
含める既存定義。cash内訳、MDD、母標準偏差volatility、exposure、target weight差、実売買notional、
数量hold、部分サイズ、quantity変更、risk overrideを口座ごとに保存する。target turnoverは提案weight
（数量×当日価格÷decision equity）の前回提案との差。部分サイズは0<abs(u)<1のdecision数。
Sharpeは標本標準偏差を用い、定数経路またはterminalではnullと明示する。

候補・配分追加価値・独立収益性は別判定。v3の数値条件を読み、期間logの値に適用する。
tieはF0平均log→全scenarioのworst MDD→F0平均実notional比→h1→fixed、許容1e-12。
候補の必要経路に不足があれば候補間選定を保留する。terminalは当該候補非採用に加え、
必要な比較は未定義のまま保持する。成功したfoldだけの集計は作らない。

#49への単純候補と#46への逐次仮説を別ファイルに保存する。#46の仮説は選定hの予測基準を
満たす場合に限定し、同forecast/4行動で将来再計画費用を減らせるかだけを引き渡す。
両hの予測基準未達、net経済条件不成立、risk不成立、測定不足を分離して記録する。
RL訓練・canonical default・旧分類・独立収益性は自動変更しない。

## 障害と検証

statusはcomplete / strategy_terminal / blocked_input / execution_error / not_run。
記録可能な実行例外ではincidentと全357セル、残経路、report/manifestを保存して例外を再送する。
入力・hash・会計違反を成功として扱わない。verifyは未完了reportにも利用でき、元市場・費用・
数量/cash・予測identity・source・runtime共通性・台帳・統計・handoffを再照合する。
入力自体が欠落・改変された場合はverifyも原因を明示して失敗し、記録を検証済みと偽らない。

9部分年・8全期待bar範囲の既知development証拠であり、未補正CI。独立trial総数不明、価格修復、
carry vintage不明、同close・逆数数量の限界を保持する。共通capはrisk一致の証明ではない。

## レポート保存障害の明示的復旧

初回の357口座評価と台帳保存後、集計risk判定のNumPy booleanをJSONへ渡したため、
report.json生成でTypeErrorが発生した。真偽値を出力境界でPython boolに確定する修正であり、
数値基準・solver・model・forecast・口座経路は変更しない。

`recover-report --directory runs/issue43-common-direction-v3`は記録済みの報告専用incidentと
全357口座・714台帳イベント・元sealのhashを必須とし、保存traceだけからreportを再構成する。
口座再実行・再推論・retry予算消費は0。初回評価runtimeと報告修正runtimeを別々に残し、
元artifactのbyte不変をverifyで照合する。既存report/manifestの上書きや通常枠の再取得は拒否する。
