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

## 実行結果（2026-10-09 JST）

実装SHA `47992880637484bb365a93cd5fcc0841a14eeddc` のclean runtimeで357口座を初回実行した。
全口座が固定区間を完走し、strategy terminal・口座実行例外・口座retryは0。レポート保存障害1件は保存traceから復旧した。全口座の会計・source/hash・統計を保存成果物から再検証した。
新規fit・PPO学習・追加fixture口座・旧metrics流用は0。実行前sealと結果を別commitに分離した。

| 方策 | F0平均期間net log | F0平均期間gross log | 全scenario最悪MDD | 開発候補 |
|---|---:|---:|---:|---|
| A1-fixed | -0.008292 | 0.002000 | 0.152781 | rejected |
| A1-cost | 0.003275 | 0.009036 | 0.134757 | rejected |
| A5-fixed | 0.004490 | 0.012405 | 0.177002 | rejected |
| A5-cost | 0.007505 | 0.011925 | 0.123143 | rejected |
| canonical | 0.040105 | 0.068799 | 0.225234 | 凍結対照 |
| ridge | -0.033347 | 0.024160 | 0.272067 | 凍結対照 |
| ppo_ens3 | 0.007409 | 0.026953 | 0.438663 | 凍結対照 |

主比較（F0、cost−fixed）の期間net log対応差：

| h | 平均差 | IID 95% CI | block 95% CI | 追加価値 |
|---|---:|---|---|---|
| 1 | 0.011567 | [-0.008314, 0.030939] | [0.000394, 0.023965] | not_supported |
| 5 | 0.003016 | [-0.022033, 0.031204] | [-0.014381, 0.021523] | not_supported |

選定候補: `None`。停止理由: `no_predictive_room, net_economic_conditions_failed`。
#46: `not_planned`、#49: `not_planned`。
両horizonの予測MSEは全区間平均・前半eraでゼロ予測を下回らず、予測基準は未達。
口座経路の完走は経済的採否の合格を意味しない。詳細な候補別不成立条件と全scenarioの数値はreport.jsonに保持する。

- [全357セル・全比較・era/CI/LOO/risk](results/issue43/campaign/report.json)
- [一覧レポート](results/issue43/campaign/report.md)
- [実行前seal](results/issue43/campaign/activation.json)
- [実行・予算状態](results/issue43/execution-status.json)
- [検証記録](results/issue43/verification.json)
- [報告障害](results/issue43/campaign/reporting-incident.json)と[報告復旧runtime](results/issue43/campaign/reporting-recovery.json)
- [#46への引渡し](results/issue43/campaign/issue46-handoff.json)
- [#49への引渡し](results/issue43/campaign/issue49-handoff.json)

新規40件＋既存関連95件、計135テストが通過した。保存済み合成fixtureを再利用し、追加fit・口座replayを検証目的で消費していない。

全4候補のrisk条件は成立した。A1-fixedはF0平均gross logが正でもnetでは負となり、
A1-costとA5-fixedもnet水準0.005を満たさない。A5-costは水準・stress・risk・全LOOを満たすが、
2009–2018 eraの平均net logが−0.014016のため非採用。予測基準未達と費用影響・net条件を
risk不成立と混同しない。#46〜#48はnot_planned、#49へ渡す単純候補も0と確定した。
中期gate未準備を理由にこの判断を保留していない。
