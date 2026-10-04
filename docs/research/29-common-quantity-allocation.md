# 共通方向の固定・費用対応配分と数量会計（Issue #42）

`issue40-common-quantity-same-close-v1` を opt-in 実装した。
本番の予測器 fit・市場口座・PPO 訓練は0。#43 の収益性判断を本Issueでは行わない。
親は [v3契約](27-short-common-direction-revision.md) と [#41予測](28-common-basket-forecaster.md)。
[ADR-0048](../decisions/0048-opt-in-quantity-hold-accounting.md) に従い、旧 target API、reward、
旧 measurement と v1/v2/v3登録bytesを保持する。forex-env の変更・別PRは不要。

## 配分と情報の境界

`common_allocation.load_forecasts(bundle)` は既存 `verify_bundle` で全34モデル・7,138予測・
train/validation境界・hash・as-ofを照合した上で、診断用の将来targetを含まない `Forecast` を返す。
キーは `(fold, horizon, decision_at_utc)`。単位は h営業日の basket price simple return。
共分散は#41の採用train行から作った `0.9*S+0.1*diag(S)+1e-8*I` をそのまま読み、
推定式・行hash・最大train label終点を確認する。evalからの再推定・追加fitは0。

`allocate(state, forecast, 'fixed'|'cost')` の切替だけで同じ forecast identity を利用する。
fixed は `sign(forecast)/9` を毎回再送し、0はflat。cost は等notional basketの `u∈[-1,1]`
と literal quantity hold を比較する。全price期待値は同じscalar。価格driftを持つholdは
現在の marked weights で評価する。追加のpair順位予測はない。

utilityは現在equity単位の price + signed financing − entry cost − holding markup −
`5*w'Σ_h*w`。signed financingは当日の `-CarryAnnual`、365日除算、markupは日次率。
登録London平日calendarの実UTC時間を使い、週末・DSTを含める。h=5は毎営業日に5日utilityを
再計画する近似。末尾も同じhを使い、未来実現price/rate、仮想決済費用、評価残日数を使わない。
予測financingとmarkupは現在notional一定の近似、実現financingは旧会計と同じ終了marked
notionalに課す。予測と実現の各成分を混同しない。

solverは両端・全費用kink・0・各区間の停留点・別hold候補を列挙する。traceには領域外も含む
全費用kinkを保存し、候補生成は `[-1,1]` 内のみ。大域最大utilityから1e-12以内を同値とし、
実売買notional小→絶対gross小→u昇順。holdのuは現在net weight和（tie時の座標）として記録する。
完全重複は決定論的なmode順。制約残差は1e-10以下、非有限・解なし・超過は明示例外。

## 数量と実現会計

研究座標のquantity=`signed exposure_jpy / inverse price`。実商品のbase数量とは異なる。
`QuantityAccount` は現在quantityを保持し、売買は `abs(q_new-q_old)*P_current` から計算する。
literal holdはrebalanceを呼ばず、丸め誤差による微小費用も発生しない。price PnLと保有費用は継続。
実現cashは既存 `PortfolioAccount` を利用し、spread/commissionは売買ごとに一度、markupは
区間開始gross notional、financingは終了signed notionalへ一度だけ計上する。

全口座は100万円・flat開始。各decisionの推論前にdrift違反を比例縮小し、`risk_drift` として保存。
提案quantityにも費用控除後のgross≤5・pair絶対weight≤1を適用する。自分の縮小費用を含む
区分一次制約を解き、最大の実行可能な比例係数を選ぶ。元提案はdecision traceに保持し、
`risk_proposal` を実売買として保存する。旧対照の提案は変更しない。共通capをrisk-matchedとは呼ばない。

marginは初期equityの0.2以下。正equityのmarginと非正equityを `strategy_terminal` として区別し、
未測定decision数・予定/実際のlast markを保存する。非正equityのlogはnull。最終markでは
仮想清算・追加risk取引を行わない。solver/input/runtime障害はincidentとして例外を再送し、
失敗時の最後の口座状態・完了traceを保持する。

## #43への接続

- `marks_from_period(prepare_period(...))`：#28の63本先行historyと測定区間を再検証し、
  同じ8特徴量・32観測windowを作る。historyで口座を動かさず、最終markはdecisionにしない。
- `load_forecasts(...)` → `allocation_policy(forecasts_for_fold_and_horizon, mode)`：
  stateの時刻に対する予測は必須。欠落・時刻/単位不一致を補完しない。
- `legacy_policy(predict)`：既存の `canonical_action`、`FrozenRidge.action`、
  `FrozenEnsemble.action` を変更せず接続する。PPO assetsはその独立口座の前回実行weight・
  leverage=1・price PnL/equityで、初回はすべて0。旧APIの観測座標を保持する。
- `replay(marks, costs, policy, runtime_sha256, policy_source)`：独立口座の完全なtraceを返す。
  F0/F1/F2ごとに新口座を開く。F1はspreadだけ2倍、F2はmarkupだけ2倍。

同じforecast/model/provenance/covarianceのidentity、equity、予測utility内訳、全solver候補、
実売買・費用・hold・risk override・terminalを保存する。#43では全357口座の実行前sealと
同一measurement/runtime検査、比較・集計を別途行う。旧metricsを再利用しない。

## 合成fixtureと有限予算

[設定](../../configs/research/issue42_common_quantity.json) と
[合成入力](protocols/issue42/fixture.json) に13口座を事前登録した。上限16の内数であり、
残り3を同一条件の再実行枠と解釈しない。障害retryはcampaign共有24口座・同一work item1回までの
別登録が必要。このCLIにretry機能はない。v3登録時点のconsumed=0を書き換えない。

対象はfixed/cost×h=1/5、hold/partial/close/short、F1/F2、drift縮小、margin、破産、
canonical、旧ridge interface、direct PPO ens3 interface。計13口座。controlは既存の
`fixture_policies` による固定係数ridge・未訓練seed42/43/44 PPOで、fit・PPO訓練は0。
実市場モデルの成績を確認するfixtureではない。

実装commit後のclean SHA、両repo・依存version・lock・CPU、入力・configとmerge済みv3をsealする。
各口座は実行前にappend-only台帳へ予約し、途中失敗も消費として残す。固定run directoryと
コミット済み保存結果の両方を検査し、別cloneでも通常枠を再取得できない。

```bash
uv run forex-common-quantity seal \
  --config configs/research/issue42_common_quantity.json \
  --dependency-merge-commit 3e82733d79d2bfb5b898803662b9f8613f2e11a8 \
  --output runs/issue42-execution-seal
uv run forex-common-quantity fixture \
  --config configs/research/issue42_common_quantity.json \
  --activation runs/issue42-execution-seal/activation.json
```

上記は#55/#41を含む確認済みmerge SHA。
口座実行を伴わない再検証は、保存済み成果物に対して行う。

```bash
uv run forex-common-quantity verify --directory docs/research/results/issue42/fixture
uv run pytest tests/test_common_allocation.py tests/test_common_quantity_*.py -q
```

数値solverのテストは合成口座を開かない。口座受入テストは保存済み13口座のcash/quantityを
数式で再照合し、再実行による追加予算を消費しない。hashを作り直した会計不整合も検出する。
テストを実装前に追加し、最初に未実装による収集失敗を確認した。実装中にテストは変更していない。

## 実行結果（2026-10-04）

実装SHA `feb089579cdc9c042dfb3f7a85b029710016e0c1`、forex-env SHA
`6024b91c0f3592611849bc231922ab60e6090aed` のclean runtimeで、初回通常13口座を実行した。
11口座は全5decision完走、margin/破産の2口座は意図したstrategy terminalを記録。
全57遷移を照合し、共有retry0、fit0、市場口座0、PPO訓練0。通常枠を再実行しない。

- [実行・予算状態](results/issue42/execution-status.json)
- [fixture manifest](results/issue42/fixture/manifest.json)
- [実行前seal](results/issue42/fixture/activation.json)
- [append-only口座台帳](results/issue42/fixture/account-ledger.jsonl)
- [検証記録](results/issue42/verification.json)

保存コピーはrunの全artifactと元bytesが一致。追加37テストと既存76テスト、計113が通過した。
旧#41の6合成fitを再消費する3テストは今回除外し、既存の保存済み学習証拠を保持した。
検証CLI・テストは保存traceを読むだけで、13口座を再度動かさない。

本番損益は閲覧・生成しておらず、4新構成と3凍結対照の357口座評価は#43の別sealへ渡す。
#43はこのmeasurementと同一runtimeで全対照を評価する必要がある。中期RL・実quote・
独立確認を有効化せず、旧成果物と分類はそのまま保持する。


## PR #58レビュー対応（2026-10-04）

配分・実現会計の計算は変えず、入力と保存成果物の検証を強化した。
`load_forecasts` は cells の一意性・全fold/horizonの存在・固定modelパス・実ファイルhash・
モデル内のfold/horizonを照合し、そのモデルのprovenance hashを各forecastへ結び付ける。
別horizonの共分散を元のmodel hashのまま渡すことや、重複cellによる上書きを拒否する。

保存口座の検証は、登録済みfixtureから全予定mark・シナリオ別price/carry/costを再構成する。
traceの各decision/next mark、input hash、cash、first/actual/planned mark、残decision数を照合し、
completeは全予定経路を通った場合のみ受理する。terminalは最初の閾値到達時に停止し、
margin/非正equity、行と口座のterminal理由、log status、台帳の完了statusを一致必須とする。
費用だけで閾値に到達する場合は同じdecision時刻で終了し、翌markへ進めない。

[追加検証記録](results/issue42/review-verification.json)：新規26ケースと既存113ケースの
計139テストが通過した。外側hashを整合させた改変コピーで、共分散参照差替え・cells重複/欠落・
短縮/空trace・terminal誤分類・時刻断絶・入力/費用差異を検証した。費用のみのterminalは
手計算のledgerを照合し、account生成・policy推論・fit・口座replayは0。

元の#41 bundle、13合成口座の全artifact、実行runtime・消費予算記録は元bytesで保持した。
再実行や再sealによる置換はしていない。元の3件のfitテストは引き続き除外した。
