# 期間利用台帳・試行台帳

監査基準日: 2026-09-09。対象repository snapshot:
`798200c88d239ab548ba65c47a59b3c3f40797ea`（[PR #23](https://github.com/TsuyoshiFujii-0915/forex-trainer/pull/23)）。
この台帳は下記の研究ノート、Issue/PR、tracked config・reportとprovenanceを照合した記録である。
gitignoredの全run、個人メモ、他口座・他repositoryの閲覧履歴を網羅した証明ではない。
「不明」は0回または未使用を意味しない。元ノートの「未使用」「合格」は当時の呼称として保存し、
本台帳の現在の資格とは区別する。

## 期間利用

範囲は暦上の対象。実効損益計測範囲は別欄に記す。train/validationで使った履歴も未使用ではない。

| 期間 | 利用目的・実効範囲 | 構成選択への使用 | 出典 | 現在の資格・未確認事項 |
|---|---|---|---|---|
| 2003-06〜2005-07-18 | 長期cache取得要求、expanding train開始の宣言 | train設計に使用 | [05](../../05-rl-architecture-campaign.md)、[fetch config](../../../../configs/fetch_1d9p_2003.yaml)、[2006 config](../../../../configs/sealed_r16_longf/sealed2006_r16_longf.yaml) | 9-pair inner join実始点より前。利用可能な独立評価データの存在を確認できない。取得要求日を実在barとしない |
| 2005-07-19〜2006-12 | 21年rule map、学習・validation。sealed2006はtrainデータ不在で実行不能 | 長期成績が研究方針変更に使用。2006 PPO成功runは未確認 | [04](../../04-rl-vs-rule-benchmark.md)、[05](../../05-rl-architecture-campaign.md)、[07 cache監査](../../07-data-sufficiency-and-scaling.md) | 使用済み履歴。PPO確認未実行を理由に再封印しない。2005はH2のみ |
| 2007〜2008 | rule mapとlegacy longf一発確認、平均+7.8%。各年の正確なdecision境界は旧run再監査が必要 | 開封後にlongfの判断へ使用。以降の設計者が結果を知っている | [05](../../05-rl-architecture-campaign.md)、[sealed configs](../../../../configs/sealed_r16_longf)、[04](../../04-rl-vs-rule-benchmark.md) | 過去確認使用済み。弱く整合的だが2008のseed依存。独立確認として再利用不可 |
| 2009〜2018 | longf architecture、checkpoint、rank、gate、regime、教師あり、固定map、会計診断 | 両era採否、研究の分岐・設計に反復使用 | [05](../../05-rl-architecture-campaign.md)、[06](../../06-checkpoint-selection-study.md)、[10](../../10-learned-apply-hold-gate.md)、[11](../../11-supervised-ranking-learnability.md)、[12](../../12-supervised-portfolio-translation.md)、[13](../../13-frozen-spread-to-net-decomposition.md) | **development folds**。年内32-row warmup + window32、2009最初のdecisionは04-03。全foldの実効時刻は[#16 steps](../../results/issue16/steps.csv) |
| 2019〜2025 | 初期walk-forward、約15構成、rule・残差scale・ensemble・データ量・上記後続研究 | +7%主張まで結果を見て設計反復し、その主張を撤回 | [04](../../04-rl-vs-rule-benchmark.md)、[07](../../07-data-sufficiency-and-scaling.md)、[PR #18](https://github.com/TsuyoshiFujii-0915/forex-trainer/pull/18)、[PR #23](https://github.com/TsuyoshiFujii-0915/forex-trainer/pull/23) | **development folds**。#15/#16は事前基準による開発証拠であり新しい独立証拠ではない |
| 2026-01〜06 | xs ens3確認と残差ens3確認。同じH1を少なくとも2構成で使用。xs実効評価04〜06の63決定 | 確認結果を比較・研究解釈に使用。alpha等の直接選択への使用範囲は不明 | [04](../../04-rl-vs-rule-benchmark.md)、[xs config](../../../../configs/confirm2026h1_1d7p_xs.yaml)、[res config](../../../../configs/confirm2026h1_1d9p_res.yaml) | 使用済み。残差確認の「合格」を独立収益性の確立と呼ばない。旧artifact完全性は未確認 |
| 2026-07〜監査日 | tracked成果物で未使用と証明できない。外部閲覧・cache取得範囲の網羅性も不明 | 不明 | [Issue #20](https://github.com/TsuyoshiFujii-0915/forex-trainer/issues/20)、上記snapshotの監査範囲 | **未確認、確認候補から除外**。暦上新しいことは未使用証拠ではない |
| 設計固定後の2027年候補 | 凍結方策の前向き観測のみ | 開始前に固定。閲覧を設計に使えば確認資格を失う | [開始・終了契約](forward-operations.md) | まだ開始していない。外部入力・記録実装・方策bundle未確定のためblocked |

**現時点で確認可能な未使用の過去期間はない。** 不明な過去ログが見つかれば既存行を書き換えず、
日付・対象行・出典・判定変更理由をこの文書末尾へ追記する。新たな結果閲覧も記録対象とする。

## 試行の数え方

今後の1 trialは、評価結果を見て次の判断に使い得る、事前登録された1つの研究構成とする。
feature/target/map/seed集合/checkpoint選択規則/採否指標の変更は別trial。
全seed・foldはそのtrialの反復測定であり独立trialや独立市場標本に水増ししない。
事前固定したvalidation grid内の選択は1構成内の選択だが、grid候補数とfit数を別記する。
結果後のgrid拡張は別trial。評価を開いた失敗・中断・事後診断も残す。
同一bytes・同一条件の障害再実行はattemptを増やし、元trial IDは維持する。

| 台帳ID・範囲 | 既知の規模 | 重複・不明点と扱い | 出典 |
|---|---|---|---|
| legacy-round0-14 | Round7〜12付近で「約15構成」。ens10、pair数、残差scale 0.15/0.3/0.5、rule sweep、overlay、H1確認等も記録 | 約15が全探索を網羅するか不明。会計世代も異なり総独立trial数は不明 | [01](../../01-diagnosis-and-protocol.md)〜[04](../../04-rl-vs-rule-benchmark.md) |
| architecture-round15-17 | [05]の探索表9行（MLP、xattention、regularized、long、RecurrentPPO、TQC、long+xattention、long2、ens5） | screenとfullの再測定がある。9行を後のtrial_count=9の完全な内訳と断定しない | [05](../../05-rl-architecture-campaign.md)、[longf configs](../../../../configs/wf_r16_long_full) |
| checkpoint-issue1 | 3選択規則、17 folds×3 seeds=51学習軌跡。lateはseed横断15 checkpoint | validation-best/last/lateは同じ軌跡を再使用。約20 validation測定を独立市場証拠としない | [06](../../06-checkpoint-selection-study.md)、[Issue #1](https://github.com/TsuyoshiFujii-0915/forex-trainer/issues/1)、[PR #10](https://github.com/TsuyoshiFujii-0915/forex-trainer/pull/10) |
| scaling-issue7 | 4履歴方針×7 folds×3 seeds=84 run、28 ens3 | CPU longfとCUDA scalingを混ぜない。過去architecture trialと重複する概念がある | [07](../../07-data-sufficiency-and-scaling.md)、[source runs](../../results/issue7/source_runs.json) |
| reporting-issue6 | 集計基盤の実装。新規研究treatmentのtrial_countではない | study-specific集計を標準化したもので、過去試行の独立性や全累積数を復元したものではない | [Issue #6](https://github.com/TsuyoshiFujii-0915/forex-trainer/issues/6)、[PR #12](https://github.com/TsuyoshiFujii-0915/forex-trainer/pull/12)、[ADR-0012](../../../decisions/0012-fold-aware-aggregate-research-reports.md) |
| rank-round18-19 | k1/k2 screenとrankf full | screenからfullへの進行を新しい独立市場試行と数えない。全失敗run数は不明 | [08](../../08-sparse-rank-allocation.md)、[PR #11](https://github.com/TsuyoshiFujii-0915/forex-trainer/pull/11) |
| regime-issue5 | 宣言trial_count=9、報告はcurrent longf 1構成、regime診断 | 値9の全独立trialへの対応は未証明。regime bucket/metric診断を追加alpha検定と解釈しない | [09](../../09-regime-tail-loss-diagnostic.md)、[campaign](../../../../configs/research/issue5_longf_baseline.yaml)、[report](../../results/issue5/baseline_report/report.json) |
| gate-issue4 | 宣言trial_count=10 = 上記9+gate treatment1。51新規学習、learned/forced/controlの3報告構成 | **9+10と合算しない**。forced applyはsame-model対照、controlは再利用 | [10](../../10-learned-apply-hold-gate.md)、[campaign](../../../../configs/research/issue4_apply_hold_gate.yaml)、[report](../../results/issue4/report/report.json) |
| ranking-issue15 | ridge family1、alpha候補4×17 folds=68候補fit、3 score比較 | 4 alphaはvalidation内部。PPO/control再利用。全研究累積trial_countの宣言なし | [11](../../11-supervised-ranking-learnability.md)、[config](../../../../configs/research/issue15_supervised_ranking.yaml)、[PR #17](https://github.com/TsuyoshiFujii-0915/forex-trainer/pull/17) |
| portfolio-issue16 | 固定map1、3 policy比較、再学習0 | #15 score/PPO再使用。新たな独立市場標本0。累積trial_countの宣言なし | [12](../../12-supervised-portfolio-translation.md)、[manifest](../../../../configs/research/issue16_supervised_portfolio.json)、[PR #18](https://github.com/TsuyoshiFujii-0915/forex-trainer/pull/18) |
| decomposition-issue19 | 新規model/map0、3 policy×5主系列の会計診断 | #15/#16同一観測再使用。系列を独立trial数に換算しない。累積trial_countの宣言なし | [13](../../13-frozen-spread-to-net-decomposition.md)、[manifest](../../../../configs/research/issue19_spread_decomposition.json)、[PR #23](https://github.com/TsuyoshiFujii-0915/forex-trainer/pull/23) |
| issue20-forward-v1 | 事前計画1、実行0、探索候補0、再学習0 | 凍結した既存2025 fold bundleを使う前向きreplication候補。開始条件未達 | [protocol](../../14-research-and-forward-protocol.md)、[運用契約](forward-operations.md) |

全研究の総trial数と独立trial数はいずれも**不明**。これらを足した値でBonferroni、deflated Sharpe等の
補正済み有意性を主張しない。既存CIは各固定campaign条件下の不確実性であり、全設計選択を補正した
確率値ではない（[ADR-0012](../../../decisions/0012-fold-aware-aggregate-research-reports.md)）。

今後の追記行は `trial_id / parent_trial_id / scope / hypothesis / changed_variable / frozen_values /
budget / folds / seeds / validation_rule / registered_at / registration_commit / opened_at /
status(planned,running,completed,failed,stopped) / attempts / artifacts_and_hashes / decision / unknowns`
を必須とする。不明は理由を記す。実行前と開封後を別行で追記し、失敗した構成を削除しない。
