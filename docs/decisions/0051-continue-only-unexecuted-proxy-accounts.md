# Title

0051. Issue #45の監査障害後は元2口座を再監査し未実行304口座だけを継続する

## Status

accepted (PR #62 merged on 2026-10-13)

## Context

PR #61の初回attemptは2口座目の保存後監査で停止した。両口座とも171遷移を保存済みだが、
独立再計算equityの1 ULP差を実行時Stateのhashと比較したため、next-closeはexecution_errorになった。
PR #61のコメントは監査のみを修正し、両口座の全件再監査と実行条件の不変確認後、
未実行304口座だけを後続PRのレビュー・マージ後に実行する方針を指定している。
ADR-0050の通常attempt、retry 0、登録済みmeasurementは変更しない。

## Decision

一度限りの独立した復旧・継続登録を追加する。元manifest・元2口座の正確なhash、
全306セルと残304セルの順序、新規実行上限304、旧消費2、fit 0、再推論retry 0を固定する。
元campaign・incident・runtime・execution_errorは変更せず、再監査結果と継続成果物を別directoryへ保存する。

監査では独立会計の数値照合を維持し、照合済み保存値から復元した判断Stateに対してidentityを完全一致で検査する。
許容する実行source差分はaudit_proxyの変更と今回の有限復旧module追加だけで、注文・約定・会計・推論、
入力/model、依存・lock・環境repo・実験条件の変更は拒否する。両口座を再監査しても旧runtimeを書き換えない。

登録と実装をマージした後にclean runtimeでsealする。元2口座の再監査が完了して初めて、
元台帳の後ろに相当する304セルを登録順に一度だけ予約・推論する。記録可能な例外時は停止して保存し、自動retryしない。
保存成果物のverifyは旧・新runtime、元口座hash、全口座、台帳、入力identity、incidentを照合する。
全306セルの照合後にのみ元と同じ9比較を復旧reportへ出す。元attemptの障害履歴は常に残す。

## Consequences

元2口座の再推論を避け、通常口座の総消費上限306を維持できる。新たな学習・探索・quote口座は有効化しない。
新旧runtimeを併記し、許可した差分だけであることを検証する必要がある。
条件不変や全件再監査が成立しない場合、再利用・継続を許可せず別登録の判断に戻す。
これは今回の1 attempt専用であり、汎用resume基盤や自動的な障害解消機構は作らない。
