# Issue #45 監査修正と未実行304口座の有限継続

## 現在の状態

[PR #61のコメント](https://github.com/TsuyoshiFujii-0915/forex-trainer/pull/61#issuecomment-6141837194)に沿い、
独立数値照合と保存identity照合を分離した。元の保存2口座は、修正版で全171遷移ずつの監査に通過した。
元campaignのincident、execution_error、消費2、口座bytesとruntimeは変更していない。
cleanな検証commit `c0e0fd09023fea71c296d5492b9106eb5e8460c0`から取得した
[正式な再監査記録](results/issue45/recovery-preflight/recovery.json)に、元manifest/account hash、旧・新runtime、
各171遷移の監査成功、推論・口座再実行0を保存した。

**本PRは修正と一度限りの継続登録であり、新規市場口座はまだ実行していない。**
コメントで要求されたレビュー・マージ前の有効新規口座予算は0。マージ後に、その正確なmerge HEADで
clean runtimeをsealして初めて未実行304口座を開始できる。旧2口座を再推論する計画ではない。

[登録JSON](../../configs/research/issue45_audit_continuation.json)と
[ADR-0051](../decisions/0051-continue-only-unexecuted-proxy-accounts.md)を参照。
元の実行・停止結果は[初回レポート](results/issue45/report.md)に残す。

## 監査修正

`audit_proxy`は元価格・carry・費用、前の数量・累積equity、売買、PnL、financing、markupを独立に再計算する。
数値照合の`rtol=1e-12 / atol=1e-8`は変更しない。監査用累積equityを保存値で置き換えない。

数値照合済みの`equity_at_decision` / `held_quantities`と元時刻・pair順・価格・carry・費用から
保存判断Stateを復元し、`freeze_order`へ渡す。ここでは従来どおりState SHA-256、注文hash、期限、数量を
完全一致で検証する。丸めたcash identityやhash検証の無効化は導入しない。

停止した元traceの最小prefix（最初の4遷移）を使い、1 ULPの独立再計算差があっても正しい保存判断を受け付け、
equity/PnL/financing/markup/spreadの許容値超過、別decision hash、注文数量・時刻改変を拒否する回帰テストを追加した。
その後、保存済み同close・next-closeの全171遷移ずつを読取専用で再監査した。推論・fit・口座再実行は0。

## 固定する復旧・継続条件

| 項目 | 固定値・規則 |
|---|---|
| 元実行commit | `fcec20ab3a90763661a31c691967bbd8f47979ed` |
| 元manifest SHA-256 | `03d8e5e59a2873c50299abc385be58a3f219c75a66af94e596c313f4afe1c9c9` |
| 再利用口座 | `2009-F0-canonical-same_close` / `2009-F0-canonical-next_close`、圧縮ファイルhashを登録に固定 |
| 元消費 | 2（元execution_errorも消費済み） |
| 全セル | 元の306セル、順序を維持 |
| 新規実行 | 元の先頭2セルを除く304セルだけ。集合と順序をJSONに明記 |
| 新規上限 / retry / fit | 304 / 0 / 0 |
| quote / 新候補 / 外部市場データ取得 | 0 / 0 / 0 |
| 新規run directory | `runs/issue45-audit-continuation-v1` |
| 新規保存先 | `docs/research/results/issue45/continuation` |

許可するsource差分は`audit_proxy`修正と専用の`execution_sensitivity_recovery.py`追加だけ。
修正版2ファイルのbytes hashを登録に固定する。元`execution_sensitivity.py`から`audit_proxy`を除いたASTが
旧実行commitと一致することも確認するので、`freeze_order`・注文・約定・価格/carry/費用計算・replayの変更は拒否する。

旧runtimeを基準に、trainer commitと登録済み2ファイル以外の全runtime値・source hashは完全一致を要求する。
環境repo、lock、依存バージョン、Python/CPU/platform、元入力/model/登録・実験条件が変われば継続できない。
新runtimeのtrainer commitはレビュー済み継続登録のmerge SHAそのものに限定する。

## 復旧記録・実行・検証

1. 旧manifest/inventory・台帳・report・incident・activation/inputを検証し、旧2口座をすべて再監査する。
2. 再監査証拠を旧manifest/account hashと新検証runtimeに結び付けて`recovery.json`へ保存する。
3. 旧2口座をbyte同一で新directoryへコピーし、台帳に`inherited`として記録する。旧runtime hashも維持する。
4. 残304セルを登録順に一度だけ予約し、既存の方策・engineで独立推論する。旧2セルの再推論はしない。
5. 各新口座を保存して全遷移を監査し、最後にruntime不変を確認する。記録可能な障害はincident・未実行セルとともに保存し例外を再送する。
6. 保存成果物のverifyは旧・新それぞれのruntime、旧ファイルhash、全口座・metrics・台帳・入力identity・report/handoff・独立incidentを照合する。

元2口座の状態は旧campaignでは変えず、復旧campaignでのみ再監査済みのcompleteとして扱う。
原本と継続先のどちらかが欠落・改変されれば照合は失敗する。
旧実行CLIの消費済み拒否は変更せず、新CLIも新run directoryまたはコミット済み継続先が存在すれば追加予約を拒否する。

全306セルの監査と条件照合が成立した後にのみ、元と同じ9比較を復旧reportに出す。
strategy terminalで17foldがそろわない比較は`pending_measurement`、新incidentがあれば全9比較を`pending_integrity`として
net/gross・cash/risk集約をnullにする。旧attemptが失敗した履歴は復旧後も残す。

## 実行手順

実行前の読取専用再監査は、実装・登録をcommitしたclean runtimeで行う。

```bash
uv run --locked python -m forex_trainer.execution_sensitivity_recovery reaudit \
  --config configs/research/issue45_audit_continuation.json \
  --output runs/issue45-recovery-reaudit-v1
```

以下はレビュー・マージ後にのみ使用する。既存CLIの再実行や旧directoryの削除は行わない。

```bash
git fetch origin
git switch --detach <continuation-registration-merge-SHA>
uv sync --locked
uv run --locked python -m forex_trainer.execution_sensitivity_recovery seal \
  --config configs/research/issue45_audit_continuation.json \
  --registration-merge-commit <continuation-registration-merge-SHA> \
  --output runs/issue45-continuation-activation-v1
uv run --locked python -m forex_trainer.execution_sensitivity_recovery run \
  --config configs/research/issue45_audit_continuation.json \
  --activation runs/issue45-continuation-activation-v1/activation.json
uv run --locked python -m forex_trainer.execution_sensitivity_recovery verify \
  --config configs/research/issue45_audit_continuation.json \
  --directory runs/issue45-audit-continuation-v1
```

未マージのcommit、source・環境変更、再監査失敗はseal/runで明示的に拒否する。
追加障害時にこの登録を再利用して再試行しない。

## 検証と引渡し

実データの11回帰ケースと、合成データの有限継続・改変検知19ケースを追加した。
継続テストでは実際の会計・未学習fixtureモデル・保存・監査を使用し、非決定的なruntime取得だけを差し替える。
元bytes保持、304新規消費、旧runtime保持、順序/予算変更拒否、未マージ拒否、runtime/台帳/復旧記録/incidentの
再hash後改変拒否を確認する。全体テストは715件が通過した。
[検証記録](results/issue45/recovery-preflight/validation.json)には、元証拠20ファイルのbytes不変と、
実際の未マージcommitによるsealが拒否されactivationを作成しなかったことも記録した。
[元campaignのverify出力](results/issue45/recovery-preflight/original-verification.json)はPR #61時点とbyte同一で、
引き続き`proxy_status=incomplete`・消費2・元incident非nullである。

#43の短期判断は変更せず、#48の新PPO候補検証は0。#50には実観測quote・provider/商品/lot/financing/depth・
非同期basket・部分約定を未解決として引き渡す。実quoteはblocked_inputのままで、継続の完走だけでは#45全体をcloseしない。
