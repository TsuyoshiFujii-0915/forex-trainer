# Issue #30後継：実行時エラーの保存契約

2026-09-26、PR #53のレビュー対応で追加した運用上の停止契約。
[元の事前登録](registration.md)・ADR-0044と保存済み成功attemptのbytes/hashは変更しない。
市場評価・モデル再学習を追加するものではない。

## 最初の実行時エラーで停止する

入力preflightが成功した後、元方策または対照のpolicy replay、あるいはtrain replayが
`RuntimeError`（その派生型を含む）で失敗した場合、当該foldに`execution-error.json`を保存する。
fold、処理段階、policy（train操作ではnull）、例外型、message、元traceback/原因連鎖を記録する。
RuntimeErrorを捕捉する範囲はreplay呼び出しに限定する。input hash、会計、親再現の検証不一致や
保存自体の失敗はそのまま例外として伝播し、正常な未完了reportへ変換しない。
プロセス強制終了・ディスク故障時にもmanifestが必ず残るという保証はしない。

既に保存した口座とtrain traceは保持し、失敗口座の部分経路を完全な成績やゼロ成績として採用しない。
失敗後は同foldの残りと後続foldを実行せず、再試行もしない。
85セルの状態は次のように区別する。

| 状況 | 状態 |
|---|---|
| 保存済みの完走口座 | `complete` |
| 推論などで失敗した評価口座 | `execution_error` |
| 元方策の再現失敗で実行できない2対照 | `blocked_dependency` |
| train replay失敗で配分を作れないconstant | `blocked_dependency` |
| 依存条件とは別に、停止方針で実行しなかった口座 | `not_run` |

train replayは評価口座とは別の操作であり、エラーの本体を上記JSONに記録し、constantセルから参照する。
train失敗時のcommon_projected、後続foldの全方策は`not_run`。
margin callと`blocked_train_terminal`の既存の意味は維持する。

## 保存・検証・終了

完走口座の帰属stepと85セルの`report.json`/`folds.csv`、未実行セルから失敗元への参照、
最終manifestを保存する。未完了の全体/era/bootstrap/LOO集計はnull。
CLIの`run`も`verify`も未完了パネルではexit 2を返す。

verifyは失敗JSONのhashだけでなく、失敗前の保存順、失敗段階とセル状態の整合、
後続foldに口座成果物が存在しないこと、必要なtrain traceの有無を検査する。
成功済み口座の会計・親再現・対照action・step帰属と集計の検証は引き続き実行する。
未完了成果物へのverifyは推論を行わない。同じ出力先への再実行・上書きは拒否する。

## 回帰検証

合成市場と保存した実PPOモデルを使い、2009 foldを完走した後、2010 foldの
PPO再現・train replay・common_projected推論で、実際の環境stepを経てRuntimeErrorを発生させる。
評価器・会計・保存処理をmockせず、成功口座のbytes保全、全85状態、集約null、
後続foldの未実行、未完了verify、非0終了、エラー証跡の改変拒否まで確認する。
input hash不一致と親会計の不一致はfail-fastのままであることも別テストで検証する。

2026-09-26の検証結果:

- `.venv/bin/python -m pytest -q`：491件通過（例外経路・fail-fastの新規5件を含む）。
- `.venv/bin/python -m forex_trainer.successor_attribution verify --output docs/research/results/issue30-successor`：85/85 complete、exit 0。
- 既存manifest内の193成果物をSHA-256で再照合し、全bytesが変更されていないことを確認。
- 保存済み市場campaignの再実行0。旧登録・config・研究判断記録の変更0。
