# Issue #45 固定bar proxy実行・比較レポート

## 結果（2026-10-13 JST）

**登録済み306口座の実行を開始したが、2口座目の保存後監査で停止した。完了1、execution_error 1、未実行304。**
固定bar proxyは`incomplete`、全9比較は`pending_integrity`であり、net/gross差やCIを報告できる状態ではない。
実観測quoteは引き続き`blocked_input`（消費0）。Issue #45全体も未完了である。

元の口座、障害、台帳、全306セル、manifestを変更せず保存した。
このPRで実装・登録・元modelを変更せず、再実行・継続・fit・PPO学習・外部市場データ取得は行っていない。

## 固定した環境と予算

| 項目 | 記録 |
|---|---|
| 事前登録・実装PR | [PR #60](https://github.com/TsuyoshiFujii-0915/forex-trainer/pull/60) |
| 実行時forex-trainer HEAD | `fcec20ab3a90763661a31c691967bbd8f47979ed`（PR #60のマージコミット） |
| 実行時forex-env-v3 HEAD | `6024b91c0f3592611849bc231922ab60e6090aed` |
| 登録SHA-256 | `6678a37ac38695e16d7a2cd0af50ad8989ecf240b949970b0758a9a3f0f36819` |
| uv.lock SHA-256 | `a1baf53c15f9681ccf5a9ed79a2dd834dccecce6aa82b3e78491fcc4ce49cd82` |
| proxy通常上限 | 17区間 × 3方策 × F0/F1/F2 × 2執行 = 306口座 |
| 消費・未実行 | 消費2（失敗1を含む）、未実行304、retry 0 |
| quote有効予算・消費 | 0 / 0（上限6は未有効化） |

`uv sync --locked`後、両repoがcleanな状態でsealし、同じHEAD・source bytes・依存バージョンでrunした。
[activation.json](campaign/activation.json)に入力/model/実行環境のidentityと全登録セルを保存している。
障害後も成果物の公開コピー前にruntimeを取得し、sealとの完全一致を確認した
（[post-failure-runtime.json](post-failure-runtime.json)）。
この障害は全口座後の最終runtimeチェックに到達する前に発生したため、後者の記録は独立した障害後照合である。

Python 3.11.15、NumPy 2.4.6、pandas 3.0.3、torch 2.12.0、stable-baselines3/sb3-contrib 2.8.0、CPU実行。
登録済み17区間には部分年9区間を含み、17通年に読み替えない。

## 全セル・比較の状態

| 状態 | 口座数 |
|---|---:|
| complete | 1 |
| execution_error | 1 |
| not_run | 304 |
| strategy_terminal | 0 |
| blocked_input（proxy） | 0 |
| 合計 | 306 |

| セル | campaign上の状態 | 保存された口座trace |
|---|---|---|
| 2009 / F0 / canonical / same_close | complete | 171判断、会計・metrics照合済み |
| 2009 / F0 / canonical / next_close | execution_error | 171判断を保存した後、監査で例外 |
| それ以降の登録セル | not_run | 304口座は開始していない |

next-closeの口座ファイル自体には`status=complete`、`remaining_decisions=0`が記録されているが、
これはengineの出力であり、監査通過を意味しない。campaignでは`execution_error`、metricsはnullのまま保持する。
この口座を完了に昇格させたり、同closeと組み合わせて損益差を算出したりしていない。
`incident.partial=null`なのは監査段階のValueErrorだからであり、元traceが存在しないという意味ではない。

| 費用条件 | 方策 | 比較状態 | net差 / gross差 / CI / cash・risk差 |
|---|---|---|---|
| F0 | canonical | pending_integrity | null |
| F0 | ridge | pending_integrity | null |
| F0 | ppo_ens3 | pending_integrity | null |
| F1 | canonical | pending_integrity | null |
| F1 | ridge | pending_integrity | null |
| F1 | ppo_ens3 | pending_integrity | null |
| F2 | canonical | pending_integrity | null |
| F2 | ridge | pending_integrity | null |
| F2 | ppo_ens3 | pending_integrity | null |

本来の比較はnext-close − same-close、17区間等重みperiod log、fold bootstrap 10,000回・seed16、
moving-block長3、era・LOOである。今回は未解決incidentと不足セルがあるため全比較を保留する。
成功区間のintersection、部分結果の年率化、方策や執行の採用判断は作らない。

完了した同close口座の記録のみを以下に示す。これは2009部分区間の単独口座値で、比較結果ではない。

| 指標 | 2009 / F0 / canonical / same_close |
|---|---:|
| net period log | -0.005530593351300005 |
| gross period log（実現経路への費用足戻し） | 0.018165603111294223 |
| price PnL（研究会計JPY） | 26,163.936248 |
| signed financing（研究会計JPY） | -8,332.349017 |
| spread / commission / markup（研究会計JPY） | 8,306.283642 / 0 / 15,040.631364 |
| 実売買notional合計（研究会計JPY） | 218,039,989.190177 |
| mean / max gross exposure | 3.200157190120 / 3.200621262884 |
| 最大drawdown | 0.13656967133632725 |
| decision coverage / fill遅延 | 1.0 / 0秒 |

same-closeにnext-close待機区間はなく、`gap_price_pnl=0`は測定上の定義である。
next-closeのgap PnL・費用・売買量・exposure/MDD・遅延・coverageは元traceに保存されているが、
監査未通過につき比較用metricsへ採用しない。

## 障害の直接原因

停止セルは`2009-F0-canonical-next_close`。監査中の判断時刻
`2009-05-10T23:00:00+00:00`（Londonでは5月11日）で次の例外となった。

```text
ValueError: proxy order at 2009-05-10T23:00:00+00:00: stale or foreign decision
```

[保存traceの読取専用診断](incident-diagnostic.json)で、同じ`audit_proxy`例外をモデル推論・口座再実行なしで再現した。

| 診断項目 | 値 |
|---|---|
| 保存されたdecision equity | `999390.5382721465` |
| 監査で再構成したequity | `999390.5382721466` |
| 差 | `1.1641532182693481e-10`（1 ULP） |
| quantitiesの完全一致 | true |
| 保存状態からのhashとdecision.state_sha256の一致 | true |
| 監査で再構成した状態hashとの一致 | false |

実行側はPortfolioAccountの価格比によるPnLと段階的なequity更新を使い、監査側は数量×価格差と
独立した加減算で再構成する。数値照合は許容誤差付きで通っても、その再構成状態を
`freeze_order`へ渡す際にはequityを含むSHA-256の完全一致が要求される。
このケースでは1 ULPの差で正しい保存decisionとの結び付けが拒否された。
これは最初の停止点の説明であり、失敗口座の全会計が検証済みという主張ではない。

修復には、保存された判断状態のidentity検証と、独立した数値会計照合との責務を整理する必要がある。
hashを無視する、全検証を緩める、保存equityで監査誤差を無条件に補正する、といった対応は採らない。
実装修正・実データの最小再現回帰テスト・保存口座の扱い・残304口座の実行条件は、別途レビューが必要である。
元のincident・消費2・失敗セルはこのPRで解消または書き換えない。

## 保存成果物の検証

[verification.json](verification.json)はCLIの出力そのもので、`status=verified`、
`proxy_status=incomplete`、`proxy_accounts_consumed=2`、`accounts_reexecuted=0`、incident非nullである。
これは保存成果物の整合性検証であって、306口座完了や障害口座の会計監査成功を意味しない。

- 8ファイルを元run directoryからbyte同一でコピーし、元manifestも保持した。
- 元run directoryと公開コピーの双方で、hash・全inventory・入力identity・activation・台帳・全306セル・report/handoff・incidentを照合した。
- completeの1口座について会計とmetricsを照合した。execution_error口座の監査通過数は0。
- 公開コピーのverify出力は元runのverify出力とbyte同一だった。
- 検証・診断時の新規モデル推論と口座再実行は0。既存の合成回帰テスト結果は[validation.json](validation.json)を参照。

元campaign manifest: `03d8e5e59a2873c50299abc385be58a3f219c75a66af94e596c313f4afe1c9c9`。
既存のpreflight.json、execution-status.json、manifest.jsonは実行前の歴史的記録として変更せず、
今回の実行状態は[run-summary.json](run-summary.json)へ分離した。

```bash
uv run --locked forex-execution-sensitivity verify \
  --directory docs/research/results/issue45/campaign
```

実行したseal/runコマンドと標準エラーは[実行記録](execution-record.json)・[stderr](execution-stderr.txt)を参照。
runコマンドは履歴用であり再実行用ではない。固定directoryと保存campaignが通常予算の再取得を拒否する。
未実行304口座は自動的に再利用可能な予算ではなく、復旧・残口座の実行には別の明示的登録が必要である。

## #43 / #48 / #50への引渡し

| 宛先 | 今回の引渡し |
|---|---|
| #43 | 既存357口座の短期判断を変更しない。執行感度は未完了で、良化・悪化とも判断できない |
| #48 | 旧3方策用の実験であり、新PPO候補の検証数は0 |
| #50 | proxyの監査障害、実quote入力、商品・lot・financing・depth、非同期basket・部分約定が未解決。候補の実執行可能性を認定しない |

固定bar proxyは同時9pair全量fill・逆数研究quantityを仮定する。provider、実商品数量、lot、
available_at、swap、depthを捏造せず、proxy差を実fillの証明やlatencyの単独因果効果と呼ばない。
実観測quoteは#44の入力と別の参照契約を待つ。新候補の執行確認が必要なら最大1候補の有限trialを別途事前登録し、
#20独立確認や追加口座を自動開始しない。Issue #45はcloseしない。
