# 同closeと固定next-closeの執行感度（Issue #45）

## 現在の状態

**2026-10-15、PR #62のマージコミットで残304口座を実行し、再利用2口座を含む全306口座の検証が完了した。**
固定bar proxyの全9比較はavailable。今回のexecution_error・terminal・未実行・incidentは0。
[継続実行・比較レポート](results/issue45/continuation-results/report.md)、
[全セルと比較値](results/issue45/continuation/report.json)、
[検証結果](results/issue45/continuation-results/verification.json)を参照。
実観測quoteはblocked_inputのためIssue #45全体は未完了である。

2026-10-13の初回attemptは2口座目の保存後監査で停止した。
[初回の障害レポート](results/issue45/report.md)と元campaign・incident・消費2・runtimeは変更せず保存している。

[登録JSON](../../configs/research/issue45_execution_sensitivity.json)を正本とする。
#40 v1/v2/v3の元bytes・停止状態や、#43の結果・予算は変更しない。
[ADR-0050](../decisions/0050-freeze-next-close-proxy-quantities.md)で新測定を定義した。

| 経路 | 測定・範囲 | 口座上限 | 現在の有効予算・状態 |
|---|---|---:|---|
| 固定bar proxy | #39の固定17後継区間、旧3方策×F0/F1/F2×同close/next-close | 306 | 旧2再利用＋新304消費、全306検証済み、retry 0。予算消費済み |
| 実観測quote | #32の固定2025 bundle、旧3方策×同close/quote、F0 | 6 | 0、blocked_input。#44のsource・商品・期間・費用・実観測入力が未確定 |

proxyの入力preflightは17区間・3,569判断時点で通過した。価格・modelファイルを検証・読込するが、
model推論、fit、口座評価、外部取得は行っていない。[保存結果](results/issue45/preflight.json)を参照。
[実行前状態とruntime](results/issue45/execution-status.json)は、市場評価0だった時点の履歴として保存する。
初回attemptの[run-summary](results/issue45/run-summary.json)と[activation](results/issue45/campaign/activation.json)も履歴として保持する。
現在の継続結果は[continuation report](results/issue45/continuation/report.json)と
[継続activation](results/issue45/continuation/activation.json)が正本である。
9区間の部分年は#39のままで、17通年へ読み替えない。既存3方策以外の候補は対象外。

## 固定したproxy会計

- 入力は#39 snapshotの同一bytes、63本の先行history、32×8のcausal windowとcarry。
  London local平日のmidnightを研究上のcloseとする。既知のavailable_atではない。
- canonical、旧凍結ridge、direct PPO ens3（seed42/43/44）を各口座で再推論する。
  初期100万円・flat、gross5・pair1・marginは初期equityの0.2。
- 同closeは#42の`issue40-common-quantity-same-close-v1`をそのまま再実行する。
  #43の口座、actions、metricsは再利用しない。
- next-closeは`issue40-next-close-proxy-v1`。判断時のequityと価格から作る数量に、
  判断時点の既存比例capを適用して凍結する。未来価格・fill equityによる数量の再選択はない。
- 次の期待closeまで旧数量を保有する。price PnLは `q_old*(P_fill-P_decision)`、
  markupは区間開始gross notional×日次率×実UTC日数、signed financingは終了notional×
  当日既知の`-CarryAnnual`×実UTC日数/365。週末・DSTを含む。将来rateは使わない。
- fill時に凍結数量を全量約定し、実delta数量×fill価格へspread/commissionを一度課す。
  F0は旧9pair spread・commission0・日次markup0.00002、F1はspreadだけ2倍、F2はmarkupだけ2倍。
  bid/ask観測値を併用しない。financingにmarkupを内包しない。
- fillでhard capを超えた場合は、全量fill後に別の`risk_fill`比例縮小を記録し、追加費用も計上する。
  これは未来価格で注文数量を書き換える操作ではない。raw proposal、凍結order、full fill、risk取引を保持する。
- gapでmarginになれば注文を`cancelled_terminal`、fill費用でmarginになればfill済みのstrategy terminal。
  以後の判断は未実行として残す。非正equityのlog・weightはnullで、flatに補完しない。
- 最終markに予定されたfillは行うが、新判断や仮想清算はしない。最終markより後の予定注文は
  `expired/after_final_mark`として売買・費用なし。現在の固定scheduleでは通常は全注文が最終markまでに到来する。
  interiorのbar欠損は例外であり、後日のcloseへの繰越・同closeへのfallbackはない。
- PPO assetsは独立口座の直近fill notional/前判断equity、leverage1、旧数量のgap price PnL/前判断equity。
  初回は0。学習時と異なる執行に凍結policyを適用する感度であり、再学習はしない。

次closeは同時9pair全量fillの**研究仮定**。depth、実商品lot、provider holiday、broker rollover、
非同期basket、部分約定は再現していない。逆数研究quantityは実商品のbase quantityではない。

## 実観測quoteとの境界

#31の`quote_replay.replay`、#32のshadow原記録・head検証を接続先として保持する。
公開・取得・判断durable保存後の最初の適格quoteというADR-0040の規則を変更しない。
quoteがなければproxyまたは同closeへ自動代用しない。本CLIは未登録quote口座を開かない。

#44から原記録が届いた後、source・商品・lot/minimum・数量・期間・calendar・費用・financing、
同close参照との情報集合・判断予定を結果閲覧前に別登録する必要がある。
現時点で不明なsource、期間、swap、depth、available_atを捏造せず、登録JSONではnullとblockerを保持する。
provider/商品座標/lot/financing等を揃えられない差は複合感度であり、全差をlatencyの因果効果と呼ばない。
quoteの実収集・6口座の比較はこのPRでは未完了。

## 実行と保存

以下は初回実行手順の履歴であり、消費した通常予算を再取得する手順ではない。
監査修正と有限継続はPR #62で登録・マージし、残304口座を実行済みである。
元run directory・campaignと継続run directory・continuationを保持する。

```bash
# 読み取り専用。outputは未使用のディレクトリを指定する。
uv run forex-execution-sensitivity preflight \
  --config configs/research/issue45_execution_sensitivity.json \
  --output runs/issue45-preflight-v1

# このPRの登録・実装がレビュー・マージされた後にだけ実行可能。
git fetch origin
git switch main
git pull --ff-only origin main
uv run forex-execution-sensitivity seal \
  --config configs/research/issue45_execution_sensitivity.json \
  --registration-merge-commit <このPRの40桁merge-SHA> \
  --output runs/issue45-activation-v1
uv run forex-execution-sensitivity run \
  --config configs/research/issue45_execution_sensitivity.json \
  --activation runs/issue45-activation-v1/activation.json
uv run forex-execution-sensitivity verify \
  --directory runs/issue45-execution-sensitivity-v1
```

registrationと実装のmerge、両repoのclean runtime、入力/model/lock/version、306セルをsealする。
固定run directoryは一度だけ作成し、各口座を推論前にfsync台帳へ予約する。
失敗口座も消費する。retry枠は0で、失敗や不利な結果を理由とした自動再実行は行わない。
別cloneでもコミット済み`results/issue45/campaign`があれば通常予算を再取得できない。

記録可能な例外ではincident、完了trace、最後の口座状態、全未実行セル、台帳、report、manifestを保存し、
原因を含む例外を再送する。OS強制終了やdisk-full等で書込不能になった場合は正常完了とみなさない。
verifyはモデル再推論・口座再実行なしで、保存hash、完了口座の会計、metrics、全セル集計と台帳を照合する。
失敗セルの途中状態はincidentとして保存し、完了口座へ昇格しない。

全口座保存後の最終runtime不一致・例外を含め、未解決のincidentがあればcampaignは
`proxy_status=incomplete`、全9比較は`pending_integrity`とし、net/gross・cash/risk集約をnullにする。
口座個別の完走状態・metrics・元trace・台帳・消費口座数は保持する。handoffにもincidentと
比較保留を伝播させる。verifyは同じ規則でreportを再構成し、hashを再作成して完了・比較可能へ
戻したreportも拒否する。`incident.json`の有無・内容はreportとは独立に確認し、report側のnull化、
incidentファイルの欠落・内容不一致・JSON nullはhashを再作成しても拒否する。
verifyの`status=verified`は保存成果物・会計の照合成功を表し、
実験の完了認定ではない。別途返す`proxy_status`と`incident`で未完了を確認する。

実装テストは`uv run python -m pytest tests/test_execution_sensitivity*.py -q`。
数量・時刻・費用・terminal・例外・保存改変の34ケースを検証する。テストは各実装変更前に追加し、
失敗を確認してから実装を修正した。テスト中に市場口座や旧方策modelの学習は行わない。
runner回帰は合成価格・既存の未学習fixtureモデルで全306セルを保存し、正常完了、strategy terminal、
最終runtime不一致、最終runtime例外を通す。非決定的なruntime取得だけを差し替え、会計・推論・
bootstrap・保存・照合は実実装を使う。異常時も元口座のhash・台帳bytes・消費306が変わらないことを検査する。

報告は各口座のnet/gross period log、gap price PnL、price/carry、spread/commission/markup、実売買量、
exposure/MDD、fill遅延・coverage、terminal・未実行判断数を保持する。
grossは同じ実現経路への費用足戻しであり、費用なしで再推論した反実仮想口座ではない。

3方策×3scenarioの9比較で、next-close − same-closeの17区間等重みperiod logを集計する。
登録済みfold bootstrap（10,000回、seed16）、moving-block（長さ3）、era、LOOを再利用する。
一つでも未完・terminalなら当該比較の17fold推定は保留し、成功区間intersectionを作らない。
CIは記述的・多重性未補正で、方策や執行条件の採用最適化には使わない。
数日のshadowを17foldとしてbootstrapしない。

## #43/#48/#50への引渡し

| 宛先 | 引渡し・未決事項 |
|---|---|
| #43 | 短期357口座の既存判断を変更しない。proxy/quote未完は短期判断の差戻し条件ではなく、実執行可能性の証拠不足 |
| #48 | この306口座は旧direct PPOを対象とし、新PPO候補を検証したと外挿しない。#43で候補0の現状では新候補実行はnot_planned |
| #50 | 実quote/商品適合と候補固有の執行確認は未解決。必要なら最大1候補の別trial・有限口座数を実行前登録。独立確認#20の自動開始は0 |

proxyが良好でも実fillを証明せず、悪化しても全horizonを否定しない。
固定bar proxyは完了したが、実quote入力・候補の追加登録を未完として引き継ぎ、Issue #45を自動closeしない。

## 監査障害後の継続

[監査修正・有限継続登録](32-execution-sensitivity-continuation.md)をPR #62でマージした。
保存済み2口座を全件再監査し、旧bytes・incidentを維持して、未実行304口座だけを実行した。
F0のnet period log差（next-close − same-close）はcanonical -0.045073、ridge +0.011243、PPO -0.014799。
canonicalは全費用条件で両95%区間が負、ridge/PPOは0を跨いだ。全表・解釈の境界は継続レポートを参照。
