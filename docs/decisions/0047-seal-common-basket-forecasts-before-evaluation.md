# Title

0047. 共通basketの候補モデルを評価前に封印し、口座decisionと診断labelを分離する

## Status

accepted

## Context

Issue #41はADR-0045/0046の共通方向ridgeを具体化する。旧#15はpair相対targetと
32-lagの別契約であり、変更すると旧成果物の意味が変わる。h=5の末尾label不足を
予測不足と混同すると、#42/#43の口座区間が事後的に短縮される。

## Decision

専用の`forex-common-basket` CLIを追加する。v3の採用境界と元cache・派生cache・calendarを
照合し、全34ケースの実値検査をfit前に行う。train-only標準化、mean-loss ridgeの3候補と
train-only縮小共分散を保存し、validationで採用した候補を再fitせずモデルファイルに封印する。
全34モデル保存後にのみ評価予測を生成する。口座decisionは固定calendarの最終mark直前まで、
診断targetは各派生cache内で利用可能なh営業日終点だけを用いる。予測bundleの欠落・重複は例外。

preflightは学習を行わない。市場fitはv3登録commitを含むマージcommit、両repoのclean SHA、
入力・config・lock・依存version・CPUのsealを検証してから開始する。通常102fitはcampaign固定の
実行ディレクトリに一度だけ予約し、各attemptは計算前に台帳へ追記する。失敗も消費として残し、
自動retryは行わない。再開には共有retry枠を使う別の明示登録が必要で、出力先変更で枠を再取得しない。

## Consequences

- 旧#15・旧PPO・v1/v2/v3登録成果物は変更しない。
- short gateのreadyと本学習可能を区別し、未マージなら実装・入力検査完了までを報告する。
- #42のfixed/costは同じhash検証済みbundleを読む。#41はportfolio採否を決めない。
- 定数列、非有限値、入力差異、予測欠損を既定値で救済しない。
- 部分年、修復価格、carry vintage不明、同close仮定は従来どおり開発証拠の限界として残る。
