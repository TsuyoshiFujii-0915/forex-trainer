# Title

0033. 凍結portfolioの回転原因を時価評価後exposureで診断する

## Status

superseded by [ADR-0034](0034-separate-overnight-fees-from-trading-costs.md)

## Context

Issue #21は順位交代と実売買costの関係を必要とする。target-weight差は時価評価後の
保有額との差ではなく、membership退出率とも単位が異なる。#19の会計は再利用する。

## Decision

#19の明示manifestと成果物sealをpinし、既存診断を再実行して元steps・fold metricsと照合する。
#15 score、#16実効weight/pair/step、#19会計をfold・decision/target時刻・pair順で完全一致させる。
前decision開始equityをE、weightをw、pair price simple寄与をPとすると、次の売買前exposureは
E(w+P)。初期exposureは0。現decisionのE'w'との差の絶対値を実売買JPY notionalとする。
環境のPortfolioAccount.rebalanceへ同じexposureを渡してspread/commissionを照合し、
保存済みtotal costともreturn単位rtol=1e-10、atol=1e-12で照合する。
本campaignはsigned carry、overnight_rate=0を必須とし、他契約へ暗黙適用しない。

pairの排他的状態をinitial_entry、entry、exit、reversal、retained、inactiveに固定する。
retainedの売買は価格変動とequity変化（carry/costを含む）後の同一targetへの再調整であり、
純粋な価格効果や独立したexecution効果と解釈しない。各状態へ実notionalとspread/commissionを
そのまま割り当てる。signed target変更とdriftを別列に保存するが、絶対値の加算分解はしない。

主診断は隣接decisionのcross-sectional Pearson score相関、平均同順位rankのSpearman相関、
score変更pair数、#16のordinal rank churn、4 slotからの退出率（初期除外）、entry/exit/反転数、
上下境界のscore gap（昇順3位−2位、8位−7位）、符号付き所属の連続decision期間。
相関の定数入力はundefinedとして件数と理由を明記し0へ置換しない。初回相関も定義しない。
所属期間はfold初期のflatから数え、最終decisionまで所属する期間を右打切りとして保存する。
foldを跨ぐ連結はしない。打切り期間を完了期間として平均しない。

集計はfold等重み、2009–2018と2019–2025。主指標の不確実性は#19と同じ10,000 IID fold／
3-fold circular moving-block、seed 16とし、同一indexを共有する。各状態の次期price/gross/netは
既存next-decision horizonだけの記述的関連としてfold内集計する。pair net寄与はpair grossから
当該pairのcost/Eを引いたsimple寄与であり、独立口座収益ではない。decision全体のmembership
維持／変更による次期収益と境界gap対退出率・costのfold内相関は探索的sliceとして別出力する。
未観測状態は件数0・平均nullとし、観測fold数を明示する。日次/pairを独立標本にしない。

既存分類を維持し、gross不確実性が残る限り判定を「未解決」とする。低回転化の収益性、
gate/bandit採用を主張せず、次実験はADR-0031と#20の別途事前登録を必要とする。

## Consequences

- 既存artifactだけで実売買額を復元でき、policy replayや再学習は不要となる。
- 会計残差、未定義相関、未観測slice、打切りを明示した監査可能なCSV/JSONを残せる。
- 配分、horizon、特徴、score、評価環境の挙動を変更しない。
