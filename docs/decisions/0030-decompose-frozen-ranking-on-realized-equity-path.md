# Title

0030. 凍結rankingを実現equity経路上で段階分解する

## Status

accepted

## Context

Issue #19は#15のpredictive evidenceと#16のportfolio evidenceの差を診断する。
再学習、配分再選択、集計法探索をすると不確実性の変化を説明できない。

## Decision

明示manifestで#15/#16のprovenance SHA-256を固定し、全sealed artifactと
config/data/PPO modelの連鎖を検証する。元dataからdecision集合、price return、carryを
再構築するがモデルは学習・推論しない。時刻、pair順、fold集合の不一致は例外にする。

主系列は順に、1.6×relative log-return tail spread、実効weight×price log return、
log1p(実効weight×price simple returnの総和)、signed carryを含む既存gross log return、
既存net log returnとする。最初の2系列の差はfloat32配分精度を明示するため保存する。
全系列を各fold内で合計し、実際の経過年数（365.25日）で割ったannual log returnで比較する。
段階間の差はこの同じ単位で比較する。元のdecision平均tail spreadと
expm1(fold annual gross/net log return)のfold算術平均も元定義で再現する。
fold重みはすべて等重み。日次平均、年率化、expm1の非線形性による差を区別する。

price-onlyは実現経路のstep開始equityを分母にしたprice寄与の連鎖であり、独立の無コスト口座ではない。
各stepでprice+carry-cost=netのsimple-return会計を照合する。
carryはmark-to-market後signed exposureに年率carryと実経過日数/365を掛ける。
会計照合は既存ADR-0029と同じrtol=1e-10、atol=1e-12（無次元return）とする。
これは9 pairの倍精度演算・equity比の丸めを吸収し、float32 weightを理想値へ置換しない許容誤差である。
累積値の許容絶対誤差はstep数×1e-12、年率化時は経過年数で割る。

17 fold、両era、全leave-one-fold-out、最大絶対寄与と上位3 fold除外を報告する。
10,000 IID fold bootstrap、3-fold circular moving-block、seed 16の同じindexを
各段階と段階間差へ適用する。元定義のCIも併記するが、単位の違う下限は直接比較しない。
canonical reversalとdirect PPOは既存対照として全step会計・元集計を再現する。
#16の分類をそのまま継承し、診断でcost-limited/tradableへ昇格させない。

## Consequences

- source hashを継承したCSV/JSON/Markdownから会計と統計の変化を監査できる。
- price logの加重和はportfolio log returnの近似であり、simple会計とは区別できる。
- 結果が未解決でも診断は完了でき、独立検証に必要な不足証拠を記録できる。
