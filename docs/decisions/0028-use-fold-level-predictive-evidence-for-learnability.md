# 0028. fold-level predictive evidenceでlearnabilityを分類する

## Status

accepted

## Context

ADR-0027で選択したridgeを評価する際、daily decisionやpair rowを独立したmarket sampleとして扱うと
不確実性を過小評価する。また、単一係数の符号だけでは、強く相関するmomentum特徴群から得たscoreが
canonical reversal orderingを実際に再現しているか判定できない。

## Decision

evaluation decisionを完全一致させ、教師ありscore、`-mom24` canonical reversal score、sealed
current-provenance `longf ens3` direct action orderingを記述的に比較する。rank IC、tail spread、
tail ordering accuracy、realized top/bottom overlap、score dispersion、consecutive average-rank churnを
各fold内でdecision集計し、foldをmarket sampling unitとする。宣言した同一seedによる10,000回の
IID fold bootstrapと3-fold circular moving-block bootstrapを併記する。

targetまたはscoreが全pair同値でrankingが定義できないdecisionはrank ICだけから除外し、rank-IC
observation fractionを記録する。return-level診断には含め、未定義ICを0へ置換しない。fold内の全ICが
未定義ならIC集計をnullとしてscoreをdegenerateと判定する。

reversal coherenceは、教師ありscoreとfrozen `-mom24` scoreのmean cross-sectional Spearmanが全foldと
両eraでpositiveかにより判定する。単一momentum係数の符号は判定条件にせず、全expanded featureの
standardized coefficient signs/magnitudesをfold、era、aggregateで別途報告する。

学習可能性の分類はIssue #15で事前登録した6条件をすべて満たす場合だけ`established learnable`、
point estimateとreversal coherenceとleave-one-fold-out方向がpositiveだが、不確実性またはera安定性が
不足する場合は`suggestive`、one-fold dominatedまたはdegenerateを含むそれ以外を`not established`とする。
結果はpredictions、model parameters、fold/era/aggregate diagnostics、source/config/data/Git/dependency
hashを含むversion付きartifactとしてtransactionalに生成する。

## Consequences

- canonical ruleとPPO orderingを同一decision、同一targetで比較できる
- 日次decisionやpair rowを独立したmarket sampleとして過大評価しない
- coefficient multicollinearityだけで経済的coherenceを誤判定しない
- scoreがdegenerateでも未定義統計を偽の0へ変換せず、診断成果物と否定分類を残せる
