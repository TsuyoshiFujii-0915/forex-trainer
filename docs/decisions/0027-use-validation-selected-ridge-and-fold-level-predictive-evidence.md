# 0027. train-only standardizationとvalidation-selected ridgeを用いる

## Status

accepted

## Context

ADR-0026のdatasetでranking可能性を診断するには、sequential credit assignmentやmodel capacity searchを
持ち込まず、evaluationによるmodel/parameter選択を防ぐ必要がある。

## Decision

モデルはpair identityを含まないpooled linear ridgeとする。standardizationはtraining rangeだけで
fitし、validation/evaluationへ変更せず適用する。alpha gridは`0.0, 0.1, 1.0, 10.0`に固定し、
直前6か月validationのmean cross-sectional Spearman rank ICだけで選択する。同値では強い
regularizationを選ぶ。evaluationはmodel、alpha、特徴量、targetの選択に使用しない。validationで
score rankingが定義できないalphaは選択対象外とし、全alphaが未定義の場合だけ最強のregularizationを
診断出力用modelとして固定し、最終分類を`not established`にする。

## Consequences

- train-only変換とvalidation-only alpha選択によりevaluation leakageを機械的に排除できる
- linear model以外のcapacity、portfolio translation、transaction cost最適化は別の決定を必要とする
