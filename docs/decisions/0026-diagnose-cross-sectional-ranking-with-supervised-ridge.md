# 0026. longf観測窓からone-step relative-return datasetを構築する

## Status

accepted

## Context

`longf`のgeneric PPO改善、履歴量拡大、sparse rank allocation、regime診断、apply/hold
gateでは、既存特徴量からout-of-sampleのpair orderingを安定して学習できる機構を確立できなかった。
一方、`mom24`のcross-sectional reversal ruleにはgross alphaが観測されている。逐次信用割当、
portfolio sizing、transaction cost最適化を切り離さない限り、PPOの失敗が問題設定に由来するのか、
特徴量とlabelの組み合わせ自体が学習不能なのかを区別できない。

## Decision

17 expanding walk-forward fold（evaluation year 2009--2025）について、`longf`と同一の未正規化
market観測から、decision timestampごと・pairごとの1行を作る。入力は各pairの32 decision lag×8
market特徴量を古いlagから現在へ安定した順序で展開した256列とする。cross-sectional特徴量も各lagで
既存計算値を使用する。account state、pair identity、将来値を入力に含めない。

primary targetは`t`から次decision timestamp`t+1`までのpair log returnから、同じdecisionにおける
9-pair平均を引いたrelative returnとする。feature decision timestampとtarget end timestampを
artifactへ明示する。各train/validation/evaluation rangeは既存環境と同じく独立にfeature warmupと
observation windowを適用し、既存`longf` evaluation decisionとの一致を維持する。

## Consequences

- `longf`に与えたmarket historyを落とさず、pair共通の教師ありmappingを学習できる
- featureとlabelの時刻境界を成果物とテストの両方で検査できる
- pair固有のmemorizationやportfolio/account stateへの依存はこのdatasetでは検証できない
