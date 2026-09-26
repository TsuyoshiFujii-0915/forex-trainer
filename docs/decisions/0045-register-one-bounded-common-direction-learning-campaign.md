# Title

0045. 共通方向の予測・配分・逐次制御を一つの有限開発campaignとして登録する

## Status

accepted

## Context

Issue #39の出口Bを受けた#29/#30後継は、固定17区間の153/85口座を完走した。
PPOの平均price寄与は共通方向が主要だが、持続的alphaと同リスクの学習追加価値は未識別。
旧17通年の欠損は残り、ADR-0036の新規学習0は測定・帰属診断の予算である。
Issue #40は次期の有限学習予算を別versionで登録することを求める。

## Decision

[issue40-common-direction-development-v1](../research/26-common-direction-learning-campaign.md)
を追加し、#39の範囲を継承した共通方向の一分岐だけを実行対象にする。
当日特徴量から固定basketの1/5営業日price returnをridgeで予測し、2予測×2配分を比較する。
中期へ進む場合も最大1予測系、同じ予測を使う1小型PPOと決定論的/greedy対照に限定する。
数値risk、採否、停止、tie-break、cross-fitを含む全fit・再試行上限をmanifestに固定する。
新しい学習/配分/quantity会計の実装・市場実行は#41以降の担当であり、本ADRでは実行しない。

これはADR-0031/0036/0042〜0044に追加する開発契約であり、既存ADRをsupersededにしない。
旧#27の学習0、#15/#16の分類、#20の独立確認契約、旧sealed成果物は保持する。
新しいquantity holdは別measurementとして実装・検証し、ADR-0004/0025のtarget再送を変更しない。
対照は同じmeasurement/runtimeで再評価し、ADR-0017の比較検証を緩めない。

## Consequences

- 部分年9区間を含む既知development期間での方法選択であり、独立収益性の認定ではない。
- 固定予測を消費するRLはhybridであり、旧pure RLの自力alpha改善と区別する。
- 共通方向仮説が失敗しても同campaignで横断分岐へ切り替えず、停止理由を記録する。
- 後続の実装SHAや実sourceは各開始前に封印する。将来成果物のhashを現在値で補完しない。
- manifest確定で#40は完了でき、後続実験の好成績や将来観測の終了を待たない。
