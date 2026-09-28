# Title

0046. 短期の共通方向実験を中期RLとは独立した入力gateで登録する

## Status

accepted

## Context

Issue #40 v2の442ケース中97不足は、短期外側分割と未選定の中期cross-fit/PPO条件を一緒に
停止条件としていた。Issue #55は既存bytes・17評価区間を保持した有限calendar検査で短期の
開始契約を確定し、中期の準備未完了を短期の依存から外すことを要求する。

## Decision

`issue40-common-direction-development-v3`でshort_data_ready、rl_data_ready、execution_readyを
分離する。短期の各foldはLondon年初cutoffから6/12/18/24か月だけを検査し、1日/5日の双方が
train≥252・validation≥60になる最短の共通境界を選ぶ。train終端もvalidation開始へ移す。
63本の先行history、gap非横断、label終点purge、元の評価区間と経済的採否を維持する。
中期の過去のみの日程・PPO連続validation・必要訓練量・予算は、#43の選定後に#46で別途登録する。

本決定はADR-0045の共通方向の有限開発campaignを具体化する追加決定であり、同ADR自体は
supersededにしない。v2 manifestの再開条件だけを新revisionで置き換える。旧manifest・台帳・
sealed preflightは元bytesで保持し、既存計数関数を使う短期用entry pointと成果物だけを追加する。

## Consequences

- 短期ready／中期blockedを表現でき、#43は中期の完了を待たず候補を判断できる。
- Calendar適格性は価格/featureの有限性、学習成功、本実行の許可を意味しない。レビュー・
  マージ後も#41/#42/#43の実装検査と実行直前sealが必要。
- 4候補が不成立なら学習を停止し、新data identityで有限の入力回復を登録する単一判断案を残す。
  最低件数・history・foldを自動で緩めず、追加候補を探索しない。
- #29/#30旧通年の未完了と後継153/85口座を区別して保持し、旧通年復旧を短期の依存に戻さない。
