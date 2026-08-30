# 0018. legacy longfはgeneric reportの前に再学習する

## Status

accepted

## Context

ADR-0013とADR-0015では旧`longf ens3`をversion 2成果物へ再生成してgeneric reportへ移す方針を
示した。しかし旧source runの`meta.json`には`requested_device`と`data_identity`がなく、現在の
`forex-eval`で再評価してもtraining-time provenanceは復元できない。現在値をmetaへ書き足す
backfillは、過去の学習条件を検証せずに捏造することになる。

## Decision

旧`longf ens3`をgeneric campaignのbaselineへ移す場合は、source fold/seed runを現行training
provenance契約で再学習し、その後に現行`forex-eval`と`forex-ensemble-eval`でversion 2成果物を
生成する。評価CLIはcurrent training provenanceが欠けたsource runを早期拒否する。再学習が完了する
までは旧成果物をstudy-specific evidenceとしてのみ扱い、generic reportへ暗黙fallbackしない。

## Consequences

- training-time provenanceを事後推測せず、generic evidenceの検証可能性を維持する
- legacy baselineのgeneric移行には再学習コストが必要になる
- 本決定はADR-0013とADR-0015のlegacy移行手順を明確化し、単純な再評価方針を置き換える
