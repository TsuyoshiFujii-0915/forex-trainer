# 0013. 旧checkpointの現行環境評価を通常runから分離する

## Status

accepted

## Context

ADR-0011より前のrunは、学習時データのSHA-256やsource worktreeを記録していない。そのため
後から現在のキャッシュを紐付けても、学習時入力を復元したとは証明できない。一方、厳密会計
(forex-env ADR-0015)導入後もR18/R19の定性的結論が維持されるかを調べるには、保存済み
checkpointを現在のデータ・コードへ固定して参考評価する価値がある。

## Decision

旧checkpointは通常の`forex-eval`へ暗黙に受け入れず、
`legacy-checkpoint-current-environment-v1`という別のattestation契約で評価する。
attestationは学習時provenanceを`unverifiable`と明記し、各modelのSHA-256、旧meta、現在の
評価データSHA-256、cache契約、source worktree、package version、現在環境での指標を新規
ディレクトリへ保存する。既存runや既存attestationは上書きしない。

現行provenanceを持つrunはこの経路で評価せず、ADR-0011の通常評価を使う。attestationは
「現在環境で旧方策がどう動くか」の証拠であり、「現在環境で学習した場合の成績」には
読み替えない。

## Consequences

- 過去成果を通常runへ偽装せず、会計変更の感度だけを定量化できる
- 厳密な現行学習結果が必要な場合は、新しいprovenance契約で再学習が必要になる
- 旧runと現行runの数値は契約が異なるため、比較時にattestationであることを明示する
