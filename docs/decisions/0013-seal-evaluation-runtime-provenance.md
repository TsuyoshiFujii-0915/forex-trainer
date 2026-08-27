# 0013. 評価実行時のprovenanceをversion付きmanifestへ固定する

## Status

accepted

## Context

学習時の`meta.json`だけでは、後から実行する`forex-eval`が実際に使ったdevice、Git revision、
依存versionを特定できない。特に`device: auto`は評価時に再解決されるため、学習時の解決結果を
評価結果へ流用すると、異なる実行環境で生成した指標を同一条件として扱ってしまう。また、
metricsだけをhashしても、評価に使ったconfig snapshotや解決済みeval envの差し替えを検出できない。

## Decision

`forex-eval`はmodelをloadする直前にdeviceを一度だけ解決し、その値をloadとmanifest記録の両方に
使う。`evaluation.json`の`manifest_version`を2とし、model、metrics、config snapshot、解決済み
eval envのSHA-256に加え、評価時の解決済みdevice、両repositoryのGit SHA、主要依存version、
data identityを記録する。`forex-report`はversion 2の全項目を必須とし、現在のファイルおよび
学習時data identityと照合する。旧manifestを暗黙補完しない。

## Consequences

- 指標を生成した評価環境を学習環境と区別して追跡できる
- 評価入力や出力の差し替えをreport生成前に検出できる
- 旧成果物はstudy固有の既存出力では参照できるが、generic reportへ入れるには再評価が必要になる
