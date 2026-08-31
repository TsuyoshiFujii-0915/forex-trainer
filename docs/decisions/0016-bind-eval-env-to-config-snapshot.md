# 0016. 評価envをconfig snapshotから再構築して照合する

## Status

accepted

## Context

`env_eval.yaml`と`config_snapshot.yaml`を個別にhashしても、両者が同じ評価条件を表す保証には
ならない。評価前にeval envが別data、cost、spread、environment parameterへ差し替わると、
metricsは差し替え後の条件で生成される一方、config由来のdata identityとprotocolをmanifestへ
記録できてしまう。またstandard runでは、reportがtraining provenanceとして読む`meta.json`が
評価manifestに封印されていなかった。

## Decision

standard評価とensemble評価は、config snapshotの`env`と`eval_range`からdeterministic full-range
eval envを再構築し、保存済み`env_eval.yaml`との完全一致を評価開始前に要求する。`forex-report`
でも同じ意味的照合を繰り返す。standard `evaluation.json`は`meta.json`のSHA-256も記録し、report時に
照合する。個別file hashだけを意味的整合性の代用にしない。

## Consequences

- metricsを生成したdata、cost、spread、environment parameterがconfig provenanceと一致する
- configとeval envを整合しない組み合わせで再封印してもgeneric reportを通過できない
- evaluation後にstandard runのtraining metadataを編集すると検出される
