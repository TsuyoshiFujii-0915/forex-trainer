# 0022. generic reportで単独configurationの記述的証拠を許可する

## Status

accepted

## Context

ADR-0012の`forex-report`はconfiguration差の採否を主目的として最低2構成とpaired comparisonを
要求した。Issue #5では、legacy成果物をgeneric evidenceへbackfillせず、再学習したcurrent
`longf ens3`単独についてfold、era、bootstrap区間をまず確定する必要がある。比較対象を偽造して
行列条件を満たすことは研究証拠を歪める。

## Decision

campaignは1件以上のconfigurationと、0件以上のcomparisonを受け付ける。comparisonが0件でも、
各configurationのartifact/provenance検証、fold/seed集約、era集約、IID fold bootstrap、moving-block
bootstrap、trial数記録をすべて実行する。paired comparisonの検証と出力はcomparisonが宣言された
場合だけ従来どおり行う。

## Consequences

- current baseline単独の記述的generic evidenceを偽の対照群なしで生成できる
- configurationの相対採否には従来どおり明示したpaired comparisonが必要である
- 既存の複数configuration campaignと比較計算の振る舞いは変わらない
