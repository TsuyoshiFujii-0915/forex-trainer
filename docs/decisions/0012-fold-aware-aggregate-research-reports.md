# 0012. fold単位の集約レポートを研究判断の標準証拠とする

## Status

accepted

## Context

`forex-compare` は評価済みrunをSharpe順に並べるため、複数fold・複数seedを使う現在の
研究で、構成差が市場期間をまたいで再現するかを判断できない。Issue #1のcheckpoint選択研究と
Issue #7のデータ量研究は、それぞれfold集計、対応差、fold bootstrapを実装したが、個別研究の
成果物に閉じている。同じ統計を手集計または別実装すると、seedを独立した市場履歴として扱う、
対応しない評価期間を差し引く、試した構成数を記録しない、といった判断上の不整合が生じる。

## Decision

今後の研究上の採否判断は、明示したrun artifact群を入力とする`forex-report`の集約結果を
標準証拠とする。campaignは名前付きconfiguration、方向付き比較、era、bootstrap制御、
campaign全体のtrial数を厳格なYAMLで宣言する。fold、seed、評価期間、指標、device、設定、
データ、Git SHA、依存versionは再入力せず、各runの`config_snapshot.yaml`、`meta.json`、
`metrics.json`、`evaluation.json`から取得する。学習開始時にdata identityを`meta.json`へ、
評価時にmodel-selection、model SHA-256、metrics SHA-256を`evaluation.json`へ固定し、report時の
ファイルと照合する。campaignの`model_selection`は期待値であり、artifactの代替にはしない。

比較は完全に一致するfold×seed行列と実効評価開始・終了時刻だけを対応付ける。集計は最初に
seedをfold内平均し、市場履歴の標本をfoldとする。95%区間はfold再標本化と、時代の連続性を
残すcircular moving-block再標本化を併記する。日次barやseed行を独立した市場標本として
再標本化しない。net/gross年率returnは、成果物世代に依存しないよう累積log returnと実効評価
期間から一意に計算する。

data identity、解決済みdevice、model-selection条件が比較群で一致しなければレポート生成を
失敗させる。configuration内では、fold固有名、三つのdate range、seed以外のconfig snapshotが
同一でなければ失敗させる。Git SHA、依存version、要求device、意図したconfiguration差は
treatment provenanceとして比較結果に記録し、実装変更の比較そのものは妨げない。

trial数と報告configuration数は常に記録する。fold集約artifactだけではreturn分布、自己相関、
独立trial数の仮定を正当化できないため、probabilistic/deflated Sharpeは算出せず、その制約を
機械可読レポートとMarkdownの両方に明記する。

## Consequences

- 単一runの好成績ではなく、fold・seed・eraを通した再現性で変更を判断できる
- 対応比較と不確実性区間が全研究で同じ標本単位と実装を使う
- campaign YAMLにrunを明示するため、「最新run」の暗黙選択や欠損セルの暗黙除外がない
- data/device/model-selectionの交絡は集計前に拒否され、config/software差は比較結果に明示される
- 多重試行による選択バイアスはtrial数として可視化されるが、deflated Sharpeの確率値は
  基礎仮定を満たす追加データがない限り得られない
- `forex-compare`は個別runの診断用途として残るが、研究上の採否根拠には使わない
