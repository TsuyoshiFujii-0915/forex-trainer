# Title

0029. 凍結scoreを固定portfolio mapで同一評価環境へ渡す

## Status

accepted

## Context

Issue #15はestablished learnableとなった。Issue #16は既存情報から得たrankingを実際のcostと
signed carryの下で取引へ変換できるかを検証する。predictionの再学習やportfolio geometryの探索を
入れると、score品質と実現performanceの関係を切り分けられない。

## Decision

Issue #15成果物のprovenance SHA-256をcampaign JSONで固定し、全生成物hash、fold config/data、
PPO source manifest/modelを照合する。sealed CSVをround-trip精度で読み、decision/target timestamp、
pair順序、relative-return target、canonical scoreを再構築したevaluation datasetと完全一致させる。
教師ありscoreはCSVの値を直接消費し、modelを再学習しない。主mapはstable ascending score sortの
末尾2 pairを+0.8、先頭2 pairを-0.8とし、残り0、decision interval 1で固定する。
同点はcanonical pair順を用いる。canonical reversalは既存mom24 ascending sortの先頭2が+0.8、
末尾2が-0.8という同点契約も維持し、実際のenv observationから既存ruleとの配分一致を確認する。

PPOはIssue #15と同じsealed ensembleを再生してsource score/metricsとの一致を確認する。その
固定action trace、教師あり配分、canonical配分を同一実行のevaluation code/Git/dependencyで再評価し、
旧metricsをpaired比較へ混在させない。PPO再学習はしない。全decision/target timestampの一致を必須とし、
早期終了時にintersectionへ縮小しない。float32 direct-action精度は既存benchmarkと揃え、実効weightも照合する。

foldをsampling unitとし、17 foldのannualized returnの算術平均をprimary aggregateとする。
net/grossの絶対値およびpaired差に10,000 IID fold bootstrapと3-fold circular moving-block
bootstrap（seed 16）を適用する。eraは2009--2018と2019--2025。最大絶対寄与fold、その絶対寄与割合、
全leave-one-fold-out平均を報告する。日次観測はbootstrapの独立sampleにしない。

grossは既存compute_metrics同様、当該stepのtransaction costをequityへ加算して復元するためsigned
carryを含む。rank別gross simple return寄与はtarget weight×price returnとmark-to-market後exposureの
signed carryの和とし、毎stepでenvのgross returnと照合する。log寄与はgross simple returnに比例配賦し、
合計がgross log returnになることを明記する。membership turnoverはlong/shortを区別した4 slot中の
前decisionからの退出率で、初期entryは除く。target-weight turnoverは初期entryを含む。

stable positive grossはmean、両era、全leave-one-fold-out mean、両bootstrap下限が正で、全foldの
mean realized gross leverageが1以上であることを要求する。これはgross target 3.2のnear-flat検出である。
tradableはさらにnet mean、全leave-one-fold-out mean、両bootstrap下限が正であることを要求する。
grossが安定してもnet meanが非正、またはcanonicalとのpaired net差の両区間上限が負ならcost-limited。
それ以外の不確実なケースはnot successfully translatedとして、判定入力も保存する。
canonical置換はtradableかつpaired net/gross差の両bootstrap下限と両era平均が正の場合だけ認める。
この分類はportfolio結果を読む前に固定する。

## Consequences

- source model/config/dataの連鎖と固定mapをreview可能に保持し、暗黙のlatest選択を排除する。
- 既存評価器・RL wrapperの挙動を変えずに、独立のsealed研究成果物を生成できる。
- early terminationやsource不一致は明示例外となり、比較を部分的に続行しない。
- suggestive sourceは別途明示的な確認決定が必要なため、このcampaignでは拒否する。
- 不確実な経済効果を安易に成功とせず、cost-aware研究へ進む条件を限定できる。
