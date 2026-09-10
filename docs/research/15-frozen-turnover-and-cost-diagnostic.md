# 凍結scoreの持続性・順位交代と売買costの発生原因

## 結論

Issue #21の診断を完了した。教師あり固定mapの高回転は**両era・全17 foldに共通**する。
隣接score/rankの持続性はcanonicalより低く、売買費用の約32%がlong/short反転、
約67%が通常のentry/exitに対応する。同一targetへの再調整は売買費用の約0.11%にすぎない。
ただし**total costの32.73%は売買量に比例しないovernight保有費用**だった。
全costを順位交代に帰属させたり、target turnoverを下げれば全costが消えると解釈しない。

判定は **「未解決」**。#19のgross不確実性は残り、#16の
**not successfully translated to portfolio alpha**を維持する。
状態別の次期収益は記述的関連であり、gate/bandit採用、低回転化による収益改善、
canonical置換を支持する因果的execution効果ではない。

## 入力・再現・事前定義

```bash
uv run forex-turnover-diagnostic \
  --campaign configs/research/issue21_turnover_diagnostic.json \
  --output-dir runs/issue21_turnover_diagnostic
```

出力先は新規directoryを指定する。
[明示manifest](../../configs/research/issue21_turnover_diagnostic.json)で#19のcampaignとprovenanceをpinする。
#19の`run_decomposition`を再利用し、元data/model/config・#15/#16 seal・fold・decision/target時刻・
pair順・会計を再検証したうえで、元#19のsteps/fold metricsと照合する。
本診断は#15のsealed scoreと#16の実効weight/pair/stepを消費する。再学習・policy推論は不要。
#19で保持するdirect PPOを含む元経済指標もJSONへ継承するが、主turnover診断は教師ありとcanonical。

[ADR-0033](../decisions/0033-diagnose-frozen-turnover-with-marked-exposures.md)で指標・状態・集約を先に固定した。
実入力照合で、signed carryでもovernight_rateが0ではないことを検出したため、最終集計前に
[ADR-0034](../decisions/0034-separate-overnight-fees-from-trading-costs.md)で訂正した。
旧accepted ADRの本文を書き換えず、supersededとした。評価環境や既存成果物は変更していない。

主診断は元の**next-decision horizonのみ**。隣接scoreのcross-sectional Pearson相関、
平均同順位rankのSpearman相関、rank churn、符号付き4 slotの退出率、entry/exit/反転、
上下境界gap、long/short所属期間を保存する。初回entryはtarget turnoverに含み、
相関・membership退出率の隣接比較からは除外する。定数入力の相関はundefinedとして件数を保存し、
0へ置換しない（実入力の隣接比較では両policyとも未定義0件）。

#16のrank churnは平均同順位rankの平均絶対変化/8であり、その定義を再現する。
ordinal版も別列に記録する。教師あり・canonicalのscoreの単位は異なるため、
境界gapの絶対値だけでpolicy間の「余裕」を比較しない。

## 実売買額と会計

前decisionの開始equityを`E`、実効weightを`w_i`、pair price simple寄与を`P_i`とすると、
次decisionの売買前保有額は`E × (w_i + P_i)`。
現decisionの開始equityを`E'`、targetを`w'_i`とすると、

```text
signed trade_i = E' w'_i − E (w_i + P_i)
actual notional_i = abs(signed trade_i)
signed target change_i = E' (w'_i − w_i)
signed drift_i = E' w_i − E (w_i + P_i)
signed trade_i = signed target change_i + signed drift_i
spread_i = actual notional_i × configured pair spread
commission_i = actual notional_i × commission rate
overnight_i = abs(E' w'_i) × overnight rate × elapsed days
total cost_i = spread_i + commission_i + overnight_i
```

初期保有額は0。signed target changeとdriftは恒等的に足せるが、絶対値を別々に取った量は
実notionalへの加算分解ではない。同じtargetを維持する場合のdriftには価格変動に加えて
equityの変化（price/carry/cost）が含まれるため、純粋な価格効果とも呼ばない。

envの`PortfolioAccount.rebalance`と`mark_to_market`の費用を照合し、さらに#16/#19 total costと照合した。
許容誤差は#19と同じreturn単位`rtol=1e-10, atol=1e-12`。最大cost残差/開始equityは
**1.63e−19**であり、この許容範囲を超える説明不能なcost残差はない。
commission_rateは0、overnight_rateは**0.00002/day**。overnightは価格適用前のtarget exposureに課され、
signed financingとは独立に存在する。実市場のslippage/market impactはこの環境に含まれず推定していない。

pair状態は`initial_entry / entry / exit / reversal / retained / inactive`の排他的6分類。
long/short反転は1回のreversalであり、entry/exitに二重計上しない。cost・notionalは全状態で足し戻せる。
next-period pair netはgross simple寄与−当該pair cost/開始equityという実現経路の寄与であり、
独立口座の収益ではない。

## fold・era別の主診断

各policyは**3,343 decision × 9 pair**。統計単位は17 foldで、fold内平均をfold等重みで集約する。
不確実性は#19と同じ10,000 IID fold、3-fold circular moving-block、seed 16の共有indexを使う。
全指標の区間・全leave-one-fold-out・fold最小/最大は[JSON](results/issue21/report.json)に保存した。

| 指標 | 教師あり全体 | 2009–2018 | 2019–2025 | canonical全体 | 2009–2018 | 2019–2025 |
|---|---:|---:|---:|---:|---:|---:|
| target-weight turnover / decision | 4.98225 | 5.12535 | 4.77782 | 1.24326 | 1.20462 | 1.29847 |
| actual notional / 開始equity / decision | 4.98708 | 5.13022 | 4.78258 | 1.26314 | 1.22631 | 1.31575 |
| membership退出率 | 77.990% | 80.237% | 74.780% | 19.270% | 18.663% | 20.137% |
| score Pearson持続性 | −0.03657 | −0.10491 | +0.06108 | +0.93909 | +0.94255 | +0.93413 |
| rank Spearman持続性 | −0.03467 | −0.09470 | +0.05108 | +0.89890 | +0.90246 | +0.89381 |
| long/short反転pair数 / 隣接decision | 1.00047 | 1.09379 | 0.86716 | 0.00451 | 0.00359 | 0.00583 |
| 元total cost / 初期equity | 5.3440% | 5.4292% | 5.2225% | 2.7582% | 2.6915% | 2.8536% |

教師ありtarget turnoverのfold範囲は**4.42211–5.43030**、canonicalは**1.05859–1.52539**。
教師ありの全LOO平均も**4.95424–5.01726**で、単一foldを除いて高回転が解消する状況ではない。
教師ありturnoverのIID 95%は**[4.86232, 5.09524]**、moving-blockは**[4.80798, 5.14994]**。
教師あり退出率のmoving-blockは**[75.253%, 80.623%]**。
score相関のmoving-blockは**[−0.11519, +0.03761]**で、安定した負相関とは確立していない。
両policyとも全隣接decisionで9 pairのscoreが変わるが、canonicalのrank/所属はよく維持される。
scoreが変わったという事実だけで売買の必要量を説明できるわけではない。

#16の教師ありturnover 4.982、退出率77.99%、cost 5.3440%を再現した。
canonicalのturnover/costも再現したが、#16はcanonicalのmembership/rank診断列を保存していない。
canonical退出率は同一定義による今回の追加測定であり、欠損値を元観測と称して補完していない。
両policyのrank churnは#15のsealed指標とも照合した。元gross/net（教師あり+4.5864%/−2.6380%）は
#19経由で保持し、annualized gross−netと初期equity基準costを混同しない。

## 売買費用の原因とovernightの分離

以下は**fold内の費用比率をfold等重みで平均**した値。全期間のJPY合計比率ではない。

| pair状態 | 教師あり：売買費用内の割合 | canonical：売買費用内の割合 |
|---|---:|---:|
| 初回entry | 0.334% | 1.239% |
| entry（flat→long/short） | 33.688% | 48.361% |
| exit（long/short→flat） | 33.692% | 48.235% |
| long/short反転 | 32.174% | 0.517% |
| 同一target維持後の再調整 | 0.111% | 1.647% |
| inactive | 0% | 0% |

教師あり反転の売買費用割合はfold別**25.50–43.37%**。反転だけの特殊な1年ではなく全foldに存在し、
通常のentry/exitが残るので、反転だけを抑えて高回転全体が解決するとは言えない。
canonicalの反転は7 foldだけで、その他10 foldは件数0をそのまま保存した。

| total costの内訳 | 教師あり全体 | 2009–2018 | 2019–2025 | canonical全体 | 2009–2018 | 2019–2025 |
|---|---:|---:|---:|---:|---:|---:|
| 売買spread + commission | 67.273% | 67.842% | 66.461% | 34.390% | 33.710% | 35.360% |
| overnight保有費用 | 32.727% | 32.158% | 33.539% | 65.610% | 66.290% | 64.640% |

canonicalはretained状態にtotal costの53.00%が対応するが、売買費用内のretained割合は1.647%だけ。
その大半がovernightであり、「同じtargetへrebalanceすることがcostの主因」とは解釈しない。

## 所属期間と探索的関連

long/short所属期間はfoldごとに初期flatから数える。終了decisionまで所属する各fold4 slot、
各policy合計**68期間を右打切り**とした。次foldへ連結せず、打切りを完了exitに変換しない。
教師ありは完了10,375期間、canonicalは2,563期間。
完了期間の中央値は教師あり1 decision、canonical2 decision、最大はそれぞれ10、59。
これは**完了期間だけの記述**であり、打切りを含む無条件の平均保有期間の推定ではない。
long/short別・各fold・打切り有無別の全期間と度数分布をCSVに保存した。

membership全体を維持したdecisionは教師あり**11件**（2020:6、2021:1、2022:2、2025:2）で、
2009–2018には**0件**。canonicalは1,336件だった。
教師ありの維持状態について、early eraの平均収益はnull、observed folds=0と明示した。
late eraの条件付きnext net simple-returnの観測fold等重み平均は約+0.000873だが、
わずか4 fold・11 decisionの事後sliceであり、変更状態との差をexecutionの因果効果としない。
維持状態にも再調整・overnightがあり、「維持した日＝費用も売買もゼロ」ではない。
全状態別の次期price/carry/gross/netと件数は`exploratory_states.csv`およびJSONに保存した。

上下境界gap対membership退出率の同時点相関のfold平均は教師あり**−0.0284 / −0.0579**、
canonical **−0.2558 / −0.2570**。gap対cost ratioは教師あり**−0.0139 / −0.0267**。
これも探索的な記述で、境界近くの交代だけに高回転が集中すると判断するには不足している。
gap閾値・hysteresis・horizonの探索をせず、次の取引を抑制した反実仮想収益も推定しない。

## 成果物・残る不足証拠・次段階

[成果物](results/issue21/report.md)には、原因別CSV、pair/decision trace、holding spells、
fold/era統計、共有bootstrap index、入力・出力・実装・env会計hashと元provenanceを保存した。
JSONのsource chainから#19の再検証実行と#15/#16元artifactを追跡できる。

会計上の対応は解決できたが、**次の一実験を採用する根拠は未解決**。
高回転が両eraに存在し、反転と通常の所属交代に費用が対応する事実は定量化できた。
一方、低回転で同じgrossを維持できる証拠、独立なstable positive gross、cost後net優位は不足する。
特に維持状態の教師あり観測が極端に少なく、低回転化の利益を既存sliceから主張できない。

#19のgross不確実性を残したまま、gate/bandit、score smoothing、配分・top-k・interval変更へ
自動進行しない。次に独立gross再現性や特定のexecution変更を検証する場合は、
[#20のprotocol](14-research-and-forward-protocol.md)に従い、変更する一変数・探索予算・
停止条件・評価期間を別途事前登録する。本診断はその新実験を実行しない。
