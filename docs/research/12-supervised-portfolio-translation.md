# 凍結した教師ありrankingの固定portfolioへの変換

## 結論

Issue #16の判定は **not successfully translated to portfolio alpha** とする。
教師あり固定配分の17-fold平均はannualized gross **+4.59%**、net **−2.64%**だった。
grossの両era平均と全leave-one-fold-out平均は正だが、3-fold moving-block 95%区間は
**[−0.16%, +9.62%]**で0をまたぐ。事前固定したstable positive grossの条件を満たさず、
**learned but cost-limitedとはまだ分類しない**。canonical reversalをbenchmarkとして維持し、
contextual bandit実装へ進む前にpredictive objectiveとportfolio economicsの差を整理する。

## 固定した実験契約

Issue #15の`established learnable`を受け、[ADR-0029](../decisions/0029-translate-sealed-scores-with-a-fixed-portfolio-map.md)
にportfolio結果を見る前の配分・集計・分類条件を記録した。

- Issue #15のsealed evaluation CSVをround-trip精度で直接消費する。モデル再学習・alpha再選択はしない。
- 9 pairの予測scoreの上位2をlong、下位2をshort、各weight magnitude 0.8、残り0、gross target 3.2。
- high scoreがlong。stable ascending sortとcanonical pair順で同点を解消する。
- decision interval 1。score threshold、gross sizing、gate、hysteresis、HPOを追加しない。
- canonical controlは既存`mom24` reversal実装を使用し、同点時も既存のmomentum ascending sortを維持する。
- PPOはIssue #15と同じcurrent-provenance `longf ens3`（seed 42/43/44）を再生する。
  source scoreと旧sealed metricsへの一致を確認した後、同じactionを今回の共通評価器で再評価する。
  PPOの既存gross cap挙動も維持し、教師ありとcanonicalの固定weightは縮小されていないことを照合する。
- canonical daily data、2009--2025の17 expanding folds、era 2009--2018と2019--2025。
  32-row feature warmupと32-row observation windowを元の契約どおり適用するため、2009の最初のdecisionは4月3日。
- source/config/data/model hash、全decision/target timestampとpair順序を照合し、intersectionで欠損を除かない。
- 同じtransaction costとsigned carry環境を使用。grossは既存`compute_metrics`どおりtransaction costを
  加算復元したreturnで、signed carryを含む。独立のcost-free accountを運用したcounterfactualではない。
- foldをmarket sampling unitとし、annualized returnの**fold算術平均**をprimary aggregateとする。
  10,000 IID fold bootstrapと3-fold circular moving-block bootstrap、seed 16。日次rowは独立sampleにしない。

stable positive grossにはmean、両era、全leave-one-fold-out meanと両bootstrap下限が正であること、
全foldのmean realized gross leverageが1以上であることを要求した。tradableにはさらにnet mean、
全leave-one-fold-out meanと両bootstrap下限が正であることを要求した。

## 結果

すべて同一decision集合・同一評価器での結果。return、Sharpe、drawdown、costはfold平均であり、
連続運用した17年CAGRではない。total cost ratioの分母は各foldのinitial equity。

| policy | annualized net | annualized gross | Sharpe | mean MDD | worst MDD | winning folds | mean gross leverage |
|---|---:|---:|---:|---:|---:|---:|---:|
| supervised fixed map | −2.64% | +4.59% | −0.339 | 10.57% | 18.21% | 7/17 | 3.200 |
| canonical reversal | +4.98% | +8.81% | 0.240 | 9.84% | 14.94% | 9/17 | 3.199 |
| direct PPO ens3 | +10.31% | +13.06% | 0.222 | 16.52% | 36.95% | 8/17 | 2.173 |

| policy | mean decision target turnover | mean fold total turnover | sum turnover over 17 folds | mean fold total cost ratio |
|---|---:|---:|---:|---:|
| supervised | 4.982 | 979.671 | 16654.400 | 5.34% |
| reversal | 1.243 | 244.424 | 4155.200 | 2.76% |
| PPO | 0.861 | 169.401 | 2879.814 | 1.91% |

| era | supervised net / gross | reversal net / gross | PPO net / gross |
|---|---:|---:|---:|
| 2009--2018 | −2.67% / +4.69% | +1.18% / +4.82% | +17.24% / +20.43% |
| 2019--2025 | −2.59% / +4.44% | +10.42% / +14.50% | +0.41% / +2.55% |

全foldとeraのSharpe、MDD、turnover、cost等は[fold_metrics.csv](results/issue16/fold_metrics.csv)と
[report.json](results/issue16/report.json)に記録した。

## 不確実性とpaired evidence

差はsupervised minus control。表の数値はpercentage points。

| control | metric | mean difference | IID fold 95% | moving-block 95% |
|---|---|---:|---:|---:|
| reversal | annualized net | −7.62 | [−13.57, −1.57] | [−12.28, −3.18] |
| reversal | annualized gross | −4.22 | [−10.51, +2.16] | [−9.19, +0.51] |
| PPO | annualized net | −12.95 | [−28.13, +0.83] | [−28.80, +0.05] |
| PPO | annualized gross | −8.48 | [−24.23, +5.69] | [−24.87, +4.98] |

supervised自体のgross IID区間は[+0.09%, +9.51%]、moving-block区間は[−0.16%, +9.62%]。
netのIID区間は[−6.83%, +1.90%]、moving-block区間は[−6.98%, +1.99%]だった。
canonicalとの差はnetで明確に負であり、置換を支持しない。PPOとの差は両指標とも不確実であり、
PPOの全期間平均が高いことを一貫した優越性とは解釈しない。

supervised grossは10/17 foldで正。最大の2025年は+30.10%で、絶対fold寄与の21.64%を占める。
2025を除くgross平均も+2.99%、正の寄与のうち上位3年の比率は54.80%、上位3年を除く平均は+1.32%。
単一年が正の平均を作っているわけではないが、時間依存を考慮した区間で正方向を確立できていない。
netは全leave-one-fold-out平均が負で、2025を除くと−4.12%になる。
各paired差にも最大絶対寄与fold、寄与率、全leave-one-fold-out平均、上位3 fold診断を保存した。

## rankingからportfolioへの診断

supervisedのmean rank churnは0.3744（pair rank変化をpair数で正規化した既存Issue #15指標）。
long/shortを区別した4 membership slotの前decisionからの退出率は77.99%で、初回entryはこの率から除く。
canonicalとlong/short membershipが異なるdecisionは99.22%。target-weight turnoverは初回entryを含み、
canonicalの約4.01倍だった。exit率とtarget-weight turnoverは異なる単位であり、混同しない。

grossとnetのannualized mean差は7.22 percentage points。foldの累積log returnを合算した場合、
cost dragはgross log returnの177.61%に相当する。これはcostが収益を超えたことを示すが、
stable positive grossの確認が欠けているため、cost-aware最適化へ進む十分条件にはしていない。
各foldの比率は`cost_drag_fraction_of_gross_log_return`に保存し、grossが負のfoldでは符号付き比率として読む。

以下はpredicted rank別のannualized gross **log-return寄与**のfold平均。rank 1が最低予測scoreで
short、rank 9が最高scoreでlong。price PnLとsigned carryをpairごとに計算し、毎stepでenvのgross
simple returnと一致を確認した後、`log1p(total)/total`で比例配賦した。合計はgross log returnに一致するが、
個別rankを独立運用したCAGRではない。

| predicted rank | 全期間 | 2009--2018 | 2019--2025 |
|---|---:|---:|---:|
| 1 (short) | +0.05262 | +0.03321 | +0.08035 |
| 2 (short) | −0.00113 | −0.02391 | +0.03142 |
| 3--7 (flat) | 0 | 0 | 0 |
| 8 (long) | −0.00643 | +0.02039 | −0.04475 |
| 9 (long) | −0.00459 | +0.01326 | −0.03009 |

後半eraではshort側が正の寄与を作る一方、long側は負になっている。Issue #15の相対log-return
spreadと、actual pair price return・signed carry・turnoverを含むportfolio economicsには差がある。
scoreを反転したり、良いrankだけを選び直したりせず、この差を次の問題整理の対象とする。

## 成果物・再現

```bash
uv run forex-supervised-portfolio \
  --campaign configs/research/issue16_supervised_portfolio.json \
  --output-dir runs/issue16_portfolio
```

出力先は新規directoryを指定する。sourceは[Issue #15成果物](results/issue15/provenance.json)と
campaignが明示するconfig/data/ensemble/modelに限定する。ローカルdataとmodelは元の明示pathに必要である。

[results/issue16](results/issue16/report.md)へ以下を保存した。

- `campaign_snapshot.json`、`provenance.json`: 固定source hash、今回のGit/dependency、fold別resolved env。
- `fold_metrics.csv`、`report.json`、`report.md`: fold/era/aggregate、paired/absolute uncertainty、分類。
- `steps.csv`: 全policyの同一decisionに対するequity、cost、carry、turnover、gross/net return。
- `pair_contributions.csv`、`rank_contributions.csv`: 実効weight、rank、price/carry/gross寄与とfold別rank集計。

今後はcanonical benchmarkを維持する。教師あり学習可能性そのものの結果を取り消すものではないが、
今回の固定mapではtradabilityは確立していない。新しいbandit issueやgeneric PPO探索は開始しない。
