# 通年・固定費用stress基準線（Issue #29）

**実装と部分実行済み、実験は未完了。72/153 policy-foldがcomplete、81/153がblocked_input。**
9foldの期待barが不足し、17foldのera/全体成績、paired CI/LOO、最終的な基準線判定は未確定。
収益の良かったfoldだけを採用したものではなく、損益閲覧前の入力監査で実行範囲を固定した。

契約は[事前登録](protocols/issue29/registration.md)と[ADR-0038](../decisions/0038-restrict-cost-sensitivity-to-frozen-full-period-sources.md)。
機械可読の[全153件](results/issue29/report.json)、[状態CSV](results/issue29/status.csv)、[seal](results/issue29/manifest.json)を保存した。

## 実行と照合

実行trainer SHAは`fb7220a3fa8fefb89297cfc692c48ee3f8e89325`、env SHAは`6024b91c0f3592611849bc231922ab60e6090aed`。
実行期間は`2026-09-11T06:48:05.831526+00:00`〜`2026-09-11T06:48:34.665543+00:00`、CPU、Python 3.11.15。
依存version・lock・source bytesはmanifest/configに固定した。開始前に実装・登録をcommitした。
開始時git statusのuntrackedは生成し始めた結果directoryのみであり、実行中にコード変更はない。
F0/F1/F2は各pair spreadのみ2倍／overnight markupのみ2倍という事前の3条件に限定した。
学習・seed/checkpoint/alpha再選択、cache補修、metadataの補完は0。

236件のsource path/hash、51 PPO member、17 ridgeを入力監査で照合した。
実行した72口座はstep会計、各scenario内の3方策の全decision/target/pair順を照合済み。
保存後に150個のartifact hash、report集計、時刻/pair順、専用感度比較を再検証した。
実行済み口座にmargin callと実行例外は0。未完了81件は戦略損失ではなく入力不足。

## 欠落範囲と必要source

期待calendarはLondon全平日label、休日除外0、available_at=nullという研究仮定。
正式なprovider営業日や実行可能sessionを証明したcalendarではない。元pair別responseがなく、
欠けたbarが休場か配信欠損かは識別不能であるため、欠落に合わせたcalendar削除をしなかった。

| 未完了fold | 不足したLondon日付label（historyを含む） |
|---|---|
| 2009 | 2008-12-29, 2008-12-30, 2008-12-31, 2009-01-01, 2009-01-02, 2009-02-05 |
| 2011 | 2011-04-15 |
| 2012 | 2012-12-04 |
| 2013 | 2012-12-04, 2013-01-16, 2013-10-08 |
| 2014 | 2013-10-08 |
| 2017 | 2017-07-11, 2017-11-16 |
| 2018 | 2017-11-16 |
| 2019 | 2019-05-22 |
| 2025 | 2025-01-01, 2025-04-18, 2025-04-21, 2025-12-25 |

各不足foldの3方策×3scenarioすべてをblocked_inputとして残した。必要なのは、元Yahooのpair別responseと
当該sourceのcalendar根拠、必要なら欠落OHLC/CarryAnnualと変換lineageである。期待するcacheのpath/hashは
[input-audit.json](protocols/issue29/input-audit.json)に列挙した。現行cacheにないbarの期待hashを捏造していない。
追加sourceで補う場合は新calendar/data identityと事前登録を要し、このattemptを上書きしない。

## 完走したfoldのF0記述値

単位は%。各セルは年率net / 年率gross。first decisionからlast markまでの実効経過時間で年率化した。
独立口座の結果であり、連続運用CAGRではない。利用可能8foldの平均・勝率は採否に使わない。

| Fold | canonical | ridge | PPO ens3 |
|---|---:|---:|---:|
| 2010 | +2.900 / +6.594 | +1.474 / +9.115 | +13.843 / +16.481 |
| 2015 | +12.878 / +17.048 | +3.254 / +11.107 | -25.342 / -23.069 |
| 2016 | -0.362 / +3.049 | -21.253 / -15.390 | -0.100 / +1.652 |
| 2020 | -3.156 / +0.252 | -8.121 / -1.808 | -13.568 / -11.725 |
| 2021 | +9.288 / +13.405 | -3.812 / +3.130 | -8.567 / -6.531 |
| 2022 | +5.506 / +9.233 | -9.077 / -2.464 | +7.011 / +9.631 |
| 2023 | -0.168 / +3.377 | +6.991 / +14.560 | -3.451 / -1.849 |
| 2024 | +17.603 / +22.133 | -9.782 / -3.127 | +20.813 / +23.544 |

個別の弱いfoldとして、PPOの2015はnet −25.342%、MDD 36.955%、ridgeの2016はnet −21.253%、
MDD 23.769%、canonicalの2020はnet −3.156%、MDD 22.225%。これは完走済みfoldの記述であり、
未測定9foldを含むworst foldや全体のリスク判定ではない。各foldのSharpe、gross/net exposure、
price/carry/spread/commission/markup、target turnover、actual notionalはreport/statusに保存した。

## 費用感度

以下は同じsource・全時刻で独立実行したF1/F2のF0に対する年率net差（percentage points）。
倍率2は感度仮定で、実口座費用水準や上限の推定ではない。最良scenarioを本当の成績として採用しない。

| Fold | canonical F1 / F2 | ridge F1 / F2 | PPO F1 / F2 |
|---|---:|---:|---:|
| 2010 | -1.218 / -2.378 | -4.875 / -2.346 | -1.011 / -1.581 |
| 2015 | -1.448 / -2.609 | -5.029 / -2.388 | -0.691 / -1.530 |
| 2016 | -1.019 / -2.303 | -3.723 / -1.822 | -0.586 / -1.143 |
| 2020 | -1.078 / -2.240 | -3.875 / -2.124 | -0.634 / -1.180 |
| 2021 | -1.476 / -2.525 | -4.354 / -2.223 | -0.790 / -1.212 |
| 2022 | -1.188 / -2.439 | -4.161 / -2.103 | -0.899 / -1.673 |
| 2023 | -1.143 / -2.307 | -4.709 / -2.471 | -0.533 / -1.049 |
| 2024 | -1.686 / -2.716 | -4.212 / -2.086 | -0.926 / -1.759 |

72口座すべてで独立resetと再推論を行った。PPOのF1/F2対F0のaction変更は実測0件。
現行assetsはweight・保有期間・price PnL/equity_startであり、equity額そのものは入力されない。
費用による残高差を観測へ追加すると凍結modelの入力契約を変えるため、追加していない。
建玉の円額・actual notional・累積費用・最終equityは変わるので、旧netから増分費用を引く方式ではない。
年率gross−netは実口座のstep returnから計算し、累積費用/初期100万円とは別指標である。

## 年初を含めた違いと旧measurement

旧#16の全51行は[別表](results/issue29/legacy_reference.csv)として保存した。
新旧のnet/gross差は`descriptive_period_changes`の記述値のみで、paired検定・採否には使わない。
新しいF0口座の年初63stepのPnLと、新旧の期間全体の差を分けて保存した。
例としてcanonical 2020の年初net PnLは−108,325.94円、新旧年率net差は−15.151ポイント。
ridge 2015の年初net PnLは+105,027.75円、新旧年率net差は+11.042ポイント。
全体差には末尾target除外、年率化時間、口座経路も含まれるため、年初部分だけの因果効果とは呼ばない。

## 後続への引渡しと判断

3方策とも「測定/データ不足で未判断」。独立収益性の認定や旧#16分類変更はしない。
次期低容量予測／コスト対応campaignでは、まず欠落の由来とcalendar根拠を解決し、
#30で共通方向・相対配分・carry/costの帰属を確認する必要がある。追加学習や費用倍率探索は行っていない。

F0のstep traceは`results/issue29/fold-YYYY/F0-{canonical,ridge,ppo_ens3}.json.gz`の`trace`、
pair traceは同directoryの`F0-*-pairs.csv.gz`。8fold×3方策を確定した。
これを17foldの最終帰属分析と称してはいけない。未完了9foldは同じ一覧に明示してある。
17foldが揃った場合に用いる10,000 IID／3-fold moving-block・seed16・全LOO・最大fold寄与は
既存`paired_evidence`を再利用する実装とfixture検証を用意したが、この市場結果では出力をnullにしている。

## 検証の残件

回帰検証は353件通過。私が先に追加した新規テスト2件には、費用変更でassets/actionが必ず変わるという
誤前提と、凍結ridge配列を直接変更できるというfixtureの誤前提がある。
AGENTS.mdの実装中テスト変更禁止に従い、訂正許可の確認中は変更せず、2件を未解決として明示する。
本番env/modelの仕様を誤ったテストに合わせて変更していない。PRはこの残件が解決するまでdraftとする。
