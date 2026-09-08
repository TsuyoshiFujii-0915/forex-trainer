# 凍結rankingのspreadからnet損益までの分解

## 結論

Issue #19の診断を完了した。**同じ単位で比較すると、moving-block区間が初めて0をまたぐのは、
加重price log-returnから実現経路のprice-only simple-return会計へ変換する段階**だった。
carryを加えるとIID区間も0をまたぎ、transaction costはさらに平均を負へ下げる。
ただしこれは同じ17 fold上の診断であり、独立した新しいalphaの検証ではない。
#16の **not successfully translated to portfolio alpha** を維持する。
**stable positive grossは未解決**であり、cost-limited／tradableへ昇格させない。

[ADR-0030](../decisions/0030-decompose-frozen-ranking-on-realized-equity-path.md)で計算前に
主系列・単位・許容誤差・bootstrapを固定した。再学習、score反転、target/features変更、
rank/top-k/weightの再選択は行っていない。canonical reversalとdirect PPOを既存対照として保持した。

## 入力と再現

```bash
uv run forex-spread-decomposition \
  --campaign configs/research/issue19_spread_decomposition.json \
  --output-dir runs/issue19_spread_decomposition
```

出力先には新規directoryを指定する。
[明示manifest](../../configs/research/issue19_spread_decomposition.json)が選択する
#15/#16のprovenance SHA-256から、全sealed CSV/JSON/設定snapshotと各foldの
元config、market data、PPO ensemble/member model/config/metaのhashを照合する。
元modelの推論・学習は不要だが、hash照合のため元の明示pathにローカルdata/modelが必要となる。
元dataからevaluation dataset、price relatives、signed carryを再構築し、予測のtarget、
decision/target timestamp、pair順序と照合する。欠損をintersectionで除外しない。

[成果物](results/issue19/report.md)には以下を保存した。

- `fold_metrics.csv`: 全17 fold × 3 policyの元指標と年率log系列。
- `steps.csv`: 全policyのprice/carry/cost会計、log系列、会計残差。
- `report.json` / `report.md`: 各段階と隣接差のfold/era/両bootstrap/全leave-one-fold-out、最大絶対寄与と上位3 fold除外。
- `bootstrap_indices.npz`: 全系列・対応差で共有する10,000回のIID／3-fold circular moving-block index（seed 16）。
- `campaign_snapshot.json` / `provenance.json`: 入力hashの継承、元fold provenance、実装hash、Git/dependencyと全出力hash。

## 会計恒等式と近似の境界

pairの次期price log returnを `r_i`、price relativeを `R_i = exp(r_i)`、
実効weightを `w_i`、step開始／終了equityを `E / E'`、transaction costを `C` とする。
日数 `d` は隣接decisionの実経過日数、`a_i` はdecision時点のraw signed annual carry。

```text
S = mean(top-2 relative r) - mean(bottom-2 relative r)
L = sum_i w_i * r_i
P = sum_i w_i * (R_i - 1)
F = sum_i -w_i * R_i * a_i * d / 365
G = P + F = (E' + C) / E - 1
N = G - C / E = E' / E - 1
```

理想配分±0.8なら `L = 1.6 S`。weight合計が0なのでcross-sectional meanは相殺する。
実装は#16どおりfloat32配分を使うため、厳密には
`L = 2 × float32(0.8) × S`。`L − 1.6 S` の年率fold平均は約 **7.43e−10**であり、
経済効果ではない。丸めると0になるが、出力JSON/CSVには丸めず保存した。

`L` と `log1p(P)` は異なる。priceのlog→simple変換とportfolio合計のsimple→log変換を
含むため、差をそのまま純粋なvolatility dragやcarryと呼ばない。恒等的には
`log1p(P) − L = sum_i w_i (expm1(r_i) − r_i) + log1p(P) − P` である。
price-only系列 `log1p(P)` は実現経路の開始equityを分母にしたprice寄与の連鎖であり、
独立の無コスト運用口座と同一とは主張しない。

pair別price/carry/gross寄与、gross logの比例配賦、equity連続性、net logを全stepで照合した。
return単位の許容誤差は#16と同じ **rtol=1e−10、atol=1e−12**。
9 pairの倍精度和、equity比やlog変換の丸めを吸収する一方、float32 weightを理想値へ
置換して誤差を隠さない。累積値はstep数×atol、年率化は経過年数によって誤差境界を調整する。
全3 policyで観測した `N − (P + F − C/E)` の最大絶対残差は **3.46e−16**だった。

## 共通単位での結果

主系列は各foldでstep値を合計し、実経過秒／(365.25×86400)で割ったannual log return。
表はそれを100倍した **年率log-returnの百分率**であり、annual simple returnではない。
foldは等重み。eraも該当foldの算術平均。日次・pair行を独立標本にしていない。

| 段階 | 17-fold平均 | 2009–2018 | 2019–2025 | IID 95% | moving-block 95% | LOO平均 min / max |
|---|---:|---:|---:|---|---|---|
| 1.6 × tail spread | +4.9851 | +5.3315 | +4.4903 | [+0.6088, +9.5842] | [+0.3603, +9.8289] | +3.5910 / +6.0004 |
| 実効weight × price log | +4.9851 | +5.3315 | +4.4903 | [+0.6088, +9.5842] | [+0.3603, +9.8289] | +3.5910 / +6.0004 |
| price-only `log1p(P)` | +4.3119 | +4.5628 | +3.9534 | [+0.0104, +8.8951] | [−0.2314, +9.0473] | +2.9093 / +5.3132 |
| carry込みgross `log1p(G)` | +4.0468 | +4.2941 | +3.6935 | [−0.2562, +8.5977] | [−0.4605, +8.7242] | +2.6551 / +5.0472 |
| net `log1p(N)` | −3.1049 | −2.9832 | −3.2786 | [−7.3680, +1.3924] | [−7.5148, +1.4868] | −4.4950 / −2.1039 |

同じ再標本化indexによる**隣接段階の対応差**も計算した。単位は年率log-returnの百分率の差。

| 差（後段−前段） | 平均 | IID 95% | moving-block 95% |
|---|---:|---|---|
| price-only − 加重price log | −0.6732 | [−0.8202, −0.5362] | [−0.8394, −0.5206] |
| gross − price-only | −0.2651 | [−0.3943, −0.1344] | [−0.3926, −0.1496] |
| net − gross | −7.1516 | [−7.2706, −7.0324] | [−7.3315, −6.9821] |

最初の会計変換による減少は両区間が負で、price-onlyのmoving-block下限が0を下回る。
carryも両区間が負の追加寄与を持ち、grossはIIDでも正を確立できない。
costによる低下はそれらより大きいが、grossの不確実性が先に残っているため
「costだけが障害」とは識別できない。
net平均は負だがnetの両区間は0をまたぐため、netの期待値が統計的に負と確立したとも主張しない。

## 元定義の再現と外れ年依存

#15のtail spreadはfold内のdecision平均を、さらにfold等重みで平均した **0.00011991 log/decision**。
元のseed 15によるIID区間 `[0.00001444, 0.00023274]`、moving-block区間
`[0.00000556, 0.00023729]` も再現した。今回の共通seed 16の区間は別途並記している。
元tailのCI下限をannual returnの下限と直接比較しない。

#16のgross/netは各foldで `expm1(annual log return)` を取ってから平均する。
主診断のannual log平均にexpm1を掛ける方法とは、非線形変換の順序が異なる。
また、`1.6 × mean_tail_spread` を全fold共通の日数で年率化する方法とも異なり、
本診断は各foldの実際のdecision数と経過年数を使う。fold重みを変える集計探索はしていない。

| policy | 元annualized gross | 元annualized net | initial-equity基準cost ratio |
|---|---:|---:|---:|
| supervised fixed map | +4.5864% | −2.6380% | 5.3440% |
| canonical reversal | +8.8088% | +4.9831% | 2.7582% |
| direct PPO ens3 | +13.0648% | +10.3098% | 1.9138% |

元grossのmoving-block区間は **[−0.1618%, +9.6192%]**、netは
**[−6.9769%, +1.9940%]**で、元seed 16の区間も照合済み。
annualized gross−netは **7.2245 percentage points**だが、
`sum(C)/initial_equity` のfold平均 **5.3440%**とは分母・年率化・非線形変換が異なる。

全段階で最大絶対寄与foldは2025年。加重price log、price-only、grossの
絶対寄与比率はそれぞれ約19.75%、19.96%、19.87%。すべてのLOO平均は正であり、
単一年だけで平均が正になっているわけではない。
各系列の有利な上位3 foldを除いた平均もそれぞれ+1.9400、+1.2926、+1.0777
（年率log-return百分率）だが、これは事後の外れ年依存診断であり新しい採否条件ではない。
netの全LOO平均は負。
全fold、era、両bootstrap、LOO範囲、最大絶対寄与比率、上位3 fold除外を
[統合表](results/issue19/report.md)にまとめ、全LOO値はJSONに保存した。

## 次段階の根拠と不足証拠

**grossの不確実性が残る。** 会計差は照合でき、positive tailからnetへの低下を
説明できたが、price-only／grossの時間依存を考慮した正方向の統計的支持は未確立。
会計診断の完了とtradabilityの確立を区別する。既存分類とcanonical benchmarkを維持する。

**新しい独立検証を設計する根拠はある。** 予測spread自体の支持は維持され、
price simple会計への変換とcarryがどれだけ差を作るか定量化できた。
次の検証では、予測targetと実効weight下のprice/carry会計の対応を事前登録し、
未使用の期間または事前に固定した独立評価設計でgrossの再現性を確認する根拠になる。
ただし本診断はその設計・実行を行っておらず、独立なpositive gross、cost控除後のnet優位、
canonical置換の証拠は不足している。同じ17 foldでCIが正になる集計や配分を選び直すこと、
gate・bandit実装、generic PPO探索を開始する根拠にはしない。
