# 研究ログ: 現実的な取引戦略で有意なプラスを出すエージェントの構築

目標: 「取引しないことで損失を回避する」のではなく、実際にポジションを取る現実的な戦略で、
取引コスト控除後に有意なプラスを出すエージェントモデルを構築する(2026-07-01 設定)。

## ドキュメント構成

| ファイル | 内容 | 対象ラウンド |
|---|---|---|
| [01-diagnosis-and-protocol.md](01-diagnosis-and-protocol.md) | 初期実験の敗因診断と実験プロトコルの確立 | 0〜2 |
| [02-long-data-and-walk-forward.md](02-long-data-and-walk-forward.md) | 長期データ導入とレジーム非定常性の発見、walk-forward への移行 | 3〜6 |
| [03-alpha-ceiling-and-reversal.md](03-alpha-ceiling-and-reversal.md) | ルールベース ceiling 診断と、クロスセクション・リバーサルの発見 | ceiling 診断 |
| [04-rl-vs-rule-benchmark.md](04-rl-vs-rule-benchmark.md) | RL 対 ルールベンチマーク、+7%主張の撤回まで | 7〜14 |
| [05-rl-architecture-campaign.md](05-rl-architecture-campaign.md) | 二時代プロトコル下の RL 本体改善キャンペーン | 15〜17 |
| [06-checkpoint-selection-study.md](06-checkpoint-selection-study.md) | validation-best / last / late-checkpoint選択方式の比較 | Issue #1 |
| [07-data-sufficiency-and-scaling.md](07-data-sufficiency-and-scaling.md) | `longf` の固有市場履歴量に対する汎化スケーリングと実効標本数 | Issue #7 |
| [08-sparse-rank-allocation.md](08-sparse-rank-allocation.md) | scoreから固定grossの疎なlong/short配分を作る構造化行動空間 | Issue #2、Round 18〜19 |
| [09-regime-tail-loss-diagnostic.md](09-regime-tail-loss-diagnostic.md) | current-provenance `longf ens3`のtail lossをfold単位でregime診断 | Issue #5 |
| [10-learned-apply-hold-gate.md](10-learned-apply-hold-gate.md) | direct proposalを適用するか既存allocationを保持するlearned execution gate | Issue #4、Round 20 |
| [11-supervised-ranking-learnability.md](11-supervised-ranking-learnability.md) | `longf` feature windowから次期pair orderingを教師ありridgeで学習できるかの診断 | Issue #15 |
| [12-supervised-portfolio-translation.md](12-supervised-portfolio-translation.md) | 凍結した教師ありscoreの固定配分への変換と、grossの統計的支持・cost dragの検証 | Issue #16 |
| [13-frozen-spread-to-net-decomposition.md](13-frozen-spread-to-net-decomposition.md) | 凍結spreadからprice/carry/costまでの会計とfold統計の分解 | Issue #19 |
| [14-research-and-forward-protocol.md](14-research-and-forward-protocol.md) | 出典付き期間・試行台帳、採否・停止・分岐、前向き記録schemaと開始・終了・開封契約 | Issue #20 |
| [15-frozen-turnover-and-cost-diagnostic.md](15-frozen-turnover-and-cost-diagnostic.md) | 凍結scoreの持続性、所属交代、実notionalとspread/overnight費用の原因別診断 | Issue #21 |
| [16-execution-and-measurement-audit.md](16-execution-and-measurement-audit.md) | 実効期間台帳、約定・費用・データの現実性監査、通年評価とpaper前の受入仕様 | Issue #22 |
| [17-bounded-development-protocol.md](17-bounded-development-protocol.md) | 開発と独立確認の適用範囲、短期予算・採否/停止/分岐、#28〜#32実行manifest | Issue #27 |
| [19-full-period-cost-baselines.md](19-full-period-cost-baselines.md) | 凍結3方策×固定費用stress、72/153完走・81入力不足、F0 step/pair traceと不足証拠 | Issue #29 |
| [18-full-period-evaluator.md](18-full-period-evaluator.md) | 63本の観測履歴を損益期間から分離する評価器、legacy再現、凍結3方策と#29引渡し | Issue #28 |
| [20-ppo-profit-attribution.md](20-ppo-profit-attribution.md) | PPOのcommon/relative/carry/cost帰属と2対照、40/85完走・45入力不足 | Issue #30 |
| [21-causal-quote-replay.md](21-causal-quote-replay.md) | 公開・保存後quoteの数量口座replay、fixture検証、実データcoverage不足の引渡し | Issue #31 |

現在の短期開発は[issue27-development-v1](17-bounded-development-protocol.md)と
[実行manifest](protocols/issue27/execution-manifest.md)を参照する。
#28 → #29 → #30最終解析を主経路とし、#30先行診断・#31約定replay・#32開発shadowを並行する。
今回の新規学習・HPO予算は0。開発shadowは成績閲覧可だが独立確認へ事後昇格しない。
旧[issue20-forward-v1](14-research-and-forward-protocol.md)は凍結確認計画として保持し、未開始のまま。
新契約は開発着手と認定判断を分けるもので、旧方策の収益性・canonical置換を新たに認定していない。
2009〜2025はdevelopment folds、2007〜2008と2026H1は過去確認使用済みであり再封印しない。
年初warmup除外部分や2026H2も、暦上新しいことだけで未使用へ再分類しない。
以下と過去ノートの成績は各実験の証拠範囲で読む。独立確認による統計的収益性の確立を意味しない。

## 確立した方法論(今後の全実験に適用)

1. **walk-forward 集計でのみ構成を比較する** — 単一の train/val/eval 分割の結果はレジームの
   引きに支配される(検証成績と評価成績の相関が **−0.37** だった実測がある)。標準は評価年
   2019〜2025 の7フォールド × 3シード以上とし、採否の標準証拠には`forex-report`の
   fold/seed/era集計、対応差、fold bootstrap、moving-block bootstrapを使う。個別run順位や
   手作業の表計算だけでは変更を採用しない。
2. **新しい特徴量空間・ペア集合では、RL の前にルールベースの ceiling 診断を行う** —
   スクリプト化したポリシーを env の実コストモデルに通し、グロスが正であることを確認して
   から学習を投入する(グロス ≈ 0 の空間では何も学べない)。
3. **グロス(コスト控除前)とネットを常に分解する** — metrics.json の
   `gross_cumulative_log_return` を参照。敗因が「シグナル欠如」か「コスト」かを区別する。
4. **学習バッチ実行中にソースコードを編集しない** — editable インストールのため、実行中の
   バッチが編集途中の不整合なコードを import して落ちる(round-7b で実証済み)。

## 主要な結論(2026-09-08 時点)

- 2ペア(JPY/USD, JPY/EUR)× 価格由来テクニカル特徴量の空間には**取れるエッジが存在しない**
  (ルールでも RL でもグロス ≈ 0)。
- 7 JPYペアの日足クロスセクションには**リバーサル(平均回帰)構造が実在**する:
  ルールベース上限はネット **+1.3%/年**(グロス +4.2%/年、Sharpe 0.29、7年中5年プラス)。
- **資金調達会計の修正(env ADR-0009)が絶対水準を変えた**: 旧・対称オーバーナイト課金は
  戦略を年約2.7%過小評価していた。signed 会計での確定値:
  - **リバーサル top2 ルール: ネット +4.5%/年、Sharpe 0.95、5/7年**(7年で t値 ≈ 2.5 —
    ルールとしては「有意なプラス」に最も近い)
  - **RL 最良(ens3、キャリー+xs特徴量、9ペア日足): ネット +3.1%/年、Sharpe 0.35、4/7年**
- RL はルール未達のまま: 2024年依存が強く(除くと ≈ −2.9%/年)、レジーム逆風年の
  ドローダウン制御ができていない。検証→評価相関は負のままでゲーティング不成立。
- direct `longf`のlearned apply/hold gateは17-foldで非採用（unproven）とした。turnoverとcostは下がったが、
  direct比net −7.39ポイント、same-model forced apply比−1.20ポイントでgross alpha損失が上回った。
- Issue #15では既存`longf`情報から教師ありridgeによる次期pair rankingの学習可能性を
  **established learnable**と判定した。
- Issue #16では予測tail spreadと固定配分grossの点推定は概ね整合した（gross +4.59%、net −2.64%）。
  ただしgrossのmoving-block区間が0をまたぎ、turnover/cost dragも大きいため、事前分類は
  **not successfully translated to portfolio alpha**。canonical reversalを維持する。
- Issue #19でfold単位の会計分解を完了した。同じ年率log単位では、moving-block区間が初めて
  0をまたぐのは加重price logからprice-only simple会計へ変換する段階。carryとcostも追加の
  低下を作る。grossの不確実性は残り、#16の分類を維持する。独立検証の設計根拠と不足証拠を記録した。

## インフラ資産

- 実験系: 決定間隔(ADR-0004)、検証区間モデル選択(ADR-0005)、シードアンサンブル評価
  (ADR-0007)、クロスセクショナル特徴量(env ADR-0008)
- データ系: Dukascopy 長期時間足キャッシュ(ADR-0006、`forex-fetch-dukascopy`)、
  yfinance 日足 7ペアキャッシュ
- CLI: `forex-train` / `forex-eval` / `forex-compare` / `forex-ensemble-eval` /
  `forex-fetch-dukascopy` / `forex-report`

`forex-report`へ登録する評価成果物はversion 2 provenanceを必須とする。既存の`longf ens3`
source runは現行のtraining provenanceを持たず、再評価だけでは移行できない。旧metaを現在値で
補完せず、現行契約で再学習してから`forex-eval`と`forex-ensemble-eval`を実行し、generic
campaignの基準方策として使用する。
