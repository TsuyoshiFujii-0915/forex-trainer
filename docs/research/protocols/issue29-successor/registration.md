# Issue #29 後継範囲の実行前登録

- trial_id: `issue29-contiguous-cost-v1`
- registered_date: 2026-09-22
- authorization: Issue #39の結果を踏まえてIssue #29の評価を進める利用者指示（2026-09-22）。
- evidence_class: development_historical。既知結果を踏まえた継続であり独立確認ではない。
- scope: #39 `issue39-contiguous-snapshot-v1` の17固定区間。8全期待bar範囲＋9部分年。
- snapshot_identity: `dab5a969ee98c7c2e2b2d01fc0038b91d8ebeef0f784a21a6d1cbab2220eb1ca`
- budget: 市場campaign 1回、3方策×3scenario×17区間＝153口座。学習・HPO・再取得・範囲変更0。
- frozen_policies: canonical mom24 reversal、元#15 ridgeとstandardizer/alpha、current PPO longf ens3 validation-best seed42/43/44。
- fixed_scenarios: F0従来費用、F1 spreadのみ2倍、F2 overnight markupのみ2倍。commission/signed carry不変。
- execution: CPU、元env/lock/依存を保持、各口座100万円・flat reset、k=1、最終bar markのみ。
- statistics: 17区間等重みの年率net/gross平均、元fold年のera 2009–2018 / 2019–2025。
  同scenario内方策対応差、同方策F1/F2−F0、絶対net/gross対zero。
  既存10,000 IIDと3-fold circular moving-block・seed16・95%区間、全LOO、最大fold寄与。
- interpretation: coverage条件付きの有限な開発パネル。不等期間の年率化であり、17通年や連続運用CAGRの推定ではない。
  区間実収益・実経過日数・volatility・MDD・費用内訳・turnover・actual notionalも併記する。
- incomplete_rule: 153セルのどれかがblocked/error/terminalなら全体/era/paired集計はnull。障害と損失を区別し、途中口座を除外しない。
- classification: 既存の開発用規則を継承。全scenario平均net>0なら現行仮定下の検討余地、F0>0かつstressで非正なら費用に脆弱、F0から非正なら正のnet基準線なし。
  不確実性・era/LOOと共に読む。独立収益性・canonical置換・#16分類変更を認定しない。
- preservation: 旧#29は72/153完了、81入力不足を保全。新結果を旧attemptへ追記・混合しない。
- handoff: 新sealと全51 F0 step/pair traceを#30へ渡す。本作業で2対照を実行しない。
- stop: 初回評価終了。結果を見て範囲・倍率・modelを変更しない。新予算は別登録。

登録と実装は市場推論前にcommitする。実行SHAと開始・終了時刻は新manifestに保存する。
再現検証は固定結果のhash・会計・統計の再照合を基本とし、追加の市場パネル評価はこの予算に含めない。
