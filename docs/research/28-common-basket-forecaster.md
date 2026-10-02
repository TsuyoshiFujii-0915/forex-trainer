# 共通basketの1日・5日ridge予測器（Issue #41）

**実装・入力検査完了、本学習は依存PR #56のレビュー・マージ待ち。**
2026-10-02時点で、v3の全34ケースは価格・特徴量・target・標準化の事前検査を通過した。
通常fit=0、採用モデル=0、市場推論=0、口座評価=0。Issue #41の市場成果物受入条件は未達であり、
実装PRだけでIssueを完了扱いにしない。中期RL・quote収集・旧通年復旧を開始条件には加えていない。

- [設定](../../configs/research/issue41_common_basket.json)
- [入力検査結果](results/issue41/preflight/report.json)
- [実装・予算・検証状態](results/issue41/implementation-status.json)
- [ADR-0047](../decisions/0047-seal-common-basket-forecasts-before-evaluation.md)

## 固定した実装

v3登録の採用splitを読む。2009は24か月、2018は12か月、残りは6か月で、
両horizonで同じtrain/validation境界を使う。v1/v2/v3の登録・旧#15・PPO成果物は変更していない。

元cacheをLondon期待平日ごとの連続ブロックに分け、63本の先行historyを要求する。
当日の9pair平均5列を固定順で作り、8列や32-lagへ戻さない。価格は正かつ有限、carry・
計算済み特徴量・targetは有限性を検査する。非有限入力は時刻・pair・欄を保存して停止する。
無効値の補間・0置換・列削除は行わない。calendar上の欠損、history不足、gap label、
境界purgeは別に保存する。全34ケースの件数を#55のsealed countsと照合する。

`mean_i(P_i(t+h)/P_i(t)-1)`をtargetに使う。train-only平均・母標準偏差、
mean-loss ridge、非罰則切片、alpha=0.01/0.1/1.0を実装した。validation MSEの
大域最小値から絶対差1e-12以内にある最大alphaを選び、再fitしない。
定数列は列名と件数を示して停止する。#42用共分散は同じtrain行集合のpair h日simple returnから
標本共分散を作り、登録済みの`0.9*S+0.1*diag(S)+1e-8*I`だけを適用する。

全34モデルをファイルへ保存し、`models-sealed.json`を書いた後にのみ評価予測へ進む。
モデルには全3候補、採用alpha、係数・切片・標準化・共分散、学習/選択範囲、使用行と
最大label終点、data/calendar/config/runtimeのhashを記録する。
評価末尾の診断label不足は`unavailable_gap_or_tail`で残し、口座decisionを削らない。
各foldでは最終markの直前までがdecisionとなり、最終markでは新規decisionを作らない。

## 入力検査結果と予算

全17区間で**3,569 decision/horizon、合計7,138予測行**を予定する。これは入力時刻から
数えた予定件数であり、予測生成済みではない。2009のtrainはh1=268/h5=264、validationは
189/181で、#55の計数と一致する。全34ケースでtrain≥252/validation≥60を満たす。

`preflight/row-identities.json.gz`に各使用行・label終点・入力値hash、
`excluded-rows.json.gz`に行ごとの除外理由、`missing-calendar.json.gz`に欠損日を保存した。
元cache、各派生cacheとcalendarは既存source sealで照合し、外部取得は行っていない。

合成fixtureは6 candidate fitを使用した。通常102fitと共有retry12fitの消費は0。
CLIの通常実行先はcampaign固定の`runs/issue41-common-basket-v3`で、新規作成のみ許可する。
各fitは計算前にappend-only台帳へ予約し、失敗も消費として残す。validation閲覧、alpha判断、
モデル封印、評価結果閲覧は別event。失敗後の再実行・自動retry・別出力先による予算再取得は拒否する。
障害時はincidentと全セルの状態を残す。共有retryを実行するには別の明示登録が必要で、
このCLIにはretry実行機能を追加していない。

## 再現と本実行の入口

入力検査はfit・市場推論なしで再現できる。出力先は未作成でなければならない。

```bash
uv run forex-common-basket preflight \
  --config configs/research/issue41_common_basket.json \
  --output runs/issue41-input-reproduction
```

#56のマージ後、最新`origin/main`を取得し、両repoの作業ツリーがcleanな状態で次を実行する。
`<MERGED_V3_COMMIT>`はv3登録を含む実際のマージ済み40桁SHAに置き換える。
sealの出力には入力検査、CLI/config、両repo SHA、lock、Python/依存version、CPUを記録する。
未マージ・dirty source・入力差異ではsealを作らない。

```bash
uv run forex-common-basket seal \
  --config configs/research/issue41_common_basket.json \
  --dependency-merge-commit <MERGED_V3_COMMIT> \
  --output runs/issue41-execution-seal
uv run forex-common-basket run \
  --config configs/research/issue41_common_basket.json \
  --activation runs/issue41-execution-seal/activation.json
uv run forex-common-basket verify --bundle runs/issue41-common-basket-v3
```

現時点では#56のHEADを指定しても`Dependency v3 merge is not verified on origin/main`で
停止することを確認した。これは入力不足とは別の`blocked_dependency_merge`状態である。
実行後は結果を研究成果物として別コミットに保存し、研究ノートとIssueの完了状態を更新する。

## #42/#43へのinterface

`verify_bundle(Path(...))`はmanifestと全artifact hash、102fit/34モデル/再fit0、
全予定decision、重複、finite予測、model hash、parameter as-ofを検証して予測行を返す。
検証に失敗したbundleやincidentがあるrunは使えない。返却行のキーは
`(fold, horizon_business_days, decision_at_utc)`、単位はh営業日のbasket price simple return。
同じ`bundle_content_sha256`をfixed/cost両方の結果へ記録する。

`model-<fold>-h<h>.json`の`selected.covariance`と`provenance`に共分散の推定式・
train行集合hash・最大label終点を保存する。`forecasts.json.gz`には両hの全予定decision、
診断labelと不足理由、model/provenance hashを保存する。`diagnostics.json`には全foldのMSE、
ゼロとの差、時間方向相関、方向一致率、予測分散、両era（2009〜2018/2019〜2025）の
等fold平均を記録する。定数系列の相関はnullと理由を残す。pairや重複5日labelを独立標本に加算しない。

まだ市場bundleは存在しない。#42はこのinterfaceに沿って合成fixtureを実装できるが、
#43への両候補の市場予測引渡しは本学習完了後となる。本Issueでportfolio採否やhorizon採択は行わない。

## 検証と限界

追加22テストのうち、合成fitを共有する3テストは6fitの1回の実行で検証した。
残る19テストと既存60テストも通過。テストを実装前に作成し、失敗後は実装を修正した。
以後の修正は台帳・入力検査・封印経路であり、数値fitの追加実行は行っていない。

検証対象はfuture perturbation、当日5列/9pair順、train-only標準化、mean/sum罰則換算、
非罰則切片、共分散、tie、1/5営業日、history/gap/purge、末尾予測保持、定数・非有限入力、
二重実行と予算超過、未マージgate、全34ケースの実値検査。
市場データによる102fitと全bundle生成のend-to-end実行は依存PR待ちで未検証。

旧#15との差は入力集約・target・選択方式を含み、純粋な容量削減効果とは呼ばない。
逆数JPY/COUNTER価格、14行修復、FRED vintage/公開時刻不明、60日lag、同close、
既知development期間・部分年という既存の限界を保持し、独立収益性を認定しない。
