# 共通basketの1日・5日ridge予測器（Issue #41）

**本学習・市場予測生成・封印検証まで完了。**
2026-10-02、依存PR #56のマージを確認し、clean runtimeで通常102fitを実行した。
全34採用モデルと7,138行の市場予測を生成・検証済み。再fit=0、障害retry=0、口座評価=0。
両horizonとも全17区間平均MSEはゼロ予測を改善せず、登録済みの予測検討gateは未達。
成績を理由とする追加学習は行わず、両候補を#42/#43へ渡す。portfolio採否はここでは行わない。

- [設定](../../configs/research/issue41_common_basket.json)
- [入力検査結果](results/issue41/preflight/report.json)
- [最新の実行・予算状態](results/issue41/execution-status.json)
- [封印bundle](results/issue41/bundle/manifest.json)・[検証結果](results/issue41/verification.json)
- [fold/era診断](results/issue41/bundle/diagnostics.json)・[実行後の判断](results/issue41/postrun-assessment.json)
- [実装時点の未実行記録（履歴）](results/issue41/implementation-status.json)
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

全17区間で**3,569 decision/horizon、合計7,138予測行**を生成した。
1日labelは全3,569行、5日labelは3,501行で利用可能。5日末尾68行にも予測を残した。
2009のtrainはh1=268/h5=264、validationは
189/181で、#55の計数と一致する。全34ケースでtrain≥252/validation≥60を満たす。

`preflight/row-identities.json.gz`に各使用行・label終点・入力値hash、
`excluded-rows.json.gz`に行ごとの除外理由、`missing-calendar.json.gz`に欠損日を保存した。
元cache、各派生cacheとcalendarは既存source sealで照合し、外部取得は行っていない。

通常102 candidate fitを全て消費し、34モデルを採用した。再fit・共有retryの消費は0。
合成fixtureの累計6fitは実装時点のままで、本実行で追加していない。
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

#56のマージcommitは`1498af1b751ef0d7c4557246e21748f665d70cb2`。
trainer=`5f3ea68b8f88c3f97395d82035c4cd91ec527f20`、
forex-env=`6024b91c0f3592611849bc231922ab60e6090aed`のclean runtimeで以下を実行済み。
実際の起動は既存環境の`.venv/bin/python -m forex_trainer.common_basket_study`を用いた
（正確なコマンドはexecution-status.json）。通常枠は消費済みであり、同じrunの再fitは行わない。
sealの出力には入力検査、CLI/config、両repo SHA、lock、Python/依存version、CPUを記録する。
未マージ・dirty source・入力差異ではsealを作らない。

```bash
uv run forex-common-basket seal \
  --config configs/research/issue41_common_basket.json \
  --dependency-merge-commit 1498af1b751ef0d7c4557246e21748f665d70cb2 \
  --output runs/issue41-execution-seal
uv run forex-common-basket run \
  --config configs/research/issue41_common_basket.json \
  --activation runs/issue41-execution-seal/activation.json
uv run forex-common-basket verify --bundle runs/issue41-common-basket-v3
```

実装時点の`blocked_dependency_merge`記録は元bytesで保持し、今回の完了記録を追加した。
`bundle/`は実行ディレクトリを元bytesのまま保存したもの。学習せず次のコマンドで検証できる。

```bash
uv run forex-common-basket verify --bundle docs/research/results/issue41/bundle
```

bundle content SHA-256:
`3b92ed21ee6fc8a92108497f030db82e270b69c93a67820fe8f456e45f8d9509`。

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

`results/issue41/bundle/`の同じ封印済み市場bundleを#42のfixed/costと#43へ渡す。
本Issueでportfolio採否やhorizon採択は行わない。

## 市場予測の診断結果

等fold平均のMSE（basket price simple returnの二乗単位）。単純に全日をpoolしたMSEではない。

| horizon | 期間 | model MSE | zero MSE | model−zero |
|---|---|---:|---:|---:|
| 1日 | 全17区間 | 4.144824817e-5 | 4.041703563e-5 | +1.031212539e-6 |
| 1日 | 2009〜2018 | 5.064849384e-5 | 4.886470831e-5 | +1.783785527e-6 |
| 1日 | 2019〜2025 | 2.830504007e-5 | 2.834893180e-5 | −4.389173133e-8 |
| 5日 | 全17区間 | 2.000386166e-4 | 1.862717017e-4 | +1.376691485e-5 |
| 5日 | 2009〜2018 | 2.451124021e-4 | 2.212120699e-4 | +2.390033219e-5 |
| 5日 | 2019〜2025 | 1.356474943e-4 | 1.363568900e-4 | −7.093956383e-7 |

両hとも後期eraでは改善したが、全区間と前期eraでは改善しないため、v2第6節の
「予測の検討余地」は未達。追加alpha・特徴量・horizon・再学習で救済しない。
採用alphaは1日: 0.01が10fold、0.1が1fold、1.0が6fold。
5日: 0.01が9fold、1.0が8fold。いずれもvalidationで選んだ候補をそのまま採用した。
全foldの時間相関・方向一致率・予測分散とeraごとの等fold平均は上記診断成果物に保存した。

## 検証と限界

追加22テストのうち、合成fitを共有する3テストは6fitの1回の実行で検証した。
残る19テストと既存60テストも通過。テストを実装前に作成し、失敗後は実装を修正した。
実装検証時の修正は台帳・入力検査・封印経路であり、合成fitは追加していない。
今回の市場実行は封印済み実装による初回の通常102fitである。

検証対象はfuture perturbation、当日5列/9pair順、train-only標準化、mean/sum罰則換算、
非罰則切片、共分散、tie、1/5営業日、history/gap/purge、末尾予測保持、定数・非有限入力、
二重実行と予算超過、未マージgate、全34ケースの実値検査。
市場データによる102fit→全34モデル封印→7,138予測→bundle検証までend-to-endで完了。
保存コピーも全bytesがrunと一致。102予約/完了・採用候補の一致・共分散hash/正定値・as-of・
全decisionを検証し、予測ファイルを改変したコピーがhash不一致で拒否されることも確認した。
追加fitなしの検証結果を`verification.json`へ保存した。

旧#15との差は入力集約・target・選択方式を含み、純粋な容量削減効果とは呼ばない。
逆数JPY/COUNTER価格、14行修復、FRED vintage/公開時刻不明、60日lag、同close、
既知development期間・部分年という既存の限界を保持し、独立収益性を認定しない。

## PR #57レビュー後の検証強化（2026-10-04）

予測行の自己申告時刻だけでなく、参照モデルJSONのfold/horizon、provenance hash、
as-of、v3のcutoff・train/validation境界を照合する。各使用decision/labelは一意・昇順、
London平日・登録h営業日と一致し、label終点は当該rangeの終端より前でなければならない。
モデルparameter as-ofはvalidation labelの最大終点、共分散as-ofと行hashはtrain側に限定する。
bundle config・モデルのconfig/runtime/registration参照も検証済み契約と元の実行sealへ結び付ける。

通常fit開始は、ローカル`runs/`に加え、コミット済みexecution-status・bundle manifest・
予約台帳を検査する。消費済み／予約済みなら、新しいcloneで`runs/`がなくてもseal/runを拒否する。
preflight/verifyは消費済みでも利用できる。v3登録のconsumed=0は事前登録時点の記録として保持し、
現在の未使用枠とは扱わない。明示的に別登録するretryの枠へ自動で切り替えることもない。

[追加検証記録](results/issue41/review-verification.json)：内部hashを整合させた不一致13種、
保存済みbundle正常系、新規checkoutでの3種の消費記録×2入口の計20テストを追加した。
mockや市場fitなしで実際の成果物コピーとCLIを検証した。既存の関連19テストも通過し、
学習fixtureを共有する3テストは追加fitを避けるため今回の再実行対象から除いた。
本修正は検証・開始拒否だけであり、既存102fit・34モデル・7,138予測と元登録のbytesは変更しない。
