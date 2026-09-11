# 観測historyと損益期間を分離する評価器（Issue #28）

採用仕様は[ADR-0037](../decisions/0037-adopt-full-period-historical-evaluation.md)。
`full-period-history-v1` / `development_historical`として実装した。
[ADR-0035案](../decisions/0035-separate-observation-history-from-measurement-range.md)と
[旧full-period仕様案](protocols/issue22/full-period-spec.md)は履歴として保持する。
#27の開発契約に従い、新しい学習・HPO・独立収益性認定は行わない。

## 実行

リポジトリrootから実行する。config内の相対pathはroot基準であり、configファイルの親基準ではない。
出力先は存在していてはいけない。入力不足・hash不一致で再取得・再学習・latest代用はしない。

```bash
# Run a short real-artifact smoke, not the 17-fold research experiment.
uv run forex-period-eval \
  --config configs/research/issue28_smoke.json \
  --output runs/full-period-history-v1/issue28-smoke

# Reproduce the existing single-model evaluator in a new directory.
uv run forex-period-eval \
  --config configs/research/issue28_legacy.json \
  --output runs/legacy/issue28-reproduction

# Run deterministic fixtures and the existing regression suite.
uv run pytest -q
```

legacyは既存`_run_evaluation`を新規出力先へ呼び出し、元runを変更しない。
`forex-eval`、`forex-ensemble-eval`、学習configの既存schema・既存成果物は変更していない。
legacyのmetrics/equity/evaluation bytes一致はserialized PPO fixtureで検証する。
legacy単体モデルとfull-periodのens3は同一方策の比較にはならない。

## full_period config

[smoke config](../../configs/research/issue28_smoke.json)が読み込み可能な実例である。
`mode`・`measurement_id`・`evidence_class`・`source`・`lineage`・`runtime`・`folds`を全て明示する。

| フィールド | 契約 |
|---|---|
| mode | `full_period`。legacyは別schemaの`mode`と`run_dir`だけ |
| measurement_id | `full-period-history-v1` |
| evidence_class | `development_historical`のみ |
| source | #15 `provenance.json`のpath/SHA-256。models.jsonとfold_sourcesを親sealから解決 |
| lineage | raw/clean/carryの各path/hash一覧、変換説明、rate vintage、availability証拠のJSONをpath/hashで固定 |
| runtime | `device: cpu`、採用env SHA、依存version。既存provenanceの5依存schemaと、numpy/pandas/pyarrow/PyYAMLを含む9依存schemaを明示的に許容。出力は常に9依存と両repo全Python source hashを記録 |
| folds | fold ID、timezone付きmeasurement_start/end、calendarのpath/hash。指定した全foldをpreflightし、不足foldを除かない |

各foldを新しいenvで評価し、ridge・canonical・PPO ens3ごとに口座をresetする。
PPOの3memberは同じensemble口座のassetsを入力とし、各memberの独立口座を平均する方式ではない。
ridgeは凍結standardizer/selected alpha/coefficients/interceptをそのまま使用し、float64の32×8 windowで推論する。
canonicalは既存float32 mom24 ascending sort、ridge固定mapはscore ascending sortという同点処理を保持する。

## calendarとhistory

[smoke calendar](protocols/issue28/smoke-calendar.json)は短い動作確認用の事前指定集合である。
全pair共通の`symbols`順、timezone、version、holidays、label_rule、sessionsを必須にする。
各sessionは`bar_label`、`session_open`、`session_close`、`available_at`の4値を持つ。
naiveなbar labelは保存labelとして扱えるが、measurement境界とsession時刻はtimezone必須。
時刻はUTC instantで比較し、名前付きtimezoneとISOのDST前後のoffsetを照合する。

historical decision/markは**仮定session_close**である。実際の配信時刻が不明ならavailable_atはnullとする。
既知available_atがcloseより後なら同close推論不能として拒否する。
過去の公表時刻・rate vintageが不明であることを、現在の取得時刻で埋めない。
新しいcalendarはデータから欠落を除いて作らず、期待するsession集合を結果閲覧前に登録する。
calendar自体の完全性は登録者のデータ契約であり、評価器がprovider営業日を推測することはない。

測定は`[S,E)`内の連続bar transitionで、`S <= decision < target < E`。
予定first decision直前の有効raw63本だけをhistoryとして切り出す。decision自身を含む入力は64本。
期待barが欠落すればpair・時刻・source pathを伴って停止し、first decisionを後ろへずらさない。
最終barには推論・rebalance・仮想清算を行わず、mark後に終了する。

既存envのwarmup32とwindow32のresetでraw index63がfirst decisionになる。
履歴ではstepを呼ばず、初期equity=100万円・exposure/assets=0、初回entry費用を一度だけ課す。
同close約定・reward・spread/commission/overnight/signed carryの会計は既存envのまま。
env変更と関連PRは不要。採用SHAは`6024b91c0f3592611849bc231922ab60e6090aed`。

## 出力と比較

- `manifest.json`: config/hash、command、calendar全session、63本の元label/UTC時刻、予定範囲、
  実効時刻集合/hash、親model/config/data/lineage、ridge parameter hash、両repo SHA、
  依存・CPU・実際のPython source hash、生成物hash。
- `report.json`: 各fold・3方策のmetrics、初期assets、時刻/pair、coverage、全step trace。
- `steps.csv`: 新規推論したscore/action、PPO assets、実効target weight/exposure、equity、
  spread/commission/overnight、signed financing、turnover、margin-call/truncation。
- `report.md`: foldごとの記述的な一覧。独立foldの年率を連続運用CAGRとは呼ばない。

coverageはS/E、first decision、予定/実際のlast mark、初期flat待機秒、末尾gap秒、実効経過秒を分ける。
年率化とSharpeはfirst decision→last markのUTC経過秒/365.25日を既存`compute_metrics`へ渡す。
宣言した暦年長での割算に置換しない。

margin callは`incomplete_margin_call`としてterminal残高と全traceを保存し、CLIはexit 2を返す。
破産して終端equityが非正なら、log/年率/Sharpeはnullと理由を記録し、envのfloor付きreward合計は
別フィールドに保存する。負の残高をゼロへ修復した成功値は作らない。
不成功foldは残し、campaign全体の比較を停止する。

`require_comparable`は同一fold内のmeasurement ID・実行環境・時刻/pair・市場値hash・
リスク設定・cost・coverageの一致を要求する。旧measurement、部分時刻集合、別依存、別入力を拒否する。
この出力を旧`forex-report`用のmetrics.jsonへ改名して混ぜてはいけない。
#29のfold対応差・10,000 IID/3-fold moving-block・seed16・era/LOO集計は既存研究統計を利用する別実験である。

## 確認結果と限界

fixtureでは現行と同じ9pair×32window×8特徴量、凍結ridgeと初期化済みPPO seed42/43/44の
実serialized modelを使用する。fixtureモデルの学習は行わず、市場成績の証拠にはしない。
63/62本、休日、DST、閏年、欠損、target=E/>E、最終mark、reset、初回費用、会計恒等式、
未来値非干渉、ridge parameter/float64精度、3方策の通年時刻整合、PPO assets、
source/hash改変、比較契約、margin call/破産、legacy/source bytes保持を検証する。

実artifactのsmokeは2025-01-01以上2025-01-11未満で実施し、3方策とも6 transitionを完了した。
first decisionは2025-01-02 00:00 UTC、last markは2025-01-10 00:00 UTCで完全一致し、
初期待機と末尾gapは各86,400秒、実効経過は691,200秒だった。市場成績の採否は行っていない。
calendarは2024-10-01〜2025-01-13の平日集合から2025-01-01だけを除く。
この歴史cacheには2024-12-25のbarが存在するため、smokeもそれを含む仮定を明記している。
bar labelを同close時刻とみなして旧会計の時刻差を保持する**動作確認限定**の写像であり、
取引可能sessionや当時の配信時刻を証明するcalendarではない。
smoke出力は`runs/`または`tmp/`に置き、実装PRに研究結果を混ぜない。

[既存lineage監査](results/issue22/data_lineage.json)のraw/clean/carry 3ファイルと凍結2025モデルは利用可能。
一方、元Yahoo pair別response/取得時刻、元FRED response/release/vintage、実行可能bid/askは未保存。
配信欠損と休日の区別をcache全体から証明することはできない。
[今回のlineage入力](protocols/issue28/source-lineage.json)に、この不足とhindsight修復を記録する。

## #29への引渡し

1. `forex-period-eval`、`full-period-history-v1`、上記env SHAを固定する。
2. `docs/research/results/issue15/provenance.json`と`models.json`を親とし、
   `fold_sources`に列挙された17foldのconfig、ensemble.json、env_eval.yaml、metrics.json、
   memberごとのmodel_final.zip/config_snapshot.yaml/meta.jsonをhash一致で用意する。
3. raw/clean/carryの3cacheとlineageを固定する。新historyやmodelが不足すれば、例外に出るpath/hashを解決する。
4. 17fold分の**研究用calendarと仮定**を事前登録し、foldsに各年のUTC `[01-01, 翌01-01)`を指定する。
   このsmoke calendarを通年calendarとして拡張せず流用することはできない。
5. #27で登録したF0/F1/F2を、同一条件内の3方策比較として別campaignで実行する。
   この評価器は親configの費用を保持するF0用であり、費用stressの明示設定・別identityと実験集計は#29の担当。

本Issueのfixture/smoke通過は17fold市場実験の完了や収益性の認定ではない。
