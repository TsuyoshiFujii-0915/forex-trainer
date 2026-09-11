# Issue #29 実行前登録

- trial_id: `issue29-full-period-cost-v1`
- parent_trial_id: `issue27-development-v1`
- scope: `development_historical`、2009〜2025の17独立fold
- hypothesis: 凍結3方策を観測historyと損益範囲を分離して再評価し、固定費用stress下の基準線と不足証拠を記録する。
- changed_variable: F0は測定範囲のみ、F1はF0からspreadのみ2倍、F2はF0からovernight markupのみ2倍。
- frozen_values: canonical mom24 top/bottom各2・各±0.8・k=1・既存tie処理、#15のridge/standardizer/alpha、current longf PPO ens3。
- budget: campaign 1、153 policy-fold、新規学習/HPO/seed/checkpoint再選択0。
- folds: 2009〜2025、era 2009〜2018 / 2019〜2025。
- seeds: PPO 42/43/44、統計16。
- validation_rule: PPO validation-best、ridge既存selected alpha。再選択なし。
- registered_at: 2026-09-11T06:45:34.956606+00:00
- registration_commit: この登録を最初に含むGit commit。実行前にcommitし、結果manifestのruntime.gitでその実行SHAを記録する。
- opened_at: 未開封。実行後は別attempt台帳へ追記し、この登録を上書きしない。
- status: planned、input-only監査済み。
- attempts: 市場評価0。fixtureは市場評価attemptに数えない。

## 固定する入力と期待calendar

[実行config](../../../../configs/research/issue29_full_period_cost.json)が親契約・#15/#16 source seal・
raw/clean/carry lineage・lock・env SHA・全9依存version・CPU・17calendarをhash固定する。
[入力監査](input-audit.json)には236件のpath/hash照合、全51 PPO memberと17 ridgeの所在、
全欠落時刻を記録した。取得元はsealに記載された元のlocal runs/dataであり、再学習やmetadata補完はしない。
元pair別Yahooレスポンス、取得時刻、FRED vintage、実quote、休場根拠は未保存。

calendarは全平日のLondon日付labelを同close仮定へ写像し、UTCの`[01-01,翌01-01)`を測る。
価格cacheの欠落に合わせたholiday除外は行わない。全平日は研究用の保守的期待集合であり、
実際のFX営業日を認定したものではない。known holidayと思われる日も、当該sourceの根拠がないため
欠落と区別できないとして停止する。実配信時刻はnullのまま。末尾barではmarkだけを行う。

入力監査段階で予定153セルのうち、8foldの72セルが実行可能、9foldの81セルが入力不足。
不足foldは2009/2011/2012/2013/2014/2017/2018/2019/2025。
この事実は損益を見る前に確定した。利用可能な8foldだけの平均・era集計・paired検定・採否を行わない。
全153セルの状態と実行可能な個別foldの記述値を納品し、実験は未完了とする。

## 実行・会計・判断

各方策/scenarioは100万円・建玉0から開始し、PPOは当該口座のobservationで毎step再推論する。
現行envのassetsは前回weight・保有期間・価格PnL/equity_startであり、equity自体は入力にない。
価格PnL/equity_startは配分から決まり、費用変更で必ずactionが変わるとは限らない。
観測schemaを変更せず、実測されたaction差（0も含む）を保存する。

#19の照合許容値関数、#21と同じ時価建玉と次targetの差によるactual notional、
#16の`paired_evidence`（10,000 IID、3-fold circular moving-block、seed16、全LOO、最大fold寄与）を利用する。
全17fold完走時だけ同一scenario内の3方策間でnet/gross対応差を集計する。
F1/F2対F0は同一source policy、全時刻/pair/runtime/market/環境/coverageを要求する専用比較。
generic比較の費用/measurement検証は維持する。費用scenarioの合成・倍率探索はしない。

各step/pairにscore/action、実効weight、時価建玉、equity、price PnL、signed carry、spread、
commission、markup、actual notional、target turnoverとdecision/target時刻を保存する。
年率gross−netとinitial-equity基準cost ratioを別記する。grossは実口座経路のstep費用加算定義を保持する。
#16は別CSVの参考とし、新旧差と新たに含む年初PnLは記述に限定する。因果的な期間処置効果やpaired検定にしない。
margin callはterminal traceを残してincomplete、入力例外と実行例外を別の状態にする。

不足解消には元providerのpair別responseとcalendar根拠が必要。
新calendarや修復dataを使うなら新しい事前登録・source identityが必要であり、このtrialを成功するまで変更しない。
全方策の判断は現状「測定/データ不足で未判断」。次の小規模campaignには欠落解消と#30の利益帰属を引き継ぐ。
#16分類、独立収益性、実口座での実行可能性は変更・認定しない。

```bash
uv run forex-cost-campaign \
  --config configs/research/issue29_full_period_cost.json \
  --output docs/research/results/issue29

uv run python -c 'from pathlib import Path; from forex_trainer.cost_campaign import verify_campaign; print(verify_campaign(Path("docs/research/results/issue29"))["status"])'
```

再現時は別の新規outputを指定する。CLI exit 2は登録153セルの未完了、exit 1はconfig/実行全体の障害。
