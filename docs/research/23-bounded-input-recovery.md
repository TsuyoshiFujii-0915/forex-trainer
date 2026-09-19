# 有限の入力回復と連続区間スナップショット（Issue #39）

**出口B。元responseを回復できなかったため、既存保存値から再生成できる新しい研究用data/calendar版を確定した。**
`issue39-contiguous-snapshot-v1` は新providerや新取得価格ではなく、hash照合済みYahoo/FRED cacheから作る派生snapshotである。
17foldすべてに実行可能な後継区間を用意したが、そのうち9件は部分年。旧#29の81/153セル、#30の45/85セルは引き続き入力不足であり、解消セルは0。
後継区間の準備完了を旧17foldの実験完了へ読み替えない。本費用campaign・帰属診断は実行していない。

## 調査範囲と停止

[取得前の有限予算](protocols/issue39/recovery-budget.json)をcommitしてから、
[保存先・取得確認結果](protocols/issue39/recovery-evidence.json)を記録した。
`data/`、`runs/`、`docs/research/`、隣接envの`data/`、yfinance cacheを一巡した。
元pair別response、join前系列、元取得ログは見つからなかった。登録された遠隔archiveは0。
yfinanceのtimezone/cookie DBは価格responseではなく、cookie内容は公開していない。
未登録のバックアップまで不存在を証明したという主張ではない。

Yahooの不足日周辺のUSDJPY日足と、代替1sourceであるDukascopyの2009年2月USDJPY時間足に
各1requestだけ行い、両方HTTP 429だった。timeout20秒、最大1MiB、redirect0、retry0。
429を恒久的な取得不能や休場の証拠にはしない。有限予算が終了したので継続取得せず出口Bを採った。
response本体はgitignoredな`runs/issue39-recovery-probes/`、hash・取得時刻・HTTP状態は証拠JSONに保存した。

[Yahoo公式provider一覧](https://uk.help.yahoo.com/kb/SLN2310.html)は検索結果の公式ページ抜粋で確認したが、本文取得は429。
歴史的な日足cutoffや休日を確定する資料は得られなかった。
[Dukascopy公式Trading Hours](https://www.dukascopy.com/swiss/english/forex/forex-trading-accounts/link/)は現在の週次sessionを示すが、Yahooの不足日を説明しない。
両資料の取得日は2026-09-19、歴史的有効期間は不明として保存した。
日付が元旦・クリスマス・復活祭らしいことだけで休場認定していない。

## データと情報時点

[manifest](results/issue39/snapshot/manifest.json)、[全欠損pair監査](results/issue39/snapshot/missing-pairs.json)、
[lineage](results/issue39/snapshot/lineage.json)、[全範囲と除外時刻](results/issue39/snapshot/coverage.json)を固定した。
旧入力監査にある17不足日×9pair＝153組（fold/historyの重複込み180行）を列挙した。
各日付はraw joined、clean、carryのすべてで既に欠落している。
従って価格修復・carry付与で消えた行ではないが、元providerのpair欠損・OHLC欠損除去・inner joinでの脱落を区別できない。
全pairの元response有無、join loss、provider休場は未識別としてnullを残す。

元envのyfinance経路はpairごとのOHLC欠損行を除き、逆数変換でHigh/Lowを交換し、最後にinner joinする。
この変換codeはruntimeのenv source hashとSHAから追跡できるが、保存されなかった入力系列を復元するものではない。
raw→cleanは同じindex・columnsで33 OHLCセル／14行だけが変更され、clean→carryは元OHLCV値を完全保持することを検証した。
修復は未来の近傍値を使う事後処理であり因果的ではない。FREDの60日lagは既存recipeであり、当時のvintageや公開時刻を証明しない。
今回の価格修復・carry再取得・補間・forward-fill・モデル学習は0。

bar labelはEurope/Londonの00:00、session closeはそのlabelと同じという既存の歴史研究仮定。
DSTはtimezoneからUTCへ変換する。実際の`available_at`と元responseの`retrieved_at`は不明のまま。
今回の取得確認時刻を過去の利用可能時刻へ付け替えない。
9pair順、OHLC/CarryAnnual、8特徴量、window32、volatility32、normalize=false、k=1をmanifestへ固定した。
元train/validation/evaluationの要求範囲、実在最初/最後のlabel、平日不足一覧もfoldごとに残した。
trainの要求開始2003年に対してjoined cacheの開始は2005-07-19であり、この履歴不足も隠していない。

## 一つの後継契約

[ADR-0042](../decisions/0042-bound-input-recovery-and-register-contiguous-successor.md)と
[実行config](../../configs/research/issue39_input_recovery.json)を推論前に固定した。
選択基準は保存値のcoverage・取得可能性・timestamp仕様・再現性。戦略損益を参照しない。
期待calendarはLondon全平日・休日除外0を維持し、欠落日をcalendarから削除していない。

各foldの連続区間を列挙し、直前63本の連続履歴と2本以上の測定barを持つ区間のうち、測定bar数が最大の一つを採る。
同数なら早い区間。全候補と採用しなかった測定時刻も保存する。
欠損後は63本を揃え直す。各区間は100万円・flatから独立開始し、最終barはmarkのみ。
欠損をまたぐ持越し、再開時の仮想清算、独立区間の口座連結はしない。
週末・DSTの経過時間によるfinancing/markupは既存envのUTC秒数会計を維持する。

全期待barを利用できる年は2010/2015/2016/2020/2021/2022/2023/2024。
部分年の正確な開始・終了・first decision・last markはcoverage JSONを正本とする。
新calendarと派生cacheに新hashを付け、旧hashは変更しない。新manifestのidentityは訓練lineageとは別である。
元51 PPO memberと17 ridgeは元source sealのまま読み込み、訓練dataを新snapshotへ付け替えない。
価格値・pair・特徴量は同じなので入力形状の変更を要しない。データ供給元変更の感度実験を行ったとも称さない。

## preflight・smokeと引渡し

[preflight](results/issue39/preflight.json)は学習0・推論0。
既存#29/#30のconfig、元#29親artifact全150件、凍結model/sourceを照合し、旧153＋85セルのstatusを出力した。
後継17区間の3方策を既存`load_campaign_sources`で読み込み、派生cache値との完全一致も検証した。
#30のtrain_constant入力は元train範囲を実envで読み込み、flat reset・初回観測時刻を確認した。
訓練replayに含まれる過去のgapは旧意味論のままであり、後継範囲での利用は#30の明示判断を要する。
common_projectedは同じ凍結PPOの観測・action形状を利用できるが、新しい親F0本評価が必要である。

[smoke](results/issue39/smoke.json)では選択規則で最初となる2009区間の2transitionだけを
canonical/ridge/PPO ens3の独立口座で実行した。全3方策complete、pair・時刻・比較契約を照合済み。
traceと一時価格はgitignoredな`runs/issue39-contiguous-snapshot-v1/smoke/`へ保存した。
市場全体の成績・era平均・採否集計は出力していない。

#29/#30へ必要な判断は、**この部分年を含む後継範囲を新しい比較契約として採用して本評価するか、別の有限な取得予算を登録するか**。
後継範囲を採る場合、3対照をすべて同じ入力・評価器で再評価し、2帰属対照も同じ新しい親F0へ結び付ける。
旧153/85セルの完了条件や統計設計へ部分年結果を混ぜない。再学習を要する新source変更は別campaignへ渡す。
今回、#29/#30の本評価・最終判断を代行していない。

## 再生成

同じhashのローカルraw/clean/carry、凍結model群が必要。現providerへの再取得で元bytesを再現できるとは保証しない。
CLIは新規出力先だけを受け付け、原本とsealed成果物を上書きしない。

```bash
uv run forex-input-recovery build \
  --config configs/research/issue39_input_recovery.json \
  --output runs/issue39-reproduction/snapshot \
  --data-output runs/issue39-reproduction/data
uv run forex-input-recovery preflight \
  --snapshot runs/issue39-reproduction/snapshot \
  --report runs/issue39-reproduction/preflight.json
uv run forex-input-recovery smoke \
  --snapshot runs/issue39-reproduction/snapshot \
  --report runs/issue39-reproduction/smoke.json \
  --private-output runs/issue39-reproduction/smoke
uv run forex-input-recovery verify --snapshot runs/issue39-reproduction/snapshot
```

`build`は外部取得も推論も行わない。`preflight`成功は入力検査の成功であり、旧campaign完了ではない。
欠損・休場・履歴不足・DST・gap後再開・pair脱落・NaN・時刻重複・未知bar・公開遅延・改変・上書き拒否・境界の待機時間を回帰テストで検証する。

実装commitは `04983ee03d0cbfeea070de18e666d65a9a72ef5a`。別出力先への[再生成照合](results/issue39/reproduction.json)で、派生cache17件とcalendar・監査などmetadata20件がbyte単位で一致した。出力先を含むmanifest identityは別attemptとして区別する。

最終検証は `uv run pytest -q` で470件通過（除外0）、新規回帰20件。
[引渡しseal](results/issue39/manifest.json)はsnapshot・preflight・smoke・再生成照合の全25artifactをhashで固定している。
