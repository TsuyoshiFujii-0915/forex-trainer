# 開発用shadow記録adapter（Issue #32）

契約は `development_shadow`、execution basisは `shadow_quote`。
**実装検証済み／実収集未開始**。外部source・broker・口座は未指定である。
#20の2027年凍結確認は開始せず、日程も旧schemaも変更しない。
開発用の損益閲覧は可能だが、この記録を後から独立holdoutへ昇格させない。

## 実行例

リポジトリrootから実行する。`uv sync --group dev`でCLIと既存のschema検証依存を導入する。
保存先は新規directoryのみ。旧runや既存eventの上書きは拒否する。

```console
uv run forex-shadow init --manifest configs/research/issue32_shadow_fixture.json
uv run forex-shadow cycle --store runs/issue32-frozen-fixture --cycle day-0
uv run forex-shadow resume --store runs/issue32-frozen-fixture --cycle day-1
uv run forex-shadow verify --store runs/issue32-frozen-fixture
uv run forex-shadow head --store runs/issue32-frozen-fixture --output runs/issue32-frozen-fixture-head.json
```

この例は**実在する2025 foldの凍結ridge、canonical mom24 reversal、PPO全seed42/43/44**を、
合成OHLCV/carry/quoteに適用する実装検証である。
[source manifest](../../configs/research/issue32_sources.json)は親seal、fold、env SHA、依存version、CPUを固定する。
#20の[artifact identity](protocols/issue20/artifact-identities.json)と照合し、validation-best model、
config/meta、ridge standardizer/alphaのhashを検証する。latest・成績選択・再学習・別foldへの代用はない。
旧bundleを新しい数量会計のassetsへ適用する制約も保存する。
欠けたmodelのpath/hashは例外になる。実modelを同梱・再配布するものではない。

ローカルにモデルがない環境のportable検証には、別trialの
`tests/fixtures/issue32/manifest.json`を明示指定する。
こちらだけはsynthetic ridgeと未学習実PPO seed42/43/44を使う。
`policy_mode=synthetic_fixture`を実観測へ指定することは禁止する。自動切替はない。

## 1サイクルの意味

1. 保存済みmanifest、runtimeのsource bytes/依存、model/configを検証し、writer lockを取る。
2. manifestに登録したraw fileの**原bytes**を保存し、欠損・時刻・9pair順・source/qualityを検証する。
3. decision自身を含む64本から既存FeaturePipelineで32×8 windowを作る。先頭63本は観測履歴で、売買しない。
4. 各独立口座のassetsで3方策を推論する。score、提案/clip後action、入力・model/config/data hash、
   feature/lag順、想定費用を3行の原子的batchとして保存する。
5. 全decisionのfsync完了を確認してbarrierを追記する。その時点より後の最初の適格quoteを
   **#31と同じ `execute_quote`**へ渡す。gap、bid/ask、commission、signed financing、markup、数量、markを追跡する。
6. quote/mark原bytesと計算根拠を保存し、3 outcomeを原子的に追記する。これが口座のcommitとなる。
   equity/数量/assetsを再現できるhead付きcheckpointを保存する。

ファイル未到着のsettlementは`awaiting_settlement` incidentを残す。
後の`resume`は保存済みdecisionを使い、再推論しない。
既に完了したcycleは`already_complete`となり、decision/outcomeを追加せず口座も二重適用しない。
quote/financingが欠けた不完全なファイル、不適格quote、hash/会計不一致、deadline超過では
incidentを残してtrialを停止する。不足を0や別barで補わず、方策を間引かない。
訂正して同じtrialへ過去decisionを補完する操作はない。

想定費用が未指定ならv2の`expected_*`は明示的な`null`と`quality=unavailable`。
これはゼロ費用ではない。観測costとsigned carryは完全なquote/financingがあるoutcomeでのみ数値化する。
sourceのsource名、quality、vintage、available_at/retrieved_atはrawと入力artifactに保持する。

## 明示manifestと実sourceに必要な設定

[fixture manifest](../../configs/research/issue32_shadow_fixture.json)を新しいtrialとして編集する。
全キー必須、未知キーは拒否。相対pathは実行cwd基準で読み、登録時に絶対pathへ固定する。
既存manifestは変更せず、登録後のcode/config変更も別trialにする。

| 設定 | 必須内容 |
|---|---|
| `mode` | `fixture`は仮想時刻。`observed_file`は実時計で新規登録し、未来の測定開始を要求 |
| `source` | 利用者指定read-only source ID、bar時刻の意味、carry lineage。外部取得処理は本CLIに含めない |
| `schedule` | cycle ID、session close、decision最早時刻/deadline、target、入力/settlement path、正確な63本の履歴＋当日event time |
| 測定範囲 | `[measurement_start, measurement_end)`内にdecisionとtarget。次cycleのsession closeは前回markと一致 |
| raw | 9pair順、OHLCV/CarryAnnual順、全rowのevent/available/retrieved時刻、carry公開/取得時刻とvintage |
| quote/mark | #31 schemaの同時9商品bid/ask/depth、発生・公開・取得時刻、tradable、source/quality、mark |
| financing | #31の有効区間、long/short別signed JPY/base/day、公開/取得時刻。decision時点で既知であること |
| 費用・商品 | lot/minimum/margin、commission、markup内包条件、休日/session/timezoneを登録。brokerを記すなら公式仕様証拠も必要 |
| 方策 | `frozen_2025`とsource manifest hash。旧bundle制約を受け入れ、CPU/runtimeを一致させる |
| 保存 | 新規`storage`、その外の`failure_storage`、初期flat100万円、`stop_policy`、確認昇格不可 |

`observed_file`のrawとquote/markはquality=`observed`のみ。
その時点で実際に取得されたsnapshotを利用者のcollectorが用意する必要がある。
後日取得した日足のlabelをavailable_atへコピーすることは実前向き観測の証拠にならない。
fileの宣言値だけでproviderの真正性を保証するものでもない。
#31のJPY quote線形商品、同期basket、financing積分、markの公開時刻条件をそのまま継承する。
非対応の部分約定やbroker rolloverを捏造しない。

実sourceが未指定なので、実観測smokeは実行していない。口座作成、契約、発注、ネットワーク収集もしない。

## 保存・障害・再開

[ADR-0041](../decisions/0041-record-development-shadow-in-atomic-event-batches.md)に従うPOSIX local filesystem専用。
NFS/分散filesystem/Windowsでの同等のlock・hard link・fsync保証は検証していない。

- `events/000000000000.jsonl`以降を名前順に連結すると単一のJSONLになる。
  各batch先頭の連番をfilenameにし、行ごとにsequence、固有event ID、前行のLFを除くraw bytes hashを記録する。
- 一時fileをfsyncし、hard linkで上書きせず公開してdirectoryをfsyncする。
  公開済みbatchの一部だけを更新するAPIはない。artifactはcontent hashで保存する。
- checkpointは検証済みheadと口座の派生物。欠けていても正本のoutcomeから復元し、不一致なら停止する。
- 保存失敗では別の`failure_storage`へ原因/origin/時刻を記録して例外にする。
  同じdiskが満杯なら両方失敗し得る。独立したvolumeを用意するのは運用側の責任で、二次障害も例外で通知する。
- 途中の`staging/`が残ったら通常cycleを拒否する。`recover --store ...`で元bytesを`orphans/`へ保存し、
  復旧incidentを追記する。別保存先の未確認障害も本ログへ取り込んでから再開する。
  3 decisionが既にcommit済みならそのdecisionから再開できる。
  rawだけ保存してdecisionが失われたcycleは停止し、後日再推論で補完しない。
- 公開済みJSONLが途中で切れている場合やchain不一致は、`recover`でも切り詰め・書換えしない。
  原本を保全して調査する。directory fsync前後の不確かな公開は、再開時に再fsyncしてから扱う。

## 訂正・閲覧・監査

```console
uv run forex-shadow correct --store runs/issue32-frozen-fixture \
  --event-id correction-001 --target day-0/decision/canonical_reversal \
  --reason 'source quality note' --evidence path/to/correction-evidence.json
uv run forex-shadow show --store runs/issue32-frozen-fixture \
  --event-id view-001 --reason 'development review'
uv run forex-shadow trial --store runs/issue32-frozen-fixture \
  --event-id design-001 --kind design_change --reason 'new candidate registered as a separate trial'
```

correctionは元event参照と理由・証拠だけを追記し、元bytes/実行済み口座を遡及変更しない。
`show`は成績閲覧をtrial ledgerへ保存してから表示する。CLIを通さないOS上の閲覧は制限できないため、
利用者も閲覧・設計変更をledgerへ記録する。`verify`と`head`は損益を表示しない。

v1のdecision/outcome/incident/correctionを基にした
[v2 schema](../../src/forex_trainer/shadow_record.schema.json)を使う。v1 schemaを上書きしない。
`verify`はSchema + FormatCheckerに加えて、artifact実体、時刻不等式、raw→feature再計算、
参照、3方策barrier、clip、quote会計の再計算、equity/数量/assets連続性、checkpointを検証する。
単なるJSON適合を記録の正しさとは扱わない。

head exportは新規fileへの出力機能で、独立保管への自動送信ではない。
権限分離、別主体による定期head固定、バックアップ・復元訓練、閲覧制御は実運用前に別途必要。
**local hash chainだけで管理者による全面改変を防げるとは主張しない。**
短期fixtureやquote観測は、独立確認済み収益性・実口座fill・長期運用の証明ではない。

## 検証

```console
uv run pytest tests/test_development_shadow.py tests/test_shadow_audit_edges.py \
  tests/test_quote_step.py tests/test_quote_replay.py tests/test_quote_replay_edges.py \
  tests/test_shadow_storage_failures.py tests/test_forward_record_schema.py -q
```

全suite **450 passed**、新shadow回帰 **31 passed**。
実固定bundleのCLI smokeは2 cycle、raw 4行、decision 6行、outcome 6行、barrier 2行、登録1行の
合計19 eventで完了した。完了cycleの再実行は`already_complete`で元bytesを保持する。
[検証記録・artifact hashes](results/issue32/fixture-smoke.json)と
[head export](results/issue32/fixture-head.json)を保存した。仮想PnLから経済的結論は出していない。

実装時はContext7 MCPでSB3 deterministic inference、jsonschema Draft202012Validator/FormatChecker、
pandas DataFrame/MultiIndexの最新ドキュメントを確認した（2026-09-17）。
