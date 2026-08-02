# 0011. run成果物に不変なデータ・コード来歴を封入する

## Status

accepted

## Context

ADR-0003は設定スナップショット、Git commit、依存バージョンをrunディレクトリへ保存するが、
入力Parquetの内容を識別していない。同じパスのキャッシュが置き換わると、`forex-eval`は学習時と
異なるデータを検知せず評価できる。またcommitが同じでもdirty worktreeの内容が異なれば、実際に
実行したコードは同一ではない。FRED系列、発表ラグ、キャッシュ契約も実験条件だが、既存の
run成果物はそれらを評価時に照合しない。

## Decision

ADR-0003の成果物契約を、明示的にversion管理されたprovenance契約で拡張する。

- `meta.json`に`run_provenance_contract_version`を保存する
- file providerでは、解決済み絶対パス、Parquetファイル全体のSHA-256、行数、実インデックスの
  最初と最後、forex-envのschema version、carry contract、timeframe、宣言範囲を保存する
- Parquet metadataに存在する`forex_trainer_*`項目を、FRED系列、発表ラグ、factor生成条件を含む
  augmentation provenanceとしてそのまま保存する
- synthetic providerでは、各splitの解決済み環境設定をcanonical JSON化してSHA-256を保存する
- mutableなremote providerから直接runを作らず、file provider用の不変キャッシュ作成を要求する
- file providerの相対パスは従来どおりtrain起動時の作業ディレクトリを基準に解決し、解決済み
  絶対パスをenv snapshotにも保存する。これにより評価時は作業ディレクトリに依存しない
- trainerとenvについて、commit、dirty状態、tracked diffとuntracked file内容から得た
  code worktree SHA-256を保存する。対象は実行コードの`src/`とpackage解決を決める
  `pyproject.toml` / `uv.lock`とし、別途fingerprintを持つデータ、成果物、テスト、文書は除外する
- `forex-eval`はモデルやデータを使用する前に、provenance契約version、評価データ、両repository、
  依存パッケージを記録値と照合する。不一致は原因を示す例外とする
- provenanceのない旧runを暗黙に許可しない。旧runを現行契約のrunとしてbackfillすることも
  禁止する。旧modelを参考評価する場合は、元学習時の来歴が不明であることを保持する別の明示的な
  legacy evaluation attestation契約を将来のADRで定義する

## Consequences

- 同じパスに別データが置かれても評価前に検出できる
- dirtyな実行コードで行った実験も、その時点のsource treeと同一かを照合できる
- runディレクトリを別マシンへ移す場合、データパスだけでなく同一内容のキャッシュを契約に沿って
  配置する移行手順が必要になる
- 既存runは通常の`forex-eval`では明示的な移行エラーになる
- remote providerを使う実験には、先にキャッシュを固定する一手間が必要になる
