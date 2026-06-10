# 0003. 実験成果物はローカルの run ディレクトリに完結させる

## Status

accepted

## Context

実験トラッキングには MLflow / W&B 等の外部サービスやサーバーを使う選択肢があるが、現段階は単独研究者のローカル(Mac / 単一GPUマシン)運用であり、セットアップと運用のコストが利益を上回る。一方で「あの結果はどの設定で出たか」が失われると実験は無価値になるため、再現に必要な情報の保存は省略できない。

## Decision

トラッキングは完全ローカルとし、`forex-train` が run ごとに `runs/<実験名>/<UTCタイムスタンプ>/` を自動生成して以下を封入する。

- `config_snapshot.yaml` — 実験YAMLの完全コピー
- `env_train.yaml` / `env_eval.yaml` — 日付レンジ注入後の解決済み環境設定
- `meta.json` — 両リポジトリの git SHA、主要パッケージのバージョン、シード、解決済みデバイス
- `tensorboard/` — 学習曲線(SB3標準)
- `model_final.zip` — 学習済みモデル
- (評価後) `metrics.json` / `equity_curve.csv` — `forex-eval` が追記

`configs/` は git にコミットし(実験設計の履歴)、`runs/` は gitignore する(結果はローカル)。run の比較は `forex-compare` が `metrics.json` を集計して行う。

## Consequences

- 外部依存ゼロで、オフラインでも実験の定義・実行・比較が完結する
- マシンを跨いだ結果共有は runs ディレクトリのコピーで行う(必要が生じたら MLflow 等の導入を新 ADR で判断)
- run ディレクトリの肥大はユーザーが手動で管理する
