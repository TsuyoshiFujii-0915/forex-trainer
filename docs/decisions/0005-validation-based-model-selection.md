# 0005. 検証区間によるベストモデル選択(early stopping 相当)

## Status

accepted

## Context

同一設定で total_timesteps を 50k → 500k に増やすと評価成績が悪化する事例が観測された(ppo_mlp_1h seed 42: Sharpe +2.0 → −2.1)。現行の train は学習終了時点のモデルのみを保存しており、学習が進むほど訓練データへの過学習が評価成績を毀損する。時系列データでの標準的な対策は、訓練区間とも評価区間とも重ならない検証区間での成績によるモデル選択である。

## Decision

実験 YAML に必須キー `val_range` を追加し、`train_range.end ≤ val_range.start < val_range.end ≤ eval_range.start` を load 時に強制する。学習は次のプロトコルで行う:

- 学習中、SB3 の `EvalCallback` が検証区間の全範囲を決定論的に1回歩く評価を定期的に(約20回)実施する
- 検証成績が最良のモデルを `model_final.zip` として保存する(= 実験の成果物。forex-eval のインターフェースは不変)
- 学習終了時点のモデルは `model_last.zip` として併置する(過学習の診断用)
- 検証成績の履歴は `evaluations.npz` として run ディレクトリに残る
- 解決済みの検証環境設定を `env_val.yaml` として封入する(ADR-0003 の再現性契約の拡張)

eval_range は最終評価専用であり、モデル選択には一切使わない。

## Consequences

- 総学習ステップ数の選択に対する感度が下がり、「長く学習したら悪化した」という失敗モードが除去される
- 訓練データが検証区間の分(直近3ヶ月)だけ減る
- 検証区間への軽度の選択バイアスは残る(eval_range での最終評価がこれを検出する)
- 既存 config は val_range の追加が必要(train_range 末尾から切り出す)
