# 0002. Stable-Baselines3 を単一バックエンドとし、実験空間を3軸レジストリで構成する

## Status

accepted

## Context

「いろんなモデルを試したい」という要求には、(a) 複数RLライブラリを抽象化する層を作る、(b) 単一ライブラリの拡張点に集中する、の2案がある。(a) は将来の拡張性のための抽象化であり、コストが先行し当面の実験には寄与しない。forex-env-v3 は Dict 観測 + Box 行動の Gymnasium 環境で、SB3 の `MultiInputPolicy` がそのまま適用でき、PPO での実動は環境側検証で確認済み。トレーディングRLで成績に効くバリエーションはアルゴリズムだけでなく、市場観測 `(ペア数, 窓長, 特徴量数)` をどう符号化するか(ネットワーク)と、何を観測に入れるか(特徴量)にある。

## Decision

Stable-Baselines3 + sb3-contrib を唯一のバックエンドとし、実験空間を3つのレジストリの直積として構成する。

1. **アルゴリズム軸** `ALGO_REGISTRY`: ppo / recurrent_ppo / sac / td3 / tqc(クラスとポリシー名の対)
2. **ネットワーク軸** `NETWORK_REGISTRY`: mlp / cnn1d / lstm / attention — SB3 の `BaseFeaturesExtractor` 差し替えとして実装し、YAML の `network:` で選択
3. **特徴量軸** `FEATURE_REGISTRY`: trainer 側で定義する因果的な特徴量関数。YAML の `env.features.selected` の名前で選択され、env の `custom_features` として構築時に注入される

他バックエンドへの抽象化レイヤーは作らない。SB3 で表現できないモデルが実際に必要になった時点で新 ADR で判断する。

## Consequences

- 新規追加コストは「特徴量=関数1個」「ネットワーク=クラス1個」「アルゴリズム=レジストリ1行」に局所化される
- レジストリに登録された全特徴量は、ルックアヘッド検査(データ末尾延長の不変性)とenv内での有限性検査をテストで自動的に受ける
- SB3 の設計制約(オンポリシー/オフポリシーの共通APIなど)を受け入れる
