# 0006. Dukascopy 公開データフィードによる長期時間足キャッシュの構築

## Status

accepted

## Context

時間足の訓練データは yfinance の制約(60分足は直近約730日)により約7.8千バーしかなく、ラウンド1/2の実験ではシード間分散が大きい(同一設定で Sharpe が −3.1〜+1.1)。観測次元(窓×ペア×特徴量)に対して訓練データが少なすぎることが、過学習と学習不安定の主因である。Dukascopy Bank の公開データフィード(datafeed.dukascopy.com)は認証不要で2003年頃からの時間足キャンドル(月単位の LZMA 圧縮 bi5、ビッグエンディアン `(Δ秒, open, close, low, high, volume:float32)`、JPY建てペアは価格×1000)を配信しており、実測で取得・解読できることを確認した(2020-01 USDJPY: OHLC整合違反 0/528、バー間連続性 中央値 0.9bp)。

## Decision

trainer に `forex-fetch-dukascopy` CLI(`src/forex_trainer/dukascopy.py`)を追加し、Dukascopy の月単位 BID 時間足キャンドルから forex-env の file プロバイダー契約に適合する parquet キャッシュを構築する。

- 環境リポジトリは変更しない。書き出しは forex-env の公開関数 `save_ohlcv_parquet`、反転は `invert_quote`、検証は `validate_ohlcv` を再利用する(ADR-0001 の依存方向を維持)
- volume ≤ 0 のバー(週末・休場)は非取引バーとして除外する
- タイムスタンプは UTC。価格は BID キャンドルを使用し、JPY/XXX への反転は env と同一の規則(High↔Low スワップ)による
- ネットワーク境界(月ファイルのダウンロード)は注入可能な関数とし、テストは合成 bi5 バイト列で解読・変換・結合を検証する
- データは個人研究用途での利用にとどめる(Dukascopy の歴史データは個人・非商用利用向けに公開されている)

## Consequences

- 訓練データが約10倍(2015年以降で約7万バー/ペア)になり、複数レジーム(2016 Brexit、2020 COVID、2022-24 日米金利差)を含む
- 学習・検証・評価の分割を長期データで再設計できる(walk-forward の余地)
- yfinance キャッシュとの価格系列の乖離はスプレッド分(BID vs 不明側)程度残る
- Dukascopy 側の仕様変更・提供停止リスクを負う(キャッシュ parquet を保持することで再取得依存を回避)
