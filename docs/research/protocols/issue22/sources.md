# Issue #22 出典台帳

確認日: **2026-09-10（Asia/Tokyo）**。Webは以下の公開一次資料を確認した。
外部資料の内容は要約であり、現在の仕様を2009〜2025年の口座仕様へ遡及適用しない。
対象broker・法域・口座・商品は未指定。OANDAの資料は相違点を示す例であり、運用先の選定ではない。

## 保存済みコード・成果物

| ID | 固定出典 | 確認内容 |
|---|---|---|
| T1 | [trainer evaluate.py](https://github.com/TsuyoshiFujii-0915/forex-trainer/blob/1c015d2974feec6174b1e184a6f2b0380a9a70fb/src/forex_trainer/evaluate.py)、`compute_metrics` | 経過秒/365.25日、step数によるSharpe年率化、cost加算によるgross |
| T2 | [trainer supervised_ranking.py](https://github.com/TsuyoshiFujii-0915/forex-trainer/blob/1c015d2974feec6174b1e184a6f2b0380a9a70fb/src/forex_trainer/supervised_ranking.py)、`build_aligned_dataset` | decision index=warmup+window−1、次bar target |
| T3 | [trainer carry.py](https://github.com/TsuyoshiFujii-0915/forex-trainer/blob/1c015d2974feec6174b1e184a6f2b0380a9a70fb/src/forex_trainer/carry.py) | FRED系列registry、fredgraph取得、日次ffillとlag、local dateへの整列 |
| T4 | [trainer data_quality.py](https://github.com/TsuyoshiFujii-0915/forex-trainer/blob/1c015d2974feec6174b1e184a6f2b0380a9a70fb/src/forex_trainer/data_quality.py) | 未来側近傍も使うspike検出・幾何補間 |
| T5 | [trainer features.py](https://github.com/TsuyoshiFujii-0915/forex-trainer/blob/1c015d2974feec6174b1e184a6f2b0380a9a70fb/src/forex_trainer/features.py) | mom24、sma20_ratio、cross-sectional特徴量、carry特徴量 |
| E1 | [env.py](https://github.com/TsuyoshiFujii-0915/forex-env-v3/blob/6024b91c0f3592611849bc231922ab60e6090aed/src/forex_env/env.py)、`__init__` / `reset` / `_observe` / `step` | range切出し後warmup、tまでの観測、close tでrebalance、close t+1でmark |
| E2 | [accounting.py](https://github.com/TsuyoshiFujii-0915/forex-env-v3/blob/6024b91c0f3592611849bc231922ab60e6090aed/src/forex_env/accounting.py) | 実notional、価格適用前overnight、適用後signed financing |
| E3 | [file_provider.py](https://github.com/TsuyoshiFujii-0915/forex-env-v3/blob/6024b91c0f3592611849bc231922ab60e6090aed/src/forex_env/data/file_provider.py)、`get_data` / `save_ohlcv_parquet` | `.loc[start:end]`の終了日込み切出し、保存metadata |
| E4 | [yfinance_provider.py](https://github.com/TsuyoshiFujii-0915/forex-env-v3/blob/6024b91c0f3592611849bc231922ab60e6090aed/src/forex_env/data/yfinance_provider.py) | ticker反転、High/Low入替、欠損価格行除去、全pair inner join、index保持 |
| E5 | [features.py](https://github.com/TsuyoshiFujii-0915/forex-env-v3/blob/6024b91c0f3592611849bc231922ab60e6090aed/src/forex_env/features.py) | volatility_window行warmup、window内正規化 |
| A16 | [#16 provenance](../../results/issue16/provenance.json)、[steps](../../results/issue16/steps.csv)、[metrics](../../results/issue16/fold_metrics.csv) | seal `d2db580d5f11ec7ca341af04549c3a09e7ddb6777c43562b1117a24ae8a57db3`。全17 resolved configとmodel/data chain |
| A19 | [#19研究ノート](../../13-frozen-spread-to-net-decomposition.md) | 既存のprice/carry/cost/net分解とgross不確実性。再実装しない |
| A20 | [#20運用契約](../issue20/forward-operations.md)、[identity](../issue20/artifact-identities.json) | broker/data未決、年内warmup、2027候補、append-only記録 |
| A21 | [#21研究ノート](../../15-frozen-turnover-and-cost-diagnostic.md)、[ADR-0034](../../../decisions/0034-separate-overnight-fees-from-trading-costs.md) | marked exposure再構築、実取引と保有費用の区別。再実装しない |

T1〜T5は#16評価revisionと監査開始時main `83e2e31`の間で差分なしを確認。
E1〜E5のローカルHEADは上記#16評価revisionと一致、作業treeはcleanだった。
#16が封印した依存versionは新しい監査の[provenance](../../results/issue22/provenance.json)にも継承する。

## FRED系列の公式定義

すべてpercent・季節調整なし。JPY/XXXのcarryは下記XXXから共通JPY系列を引き0.01倍する（T3）。
「すべて月次政策金利」という理解は不正確。accepted ADR-0008は書き換えず、本台帳で補足する。

| 通貨 | ID・一次資料（確認日はいずれも2026-09-10） | 定義・頻度 |
|---|---|---|
| USD | [FEDFUNDS](https://fred.stlouisfed.org/series/FEDFUNDS) | effective federal funds rate、日次値の月平均。政策目標そのものではない |
| EUR | [ECBDFR](https://fred.stlouisfed.org/series/ECBDFR) | ECB deposit facility、日次7-day |
| GBP | [IR3TIB01GBM156N](https://fred.stlouisfed.org/series/IR3TIB01GBM156N) | OECD 3-month/90-day interbank、月次 |
| AUD | [IR3TIB01AUM156N](https://fred.stlouisfed.org/series/IR3TIB01AUM156N) | 同上、Australia |
| CHF | [IR3TIB01CHM156N](https://fred.stlouisfed.org/series/IR3TIB01CHM156N) | 同上、Switzerland |
| CAD | [IR3TIB01CAM156N](https://fred.stlouisfed.org/series/IR3TIB01CAM156N) | 同上、Canada |
| NZD | [IR3TIB01NZM156N](https://fred.stlouisfed.org/series/IR3TIB01NZM156N) | 同上、New Zealand |
| NOK | [IR3TIB01NOM156N](https://fred.stlouisfed.org/series/IR3TIB01NOM156N) | 同上、Norway |
| SEK | [IR3TIB01SEM156N](https://fred.stlouisfed.org/series/IR3TIB01SEM156N) | 同上、Sweden |
| JPY | [IR3TIB01JPM156N](https://fred.stlouisfed.org/series/IR3TIB01JPM156N) | 同上、Japan。全pair共通の控除系列 |

## 公開・執行仕様

| ID | 一次資料 | 確認した範囲・限界 |
|---|---|---|
| W1 | [FRED API Real-Time Periods](https://fred.stlouisfed.org/docs/api/fred/realtime_period.html) | FREDの通常値は現在わかっている過去。ALFREDのreal-time期間指定は過去時点の情報を区別する。現在の取得値を60日遅らせるだけではvintage検証にならない。各系列の過去vintage網羅性は今回未検証 |
| W2 | [OANDA US financing fees](https://www.oanda.com/us-en/trading/financing-fees/) | 17時ETの保有にlong/short別rate、tom-next由来+admin、通常水曜3日分、休日で変動。indicativeとfinalを区別。現在の米国仕様の例に限定し、対象口座のrateには使わない |
| W3 | [yfinance PriceHistory API](https://ranaroussi.github.io/yfinance/reference/yfinance.price_history.html)（Context7公式repo文書で確認） | historyのstartはinclusive、endはexclusive。日足indexのlabelだけではOHLCの確定・取得可能時刻を証明しない |
| W4 | [pandas IO](https://pandas.pydata.org/docs/user_guide/io.html)、[timedelta](https://pandas.pydata.org/docs/reference/api/pandas.Series.dt.total_seconds.html)（Context7で確認） | CSV読取と実経過秒による台帳生成。UTC offsetの違いを維持して計算 |

Web資料は出典と確認日・要約を保存した。元のhistorical Yahooレスポンス、FRED vintageレスポンスや
対象brokerの公式契約書は取得できておらず、これらを保存済みと主張しない。
