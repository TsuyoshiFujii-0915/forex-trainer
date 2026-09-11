# Issue #30 実行前研究契約

trial_id: `issue30-profit-attribution-v1`。親: `issue27-development-v1`。
証拠区分: `development_historical`。登録commitはこの文書を最初に含むcommit。
市場実行前にcommitし、実行SHA・source bytes・開始終了時刻を結果manifestへ保存する。
この登録は実行結果で上書きしない。

## 固定範囲・予算

2009〜2025の全17fold、era 2009〜2018 / 2019〜2025。#29 F0だけを主入力に使う。
[config](../../../../configs/research/issue30_profit_attribution.json)で親seal、研究契約、lockをhash固定。
PPOはcurrent-provenance validation-best seed42/43/44、ridgeは#15凍結係数/standardizer/alpha、
canonicalはmom24 reversal各±0.8・上下各2・既存tie処理。
新規学習・HPO・seed/checkpoint選択・追加対照・サイズ探索・regime再screenは0。
診断campaign 1、元F0の51帰属、最大51再現確認、17 train replay、34対照eval。
不足sourceはpath/hashと親errorを引き継ぎ、都合のよいfoldだけの集約は行わない。
legacy先行解析は今回実行しない。F1/F2、旧年内warmupとは混合しない。

## 会計・対照

[ADR-0039](../../../decisions/0039-attribute-profit-and-replay-two-frozen-controls.md)の式と順序を固定する。
毎step、effective weights、pair順、decision/target、開始equity連続性、price/carry/費用を照合。
returnのrtol=1e-10、atol=1e-12。円単位は開始equityで除して照合する。
logは全体netの共通変換係数を配賦し、累積log・年率log・実現JPY寄与を別列に保存する。
成分和は全体に一致するが、成分単独運用CAGRではない。commonはこの9pair座標系の平均であり、
純粋JPY factorや独立因子とは断定しない。2固定mapのcommonほぼ0は構造確認に限る。

constantはtrain [start,end)の既存warmup後のdecision等重み平均。学習区間の記述的対照であり
独立alpha発見ではない。train terminalなら配分を作らない。評価期間で平均・符号・サイズを推定しない。
projectedは独立口座のobservationでPPOを毎decision再推論し、clip/cap後weightのpair平均へ投影。
全方策のcapはsource config、初期状態は100万円・flat。float32 rounding後に既存capを適用する。
投影はtieなし、平均0ならflat。費用は各口座の時価建玉と実売買から計算する。
固定action列再生、relative-only、gross合わせ、事後リスク調整は行わない。

## 集約・判断

全85 policy-foldのstatusを保存し、未完了があれば個別foldの記述値のみを報告する。
全体・era・最大fold寄与・全LOO・対応CIはnullとし、判断は不確実/情報不足と明記する。
全完走時はfold等重み、10,000 IIDと3-fold circular moving-block bootstrap、seed16。
common/relative/financing/各費用/netの累積・年率log寄与、JPY寄与、net/gross return、MDD、
gross/net exposure、年率net log-return volatility、turnover・actual notionalを保存する。
元PPOと各対照のnet/gross対応差に同じ既存fold統計を用いる。日次/seedを独立標本に加算しない。

結論候補は共通方向主要、相対配分主要、era依存、不確実。事後境界値で自動分類しない。
対照間のnet差だけで因果的優越性を主張しない。次期候補は横断予測＋cost-aware配分、
少数共通要因への制御、情報不足のいずれかを根拠とともに研究ノートに明示する。
現時点では#29の9欠損foldが既知であり、最終17fold結果の未完了を隠さない。
不足の取得元は#29 input-auditに記された元providerのpair別responseとcalendar根拠。
新しいdata/calendarを採用する場合は新しい親契約・seal・登録を必要とする。
