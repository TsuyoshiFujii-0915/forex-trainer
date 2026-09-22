# Issue #30 後継利益帰属の実行前契約

trial_id: `issue30-contiguous-attribution-v1`。親: `issue27-development-v1`。
証拠: `development_historical`。市場実行前にこの契約・config・実装をcommitする。

## 入力・有限予算

#39の固定snapshot、#29後継の親manifestとF0 handoffをhash固定する。
全17区間（8全範囲＋9部分年）×5方策、85口座。元F0 51口座の帰属・同runtime再現51口座、
train replay17件、独立対照34口座、campaign1回。新規学習・取得・探索・サイズ選択0。
旧#30の40/85完走・45不足は保持し、新結果を継ぎ足さない。
範囲は#39で損益を使わずに固定済みのcoverageをそのまま採り、再選択しない。

## 会計・対照・訓練gap

ADR-0039/0044と既存attribute_account・train_allocation・CommonProjectionを使う。
price = common + relative、net = price + signed financing − spread − commission − markup。
実効weightは開始equity単位。各成分のlogは共通log1p(net)/net係数、net=0では1。
非正terminal equityのlogはnull。累積/年率log・JPYは実現経路の会計帰属であり単独CAGRではない。
commonは9pairの平均であり純粋JPY因子の識別ではない。

constantは元configのtrain [start,end)、元cache・63本warmup後の全decisionで実効weightを平均。
2003年の要求開始に対する2005年の実データ開始、訓練内の不足平日、gapをまたぐ旧replayと
実UTC時間によるfinancingを明示的に採用する。coverageの元train監査とtrain traceをsealする。
後継の連続区間規則をtrainへ遡及適用せず、validation/evalから平均・符号・サイズを推定しない。
train terminalなら平均を採用せず全体集約を保留する。
projectedは自身のassetsで毎decision再推論し、元提案をclip/capしてpair平均に投影する。
全口座は100万円・flat、共通cap=5、k=1、同close・F0費用。各経路で費用を計算する。
元PPOと同grossへの再拡大、固定action列再生、追加対照は行わない。

## 集約と判定

85セルの未完了はすべて残し、未完了があれば全体/era/CI/LOOをnullとする。
完走時は17区間等重み、era 2009–2018 / 2019–2025、10,000 IID・3-fold circular
moving-block、seed16、95%区間、最大寄与fold、全17 LOOを既存統計で算出する。
両対照−PPOのnet/gross対応差を出す。componentの年率logに加え期間log/JPY・期間return、
実経過時間、MDD、gross/net exposure、volatility、turnover、actual notionalを残す。
volatilityは既存帰属の標本標準偏差(ddof=1)と#29後継metricsの母標準偏差(ddof=0)を区別する。

結論は共通方向主要・相対配分主要・era依存・不確実の根拠を研究ノートに記し、事後閾値で分類しない。
次期候補は横断予測＋cost-aware配分、少数共通要因への制御、情報不足を検討する。
net差だけを因果的・リスク調整済み優越性としない。短区間年率化とcoverage選択の不確実性は残る。
旧通年完了、独立収益性、旧#16判断の変更は主張しない。
