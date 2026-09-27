# 次期の共通方向予測・配分・逐次RL実行manifest（Issue #40）

**`issue40-common-direction-development-v1`を登録する。分岐は共通方向のみ。**
登録日2026-09-26。状態は契約確定・新規学習0・新規市場評価0。
正本は本manifest、入力hash・登録commit・全foldの範囲・cross-fit日程・試行台帳は
[registration.json](protocols/issue40/registration.json)。実行CLI configではない。
実装と本実験は#41〜#50の担当とし、本Issueは登録までで完了する。
[ADR-0045](../decisions/0045-register-one-bounded-common-direction-learning-campaign.md)に基づく。

## 1. 分岐と比較可能な範囲

入力は[#39出口B](23-bounded-input-recovery.md)の`issue39-contiguous-snapshot-v1`。
[#29後継](24-successor-cost-baselines.md)153口座、[#30後継](25-successor-profit-attribution.md)85口座と
[判断記録](protocols/issue30-successor/postrun-assessment.json)を引き継ぐ。
PPOの区間平均年率log寄与はcommon +5.378、relative −0.767ポイント。
両eraのcommon平均は正、projected対照のnet差も正だが、commonの区間推定はゼロをまたぎ、
projectedのworst MDDは43.33%。**主要な実現会計寄与と、持続的な学習alphaの識別は別である。**
自動reportの`uncertain_not_identified`を消さず、共通方向の符号・量・費用制御を一つの探索的仮説に選ぶ。
純粋JPY因子とは呼ばず、横断相対との同時探索・事後ブレンドは0とする。

2009〜2025の17区間を等重みで使う。eraは2009〜2018／2019〜2025。
範囲は[#39 evaluation](results/issue39/snapshot/evaluation.json)と
[coverage](results/issue39/snapshot/coverage.json)の全17行をそのまま採用する。
8区間は全期待bar範囲、9区間は部分年。元の#29は81セル、#30は45セルが入力不足のまま。
この範囲は既存coverage規則で損益を見る前に選ばれたが、今回の仮説選択ではその結果を閲覧済み。
旧17通年の完了、新しい独立標本、欠損期間込みの年間損益と称さない。
新予測の不調・欠損を理由に範囲を縮めず、17区間の比較が成立しなければ不足解消へ戻る。

## 2. 入力、時刻、fit境界

9pair順は `JPY/USD, JPY/EUR, JPY/GBP, JPY/AUD, JPY/CHF, JPY/CAD, JPY/NZD, JPY/NOK, JPY/SEK`。
元carry cacheのSHA-256は`723db7c935dcc27c147007728358d098243eae96998eae146db4c7df02bd081e`。
評価は#39の派生cache/calendar hash、訓練は同じ元cacheとfold別longf configの要求範囲を使う。
要求開始2003-06-01に対する実在開始2005-07-19、元response不在、14行の未来近傍OHLC修復、
FRED vintage/公開時刻不明・既存60日lagを引き継ぐ。現在取得値へ差し替えず、point-in-timeと認定しない。

London全平日・休日除外0・label=London午前0時=仮定close、DSTはEurope/LondonからUTC変換。
`available_at`/元`retrieved_at`は不明のまま。同closeは歴史研究仮定であり実執行可能性ではない。
観測window32、volatility32、normalize=false、必要history63本、decision interval=1を保持する。
新fit用のfeature/labelは**欠損をまたがない連続期待barブロック**内で作る。
不足行・gap・先頭history不足・無効labelを理由別に保存し、補間やh営業日→h観測行への読み替えをしない。
旧modelの訓練gap契約を遡及変更せず、新fitの有効行集合を別artifactに封印する。

外側fold Yはtrain `[2003-06-01, Y-1年07-01)`、validation `[Y-1年07-01, Y年01-01)`、
evalは上記後継範囲。元configとの一致を検証する。境界はLondon local date。
label終点が次rangeに入る行はpurge。過去feature historyだけは境界をまたいで参照できる。
fitには有効train252時刻以上、validation60時刻以上を必要とし、不足foldは明示的に開始不可。
評価末尾のh日labelが未成立でも予測・配分は毎decisionで行い、診断label不足と口座coverageを分ける。

## 3. 予測は2候補に固定（#41）

元8列の順は `log_return, volatility, sma20_ratio, mom24, xz_mom24, xr_mom24, carry_annual, xz_carry`。
各decisionの9pair算術平均を作り、恒等的に平均0の`xz_mom24, xr_mom24, xz_carry`を事前に除外する。
入力は残る5列を元順で使用。32-lag展開、pair ID、追加特徴量は0。
残る列がtrainで定数なら列名・件数を記録してfitを停止し、既定分散へ置き換えない。

価格座標は既存逆数JPY/COUNTERのまま。P_i(t)をこの価格とし、単位初期notionalを
各pairへ1/9ずつ配るbasketのh営業日price targetを
`y_h(t) = mean_i(P_i(t+h) / P_i(t) - 1)`とする（h=1,5）。
初期の単位数量をh日保持したpriceだけのreturnであり、日次simple returnの和ではない。
逆方向の実商品returnを符号だけ反転した値とも同一視しない。将来carry/売買費用はtargetに入れない。

train-onlyの平均・母標準偏差でXを標準化。切片は罰則なし。
目的は`mean((y - intercept - X beta)^2) + alpha * sum(beta^2)`、alpha=`0.01, 0.1, 1.0`。
mean loss単位なのでsum-loss実装なら罰則をtrain行数倍する。validation MSE最小、
差が絶対値1e-12以下なら大きいalpha。採用済み候補modelをそのまま封印し、train+val再fitは0。
ゼロ予測との差、時間方向の相関・方向一致率、予測分散・係数を診断し、rank ICで選ばない。
両horizonの全model/予測を#43へ渡す。予測改善とportfolio採否は別判断。

## 4. 4構成と配分契約（#42/#43）

| trial | horizon | 配分 |
|---|---:|---|
| A1-fixed | 1営業日 | 予測符号の等notional basket |
| A1-cost | 1営業日 | 同じ封印予測による費用対応 |
| A5-fixed | 5営業日 | 予測符号の等notional basket |
| A5-cost | 5営業日 | 同じ封印予測による費用対応 |

正のbasketは全pair正、負は全pair負。fixedは`u = sign(prediction)`、各target weight=`u/9`、
prediction=0はflat。サイズはgross1、共通basketの向きやサイズをevalで選ばない。
costは`u in [-1,1]`の同じbasket targetと**現在数量を維持する候補**を比較する。
数量hold後のweight driftは許すが、新しいrelative予測やpair選別は導入しない。
qty(u)は現在equityと価格から計算する。最適uによる部分変更、close、数量holdを許す。
fixedも毎日同じaccount engineを使い、quantity holdとtarget再送を区別して記録する。

目的関数は現在equity単位で
`predicted price + known-rate signed financing - entry trading cost - holding markup - 5 * w' Sigma_h w`。
全pairのprice予測は同じbasket予測値。現数量holdでは現在のmarked weightsを使う。
Sigma_hはtrainのgapをまたがないpair h日simple returnの標本共分散に
`0.9*S + 0.1*diag(S) + 1e-8*I`を適用する。縮小率・floor・risk係数を探索しない。
予測校正・volatility目標・追加size fitは0。実効riskが同じとは主張しない。

取引費用はcurrent marked quantityからqty(u)への実notionalで算定し、前回target差で代用しない。
financingは当日既知のCarryAnnualをhorizon内で一定と仮定し、calendarが定める実UTC経過日で
signed carryとmarkupを別計算する。将来実現rateは参照しない。終端決済costの予測上乗せは0。
h=5でも毎営業日に5営業日utilityを再計画する近似で、1日予測に割り直さない。
測定末端も同じhを使用し、最終markでは仮想決済しない。

solverは1次元の区分二次目的の決定論的解法に固定する。uの両端、notional差/financingの
符号が変わる全kink、各区間内の停留点、別候補のliteral quantity holdを全て比較する。
utility同値許容1e-12では実売買notional小→絶対gross小→u昇順。制約残差許容1e-10。
非有限値・解なし・残差超過は例外とincident。flat/holdへの暗黙fallbackは禁止。

全口座のhard capはgross5、pair絶対weight1、margin threshold0.2、初期100万円・flat。
新basket targetのgrossは1以下、既存対照のtargetは変更しない。
driftでhard capを超える場合だけ全数量を比例縮小し、費用とoverrideを保存する。
これはholdであっても明示risk取引であり、売買0と記録しない。旧対照への適用も同じ新契約で行う。

**測定ID `issue40-common-quantity-same-close-v1`を予約する。実装済みとは扱わない。**
#42で最小quantity adapterを実装し、#19/#21会計と#28のhistory/measurement分離を再利用する。
旧target API・旧報酬・旧ADRを変えず、新契約で全対照をreplayする。
主対照はcanonical mom24 top/bottom各2・各±0.8・既存tie、旧#15 ridge固定map、direct PPO ens3。
sourceは#20 identity/#15 models/provenance/#16 provenanceをhash固定、PPOはseed42/43/44のvalidation-best。
#30 constant/projectedは帰属の参考のみで、新たな勝ちやすい主対照として追加しない。

## 5. 費用、主指標、失敗

F0 spreadはpair順に`0.000015, 0.000025, 0.000030, 0.000030, 0.000035, 0.000035, 0.000040, 0.000060, 0.000060`。
commission=0、signed carry、markup=0.00002/day。F1はspreadのみ2倍、F2はmarkupのみ2倍。
合成stress・倍率探索・stressでの再学習0。各scenarioで独立口座を再推論する。
stress間は専用の費用感度とし、generic comparisonの同measurement検証を緩めない。

一次量はfold fの期間net log-return `L_f=log(E_last/E_initial)`の対照差。
#43の主比較は各hのcost−fixedの2本。全4構成の各3既存対照差、同配分内のhorizon差、
交互作用`(A5-cost−A5-fixed)−(A1-cost−A1-fixed)`も全て報告する。
期間の違いを隠さず、経過年`T_f=(last_mark-first_decision).total_seconds()/(365.25*86400)`、
年率log=`L_f/T_f`、年率simple=`expm1(L_f/T_f)`を別欄にする。
以下の数値採否は**期間logの17区間平均**に適用し、年率値へ読み替えない。
年次resetの平均を連続CAGRと呼ばず、部分区間を未観測期間へ外挿しない。

net/gross、Sharpe、MDD平均/最悪、price/carry/spread/commission/markup、gross/net exposure、
volatility、target turnover、実売買notional、数量hold/部分取引/override、coverageを併記する。
grossはsigned financingを含む既存定義。年率gross−netと初期equity基準cost ratioを分ける。
既存`forex-report`の比較検証・`research_statistics`の10,000 IID fold／3-fold circular moving-block、
seed16、95%区間、両era、全17 LOO、最大fold絶対寄与比を再利用する。
pair/日次/重複5日label/seedを市場標本に加算しない。全試行の区間は未補正の開発証拠。
全研究の独立trial総数は不明。reportを新測定へ接続する最小変更は#43/#48で検証する。

入力欠損・hash不一致・runtime/solver例外は`blocked_input`/`execution_error`。
margin/破産は`strategy_terminal`として全セルに残す。時刻・最終equity・費用・未測定範囲を保存する。
非正equityではlog=nullと理由を保存し、0・有限clip・失敗fold除外で統計を作らない。
正equityの途中terminalも通年外挿しない。全17区間の主統計が定義できなければ採否は未判定、
戦略terminalの候補は非採用。障害の残存とは区別し、成功セルだけのintersectionは禁止。

## 6. 結果前の数値採否・停止・tie-break

次の値は実口座の安全性や収益保証ではなく、この有限開発の選別規則である。
新候補は全F0/F1/F2でmargin/破産0、worst MDD≤25%、区間平均MDD≤15%、
区間平均の年率net log volatility≤20%を必要とする（母標準偏差ddof=0、既存metrics式）。
既存対照にも同じrisk欄を表示するが、不適合な対照を消したりサイズを合わせたりしない。

| 判断 | 固定条件 | 不成立時 |
|---|---|---|
| 予測の検討余地 | hごとに区間平均MSEがゼロ予測より小さく、両eraでも小さい。相関は診断のみ | 両h不成立なら予測余地なし。horizon/feature/family追加0 |
| 単純な開発候補 | 上記risk、F0期間net log平均≥0.005、両eraと全LOOのnet平均>0。F1/F2も平均net≥0、各stress−F0平均≥−0.005 | 費用消滅／risk不成立を分ける。常時flatは0なので不合格、一時flatに下限を設けない |
| 配分の追加価値を支持 | costと同h fixedのF0平均log差≥0.005、両era・全LOO差>0、IID/MB両95%下限>0、F1/F2の差≥0。cost自身も開発候補条件を満たす | CIをまたげば未証明。fixed維持可。gross低下より費用減が大きいnet改善は可 |
| 中期RLへ進む | 最大1hが開発候補を持ち、建玉保持・変更後の将来費用に未解決の逐次仮説を特定できる | 単純配分を維持可。CIが0をまたぐだけで探索着手を一律禁止しない |
| RL追加価値を開発上支持 | 合議−allocatorと合議−greedyの双方に配分追加価値と同じ差/risk/era/LOO/stress条件 | 単純配分以下は非採用、点推定のみ良い場合は未証明。別architectureに移らない |
| 停止/再定義 | 予測余地なし、費用で消滅、risk不成立、測定不能を区別して記録 | 後続#46〜#48はnot_planned。変更するなら情報源・期間・対象の一軸を別campaignに登録 |

「最大fold寄与」は`max(abs(d_f))/sum(abs(d_f))`、分母0はundefinedと理由を残す。
追加価値の支持には≤0.50も要求する。最大foldを削って合格を作らず全LOOを報告する。
候補が複数なら適格な4セルのF0平均期間net log降順→worst MDD昇順→
平均actual traded notional/initial equity昇順→h=1優先→fixed優先。
数値差のtie許容は1e-12。最大1hとそのセルを選び、同hの両配分結果は保持する。
進行の未解決仮説は「同じ予測/4行動で、myopic utilityを超えて再計画後の費用を減らせるか」に限定する。
選定/停止の理由、全条件のpass/fail、CI、未識別事項を閲覧後の台帳へ追記する。
canonical default、#15 learnability、#16 portfolio分類を自動変更せず、独立収益性は#50で別途設計する。

## 7. 有限予算と因果的cross-fit（#46）

| 作業 | 通常上限（失敗attemptを含めた追加枠は下記） |
|---|---:|
| #41 ridge | 2h×3alpha×17fold=102 fit。採用model34、再fit0 |
| #43全評価 | 4新構成×17×3=204口座＋3凍結対照×17×3=153再評価、合計357 |
| #46 cross-fit | 187更新×3alpha=561 fit（選定1hのみ）。採用model187、再fit0 |
| #48 PPO | 1構成×17×3seed=51学習、13,369,344 train step、408 validation口座 |
| #48評価 | 3seed＋合議＋allocator＋greedy=6方策×17×3=306、canonical/旧PPO参考102、合計408 |
| fixture/smoke | #41/#46各6 ridge fit=12、#47 PPO1学習4096step。#42合成口座16、#47合成口座12まで |
| #45 historical proxy | 既存3方策×17×同close/proxyの2条件×F0/F1/F2=306口座、fit0 |
| #44/#45実source | read-only1系統、最大30暦日か20decisionの早い方。quote比較は3方策×2条件=6口座、fit0 |
| #49連続口座 | 最終候補1＋canonical＋旧PPOの3方策×3scenario=9口座、再学習0 |
| #50確認設計 | protocol1、実観測/学習/発注0、自動開始なし |

fixtureの市場評価0、smoke seed=42、予測器の再fitや設定選びには使用しない。
solver・標準化・共分散の生成回数もmodelごとに記録し、追加の校正fitは0とする。
新quantity契約を採るため#43/#48の対照は**全数再評価を予算化し、旧metrics再利用0**。
同じ市場を再評価しても新しい市場標本ではない。未使用の予算は他作業へ移さない。
追加の障害再試行は同じbytes/条件につき1回まで、campaign全体でridge fit12、PPO学習3、
評価口座24を上限とする（PPO再試行のvalidationもこの24口座に含む）。戦略失敗・低成績の再試行0。
fixture/smokeもこの上限以外の再実行枠はない。実行前にattemptを予約し、失敗分も消費として数える。
合計ridge fit上限=102+561+12 fixture+12 retry=687、PPO学習上限=51+1 smoke+3 retry=55。
PPO retryは各262,144step以下、smoke4096stepを含む総train上限14,159,872step。

#46の更新は外側Yのtrain内各年1月1日（2008〜Y-1）、外側validation開始のY-1年7月1日、
eval用Y年1月1日。全てLondon午前0時、1月1日更新からeval終了まで再更新0。
foldごと3〜19更新、合計187。全日程をregistrationに列挙し、同cutoffの再利用をしても上限は増やさない。
各cutoff Cでinner train=`[2005-07-19,C-6か月)`、inner validation=`[C-6か月,C)`。
既知labelだけで3alphaをfit/選択し、終点≥境界をpurge、標準化・共分散もinner trainだけ。
最低252/60有効時刻を満たさなければその範囲を欠落と記録し、後年modelやゼロで埋めない。
最初のRL forecastは2008-01-01以降の有効decision。各行に全parameterのas-of cutoffを記録する。
これはcacheの歴史的公開時刻を証明せず、固定cache上での過去情報制約である。
外側validationで選んだalphaのtrainへの遡及適用は禁止。#41とは別bundleとして封印し、
PPO/allocator/greedyは全て新bundleを共有する。#43とのmetrics差をRL効果としない。

## 8. 中期PPOの一構成（#47/#48）

CPU、seed42/43/44、1env、MLP actor/critic各[32,32]・tanh、PPOのみ。
learning_rate=0.0003一定、gamma=0.99、gae_lambda=0.95、clip_range=0.2、
batch_size=64、n_epochs=10、ent_coef=0、vf_coef=0.5、max_grad_norm=0.5、advantage normalization有効。
rollout=2048、requested/actual steps=262144（128 rollout）。32,768stepごと8 checkpointを
同じvalidation F0のnet log-returnで選ぶ。同値1e-12なら早いcheckpoint。lastへの切替0。
checkpoint評価はvalidation内の最長の有効連続ブロック1つをflatから歩く。
必要history63本と60decision以上、同長なら早い区間というcoverageだけの規則で、初回fit前に固定する。
6か月全体の範囲と除外時刻を併記し、#41の全有効validation行によるMSE選択とは区別する。
不足で選択区間が成立しなければ停止する。terminalのcheckpointは選択不適格として全件記録し、
全checkpoint不適格なら学習失敗。runtime障害を不適格成績に置き換えて残りから選択しない。
実装でstep丸めが変われば超過を記録し、上限を越えて実行しない。

状態は順にh日forecast、train由来basket volatility、9pair marked equity weights（上記pair順）、
保有営業日数、既知9pair carry、9pair spread、commission、markup。
forecast/volatilityはreturn、weightsはequity比、保有日数は252で除す、費用はF0/F1/F2の比率。
追加market特徴量・観測normalizerのeval更新0。入力の非有限値は拒否する。
q*は#42のfixed符号basket target、4行動はhold_quantity/close/move_half/move_full。
greedyは同じ4候補を#42の予測utilityで比較。重複行動はmaskせず、tie順はhold→close→half→full。
合議は同じ独立口座観測に対する3seedの行動確率平均のargmax、tieも同順とする。

報酬は費用込みnet equity log成長のみ。非正equity時はtraining信号−20でterminal、
評価logの捏造には使わない。margin terminalも区別し、risk overrideは全対照で同一。
episodeは固定cacheの有効連続ブロックからseed付き一様に開始時刻を選び、最大128decision、flat開始。
未来returnで開始を選ばない。time-limitはtruncationとbootstrap、gap/ブロック末端はterminalとし、
その時刻のmarkで終える（仮想決済0）。再entry費用、gap、truncationの意味をfixtureで検証する。
この方策は固定予測を消費するhybridであり、旧`residual:none`のpure RL改善とは別主張。

## 9. 後続Issue、並行性、入力と出力

Issueリンクは[リポジトリのIssue一覧](https://github.com/TsuyoshiFujii-0915/forex-trainer/issues)から番号で参照できる。

| Issue | 着手条件・並行性 | 入力 → 出力 |
|---|---|---|
| #39 → #29 → #30 | 後継範囲は引渡し済み、旧通年は不足を保持 | data/calendar → 費用基準線 → 帰属・共通方向仮説 |
| #40 | 本manifest登録まで | 上記seal/判断 → 一分岐・4構成・予算・採否・登録commit |
| #41 | #40確定後。fixtureは#42と並行可 | 入力/5列/2h/3alpha → 全34 modelと予測、fit/purge/status/provenance |
| #42 | #40/#41 interface。#41本学習と並行可 | 同一予測・marked holdings → fixed/cost/quantity adapter、会計fixture、measurement seal |
| #43 | #41/#42、同runtimeの基準線 | 4構成と357セル → 全結果・対応差・最大1hまたは停止。実装と結果commitを分離 |
| #44 | #31/#32利用、#29/#30/#43待ち不要 | source/時刻/費用仕様 → read-only raw/decision/quote記録・coverage。実収集とfixtureを区別 |
| #45 | quoteは#44、proxyは#39/#40で先行可 | 同closeと単一遅延条件 → 独立経路感度、source不足・実執行未証明の一覧 |
| #46 | #43でRL進行の場合のみ | 選定1h・187cutoff → 全as-of付きforecast bundle・561 fit台帳 |
| #47 | #42/#46、#43進行判断 | causal forecast/quantity → PPO/4行動greedy、fixtureと4096step smokeまで |
| #48 | #46/#47と登録済runtime | 51学習＋408評価 → 2主比較・risk/費用・合議bundleまたは単純方策維持 |
| #49 | #43の単純候補で先行可、RLなら#48後 | 選定bundle・切替規則 → 年次resetなし口座、切替cost、欠損/gap可否 |
| #50 | 候補と#44/#45/#49の成果 | 期間利用台帳・実執行制約 → 新候補の独立確認protocol。設計のみ、自動開始0 |

#45の主執行方式は#31の「公開・durable decision後の最初の適格quote」。
実quote未取得のhistorical枠には別ID `issue40-next-close-proxy-v1` を事前登録する。
proxyはdecision後の**次の期待bar close**で凍結注文数量を全量fillする仮定、約定まで旧数量を保有。
新closeで約定notional/費用/marginを算定する。最終markまでにfillできない注文はexpiredとして記録する。
未来価格で数量を選び直さず、実quoteが無いことを同closeへfallbackする理由にしない。
latency/depth/partial fillの現実性は未検証。実quoteとproxyは別表・別IDで集約する。

外部source・料金/認証・公開/quote/費用仕様・開始/終了UTC・保存先は#44開始前に確定する。
未指定の現在は実収集blocked、注文0。最大1,000 request、retryは各request1回までで総数に含む。
429ではRetry-Afterを尊重し、その日の取得を停止する。大規模収集・常駐延長はこの予算に含めない。
#49は既存部分区間を連結してCAGRを作らず、全対象期間のgap/切替会計を結果前に登録する。
必要な連続入力がなければcoverage不足として終了する。入力変更は別data identityと有限契約を要する。
実quote/連続口座/独立確認が未完でも、#41〜#43のhistorical開発を一律停止しない。

## 10. 登録、検証、保持する契約

本manifestを最初にcommitし、そのSHAをregistrationに記録する。後続作業はこの登録commitと
manifest SHA-256を親とし、実装済みCLI/config、両repoの実際のclean SHA、lock hash、
Python/主要依存version、CPU、全model/data/forecast/calendar hashを初回fit/評価前に封印する。
現在mainのSHAを将来runtimeとして流用しない。実行中のsource編集0、比較内のruntime一致必須。
旧modelのtraining lineageを現在値で補完しない。将来のhashは開始前に取得すべき出力であり空欄の既定値ではない。

台帳の数え方は[#20](protocols/issue20/period-and-trial-ledgers.md)を継承し、旧台帳へ上書きしない。
2予測trial、4配分trial、条件付き1PPO trialの7行を新台帳に登録する（独立市場trial数ではない）。
alpha候補fit、187 cross-fit更新、seed/checkpoint、fixture、失敗attempt、対照再評価を別カウンタで数える。
planned→実行→閲覧→終了を追記し、同じ条件の再試行でもattempt ID・消費予算・失敗原因を残す。
各行はscope/hypothesis/changed_variable/frozen_values/budget/folds/seeds/validation_rule/
registered_at/registration_commit/opened_at/status/attempts/artifacts_and_hashes/decision/unknownsを保持する。
開封後の閾値変更は本campaignの救済に使わず、別revisionとして理由と旧結果を保持する。

受入確認は入力sealのhash、17範囲と9部分年の一致、187更新と全予算の算術、リンク、
登録commitとmanifest bytesの一致を検査する。#40では学習・市場推論・外部収集を行わない。
元入力の再生成は[#39の固定コマンド](23-bounded-input-recovery.md#再生成)、保存済み親結果の検証は
`uv run forex-successor-cost-campaign verify --output docs/research/results/issue29-successor`と
`uv run forex-successor-attribution verify --output docs/research/results/issue30-successor`を使う。
新予測/配分/PPOの実行CLIは未実装であり、#41/#42/#47の成果物を開始前登録へ追記する。
後続ではfuture perturbation、label purge、literal hold、stress別独立口座、solver例外、
terminal、全セルcoverageとhashの振る舞いテストを実装前に作る。

旧[#27](17-bounded-development-protocol.md)の学習/HPO=0はその診断campaignで保持する。
#20のB1〜B4・未開始の確認計画、ADR-0017/0031/0036/0042〜0044、#15/#16分類を変更しない。
2009〜2025はdevelopment、2007〜2008/2026H1は確認使用済み、2026H2は未使用と証明されていない。
前向きdevelopment shadowを独立確認へ昇格しない。開発候補・canonical置換・予測改善・
配分/RL追加価値・独立収益性を、それぞれ別の判断として保存する。
