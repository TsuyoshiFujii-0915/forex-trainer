# 次期研究の採否・停止と前向き確認契約（Issue #20）

契約ID: `issue20-forward-v1`。登録日: 2026-09-09。状態: **設計登録済み、前向き収集は開始条件未達**。
本PRのmerge commitを登録revisionとし、開始時には本書・台帳・schema・運用契約のraw bytesを
まとめたmanifestのSHA-256を凍結する。これは運用の開始や新規学習を実行するPRではない。

## 現在の判断と成果物

[Issue #19](https://github.com/TsuyoshiFujii-0915/forex-trainer/issues/19)は
[PR #23](https://github.com/TsuyoshiFujii-0915/forex-trainer/pull/23)で完了した。
同じ17 foldでの会計診断はgrossの安定したプラスを確立しなかった。
#15の`established learnable`と#16の`not successfully translated to portfolio alpha`を維持する。
cost-limited、tradable、canonical置換へ昇格しない。#19の診断は本契約待ちを要しない。

- [出典付き期間利用・試行台帳](protocols/issue20/period-and-trial-ledgers.md)
- [前向き記録の開始・終了・開封、障害・閲覧・訂正手順](protocols/issue20/forward-operations.md)
- [JSON Schema](protocols/issue20/forward-record.schema.json)・[架空の記録例](protocols/issue20/forward-record.examples.jsonl)
- [固定artifact identity](protocols/issue20/artifact-identities.json)
- [ADR-0031](../decisions/0031-separate-development-and-prospective-evidence.md)・[ADR-0032](../decisions/0032-append-prospective-observations-without-overwrite.md)

2009〜2025はすべてdevelopment folds。2007〜2008、2026H1は過去確認使用済みであり、
新しい設計の未使用確認には使わない。現時点で未使用と確認できる過去期間はない。

## 方策の役割とidentity

下記のhashは[identity manifest](protocols/issue20/artifact-identities.json)に実bytesから保存した。
`runs/latest`や最新timestamp検索で代用しない。data/modelがローカルにない場合は取得元を特定して
同じhashを照合するまで実行しない。legacy longfの旧meta補完や再学習への暗黙切替は禁止。

| 方策 | 役割 | 固定source・定義 |
|---|---|---|
| current direct `longf ens3` | RL対照。点推定のnet +10.31%を独立収益性と解釈しない | #15 provenanceの`fold_sources[year].ppo`、17 ensemble manifestと51 member model hash。seed 42/43/44、validation-best、CPU。legacy +4.7%系列とは別identity |
| canonical reversal | 維持する単純ルール対照。RLの追加価値は必ずこの対照とも比較 | #16 sealed steps/configと共通evaluator。mom24 ascendingの先頭2を+0.8、末尾2を−0.8、同点も既存順、k=1。過去に選択されたbenchmarkであり無選択の独立証拠ではない |
| supervised fixed map | 予測学習とportfolio変換の診断対象 | #15 predictions/modelsと#16 source hash。stable ascending predicted scoreの末尾2を+0.8、先頭2を−0.8、canonical pair順で同点解消、gross 3.2、k=1 |

#15 provenance SHA-256:
`a398ee18a01de364e21deb1d5f0ac5a66c1c1e445e60dc21c7c713215cebfb61`。
#16 provenance SHA-256:
`d2db580d5f11ec7ca341af04549c3a09e7ddb6777c43562b1117a24ae8a57db3`。
共通historical data SHA-256:
`723db7c935dcc27c147007728358d098243eae96998eae146db4c7df02bd081e`。
上記は過去比較のidentityであり、将来到着するdataのhashではない。

## 測定契約と採否表

以下は**今後のdevelopment比較**に対する基準であり、#15/#16の分類式を事後変更しない。
2009〜2025の17 folds、era 2009〜2018 / 2019〜2025、9 JPY pairs、日足、signed carry、
既存のrange内warmup/window、decision interval、cost会計を固定する。
同一fold・実効decision/target時刻・pair順、data、評価Git/依存、device、model selectionを要求し、
ensemble同士はmember seed集合42/43/44も完全一致させる。不一致・欠損は例外で停止し、
intersection、seed除外、評価器の混在をしない。

主指標は各foldの `expm1(net累積log return / 実経過年数)` の算術平均。
paired差はcandidate minus comparatorをfold内で取り、その17-fold平均を用いる。
grossも同じ年率化でsigned carryを含む既存定義を使う。
`forex-report`および#16/#19の既存出力・統計関数を使用し、別の集計基盤を作らない。
10,000 IID fold bootstrapと3-fold circular moving-block、95%区間、seed16、
同じ再標本化indexによるpaired比較を固定する。seedや日次barを独立市場標本として扱わない。
LOOは全17通りを報告し、除外後の平均を新たなprimaryにしない。

| 判定対象 | 必要条件（すべてAND） | 未達時 |
|---|---|---|
| stable positive gross | gross平均>0、両era平均>0、全LOO平均>0、IIDとmoving-block両下限>0、各foldのmean realized gross leverage≥1 | 未確立。costだけが障害と呼ばない |
| ネット正の開発証拠 | 上記gross条件 + net平均>0、両era平均>0、全LOO平均>0、両CI下限>0 | 平均のみ正なら記述的。独立確認済み収益性を主張しない |
| canonicalルール置換 | ネット正 + candidate−canonicalのnet/gross対応差の平均・両era・全LOO・両CI下限がすべて>0 + 下記risk条件 | canonical維持 |
| RLによる追加価値 | RLのネット正 + RL−canonical **および** RL−凍結教師あり固定mapのnet/gross対応差が上行の条件を満たす + 両対照へのrisk条件 | RL追加価値は未確立。RL自体のnet正やridgeのIC正で代用しない |
| risk条件 | candidateのmean MDDとworst MDDが各必要対照以下。fold mean realized gross leverage≥1かつcandidate/対照のfold等重みmean gross leverage≥0.8 | 縮退・tail悪化を伴う変更は非採用 |
| exposure / turnover / cost監査 | 全fold・eraの実効gross、net exposure、target-weight turnover（初期entryを含む）、cost JPY、initial-equity基準cost ratio、annual gross−net差を欠損なく報告 | 欠損は判定不能。turnover低下だけでは採用しない |
| cost削減を目的にする将来trial | 上記の必要な収益・risk条件 + 同一対照比でmean target turnoverとcost ratioがともに低下 | cost-aware成功と呼ばない |

MDDは正の損失割合、meanはfold等重み、worstはfold最大。risk条件の非劣性幅は0と固定する。
0.8 exposure比は小さなnet改善がほぼflat化に由来するのを排除する今後の保守的な運用基準であり、
過去の#16分類への追加条件ではない。Sharpe、勝ちfold数、最大絶対寄与foldと寄与率は副指標。
primaryが未達なら副指標への切替や+7%目標の再解釈で採用しない。
CIは設計探索の全多重性を補正していない。独立確認の有意性とは明確に区別する。

## 次の実験・停止・条件分岐

`G`を事前基準を満たすstable gross、`I`を未使用期間・事前統計設計に基づく独立証拠とする。
#19は会計を照合したが`G`を新たに確立せず、`I`でもない。実際に該当するのはB1である。

| 分岐 | 条件 | 次の作業・変更可能な一変数 | 探索予算と停止 |
|---|---|---|---|
| B0 | #19未完了、またはhash/時刻/会計不一致 | #19既存契約で原因を診断。研究treatment変更なし | 学習0、探索0。照合が通るまで新しい比較を止める |
| **B1（現在）** | #19完了、grossの不確実性が残る | **方策を変えず観測期間だけを将来へ移す**前向きreplicationの準備 | 事前登録1 campaign、3固定方策、候補探索0、再学習0。運用開始条件未達なら未開始で終了。bandit/gate/generic PPOへ自動進行しない |
| B2 | 将来の独立契約で`G AND I`、net未達でcost dragが障害と識別 | cost-aware問題を設計する別Issue。変更軸はtransaction-cost penalty係数1つに限定 | **本契約の実行予算0**。正確な係数候補、学習上限、seed/validation選択を実行前に別契約で固定するまで学習禁止。現時点で値を推測してbanditを作らない |
| B3 | `G AND I`、net支持あり、canonical paired優位なし | canonical維持。RL探索は開始しない | 新規学習0、同じ期間で再選択しない |
| B4 | 独立契約の収益・paired・risk条件をすべて満たす | その契約の対象範囲だけで確認報告 | 新規探索0。実口座導入・売買は別作業 |

B1の方策選択は成績の最大fold選びを避け、**時間順で最後の2025 foldの既存bundle**を機械的に使う。
ridgeは#15 `models.json`の2025 modelを凍結し、alpha再選択なし。
PPOは#15 provenanceが指す2025 ensembleのvalidation-best seed42/43/44を全員使い、
seed選抜・last/lateへの切替・再学習なし。canonicalも#16定義を維持する。
既存historical evaluation CSVは将来predictionの代用にできない。将来inputで同じmodelを推論する
記録用adapterの検証を別作業で完了することを開始条件とする。
方策が古いことはこの凍結replicationの制約であり、結果を見て更新しない。

1 campaign終了後に失敗した場合は停止し、期間延長、seed追加、top-k/weight/feature/horizon変更で
同じ確認を救済しない。次の仮説が必要なら別trial・別の将来期間・実行前の契約を必要とする。
過去開発結果または2027年1年間の記述的観測だけでは`G AND I`の成立とせず、B2以降へ進まない。
