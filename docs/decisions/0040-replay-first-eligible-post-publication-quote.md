# Title

0040. 公開・判断保存後の最初の適格quoteで数量口座をreplayする

## Status

accepted

## Context

Issue #31はADR-0036の有限開発として同close依存を検証する。ADR-0037の
same-close会計は維持する。broker・口座・商品は未指定であり、日足labelは
公開時刻、次barは実行可能quote、逆数価格のreturnは実商品PnLの証拠にならない。

## Decision

独立したexecution ID `first-eligible-quote-v1`、measurement ID
`post-publication-quote-jpy-v1`を採用する。方式は次の1つに固定する。

- 全情報のavailable_atとretrieved_atがdecision生成以前、生成がdurable記録時刻以前。
  記録時刻より厳密に後のquoteを時刻順に走査する。quoteは9商品同時snapshotで、
  available_at <= retrieved_at、quote時刻からretrieved_atまで1秒以内、記録から60秒以内、
  明示した取引session内かつ休日外を要求する。calendarはtimezone/UTC境界を保存する。
  条件不適格quoteは理由付きで保存し、適格quote欠損はincidentとする。
- 事前固定した最初の適格basketでbid/askを使う。各商品の数量に足りるdepthを要求し、
  足りなければ停止する。後の有利なquoteへ進まない。部分約定・非同期basket・実発注は未対応。
  この全量basketは研究仮定であって、原子的な実市場約定の保証ではない。
- 商品境界はJPY quote通貨の線形cash-settled契約、baseはモデルpair JPY/XXXのXXX。
  model weight wに対して商品base数量q = -w E_fill / midをlot単位でゼロ方向に丸める。
  wは既存float32提案へpair clip ±1とgross cap5を適用する。最小取引数量未満の非ゼロ
  delta、数量不足、margin不足を拒否する。broker仕様を推測しない。
  PnL = q (P_end - P_start) JPY、quote→JPY換算係数は厳密に1。
  JPY base・非JPY quote・現物多通貨cash・netting/hedging・intrabar強制決済は未対応として拒否する。
- 前回mark=当該decisionの口座から、前数量のdecision→fill gap PnLとfinancingを計上する。
  そのfill equityで新数量を決め、deltaのbuyはask、sellはbidで約定する。
  spreadはmidとの差として1回のみ、commissionは実fill価格の売買notionalに課す。
  最終markまでの新数量のPnLとfinancingを分離し、最終清算はしない。
- financingは明示した有効区間内のbase単位・暦日当たりsigned JPY額をactual UTC秒/86400で
  積分する研究仮定。商品long/short別ratesと別markupを保存し、swapにmarkup内包なら
  別markupは0のみ認める。gapと保有区間の両方を対象とする。休日/DSTでも実秒を使う。
  broker rolloverの離散請求はこの線形モデルの対象外。
- canonical/ridge/凍結PPO ens3は同じ市場入力・開始flat100万円で独立replayする。
  PPO assetsは実現数量のmodel向きfill notional/E_fill、leverage1、前区間商品price PnL/
  decision equity。初期は全0。gapもprice PnLへ含む。旧inverse-return assetsと混同しない。
  旧modelを新座標へ適用する感度仮定として明記し、再学習しない。
- 入力・calendar・費用・モデル・コードのhashを保存し、decisionファイルを実際にfsyncした後に
  fillを処理する。シナリオのdecision時刻は仮想replay clockであり、保存wall-clockを別記する。
  過去に計算・保存できたことの証明とはしない。段階ごとに最後の妥当な口座状態を保持し、
  例外時にincidentと未完了を保存する。same-close/翌bar/ゼロactionへのfallbackは存在しない。

入力schemaは`src/forex_trainer/quote_replay.schema.json`。fixtureは
`tests/fixtures/issue31/scenario.json`。明示CLI scenarioは`fixture`と`shadow_quote`。
双方とも仮定付き研究replayで、後者のみ観測quote入力を要求する。`actual_fill`は拒否する。
fixtureのPPOはseed42/43/44の未学習実PPOで実装検証専用。shadowは既存封印済み
canonical/ridge/validation-best PPOのみ利用し、fixtureへ自動代用しない。

## Consequences

時刻・数量・資金調達を検証できるが、broker仕様・available_at・実quote・depthのcoverageが
揃うまで経済的影響は未検証。日足same-close結果との全差をlatency効果と呼ばない。
coverageをPnL閲覧前に固定し、17fold不足を縮小集計や長期外挿で埋めない。
既存#28/#29/#30とaccepted ADRの会計・統計・成果物は変更しない。
