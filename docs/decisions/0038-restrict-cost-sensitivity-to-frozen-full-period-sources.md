# Title

0038. 通年費用感度比較を同一凍結sourceの独立口座に限定する

## Status

accepted

## Context

Issue #29はADR-0036/0037に従い、同じ3方策をF0/F1/F2で比較する。
費用変更はPPOのassetsと次のactionを変え、旧netへの費用加算では再現できない。
また歴史cacheの平日欠落を休日と推測すると通年測定の不足を隠してしまう。

## Decision

Issue #28評価器の上に今回限りの153セルcampaignを置く。方策ごと・scenarioごとに
口座をresetし、F1はspreadのみ2倍、F2はovernight markupのみ2倍にする。
汎用比較のmeasurement検証を維持し、専用感度比較だけが明示した費用差を許可する。
専用比較でもsource方策hash・runtime・市場値・全時刻・pair・環境・coverage一致を必須とする。
既存#19の会計照合関数、#21と同じ時価建玉基準のactual notional、#16のfold統計を再利用する。

calendarは結果閲覧前に、Europe/Londonの日付labelに対応する全平日という研究用期待集合を
登録する。休日と配信欠損を区別できるsourceが未保存なので休日除外は0とし、cacheから
欠けた日をcalendarから消さない。これはFX営業日や利用可能時刻の認定ではない。
欠落のあるfoldは全3方策×3scenarioをblocked_inputとして残す。他foldは独立して評価できるが、
153セルすべてが完走するまでera/全体集計、paired検定、方策の経済的判定は公表しない。
margin callはterminal trace付きのincomplete_margin_call、実行例外はexecution_errorと区別する。

## Consequences

- 独立した口座経路のstep/pair traceを後続の帰属分析へ渡せる。
- データ不足による部分実行は実験完了ではない。全体の不足範囲と必要sourceを明示できる。
- 休日の根拠や不足barを得た場合は新しいcalendar/data契約の事前登録が必要となる。
  元のsealed cacheを補修して同じhashと称することはできない。
- 旧#16は別表の記述的参考とし、新旧measurementのpaired検定や旧分類変更は行わない。
