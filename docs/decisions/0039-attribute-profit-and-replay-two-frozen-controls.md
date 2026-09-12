# Title

0039. PPO利益を実効weightで帰属し、固定した2対照を独立口座でreplayする

## Status

accepted

## Context

Issue #30は共通方向、横断相対配分、signed financing、費用の寄与を区別する。
ADR-0030/0034の会計とADR-0036の有限予算を継承する。#29 F0は現在8/17foldのみ完走している。

## Decision

#29 F0の親sealをpinし、全3方策の実効post-rebalance weightとprice simple returnから
common = sum(w) mean(r)、relative = sum(w (r-mean(r)))を計算する。
financingはsigned、spread/commission/overnightは別々の負の寄与にする。
開始equity単位で毎step照合し、同じlog1p(net)/net係数（net=0では1）を全成分に配賦する。
非正terminal equityのlogはnullと原因を保存し、terminalを通年推論へ入れない。

対照はtrain-derived constantとcommon-only projectedの2つに限定する。
constantは元configのtrain [start,end)内の既存観測warmup後の全decisionを独立口座でreplayし、
実効weightのdecision等重み平均を凍結する。train範囲外の履歴・validation/evalを含めない。
train terminalの場合は平均を採用せず、traceと未完了を保存する。
projectedはその独立口座のassetsで凍結ens3を再推論し、元提案へpair clip・gross capを
適用した実効weightを平均して9pairへ同じ値を渡す。float32化と既存env capを再適用する。
全方策は同じ上限、初期100万円・flat、k=1、同close、F0費用。投影に順位/tie処理はなく、
平均0は全flatとする。grossを元方策へ合わせる再拡大や事後サイズ調整はしない。

新runtimeで3元方策も再実行し、親F0の全trace/metric/時刻/環境を完全一致で照合する。
この明示的な再現検証に限りtrainer runtime差を記録して認める。対照のpaired比較は
新runtime内で既存require_comparableを通し、汎用measurement検証を緩めない。

17fold×5方策の全85セルを残し、全完走までera/全体/CI/LOO/主要源泉判定を保留する。
完走時はfold等重み、既存10,000 IID/3-fold circular moving-block・seed16を使う。
記述的な会計帰属を因果的優越性や単独成分CAGRと呼ばず、任意の境界値で主要源泉を自動分類しない。

## Consequences

- 元source hash、pair順、時刻、model/data/runtime、train replayを辿れる再現可能な診断となる。
- #29の欠損をcalendar削除やlegacy代用で埋めない。部分実行は最終通年解析の完了ではない。
- 新規学習、追加対照、regime再screen、旧#16判断変更、収益性認定は行わない。
