# Title

0041. 開発用shadowを原子的な追記batchと保存済みdecisionから再開する

## Status

accepted

## Context

Issue #32はADR-0032/0036の開発用記録adapterであり、#20の2027年確認を開始しない。
3方策の保存前適用、途中JSONL、再起動時の再推論・二重適用を防ぐ必要がある。

## Decision

local POSIX filesystem専用のsingle-writer lockと、連番のimmutable JSONL batchを使う。
各batchは同じfilesystem内の一時ファイルをfsync後、上書きしないhard linkで公開し、
directoryをfsyncする。3 decisionおよび3 outcomeは各々1 batchで公開する。
全行はv2 schema、連番、event ID、改行を除く前行のraw bytes hashで連結する。
既存v1 schema/成果物は変更しない。raw、登録、保存完了barrier、trial ledgerをv2へ追加する。

口座の正本はoutcomeのみ。全decisionのfsync完了barrierより後のquoteをADR-0040の共通会計へ渡し、
3口座のoutcome batchがdurableになって初めて適用済みとする。途中の計算は口座のcommitではない。
再開は保存済みdecisionを利用し、過去decisionを再推論しない。checkpointはheadに対応する
検証可能な派生物で、正本の代用にしない。公開前の途中ファイルは保存して明示recoverで障害を追記する。
公開済みbatchの破損は切り詰めず停止する。記録失敗は別directoryへ記録し、通常cycleは停止する。

fixtureは仮想scheduleと保存wall-clockを区別する。observed_fileはwall-clockで登録・deadlineを
検証し、当時のavailable_at/retrieved_atがある利用者指定read-only snapshotだけ受け付ける。
現在の実sourceは未指定。後日取得historical dataをobservedへ代用しない。
方策は#20の2025 foldとseed42/43/44をpinする。portable fixtureでは明示的なsynthetic modelを
使用できるが、observed_fileでは禁止し、実bundle fixtureの検証と区別する。

## Consequences

追記・再開・訂正・hash/時刻/口座検証をlocalでテストできる。quote/financing不足では
保存済みraw/decisionを残しoutcomeを作らない。訂正は証拠の追記で口座を遡及変更しない。
損益閲覧・設計変更をtrial ledgerへ記録するが、OS経由の閲覧は技術的には防げない。
このデータは独立holdoutへ昇格しない。独立head保管、権限分離、バックアップ、実source接続、
長期常駐運用は保証しない。local管理者による全履歴改変をhash chainだけで防げるとは主張しない。
