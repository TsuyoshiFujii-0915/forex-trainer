# Title

0037. 履歴を分離したhistorical通年評価を採用する

## Status

accepted

## Context

Issue #28でADR-0035案とIssue #22のfull-period仕様を実装へ移す。
ADR-0036の`issue27-development-v1`に従う開発測定であり、旧確認契約やsealed成果物は保持する。

## Decision

`full-period-history-v1`を`development_historical`専用に採用する。
UTCへ変換した明示的なtimezone付き境界`[S,E)`と、hashで固定したcalendarを必須にする。
calendarにはbar label、仮定session open/close、実際のavailable_at（不明ならnull）、
timezone、休日、写像規則とversionを記録する。closeはhistorical同close執行の仮定時刻であり、
available_atや当時のvintageの証明ではない。既知の公開遅延がcloseを超える入力は拒否する。

予定first decisionの直前63本と、S以上E未満の全対象barだけを独立cacheへ切り出す。
現行の8特徴量・window32・volatility32・normalize=false・k=1に限定する。
calendarとの欠落・余分なbar・pair差は例外とし、intersectionや日数による履歴代用をしない。
既存ForexEnvのfeature warmup32とreset index31がraw index63に一致するため、env API変更は不要。
同close約定・報酬・cost会計はenv SHA `6024b91c0f3592611849bc231922ab60e6090aed`を利用する。

各fold・各方策で100万円・exposure/assetsゼロから開始し、履歴ではstepを呼ばない。
全transitionは`S <= decision < target < E`、最終barはmarkのみで仮想清算しない。
年率化はfirst decisionから実際のlast markまでのUTC経過秒/365.25日を既存compute_metricsへ渡す。
予定範囲、初期flat待機、予定/実際のlast mark、末尾gapは別記する。
margin callはterminal traceを保存したincomplete結果とし、比較を停止する。データ例外と混同しない。

ridgeは封印済みfloat64特徴量window・standardizer・alphaを保持し、canonicalは既存float32
mom24 reversal、PPOはseed42/43/44の同一口座observationに対するaction平均を使用する。
新decisionを凍結modelから推論し、CSV継ぎ足し・再学習・latest選択をしない。
config、raw/clean/carry lineage、親seal/model、calendar、実効時刻、両repo SHA、依存とCPUを封印する。
出力は新規directoryのみとし、比較は同一measurement ID・実行条件・fold内時刻/pairを要求する。
legacyモードは既存評価関数を新規出力先へ実行し、sourceのbytesを変更しない。

## Consequences

- trainerのみの変更で観測履歴と損益期間を分離でき、関連env PRは不要。
- 過去の配信時刻や改訂vintageを捏造せず、仮定を明示した開発研究として実行する。
- margin call foldを除いた集計や、fold算術平均を連続運用CAGRとする表現は許可しない。
- 17fold×3費用条件の実験と研究結果コミットは#29へ渡す。fixture/smokeは市場実験完了ではない。
- ADR-0035と旧仕様案は履歴として保持し、本ADRが採用仕様を定める。旧accepted ADRは変更しない。
