# Title

0035. 観測用履歴と損益計測期間を分離する通年評価契約案

## Status

proposed（Issue #22の仕様案。評価器への採用・実装は別作業）

## Context

現行の17 foldは年内の先頭32行をfeature warmupに使い、32行の観測windowを作るため
最初のdecisionはraw index 63になる。file providerは終了日を含み、10 foldでは翌年1月1日が
最後のtargetである。同じdecision集合による既存3対照の比較は有効だが、指定した暦年の
取引可能な全期間を測った結果ではない。

## Decision

採用する場合、観測用data範囲と損益範囲を別の必須値として封印する。
新契約ID案は`full-period-history-v1`。UTCの半開区間`[measurement_start, measurement_end)`内に
含まれる事前登録済みdecision/targetだけを測り、最初のdecisionより前の因果的履歴を読み込む。
現行8特徴量・32行windowには直前63本の有効なraw barを要求し、暦日数で代用しない。
履歴に損益を発生させず、最初のdecisionで1,000,000 JPY・建玉ゼロから開始する。
休日・未到着・bar labelとavailable_atの差を区別し、欠損で開始を後ろへずらさない。
詳しい境界・終端・特徴量・受入条件は[仕様案](../research/protocols/issue22/full-period-spec.md)に従う。

この一決定は履歴と測定範囲の分離である。broker費用モデルや約定タイミングの採用は含めない。
それらを変更する場合は別決定・別measurement identityとする。
旧期間契約と混在させず、凍結教師あり固定map・canonical reversal・current direct longf ens3の
全3対照を同じ新評価器・期間・データで再評価する。過去score CSVにないdecisionは凍結modelから
生成し、新artifactとして保存する。再学習や旧結果の上書きで代用しない。

## Consequences

- 年初の観測用履歴を損益に混ぜず、指定期間の最初の取引可能decisionから測れる。
- 観測範囲の変更でも特徴量とactionが変わり得るため、単なる表示修正ではない。
- #20の`issue20-forward-v1`は年内warmupのまま有効であり、本提案だけで置換されない。
  採用時には新しい前向き契約を開封前に登録し、既存accepted ADRを改変しない。
- 日足labelだけでは実際の公開・約定時刻を証明できない。通年評価の実装完了と
  brokerでの実行可能性の確認は別の受入条件となる。
