# 0010. 共通BASEに基づくFRED金利差・PPPファクター

## Status

accepted

## Context

ADR-0008 は `JPY/COUNTER` のみを扱う環境を前提として、`CarryAnnual` を
`rate_COUNTER - rate_JPY` と定義した。その後、forex-env ADR-0010 により、全ペアが
同じ第一レグを共有する制約のもとで任意の `BASE/COUNTER` を扱えるようになり、
forex-env ADR-0012 は `CarryAnnual` を `rate_COUNTER - rate_BASE` へ一般化した。

trainer は短期金利に加えて、10年国債金利差と相対PPPもFRED系列から生成する。これらを
JPY固定のまま非JPY環境へ渡すと、観測と資金調達損益が実際の二通貨間の差を表さなくなる。

## Decision

FREDの金利・CPI系列は3文字の通貨コードをキーとして管理し、各
`BASE/COUNTER`シンボルについて次の契約でペア相対フィールドを生成する。

- `CarryAnnual = rate_COUNTER - rate_BASE`
- `TermCarryAnnual = rate10y_COUNTER - rate10y_BASE`
- `PppGap = log(Close(t) / Close(t0)) - (log(CPI_COUNTER(t) / CPI_COUNTER(t0)) - log(CPI_BASE(t) / CPI_BASE(t0)))`

短期金利の`CarryAnnual`はforex-env ADR-0012の資金調達契約に従う。金利・CPI系列の
forward-fill、必須の発表ラグ、キャッシュ時刻への整列、未対応通貨および非有限値の
明示的エラーというADR-0008の因果性・fail-fast契約は維持する。VIXと原油のような
グローバル系列はペア相対値ではないため、全シンボルへ同じ値を配布する。

本ADRはADR-0008をsupersedeする。

## Consequences

- JPYをBASEとする既存キャッシュでは、生成値はADR-0008から変わらない
- USDやEURなどをBASEとする環境でも、観測と資金調達損益が実際の二通貨間の差を表す
- 通貨レジストリに存在しないBASEまたはCOUNTERを含むキャッシュは明示的に拒否される
- キャッシュ生成時に選んだ金利系列、CPI系列、発表ラグは引き続き実験条件の一部になる
