# Title

0050. next-close proxyの注文数量を判断時に凍結し、実quoteと別契約にする

## Status

accepted

## Context

Issue #45 は旧3方策の執行感度を検証する。ADR-0040 の実商品quoteと ADR-0048 の
逆数研究数量は異なる測定であり、日足の次closeを観測quoteとして扱えない。
v3の短期gateは執行評価を許可していない。

## Decision

`issue40-next-close-proxy-v1` を追加し、既存same-close数量会計は変更しない。
判断時の独立口座から旧方策を再推論し、既存の比例capを判断時に適用した数量と
London平日の次の期待closeを注文へ保存する。約定時equityで数量を再計算しない。
旧数量を次closeまで保持し、price PnL、開始notionalのmarkup、終了notionalの
signed financingを既存PortfolioAccountで一度ずつ計上する。

次closeでは凍結数量を全量約定し、当該価格のdelta notionalにspread/commissionを課す。
fill後のhard cap超過は別の明示的risk縮小取引とし、注文自体の書換えをしない。
gapまたはfill費用でmarginに達した場合はstrategy terminalとし、gap terminalでは
未約定注文をcancelled_terminalとする。最終markで予定約定は許すが、新判断・仮想清算は
行わない。最終markより後が予定時刻の注文はexpired。欠損barは別closeへ飛ばさず例外とする。

PPOには自身の前区間price PnL/判断時equity、直近fill数量のfill notional/判断時equity、
leverage=1を渡す。初期assetsは0。待機中の資産変化を他の条件・方策から流用しない。

独立事前登録をレビュー・マージし、clean runtimeをsealした後にだけ、17区間×3費用×
3方策×2執行の306口座を一度実行できる。実quoteの上限6口座は入力未取得のため有効0。
自動retry・fallback・新候補の検証は行わない。全セル、失敗、最後の口座状態、trace、hashを保存する。

## Consequences

proxyの差は同じ逆数研究会計内の固定執行ルール感度で、実市場latencyの因果効果ではない。
future fillでのrisk縮小や状態に応じた再推論も差に含む。実quoteではprovider、商品、lot、
financing、公開・受信時刻、depthなどの差も残るため複合感度として別登録する。
実quote不足や新候補の追加登録要否は#43/#48/#50へ引き継ぎ、Issue #45は市場評価・
実quote検証が未完の間は完了扱いにしない。
