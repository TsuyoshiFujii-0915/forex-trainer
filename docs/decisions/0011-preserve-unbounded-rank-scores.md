# 0011. rank allocationのスコア順位を全有限float32域で保持する

## Status

accepted

## Context

ADR-0010では方策のpair scoreを`[-1, 1]`のBoxで公開し、wrapperでも同範囲へclipしてから
順位へ変換した。しかしPPOが出力する連続値は順位だけに意味があり、例えば
`5 > 2 > 1`を`1 = 1 = 1`へ潰すclipは順位情報を破壊する。同点後は設定pair順の
tie-breakに結果が依存するため、方策が学習した大小関係と異なるportfolioが選ばれる。

## Decision

rank allocationが有効な場合のagent-facing action spaceを、pairごとのfloat32で表現可能な
全有限域のBox、shape `(N, 1)`とする。Stable-Baselines3は有限な上下限を必須とするため、
境界には`±np.finfo(np.float32).max`を使う。これはPPOのfloat32出力すべてを包含するので、
学習・予測時のBox境界clipは恒等写像になる。wrapperはfiniteかつshapeが一致するscoreを
clipせず、生の値をstable sortする。同値のscoreだけはADR-0010どおり設定pair順で
tie-breakする。

この決定はADR-0010のscore範囲とclipに関する決定を置き換える。top-k配分、固定gross、
market neutrality、設定検証、ensembleでscore平均後に一度だけ変換する決定は維持する。

## Consequences

- 方策が出力した有限scoreの厳密な大小関係がportfolio選択まで保持される
- Stable-Baselines3が要求する有限Box境界を満たしつつ、PPOのfloat32 scoreを変更しない
- portfolio weight自体は従来どおり有限かつ固定grossであり、ForexEnvへ非有界値は渡らない
- NaNと無限scoreはwrapper境界で明示的な`ValueError`になる
