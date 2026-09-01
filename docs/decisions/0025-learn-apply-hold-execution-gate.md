# 0025. direct policyに学習可能なapply/hold execution gateを追加する

## Status

accepted

## Context

current-provenanceのdirect `longf ens3`は主要RL baselineである。疎なrank allocationは
drawdownを改善した一方でout-of-sampleのgross/net alphaを失い、Issue #5のregime診断でも
agentの将来net returnを安定して説明するmarket変数や単純なpolicy stateは見つからなかった。
gross exposureとturnoverは将来costとは安定して関係したため、残るexecution仮説は、現在の
allocationから新しいdirect-weight proposalへ移る便益が取引costに見合うかをpolicy自身が
end-to-endで学習できるかに限定される。

この検証ではgateの効果を、低いexposure、異なるdirect-weight head、rank/residual action、
または別の評価実装の効果と混同してはならない。またIssue #5で再学習したcontrol artifactの
`config_snapshot.yaml`はgate key導入前のstrict schemaで封印されており、その内容を変更せず
現行評価器で再評価できる必要がある。

## Decision

direct policyのpair数を`N`とすると、gate有効時のagent-facing actionを`(N + 1, 1)`の有限な
`[-1, 1]` Boxとする。先頭`N`要素は既存と同じdirect-weight proposal、末尾1要素はgate signalで
あり、事前登録した閾値を次のように固定する。

- `gate < 0`: hold
- `gate >= 0`: apply

閾値はzeroだけを実装し、evaluation fold上で選択可能なthreshold parameterは設けない。
configは`run.apply_hold_gate: zero_threshold`でgateを有効化し、`none`で無効化する。

holdは、直前のstepで環境が返したglobal gross cap適用後のeffective target allocationを厳密に
再送する。初回decisionではreset時のflat allocationを再送する。proposalをzero方向へblendせず、
gross exposureを暗黙に縮小しない。この意味は、同じtarget allocationを再送して配分を保持する
ADR-0004と一致する。absolute JPY exposureを固定する別のno-rebalance会計には変更しない。
gate decisionは`DecisionInterval`の外側でagent decisionごとに一度だけ行い、選ばれたtargetを
interval内の各barへ従来どおり再送する。

primary treatmentはdirect-weight PPO+MLPだけに限定する。gate有効時は
`allow_action_leverage: false`、`run.residual: none`、`run.rank_allocation: none`を必須とし、
違反をconfig読込時に拒否する。観測、feature、network、学習budget、model selection、device、
data、costおよび評価指標の意味は変更しない。

action-mean ensembleでは、各memberのdirect-weight proposalとgate signalを含むaction全体を先に
算術平均し、平均gateに対するapply/hold判定を一度だけ行う。memberごとにgate判定したportfolioを
後から平均してはならない。

各gated modelとensembleは、同じvalidation-best modelについて次の2 modeを別々に評価する。

- `learned`: zero-thresholdのapply/holdを通常どおり適用する
- `forced_apply`: gate signalを記録するが判定を無視し、毎decisionで平均後proposalを適用する

両modeは互いを上書きしない別artifactとして保存する。artifactはevaluation mode、source model、
member seeds、config、resolved eval env、data identity、device、evaluation-time Git/dependency、
metricsおよびgate decision traceのhashを封印する。forced applyは再学習した別modelではなく、
learned gateと同じmodelのdirect-weight headを使う。これによりgated minus forced-applyの対応差を
gate behavior自体のattribution controlとする。

gate decision traceには少なくともgate signal、learned decision、実際のevaluation modeでの
apply/hold、proposal、decision直前のeffective target、実際に適用したtarget、proposal distance、
holdで回避したtarget-weight turnover、決定的に算出できる即時取引costの支払額と回避額を記録する。
集約artifactはapply/hold比率、連続hold-run長、realized turnover、total cost ratio、mean gross
leverageを報告できるようにする。

gate keyを持たないIssue #5のrun sectionは、その既知のexact legacy key setに一致する場合だけ
明示的なlegacy direct schemaとして受理する。未知keyや部分的な新schemaをdirectへfallbackしない。
legacy artifact自体は変更しない。gateが無効なconfigではgate wrapperを構築せず、action space、
学習、validation、standard/ensemble evaluationの既存direct behaviorを維持する。

## Consequences

- policyは既存featureだけからproposalを適用するか現在allocationを保持するかを学習できる
- learned gateとsame-model forced applyの差により、direct-weight headの学習差とgate behaviorを
  分離できる
- holdはtarget-allocation契約を厳密に維持するが、市場変動後のabsolute JPY exposureを固定する
  no-trade命令ではないため、既存環境と同様に配分維持の小さなrebalance costが生じ得る
- action dimensionが一つ増えるためgated modelは既存direct modelと互換ではなく、primary comparison
  ではこの不可避な差だけをtraining treatmentとして扱う
- gate無効時にはwrapperもaction dimensionも追加されず、既存policyの数値挙動を変更しない
- Issue #5 control modelは再学習せず再利用できるが、ADR-0017に従いcandidateと同じmeasurement
  contractでcontrol ensembleを再評価してからpaired reportへ入れる必要がある
- evaluation modeとtraceを封印するartifact契約の追加実装と、17 folds × seeds 42/43/44のgated
  trainingおよびlearned/forced-apply評価が必要になる
