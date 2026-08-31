# issue5_longf_tail_loss_regime_diagnostic

## Reproducibility checks

- The baseline-only generic report passed current artifact and provenance validation.
- Every action-mean ensemble was replayed deterministically and matched its sealed metrics.
- The rule was walked independently in the same resolved evaluation environments and timestamps aligned exactly.

## Stable relationships

- agent / decision_gross_exposure / next_cost_ratio
- agent / decision_turnover / next_cost_ratio
- rule / momentum_dispersion / next_net_return
- rule / momentum_dispersion / next_gross_return
- rule / momentum_dispersion / next_cost_ratio
- rule / momentum_dispersion / next_drawdown_change
- rule / decision_gross_exposure / next_net_return
- rule / decision_gross_exposure / next_gross_return
- rule / decision_gross_exposure / next_cost_ratio
- rule / decision_turnover / next_cost_ratio

## Legacy sanity comparison

Legacy Issue 1 evidence is reported descriptively only; it is not paired or bootstrapped with current evidence.

See `sanity_comparison.csv`, `fold_effects.csv`, and `provenance.json` for auditable details.
