"""Opt-in same-close quantity adapter over the unchanged portfolio accounting."""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from forex_env.accounting import PortfolioAccount
from forex_env.config import TransactionCostsConfig

from .common_allocation import Costs, Decision, State, RESIDUAL, label, proportional_cap, vector
from .common_basket import PAIRS, advance, utc
from .common_basket_study import digest
from .full_period import PeriodPlan, Policy

MEASUREMENT_ID = 'issue40-common-quantity-same-close-v1'
INITIAL_EQUITY = 1_000_000.
MARGIN_EQUITY = INITIAL_EQUITY * .2


@dataclass(frozen=True)
class Mark:
    """One calendar mark and current known inputs; windows exclude future rows."""

    at: str
    prices: np.ndarray
    carry: np.ndarray
    window: np.ndarray

    def __post_init__(self) -> None:
        label(self.at, 'mark.at')
        for name in ('prices', 'carry'):
            object.__setattr__(self, name, vector(getattr(self, name), f'mark.{name} at {self.at}'))
        window = np.array(self.window, dtype=float, copy=True)
        if window.shape != (9, 32, 8) or not np.isfinite(window).all() or (self.prices <= 0).any():
            raise ValueError(f'mark at {self.at}: invalid price or causal window')
        window.setflags(write=False)
        object.__setattr__(self, 'window', window)


QuantityPolicy = Callable[[State, dict[str, np.ndarray], np.ndarray], Decision]


class QuantityExecutionError(ValueError):
    """Account failure with the last cash state and all completed transitions."""

    def __init__(self, message: str, evidence: dict[str, Any]) -> None:
        super().__init__(message)
        self.evidence = evidence


def instruction(state: State, quantities: np.ndarray, mode: str) -> Decision:
    """Bind a literal quantity instruction to its current account state.

    Args:
        state: Current account after drift risk checks.
        quantities: Requested research-coordinate quantities.
        mode: Explicit hold, close, partial, or legacy target semantics.

    Returns:
        Validated instruction without fabricated forecast utility.
    """
    if mode not in ('hold_quantity', 'close', 'partial', 'legacy_target'):
        raise ValueError(f'quantity instruction: unknown mode {mode}')
    q = vector(quantities, 'instruction.quantities')
    if mode == 'hold_quantity' and not np.array_equal(q, state.quantities):
        raise ValueError('literal hold must preserve quantities exactly')
    if mode == 'close' and np.any(q):
        raise ValueError('close must request zero quantities')
    return Decision(mode, q, None, state.identity, None, None, {'status': 'quantity_instruction'})


def legacy_policy(predict: Policy) -> QuantityPolicy:
    """Adapt existing canonical, ridge, or PPO target proposals unchanged.

    Args:
        predict: Existing direct-policy inference callable.

    Returns:
        A fresh-account quantity instruction provider with no training.
    """
    def decide(state: State, observation: dict[str, np.ndarray], window: np.ndarray) -> Decision:
        scores, action = predict(observation, window)
        if action.shape != (9, 1) or not np.isfinite(action).all() or not np.isfinite(scores).all():
            raise ValueError(f'legacy policy at {state.at}: invalid direct target/scores')
        weights = action[:, 0].astype(float)
        if np.abs(weights).max() > 1 + 1e-7:
            raise ValueError(f'legacy policy at {state.at}: pair target exceeds 1')
        return instruction(state, weights * state.equity / state.prices, 'legacy_target')
    return decide


class QuantityAccount:
    """Maintain quantity holds while delegating all cash accounting to the env."""

    def __init__(self, first: Mark, costs: Costs) -> None:
        self.costs = costs
        self.mark = first
        self.account = PortfolioAccount(INITIAL_EQUITY, PAIRS, TransactionCostsConfig(
            commission_rate=costs.commission, overnight_rate=costs.markup,
            carry_mode='signed', spreads=dict(zip(PAIRS, costs.spreads, strict=True))))
        self.quantities = np.zeros(9)
        self.assets = np.zeros((9, 3), dtype=np.float32)
        self.pending: list[dict[str, Any]] = []
        self.equity_start = INITIAL_EQUITY
        self.prepared = False
        self.terminal_reason: str | None = None

    def state(self) -> State:
        """Return the current known input snapshot.

        Returns:
            Immutable current state, rejecting terminal equity.
        """
        return State(self.mark.at, PAIRS, self.account.equity_jpy, self.quantities,
                     self.mark.prices, self.mark.carry, self.costs)

    def trade(self, quantities: np.ndarray, reason: str) -> dict[str, Any]:
        """Execute exactly one quantity delta at the current close.

        Args:
            quantities: Explicit post-trade quantities.
            reason: Voluntary instruction or compulsory risk reduction.

        Returns:
            Actual notional and each charged cash cost.
        """
        before = self.quantities.copy()
        turnover = np.abs(quantities - before) * self.mark.prices
        if np.array_equal(quantities, before):
            spread, commission = 0., 0.
        else:
            spread, commission = self.account.rebalance(quantities * self.mark.prices)
        self.quantities = quantities.copy()
        if not np.isfinite([spread, commission, self.account.equity_jpy]).all():
            raise ValueError(f'account at {self.mark.at}: nonfinite execution')
        return {'reason': reason, 'quantities_before': before.tolist(), 'quantities_after': quantities.tolist(),
                'traded_notional': turnover.tolist(), 'spread': spread, 'commission': commission}

    def prepare(self) -> State:
        """Apply mandatory drift reduction before inference on any policy.

        Returns:
            State after proportional risk trading, including its own costs.
        """
        if self.prepared or self.terminal_reason is not None:
            raise ValueError(f'account at {self.mark.at}: duplicate prepare or terminal account')
        self.equity_start = self.account.equity_jpy
        current = self.state()
        capped = proportional_cap(current, current.quantities)
        if not np.array_equal(capped, current.quantities):
            self.pending.append(self.trade(capped, 'risk_drift'))
        self.prepared = True
        return self.state()

    def step(self, decision: Decision, next_mark: Mark) -> dict[str, Any]:
        """Execute a bound decision and mark held quantities through one interval.

        Args:
            decision: Instruction created from this prepared account state.
            next_mark: Next expected bar, not available to decision inference.

        Returns:
            Complete cash, quantity, utility, risk, and terminal trace.
        """
        if not self.prepared or self.terminal_reason is not None:
            raise ValueError(f'account at {self.mark.at}: prepare required or terminal account')
        current = self.state()
        if decision.state_sha256 != current.identity:
            raise ValueError(f'account at {self.mark.at}: stale or foreign decision')
        day = label(self.mark.at, 'account.at').tz_convert('Europe/London').date()
        if label(next_mark.at, 'account.next') != label(utc(advance(day, 1)), 'account.expected'):
            raise ValueError(f'account at {self.mark.at}: gap/nonconsecutive next mark')
        q = vector(decision.quantities, 'account.proposal')
        if decision.mode == 'hold_quantity' and not np.array_equal(q, current.quantities):
            raise ValueError('account: hold instruction changed quantities')
        capped = proportional_cap(current, q)
        override = not np.array_equal(capped, q)
        trades = [*self.pending, self.trade(capped, 'risk_proposal' if override else decision.mode)]
        if self.account.equity_jpy <= MARGIN_EQUITY:
            next_mark = self.mark
        execution_weights = capped * current.prices / self.account.equity_jpy
        residual = max(0., float(np.abs(execution_weights).sum()) - 5, float(np.abs(execution_weights).max()) - 1)
        if residual > RESIDUAL or not np.isfinite(residual):
            raise ValueError(f'account at {self.mark.at}: constraint residual {residual}')
        elapsed = float((label(next_mark.at, 'next') - label(current.at, 'current')).total_seconds() / 86400)
        relatives = next_mark.prices / current.prices
        pnl, markup = self.account.mark_to_market(relatives, elapsed)
        self.account.exposures_jpy = capped * next_mark.prices
        financing = self.account.accrue_financing(-current.carry, elapsed)
        equity = self.account.equity_jpy
        if not np.isfinite([pnl, markup, financing, equity]).all() or not np.isfinite(self.account.exposures_jpy).all():
            raise ValueError(f'account at {self.mark.at}: nonfinite mark accounting')
        if equity <= MARGIN_EQUITY:
            self.terminal_reason = 'nonpositive_equity' if equity <= 0 else 'margin_threshold'
        traded = sum(sum(t['traded_notional']) for t in trades)
        result = {'at': current.at, 'next_at': next_mark.at, 'prices': current.prices.tolist(),
                  'next_prices': next_mark.prices.tolist(), 'carry': current.carry.tolist(),
                  'equity_start': self.equity_start, 'equity_at_decision': current.equity,
                  'decision': decision.record(), 'trades': trades, 'quantities': capped.tolist(),
                  'marked_weights_before': current.weights.tolist(),
                  'execution_weights_after_cost': execution_weights.tolist(),
                  'marked_weights_after': (capped * next_mark.prices / equity).tolist() if equity > 0 else None,
                  'actual_traded_notional': traded, 'literal_hold': decision.mode == 'hold_quantity' and traded == 0,
                  'risk_override': override or bool(self.pending), 'constraint_residual': residual,
                  'elapsed_days': elapsed, 'price_pnl': pnl, 'financing': financing, 'markup': markup,
                  'equity_end': equity, 'assets_before': self.assets.tolist(), 'terminal_reason': self.terminal_reason}
        self.assets = np.stack([capped * current.prices / current.equity, np.ones(9),
                                capped * (next_mark.prices - current.prices) / current.equity], axis=1).astype(np.float32)
        self.mark = next_mark
        self.pending = []
        self.prepared = False
        return result


def replay(marks: tuple[Mark, ...], costs: Costs, policy: QuantityPolicy,
           runtime_sha256: str, policy_source: dict[str, Any]) -> dict[str, Any]:
    """Evaluate one independent account; history and final mark never trade.

    Args:
        marks: Complete measurement axis, excluding any observation-only history.
        costs: This account's explicitly registered scenario rates.
        policy: Inference using current information only.
        runtime_sha256: Shared sealed implementation/runtime identity.
        policy_source: Policy artifact identity, including fixture status.

    Returns:
        Complete trace or strategy terminal with uncovered decisions retained.
    """
    if len(marks) < 2:
        raise ValueError('quantity replay: at least one decision and final mark required')
    for first, second in zip(marks[:-1], marks[1:], strict=True):
        day = label(first.at, 'replay.axis').tz_convert('Europe/London').date()
        if label(second.at, 'replay.axis') != label(utc(advance(day, 1)), 'replay.expected'):
            raise ValueError(f'quantity replay: missing expected mark after {first.at}')
    account = QuantityAccount(marks[0], costs)
    trace = []
    try:
        for mark in marks[1:]:
            current = account.prepare()
            observation = {'market': account.mark.window.astype(np.float32), 'assets': account.assets.copy()}
            if current.equity <= MARGIN_EQUITY:
                decision = Decision('terminal_risk', current.quantities, None, current.identity, None, None,
                                    {'status': 'strategy_terminal_before_inference'})
            else:
                decision = policy(current, observation, account.mark.window.copy())
            trace.append(account.step(decision, mark))
            if account.terminal_reason is not None:
                break
    except Exception as exc:
        raise QuantityExecutionError(
            f'quantity replay at {account.mark.at}: {type(exc).__name__}: {exc}',
            {'trace': trace, 'at': account.mark.at, 'equity_repr': repr(account.account.equity_jpy),
             'quantities_repr': repr(account.quantities), 'pending_risk_trades': account.pending},
        ) from exc
    equity = account.account.equity_jpy
    return {'measurement_id': MEASUREMENT_ID, 'runtime_sha256': runtime_sha256,
            'policy_source': policy_source, 'costs': costs.record(), 'trace': trace,
            'status': 'strategy_terminal' if account.terminal_reason is not None else 'complete',
            'terminal_reason': account.terminal_reason, 'first_decision': marks[0].at,
            'planned_last_mark': marks[-1].at, 'actual_last_mark': account.mark.at,
            'remaining_decisions': len(marks) - 1 - len(trace), 'final_equity': equity,
            'net_log_return': float(np.log(equity / INITIAL_EQUITY)) if equity > 0 else None,
            'net_log_status': 'defined' if equity > 0 else 'undefined_nonpositive_equity',
            'final_quantities': account.quantities.tolist(), 'virtual_terminal_liquidation': False,
            'input_sha256': digest([{'at': m.at, 'prices': m.prices.tolist(), 'carry': m.carry.tolist(),
                                     'window': m.window.tolist()} for m in marks])}


def marks_from_period(plan: PeriodPlan) -> tuple[Mark, ...]:
    """Build causal policy windows from the existing history/measurement plan.

    Args:
        plan: A history-separated plan returned by prepare_period.

    Returns:
        Measurement marks only, with exactly 63 preceding history bars.
    """
    from forex_env.config import FeaturesConfig
    from forex_env.features import FeaturePipeline
    from .features import CROSS_FEATURE_REGISTRY, FEATURE_REGISTRY
    from .full_period import FEATURES, prepare_period

    validated = prepare_period(plan.market, plan.calendar, plan.measurement_start,
                               plan.measurement_end, plan.symbols, Path(plan.origin))
    if (validated.timestamps != plan.timestamps or validated.history_labels != plan.history_labels
            or plan.symbols != PAIRS or len(plan.history_labels) != 63):
        raise ValueError(f'quantity period: history/measurement mismatch: {plan.origin}')
    pipeline = FeaturePipeline(FeaturesConfig(32, False, FEATURES),
                               {name: FEATURE_REGISTRY[name] for name in FEATURES if name in FEATURE_REGISTRY},
                               custom_cross_features={name: CROSS_FEATURE_REGISTRY[name] for name in FEATURES if name in CROSS_FEATURE_REGISTRY})
    tensor = pipeline.compute(plan.market, PAIRS)
    return tuple(Mark(at, np.array([plan.market[pair, 'Close'].iloc[i] for pair in PAIRS]),
                      np.array([plan.market[pair, 'CarryAnnual'].iloc[i] for pair in PAIRS]),
                      tensor[:, i-31:i+1, :]) for i, at in enumerate(plan.timestamps, start=63))
