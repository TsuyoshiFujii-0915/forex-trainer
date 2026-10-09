"""Frozen-quantity next-close proxy over the existing research account."""
from __future__ import annotations

from typing import Any

import numpy as np

from .common_allocation import Costs, Decision, State, label, proportional_cap, vector
from .common_basket import PAIRS, advance, utc
from .common_basket_study import digest
from .common_quantity import (
    INITIAL_EQUITY, MARGIN_EQUITY, Mark, QuantityAccount,
    QuantityExecutionError, QuantityPolicy,
)

MEASUREMENT_ID = 'issue40-next-close-proxy-v1'


def next_close(at: str) -> str:
    """Return the next expected close in the registered weekday calendar.

    Args:
        at: London midnight decision label.

    Returns:
        UTC label including weekend and DST transitions.
    """
    return utc(advance(label(at, 'proxy.decision').tz_convert('Europe/London').date(), 1))


def input_identity(marks: tuple[Mark, ...]) -> str:
    """Hash all registered market inputs, including causal windows.

    Args:
        marks: Complete ordered measurement axis.

    Returns:
        Shared input SHA-256 for both execution conditions.
    """
    return digest([{'at': m.at, 'prices': m.prices.tolist(), 'carry': m.carry.tolist(),
                    'window': m.window.tolist()} for m in marks])


def terminal(equity: float) -> str | None:
    """Classify the existing initial-equity margin rule.

    Args:
        equity: Current finite account equity.

    Returns:
        Terminal reason, or None for a live account.
    """
    if not np.isfinite(equity):
        raise ValueError('proxy accounting: nonfinite equity')
    if equity <= MARGIN_EQUITY:
        return 'nonpositive_equity' if equity <= 0 else 'margin_threshold'
    return None


def freeze_order(state: State, decision: Decision) -> dict[str, Any]:
    """Freeze an order using only the decision-time account information.

    Args:
        state: Prepared current account state.
        decision: Instruction bound to this exact state.

    Returns:
        Quantity, due time, original proposal and decision identity.
    """
    if decision.state_sha256 != state.identity:
        raise ValueError(f'proxy order at {state.at}: stale or foreign decision')
    q = vector(decision.quantities, 'proxy.order.quantities')
    if decision.mode == 'hold_quantity' and not np.array_equal(q, state.quantities):
        raise ValueError(f'proxy order at {state.at}: hold changed quantity')
    capped = proportional_cap(state, q)
    return {'decision_at': state.at, 'expected_fill_at': next_close(state.at),
            'quantities': capped.tolist(), 'decision': decision.record(),
            'decision_equity': state.equity, 'decision_prices': state.prices.tolist(),
            'decision_risk_override': not np.array_equal(capped, q)}


def fill_order(account: QuantityAccount, order: dict[str, Any], final_at: str) -> dict[str, Any]:
    """Fill the frozen quantity and separately record compulsory risk trades.

    Args:
        account: Account already marked to the expected fill instant.
        order: Frozen decision-time order.
        final_at: Registered final mark beyond which orders expire.

    Returns:
        Explicit filled, expired, or terminal-cancelled outcome with cash costs.
    """
    expected = next_close(order['decision_at'])
    if order['expected_fill_at'] != expected:
        raise ValueError('proxy order: expected fill schedule mismatch')
    q = vector(np.array(order['quantities']), 'proxy.fill.quantity')
    outcome = {**order, 'order_sha256': digest(order), 'fill_at': None,
               'equity_before_fill': account.account.equity_jpy, 'trades': []}
    if label(expected, 'proxy.expected') > label(final_at, 'proxy.final'):
        if account.mark.at != final_at:
            raise ValueError('proxy expiry requires the final mark')
        return {**outcome, 'status': 'expired', 'reason': 'after_final_mark'}
    if account.mark.at != expected:
        raise ValueError(f'proxy expected fill {expected}, got {account.mark.at}')
    if terminal(account.account.equity_jpy) is not None:
        return {**outcome, 'status': 'cancelled_terminal', 'reason': 'gap_margin'}
    trades = [account.trade(q, 'frozen_next_close')]
    if terminal(account.account.equity_jpy) is None:
        capped = proportional_cap(account.state(), account.quantities)
        if not np.array_equal(capped, account.quantities):
            trades.append(account.trade(capped, 'risk_fill'))
    for trade in trades:
        trade.update(at=account.mark.at, prices=account.mark.prices.tolist())
    return {**outcome, 'status': 'filled', 'reason': None, 'fill_at': account.mark.at,
            'trades': trades, 'equity_after_fill': account.account.equity_jpy}


def replay_next_close(marks: tuple[Mark, ...], costs: Costs, policy: QuantityPolicy,
                      runtime_sha256: str, policy_source: dict[str, Any]) -> dict[str, Any]:
    """Infer independently, hold old quantities through the gap, then fill.

    Args:
        marks: Consecutive registered marks; final mark never generates a decision.
        costs: Explicit spread, commission and separate markup.
        policy: Frozen policy using this account's current information only.
        runtime_sha256: Sealed execution runtime identity.
        policy_source: Frozen model lineage or explicit synthetic-test identity.

    Returns:
        All transitions, order outcomes, accounting and terminal coverage.

    Raises:
        QuantityExecutionError: Failure with completed trace and last account state.
    """
    if len(marks) < 2:
        raise ValueError('proxy requires a decision and final mark')
    for a, b in zip(marks[:-1], marks[1:], strict=True):
        if b.at != next_close(a.at):
            raise ValueError(f'proxy missing expected bar after {a.at}: {b.at}')
    account = QuantityAccount(marks[0], costs)
    trace: list[dict[str, Any]] = []
    order: dict[str, Any] | None = None
    try:
        for end in marks[1:]:
            order = None
            start = account.mark
            equity_start = account.account.equity_jpy
            state = account.prepare()
            drift = [{**t, 'at': start.at, 'prices': start.prices.tolist()} for t in account.pending]
            if terminal(state.equity) is not None:
                raise ValueError(f'proxy drift risk caused margin before decision at {start.at}')
            observation = {'market': start.window.astype(np.float32), 'assets': account.assets.copy()}
            decision = policy(state, observation, start.window.copy())
            order = freeze_order(state, decision)
            held = account.quantities.copy()
            elapsed = (label(end.at, 'proxy.fill') - label(start.at, 'proxy.decision')).total_seconds() / 86400
            pnl, markup = account.account.mark_to_market(end.prices / start.prices, elapsed)
            account.account.exposures_jpy = held * end.prices
            financing = account.account.accrue_financing(-start.carry, elapsed)
            if not np.isfinite([pnl, markup, financing]).all():
                raise ValueError(f'proxy nonfinite gap accounting at {start.at}')
            account.mark = end
            account.pending = []
            account.prepared = False
            filled = fill_order(account, order, marks[-1].at)
            trades = [*drift, *filled['trades']]
            equity = account.account.equity_jpy
            account.terminal_reason = terminal(equity)
            weights = account.quantities * end.prices / equity if equity > 0 else None
            residual = (max(0., float(np.abs(weights).sum()) - 5, float(np.abs(weights).max()) - 1)
                        if weights is not None else None)
            if account.terminal_reason is None and (residual is None or residual > 1e-10):
                raise ValueError(f'proxy fill constraint residual at {end.at}: {residual}')
            traded = sum(sum(t['traded_notional']) for t in trades)
            row = {'at': start.at, 'next_at': end.at, 'prices': start.prices.tolist(),
                   'next_prices': end.prices.tolist(), 'carry': start.carry.tolist(),
                   'equity_start': equity_start, 'equity_at_decision': state.equity,
                   'decision': decision.record(), 'orders': [filled], 'trades': trades,
                   'held_quantities': held.tolist(), 'quantities': account.quantities.tolist(),
                   'marked_weights_before': state.weights.tolist(),
                   'execution_weights_after_cost': weights.tolist() if weights is not None else None,
                   'marked_weights_after': weights.tolist() if weights is not None else None,
                   'actual_traded_notional': traded, 'literal_hold': decision.mode == 'hold_quantity' and traded == 0,
                   'risk_override': bool(drift) or order['decision_risk_override'] or len(filled['trades']) > 1,
                   'constraint_residual': residual, 'elapsed_days': elapsed,
                   'price_pnl': pnl, 'gap_price_pnl': pnl, 'financing': financing, 'markup': markup,
                   'equity_end': equity, 'assets_before': account.assets.tolist(),
                   'terminal_reason': account.terminal_reason}
            trace.append(row)
            account.assets = np.stack([account.quantities * end.prices / state.equity, np.ones(9),
                                       held * (end.prices - start.prices) / state.equity], axis=1).astype(np.float32)
            order = None
            if account.terminal_reason is not None:
                break
    except Exception as exc:
        raise QuantityExecutionError(
            f'next-close proxy at {account.mark.at}: {type(exc).__name__}: {exc}',
            {'trace': trace, 'at': account.mark.at, 'equity_repr': repr(account.account.equity_jpy),
             'quantities_repr': repr(account.quantities), 'pending_order': order,
             'pending_risk_trades': account.pending},
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
            'input_sha256': input_identity(marks)}


def audit_proxy(result: dict[str, Any], marks: tuple[Mark, ...], costs: Costs, origin: str) -> None:
    """Reconcile saved cash and frozen orders without model inference.

    Args:
        result: Complete or strategy-terminal saved proxy trace.
        marks: Original sealed input axis.
        costs: Independently registered cost scenario.
        origin: Artifact identity for explicit errors.
    """
    def equal(actual: Any, expected: Any, field: str) -> None:
        if (np.shape(actual) != np.shape(expected) or not np.isfinite(actual).all()
                or not np.allclose(actual, expected, rtol=1e-12, atol=1e-8)):
            raise ValueError(f'proxy accounting reconciliation: {origin}/{field}')

    if (result['measurement_id'] != MEASUREMENT_ID or result['input_sha256'] != input_identity(marks)
            or result['costs'] != costs.record() or not 0 < len(result['trace']) < len(marks)):
        raise ValueError(f'proxy input/coverage mismatch: {origin}')
    equity, q = INITIAL_EQUITY, np.zeros(9)
    assets = np.zeros((9, 3), dtype=np.float32)
    reason: str | None = None
    for i, row in enumerate(result['trace']):
        a, b = marks[i:i + 2]
        if reason is not None or row['at'] != a.at or row['next_at'] != b.at or b.at != next_close(a.at):
            raise ValueError(f'proxy coverage/terminal mismatch: {origin}/{i}')
        equal(row['equity_start'], equity, 'equity_start')
        equal(row['prices'], a.prices, 'prices')
        equal(row['next_prices'], b.prices, 'next_prices')
        equal(row['carry'], a.carry, 'carry')
        equal(row['assets_before'], assets, 'assets')
        traded = 0.
        for trade in row['trades']:
            if trade['at'] == a.at:
                if trade['reason'] != 'risk_drift':
                    raise ValueError(f'proxy unexpected decision-time trade: {origin}/{i}')
                equal(trade['quantities_before'], q, 'drift.before')
                target = proportional_cap(State(a.at, PAIRS, equity, q, a.prices, a.carry, costs), q)
                equal(trade['quantities_after'], target, 'drift.after')
                notional = np.abs(target - q) * a.prices
                equal(trade['traded_notional'], notional, 'drift.notional')
                equal(trade['spread'], notional @ costs.spreads, 'drift.spread')
                equal(trade['commission'], notional.sum() * costs.commission, 'drift.commission')
                equity -= trade['spread'] + trade['commission']
                traded += float(notional.sum())
                q = target
        state = State(a.at, PAIRS, equity, q, a.prices, a.carry, costs)
        equal(row['equity_at_decision'], equity, 'decision.equity')
        equal(row['marked_weights_before'], state.weights, 'decision.weights')
        equal(row['held_quantities'], q, 'held.quantity')
        saved = row['decision']
        decision = Decision(saved['mode'], np.array(saved['quantities']), saved['u'], saved['state_sha256'],
                            saved['forecast_identity'], saved['utility'], saved['solver'])
        frozen = freeze_order(state, decision)
        if len(row['orders']) != 1:
            raise ValueError(f'proxy order count mismatch: {origin}/{i}')
        order = row['orders'][0]
        if any(order[k] != v for k, v in frozen.items()) or order['order_sha256'] != digest(frozen):
            raise ValueError(f'proxy frozen order mismatch: {origin}/{i}')
        days = (label(b.at, origin) - label(a.at, origin)).total_seconds() / 86400
        pnl = float(q @ (b.prices - a.prices))
        financing = float((q * b.prices) @ -a.carry) * days / 365
        markup = float(np.abs(q * a.prices).sum()) * costs.markup * days
        for field, value in [('elapsed_days', days), ('price_pnl', pnl), ('gap_price_pnl', pnl),
                             ('financing', financing), ('markup', markup)]:
            equal(row[field], value, field)
        equity += pnl + financing - markup
        equal(order['equity_before_fill'], equity, 'fill.equity')
        fill_trades = [t for t in row['trades'] if t['at'] == b.at]
        if order['trades'] != fill_trades or len(fill_trades) + sum(t['at'] == a.at for t in row['trades']) != len(row['trades']):
            raise ValueError(f'proxy trade times mismatch: {origin}/{i}')
        if terminal(equity) is not None:
            if order['status'] != 'cancelled_terminal' or fill_trades or order['fill_at'] is not None:
                raise ValueError(f'proxy terminal order mismatch: {origin}/{i}')
        else:
            if order['status'] != 'filled' or order['fill_at'] != b.at or not 1 <= len(fill_trades) <= 2:
                raise ValueError(f'proxy fill order mismatch: {origin}/{i}')
            for j, trade in enumerate(fill_trades):
                target = (np.array(frozen['quantities']) if j == 0
                          else proportional_cap(State(b.at, PAIRS, equity, q, b.prices, b.carry, costs), q))
                if trade['reason'] != ('frozen_next_close' if j == 0 else 'risk_fill'):
                    raise ValueError(f'proxy fill reason mismatch: {origin}/{i}')
                equal(trade['quantities_before'], q, 'fill.before')
                equal(trade['quantities_after'], target, 'fill.after')
                equal(trade['prices'], b.prices, 'fill.prices')
                notional = np.abs(target - q) * b.prices
                equal(trade['traded_notional'], notional, 'fill.notional')
                equal(trade['spread'], notional @ costs.spreads, 'fill.spread')
                equal(trade['commission'], notional.sum() * costs.commission, 'fill.commission')
                equity -= trade['spread'] + trade['commission']
                traded += float(notional.sum())
                q = target
            equal(order['equity_after_fill'], equity, 'fill.equity_after')
        equal(row['quantities'], q, 'quantity')
        equal(row['actual_traded_notional'], traded, 'turnover')
        equal(row['equity_end'], equity, 'equity_end')
        expected_risk = (any(t['reason'] == 'risk_drift' for t in row['trades'])
                         or frozen['decision_risk_override'] or len(fill_trades) > 1)
        if (row['literal_hold'] != (decision.mode == 'hold_quantity' and traded == 0)
                or row['risk_override'] != expected_risk):
            raise ValueError(f'proxy execution diagnostic mismatch: {origin}/{i}')
        reason = terminal(equity)
        if row['terminal_reason'] != reason:
            raise ValueError(f'proxy terminal classification mismatch: {origin}/{i}')
        if equity > 0:
            weights = q * b.prices / equity
            equal(row['execution_weights_after_cost'], weights, 'fill.weights')
            equal(row['marked_weights_after'], weights, 'mark.weights')
            residual = max(0., float(np.abs(weights).sum()) - 5, float(np.abs(weights).max()) - 1)
            equal(row['constraint_residual'], residual, 'constraint_residual')
            if reason is None and (np.abs(weights).sum() > 5 + 1e-10 or np.abs(weights).max() > 1 + 1e-10):
                raise ValueError(f'proxy fill risk constraint mismatch: {origin}/{i}')
        elif any(row[field] is not None for field in
                 ('execution_weights_after_cost', 'marked_weights_after', 'constraint_residual')):
            raise ValueError(f'proxy nonpositive equity diagnostic must be null: {origin}/{i}')
        assets = np.stack([q * b.prices / state.equity, np.ones(9),
                           np.array(row['held_quantities']) * (b.prices - a.prices) / state.equity], axis=1).astype(np.float32)
    remaining = len(marks) - 1 - len(result['trace'])
    if (result['status'] != ('complete' if reason is None else 'strategy_terminal')
            or result['terminal_reason'] != reason or result['remaining_decisions'] != remaining
            or (reason is None and remaining != 0) or result['first_decision'] != marks[0].at
            or result['planned_last_mark'] != marks[-1].at or result['actual_last_mark'] != b.at
            or result['virtual_terminal_liquidation'] is not False):
        raise ValueError(f'proxy final coverage mismatch: {origin}')
    equal(result['final_equity'], equity, 'final_equity')
    equal(result['final_quantities'], q, 'final_quantities')
    if result['net_log_status'] != ('defined' if equity > 0 else 'undefined_nonpositive_equity'):
        raise ValueError(f'proxy net log diagnostic mismatch: {origin}')
    if equity > 0:
        equal(result['net_log_return'], np.log(equity / INITIAL_EQUITY), 'net_log')
    elif result['net_log_return'] is not None:
        raise ValueError(f'proxy nonpositive equity log must be null: {origin}')
