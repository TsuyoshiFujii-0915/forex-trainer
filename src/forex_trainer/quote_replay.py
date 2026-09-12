"""First eligible post-publication quote replay, with JPY quantity accounting."""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np

from .artifact_provenance import sha256_file
from .full_period import Policy, json_hash, read_json, runtime_identity

EXECUTION_ID = 'first-eligible-quote-v1'
MEASUREMENT_ID = 'post-publication-quote-jpy-v1'
SYMBOLS = tuple(f'JPY/{base}' for base in ('USD', 'EUR', 'GBP', 'AUD', 'CHF', 'CAD', 'NZD', 'NOK', 'SEK'))
SCHEMA = Path(__file__).with_name('quote_replay.schema.json')


class ReplayError(ValueError):
    """A replay incident whose last valid state has been durably saved."""


def instant(value: str) -> datetime:
    """Parse a required aware instant.

    Args:
        value: ISO timestamp.

    Returns:
        UTC instant.
    """
    parsed = datetime.fromisoformat(value.replace('Z', '+00:00'))
    if parsed.tzinfo is None:
        raise ValueError(f'Timezone required: {value}')
    return parsed.astimezone(timezone.utc)


def write_json(path: Path, value: Any) -> None:
    """Create and durably record a finite JSON artifact without overwriting.

    Args:
        path: New artifact path.
        value: JSON payload.
    """
    encoded = json.dumps(value, indent=2, allow_nan=False) + '\n'
    with path.open('x', encoding='utf-8') as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def evidence(record: dict[str, Any], cutoff: datetime, origin: str) -> None:
    """Require known release and retrieval before use.

    Args:
        record: Timestamped input evidence.
        cutoff: Earliest permissible consumer instant.
        origin: Traceable input location.
    """
    if not instant(record['available_at']) <= instant(record['retrieved_at']) <= cutoff:
        raise ValueError(f'Information unavailable at use: {origin}')


def validate(raw: dict[str, Any]) -> None:
    """Validate the closed input schema and fixed instrument/cost boundary.

    Args:
        raw: Explicit scenario document.
    """
    try:
        from jsonschema import Draft202012Validator, FormatChecker, ValidationError
        import rfc3339_validator  # noqa: F401
    except ImportError as exc:
        raise RuntimeError('Quote replay requires declared dev dependencies: uv sync --group dev') from exc
    json_hash(raw)
    schema = read_json(SCHEMA)
    try:
        Draft202012Validator(schema, format_checker=FormatChecker()).validate(raw)
    except ValidationError as exc:
        location = '/'.join(str(part) for part in exc.absolute_path)
        raise ValueError(f'scenario/{location}: {exc.message}') from exc
    if tuple(item['model_pair'] for item in raw['instruments']) != SYMBOLS:
        raise ValueError('scenario/instruments: exact ordered nine model pairs required')
    for item in raw['instruments']:
        if item['model_pair'] != f"JPY/{item['base']}" or item['base'] == 'JPY':
            raise ValueError(f'Unsupported base/quote quantity mapping: {item}')
    if raw['costs']['swap_includes_markup'] and any(raw['costs']['markup_per_base_day']):
        raise ValueError('scenario/costs: swap already includes markup; duplicate debit rejected')
    if (raw['broker'] is None) != (raw['account'] is None):
        raise ValueError('scenario: broker and account must both be specified or unresolved')
    if raw['broker'] is not None and raw['product_specification'] is None:
        raise ValueError('scenario: named broker/account require dated official product specifications')
    ZoneInfo(raw['calendar']['timezone'])
    previous: datetime | None = None
    for session in raw['calendar']['sessions']:
        begin, end = instant(session['open']), instant(session['close'])
        if begin >= end or (previous is not None and begin < previous):
            raise ValueError('scenario/calendar: unordered or overlapping sessions')
        previous = end
    previous = None
    for funding in raw['funding']:
        begin, end = instant(funding['start']), instant(funding['end'])
        if begin >= end or (previous is not None and begin < previous):
            raise ValueError('scenario/funding: unordered or overlapping intervals')
        previous = end
    if raw['evidence_kind'] == 'shadow_quote':
        records = [raw['initial_mark']]
        for step in raw['steps']:
            records.extend([*step['inputs'], *step['quotes'], step['mark']])
        if any(record['quality'] != 'observed' for record in records):
            raise ValueError('scenario/evidence: shadow_quote requires observed market/release/quote evidence')


def choose_quote(step: dict[str, Any], calendar: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Select the first eligible timestamp without optimizing price or depth.

    Args:
        step: Decision schedule and ordered quote snapshots.
        calendar: Explicit timezone, sessions and holidays.

    Returns:
        First eligible quote and rejected candidates with reasons.
    """
    saved = instant(step['decision_recorded_at'])
    previous: datetime | None = None
    previous_receipt: datetime | None = None
    for quote in step['quotes']:
        at = instant(quote['timestamp'])
        if previous is not None and at <= previous:
            raise ValueError('scenario/quotes: timestamps must be strictly increasing')
        received = instant(quote['retrieved_at'])
        if previous_receipt is not None and received <= previous_receipt:
            raise ValueError('scenario/quotes: receipt order must be strictly increasing')
        previous, previous_receipt = at, received
        if any(bid > ask for bid, ask in zip(quote['bid'], quote['ask'], strict=True)):
            raise ValueError('scenario/quotes: crossed bid/ask')
        evidence(quote, instant(quote['retrieved_at']), 'scenario/quote')
        if not at <= instant(quote['available_at']):
            raise ValueError('scenario/quote: publication before quote timestamp')
    rejected: list[dict[str, Any]] = []
    for index, quote in enumerate(step['quotes']):
        at = instant(quote['timestamp'])
        age = (instant(quote['retrieved_at']) - at).total_seconds()
        delay = (instant(quote['retrieved_at']) - saved).total_seconds()
        local_date = at.astimezone(ZoneInfo(calendar['timezone'])).date().isoformat()
        received = instant(quote['retrieved_at'])
        receipt_date = received.astimezone(ZoneInfo(calendar['timezone'])).date().isoformat()
        in_session = any(instant(s['open']) <= at <= received < instant(s['close']) for s in calendar['sessions'])
        reasons = []
        if at <= saved:
            reasons.append('before_durable_decision')
        if age > 1:
            reasons.append('stale_quote')
        if delay > 60:
            reasons.append('deadline_exceeded')
        if not quote['tradable'] or not in_session or local_date in calendar['holidays'] or receipt_date in calendar['holidays']:
            reasons.append('market_closed_or_not_tradable')
        if reasons:
            rejected.append({'index': index, 'timestamp': quote['timestamp'], 'reasons': reasons})
        else:
            return quote, rejected
    raise ValueError(f'scenario/quotes: no eligible quote; rejected={rejected}')


def financing(raw: dict[str, Any], quantity: np.ndarray, start: datetime, end: datetime, known: datetime) -> tuple[float, float]:
    """Integrate explicit product-direction financing over actual elapsed time.

    Args:
        raw: Scenario funding schedules and costs.
        quantity: Signed product base units.
        start: Interval start.
        end: Interval end.
        known: Decision information cutoff.

    Returns:
        Signed financing credit and separate positive markup debit in JPY.
    """
    if end < start:
        raise ValueError('scenario/funding: negative elapsed interval')
    cursor, credit = start, 0.
    for row in raw['funding']:
        begin, stop = instant(row['start']), instant(row['end'])
        if stop <= cursor or begin >= end:
            continue
        if begin > cursor:
            raise ValueError(f'scenario/funding: missing interval at {cursor.isoformat()}')
        evidence(row, known, 'scenario/funding')
        until = min(end, stop)
        rates = np.where(quantity >= 0, row['long_jpy_per_base_day'], row['short_jpy_per_base_day'])
        credit += float(np.sum(np.abs(quantity) * rates)) * (until - cursor).total_seconds() / 86400
        cursor = until
        if cursor == end:
            break
    if cursor != end:
        raise ValueError(f'scenario/funding: missing coverage through {end.isoformat()}')
    markup = float(np.sum(np.abs(quantity) * raw['costs']['markup_per_base_day'])) * (end - start).total_seconds() / 86400
    if not np.isfinite([credit, markup]).all():
        raise ValueError('scenario/funding: non-finite financing or markup')
    return credit, markup


def initial_state() -> dict[str, Any]:
    """Create the registered flat account before any input is accepted.

    Returns:
        Initial JPY account with no assumed market timestamp.
    """
    return {'phase': 'initial', 'timestamp': None, 'equity_jpy': 1_000_000.,
            'quantity_base': [0.] * 9, 'mid': None, 'assets': np.zeros((9, 3)).tolist()}


@np.errstate(over='raise', invalid='raise', divide='raise')
def replay(raw: dict[str, Any], predict: Policy, output: Path) -> dict[str, Any]:
    """Replay one independent account and persist exceptions with its valid state.

    Args:
        raw: Explicit input snapshot; never modified.
        predict: Frozen inference using this account's assets.
        output: New account artifact directory.

    Returns:
        Completed or margin-terminal account, with trace and coverage.

    Raises:
        ReplayError: Invalid inputs or unsupported execution, after incident save.
    """
    output.mkdir(parents=True, exist_ok=False)
    state = initial_state()
    trace: list[dict[str, Any]] = []
    result: dict[str, Any] = {'measurement_id': MEASUREMENT_ID, 'execution_id': EXECUTION_ID,
                             'status': 'running', 'trace': trace, 'last_state': state}
    origin = 'scenario'
    try:
        validate(raw)
        first = raw['initial_mark']
        evidence(first, instant(first['timestamp']), 'scenario/initial_mark')
        if instant(first['available_at']) < instant(first['timestamp']):
            raise ValueError('scenario/initial_mark: price published before mark timestamp')
        state.update(timestamp=first['timestamp'], mid=first['mid'])
        for index, step in enumerate(raw['steps']):
            origin = f'scenario/steps/{index}'
            close = instant(step['session_close'])
            generated = instant(step['decision_generated_at'])
            saved = instant(step['decision_recorded_at'])
            if instant(state['timestamp']) != close or not close <= generated <= saved:
                raise ValueError('decision must follow observation and previous mark, then durable recording')
            if close.date().isoformat() != step['bar_label']:
                raise ValueError('bar_label must identify the UTC session-close date')
            for info in step['inputs']:
                evidence(info, generated, origin + '/inputs')
            windows = [info for info in step['inputs'] if info['name'] == 'market_and_carry_window']
            if len(windows) != 1 or windows[0]['payload_sha256'] != json_hash(step['window']):
                raise ValueError('market_and_carry_window payload hash mismatch')
            if instant(windows[0]['available_at']) < close:
                raise ValueError('complete window cannot be available before session close')
            window = np.array(step['window'], dtype=np.float64)
            assets = np.array(state['assets'], dtype=np.float32)
            scores, action = predict({'market': window.astype(np.float32), 'assets': assets.copy()}, window.copy())
            if scores.shape != (9,) or action.shape != (9, 1) or not np.isfinite(scores).all() or not np.isfinite(action).all():
                raise ValueError('policy must return finite nine-pair scores and direct action')
            weights = np.clip(action[:, 0].astype(np.float32), -1, 1).astype(float)
            gross = float(np.sum(np.abs(weights)))
            if gross > 5:
                weights *= 5 / gross
            decision_path = output / f'decision-{index:04d}.json'
            write_json(decision_path, {'clock_basis': 'virtual_replay_schedule',
                       'generated_at': step['decision_generated_at'], 'recorded_at': step['decision_recorded_at'],
                       'write_started_at_utc': datetime.now(timezone.utc).isoformat(),
                       'input_sha256': json_hash({'window': step['window'], 'inputs': step['inputs']}), 'assets': assets.tolist(),
                       'scores': scores.tolist(), 'action': action[:, 0].tolist(), 'weights': weights.tolist()})
            durable_wall_clock = datetime.now(timezone.utc).isoformat()
            quote, rejected = choose_quote(step, raw['calendar'])
            fill_time = instant(quote['retrieved_at'])
            mid = (np.array(quote['bid']) + np.array(quote['ask'])) / 2
            old_q = np.array(state['quantity_base'])
            equity_before = float(state['equity_jpy'])
            gap = old_q * (mid - state['mid'])
            gap_financing, gap_markup = financing(raw, old_q, close, fill_time, generated)
            fill_equity = equity_before + float(gap.sum()) + gap_financing - gap_markup
            if not np.isfinite(fill_equity):
                raise ValueError('non-finite fill equity')
            state.update(phase='pre_fill', timestamp=quote['retrieved_at'], equity_jpy=fill_equity, mid=mid.tolist())
            if fill_equity <= 200_000:
                result['status'] = 'terminal_margin'
                result['terminal'] = {'origin': origin, 'phase': 'pre_fill', 'gap_pnl_jpy': float(gap.sum()),
                                      'gap_financing_jpy': gap_financing, 'gap_markup_jpy': gap_markup}
                break
            lots = np.array([item['lot_size'] for item in raw['instruments']])
            quantity = np.trunc((-weights * fill_equity / mid) / lots) * lots
            delta = quantity - old_q
            minimum = np.array([item['minimum_trade'] for item in raw['instruments']])
            if np.any((np.abs(delta) > 0) & (np.abs(delta) < minimum)):
                raise ValueError('minimum trade size violated; no silent quantity substitution')
            if np.any(np.abs(delta) > quote['capacity_base']):
                raise ValueError('quote capacity insufficient; partial fill unsupported')
            fill_prices = np.where(delta >= 0, quote['ask'], quote['bid'])
            spread = float(np.sum(delta * (fill_prices - mid)))
            traded_notional = np.abs(delta) * fill_prices
            commission = float(np.sum(traded_notional)) * raw['costs']['commission_rate']
            post_fill = fill_equity - spread - commission
            margin = float(np.sum(np.abs(quantity * mid) * [item['margin_rate'] for item in raw['instruments']]))
            if not np.isfinite([post_fill, margin, spread, commission]).all():
                raise ValueError('non-finite fill costs or margin')
            if post_fill < margin or post_fill <= 200_000:
                raise ValueError('insufficient margin for full proposed basket')
            effective = -quantity * mid / fill_equity
            state.update(phase='fill', equity_jpy=post_fill, quantity_base=quantity.tolist())
            write_json(output / f'fill-{index:04d}.json', {'state': state, 'quote': quote,
                       'delta_base': delta.tolist(), 'fill_prices': fill_prices.tolist(), 'spread_jpy': spread,
                       'commission_jpy': commission, 'decision_sha256': sha256_file(decision_path),
                       'decision_fsync_completed_at_utc': durable_wall_clock})
            mark = step['mark']
            mark_time = instant(mark['timestamp'])
            if mark_time <= fill_time:
                raise ValueError('final mark must follow fill')
            evidence(mark, mark_time, origin + '/mark')
            if instant(mark['available_at']) < mark_time:
                raise ValueError('mark price cannot be published before mark timestamp')
            holding_financing, holding_markup = financing(raw, quantity, fill_time, mark_time, generated)
            holding = quantity * (np.array(mark['mid']) - mid)
            equity = post_fill + float(holding.sum()) + holding_financing - holding_markup
            if not np.isfinite(equity):
                raise ValueError('non-finite mark equity')
            price_pnl = gap + holding
            next_assets = np.stack([effective, np.ones(9), price_pnl / equity_before], axis=1).astype(np.float32)
            state.update(phase='mark', timestamp=mark['timestamp'], equity_jpy=equity, mid=mark['mid'], assets=next_assets.tolist())
            trace.append({'bar_label': step['bar_label'], 'decision_timestamp': step['decision_generated_at'],
                          'decision_recorded_at': step['decision_recorded_at'], 'fill_timestamp': quote['retrieved_at'], 'quote_timestamp': quote['timestamp'],
                          'mark_timestamp': mark['timestamp'], 'quote_sha256': json_hash(quote),
                          'mark_sha256': json_hash(mark), 'input_sha256': json_hash(step),
                          'decision_sha256': sha256_file(decision_path), 'assets_before': assets.tolist(),
                          'scores': scores.tolist(), 'action': action[:, 0].tolist(),
                          'effective_weights': effective.tolist(), 'quantity_base': quantity.tolist(),
                          'delta_base': delta.tolist(), 'fill_prices': fill_prices.tolist(), 'quote_to_jpy': [1.] * 9,
                          'fill_notional_jpy': (quantity * mid).tolist(), 'traded_notional_jpy': traded_notional.tolist(),
                          'equity_before': equity_before, 'fill_equity_jpy': fill_equity, 'equity_after_fill_jpy': post_fill,
                          'gap_pnl_jpy': float(gap.sum()), 'holding_pnl_jpy': float(holding.sum()),
                          'price_pnl_by_pair': price_pnl.tolist(), 'gap_financing_jpy': gap_financing,
                          'holding_financing_jpy': holding_financing, 'gap_markup_jpy': gap_markup,
                          'holding_markup_jpy': holding_markup, 'spread_jpy': spread, 'commission_jpy': commission,
                          'equity_jpy': equity, 'gap_seconds': (fill_time - close).total_seconds(),
                          'holding_seconds': (mark_time - fill_time).total_seconds(), 'rejected_quotes': rejected})
            if equity <= 200_000 or equity < float(np.sum(np.abs(quantity * np.array(mark['mid'])) * [item['margin_rate'] for item in raw['instruments']])):
                result['status'] = 'terminal_margin'
                result['terminal'] = {'origin': origin, 'phase': 'mark'}
                break
        else:
            result['status'] = 'complete'
        result['coverage'] = {'first_decision': raw['steps'][0]['session_close'],
                              'last_mark': state['timestamp'], 'planned_last_mark': raw['steps'][-1]['mark']['timestamp'],
                              'completed_steps': len(trace), 'planned_steps': len(raw['steps']),
                              'elapsed_seconds': (instant(state['timestamp']) - instant(first['timestamp'])).total_seconds()}
    except (ValueError, TypeError, KeyError, OSError, RuntimeError, ArithmeticError) as exc:
        result['status'] = 'incident'
        result['incident'] = {'origin': origin, 'type': type(exc).__name__, 'message': str(exc)}
        write_json(output / 'result.json', result)
        raise ReplayError(f'{origin}: {exc}; last valid account saved at {output}') from exc
    write_json(output / 'result.json', result)
    return result


def run_scenario(input_path: Path, scenario: str, source_path: Path | None, output: Path) -> dict[str, Any]:
    """Run all three policies on one explicit quote scenario and seal provenance.

    Args:
        input_path: Timestamped scenario JSON.
        scenario: Explicit fixture or shadow_quote evidence selection.
        source_path: Required frozen-source manifest for shadow; absent for fixture.
        output: New campaign directory.

    Returns:
        Panel status and paths. Incidents prohibit paired interpretation.
    """
    from .quote_replay_policies import fixture_policies, frozen_policies

    if output.exists():
        raise ValueError(f'Output exists: {output}')
    try:
        raw = read_json(input_path)
        validate(raw)
    except (ValueError, TypeError, KeyError) as exc:
        output.mkdir(parents=True)
        write_json(output / 'incident.json', {'origin': str(input_path), 'message': str(exc),
                   'last_state': initial_state(), 'status': 'preflight_incident'})
        raise ValueError(f'{input_path}: {exc}') from exc
    if scenario not in {'fixture', 'shadow_quote'} or scenario != raw['evidence_kind']:
        raise ValueError('Explicit scenario must match input evidence_kind')
    if scenario == 'fixture':
        if source_path is not None:
            raise ValueError('fixture scenario does not accept frozen sources')
        policies, identity = fixture_policies()
    else:
        if source_path is None:
            raise ValueError('shadow_quote scenario requires a pinned source manifest')
        policies, identity = frozen_policies(source_path)
    output.mkdir(parents=True)
    write_json(output / 'input.json', raw)
    write_json(output / 'coverage-plan.json', {'input_sha256': sha256_file(input_path),
               'decision_count': len(raw['steps']), 'symbols': list(SYMBOLS), 'scenario': scenario,
               'planned_start': raw['initial_mark']['timestamp'], 'planned_end': raw['steps'][-1]['mark']['timestamp'],
               'quote_counts': [len(step['quotes']) for step in raw['steps']], 'policy_names': list(policies),
               'economic_validation': 'unverified'})
    results = {}
    for name, policy in policies.items():
        try:
            results[name] = replay(raw, policy, output / name)
        except ReplayError:
            results[name] = read_json(output / name / 'result.json')
    complete = all(result['status'] == 'complete' for result in results.values())
    if complete:
        coverage = [result['coverage'] for result in results.values()]
        timelines = [[(row['decision_timestamp'], row['fill_timestamp'], row['mark_timestamp']) for row in result['trace']] for result in results.values()]
        if any(item != coverage[0] for item in coverage) or any(item != timelines[0] for item in timelines):
            raise ValueError('Independent accounts do not share the registered comparison timeline')
    summary = {'status': 'complete' if complete else 'incomplete', 'policies': {name: result['status'] for name, result in results.items()},
               'economic_validation': 'unverified', 'interpretation': 'assumption_based_research_replay'}
    write_json(output / 'summary.json', summary)
    write_json(output / 'provenance.json', {'input_path': str(input_path), 'input_sha256': sha256_file(input_path),
               'schema_sha256': sha256_file(SCHEMA), 'sources': identity, 'runtime': runtime_identity(),
               'scenario': scenario, 'economic_validation': 'unverified',
               'artifacts': {str(path.relative_to(output)): sha256_file(path) for path in sorted(output.rglob('*.json'))}})
    return summary


def main() -> None:
    """Run the explicitly selected scenario CLI."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--scenario', choices=('fixture', 'shadow_quote'), required=True)
    parser.add_argument('--sources', type=Path, help='Required frozen-source manifest for shadow_quote; forbidden for fixture')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    try:
        result = run_scenario(args.input, args.scenario, args.sources, args.output)
    except (ValueError, OSError) as exc:
        parser.exit(2, f'{exc}\n')
    print(json.dumps(result, indent=2))
    if result['status'] != 'complete':
        sys.exit(2)


if __name__ == '__main__':
    main()
