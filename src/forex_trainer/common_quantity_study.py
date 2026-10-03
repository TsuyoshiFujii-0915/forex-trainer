"""Bounded synthetic execution, immutable measurement seals, and trace audit."""
from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import date
import json
import os
from pathlib import Path
from typing import Any, Literal

import numpy as np

from .artifact_provenance import sha256_file
from .common_allocation import Costs, Decision, Forecast, State, UNITS, allocate, label
from .common_basket import PAIRS, advance, utc
from .common_basket_study import CAMPAIGN, ROOT, checked, digest, execution_runtime, read, require_merge, write
from .common_quantity import MEASUREMENT_ID, Mark, QuantityExecutionError, QuantityPolicy, instruction, legacy_policy, replay
from .quote_replay_policies import fixture_policies

RUN = ROOT / 'runs/issue42-common-quantity-v3'
SAVED = ROOT / 'docs/research/results/issue42/fixture'
ACCOUNT_NAMES = ('fixed-h1', 'cost-h1', 'fixed-h5', 'cost-h5', 'hold', 'F1', 'F2',
                 'risk', 'margin', 'bankruptcy', 'canonical', 'ridge', 'ppo_ens3')


def load_config(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """Validate the fixed synthetic scope and inherited v3 budget.

    Args:
        path: Explicit Issue 42 configuration.

    Returns:
        Validated config and original immutable campaign registration.
    """
    config = read(path)
    expected = {'campaign_id': CAMPAIGN, 'measurement_id': MEASUREMENT_ID,
                'run_directory': str(RUN.relative_to(ROOT)), 'fixture_account_limit': 16,
                'planned_fixture_accounts': 13, 'fixture_input': 'docs/research/protocols/issue42/fixture.json',
                'market_accounts': 0, 'fit_calls': 0, 'ppo_training': 0,
                'risk': {'gross_cap': 5., 'pair_cap': 1., 'margin_threshold': .2, 'initial_equity': 1_000_000.},
                'solver': {'risk_coefficient': 5., 'tie_tolerance': 1e-12, 'constraint_tolerance': 1e-10}}
    if set(config) != set(expected) | {'registration', 'fixture_sha256'} or any(config[k] != v for k, v in expected.items()):
        raise ValueError(f'Issue 42 contract mismatch: {path}')
    if config['registration'] != {'path': 'docs/research/protocols/issue40/revisions/v3/registration.json',
                                  'sha256': '5d096705e68536faa48eadf7a701e7d034f918e5203b45fb8191bbefeae6e8aa'}:
        raise ValueError(f'Issue 42 parent contract mismatch: {path}')
    registration = read(checked(config['registration']))
    if (registration['campaign_id'] != CAMPAIGN
            or registration['gates']['short_data_ready']['status'] != 'ready'
            or registration['counters']['issue42_fixture_accounts']['limit'] != 16):
        raise ValueError('Issue 42 v3 gate/budget contract mismatch')
    fixture = read(checked({'path': config['fixture_input'], 'sha256': config['fixture_sha256']}))
    if (fixture['kind'] != 'synthetic_only' or tuple(fixture['accounts']) != ACCOUNT_NAMES
            or fixture['control_training'] != 0 or fixture['control_seeds'] != [42, 43, 44]):
        raise ValueError('Issue 42 fixture contract mismatch')
    return config, registration


def require_unused_budget(run: Path, saved: Path) -> None:
    """Reject any previous reserved run, including a fresh clone's saved run.

    Args:
        run: Campaign-fixed execution directory.
        saved: Committed results directory consulted across clones.
    """
    for path in (run, saved):
        if path.exists():
            raise ValueError(f'Issue 42 fixture budget already reserved or consumed: {path}; retry requires separate shared-budget registration')


def seal(config_path: Path, merge_commit: str, output: Path) -> dict[str, Any]:
    """Seal the clean implementation before the single synthetic campaign.

    Args:
        config_path: Registered fixture configuration.
        merge_commit: Merged v3 dependency commit on origin/main.
        output: New activation directory.

    Returns:
        Runtime, input, config, budget, and measurement activation.
    """
    config, registration = load_config(config_path)
    require_unused_budget(RUN, SAVED)
    require_merge(registration, merge_commit)
    runtime = execution_runtime()
    activation = {'campaign_id': CAMPAIGN, 'measurement_id': MEASUREMENT_ID,
                  'dependency_merge_commit': merge_commit, 'registration': config['registration'],
                  'runtime': runtime, 'config_sha256': sha256_file(config_path),
                  'fixture_sha256': config['fixture_sha256'], 'planned_accounts': list(ACCOUNT_NAMES),
                  'fixture_account_limit': 16, 'market_accounts': 0, 'fit_calls': 0, 'ppo_training': 0}
    output.mkdir(parents=True, exist_ok=False)
    write(output / 'activation.json', activation)
    return activation


def append_event(path: Path, event: dict[str, Any]) -> None:
    """Durably append a reservation or outcome before advancing the campaign.

    Args:
        path: Append-only account ledger.
        event: Finite JSON event.
    """
    with path.open('a', encoding='utf-8') as stream:
        stream.write(json.dumps(event, sort_keys=True, allow_nan=False) + '\n')
        stream.flush()
        os.fsync(stream.fileno())


def fixture_marks(raw: dict[str, Any]) -> tuple[Mark, ...]:
    """Construct the registered deterministic synthetic inputs without fitting.

    Args:
        raw: Immutable fixture design.

    Returns:
        Six measured marks spanning a DST weekend.
    """
    marks = []
    for i in range(raw['mark_count']):
        at = utc(advance(date.fromisoformat(raw['first_london_date']), i))
        prices = np.array(raw['initial_prices']) * (1 + np.array(raw['daily_pair_growth'])) ** i
        window = np.zeros((9, 32, 8))
        window[:, :, 0] = np.log1p(raw['daily_pair_growth'])[:, None]
        window[:, :, 1] = .01
        window[:, :, 2] = np.arange(9)[:, None] * .001
        window[:, :, 3] = (np.arange(9)[:, None] - 4) * .01 + i * .001
        carry = np.array(raw['carry']) + i * raw['daily_carry_increment']
        window[:, :, 6] = carry[:, None]
        marks.append(Mark(at, prices, carry, window))
    return tuple(marks)


def synthetic_forecasts(raw: dict[str, Any], marks: tuple[Mark, ...], horizon: int) -> dict[str, Forecast]:
    """Seal fixed numerical fixture forecasts without training or future labels.

    Args:
        raw: Registered fixture values.
        marks: Decision times and final mark.
        horizon: Registered one or five business days.

    Returns:
        Identical forecast objects for fixed/cost consumption.
    """
    covariance = np.eye(9) * raw['covariance_diagonal']
    model = digest({'synthetic_covariance': covariance.tolist(), 'horizon': horizon})
    source = digest({'kind': 'synthetic_not_fitted', 'model': model})
    bundle = digest({'forecasts': raw['forecasts'], 'horizon': horizon, 'times': [m.at for m in marks[:-1]], 'model': model})
    return {mark.at: Forecast('synthetic', horizon, mark.at, '2024-01-01T00:00:00+00:00',
                             '2023-12-29T00:00:00+00:00', '2023-06-30T23:00:00+00:00', value,
                             UNITS, PAIRS, covariance, bundle, model, source)
            for mark, value in zip(marks[:-1], raw['forecasts'], strict=True)}


def allocation_policy(forecasts: dict[str, Forecast], mode: Literal['fixed', 'cost']) -> QuantityPolicy:
    """Bind one allocation treatment to the shared forecast series.

    Args:
        forecasts: Exact decision-keyed sealed predictions.
        mode: Fixed or cost-aware allocation.

    Returns:
        Causal quantity policy for an independent account.
    """
    def decide(state: State, observation: dict[str, np.ndarray], window: np.ndarray) -> Decision:
        if state.at not in forecasts:
            raise ValueError(f'quantity policy: missing forecast at {state.at}')
        return allocate(state, forecasts[state.at], mode)
    return decide


def scripted_policy(raw: dict[str, Any], marks: tuple[Mark, ...], name: str) -> QuantityPolicy:
    """Define literal hold and stress paths solely for cash reconciliation.

    Args:
        raw: Registered fixture values.
        marks: Exact decision axis.
        name: One registered synthetic accounting scenario.

    Returns:
        Deterministic instruction provider without a predictor fit.
    """
    def decide(state: State, observation: dict[str, np.ndarray], window: np.ndarray) -> Decision:
        index = next(i for i, mark in enumerate(marks) if mark.at == state.at)
        if name == 'hold':
            mode = raw['hold_modes'][index]
            if mode == 'hold':
                return instruction(state, state.quantities, 'hold_quantity')
            if mode == 'partial':
                return instruction(state, state.quantities / 2, 'partial')
            if mode == 'close':
                return instruction(state, np.zeros(9), 'close')
            if mode not in ('entry', 'short'):
                raise ValueError(f'Unknown fixture instruction: {mode}')
            direction = 1 if mode == 'entry' else -1
            q = np.full(9, direction * raw['hold_gross'] / 9) * state.equity / state.prices
            return instruction(state, q, 'legacy_target')
        if index > 0:
            return instruction(state, state.quantities, 'hold_quantity')
        if name == 'risk':
            weights = np.array(raw['risk_weights'])
        elif name in ('margin', 'bankruptcy'):
            weights = np.full(9, -1 / 9)
        else:
            raise ValueError(f'Unknown fixture policy: {name}')
        return instruction(state, weights * state.equity / state.prices, 'legacy_target')
    return decide


def run_fixture(config_path: Path, activation_path: Path) -> dict[str, Any]:
    """Reserve and run exactly thirteen synthetic accounts, with no auto retry.

    Args:
        config_path: Fixed Issue 42 configuration.
        activation_path: Clean runtime seal made before this run.

    Returns:
        Sealed artifact manifest; failures leave consumed reservations and incident.
    """
    config, registration = load_config(config_path)
    require_unused_budget(RUN, SAVED)
    activation = read(activation_path)
    require_merge(registration, activation['dependency_merge_commit'])
    if (activation['campaign_id'] != CAMPAIGN or activation['measurement_id'] != MEASUREMENT_ID
            or activation['config_sha256'] != sha256_file(config_path)
            or activation['fixture_sha256'] != config['fixture_sha256']
            or activation['registration'] != config['registration']
            or activation['planned_accounts'] != list(ACCOUNT_NAMES)
            or activation['fixture_account_limit'] != 16
            or any(activation[k] != 0 for k in ('market_accounts', 'fit_calls', 'ppo_training'))
            or activation['runtime'] != execution_runtime()):
        raise ValueError('Issue 42 activation/input/runtime mismatch')
    RUN.mkdir(parents=True, exist_ok=False)
    (RUN / 'accounts').mkdir()
    ledger = RUN / 'account-ledger.jsonl'
    completed: list[str] = []
    reserved: list[str] = []
    try:
        write(RUN / 'activation.json', activation)
        write(RUN / 'config.json', config)
        raw = read(ROOT / config['fixture_input'])
        write(RUN / 'fixture.json', raw)
        marks = fixture_marks(raw)
        costs = Costs(np.array(raw['spreads']), raw['commission'], raw['markup'])
        forecasts = {h: synthetic_forecasts(raw, marks, h) for h in (1, 5)}
        controls, identity = fixture_policies()
        write(RUN / 'policy-sources.json', identity)
        runtime_hash = digest(activation['runtime'])
        for name in ACCOUNT_NAMES:
            scenario_marks, scenario_costs = marks, costs
            source: dict[str, Any] = {'kind': 'synthetic_untrained_interface_fixture', 'name': name}
            if name in controls:
                policy = legacy_policy(controls[name])
                source['parameters'] = identity
            elif name in ('hold', 'risk', 'margin', 'bankruptcy'):
                if name != 'hold':
                    relatives = np.array(raw['risk_first_relatives']) if name == 'risk' else np.full(9, raw[f'{name}_first_relative'])
                    scenario_marks = (marks[0], *(replace(m, prices=marks[0].prices * relatives) for m in marks[1:]))
                policy = scripted_policy(raw, scenario_marks, name)
            else:
                horizon = 5 if name.endswith('h5') else 1
                mode = 'cost' if name.startswith('cost') else 'fixed'
                policy = allocation_policy(forecasts[horizon], mode)
                if name == 'F1':
                    scenario_costs = replace(costs, spreads=costs.spreads * 2)
                if name == 'F2':
                    scenario_costs = replace(costs, markup=costs.markup * 2)
            append_event(ledger, {'event': 'reserved', 'counter': 'issue42_fixture_accounts', 'work_item': name,
                                  'attempt': 1, 'fixture_input_sha256': config['fixture_sha256'], 'runtime_sha256': runtime_hash})
            reserved.append(name)
            result = replay(scenario_marks, scenario_costs, policy, runtime_hash, source)
            write(RUN / 'accounts' / f'{name}.json', result)
            append_event(ledger, {'event': 'complete', 'work_item': name, 'status': result['status']})
            completed.append(name)
        if execution_runtime() != activation['runtime']:
            raise ValueError('Issue 42 runtime changed during fixture execution')
        load_config(config_path)
        append_event(ledger, {'event': 'results_opened', 'fixture_accounts': len(reserved), 'portfolio_decision': 'deferred_to_issue43'})
        manifest = {'campaign_id': CAMPAIGN, 'measurement_id': MEASUREMENT_ID, 'status': 'complete',
                    'fixture_accounts': len(reserved), 'fixture_limit': 16, 'market_accounts': 0,
                    'fit_calls': 0, 'ppo_training': 0, 'shared_retry_accounts': 0,
                    'accounts': list(ACCOUNT_NAMES), 'runtime_sha256': runtime_hash,
                    'artifacts': {str(p.relative_to(RUN)): sha256_file(p) for p in sorted(RUN.rglob('*')) if p.is_file()}}
        manifest['content_sha256'] = digest(manifest)
        write(RUN / 'manifest.json', manifest)
        verify_fixture(RUN)
        return manifest
    except Exception as exc:
        write(RUN / 'incident.json', {'status': 'execution_error', 'error': f'{type(exc).__name__}: {exc}',
                                     'reserved': reserved, 'completed': completed, 'fixture_accounts': len(reserved),
                                     'shared_retry_accounts': 0, 'automatic_retry': False,
                                     'account_evidence': exc.evidence if isinstance(exc, QuantityExecutionError) else None})
        raise


def verify_fixture(directory: Path) -> dict[str, Any]:
    """Verify saved evidence without fitting, inference, or opening an account.

    Args:
        directory: Original or byte-identical saved fixture directory.

    Returns:
        Verified complete manifest.
    """
    if (directory / 'incident.json').exists():
        raise ValueError('Issue 42 fixture has an unresolved incident')
    manifest = read(directory / 'manifest.json')
    unsigned = {k: v for k, v in manifest.items() if k != 'content_sha256'}
    if digest(unsigned) != manifest['content_sha256']:
        raise ValueError('Issue 42 manifest content hash mismatch')
    if (manifest['status'] != 'complete' or manifest['measurement_id'] != MEASUREMENT_ID
            or manifest['accounts'] != list(ACCOUNT_NAMES) or manifest['fixture_accounts'] != 13
            or any(manifest[k] != 0 for k in ('market_accounts', 'fit_calls', 'ppo_training', 'shared_retry_accounts'))):
        raise ValueError('Issue 42 fixture status/counter/measurement mismatch')
    required = {'activation.json', 'config.json', 'fixture.json', 'policy-sources.json', 'account-ledger.jsonl',
                *(f'accounts/{name}.json' for name in ACCOUNT_NAMES)}
    actual = {str(p.relative_to(directory)) for p in directory.rglob('*') if p.is_file()} - {'manifest.json'}
    if set(manifest['artifacts']) != required or actual != required:
        raise ValueError('Issue 42 artifact inventory mismatch')
    for name, expected in manifest['artifacts'].items():
        path = (directory / name).resolve()
        if not path.is_relative_to(directory.resolve()) or sha256_file(path) != expected:
            raise ValueError(f'Issue 42 artifact hash mismatch: {name}')
    events = [json.loads(line) for line in (directory / 'account-ledger.jsonl').read_text().splitlines()]
    if ([e['work_item'] for e in events if e['event'] == 'reserved'] != list(ACCOUNT_NAMES)
            or [e['work_item'] for e in events if e['event'] == 'complete'] != list(ACCOUNT_NAMES)):
        raise ValueError('Issue 42 reservation/completion ledger mismatch')
    activation = read(directory / 'activation.json')
    if digest(activation['runtime']) != manifest['runtime_sha256']:
        raise ValueError('Issue 42 runtime seal mismatch')
    for name in ACCOUNT_NAMES:
        account = read(directory / 'accounts' / f'{name}.json')
        if account['measurement_id'] != MEASUREMENT_ID or account['runtime_sha256'] != manifest['runtime_sha256']:
            raise ValueError(f'Issue 42 account comparison identity mismatch: {name}')
        audit_account(account, name)
    return manifest


def audit_account(account: dict[str, Any], origin: str) -> None:
    """Reconcile saved quantities and cash without executing an account.

    Args:
        account: Complete saved synthetic account.
        origin: Account identity included in any mismatch.
    """
    equity, quantity = 1_000_000., np.zeros(9)
    costs = Costs(np.array(account['costs']['spreads']), account['costs']['commission'], account['costs']['markup'])

    def equal(actual: Any, expected: Any, field: str) -> None:
        if not np.isfinite(actual).all() or not np.allclose(actual, expected, rtol=1e-12, atol=1e-8):
            raise ValueError(f'Issue 42 accounting reconciliation: {origin}/{field}')

    for index, row in enumerate(account['trace']):
        prefix = f'{index}/'
        prices, next_prices = np.array(row['prices']), np.array(row['next_prices'])
        equal(row['equity_start'], equity, prefix + 'equity_start')
        traded = 0.
        for trade in row['trades']:
            equal(trade['quantities_before'], quantity, prefix + 'quantities_before')
            new = np.array(trade['quantities_after'])
            notional = np.abs(new - quantity) * prices
            equal(trade['traded_notional'], notional, prefix + 'traded_notional')
            equal(trade['spread'], float(notional @ costs.spreads), prefix + 'spread')
            equal(trade['commission'], float(notional.sum() * costs.commission), prefix + 'commission')
            equity -= trade['spread'] + trade['commission']
            traded += float(notional.sum())
            quantity = new
        equal(row['quantities'], quantity, prefix + 'quantities')
        equal(row['actual_traded_notional'], traded, prefix + 'turnover')
        elapsed = (label(row['next_at'], origin) - label(row['at'], origin)).total_seconds() / 86400
        equal(row['elapsed_days'], elapsed, prefix + 'elapsed_days')
        equal(row['price_pnl'], float(quantity @ (next_prices - prices)), prefix + 'price_pnl')
        equal(row['financing'], float((quantity * next_prices) @ -np.array(row['carry'])) * elapsed / 365, prefix + 'financing')
        equal(row['markup'], float(np.abs(quantity * prices).sum()) * costs.markup * elapsed, prefix + 'markup')
        weights = quantity * prices / equity
        if np.abs(weights).sum() > 5 + 1e-10 or np.abs(weights).max() > 1 + 1e-10:
            raise ValueError(f'Issue 42 accounting constraint violation: {origin}/{index}')
        equity += row['price_pnl'] + row['financing'] - row['markup']
        equal(row['equity_end'], equity, prefix + 'equity_end')
    equal(account['final_equity'], equity, 'final_equity')
    equal(account['final_quantities'], quantity, 'final_quantities')
    if equity > 0:
        equal(account['net_log_return'], np.log(equity / 1_000_000), 'net_log_return')
    elif account['net_log_return'] is not None:
        raise ValueError(f'Issue 42 accounting: nonpositive equity log must be null: {origin}')


def main() -> int:
    """Run explicit synthetic seal/run or read-only evidence verification.

    Returns:
        Zero on success; exceptions propagate with their origin.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    sealing = sub.add_parser('seal')
    sealing.add_argument('--config', type=Path, required=True)
    sealing.add_argument('--dependency-merge-commit', required=True)
    sealing.add_argument('--output', type=Path, required=True)
    running = sub.add_parser('fixture')
    running.add_argument('--config', type=Path, required=True)
    running.add_argument('--activation', type=Path, required=True)
    verifying = sub.add_parser('verify')
    verifying.add_argument('--directory', type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'seal':
        result = seal(args.config, args.dependency_merge_commit, args.output)
    elif args.command == 'fixture':
        result = run_fixture(args.config, args.activation)
    else:
        result = verify_fixture(args.directory)
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
