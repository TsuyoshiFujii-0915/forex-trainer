"""Frozen common-direction decisions and deterministic scalar allocation."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd

from .common_basket import PAIRS, advance, utc
from .common_basket_study import digest, read, verify_bundle
from .full_period import utc_instant

UNITS = 'basket_h_business_day_price_simple_return'
TIE = 1e-12
RESIDUAL = 1e-10


def vector(value: np.ndarray, origin: str) -> np.ndarray:
    """Copy and freeze a finite nine-pair vector.

    Args:
        value: Ordered numerical vector.
        origin: Input identity for errors.

    Returns:
        Immutable float64 vector.
    """
    result = np.array(value, dtype=float, copy=True)
    if result.shape != (9,) or not np.isfinite(result).all():
        raise ValueError(f'{origin}: expected nine finite values')
    result.setflags(write=False)
    return result


def label(value: str, origin: str) -> pd.Timestamp:
    """Validate one registered London midnight weekday label.

    Args:
        value: Timezone-aware ISO instant.
        origin: Error context.

    Returns:
        UTC pandas Timestamp.
    """
    at = utc_instant(value, origin)
    day = at.tz_convert('Europe/London').date()
    if day.weekday() >= 5 or at != utc_instant(utc(day), origin):
        raise ValueError(f'{origin}: expected London midnight weekday, got {value}')
    return at


@dataclass(frozen=True)
class Costs:
    """One-way notional spread/commission and daily absolute holding markup."""

    spreads: np.ndarray
    commission: float
    markup: float

    def __post_init__(self) -> None:
        spreads = vector(self.spreads, 'costs.spreads')
        if (spreads < 0).any() or not np.isfinite([self.commission, self.markup]).all() or min(self.commission, self.markup) < 0:
            raise ValueError('costs: rates must be finite and nonnegative')
        object.__setattr__(self, 'spreads', spreads)

    def record(self) -> dict[str, Any]:
        """Return explicit serializable cost rates.

        Returns:
            Rates in registered units.
        """
        return {'spreads': self.spreads.tolist(), 'commission': self.commission, 'markup': self.markup}


@dataclass(frozen=True)
class State:
    """Current information only; future marks and rates are absent."""

    at: str
    pairs: tuple[str, ...]
    equity: float
    quantities: np.ndarray
    prices: np.ndarray
    carry: np.ndarray
    costs: Costs

    def __post_init__(self) -> None:
        label(self.at, 'state.at')
        if self.pairs != PAIRS or not np.isfinite(self.equity) or self.equity <= 0:
            raise ValueError(f'state at {self.at}: pair order or equity invalid')
        for name in ('quantities', 'prices', 'carry'):
            object.__setattr__(self, name, vector(getattr(self, name), f'state.{name} at {self.at}'))
        if (self.prices <= 0).any():
            raise ValueError(f'state.prices at {self.at}: positive prices required')
        if not np.isfinite(self.weights).all():
            raise ValueError(f'state at {self.at}: nonfinite marked weights')

    @property
    def weights(self) -> np.ndarray:
        """Return current marked equity weights."""
        return self.quantities * self.prices / self.equity

    @property
    def identity(self) -> str:
        """Return the state identity binding a decision to its originating account."""
        return digest({'at': self.at, 'pairs': self.pairs, 'equity': self.equity,
                       'quantities': self.quantities.tolist(), 'prices': self.prices.tolist(),
                       'carry': self.carry.tolist(), 'costs': self.costs.record()})


@dataclass(frozen=True)
class Forecast:
    """One sealed prediction with its train-only risk and information lineage."""

    fold: str
    horizon: int
    at: str
    cutoff: str
    parameter_as_of: str
    covariance_as_of: str
    value: float
    units: str
    pairs: tuple[str, ...]
    covariance: np.ndarray
    bundle_sha256: str
    model_sha256: str
    provenance_sha256: str

    def __post_init__(self) -> None:
        at = label(self.at, 'forecast.decision')
        cutoff = utc_instant(self.cutoff, 'forecast.cutoff')
        if not (utc_instant(self.covariance_as_of, 'forecast.covariance_asof')
                <= utc_instant(self.parameter_as_of, 'forecast.parameter_asof') < cutoff <= at):
            raise ValueError(f'forecast at {self.at}: future parameter/covariance as-of')
        if self.horizon not in (1, 5) or self.units != UNITS or self.pairs != PAIRS or not np.isfinite(self.value):
            raise ValueError(f'forecast at {self.at}: horizon/units/pairs/value mismatch')
        covariance = np.array(self.covariance, dtype=float, copy=True)
        if (covariance.shape != (9, 9) or not np.isfinite(covariance).all()
                or not np.allclose(covariance, covariance.T, rtol=0, atol=1e-14)
                or np.linalg.eigvalsh(covariance).min() <= 0):
            raise ValueError(f'forecast at {self.at}: invalid train covariance')
        covariance.setflags(write=False)
        object.__setattr__(self, 'covariance', covariance)
        for value in (self.bundle_sha256, self.model_sha256, self.provenance_sha256):
            if len(value) != 64 or any(c not in '0123456789abcdef' for c in value):
                raise ValueError(f'forecast at {self.at}: invalid source hash')

    @property
    def identity(self) -> dict[str, Any]:
        """Return shared fixed/cost forecast and covariance provenance."""
        return {name: getattr(self, name) for name in (
            'fold', 'horizon', 'at', 'cutoff', 'parameter_as_of', 'covariance_as_of',
            'value', 'units', 'pairs', 'bundle_sha256', 'model_sha256', 'provenance_sha256')}


def horizon_days(forecast: Forecast) -> float:
    """Measure the full registered horizon using actual UTC calendar duration.

    Args:
        forecast: One- or five-business-day prediction.

    Returns:
        Elapsed fractional days, including weekends and DST.
    """
    at = label(forecast.at, 'forecast.at')
    end = utc_instant(utc(advance(at.tz_convert('Europe/London').date(), forecast.horizon)), 'horizon.end')
    return float((end - at).total_seconds() / 86400)


def utility(state: State, forecast: Forecast, weights: np.ndarray) -> dict[str, float]:
    """Evaluate current-equity utility with constant known rates and notionals.

    Args:
        state: Current marked account and costs.
        forecast: Shared horizon prediction and train covariance.
        weights: Candidate marked weights, including literal hold drift.

    Returns:
        Separate price, financing, entry, markup, risk, and total terms.
    """
    if label(state.at, 'state.at') != label(forecast.at, 'forecast.at'):
        raise ValueError('utility: decision timestamp mismatch')
    w = vector(weights, 'utility.weights')
    days = horizon_days(forecast)
    parts = {'predicted_price': float(w.sum() * forecast.value),
             'signed_financing': float(w @ -state.carry) * days / 365,
             'entry_cost': float(np.abs(w - state.weights) @ (state.costs.spreads + state.costs.commission)),
             'holding_markup': float(np.abs(w).sum()) * state.costs.markup * days,
             'risk_penalty': float(5 * w @ forecast.covariance @ w)}
    parts['total'] = parts['predicted_price'] + parts['signed_financing'] - parts['entry_cost'] - parts['holding_markup'] - parts['risk_penalty']
    if not np.isfinite(list(parts.values())).all():
        raise ValueError(f'utility at {state.at}: nonfinite components')
    return parts


@dataclass(frozen=True)
class Candidate:
    """One enumerated solver candidate with explicit economic tie keys."""

    mode: str
    u: float
    quantities: np.ndarray
    total: float
    traded_notional: float
    gross: float


def select_candidate(candidates: list[Candidate]) -> Candidate:
    """Select relative to the global maximum, then apply economic tie order.

    Args:
        candidates: All feasible stationary, kink, endpoint, and hold choices.

    Returns:
        Deterministic winning candidate.
    """
    if not candidates:
        raise ValueError('solver: no feasible candidate')
    if not all(np.isfinite([c.total, c.traded_notional, c.gross, c.u]).all() and np.isfinite(c.quantities).all() for c in candidates):
        raise ValueError('solver: nonfinite candidate')
    best = max(c.total for c in candidates)
    return min((c for c in candidates if best - c.total <= TIE),
               key=lambda c: (c.traded_notional, c.gross, c.u, c.mode))


@dataclass(frozen=True)
class Decision:
    """Quantity instruction bound to the current account, with prediction trace."""

    mode: str
    quantities: np.ndarray
    u: float | None
    state_sha256: str
    forecast_identity: dict[str, Any] | None
    utility: dict[str, float] | None
    solver: dict[str, Any]

    def record(self) -> dict[str, Any]:
        """Serialize the instruction without omitting its solver state.

        Returns:
            Finite JSON-compatible trace.
        """
        record = asdict(self)
        record['quantities'] = self.quantities.tolist()
        return record


def proportional_cap(state: State, proposed: np.ndarray) -> np.ndarray:
    """Find the greatest proportional quantity feasible after its own costs.

    Args:
        state: Marked quantities and equity before this trade.
        proposed: Unmodified proposal or current quantities for a drift override.

    Returns:
        Proportional quantities satisfying gross 5 and pair 1 to 1e-10.

    Raises:
        ValueError: No feasible shrink exists, or a numerical residual is invalid.
    """
    q = vector(proposed, 'risk.proposed')
    x, old = q * state.prices, state.quantities * state.prices
    rates = state.costs.spreads + state.costs.commission
    required = max(float(np.abs(x).sum()) / 5, float(np.max(np.abs(x))))
    kinks = sorted({0., 1., *(float(a / b) for a, b in zip(old, x, strict=True) if b != 0 and 0 < a / b < 1)})

    def slack(scale: float) -> float:
        return float(state.equity - np.abs(scale * x - old) @ rates - scale * required)

    feasible = [s for s in kinks if slack(s) >= 0]
    for lo, hi in zip(kinks[:-1], kinks[1:], strict=True):
        a, b = slack(lo), slack(hi)
        if not np.isfinite([a, b]).all():
            raise ValueError(f'risk at {state.at}: nonfinite constraint')
        if a * b < 0:
            feasible.append(lo + (hi - lo) * a / (a - b))
    if not feasible:
        raise ValueError(f'risk at {state.at}: no feasible proportional quantity')
    scale = max(feasible)
    result = q * scale
    equity = state.equity - float(np.abs(result * state.prices - old) @ rates)
    if equity <= 0 or not np.isfinite(equity):
        raise ValueError(f'risk at {state.at}: no feasible positive equity')
    w = result * state.prices / equity
    residual = max(0., float(np.abs(w).sum()) - 5, float(np.abs(w).max()) - 1)
    if not np.isfinite(residual) or residual > RESIDUAL:
        raise ValueError(f'risk at {state.at}: constraint residual {residual}')
    return result


def allocate(state: State, forecast: Forecast, mode: Literal['fixed', 'cost']) -> Decision:
    """Solve the registered common basket, keeping literal quantity hold distinct.

    Args:
        state: Current information after mandatory drift reduction.
        forecast: The same sealed prediction for either allocation mode.
        mode: Registered fixed or cost-aware allocation treatment.

    Returns:
        Instruction with all solver candidates and expected utility components.
    """
    if mode not in ('fixed', 'cost'):
        raise ValueError(f'allocator: unsupported mode {mode}')
    if np.abs(state.weights).sum() > 5 + RESIDUAL or np.abs(state.weights).max() > 1 + RESIDUAL:
        raise ValueError('allocator: risk drift must be reduced before inference')
    kinks = sorted({-1., 0., 1., *(float(9 * w) for w in state.weights)})
    partitions = [point for point in kinks if -1 <= point <= 1]
    points = {float(np.sign(forecast.value))} if mode == 'fixed' else set(partitions)
    if mode == 'cost':
        basket = np.ones(9) / 9
        quadratic = float(5 * basket @ forecast.covariance @ basket)
        linear = forecast.value + float(basket @ -state.carry) * horizon_days(forecast) / 365
        for lo, hi in zip(partitions[:-1], partitions[1:], strict=True):
            mid = (lo + hi) / 2
            derivative = linear - float(np.sign(mid / 9 - state.weights) @ (state.costs.spreads + state.costs.commission)) / 9
            derivative -= np.sign(mid) * state.costs.markup * horizon_days(forecast)
            stationary = derivative / (2 * quadratic)
            if lo < stationary < hi:
                points.add(float(stationary))
    candidates = []
    for u in sorted(points):
        w = np.full(9, u / 9)
        q = w * state.equity / state.prices
        parts = utility(state, forecast, w)
        candidates.append(Candidate('basket_target', u, q, parts['total'],
                                    float(np.abs((q - state.quantities) * state.prices).sum()), abs(u)))
    if mode == 'cost':
        parts = utility(state, forecast, state.weights)
        candidates.append(Candidate('hold_quantity', float(state.weights.sum()), state.quantities.copy(),
                                    parts['total'], 0., float(np.abs(state.weights).sum())))
    chosen = select_candidate(candidates)
    residual = max(0., abs(chosen.u) - 1) if chosen.mode == 'basket_target' else max(
        0., chosen.gross - 5, float(np.abs(state.weights).max()) - 1)
    if not np.isfinite(residual) or residual > RESIDUAL:
        raise ValueError(f'allocator at {state.at}: constraint residual {residual}')
    return Decision(chosen.mode, vector(chosen.quantities, 'decision.quantities'), chosen.u, state.identity,
                    forecast.identity, utility(state, forecast, chosen.quantities * state.prices / state.equity),
                    {'status': 'optimal' if mode == 'cost' else 'fixed_sign', 'kinks': kinks,
                     'constraint_residual': residual, 'horizon_elapsed_days': horizon_days(forecast),
                     'candidates': [{'mode': c.mode, 'u': c.u, 'utility': c.total,
                                     'traded_notional': c.traded_notional, 'gross': c.gross} for c in candidates]})


def load_forecasts(directory: Path) -> dict[tuple[str, int, str], Forecast]:
    """Consume the complete verified Issue 41 handoff without future labels.

    Args:
        directory: Sealed complete forecast bundle.

    Returns:
        Decision-keyed immutable forecasts with train-only covariance.
    """
    rows = verify_bundle(directory)
    manifest = read(directory / 'manifest.json')
    models = {(row['fold'], row['horizon']): read(directory / row['model_path']) for row in manifest['cells']}
    result = {}
    for row in rows:
        key = row['fold'], row['horizon_business_days'], row['decision_at_utc']
        model = models[key[:2]]
        p = model['provenance']
        if p['prediction_unit'] != UNITS or p['covariance_formula'] != '0.9*S+0.1*diag(S)+1e-8*I; S=sample_cov(pair_h_simple_returns,ddof=1)':
            raise ValueError(f'forecast {key}: prediction units/covariance formula mismatch')
        result[key] = Forecast(row['fold'], row['horizon_business_days'], row['decision_at_utc'],
                               row['information_cutoff_utc'], row['parameter_as_of'], p['covariance_as_of'],
                               row['predicted_basket_simple_return'], p['prediction_unit'], tuple(p['pair_order']),
                               np.array(model['selected']['covariance']), manifest['bundle_content_sha256'],
                               row['model_sha256'], row['provenance_sha256'])
    return result
