"""Causal five-feature common-basket ridge for the frozen Issue 41 contract."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any, Protocol

import numpy as np
import pandas as pd

from .features import carry_annual, mom24, sma20_ratio

PAIRS = ('JPY/USD', 'JPY/EUR', 'JPY/GBP', 'JPY/AUD', 'JPY/CHF',
         'JPY/CAD', 'JPY/NZD', 'JPY/NOK', 'JPY/SEK')
FEATURES = ('log_return', 'volatility', 'sma20_ratio', 'mom24', 'carry_annual')
ALPHAS = (0.01, 0.1, 1.0)
HISTORY = 63


class InputError(ValueError):
    """Invalid inputs with machine-readable row-level evidence."""

    def __init__(self, message: str, rows: list[dict[str, Any]]) -> None:
        super().__init__(message)
        self.rows = rows


class Ledger(Protocol):
    """The candidate fit must reserve its attempt before numerical work."""

    def reserve(self, work_item: str) -> None: ...
    def complete(self, work_item: str) -> None: ...


@dataclass(frozen=True)
class Panel:
    """Original market rows with causal features and block positions."""

    dates: tuple[date, ...]
    prices: np.ndarray
    features: np.ndarray
    history: np.ndarray
    block: np.ndarray


@dataclass(frozen=True)
class Rows:
    """Purged labeled observations and complete exclusion evidence."""

    dates: tuple[date, ...]
    target_dates: tuple[date, ...]
    features: np.ndarray
    targets: np.ndarray
    pair_returns: np.ndarray
    counts: dict[str, int]
    excluded: list[dict[str, Any]]


@dataclass(frozen=True)
class Model:
    """One train-only ridge candidate, including allocation covariance."""

    alpha: float
    mean: np.ndarray
    scale: np.ndarray
    coefficients: np.ndarray
    intercept: float
    covariance: np.ndarray
    validation_mse: float

    def predict(self, features: np.ndarray) -> np.ndarray:
        """Predict basket simple returns from current five-column observations.

        Args:
            features: Finite decisions by five features.

        Returns:
            Finite scalar prediction for every supplied decision.
        """
        if features.ndim != 2 or features.shape[1] != 5 or not np.isfinite(features).all():
            raise InputError('Prediction requires finite five-column features', [{'reason': 'features'}])
        result = ((features - self.mean) / self.scale) @ self.coefficients + self.intercept
        if not np.isfinite(result).all():
            raise InputError('Nonfinite basket prediction', [{'reason': 'prediction'}])
        return result


def advance(day: date, horizon: int) -> date:
    """Advance by registered London weekdays, excluding no holidays.

    Args:
        day: London decision date.
        horizon: Number of expected business days.

    Returns:
        Expected label end date.
    """
    for _ in range(horizon):
        day += timedelta(days=1)
        while day.weekday() >= 5:
            day += timedelta(days=1)
    return day


def utc(day: date) -> str:
    """Convert a London midnight label to its DST-aware UTC instant.

    Args:
        day: London calendar date.

    Returns:
        ISO UTC timestamp.
    """
    return pd.Timestamp(day, tz='Europe/London').tz_convert('UTC').isoformat()


def build_panel(market: pd.DataFrame) -> Panel:
    """Compute current features separately inside every observed weekday block.

    Args:
        market: Ordered nine-pair OHLCV and CarryAnnual market frame.

    Returns:
        Causal panel with explicit leading-history ineligibility.

    Raises:
        InputError: Pair order, timestamps, prices, carry, or features are invalid.
    """
    def fail(reason: str) -> None:
        raise InputError(reason, [{'reason': reason}])

    if not isinstance(market.columns, pd.MultiIndex) or market.columns.has_duplicates:
        fail('Market requires unique pair/field columns')
    if tuple(dict.fromkeys(market.columns.get_level_values(0))) != PAIRS:
        fail('Market pair order differs from registered nine-pair order')
    index = market.index
    if not isinstance(index, pd.DatetimeIndex) or index.tz is None:
        fail('Market timestamps require an explicit timezone')
    if len(index) == 0 or index.hasnans or index.has_duplicates or not index.is_monotonic_increasing:
        fail('Market timestamps must be nonempty, unique and monotonic')
    local = index.tz_convert('Europe/London')
    if not local.equals(local.normalize()) or any(day.weekday() >= 5 for day in local):
        fail('Market timestamps must be London weekday midnight labels')
    missing = [(pair, field) for pair in PAIRS for field in ('Close', 'CarryAnnual')
               if (pair, field) not in market.columns]
    if missing:
        fail(f'Missing market columns: {missing}')
    invalid: list[dict[str, Any]] = []
    for pair in PAIRS:
        for field in ('Close', 'CarryAnnual'):
            values = market[pair, field].to_numpy(dtype=float)
            bad = ~np.isfinite(values)
            if field == 'Close':
                bad |= values <= 0
            invalid.extend({'decision_at_utc': index[i].isoformat(), 'pair': pair,
                            'field': field, 'reason': 'nonfinite_or_nonpositive_price' if field == 'Close' else 'nonfinite'}
                           for i in np.flatnonzero(bad))
    if invalid:
        raise InputError(f'Invalid market values: {invalid[0]}', invalid)
    dates = tuple(local.date)
    starts = [0] + [i for i in range(1, len(dates)) if advance(dates[i-1], 1) != dates[i]]
    features = np.full((len(index), 5), np.nan)
    history = np.empty(len(index), dtype=int)
    block_ids = np.empty(len(index), dtype=int)
    for block_id, (start, end) in enumerate(zip(starts, starts[1:] + [len(index)])):
        block = market.iloc[start:end]
        per_pair = []
        for pair in PAIRS:
            frame = block[pair]
            close = frame['Close']
            log = np.log(close / close.shift(1))
            per_pair.append(np.column_stack((log, log.rolling(32, min_periods=32).std(ddof=1),
                                            sma20_ratio(frame), mom24(frame), carry_annual(frame))))
        tensor = np.stack(per_pair)
        bad = np.argwhere(~np.isfinite(tensor[:, HISTORY:, :]))
        for pair_i, row_i, feature_i in bad:
            invalid.append({'decision_at_utc': utc(dates[start + HISTORY + row_i]),
                            'pair': PAIRS[pair_i], 'feature': FEATURES[feature_i], 'reason': 'nonfinite_feature'})
        features[start:end] = tensor.mean(axis=0)
        history[start:end] = np.arange(end-start)
        block_ids[start:end] = block_id
    if invalid:
        raise InputError(f'Invalid computed feature: {invalid[0]}', invalid)
    prices = np.column_stack([market[pair, 'Close'].to_numpy(dtype=float) for pair in PAIRS])
    return Panel(dates, prices, features, history, block_ids)


def training_rows(panel: Panel, start: date, end: date, horizon: int) -> Rows:
    """Select labels without crossing gaps or the next split boundary.

    Args:
        panel: Causal source panel.
        start: Inclusive London decision boundary.
        end: Exclusive decision and target-end boundary.
        horizon: Registered one or five business days.

    Returns:
        Labeled rows, purge counts and individual excluded observations.
    """
    if horizon not in (1, 5) or start >= end:
        raise ValueError(f'Invalid training range/horizon: {start}/{end}/{horizon}')
    counts = dict(present_rows=0, history_excluded=0, label_gap_excluded=0, boundary_purged=0)
    selected: list[int] = []
    excluded: list[dict[str, Any]] = []
    for i, day in enumerate(panel.dates):
        if not start <= day < end:
            continue
        counts['present_rows'] += 1
        reason = None
        if panel.history[i] < HISTORY:
            reason = 'history_excluded'
        elif i+horizon >= len(panel.dates) or panel.block[i] != panel.block[i+horizon]:
            reason = 'label_gap_excluded'
        elif panel.dates[i+horizon] >= end:
            reason = 'boundary_purged'
        if reason is not None:
            counts[reason] += 1
            excluded.append({'decision_at_utc': utc(day), 'reason': reason,
                             'target_end_utc': utc(advance(day, horizon))})
        else:
            selected.append(i)
    indices = np.asarray(selected, dtype=int)
    pair_returns = panel.prices[indices+horizon] / panel.prices[indices] - 1
    if not np.isfinite(pair_returns).all():
        raise InputError('Nonfinite training target', [{'reason': 'nonfinite_target', 'start': str(start), 'end': str(end)}])
    counts['eligible_rows'] = len(indices)
    return Rows(tuple(panel.dates[i] for i in indices), tuple(panel.dates[i+horizon] for i in indices),
                panel.features[indices], pair_returns.mean(axis=1), pair_returns, counts, excluded)


def standardize(features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Estimate population scaling using only supplied training observations.

    Args:
        features: Finite training decisions by five features.

    Returns:
        Training mean and population standard deviation.
    """
    if features.ndim != 2 or features.shape[1] != 5 or len(features) < 2 or not np.isfinite(features).all():
        raise InputError('Invalid training features for standardization', [{'reason': 'training_features'}])
    mean, scale = features.mean(axis=0), features.std(axis=0, ddof=0)
    constant = (np.ptp(features, axis=0) == 0) | (scale == 0)
    if constant.any() or not np.isfinite(mean).all() or not np.isfinite(scale).all():
        rows = [{'feature': FEATURES[i], 'reason': 'constant_training_feature', 'rows': len(features)}
                for i in np.flatnonzero(constant)]
        raise InputError(f'Invalid/constant training columns: {rows}', rows)
    return mean, scale


def fit_candidates(train: Rows, validation: Rows, ledger: Ledger, work_item: str) -> tuple[Model, ...]:
    """Fit exactly three mean-loss candidates without train+validation refitting.

    Args:
        train: Purged training labels and features.
        validation: Purged selection labels and features.
        ledger: Durable reservation sink, invoked before each fit.
        work_item: Fold/horizon identity unique within the attempt.

    Returns:
        Three train-only candidates with validation MSE and covariance.
    """
    if len(train.targets) < 252 or len(validation.targets) < 60:
        raise InputError(f'Insufficient train/validation rows: {len(train.targets)}/{len(validation.targets)}',
                         [{'reason': 'minimum_rows', 'work_item': work_item}])
    mean, scale = standardize(train.features)
    z = (train.features - mean) / scale
    z_mean = z.mean(axis=0)
    centered = z - z_mean
    y_mean = float(train.targets.mean())
    sample = np.cov(train.pair_returns, rowvar=False, ddof=1)
    covariance = 0.9 * sample + 0.1 * np.diag(np.diag(sample)) + 1e-8 * np.eye(9)
    if not np.isfinite(covariance).all():
        raise InputError('Nonfinite train covariance', [{'reason': 'covariance', 'work_item': work_item}])
    candidates = []
    for alpha in ALPHAS:
        identity = f'{work_item}/alpha={alpha}'
        ledger.reserve(identity)
        coefficients = np.linalg.solve(centered.T @ centered + len(z) * alpha * np.eye(5),
                                       centered.T @ (train.targets - y_mean))
        intercept = y_mean - float(z_mean @ coefficients)
        prediction = ((validation.features - mean) / scale) @ coefficients + intercept
        mse = float(np.mean((prediction - validation.targets) ** 2))
        if not np.isfinite(coefficients).all() or not np.isfinite(intercept) or not np.isfinite(mse):
            raise InputError(f'Nonfinite ridge candidate: {identity}', [{'reason': 'nonfinite_model', 'work_item': identity}])
        candidates.append(Model(alpha, mean, scale, coefficients, intercept, covariance, mse))
        ledger.complete(identity)
    return tuple(candidates)


def choose_alpha(candidates: tuple[Model, ...] | list[Model]) -> Model:
    """Return the existing candidate with minimum MSE, stronger alpha on ties.

    Args:
        candidates: The registered three candidates.

    Returns:
        Selected model without any further fitting.
    """
    if tuple(m.alpha for m in candidates) != ALPHAS or not all(np.isfinite(m.validation_mse) for m in candidates):
        raise ValueError('Selection requires the three finite registered alpha candidates')
    best = min(m.validation_mse for m in candidates)
    return max((m for m in candidates if abs(m.validation_mse-best) <= 1e-12), key=lambda m: m.alpha)


def forecast_rows(panel: Panel, model: Model, decisions: tuple[date, ...], horizon: int,
                  cutoff: date, fold: str, model_sha256: str) -> list[dict[str, Any]]:
    """Predict every account decision even when the diagnostic label is absent.

    Args:
        panel: Frozen evaluation panel, ending at the registered last mark.
        model: Previously sealed selected model.
        decisions: Complete ordered account decisions, excluding final mark.
        horizon: Registered one or five business days.
        cutoff: Parameter information cutoff preceding all decisions.
        fold: Registered fold identity.
        model_sha256: Exact model artifact identity.

    Returns:
        Shared fixed/cost forecast rows with separate label coverage.
    """
    if horizon not in (1, 5) or not decisions or tuple(sorted(set(decisions))) != decisions:
        raise InputError('Invalid forecast decision axis or horizon', [{'reason': 'decision_axis'}])
    lookup = {day: i for i, day in enumerate(panel.dates)}
    rows = []
    for day in decisions:
        if day < cutoff:
            raise InputError(f'Forecast cutoff is after decision: {day}', [{'reason': 'cutoff'}])
        if day not in lookup or panel.history[lookup[day]] < HISTORY:
            raise InputError(f'Forecast missing input/history: {day}', [{'reason': 'missing_history', 'decision': str(day)}])
        i = lookup[day]
        target_end = advance(day, horizon)
        available = (target_end in lookup and panel.block[lookup[target_end]] == panel.block[i])
        target = float(np.mean(panel.prices[lookup[target_end]] / panel.prices[i] - 1)) if available else None
        if target is not None and not np.isfinite(target):
            raise InputError(f'Nonfinite evaluation target: {day}', [{'reason': 'target', 'decision': str(day)}])
        rows.append({'fold': fold, 'horizon_business_days': horizon,
                     'decision_at_utc': utc(day), 'information_cutoff_utc': utc(cutoff),
                     'predicted_basket_simple_return': float(model.predict(panel.features[i:i+1])[0]),
                     'diagnostic_target_end_utc': utc(target_end),
                     'diagnostic_target_status': 'available' if available else 'unavailable_gap_or_tail',
                     'diagnostic_target': target, 'model_sha256': model_sha256})
    return rows


def diagnostics(target: np.ndarray, prediction: np.ndarray) -> dict[str, Any]:
    """Describe time-series predictive evidence without independent-pair counts.

    Args:
        target: Available diagnostic basket labels.
        prediction: Predictions at exactly those label timestamps.

    Returns:
        MSE, zero difference, correlation, direction, and variance diagnostics.
    """
    if target.shape != prediction.shape or target.ndim != 1 or not np.isfinite(target).all() or not np.isfinite(prediction).all():
        raise ValueError('Diagnostics require matching finite one-dimensional series')
    if len(target) == 0:
        return {'rows': 0, 'status': 'unavailable_no_labels'}
    mse = float(np.mean((target-prediction)**2))
    zero = float(np.mean(target**2))
    defined = len(target) >= 2 and np.ptp(target) != 0 and np.ptp(prediction) != 0
    return {'rows': len(target), 'status': 'available', 'mse': mse, 'zero_mse': zero,
            'mse_minus_zero': mse-zero,
            'correlation': float(np.corrcoef(target, prediction)[0, 1]) if defined else None,
            'correlation_status': 'defined' if defined else 'undefined_constant_series',
            'direction_agreement': float(np.mean(np.sign(target) == np.sign(prediction))),
            'prediction_variance': float(np.var(prediction, ddof=0))}
