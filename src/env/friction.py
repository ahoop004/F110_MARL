"""Explicit per-episode grip protocols, independent of spawn and sensor RNGs."""
from copy import deepcopy
from collections.abc import Mapping

import numpy as np

FRICTION_STREAM = 0x46524943  # Stable named SeedSequence stream ('FRIC').


def copy_friction_metadata(metadata: dict) -> dict:
    """Detach the validated sample schema for an independently mutable info.

    All sample fields are scalars except the protocol dict and its optional
    grid-values list. Copy those containers without per-scalar deepcopy work.
    """
    result = metadata.copy()
    result['protocol'] = metadata['protocol'].copy()
    if 'values' in result['protocol']:
        result['protocol']['values'] = result['protocol']['values'].copy()
    return result


def validate_friction_protocol(config, *, nonlinear: bool) -> dict | None:
    if config is None:
        return None
    if not nonlinear:
        raise ValueError('friction requires combined_slip_st physics')
    if (not isinstance(config, Mapping) or set(config) != {'version', 'scope', 'train', 'eval'}
            or type(config['version']) is not int or config['version'] != 1
            or config['scope'] != 'shared'):
        raise ValueError('friction requires version: 1, scope: shared, and explicit train/eval protocols')
    result = deepcopy(dict(config))
    def number(value):
        if isinstance(value, (bool, np.bool_, str)) or not np.isscalar(value):
            raise ValueError('friction coefficients must be finite nonnegative numbers')
        try:
            value = float(value)
        except (ValueError, TypeError, OverflowError) as exc:
            raise ValueError('friction coefficient is invalid') from exc
        if not np.isfinite(value) or value < 0:
            raise ValueError('friction coefficients must be finite and nonnegative')
        return value
    for phase in ('train', 'eval'):
        spec = result[phase]
        if not isinstance(spec, Mapping):
            raise ValueError('friction phase must be a mapping')
        mode = spec.get('mode')
        fields = {'fixed': {'mode', 'mu'}, 'uniform': {'mode', 'low', 'high'},
                  'grid': {'mode', 'values'}, 'gaussian': {'mode', 'relative_std'}}
        if not isinstance(mode, str) or mode not in fields or set(spec) != fields[mode]:
            raise ValueError('friction mode/fields must describe fixed, uniform, grid, or gaussian')
        if phase == 'eval' and mode in {'uniform', 'gaussian'}:
            raise ValueError('Evaluation friction must be fixed or a deterministic grid')
        if mode == 'fixed':
            result[phase] = {'mode': mode, 'mu': number(spec['mu'])}
        elif mode == 'gaussian':
            result[phase] = {'mode': mode, 'relative_std': number(spec['relative_std'])}
        elif mode == 'uniform':
            lo, hi = number(spec['low']), number(spec['high'])
            if lo >= hi:
                raise ValueError('friction uniform low must be less than high')
            result[phase] = {'mode': mode, 'low': lo, 'high': hi}
        else:
            if not isinstance(spec['values'], list) or not spec['values']:
                raise ValueError('friction grid values must be a nonempty list')
            result[phase] = {'mode': mode, 'values': [number(v) for v in spec['values']]}
    return result


class EpisodeFriction:
    def __init__(self, config, *, nominal_mu: float, seed: int, phase: str):
        if phase not in {'train', 'eval'}:
            raise ValueError('physics phase must be train or eval')
        self.config = validate_friction_protocol(config, nonlinear=True)
        self.nominal_mu = float(nominal_mu)
        self.phase = phase
        self.reseed(seed)

    def reseed(self, seed: int) -> None:
        self.seed = int(seed)
        self.draw = 0
        self.rng = np.random.default_rng(np.random.SeedSequence([self.seed, FRICTION_STREAM]))

    def sample(self) -> dict:
        spec = self.config[self.phase] if self.config else {'mode': 'fixed', 'mu': self.nominal_mu}
        mode = spec['mode']
        grid_index = None
        if mode == 'uniform':
            mu = float(self.rng.uniform(spec['low'], spec['high']))
        elif mode == 'gaussian':
            mu = float(self.nominal_mu * self.rng.normal(1.0, spec['relative_std']))
            if not np.isfinite(mu) or mu < 0:
                raise ValueError('Gaussian friction draw is negative/nonfinite; check relative_std')
        elif mode == 'grid':
            # Consecutive evaluation seeds cover the grid; reset(None) advances it.
            grid_index = (self.seed + self.draw) % len(spec['values'])
            mu = spec['values'][grid_index]
        else:
            mu = spec['mu']
        metadata = {'version': 1, 'scope': 'shared', 'phase': self.phase,
                    'nominal_mu': self.nominal_mu, 'mu': mu, 'protocol': deepcopy(spec),
                    'seed': self.seed, 'stream': FRICTION_STREAM, 'draw': self.draw,
                    'grid_index': grid_index}
        self.draw += 1
        return metadata
