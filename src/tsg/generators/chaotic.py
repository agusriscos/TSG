"""Chaotic regime generators: Lorenz attractor (RK4) and logistic map."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from tsg.generators.base import GeneratedSeries, Regime, SeriesGenerator
from tsg.generators.config import LogisticParams, LorenzParams

logger = logging.getLogger(__name__)

_DEGENERATE_LOGISTIC_X0 = frozenset({0.0, 0.5, 1.0})
_SAFE_LOGISTIC_X0 = 0.1


@dataclass(frozen=True)
class LorenzX(SeriesGenerator):
    """Continuous Lorenz attractor integrated with classical RK4; observes x(t)."""

    params: LorenzParams = LorenzParams()

    @property
    def dataset_id(self) -> str:
        return "lorenz"

    @property
    def regime(self) -> Regime:
        return "chaotic"

    @property
    def variable(self) -> str:
        return "x-value"

    def params_dict(self) -> dict[str, Any]:
        p = self.params
        return {
            "sigma": p.sigma,
            "rho": p.rho,
            "beta": p.beta,
            "dt": p.dt,
            "burn_in": p.burn_in,
            "x0": p.x0,
            "y0": p.y0,
            "z0": p.z0,
        }

    def _deriv(self, state: NDArray[np.floating]) -> NDArray[np.floating]:
        p = self.params
        x, y, z = state
        return np.array(
            [
                p.sigma * (y - x),
                x * (p.rho - z) - y,
                x * y - p.beta * z,
            ],
            dtype=float,
        )

    def generate(
        self,
        n_steps: int,
        *,
        seed: int = 0,
        burn_in: int | None = None,
    ) -> GeneratedSeries:
        p = self.params
        transient = p.burn_in if burn_in is None else burn_in
        total = transient + n_steps
        rng = np.random.default_rng(seed)
        # Perturb ICs with seed so ensemble realizations explore the attractor.
        state = np.array(
            [
                p.x0 + float(rng.normal(0.0, 0.5)),
                p.y0 + float(rng.normal(0.0, 0.5)),
                p.z0 + float(rng.normal(0.0, 0.5)),
            ],
            dtype=float,
        )
        xs = np.empty(total, dtype=float)
        dt = p.dt

        for i in range(total):
            xs[i] = state[0]
            k1 = self._deriv(state)
            k2 = self._deriv(state + 0.5 * dt * k1)
            k3 = self._deriv(state + 0.5 * dt * k2)
            k4 = self._deriv(state + dt * k3)
            state = state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

        values = xs[transient : transient + n_steps]
        return self._finalize(values, seed=seed, burn_in=transient)


def validate_logistic_x0(x0: float) -> float:
    """Reject trivial fixed points and values outside (0, 1); fall back to 0.1."""
    if not (0.0 < x0 < 1.0) or x0 in _DEGENERATE_LOGISTIC_X0:
        logger.warning(
            "Invalid logistic x0=%s; using safe default x0=%s",
            x0,
            _SAFE_LOGISTIC_X0,
        )
        return _SAFE_LOGISTIC_X0
    return float(x0)


@dataclass(frozen=True)
class LogisticMap(SeriesGenerator):
    """Discrete logistic map in fully chaotic regime (r=4)."""

    params: LogisticParams = LogisticParams()

    @property
    def dataset_id(self) -> str:
        return "logistic"

    @property
    def regime(self) -> Regime:
        return "chaotic"

    def params_dict(self) -> dict[str, Any]:
        p = self.params
        return {
            "r": p.r,
            "x0": validate_logistic_x0(p.x0),
            "x0_requested": p.x0,
            "burn_in": p.burn_in,
        }

    def generate(
        self,
        n_steps: int,
        *,
        seed: int = 0,
        burn_in: int | None = None,
    ) -> GeneratedSeries:
        p = self.params
        transient = p.burn_in if burn_in is None else burn_in
        rng = np.random.default_rng(seed)
        # Sample a non-degenerate x0 in (0,1); fall back to params.x0 if seed unused.
        lo, hi = p.x0_range
        x0 = float(rng.uniform(lo, hi))
        x = validate_logistic_x0(x0)
        for _ in range(transient):
            x = p.r * x * (1.0 - x)
        out = np.empty(n_steps, dtype=float)
        for i in range(n_steps):
            out[i] = x
            x = p.r * x * (1.0 - x)
        return self._finalize(out, seed=seed, burn_in=transient)
