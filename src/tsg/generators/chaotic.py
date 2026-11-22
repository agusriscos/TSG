"""Chaotic regime generators: Lorenz attractor (RK4) and logistic map."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from tsg.generators.base import GeneratedSeries, Regime, SeriesGenerator
from tsg.generators.config import (
    HenonParams,
    LogisticParams,
    LogisticWeakParams,
    LorenzParams,
    RosslerParams,
)

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
            "sigma_range": list(p.sigma_range),
            "rho_range": list(p.rho_range),
        }

    def _deriv(
        self,
        state: NDArray[np.floating],
        *,
        sigma: float,
        rho: float,
        beta: float,
    ) -> NDArray[np.floating]:
        x, y, z = state
        return np.array(
            [
                sigma * (y - x),
                x * (rho - z) - y,
                x * y - beta * z,
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
        # Safe parametric jitter inside the classical chaotic window; beta fixed.
        sigma = float(rng.uniform(*p.sigma_range))
        rho = float(rng.uniform(*p.rho_range))
        beta = float(p.beta)
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
            k1 = self._deriv(state, sigma=sigma, rho=rho, beta=beta)
            k2 = self._deriv(state + 0.5 * dt * k1, sigma=sigma, rho=rho, beta=beta)
            k3 = self._deriv(state + 0.5 * dt * k2, sigma=sigma, rho=rho, beta=beta)
            k4 = self._deriv(state + dt * k3, sigma=sigma, rho=rho, beta=beta)
            state = state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

        values = xs[transient : transient + n_steps]
        used = {
            "sigma": sigma,
            "rho": rho,
            "beta": beta,
            "dt": float(p.dt),
            "burn_in": transient,
            "x0": float(p.x0),
            "y0": float(p.y0),
            "z0": float(p.z0),
            "ic_jittered": True,
            "sigma_range": list(p.sigma_range),
            "rho_range": list(p.rho_range),
        }
        return self._finalize(values, seed=seed, burn_in=transient, params=used)


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
            "burn_in_range": list(p.burn_in_range),
        }

    def generate(
        self,
        n_steps: int,
        *,
        seed: int = 0,
        burn_in: int | None = None,
    ) -> GeneratedSeries:
        p = self.params
        if p.r != 4.0:
            raise ValueError(
                f"LogisticMap requires r=4.0 (fully chaotic); got r={p.r}. "
                "Do not perturb r — use x0 + random burn_in instead."
            )
        rng = np.random.default_rng(seed)
        if burn_in is None:
            lo, hi = p.burn_in_range
            transient = int(rng.integers(lo, hi))
        else:
            transient = int(burn_in)
        lo, hi = p.x0_range
        x0 = float(rng.uniform(lo, hi))
        x = validate_logistic_x0(x0)
        for _ in range(transient):
            x = p.r * x * (1.0 - x)
        out = np.empty(n_steps, dtype=float)
        for i in range(n_steps):
            out[i] = x
            x = p.r * x * (1.0 - x)
        used = {
            "r": float(p.r),
            "x0": float(x0),
            "x0_requested": float(x0),
            "burn_in": transient,
            "burn_in_range": list(p.burn_in_range),
        }
        return self._finalize(out, seed=seed, burn_in=transient, params=used)


def _rk4_step(
    state: NDArray[np.floating],
    deriv,
    dt: float,
) -> NDArray[np.floating]:
    """One classical RK4 step for an autonomous 3D vector field."""
    k1 = deriv(state)
    k2 = deriv(state + 0.5 * dt * k1)
    k3 = deriv(state + 0.5 * dt * k2)
    k4 = deriv(state + dt * k3)
    return state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


@dataclass(frozen=True)
class Rossler(SeriesGenerator):
    """Rössler flow in the classical chaotic window; observes ``x(t)``.

    Parameters are drawn once per realization
    (``a ~ U(0.18, 0.22)``, ``b ~ U(0.18, 0.22)``, ``c ~ U(5.5, 5.9)``).
    Initial conditions are perturbed around ``(x0, y0, z0)`` and a transient
    is discarded before the retained samples.
    """

    params: RosslerParams = RosslerParams()

    @property
    def dataset_id(self) -> str:
        return "rossler"

    @property
    def regime(self) -> Regime:
        return "chaotic"

    @property
    def variable(self) -> str:
        return "x-value"

    def params_dict(self) -> dict[str, Any]:
        p = self.params
        return {
            "a_range": list(p.a_range),
            "b_range": list(p.b_range),
            "c_range": list(p.c_range),
            "dt": p.dt,
            "burn_in": p.burn_in,
            "x0": p.x0,
            "y0": p.y0,
            "z0": p.z0,
        }

    @staticmethod
    def _deriv(
        state: NDArray[np.floating],
        *,
        a: float,
        b: float,
        c: float,
    ) -> NDArray[np.floating]:
        x, y, z = state
        return np.array(
            [-y - z, x + a * y, b + z * (x - c)],
            dtype=float,
        )

    def _integrate(
        self,
        *,
        n_steps: int,
        transient: int,
        a: float,
        b: float,
        c: float,
        state: NDArray[np.floating],
    ) -> NDArray[np.floating]:
        p = self.params
        total = transient + n_steps
        xs = np.empty(total, dtype=float)
        dt = float(p.dt)

        def deriv(s: NDArray[np.floating]) -> NDArray[np.floating]:
            return self._deriv(s, a=a, b=b, c=c)

        for i in range(total):
            xs[i] = state[0]
            state = _rk4_step(state, deriv, dt)
            if not np.isfinite(state).all() or float(np.max(np.abs(state))) > 1e6:
                raise RuntimeError("Rössler trajectory diverged")
        return xs[transient : transient + n_steps]

    def generate(
        self,
        n_steps: int,
        *,
        seed: int = 0,
        burn_in: int | None = None,
    ) -> GeneratedSeries:
        p = self.params
        if p.dt <= 0.0:
            raise ValueError(f"Rössler dt must be positive; got {p.dt}")
        transient = p.burn_in if burn_in is None else int(burn_in)
        rng = np.random.default_rng(seed)
        a = float(rng.uniform(*p.a_range))
        b = float(rng.uniform(*p.b_range))
        c = float(rng.uniform(*p.c_range))
        jitter = rng.normal(0.0, p.ic_sigma, size=3)
        state = np.array([p.x0, p.y0, p.z0], dtype=float) + jitter
        try:
            values = self._integrate(
                n_steps=n_steps,
                transient=transient,
                a=a,
                b=b,
                c=c,
                state=state,
            )
            ic_fallback = False
        except RuntimeError:
            logger.warning(
                "Rössler diverged from perturbed IC (seed=%d); restarting from (x0, y0, z0)",
                seed,
            )
            values = self._integrate(
                n_steps=n_steps,
                transient=transient,
                a=a,
                b=b,
                c=c,
                state=np.array([p.x0, p.y0, p.z0], dtype=float),
            )
            ic_fallback = True
        used = {
            "a": a,
            "b": b,
            "c": c,
            "dt": float(p.dt),
            "burn_in": transient,
            "x0": float(p.x0),
            "y0": float(p.y0),
            "z0": float(p.z0),
            "ic_sigma": float(p.ic_sigma),
            "ic_fallback": ic_fallback,
            "a_range": list(p.a_range),
            "b_range": list(p.b_range),
            "c_range": list(p.c_range),
        }
        return self._finalize(values, seed=seed, burn_in=transient, params=used)


@dataclass(frozen=True)
class HenonMap(SeriesGenerator):
    """Hénon map ``(x, y) -> (1 - a x^2 + y, b x)``; observes ``x``.

    ``a`` is drawn from ``U(1.38, 1.41)`` and ``b`` stays at 0.3. Initial
    conditions are perturbed inside the basin. A divergent orbit is restarted
    from the nominal ``(x0, y0)`` with the same ``a`` so the draw stays
    reproducible.
    """

    params: HenonParams = HenonParams()

    @property
    def dataset_id(self) -> str:
        return "henon"

    @property
    def regime(self) -> Regime:
        return "chaotic"

    @property
    def variable(self) -> str:
        return "x-value"

    def params_dict(self) -> dict[str, Any]:
        p = self.params
        return {
            "a_range": list(p.a_range),
            "b": p.b,
            "burn_in": p.burn_in,
            "x0": p.x0,
            "y0": p.y0,
            "ic_sigma": p.ic_sigma,
        }

    def _roll(
        self,
        x: float,
        y: float,
        *,
        a: float,
        b: float,
        steps: int,
    ) -> tuple[float, float, bool]:
        bound = self.params.bound
        for _ in range(steps):
            x, y = 1.0 - a * x * x + y, b * x
            if not np.isfinite(x) or abs(x) > bound or abs(y) > bound:
                return x, y, False
        return x, y, True

    def generate(
        self,
        n_steps: int,
        *,
        seed: int = 0,
        burn_in: int | None = None,
    ) -> GeneratedSeries:
        p = self.params
        transient = p.burn_in if burn_in is None else int(burn_in)
        rng = np.random.default_rng(seed)
        a = float(rng.uniform(*p.a_range))
        b = float(p.b)
        x = float(p.x0 + rng.normal(0.0, p.ic_sigma))
        y = float(p.y0 + rng.normal(0.0, p.ic_sigma))
        x, y, ok = self._roll(x, y, a=a, b=b, steps=transient)
        ic_fallback = False
        if not ok:
            logger.warning(
                "Hénon left the basin (seed=%d, a=%.4f); restarting from (x0, y0)",
                seed,
                a,
            )
            ic_fallback = True
            x, y = float(p.x0), float(p.y0)
            x, y, ok = self._roll(x, y, a=a, b=b, steps=transient)
            if not ok:
                raise RuntimeError(
                    f"Hénon diverged from the nominal IC (a={a}, b={b})"
                )
        out = np.empty(n_steps, dtype=float)
        for i in range(n_steps):
            out[i] = x
            x, y, ok = self._roll(x, y, a=a, b=b, steps=1)
            if not ok:
                raise RuntimeError(
                    f"Hénon diverged while recording (a={a}, b={b}, step={i})"
                )
        used = {
            "a": a,
            "b": b,
            "burn_in": transient,
            "x0": float(p.x0),
            "y0": float(p.y0),
            "ic_sigma": float(p.ic_sigma),
            "ic_fallback": ic_fallback,
            "a_range": list(p.a_range),
        }
        return self._finalize(out, seed=seed, burn_in=transient, params=used)


def _logistic_lyapunov(r: float, orbit: NDArray[np.floating]) -> float:
    """Finite-time Lyapunov exponent of the logistic map along ``orbit``."""
    deriv = np.abs(r - 2.0 * r * orbit)
    return float(np.mean(np.log(np.maximum(deriv, 1e-15))))


def _logistic_orbit(
    r: float,
    x0: float,
    *,
    transient: int,
    n_steps: int,
) -> NDArray[np.floating]:
    x = float(x0)
    for _ in range(transient):
        x = r * x * (1.0 - x)
    out = np.empty(n_steps, dtype=float)
    for i in range(n_steps):
        out[i] = x
        x = r * x * (1.0 - x)
    return out


@dataclass(frozen=True)
class LogisticWeak(SeriesGenerator):
    """Logistic map in the weak-chaos band ``r ~ U(3.58, 3.65)``.

    Periodic windows inside that interval (including the period-6 window near
    ``r ≈ 3.63``) are rejected: a candidate ``r`` is kept only when a
    finite-time Lyapunov probe is strictly positive. If every draw fails, the
    generator falls back to ``r_fallback`` (default 3.60), which is chaotic.
    """

    params: LogisticWeakParams = LogisticWeakParams()

    @property
    def dataset_id(self) -> str:
        return "logistic_weak"

    @property
    def regime(self) -> Regime:
        return "chaotic"

    def params_dict(self) -> dict[str, Any]:
        p = self.params
        return {
            "r_range": list(p.r_range),
            "r_fallback": p.r_fallback,
            "x0_range": list(p.x0_range),
            "burn_in_range": list(p.burn_in_range),
            "lyapunov_probe": p.lyapunov_probe,
        }

    def _accepts(self, r: float, x0: float) -> bool:
        probe = _logistic_orbit(
            r,
            x0,
            transient=200,
            n_steps=int(self.params.lyapunov_probe),
        )
        if not np.isfinite(probe).all() or np.any((probe <= 0.0) | (probe >= 1.0)):
            return False
        return _logistic_lyapunov(r, probe) > 0.0

    def generate(
        self,
        n_steps: int,
        *,
        seed: int = 0,
        burn_in: int | None = None,
    ) -> GeneratedSeries:
        p = self.params
        rng = np.random.default_rng(seed)
        if burn_in is None:
            transient = int(rng.integers(p.burn_in_range[0], p.burn_in_range[1]))
        else:
            transient = int(burn_in)
        lo, hi = p.x0_range
        r_lo, r_hi = p.r_range
        r = float(p.r_fallback)
        x0 = float(p.x0)
        accepted = False
        for _ in range(int(p.max_attempts)):
            r_try = float(rng.uniform(r_lo, r_hi))
            x0_try = validate_logistic_x0(float(rng.uniform(lo, hi)))
            if self._accepts(r_try, x0_try):
                r = r_try
                x0 = x0_try
                accepted = True
                break
        if not accepted:
            logger.warning(
                "No chaotic logistic_weak draw in %d attempts (seed=%d); using r=%s",
                p.max_attempts,
                seed,
                p.r_fallback,
            )
            r = float(p.r_fallback)
            x0 = validate_logistic_x0(float(p.x0))
        out = _logistic_orbit(r, x0, transient=transient, n_steps=n_steps)
        used = {
            "r": float(r),
            "x0": float(x0),
            "burn_in": transient,
            "window_rejected": not accepted,
            "r_range": list(p.r_range),
            "r_fallback": float(p.r_fallback),
            "burn_in_range": list(p.burn_in_range),
        }
        return self._finalize(out, seed=seed, burn_in=transient, params=used)
