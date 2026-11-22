"""Stochastic regime generators: white noise, Brownian motion, fBm and ARMA."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

from tsg.generators.base import GeneratedSeries, Regime, SeriesGenerator
from tsg.generators.config import (
    ARMAParams,
    BrownianParams,
    FractionalBrownianParams,
    WhiteNoiseParams,
)

logger = logging.getLogger(__name__)

FbmVariant = Literal["antipersistent", "persistent"]


@dataclass(frozen=True)
class WhiteNoise(SeriesGenerator):
    """Uncorrelated Gaussian control: N(0, sigma^2)."""

    params: WhiteNoiseParams = WhiteNoiseParams()

    @property
    def dataset_id(self) -> str:
        return "white_noise"

    @property
    def regime(self) -> Regime:
        return "stochastic"

    def params_dict(self) -> dict[str, Any]:
        return {"sigma": self.params.sigma}

    def generate(
        self,
        n_steps: int,
        *,
        seed: int = 0,
        burn_in: int | None = None,
    ) -> GeneratedSeries:
        del burn_in  # white noise has no transient dynamics
        rng = np.random.default_rng(seed)
        values = rng.normal(0.0, self.params.sigma, size=n_steps)
        return self._finalize(values, seed=seed, burn_in=0)


@dataclass(frozen=True)
class BrownianMotion(SeriesGenerator):
    """Non-stationary random walk: cumulative sum of N(0, sigma^2) increments."""

    params: BrownianParams = BrownianParams()

    @property
    def dataset_id(self) -> str:
        return "brownian"

    @property
    def regime(self) -> Regime:
        return "stochastic"

    def params_dict(self) -> dict[str, Any]:
        return {"sigma": self.params.sigma}

    def generate(
        self,
        n_steps: int,
        *,
        seed: int = 0,
        burn_in: int | None = None,
    ) -> GeneratedSeries:
        del burn_in
        rng = np.random.default_rng(seed)
        increments = rng.normal(0.0, self.params.sigma, size=n_steps)
        values = np.cumsum(increments)
        return self._finalize(values, seed=seed, burn_in=0)


def _fgn_acf(lags: NDArray[np.floating], hurst: float) -> NDArray[np.floating]:
    """Autocovariance of unit-variance fractional Gaussian noise."""
    k = np.asarray(lags, dtype=float)
    two_h = 2.0 * hurst
    return 0.5 * (
        np.abs(k + 1.0) ** two_h + np.abs(k - 1.0) ** two_h - 2.0 * np.abs(k) ** two_h
    )


def _fgn_from_embedding(
    eigenvalues: NDArray[np.floating],
    rng: np.random.Generator,
) -> NDArray[np.floating]:
    """Draw a stationary series from a circulant embedding (length = embedding)."""
    m = int(eigenvalues.size)
    half = m // 2
    z_real = rng.normal(size=m)
    z_imag = rng.normal(size=m)
    z = np.zeros(m, dtype=np.complex128)
    z[0] = z_real[0]
    z[half] = z_real[half]
    z[1:half] = (z_real[1:half] + 1j * z_imag[1:half]) / np.sqrt(2.0)
    z[half + 1 :] = np.conj(z[1:half][::-1])
    spectrum = np.sqrt(np.maximum(eigenvalues, 0.0)) * z
    return np.fft.ifft(spectrum).real * np.sqrt(m)


def _fgn_cholesky(
    n: int,
    hurst: float,
    rng: np.random.Generator,
) -> NDArray[np.floating]:
    """Exact fGn via a Toeplitz Cholesky factor (fallback for small ``n``)."""
    cov = _fgn_acf(np.arange(n, dtype=float), hurst)
    idx = np.arange(n)
    toeplitz = cov[np.abs(idx[:, None] - idx[None, :])]
    jitter = 1e-10
    chol: NDArray[np.floating] | None = None
    for _ in range(6):
        try:
            chol = np.linalg.cholesky(toeplitz + jitter * np.eye(n))
            break
        except np.linalg.LinAlgError:
            jitter *= 10.0
    if chol is None:
        raise RuntimeError(
            f"Fractional Gaussian noise covariance is not PSD (n={n}, H={hurst})"
        )
    return chol @ rng.normal(size=n)


def fractional_gaussian_noise(
    n: int,
    hurst: float,
    rng: np.random.Generator,
) -> NDArray[np.floating]:
    """Simulate unit-variance fGn with the Davies–Harte method.

    The circulant embedding has size ``m = 2(n-1)`` and is diagonalized by an
    FFT, so one series costs ``O(n log n)``. If roundoff makes the embedding
    indefinite the size is doubled (Wood–Chan); a Cholesky factorization is
    used only when that still fails.

    Parameters
    ----------
    n
        Number of increments.
    hurst
        Hurst exponent in ``(0, 1)``.
    rng
        Explicit NumPy generator. The draw is fully determined by ``rng``.
    """
    if n < 1:
        raise ValueError(f"n must be positive; got {n}")
    if not 0.0 < hurst < 1.0:
        raise ValueError(f"Hurst exponent must lie in (0, 1); got {hurst}")
    if n == 1:
        return rng.normal(size=1)

    m = 2 * (n - 1)
    max_m = max(m, 16 * n)
    eigenvalues: NDArray[np.floating] | None = None
    while m <= max_m:
        lags = np.minimum(np.arange(m), m - np.arange(m)).astype(float)
        circulant = _fgn_acf(lags, hurst)
        eigenvalues = np.fft.fft(circulant).real
        scale = max(1.0, float(np.max(np.abs(eigenvalues))))
        if float(np.min(eigenvalues)) >= -1e-8 * scale:
            eigenvalues = np.maximum(eigenvalues, 0.0)
            break
        logger.warning(
            "Davies–Harte embedding not PSD at m=%d (min eig=%g, H=%g); enlarging",
            m,
            float(np.min(eigenvalues)),
            hurst,
        )
        m *= 2
        eigenvalues = None
    if eigenvalues is None:
        logger.warning(
            "Davies–Harte embedding failed for n=%d H=%g; using Cholesky",
            n,
            hurst,
        )
        return _fgn_cholesky(n, hurst, rng)
    return _fgn_from_embedding(eigenvalues, rng)[:n]


@dataclass(frozen=True)
class FractionalBrownianMotion(SeriesGenerator):
    """Fractional Brownian motion via cumulative Davies–Harte fGn.

    ``variant='antipersistent'`` samples ``H ~ U(0.15, 0.25)`` (negative lag-1
    increment correlation). ``variant='persistent'`` samples ``H ~ U(0.75, 0.85)``.
    The returned series is the integrated path (levels), matching ``brownian``.
    """

    params: FractionalBrownianParams = FractionalBrownianParams()
    variant: FbmVariant = "persistent"

    def __post_init__(self) -> None:
        if self.variant not in ("antipersistent", "persistent"):
            raise ValueError(
                f"Unknown fBm variant {self.variant!r}; "
                "expected 'antipersistent' or 'persistent'"
            )

    @property
    def dataset_id(self) -> str:
        return f"fbm_{self.variant}"

    @property
    def regime(self) -> Regime:
        return "stochastic"

    def params_dict(self) -> dict[str, Any]:
        p = self.params
        window = (
            p.hurst_antipersistent_range
            if self.variant == "antipersistent"
            else p.hurst_persistent_range
        )
        return {
            "variant": self.variant,
            "sigma": p.sigma,
            "hurst_range": list(window),
            "method": "davies_harte",
        }

    def generate(
        self,
        n_steps: int,
        *,
        seed: int = 0,
        burn_in: int | None = None,
    ) -> GeneratedSeries:
        del burn_in  # fBm increments are stationary; the path has no transient
        if self.params.sigma <= 0.0:
            raise ValueError(f"fBm sigma must be positive; got {self.params.sigma}")
        rng = np.random.default_rng(seed)
        window = (
            self.params.hurst_antipersistent_range
            if self.variant == "antipersistent"
            else self.params.hurst_persistent_range
        )
        hurst = float(rng.uniform(*window))
        increments = fractional_gaussian_noise(n_steps, hurst, rng)
        values = self.params.sigma * np.cumsum(increments)
        used = {
            "variant": self.variant,
            "sigma": float(self.params.sigma),
            "hurst": hurst,
            "hurst_range": list(window),
            "method": "davies_harte",
        }
        return self._finalize(values, seed=seed, burn_in=0, params=used)


@dataclass(frozen=True)
class ARMACorrelated(SeriesGenerator):
    """Stationary ARMA(2, 1) with strong positive autocorrelation.

    Each realization draws a modulus ``rho``, a frequency ``omega`` and an MA
    coefficient ``theta``. The AR polynomial is
    ``1 - 2 rho cos(omega) z + rho^2 z^2``, so both roots lie on the circle of
    radius ``rho < 1`` and the process is stationary. A long burn-in is
    discarded so the retained samples sit on the stationary law.
    """

    params: ARMAParams = ARMAParams()

    @property
    def dataset_id(self) -> str:
        return "arma_correlated"

    @property
    def regime(self) -> Regime:
        return "stochastic"

    def params_dict(self) -> dict[str, Any]:
        p = self.params
        return {
            "sigma": p.sigma,
            "rho_range": list(p.rho_range),
            "omega_range": list(p.omega_range),
            "theta_range": list(p.theta_range),
            "burn_in_range": list(p.burn_in_range),
            "model": "ARMA(2,1)",
        }

    def generate(
        self,
        n_steps: int,
        *,
        seed: int = 0,
        burn_in: int | None = None,
    ) -> GeneratedSeries:
        if n_steps < 1:
            raise ValueError(f"n_steps must be positive; got {n_steps}")
        p = self.params
        if p.sigma <= 0.0:
            raise ValueError(f"ARMA sigma must be positive; got {p.sigma}")
        rng = np.random.default_rng(seed)
        rho = float(rng.uniform(*p.rho_range))
        omega = float(rng.uniform(*p.omega_range))
        theta = float(rng.uniform(*p.theta_range))
        if burn_in is None:
            transient = int(rng.integers(p.burn_in_range[0], p.burn_in_range[1]))
        else:
            transient = int(burn_in)
        phi1 = float(2.0 * rho * np.cos(omega))
        phi2 = float(-(rho**2))
        # Stationarity of a real AR(2): |phi2| < 1, phi1 + phi2 < 1, phi2 - phi1 < 1.
        if abs(phi2) >= 1.0 or (phi1 + phi2) >= 1.0 or (phi2 - phi1) >= 1.0:
            raise RuntimeError(
                f"Sampled AR(2) is not stationary (phi1={phi1}, phi2={phi2}, rho={rho})"
            )
        total = transient + n_steps
        innovations = rng.normal(0.0, p.sigma, size=total)
        values_full = np.empty(total, dtype=float)
        x1 = 0.0
        x2 = 0.0
        e1 = 0.0
        for t in range(total):
            innov = float(innovations[t])
            xt = phi1 * x1 + phi2 * x2 + innov + theta * e1
            values_full[t] = xt
            x2 = x1
            x1 = xt
            e1 = innov
        values = values_full[transient:]
        if not np.isfinite(values).all():
            raise RuntimeError("ARMA realization produced non-finite values")
        used = {
            "sigma": float(p.sigma),
            "rho": rho,
            "omega": omega,
            "theta": theta,
            "phi1": phi1,
            "phi2": phi2,
            "burn_in": transient,
            "model": "ARMA(2,1)",
            "rho_range": list(p.rho_range),
            "omega_range": list(p.omega_range),
            "theta_range": list(p.theta_range),
        }
        return self._finalize(values, seed=seed, burn_in=transient, params=used)
