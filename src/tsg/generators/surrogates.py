"""Paired surrogate controls for temporal-order / nonlinearity audits.

Implements three classical null models (NumPy/SciPy only), all deterministic
given an injected ``np.random.Generator``:

* **shuffled** — destroy temporal order, keep the exact amplitude multiset.
* **ft** — Fourier phase randomization (preserve linear power spectrum).
* **iaaft** — Iterated Amplitude-Adjusted Fourier Transform (Schreiber & Schmitz),
  jointly preserving the amplitude distribution and the linear spectrum.
"""

from __future__ import annotations

import logging

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)

SurrogateKind = str  # "shuffled" | "ft" | "iaaft" | "original"

# Stopping rule of the corrected IAAFT (not the amplitude norm ``||y||``).
IAAFT_CONVERGENCE_CRITERION = "relative_spectral_error"
IAAFT_DEFAULT_MAX_ITER = 250
IAAFT_DEFAULT_MIN_ITER = 10
IAAFT_DEFAULT_TOL = 1e-6


def shuffled_surrogate(
    x: NDArray[np.floating],
    rng: np.random.Generator,
) -> NDArray[np.floating]:
    """Random permutation of ``x`` (exact marginal, no temporal structure)."""
    values = np.asarray(x, dtype=float).ravel()
    out = values.copy()
    rng.shuffle(out)
    return out


def ft_surrogate(
    x: NDArray[np.floating],
    rng: np.random.Generator,
) -> NDArray[np.floating]:
    """Fourier phase-randomized surrogate (preserve ``|FFT|``, destroy phases).

    Uses the real-FFT Hermitian construction so the inverse transform is real
    and length-matched to ``x``.
    """
    values = np.asarray(x, dtype=float).ravel()
    n = values.size
    if n == 0:
        return values.copy()

    spectrum = np.fft.rfft(values)
    amplitudes = np.abs(spectrum)
    # DC (and Nyquist when present) must stay real for a real-valued inverse.
    n_bins = amplitudes.size
    phases = rng.uniform(0.0, 2.0 * np.pi, size=n_bins)
    phases[0] = 0.0
    if n % 2 == 0 and n_bins > 1:
        phases[-1] = 0.0
    randomized = amplitudes * np.exp(1j * phases)
    surrogate = np.fft.irfft(randomized, n=n)
    return np.asarray(surrogate, dtype=float)


def _spectral_relative_error(
    current: NDArray[np.floating],
    target_spectrum: NDArray[np.floating],
) -> float:
    """Relative L2 error between ``|rFFT(current)|`` and a target amplitude spectrum."""
    amp = np.abs(np.fft.rfft(current))
    denom = float(np.linalg.norm(target_spectrum)) + 1e-15
    return float(np.linalg.norm(amp - target_spectrum) / denom)


def iaaft_surrogate(
    x: NDArray[np.floating],
    rng: np.random.Generator,
    *,
    max_iter: int = IAAFT_DEFAULT_MAX_ITER,
    tol: float = IAAFT_DEFAULT_TOL,
    min_iter: int = IAAFT_DEFAULT_MIN_ITER,
) -> NDArray[np.floating]:
    """Iterated Amplitude-Adjusted Fourier Transform (Schreiber & Schmitz).

    Preserves the exact sorted amplitude distribution of ``x`` and approximately
    the linear power spectrum, while destroying nonlinear phase correlations.

    Convergence is tracked on the **relative spectral error**
    ``|| |rFFT(y)| - |rFFT(x)| ||_2 / || |rFFT(x)| ||_2``, not on ``||y||_2``.
    The latter is invariant under the rank-order step (same amplitude multiset)
    and would stop after a single iteration, leaving a large spectral mismatch
    that creates false positives for linear Gaussian processes such as ARMA.

    Parameters
    ----------
    x
        Input series.
    rng
        NumPy Generator (reproducibility).
    max_iter
        Maximum alternating spectrum / rank iterations.
    tol
        Stop when the spectral relative error falls below ``tol``, or when the
        absolute change in that error falls below ``tol`` (plateau), after
        ``min_iter`` iterations.
    min_iter
        Minimum iterations before plateau / tolerance stopping is allowed.
    """
    values = np.asarray(x, dtype=float).ravel()
    n = values.size
    if n == 0:
        return values.copy()

    sorted_amps = np.sort(values)
    target_spectrum = np.abs(np.fft.rfft(values))

    # Start from a random shuffle (amplitude-exact); then iterate.
    y = shuffled_surrogate(values, rng)
    prev_err = np.inf
    err = np.inf
    it_done = 0

    for it in range(max_iter):
        # 1) Impose target power spectrum (keep current phases).
        y_fft = np.fft.rfft(y)
        phases = np.angle(y_fft)
        # Protect DC / Nyquist reality.
        phases[0] = 0.0
        if n % 2 == 0 and phases.size > 1:
            phases[-1] = 0.0
        y = np.fft.irfft(target_spectrum * np.exp(1j * phases), n=n)

        # 2) Rank-order replace to restore exact amplitude distribution.
        ranks = np.argsort(np.argsort(y))
        y = sorted_amps[ranks]

        err = _spectral_relative_error(y, target_spectrum)
        it_done = it + 1
        delta = abs(prev_err - err)
        prev_err = err

        if it_done >= min_iter and (err < tol or delta < tol):
            logger.debug(
                "IAAFT converged at iter=%d spectral_rel_err=%.3e delta=%.3e",
                it_done,
                err,
                delta,
            )
            break
    else:
        logger.debug(
            "IAAFT reached max_iter=%d spectral_rel_err=%.3e",
            max_iter,
            err,
        )

    return np.asarray(y, dtype=float)


def make_surrogate(
    x: NDArray[np.floating],
    kind: SurrogateKind,
    rng: np.random.Generator,
    *,
    max_iter: int = IAAFT_DEFAULT_MAX_ITER,
    tol: float = IAAFT_DEFAULT_TOL,
    min_iter: int = IAAFT_DEFAULT_MIN_ITER,
) -> NDArray[np.floating]:
    """Dispatch helper for ``original`` / ``shuffled`` / ``ft`` / ``iaaft``.

    ``min_iter`` applies only to IAAFT (spectral-error stopping).
    """
    if kind == "original":
        return np.asarray(x, dtype=float).ravel().copy()
    if kind == "shuffled":
        return shuffled_surrogate(x, rng)
    if kind == "ft":
        return ft_surrogate(x, rng)
    if kind == "iaaft":
        return iaaft_surrogate(
            x,
            rng,
            max_iter=max_iter,
            tol=tol,
            min_iter=min_iter,
        )
    raise ValueError(f"Unknown surrogate kind: {kind!r}")
