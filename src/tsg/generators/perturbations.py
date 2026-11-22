"""Deterministic signal transforms and perturbations for invariance audits.

All stochastic transforms take an injected ``np.random.Generator`` so every
run is reproducible given a seed. Pure geometric / amplitude maps (affine,
monotone, quantize, downsample) are fully deterministic without RNG.
"""

from __future__ import annotations

import logging

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)

TransformName = str  # affine | monotone | outliers | quantize | downsample | snr_*


def affine_transform(
    x: NDArray[np.floating],
    *,
    scale: float = 3.5,
    shift: float = 10.0,
) -> NDArray[np.floating]:
    """Affine map ``y = a·x + b`` (scale + shift invariance probe)."""
    values = np.asarray(x, dtype=float).ravel()
    return np.asarray(scale * values + shift, dtype=float)


def monotone_nonlinear(x: NDArray[np.floating]) -> NDArray[np.floating]:
    """Strictly increasing nonlinear map after z-score standardisation.

    Uses ``y = sinh(1.5 · z)``. Ordinal / horizontal-visibility structure
    (``HVG``, ``OPN``) should stay invariant; natural visibility and recurrence
    graphs generally change.
    """
    values = np.asarray(x, dtype=float).ravel()
    if values.size == 0:
        return values.copy()
    mu = float(np.mean(values))
    sigma = float(np.std(values))
    if sigma < 1e-15:
        z = values - mu
    else:
        z = (values - mu) / sigma
    return np.asarray(np.sinh(1.5 * z), dtype=float)


def impulsive_outliers(
    x: NDArray[np.floating],
    rng: np.random.Generator,
    *,
    rate: float = 0.02,
    magnitude: float = 6.0,
) -> NDArray[np.floating]:
    """Inject impulsive spikes of amplitude ``±magnitude·σ`` at fraction ``rate``."""
    values = np.asarray(x, dtype=float).ravel().copy()
    n = values.size
    if n == 0 or rate <= 0.0:
        return values
    n_spikes = max(1, int(round(rate * n)))
    n_spikes = min(n_spikes, n)
    idx = rng.choice(n, size=n_spikes, replace=False)
    sigma = float(np.std(values))
    if sigma < 1e-15:
        sigma = 1.0
    signs = rng.choice(np.asarray([-1.0, 1.0]), size=n_spikes)
    values[idx] = values[idx] + signs * magnitude * sigma
    return values


def quantize_ties(
    x: NDArray[np.floating],
    *,
    n_levels: int = 8,
) -> NDArray[np.floating]:
    """Uniform quantisation to at most ``n_levels`` distinct amplitudes.

    Introduces massive ties that stress ordinal / visibility constructions
    (``OPN``, ``HVG``).
    """
    values = np.asarray(x, dtype=float).ravel()
    if values.size == 0:
        return values.copy()
    if n_levels < 2:
        raise ValueError(f"n_levels must be >= 2, got {n_levels}")
    vmin = float(np.min(values))
    vmax = float(np.max(values))
    if abs(vmax - vmin) < 1e-15:
        return np.full_like(values, vmin, dtype=float)
    # Map to integer bins [0, n_levels-1], then back to bin centres.
    scaled = (values - vmin) / (vmax - vmin)
    bins = np.floor(scaled * n_levels).astype(int)
    bins = np.clip(bins, 0, n_levels - 1)
    centres = vmin + (bins.astype(float) + 0.5) * (vmax - vmin) / n_levels
    return np.asarray(centres, dtype=float)


def downsample_series(
    x: NDArray[np.floating],
    *,
    factor: int = 2,
) -> NDArray[np.floating]:
    """Subsample by ``factor`` then linearly interpolate back to length ``N``.

    Keeping the original length aligns feature rows for OOD classification
    (train on clean, evaluate on transformed). The intermediate decimation
    removes high-frequency content that the interpolation cannot restore.
    """
    values = np.asarray(x, dtype=float).ravel()
    n = values.size
    if n == 0:
        return values.copy()
    if factor < 2:
        raise ValueError(f"factor must be >= 2, got {factor}")
    coarse = values[::factor]
    if coarse.size < 2:
        return np.full(n, float(coarse[0]) if coarse.size else 0.0, dtype=float)
    t_coarse = np.linspace(0.0, 1.0, num=coarse.size)
    t_fine = np.linspace(0.0, 1.0, num=n)
    return np.asarray(np.interp(t_fine, t_coarse, coarse), dtype=float)


def add_white_noise_snr(
    x: NDArray[np.floating],
    snr_db: float,
    rng: np.random.Generator,
) -> NDArray[np.floating]:
    """Add white Gaussian noise calibrated to an exact SNR in decibels.

    Defines ``SNR_dB = 10 · log10(Var(x) / Var(ε))``, so
    ``Var(ε) = Var(x) / 10^(SNR_dB / 10)``.
    """
    values = np.asarray(x, dtype=float).ravel()
    n = values.size
    if n == 0:
        return values.copy()
    var_x = float(np.var(values))
    if var_x < 1e-15:
        logger.warning("add_white_noise_snr: near-zero signal variance; injecting unit noise")
        var_x = 1.0
    var_eps = var_x / (10.0 ** (float(snr_db) / 10.0))
    noise = rng.normal(0.0, np.sqrt(var_eps), size=n)
    return np.asarray(values + noise, dtype=float)


def apply_transform(
    x: NDArray[np.floating],
    name: str,
    rng: np.random.Generator | None = None,
    *,
    scale: float = 3.5,
    shift: float = 10.0,
    outlier_rate: float = 0.02,
    outlier_magnitude: float = 6.0,
    n_levels: int = 8,
    downsample_factor: int = 2,
) -> NDArray[np.floating]:
    """Dispatch a named transform. Stochastic names require ``rng``."""
    key = name.strip().lower()
    if key in {"affine", "affine_transform"}:
        return affine_transform(x, scale=scale, shift=shift)
    if key in {"monotone", "monotone_nonlinear"}:
        return monotone_nonlinear(x)
    if key in {"outliers", "impulsive_outliers"}:
        if rng is None:
            raise ValueError("impulsive_outliers requires an rng")
        return impulsive_outliers(
            x, rng, rate=outlier_rate, magnitude=outlier_magnitude
        )
    if key in {"quantize", "quantize_ties"}:
        return quantize_ties(x, n_levels=n_levels)
    if key in {"downsample", "downsample_series"}:
        return downsample_series(x, factor=downsample_factor)
    if key.startswith("snr_") or key.startswith("noise_snr_"):
        if rng is None:
            raise ValueError("add_white_noise_snr requires an rng")
        snr_db = float(key.rsplit("_", 1)[-1])
        return add_white_noise_snr(x, snr_db, rng)
    if key in {"noise", "white_noise", "add_white_noise_snr"}:
        raise ValueError(
            "Use apply_transform(..., name='snr_<dB>') or call "
            "add_white_noise_snr directly with an explicit snr_db"
        )
    raise ValueError(f"Unknown transform {name!r}")
