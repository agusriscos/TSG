"""Bootstrap CIs, permutation p-values and Holm adjustment for Gaussian Ω.

Reuses :func:`tsg.audit.information.o_information_gaussian` on the four
family PCA scores. Rows of the already-fitted score matrix are resampled
with replacement (consistent with the conditional-Ω definition used in the
MMI pipeline: PCA is fit once on the full panel). Independence under
:math:`H_0: \\Omega = 0` is probed by column-wise permutations.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from typing import Any, Sequence

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from tsg.audit.information import (
    DYNAMIC_GROUPS,
    GAUSSIAN_RIDGE,
    family_principal_scores,
    interpret_omega,
    o_information_gaussian,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class OmegaBootstrapConfig:
    """Resampling settings for multivariate O-information."""

    n_bootstrap: int = 1000
    n_permutations: int = 1000
    ci_level: float = 0.95
    seed: int = 0
    ridge: float = GAUSSIAN_RIDGE
    min_samples: int = 8
    # Store every replicate only when debugging; default keeps percentiles.
    store_replicates: bool = False


@dataclass
class OmegaBootstrapResult:
    """Point estimate of Ω with bootstrap and permutation uncertainty."""

    scope: str
    group: str
    omega_bits: float
    n_samples: int
    n_bootstrap: int
    ci_low: float
    ci_high: float
    p_value: float
    reading: str
    significant_vs_zero: bool
    n_permutations: int = 0
    p_perm: float = float("nan")
    p_holm: float = float("nan")
    replicates: list[float] | None = None

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        if not self.replicates:
            payload.pop("replicates", None)
        return payload


def bootstrap_indices(
    n_samples: int,
    n_bootstrap: int,
    *,
    seed: int,
) -> NDArray[np.int_]:
    """Draw ``n_bootstrap`` row-index vectors of length ``n_samples``."""
    if n_samples < 1:
        raise ValueError("n_samples must be positive")
    rng = np.random.default_rng(seed)
    return rng.integers(0, n_samples, size=(n_bootstrap, n_samples), endpoint=False)


def _percentile_ci(
    replicates: NDArray[np.floating],
    level: float,
) -> tuple[float, float]:
    alpha = 1.0 - float(level)
    low = float(np.quantile(replicates, 0.5 * alpha))
    high = float(np.quantile(replicates, 1.0 - 0.5 * alpha))
    return low, high


def two_sided_pvalue_vs_zero(replicates: NDArray[np.floating]) -> float:
    """Two-sided bootstrap p-value for :math:`H_0: \\Omega = 0`.

    Uses the proportion of replicates on each side of zero, floored at
    ``1 / (B + 1)`` so a never-crossing sample is not reported as ``p = 0``.
    """
    values = np.asarray(replicates, dtype=float)
    values = values[np.isfinite(values)]
    b = int(values.size)
    if b == 0:
        return float("nan")
    prop_pos = float(np.mean(values >= 0.0))
    prop_neg = float(np.mean(values <= 0.0))
    raw = 2.0 * min(prop_pos, prop_neg)
    return float(min(1.0, max(raw, 1.0 / (b + 1))))


def holm_adjust(p_values: Sequence[float]) -> list[float]:
    """Holm–Bonferroni step-down adjustment (monotone, ≤ 1)."""
    raw = np.asarray(list(p_values), dtype=float)
    n = int(raw.size)
    if n == 0:
        return []
    order = np.argsort(raw)
    adjusted = np.empty(n, dtype=float)
    running = 0.0
    for rank, idx in enumerate(order):
        if not np.isfinite(raw[idx]):
            adjusted[idx] = float("nan")
            continue
        candidate = min(1.0, float(raw[idx]) * (n - rank))
        running = max(running, candidate)
        adjusted[idx] = running
    return [float(v) for v in adjusted]


def permutation_pvalue_omega(
    matrix: NDArray[np.floating],
    omega_obs: float,
    *,
    n_permutations: int,
    seed: int,
    ridge: float,
) -> float:
    """Two-sided permutation p-value under column-wise independence.

    Each null draw independently shuffles every column (destroying dependence
    while preserving marginals). Returns
    ``(1 + #{|Ω_null| ≥ |Ω_obs|}) / (n_perm + 1)``.
    """
    data = np.asarray(matrix, dtype=float)
    n_perm = int(max(n_permutations, 0))
    if n_perm < 1 or not np.isfinite(omega_obs):
        return float("nan")
    rng = np.random.default_rng(seed)
    abs_obs = abs(float(omega_obs))
    count = 0
    for _ in range(n_perm):
        null = data.copy()
        for col in range(null.shape[1]):
            rng.shuffle(null[:, col])
        omega_null = float(o_information_gaussian(null, ridge=ridge))
        if abs(omega_null) >= abs_obs:
            count += 1
    return float((1 + count) / (n_perm + 1))


def bootstrap_omega(
    matrix: NDArray[np.floating],
    config: OmegaBootstrapConfig,
    *,
    seed_offset: int = 0,
) -> tuple[float, float, float, float, float, NDArray[np.floating]]:
    """Point Ω plus CI bounds, bootstrap p, permutation p and replicates.

    Returns
    -------
    omega, ci_low, ci_high, p_boot, p_perm, replicates
    """
    data = np.asarray(matrix, dtype=float)
    if data.ndim != 2 or data.shape[1] < 2:
        raise ValueError("Expected matrix with shape (n_samples, n_variables>=2)")
    n = int(data.shape[0])
    omega = float(o_information_gaussian(data, ridge=config.ridge))
    if n < config.min_samples:
        return (
            omega,
            float("nan"),
            float("nan"),
            float("nan"),
            float("nan"),
            np.array([omega]),
        )

    indices = bootstrap_indices(
        n, config.n_bootstrap, seed=config.seed + seed_offset
    )
    replicates = np.empty(config.n_bootstrap, dtype=float)
    for b in range(config.n_bootstrap):
        replicates[b] = o_information_gaussian(data[indices[b]], ridge=config.ridge)
    ci_low, ci_high = _percentile_ci(replicates, config.ci_level)
    p_boot = two_sided_pvalue_vs_zero(replicates)
    p_perm = permutation_pvalue_omega(
        data,
        omega,
        n_permutations=config.n_permutations,
        seed=config.seed + 50_000 + seed_offset,
        ridge=config.ridge,
    )
    return omega, ci_low, ci_high, p_boot, p_perm, replicates


def _pack_result(
    *,
    scope: str,
    group: str,
    omega: float,
    n_samples: int,
    config: OmegaBootstrapConfig,
    ci_low: float,
    ci_high: float,
    p_value: float,
    p_perm: float,
    replicates: NDArray[np.floating],
) -> OmegaBootstrapResult:
    significant = bool(
        np.isfinite(ci_low)
        and np.isfinite(ci_high)
        and (ci_high < 0.0 or ci_low > 0.0)
    )
    return OmegaBootstrapResult(
        scope=scope,
        group=group,
        omega_bits=float(omega),
        n_samples=int(n_samples),
        n_bootstrap=int(config.n_bootstrap),
        ci_low=float(ci_low),
        ci_high=float(ci_high),
        p_value=float(p_value),
        reading=interpret_omega(omega),
        significant_vs_zero=significant,
        n_permutations=int(config.n_permutations),
        p_perm=float(p_perm),
        p_holm=float("nan"),
        replicates=replicates.tolist() if config.store_replicates else None,
    )


def apply_holm_correction(
    results: Sequence[OmegaBootstrapResult],
) -> list[OmegaBootstrapResult]:
    """Attach Holm-adjusted ``p_holm`` from each row's ``p_perm``."""
    p_raw = [float(r.p_perm) for r in results]
    adjusted = holm_adjust(p_raw)
    out: list[OmegaBootstrapResult] = []
    for row, p_h in zip(results, adjusted):
        row.p_holm = float(p_h)
        out.append(row)
    return out


def o_information_bootstrap_report(
    scores: pd.DataFrame,
    y: pd.Series | Sequence[Any],
    *,
    config: OmegaBootstrapConfig | None = None,
    regimes: Sequence[str] | None = None,
) -> list[OmegaBootstrapResult]:
    """Global, macro-group and per-regime Ω with bootstrap CIs + perm p-values.

    Parameters
    ----------
    scores
        Columns = TS2G families (PCA scores). Prefer reusing a matrix already
        built by :func:`family_principal_scores`.
    y
        Regime labels.
    config
        Bootstrap / permutation settings.
    regimes
        Optional ordered regime list for the per-regime block; defaults to the
        unique labels in ``y``.
    """
    cfg = config or OmegaBootstrapConfig()
    labels = pd.Series(list(y), dtype=object)
    matrix = scores.to_numpy(dtype=float)
    results: list[OmegaBootstrapResult] = []

    omega, lo, hi, p_boot, p_perm, reps = bootstrap_omega(matrix, cfg, seed_offset=0)
    results.append(
        _pack_result(
            scope="global",
            group="",
            omega=omega,
            n_samples=len(labels),
            config=cfg,
            ci_low=lo,
            ci_high=hi,
            p_value=p_boot,
            p_perm=p_perm,
            replicates=reps,
        )
    )
    logger.info(
        "Ω global=%+.4f bits CI[%.4f, %.4f] p_boot=%.4g p_perm=%.4g (%s)",
        omega,
        lo,
        hi,
        p_boot,
        p_perm,
        interpret_omega(omega),
    )

    for offset, (group_name, members) in enumerate(DYNAMIC_GROUPS.items(), start=1):
        mask = labels.isin(members).to_numpy()
        n = int(mask.sum())
        if n < cfg.min_samples:
            logger.warning("Skipping Ω bootstrap for group %s (n=%d)", group_name, n)
            continue
        omega, lo, hi, p_boot, p_perm, reps = bootstrap_omega(
            matrix[mask], cfg, seed_offset=1000 * offset
        )
        results.append(
            _pack_result(
                scope="group",
                group=group_name,
                omega=omega,
                n_samples=n,
                config=cfg,
                ci_low=lo,
                ci_high=hi,
                p_value=p_boot,
                p_perm=p_perm,
                replicates=reps,
            )
        )
        logger.info(
            "Ω group %-16s %+.4f bits CI[%.4f, %.4f] p_boot=%.4g p_perm=%.4g",
            group_name,
            omega,
            lo,
            hi,
            p_boot,
            p_perm,
        )

    regime_list = list(regimes) if regimes is not None else sorted(set(labels.astype(str)))
    for offset, regime in enumerate(regime_list, start=1):
        mask = labels.to_numpy() == regime
        n = int(mask.sum())
        if n < cfg.min_samples:
            continue
        omega, lo, hi, p_boot, p_perm, reps = bootstrap_omega(
            matrix[mask], cfg, seed_offset=10_000 + 100 * offset
        )
        results.append(
            _pack_result(
                scope="regime",
                group=str(regime),
                omega=omega,
                n_samples=n,
                config=cfg,
                ci_low=lo,
                ci_high=hi,
                p_value=p_boot,
                p_perm=p_perm,
                replicates=reps,
            )
        )
        logger.info(
            "Ω regime %-20s %+.4f bits CI[%.4f, %.4f] p_boot=%.4g p_perm=%.4g sig=%s",
            regime,
            omega,
            lo,
            hi,
            p_boot,
            p_perm,
            results[-1].significant_vs_zero,
        )
    return apply_holm_correction(results)


def bootstrap_report_from_features(
    features: pd.DataFrame,
    y: pd.Series | Sequence[Any],
    *,
    config: OmegaBootstrapConfig | None = None,
    regimes: Sequence[str] | None = None,
    scores: pd.DataFrame | None = None,
) -> tuple[list[OmegaBootstrapResult], pd.DataFrame]:
    """Fit (or reuse) family PCA scores and run the bootstrap report."""
    if scores is None:
        scores = family_principal_scores(features)
    return o_information_bootstrap_report(
        scores, y, config=config, regimes=regimes
    ), scores


def results_to_frame(results: Sequence[OmegaBootstrapResult]) -> pd.DataFrame:
    """Tabular export without bootstrap replicates."""
    rows = []
    for item in results:
        payload = item.as_dict()
        payload.pop("replicates", None)
        rows.append(payload)
    return pd.DataFrame(rows)
