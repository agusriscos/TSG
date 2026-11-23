"""Shapley Interaction Index (SII) over the four TS2G families.

Reuses the 16-coalition Macro-F1 characteristic function already persisted in
``information_metrics.json`` (``subset_scores``), including the majority
baseline :math:`v(\\emptyset)`. Uncertainty on :math:`\\phi_i` is obtained by
resampling each coalition score from its CV mean/std and recomputing Shapley.
No graphs are recomputed.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from itertools import combinations
from math import factorial
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from tsg.audit.information import (
    FAMILY_ORDER,
    FAMILY_SHORT,
    exact_shapley,
    subset_label,
)

logger = logging.getLogger(__name__)

# Macro-F1 of a constant predictor on K balanced classes:
# F1_c = 2/(K+1) for the predicted class and 0 elsewhere → macro = 2/(K(K+1)).
# For K=12 this is 2/156 = 1/78 (not 1/K = 1/12).
NOTE_V_EMPTY = (
    "Macro-F1 of constant/majority predictor = 2 / (K * (K + 1)) = 1/78 "
    "for K=12 balanced regimes"
)


def exact_sii_pairwise(
    values: Mapping[frozenset[str], float],
    players: Sequence[str],
) -> dict[tuple[str, str], float]:
    """Pairwise Shapley Interaction Index for every unordered pair.

    For two distinct players :math:`i,j`,

    .. math::

        I_{ij} = \\sum_{S \\subseteq N \\setminus \\{i,j\\}}
        \\frac{|S|!\\,(M-|S|-2)!}{(M-1)!}
        \\delta_{ij}(S),

    where
    :math:`\\delta_{ij}(S) = v(S\\cup\\{i,j\\}) - v(S\\cup\\{i\\})
    - v(S\\cup\\{j\\}) + v(S)`.
    Positive :math:`I_{ij}` indicates algorithmic synergy of the pair.
    """
    ordered = tuple(players)
    n_players = len(ordered)
    if n_players < 2:
        return {}
    denom = float(factorial(n_players - 1))
    interactions: dict[tuple[str, str], float] = {}
    for left, right in combinations(ordered, 2):
        others = [p for p in ordered if p not in {left, right}]
        total = 0.0
        for size in range(len(others) + 1):
            for coalition_tuple in combinations(others, size):
                coalition = frozenset(coalition_tuple)
                delta = (
                    values[coalition | {left, right}]
                    - values[coalition | {left}]
                    - values[coalition | {right}]
                    + values[coalition]
                )
                weight = (
                    factorial(len(coalition))
                    * factorial(n_players - len(coalition) - 2)
                    / denom
                )
                total += weight * delta
        interactions[(left, right)] = float(total)
        logger.info(
            "SII %s+%s = %+.5f",
            FAMILY_SHORT.get(left, left),
            FAMILY_SHORT.get(right, right),
            total,
        )
    return interactions


def empty_set_baseline(
    values: Mapping[frozenset[str], float],
) -> float:
    """Return :math:`v(\\emptyset)` from a characteristic function."""
    return float(values[frozenset()])


def characteristic_from_subset_rows(
    rows: Sequence[Mapping[str, Any]],
    model: str,
    *,
    players: Sequence[str] = FAMILY_ORDER,
) -> dict[frozenset[str], float]:
    """Build :math:`v(S)` for one model from ``subset_scores`` JSON rows."""
    allowed = set(players)
    values: dict[frozenset[str], float] = {}
    for row in rows:
        if str(row.get("model")) != model:
            continue
        families = [str(f) for f in row.get("families", []) if str(f) in allowed]
        key = frozenset(families)
        values[key] = float(row["macro_f1_mean"])
    # Ensure the empty coalition is present (majority baseline).
    if frozenset() not in values:
        raise KeyError(
            f"subset_scores for model={model!r} is missing v(∅) "
            "(label 'empty' / families=[])"
        )
    missing = []
    for size in range(len(players) + 1):
        for subset in combinations(players, size):
            if frozenset(subset) not in values:
                missing.append(subset_label(subset))
    if missing:
        raise KeyError(
            f"subset_scores for model={model!r} missing coalitions: {missing}"
        )
    return values


@dataclass
class ModelSIIResult:
    """First-order Shapley values + pairwise SII for one classifier."""

    model: str
    v_empty: float
    v_full: float
    phi: dict[str, float]
    sii: dict[str, float]
    subset_scores: dict[str, float] = field(default_factory=dict)
    phi_std: dict[str, float] = field(default_factory=dict)
    phi_ci_low: dict[str, float] = field(default_factory=dict)
    phi_ci_high: dict[str, float] = field(default_factory=dict)
    n_bootstrap: int = 0

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SIIReport:
    """SII artifacts for every model found in ``subset_scores``."""

    by_model: dict[str, ModelSIIResult]
    source_path: str
    players: tuple[str, ...] = FAMILY_ORDER

    def as_dict(self) -> dict[str, Any]:
        return {
            "source_path": self.source_path,
            "players": list(self.players),
            "by_model": {
                name: result.as_dict() for name, result in self.by_model.items()
            },
        }


def _pair_key(left: str, right: str) -> str:
    return f"{FAMILY_SHORT[left]}+{FAMILY_SHORT[right]}"


def characteristic_mean_std_from_rows(
    rows: Sequence[Mapping[str, Any]],
    model: str,
    *,
    players: Sequence[str] = FAMILY_ORDER,
) -> tuple[dict[frozenset[str], float], dict[frozenset[str], float]]:
    """Build mean and CV-std characteristic functions for one model."""
    allowed = set(players)
    means: dict[frozenset[str], float] = {}
    stds: dict[frozenset[str], float] = {}
    for row in rows:
        if str(row.get("model")) != model:
            continue
        families = [str(f) for f in row.get("families", []) if str(f) in allowed]
        key = frozenset(families)
        means[key] = float(row["macro_f1_mean"])
        stds[key] = float(row.get("macro_f1_std", 0.0) or 0.0)
    if frozenset() not in means:
        raise KeyError(
            f"subset_scores for model={model!r} is missing v(∅) "
            "(label 'empty' / families=[])"
        )
    return means, stds


def bootstrap_shapley_phi(
    means: Mapping[frozenset[str], float],
    stds: Mapping[frozenset[str], float],
    players: Sequence[str],
    *,
    n_bootstrap: int = 1000,
    seed: int = 0,
    ci_level: float = 0.95,
) -> tuple[dict[str, float], dict[str, float], dict[str, float]]:
    """Resample :math:`v(S)\\sim\\mathcal{N}(\\mu_S,\\sigma_S)` and recompute φ.

    Returns ``(phi_std, phi_ci_low, phi_ci_high)`` dictionaries keyed by family.
    Scores are clipped to ``[0, 1]``. When every ``σ_S = 0`` the intervals
    collapse to the point estimate.
    """
    ordered = tuple(players)
    point = exact_shapley(dict(means), ordered)
    n_boot = int(max(n_bootstrap, 0))
    if n_boot < 2:
        return (
            {p: 0.0 for p in ordered},
            {p: float(point[p]) for p in ordered},
            {p: float(point[p]) for p in ordered},
        )

    keys = list(means)
    mu = np.array([float(means[k]) for k in keys], dtype=float)
    sigma = np.array([float(stds.get(k, 0.0) or 0.0) for k in keys], dtype=float)
    rng = np.random.default_rng(seed)
    samples = np.empty((n_boot, len(ordered)), dtype=float)
    for b in range(n_boot):
        draws = mu + sigma * rng.standard_normal(len(keys))
        draws = np.clip(draws, 0.0, 1.0)
        values = {key: float(val) for key, val in zip(keys, draws)}
        # Keep efficiency: empty / full use their point means when σ=0.
        for key, sig in zip(keys, sigma):
            if sig <= 0.0:
                values[key] = float(means[key])
        phi_b = exact_shapley(values, ordered)
        samples[b] = [float(phi_b[p]) for p in ordered]

    alpha = 1.0 - float(ci_level)
    phi_std = {
        p: float(np.std(samples[:, i], ddof=1)) for i, p in enumerate(ordered)
    }
    phi_lo = {
        p: float(np.quantile(samples[:, i], 0.5 * alpha))
        for i, p in enumerate(ordered)
    }
    phi_hi = {
        p: float(np.quantile(samples[:, i], 1.0 - 0.5 * alpha))
        for i, p in enumerate(ordered)
    }
    return phi_std, phi_lo, phi_hi


def compute_sii_report(
    subset_rows: Sequence[Mapping[str, Any]],
    *,
    models: Sequence[str] | None = None,
    players: Sequence[str] = FAMILY_ORDER,
    source_path: str = "",
    n_bootstrap: int = 1000,
    seed: int = 0,
    ci_level: float = 0.95,
) -> SIIReport:
    """Exact Shapley + pairwise SII from persisted subset Macro-F1 scores."""
    if models is None:
        models = tuple(sorted({str(r["model"]) for r in subset_rows}))
    by_model: dict[str, ModelSIIResult] = {}
    for model_idx, model in enumerate(models):
        means, stds = characteristic_mean_std_from_rows(
            subset_rows, model, players=players
        )
        # Keep legacy helper for missing-coalition checks.
        values = characteristic_from_subset_rows(subset_rows, model, players=players)
        phi = exact_shapley(values, players)
        sii_raw = exact_sii_pairwise(values, players)
        sii = {_pair_key(a, b): float(v) for (a, b), v in sii_raw.items()}
        subsets = {
            subset_label(key): float(score) for key, score in values.items()
        }
        v_empty = empty_set_baseline(values)
        v_full = float(values[frozenset(players)])
        phi_std, phi_lo, phi_hi = bootstrap_shapley_phi(
            means,
            stds,
            players,
            n_bootstrap=n_bootstrap,
            seed=seed + 17 * model_idx,
            ci_level=ci_level,
        )
        by_model[model] = ModelSIIResult(
            model=model,
            v_empty=v_empty,
            v_full=v_full,
            phi={p: float(phi[p]) for p in players},
            sii=sii,
            subset_scores=subsets,
            phi_std=phi_std,
            phi_ci_low=phi_lo,
            phi_ci_high=phi_hi,
            n_bootstrap=int(n_bootstrap),
        )
        logger.info(
            "SII model=%-14s v(∅)=%.4f v(full)=%.4f sum_phi=%.4f "
            "(|φ| uncertainty via B=%d)",
            model,
            v_empty,
            v_full,
            sum(phi.values()),
            n_bootstrap,
        )
        # Efficiency identity: Σφ + v(∅) = v(N).
        gap = abs(sum(phi.values()) + v_empty - v_full)
        if gap > 1e-6:
            logger.warning(
                "Shapley efficiency gap for %s is %.3e "
                "(expected Σφ + v(∅) = v(N); "
                "v(∅)=2/(K(K+1))=1/78 for K=12 balanced)",
                model,
                gap,
            )
    return SIIReport(by_model=by_model, source_path=source_path, players=tuple(players))


def sii_report_from_metrics_json(
    path: Path | str,
    *,
    n_bootstrap: int = 1000,
    seed: int = 0,
    ci_level: float = 0.95,
) -> SIIReport:
    """Load ``information_metrics.json`` and compute the SII report."""
    metrics_path = Path(path)
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    rows = payload.get("subset_scores")
    if not rows:
        raise KeyError(f"No subset_scores in {metrics_path}")
    return compute_sii_report(
        rows,
        source_path=str(metrics_path),
        n_bootstrap=n_bootstrap,
        seed=seed,
        ci_level=ci_level,
    )


def sii_to_frames(report: SIIReport) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return ``(phi_frame, sii_frame, baseline_frame)`` for CSV export."""
    phi_rows: list[dict[str, Any]] = []
    sii_rows: list[dict[str, Any]] = []
    base_rows: list[dict[str, Any]] = []
    for model, block in report.by_model.items():
        base_rows.append(
            {
                "model": model,
                "v_empty": block.v_empty,
                "v_full": block.v_full,
                "gap_full_minus_empty": block.v_full - block.v_empty,
                "note_v_empty": NOTE_V_EMPTY,
            }
        )
        for family, value in block.phi.items():
            phi_rows.append(
                {
                    "model": model,
                    "family": family,
                    "shapley_phi": value,
                    "phi_std": float(block.phi_std.get(family, float("nan"))),
                    "ci_low": float(block.phi_ci_low.get(family, float("nan"))),
                    "ci_high": float(block.phi_ci_high.get(family, float("nan"))),
                    "n_bootstrap": int(block.n_bootstrap),
                    "v_empty": block.v_empty,
                    "v_full": block.v_full,
                }
            )
        for pair, value in block.sii.items():
            sii_rows.append(
                {
                    "model": model,
                    "pair": pair,
                    "sii": value,
                    "v_empty": block.v_empty,
                }
            )
    return pd.DataFrame(phi_rows), pd.DataFrame(sii_rows), pd.DataFrame(base_rows)
