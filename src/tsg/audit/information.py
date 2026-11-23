"""Partial information, O-information and exact family-level Shapley values.

The four TS2G families (visibility, proximity, transition, similarity) are
treated as views of the same series. Three complementary summaries are
computed from a cached descriptor matrix, without adding information-theory
dependencies beyond NumPy, SciPy and scikit-learn:

* Pairwise partial information decomposition under the MMI rule, using a
  leakage-free plugin estimator of :math:`I(F; Y)`.
* Gaussian O-information of the four family representations, globally and
  inside dynamic groups.
* Exact Shapley values of Macro-F1, which are well-defined because there are
  only :math:`2^4 = 16` subsets.
"""

from __future__ import annotations

import json
import logging
import pickle
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.special import digamma
from sklearn.base import clone
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold

from tsg.audit.artifacts import (
    atomic_write_text,
    meta_from_records,
    save_features_npz,
    save_oof_predictions,
    save_table_csv,
    write_config_snapshot,
    write_metrics_alias,
)
from tsg.audit.classification import _evaluate_view, _make_models
from tsg.audit.config import AuditConfig
from tsg.audit.ensemble import (
    generate_one_matrix,
    load_generator_suite_config,
    load_record,
    realization_seed,
    save_record,
    versioned_cache_dir,
)
from tsg.audit.features import build_feature_table, family_feature_mask
from tsg.audit.health import diagnose_and_clean
from tsg.audit.pareto import (
    SeriesFamilyTiming,
    benchmark_family_times,
    mean_and_sample_std,
    pareto_front_mask,
    subset_cost_stats,
    timing_lengths_for_profile,
)

logger = logging.getLogger(__name__)

FAMILY_ORDER: tuple[str, ...] = (
    "visibility",
    "proximity",
    "transition",
    "similarity",
)
FAMILY_SHORT: dict[str, str] = {
    "visibility": "vis",
    "proximity": "prox",
    "transition": "trans",
    "similarity": "sim",
}
FAMILY_COLORS: dict[str, str] = {
    "visibility": "#4c78a8",
    "transition": "#f28e2b",
    "proximity": "#59a14f",
    "similarity": "#b07aa1",
}
MODEL_LABELS: dict[str, str] = {
    "random_forest": "Random Forest",
    "logistic": "Logística L2",
}
# Exhaustive partition of the 12-regime panel (360 series when R=30).
# ``structured_other`` holds quasiperiodic and regime-switching so every
# realization belongs to exactly one macro-group.
DYNAMIC_GROUPS: dict[str, tuple[str, ...]] = {
    "stochastic": (
        "white_noise",
        "fbm_antipersistent",
        "brownian",
        "fbm_persistent",
        "arma_correlated",
    ),
    "chaotic_flows": ("lorenz", "rossler"),
    "discrete_maps": ("henon", "logistic_weak", "logistic"),
    "structured_other": ("quasiperiodic", "regime_switching"),
}
GROUP_LABELS: dict[str, str] = {
    "stochastic": "estocásticos",
    "chaotic_flows": "flujos caóticos",
    "discrete_maps": "mapas discretos",
    "structured_other": "estructurados / conmutados",
}
GAUSSIAN_RIDGE = 1e-3

NOTES: dict[str, str] = {
    "mi": (
        "I(F; Y) es el estimador plugin H(Y) − H(Y|F), en bits. H(Y|F) es la "
        "entropía media del posterior de una regresión logística L2 ajustada "
        "out-of-fold (el pliegue de test no entra en el ajuste). H(Y) incluye "
        "la corrección de Miller–Madow: esa constante se suma a las tres "
        "informaciones mutuas de un par y solo desplaza la redundancia; la "
        "información única y la sinergia no cambian. Las MI negativas del "
        "plugin se truncan a 0 antes de aplicar MMI."
    ),
    "o_information": (
        "Cada familia se resume en su primer componente principal (columnas "
        "estandarizadas). La O-información usa la entropía diferencial "
        "gaussiana, con covarianza regularizada cuando no es definida positiva "
        "y con la corrección de sesgo de Wishart del log-determinante cuando "
        "sí lo es. Ω>0 indica dominancia de redundancia y Ω<0 de sinergia. "
        "Los componentes se ajustan una vez en el panel completo; el valor "
        "condicional reevalúa esa misma representación dentro del grupo o del régimen."
    ),
    "shapley": (
        "Valores de Shapley exactos del Macro-F1 sobre las 4 familias "
        "(los 16 subconjuntos, sin Monte Carlo). v(∅) es el Macro-F1 de un "
        "clasificador de clase mayoritaria reajustado en cada pliegue, con "
        "desempate lexicográfico. La suma de los cuatro valores es "
        "Score(full) − Score(∅)."
    ),
    "pareto": (
        "El coste de una combinación es la suma de los tiempos de construir "
        "el grafo y extraer los descriptores de las familias que contiene. "
        "La generación de la serie no entra en el cronómetro. Una combinación "
        "está en la frontera si ninguna otra es a la vez más rápida y más precisa."
    ),
}


def subset_label(families: Sequence[str]) -> str:
    """Stable short label. The full set is ``full`` and the empty set is ``empty``."""
    ordered = tuple(family for family in FAMILY_ORDER if family in set(families))
    if not ordered:
        return "empty"
    if len(ordered) == len(FAMILY_ORDER):
        return "full"
    return "+".join(FAMILY_SHORT[family] for family in ordered)


def nonempty_family_subsets() -> list[tuple[str, ...]]:
    """The 15 non-empty subsets, ordered by size and then by :data:`FAMILY_ORDER`."""
    subsets: list[tuple[str, ...]] = []
    for size in range(1, len(FAMILY_ORDER) + 1):
        subsets.extend(combinations(FAMILY_ORDER, size))
    return subsets


def mmi_decomposition(
    mi_i: float,
    mi_j: float,
    mi_joint: float,
) -> tuple[float, float, float, float]:
    """MMI atoms ``(redundancy, unique_i, unique_j, synergy)``.

    The definitions are

    .. math::

        Red = \\min\\{I_i, I_j\\},\\quad
        Un_i = I_i - Red,\\quad
        Un_j = I_j - Red,\\quad
        Syn = I_{ij} - I_i - I_j + Red.

    They satisfy :math:`I_{ij} = Red + Un_i + Un_j + Syn` for any finite
    inputs. Redundancy and both unique terms are non-negative when ``mi_i``
    and ``mi_j`` are non-negative. Synergy equals :math:`I_{ij} - \\max(I_i, I_j)`
    and is negative when an estimator breaks monotonicity of mutual information.
    """
    redundancy = float(min(mi_i, mi_j))
    unique_i = float(mi_i - redundancy)
    unique_j = float(mi_j - redundancy)
    synergy = float(mi_joint - mi_i - mi_j + redundancy)
    return redundancy, unique_i, unique_j, synergy


def discrete_entropy_bits(labels: np.ndarray, *, miller_madow: bool = True) -> float:
    """Shannon entropy of a discrete sample, in bits.

    The Miller–Madow term ``(K - 1) / (2 n ln 2)`` is the first-order bias
    correction of the plugin estimator.
    """
    values, counts = np.unique(np.asarray(labels), return_counts=True)
    total = float(counts.sum())
    if total <= 0:
        raise ValueError("Cannot estimate entropy of an empty sample")
    probabilities = counts / total
    entropy = float(-np.sum(probabilities * np.log2(probabilities)))
    if miller_madow:
        entropy += (len(values) - 1) / (2.0 * total * np.log(2.0))
    return entropy


def mean_posterior_entropy_bits(proba: np.ndarray) -> float:
    """Mean row-wise entropy of a row-stochastic predictive distribution."""
    probabilities = np.clip(np.asarray(proba, dtype=float), 1e-15, 1.0)
    probabilities = probabilities / probabilities.sum(axis=1, keepdims=True)
    row_entropy = -np.sum(probabilities * np.log2(probabilities), axis=1)
    return float(row_entropy.mean())


def predictive_mi_bits(proba: np.ndarray, h_y_bits: float) -> float:
    """Plugin mutual information :math:`H(Y) - H(Y \\mid F)` in bits."""
    return float(h_y_bits - mean_posterior_entropy_bits(proba))


def _wishart_logdet_bias(degrees_of_freedom: int, n_dim: int) -> float:
    """Bias of ``log det`` of the unbiased Gaussian covariance, in nats.

    With ``nu = n - 1``,

    ``E[log det Σ_hat] - log det Σ = Σ_j [ψ((ν - j) / 2) - log(ν / 2)]``.
    """
    indices = np.arange(n_dim, dtype=float)
    return float(
        np.sum(
            digamma(0.5 * (degrees_of_freedom - indices))
            - np.log(0.5 * degrees_of_freedom)
        )
    )


def gaussian_entropy_bits(
    samples: np.ndarray,
    *,
    ridge: float = GAUSSIAN_RIDGE,
    bias_correction: bool = True,
) -> float:
    """Regularized Gaussian differential entropy, in bits.

    Parameters
    ----------
    samples
        Array of shape ``(n_samples, n_dim)``.
    ridge
        Relative ridge added to the covariance when it is not safely positive
        definite. The ridge is ``ridge`` times the mean non-negative eigenvalue,
        so a global rescaling of the variables does not change the regularization.
    bias_correction
        Subtract the Wishart bias of ``log det Σ_hat`` when the sample
        covariance is well conditioned. The correction is omitted under ridge
        regularization, where that Wishart identity no longer holds.
    """
    data = np.asarray(samples, dtype=float)
    if data.ndim == 1:
        data = data.reshape(-1, 1)
    if data.ndim != 2:
        raise ValueError("samples must have shape (n_samples, n_dim)")
    n_samples, n_dim = data.shape
    if n_samples < 2 or n_dim < 1:
        raise ValueError(f"Need at least 2 samples and 1 dimension, got {data.shape}")
    centered = data - data.mean(axis=0, keepdims=True)
    degrees = n_samples - 1
    covariance = (centered.T @ centered) / float(degrees)
    covariance = 0.5 * (covariance + covariance.T)
    eigenvalues = np.linalg.eigvalsh(covariance)
    scale = float(np.mean(np.clip(eigenvalues, 0.0, None)))
    scale = max(scale, 1e-12)
    jitter = float(ridge) * scale
    well_conditioned = bool(float(eigenvalues.min()) > 1e-10 * scale)
    correction_ok = bias_correction and degrees - (n_dim - 1) >= 2
    if well_conditioned:
        matrix = covariance
        correction = _wishart_logdet_bias(degrees, n_dim) if correction_ok else 0.0
    else:
        matrix = covariance + jitter * np.eye(n_dim)
        correction = 0.0
    sign, logdet = np.linalg.slogdet(matrix)
    if sign <= 0.0:
        matrix = matrix + (jitter + 1e-8) * np.eye(n_dim)
        correction = 0.0
        sign, logdet = np.linalg.slogdet(matrix)
    if sign <= 0.0:
        raise np.linalg.LinAlgError("Regularized covariance is not positive definite")
    entropy_nats = 0.5 * (
        n_dim * np.log(2.0 * np.pi * np.e) + float(logdet) - correction
    )
    return float(entropy_nats / np.log(2.0))


def o_information_from_entropies(
    h_joint: float,
    h_marginal: Sequence[float],
    h_leave_one_out: Sequence[float],
) -> float:
    """O-information :math:`(n-2) H(X) + Σ_i [H(X_i) - H(X_{-i})]`.

    The entropies must already share a unit. For ``n = 2`` the value is
    identically zero.
    """
    marginal = [float(value) for value in h_marginal]
    leave = [float(value) for value in h_leave_one_out]
    n_vars = len(marginal)
    if n_vars < 2:
        raise ValueError("O-information needs at least 2 variables")
    if len(leave) != n_vars:
        raise ValueError("leave-one-out entropies must match the marginals")
    interaction = sum(h_i - h_rest for h_i, h_rest in zip(marginal, leave))
    return float((n_vars - 2) * h_joint + interaction)


def o_information_gaussian(
    samples: np.ndarray,
    *,
    ridge: float = GAUSSIAN_RIDGE,
    bias_correction: bool = True,
) -> float:
    """Gaussian O-information of the columns of ``samples``, in bits."""
    data = np.asarray(samples, dtype=float)
    if data.ndim != 2 or data.shape[1] < 2:
        raise ValueError("Expected samples with shape (n_samples, n_variables>=2)")
    h_joint = gaussian_entropy_bits(data, ridge=ridge, bias_correction=bias_correction)
    marginals: list[float] = []
    leave: list[float] = []
    for index in range(data.shape[1]):
        marginals.append(
            gaussian_entropy_bits(data[:, [index]], ridge=ridge, bias_correction=bias_correction)
        )
        rest = np.delete(data, index, axis=1)
        leave.append(gaussian_entropy_bits(rest, ridge=ridge, bias_correction=bias_correction))
    return o_information_from_entropies(h_joint, marginals, leave)


def interpret_omega(omega: float, *, tol: float = 0.05) -> str:
    """Map the sign of Ω to a redundancy / synergy / balance label."""
    if not np.isfinite(omega):
        return "no estimable"
    if omega > tol:
        return "redundancia"
    if omega < -tol:
        return "sinergia"
    return "equilibrio"


def exact_shapley(
    values: Mapping[frozenset[str], float],
    players: Sequence[str],
) -> dict[str, float]:
    """Exact Shapley values for a cooperative game on a small player set.

    ``values`` must contain every subset of ``players``, including the empty
    set. The weights are the classical ones,

    .. math::

        \\phi_i = \\sum_{S \\subseteq N \\setminus \\{i\\}}
        \\frac{|S|!\\, (M-|S|-1)!}{M!}
        \\bigl(v(S \\cup \\{i\\}) - v(S)\\bigr).
    """
    ordered = tuple(players)
    if len(set(ordered)) != len(ordered):
        raise ValueError(f"Duplicated players: {ordered}")
    n_players = len(ordered)
    if n_players == 0:
        return {}
    from math import factorial

    denominator = float(factorial(n_players))
    attribution = {player: 0.0 for player in ordered}
    for player in ordered:
        others = [name for name in ordered if name != player]
        for size in range(len(others) + 1):
            for coalition_tuple in combinations(others, size):
                coalition = frozenset(coalition_tuple)
                with_player = coalition | {player}
                if coalition not in values or with_player not in values:
                    raise KeyError(
                        f"Characteristic function is missing {subset_label(coalition)} "
                        f"or {subset_label(with_player)}"
                    )
                weight = (
                    factorial(len(coalition))
                    * factorial(n_players - len(coalition) - 1)
                    / denominator
                )
                attribution[player] += weight * (values[with_player] - values[coalition])
    return {player: float(attribution[player]) for player in ordered}


def first_principal_component(block: np.ndarray) -> np.ndarray:
    """Scores of the leading principal component of a standardized block.

    Columns with non-finite or near-zero standard deviation are dropped. The
    sign of the component is the sign returned by the SVD (deterministic).
    """
    data = np.asarray(block, dtype=float)
    if data.ndim != 2:
        raise ValueError("Expected a 2-D feature block")
    n_samples = data.shape[0]
    std = data.std(axis=0)
    keep = np.isfinite(std) & (std > 1e-8)
    if not np.any(keep):
        return np.zeros(n_samples, dtype=float)
    kept = data[:, keep]
    centered = kept - kept.mean(axis=0, keepdims=True)
    scaled = centered / std[keep]
    # Full matrices are unnecessary: the first right singular vector is enough.
    _, _, vt = np.linalg.svd(scaled, full_matrices=False)
    return scaled @ vt[0]


def family_principal_scores(features: pd.DataFrame) -> pd.DataFrame:
    """One principal-component score per TS2G family, aligned with ``features``."""
    scores: dict[str, np.ndarray] = {}
    for family in FAMILY_ORDER:
        mask = family_feature_mask(features.columns, family)
        if not bool(mask.any()):
            raise ValueError(f"Family {family!r} has no usable columns")
        scores[family] = first_principal_component(features.loc[:, mask].to_numpy(dtype=float))
    return pd.DataFrame(scores, index=features.index)


def oof_logistic_proba(
    features: pd.DataFrame,
    y: pd.Series,
    config: AuditConfig,
) -> np.ndarray:
    """Out-of-fold class probabilities of the phase-2 L2 logistic pipeline.

    Folds match :func:`tsg.audit.classification._evaluate_view`: stratified,
    shuffled with ``config.seed``. Every sample is predicted by a model that
    was not trained on it.
    """
    model = _make_models(config.seed)["logistic"]
    labels = y.to_numpy()
    matrix = features.to_numpy(dtype=float)
    n_splits = min(config.n_folds, int(pd.Series(labels).value_counts().min()))
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=config.seed)
    proba: np.ndarray | None = None
    for train_idx, test_idx in splitter.split(matrix, labels):
        classifier = clone(model)
        classifier.fit(matrix[train_idx], labels[train_idx])
        part = classifier.predict_proba(matrix[test_idx])
        if proba is None:
            proba = np.zeros((len(labels), part.shape[1]), dtype=float)
        elif part.shape[1] != proba.shape[1]:
            raise RuntimeError("OOF logistic folds do not share the same class set")
        proba[test_idx] = part
    if proba is None or not np.isfinite(proba).all():
        raise RuntimeError("OOF logistic probabilities were not filled")
    return proba


def majority_baseline_macro_f1(
    y: pd.Series,
    config: AuditConfig,
) -> tuple[float, float]:
    """Macro-F1 of a fold-wise majority classifier.

    Ties break toward the lexicographically smallest label so the empty-set
    value does not depend on row order. The fold standard deviation uses the
    same population ``ddof=0`` convention as the phase-2 view metrics.
    """
    labels = list(config.regimes)
    observed = y.to_numpy()
    n_splits = min(config.n_folds, int(y.value_counts().min()))
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=config.seed)
    fold_scores: list[float] = []
    dummy_x = np.zeros((len(observed), 1), dtype=float)
    for train_idx, test_idx in splitter.split(dummy_x, observed):
        values, counts = np.unique(observed[train_idx], return_counts=True)
        top = counts.max()
        chosen = sorted(str(value) for value, count in zip(values, counts) if count == top)[0]
        predicted = np.full(test_idx.shape, chosen)
        fold_scores.append(
            float(
                f1_score(
                    observed[test_idx],
                    predicted,
                    average="macro",
                    labels=labels,
                    zero_division=0,
                )
            )
        )
    scores = np.asarray(fold_scores, dtype=float)
    return float(scores.mean()), float(np.std(scores))


@dataclass
class PIDAtom:
    """MMI atoms of one family pair, in bits."""

    family_i: str
    family_j: str
    mi_i: float
    mi_j: float
    mi_joint: float
    mi_i_raw: float
    mi_j_raw: float
    mi_joint_raw: float
    redundancy: float
    unique_i: float
    unique_j: float
    synergy: float

    @property
    def pair_label(self) -> str:
        return f"{FAMILY_SHORT[self.family_i]} | {FAMILY_SHORT[self.family_j]}"

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["pair_label"] = self.pair_label
        payload["conservation_residual"] = float(
            self.mi_joint
            - (self.redundancy + self.unique_i + self.unique_j + self.synergy)
        )
        return payload


@dataclass
class SubsetEvaluation:
    """Macro-F1 characteristic functions, including the empty subset."""

    scores: dict[str, dict[frozenset[str], float]]
    stds: dict[str, dict[frozenset[str], float]]
    models: tuple[str, ...]


def _floor_mi(value: float) -> float:
    if not np.isfinite(value):
        raise ValueError(f"Mutual information is not finite: {value}")
    return float(max(0.0, value))


def pairwise_pid(
    features: pd.DataFrame,
    y: pd.Series,
    config: AuditConfig,
) -> tuple[list[PIDAtom], float, float]:
    """MMI table for the six family pairs.

    Returns the atoms, the plugin entropy of ``Y`` and the Miller–Madow
    corrected entropy (both in bits).
    """
    h_plugin = discrete_entropy_bits(y.to_numpy(), miller_madow=False)
    h_corrected = discrete_entropy_bits(y.to_numpy(), miller_madow=True)
    marginals: dict[str, tuple[float, float]] = {}
    for family in FAMILY_ORDER:
        block = features.loc[:, family_feature_mask(features.columns, family)]
        if block.shape[1] == 0:
            raise ValueError(f"Family {family!r} has no columns for PID")
        raw = predictive_mi_bits(oof_logistic_proba(block, y, config), h_corrected)
        marginals[family] = (raw, _floor_mi(raw))
        logger.info("PID marginal %-12s I(F;Y)=%.4f bits", family, raw)

    atoms: list[PIDAtom] = []
    for left, right in combinations(FAMILY_ORDER, 2):
        mask = family_feature_mask(features.columns, left) | family_feature_mask(
            features.columns, right
        )
        block = features.loc[:, mask]
        raw_joint = predictive_mi_bits(oof_logistic_proba(block, y, config), h_corrected)
        used_i = marginals[left][1]
        used_j = marginals[right][1]
        used_joint = _floor_mi(raw_joint)
        redundancy, unique_i, unique_j, synergy = mmi_decomposition(
            used_i, used_j, used_joint
        )
        atom = PIDAtom(
            family_i=left,
            family_j=right,
            mi_i=used_i,
            mi_j=used_j,
            mi_joint=used_joint,
            mi_i_raw=marginals[left][0],
            mi_j_raw=marginals[right][0],
            mi_joint_raw=raw_joint,
            redundancy=redundancy,
            unique_i=unique_i,
            unique_j=unique_j,
            synergy=synergy,
        )
        atoms.append(atom)
        logger.info(
            "PID %s red=%.3f un_%s=%.3f un_%s=%.3f syn=%.3f",
            atom.pair_label,
            redundancy,
            FAMILY_SHORT[left],
            unique_i,
            FAMILY_SHORT[right],
            unique_j,
            synergy,
        )
    return atoms, h_plugin, h_corrected


def evaluate_subset_scores(
    features: pd.DataFrame,
    y: pd.Series,
    config: AuditConfig,
    *,
    model_names: tuple[str, ...] = ("random_forest", "logistic"),
    rf_estimators: int | None = None,
    n_jobs: int | None = None,
) -> SubsetEvaluation:
    """Macro-F1 of every family subset, plus the majority baseline as ``v(∅)``.

    Production uses the same logistic and random-forest pipelines as the
    phase-2 audit. Tests may shrink the forest via ``rf_estimators``.
    """
    models = _make_models(config.seed)
    overrides: dict[str, int] = {}
    if rf_estimators is not None:
        overrides["clf__n_estimators"] = rf_estimators
    if n_jobs is not None:
        overrides["clf__n_jobs"] = n_jobs
    if overrides:
        models["random_forest"].set_params(**overrides)
    unknown = [name for name in model_names if name not in models]
    if unknown:
        raise KeyError(f"Unknown models {unknown}; known={tuple(models)}")

    empty_mean, empty_std = majority_baseline_macro_f1(y, config)
    scores: dict[str, dict[frozenset[str], float]] = {}
    stds: dict[str, dict[frozenset[str], float]] = {}
    subsets = nonempty_family_subsets()
    logger.info(
        "Scoring %d non-empty family subsets × %s (empty Macro-F1=%.4f)",
        len(subsets),
        ", ".join(model_names),
        empty_mean,
    )
    for model_name in model_names:
        scores[model_name] = {frozenset(): empty_mean}
        stds[model_name] = {frozenset(): empty_std}
        for subset in subsets:
            mask = np.zeros(features.shape[1], dtype=bool)
            for family in subset:
                mask |= family_feature_mask(features.columns, family)
            label = subset_label(subset)
            if not bool(mask.any()):
                scores[model_name][frozenset(subset)] = empty_mean
                stds[model_name][frozenset(subset)] = empty_std
                continue
            metrics = _evaluate_view(
                features.loc[:, mask],
                y,
                view=label,
                model_name=model_name,
                model=clone(models[model_name]),
                config=config,
            )
            scores[model_name][frozenset(subset)] = float(metrics.macro_f1_mean)
            stds[model_name][frozenset(subset)] = float(metrics.macro_f1_std)
            logger.info(
                "Subset %-16s %-14s Macro-F1=%.4f ± %.4f",
                label,
                model_name,
                metrics.macro_f1_mean,
                metrics.macro_f1_std,
            )
    return SubsetEvaluation(scores=scores, stds=stds, models=tuple(model_names))


def _omega_record(omega: float, n_samples: int) -> dict[str, Any]:
    return {
        "omega_bits": float(omega),
        "n_samples": int(n_samples),
        "reading": interpret_omega(omega),
    }


def o_information_report(
    scores: pd.DataFrame,
    y: pd.Series,
    config: AuditConfig,
    *,
    ridge: float = GAUSSIAN_RIDGE,
) -> dict[str, Any]:
    """Global, group-wise and per-regime Gaussian O-information."""
    matrix = scores.to_numpy(dtype=float)
    labels = y.to_numpy()
    report: dict[str, Any] = {
        "global": _omega_record(o_information_gaussian(matrix, ridge=ridge), len(labels)),
        "by_group": {},
        "by_regime": {},
        "excluded_from_dynamic_groups": [
            regime
            for regime in config.regimes
            if regime not in {name for group in DYNAMIC_GROUPS.values() for name in group}
        ],
    }
    for group_name, members in DYNAMIC_GROUPS.items():
        mask = np.isin(labels, members)
        if int(mask.sum()) < 8:
            logger.warning("Skipping O-information for group %s (n=%d)", group_name, mask.sum())
            continue
        omega = o_information_gaussian(matrix[mask], ridge=ridge)
        report["by_group"][group_name] = _omega_record(omega, int(mask.sum()))
        logger.info(
            "O-information group %-16s Ω=%+.4f bits (%s, n=%d)",
            group_name,
            omega,
            interpret_omega(omega),
            int(mask.sum()),
        )
    for regime in config.regimes:
        mask = labels == regime
        if int(mask.sum()) < 8:
            continue
        omega = o_information_gaussian(matrix[mask], ridge=ridge)
        report["by_regime"][regime] = _omega_record(omega, int(mask.sum()))
    logger.info(
        "O-information global Ω=%+.4f bits (%s)",
        report["global"]["omega_bits"],
        report["global"]["reading"],
    )
    return report


def _assemble_pareto(
    evaluation: SubsetEvaluation,
    timing: SeriesFamilyTiming,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for model_name in evaluation.models:
        candidates: list[dict[str, Any]] = []
        for subset in nonempty_family_subsets():
            key = frozenset(subset)
            time_mean, time_std = subset_cost_stats(timing, subset)
            candidates.append(
                {
                    "label": subset_label(subset),
                    "families": list(subset),
                    "n_families": len(subset),
                    "model": model_name,
                    "macro_f1_mean": float(evaluation.scores[model_name][key]),
                    "macro_f1_std": float(evaluation.stds[model_name][key]),
                    "time_ms_mean": float(time_mean),
                    "time_ms_std": float(time_std),
                    "n_steps": int(timing.n_steps),
                }
            )
        mask = pareto_front_mask(
            np.array([row["time_ms_mean"] for row in candidates], dtype=float),
            np.array([row["macro_f1_mean"] for row in candidates], dtype=float),
        )
        for row, on_front in zip(candidates, mask):
            row["on_pareto_front"] = bool(on_front)
            rows.append(row)
    return rows


def _timing_records(timings: Mapping[int, SeriesFamilyTiming]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for n_steps in sorted(timings):
        timing = timings[n_steps]
        for family in timing.families:
            column = timing.column(family)
            mean, std = mean_and_sample_std(column)
            records.append(
                {
                    "n_steps": int(n_steps),
                    "family": family,
                    "n_series": len(timing.dataset_ids),
                    "time_ms_mean": mean,
                    "time_ms_std": std,
                    "per_series_ms": {
                        dataset_id: float(column[index])
                        for index, dataset_id in enumerate(timing.dataset_ids)
                    },
                }
            )
    return records


def _shapley_block(evaluation: SubsetEvaluation) -> dict[str, Any]:
    block: dict[str, Any] = {
        "empty_macro_f1": float(evaluation.scores[evaluation.models[0]][frozenset()]),
        "by_model": {},
    }
    for model_name in evaluation.models:
        values = evaluation.scores[model_name]
        phi = exact_shapley(values, FAMILY_ORDER)
        full = float(values[frozenset(FAMILY_ORDER)])
        empty = float(values[frozenset()])
        gap = float(sum(phi.values()) - (full - empty))
        if abs(gap) > 1e-6:
            logger.warning("Shapley efficiency gap for %s is %.3e", model_name, gap)
        block["by_model"][model_name] = {
            "phi": {family: float(phi[family]) for family in FAMILY_ORDER},
            "full_macro_f1": full,
            "sum_phi": float(sum(phi.values())),
            "target_full_minus_empty": float(full - empty),
            "efficiency_gap": gap,
        }
        logger.info(
            "Shapley %-14s %s",
            model_name,
            " ".join(f"{FAMILY_SHORT[name]}={phi[name]:+.4f}" for name in FAMILY_ORDER),
        )
    return block


@dataclass
class InformationAuditResult:
    """Artifacts of one information-and-Pareto audit."""

    config: AuditConfig
    pid: list[PIDAtom]
    h_y_plugin_bits: float
    h_y_bits: float
    o_information: dict[str, Any]
    shapley: dict[str, Any]
    subset_rows: list[dict[str, Any]]
    pareto: list[dict[str, Any]]
    timing: list[dict[str, Any]]
    family_widths: dict[str, int]
    n_samples: int
    n_features_raw: int
    n_features_clean: int
    cache_dir: str
    cache_hits: int
    cache_misses: int
    reference_n_steps: int
    generated_at_utc: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    figure_paths: list[Path] = field(default_factory=list)
    records: list[dict[str, Any]] = field(default_factory=list)
    x_raw: pd.DataFrame | None = None
    x_clean: pd.DataFrame | None = None
    y: pd.Series | None = None

    def pareto_front(self, model: str) -> list[dict[str, Any]]:
        front = [
            row
            for row in self.pareto
            if row["model"] == model and row["on_pareto_front"]
        ]
        return sorted(front, key=lambda row: row["time_ms_mean"])

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "generated_at_utc": self.generated_at_utc,
            "notes": NOTES,
            "config": {
                "profile": self.config.profile,
                "seed": self.config.seed,
                "n_steps": self.config.n_steps,
                "n_realizations": self.config.n_realizations,
                "n_folds": self.config.n_folds,
                "brownian_mode": self.config.brownian_mode,
                "regimes": list(self.config.regimes),
                "families": {name: list(methods) for name, methods in self.config.families.items()},
            },
            "cache_dir": self.cache_dir,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "n_samples": self.n_samples,
            "n_features_raw": self.n_features_raw,
            "n_features_clean": self.n_features_clean,
            "family_widths": self.family_widths,
            "h_y_plugin_bits": self.h_y_plugin_bits,
            "h_y_bits": self.h_y_bits,
            "miller_madow_bits": float(self.h_y_bits - self.h_y_plugin_bits),
            "reference_n_steps": self.reference_n_steps,
            "pid_pairs": [atom.as_dict() for atom in self.pid],
            "o_information": self.o_information,
            "shapley": self.shapley,
            "subset_scores": self.subset_rows,
            "timing": self.timing,
            "pareto": self.pareto,
            "figures": [str(path) for path in self.figure_paths],
        }
        return _jsonify(payload)


def _jsonify(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonify(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonify(item) for item in value]
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    return value



def _load_records(
    config: AuditConfig,
    cache_dir: Path,
) -> tuple[list[dict[str, Any]], Path, int, int]:
    """Load the phase-2 ensemble cache, computing a record only on a miss."""
    generator_config = load_generator_suite_config()
    cache = versioned_cache_dir(cache_dir, config, generator_config)
    cache.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    hits = 0
    misses = 0
    for dataset_id in config.regimes:
        for realization in range(config.n_realizations):
            seed = realization_seed(config.seed, dataset_id, realization)
            record = load_record(cache, dataset_id, realization, seed)
            if record is None:
                misses += 1
                logger.info(
                    "Cache miss dataset=%s realization=%d; computing descriptor matrix",
                    dataset_id,
                    realization,
                )
                record = generate_one_matrix(
                    dataset_id,
                    seed=seed,
                    audit_config=config,
                    generator_config=generator_config,
                )
                record["realization"] = realization
                save_record(record, cache)
            else:
                hits += 1
                record.setdefault("realization", realization)
            records.append(record)
    logger.info("Ensemble cache %s hits=%d misses=%d", cache, hits, misses)
    return records, cache, hits, misses


def _subset_rows(evaluation: SubsetEvaluation) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    catalog: list[tuple[str, ...]] = [tuple(), *nonempty_family_subsets()]
    for model_name in evaluation.models:
        for subset in catalog:
            key = frozenset(subset)
            rows.append(
                {
                    "label": subset_label(subset),
                    "families": list(subset),
                    "n_families": len(subset),
                    "model": model_name,
                    "macro_f1_mean": float(evaluation.scores[model_name][key]),
                    "macro_f1_std": float(evaluation.stds[model_name][key]),
                }
            )
    return rows


class InformationAuditPipeline:
    """Run PID, O-information, exact Shapley and the cost-accuracy Pareto front."""

    def __init__(
        self,
        config: AuditConfig,
        cache_dir: Path | str,
        *,
        lang: str = "en",
        n_workers: int = 1,
    ) -> None:
        self.config = config
        self.cache_dir = Path(cache_dir)
        self.lang = lang
        self.n_workers = max(1, int(n_workers))

    def run(self, output_dir: Path | str | None = None) -> InformationAuditResult:
        config = self.config
        records, cache, hits, misses = _load_records(config, self.cache_dir)
        features, labels = build_feature_table(records, config)
        health = diagnose_and_clean(features, config)
        clean = health.x_clean
        clean.attrs.clear()
        family_widths = {
            family: int(family_feature_mask(clean.columns, family).sum())
            for family in FAMILY_ORDER
        }
        logger.info(
            "Descriptor matrix n=%d raw=%d clean=%d widths=%s",
            len(labels),
            health.n_features,
            clean.shape[1],
            family_widths,
        )
        pid, h_plugin, h_y = pairwise_pid(clean, labels, config)
        scores = family_principal_scores(clean)
        o_info = o_information_report(scores, labels, config)
        evaluation = evaluate_subset_scores(clean, labels, config)
        shapley = _shapley_block(evaluation)
        lengths = timing_lengths_for_profile(config.profile)
        # Always time the active panel length so smoke / custom N never falls back
        # to a heavier default (e.g. N=1000 when the cache was built at N=200).
        if int(config.n_steps) not in lengths:
            lengths = tuple(sorted({*lengths, int(config.n_steps)}))
        timings = benchmark_family_times(config, families=FAMILY_ORDER, lengths=lengths)
        reference_n = int(config.n_steps) if int(config.n_steps) in timings else int(lengths[-1])
        pareto = _assemble_pareto(evaluation, timings[reference_n])
        for model_name in evaluation.models:
            front = [
                row["label"]
                for row in sorted(
                    (row for row in pareto if row["model"] == model_name and row["on_pareto_front"]),
                    key=lambda row: row["time_ms_mean"],
                )
            ]
            logger.info("Pareto front %-14s %s", model_name, " < ".join(front))

        result = InformationAuditResult(
            config=config,
            pid=pid,
            h_y_plugin_bits=h_plugin,
            h_y_bits=h_y,
            o_information=o_info,
            shapley=shapley,
            subset_rows=_subset_rows(evaluation),
            pareto=pareto,
            timing=_timing_records(timings),
            family_widths=family_widths,
            n_samples=int(len(labels)),
            n_features_raw=int(health.n_features),
            n_features_clean=int(clean.shape[1]),
            cache_dir=str(cache),
            cache_hits=hits,
            cache_misses=misses,
            reference_n_steps=reference_n,
            generated_at_utc=datetime.now(timezone.utc).isoformat(),
            records=records,
            x_raw=features,
            x_clean=clean,
            y=labels,
        )
        if output_dir is not None:
            self.persist(result, Path(output_dir))
        return result

    def persist(self, result: InformationAuditResult, output_dir: Path) -> Path:
        """Write JSON, NPZ/CSV data, figures, Markdown and a self-contained HTML report."""
        from tsg.audit.information_plots import write_information_figures
        from tsg.reporting.information_report import write_information_reports

        output_dir.mkdir(parents=True, exist_ok=True)
        data_dir = output_dir / "data"
        tables_dir = data_dir / "tables"
        data_dir.mkdir(parents=True, exist_ok=True)
        tables_dir.mkdir(parents=True, exist_ok=True)

        self._persist_data(result, data_dir, tables_dir)

        result.figure_paths = write_information_figures(
            result,
            output_dir,
            dpi=result.config.figure_dpi,
            lang=self.lang,
        )

        metrics_path = output_dir / "information_metrics.json"
        atomic_write_text(
            metrics_path,
            json.dumps(result.to_dict(), indent=2),
        )
        write_metrics_alias(output_dir, metrics_path)
        write_config_snapshot(
            output_dir,
            block="information",
            profile=result.config.profile,
            seed=result.config.seed,
            config=result.to_dict().get("config", {}),
            extra={"lang": self.lang, "n_workers": self.n_workers},
        )
        write_information_reports(result, output_dir)
        logger.info("Information-audit artifacts written to %s", output_dir)
        return output_dir

    def plot_only(
        self,
        output_dir: Path | str,
        *,
        lang: str | None = None,
    ) -> InformationAuditResult:
        """Regenerate figures from persisted JSON without recomputing graphs.

        Parameters
        ----------
        output_dir : Path | str
            Directory previously written by :meth:`persist`.
        lang : str | None
            Override language; falls back to stored value or ``self.lang``.

        Returns
        -------
        InformationAuditResult
            Reconstructed result with updated ``figure_paths``.
        """
        from tsg.audit.information_plots import write_information_figures

        out = Path(output_dir)
        metrics_path = out / "information_metrics.json"
        if not metrics_path.exists():
            metrics_path = out / "metrics.json"
        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
        use_lang = lang or payload.get("lang") or self.lang

        result = _result_from_payload(payload, self.config)
        result.figure_paths = write_information_figures(
            result,
            out,
            dpi=result.config.figure_dpi,
            lang=use_lang,
        )
        payload["figures"] = [str(p) for p in result.figure_paths]
        payload["lang"] = use_lang
        atomic_write_text(
            out / "information_metrics.json",
            json.dumps(payload, indent=2, default=str),
        )
        write_metrics_alias(out, out / "information_metrics.json")
        logger.info("Plot-only figures regenerated in %s (lang=%s)", out, use_lang)
        return result

    def _persist_data(
        self,
        result: InformationAuditResult,
        data_dir: Path,
        tables_dir: Path,
    ) -> None:
        """Write features NPZ, CSV tables and records pickle."""
        config = result.config

        if result.x_raw is not None and result.y is not None:
            meta = meta_from_records(result.records) if result.records else {}
            save_features_npz(
                data_dir / "features_raw.npz",
                x=result.x_raw,
                y=result.y,
                regime_names=list(config.regimes),
                realization_ids=meta.get("realization_ids"),
                transform_or_kind=meta.get("transform_or_kind"),
            )
        if result.x_clean is not None and result.y is not None:
            meta = meta_from_records(result.records) if result.records else {}
            save_features_npz(
                data_dir / "features_clean.npz",
                x=result.x_clean,
                y=result.y,
                regime_names=list(config.regimes),
                realization_ids=meta.get("realization_ids"),
                transform_or_kind=meta.get("transform_or_kind"),
            )
            from tsg.audit.classification import evaluate_classification

            clf = evaluate_classification(result.x_clean, result.y, config)
            save_oof_predictions(data_dir / "oof_predictions.npz", clf.oof_bundles())

        # CSV tables
        save_table_csv(
            tables_dir / "pid.csv",
            pd.DataFrame([atom.as_dict() for atom in result.pid]),
        )
        save_table_csv(
            tables_dir / "o_information.csv",
            pd.DataFrame(_flatten_o_info(result.o_information)),
        )
        save_table_csv(
            tables_dir / "shapley.csv",
            pd.DataFrame(_flatten_shapley(result.shapley)),
        )
        save_table_csv(
            tables_dir / "pareto.csv",
            pd.DataFrame(result.pareto),
        )
        save_table_csv(
            tables_dir / "timing.csv",
            pd.DataFrame(result.timing),
        )

        if result.records:
            records_path = data_dir / "records.pkl"
            with records_path.open("wb") as fh:
                pickle.dump(result.records, fh, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info("Persisted information data to %s", data_dir)


def _flatten_o_info(o_info: dict[str, Any]) -> list[dict[str, Any]]:
    """Flatten the nested o-information dict into a list of rows."""
    rows: list[dict[str, Any]] = []
    g = o_info.get("global", {})
    rows.append({"scope": "global", "group": "", "omega_bits": g.get("omega_bits", float("nan")),
                  "n_samples": g.get("n_samples", 0), "reading": g.get("reading", "")})
    for group, data in o_info.get("by_group", {}).items():
        rows.append({"scope": "group", "group": group, "omega_bits": data.get("omega_bits", float("nan")),
                      "n_samples": data.get("n_samples", 0), "reading": data.get("reading", "")})
    for regime, data in o_info.get("by_regime", {}).items():
        rows.append({"scope": "regime", "group": regime, "omega_bits": data.get("omega_bits", float("nan")),
                      "n_samples": data.get("n_samples", 0), "reading": data.get("reading", "")})
    return rows


def _flatten_shapley(shapley: dict[str, Any]) -> list[dict[str, Any]]:
    """Flatten the nested Shapley dict into a list of rows."""
    rows: list[dict[str, Any]] = []
    for model_name, data in shapley.get("by_model", {}).items():
        for family, phi in data.get("phi", {}).items():
            rows.append({
                "model": model_name,
                "family": family,
                "shapley_value": phi,
                "full_macro_f1": data.get("full_macro_f1", float("nan")),
            })
    return rows


def _result_from_payload(
    payload: dict[str, Any],
    config: AuditConfig,
) -> InformationAuditResult:
    """Reconstruct an InformationAuditResult from a JSON payload (for plot_only)."""
    pid_dicts = payload.get("pid_pairs", [])
    pid_atoms: list[PIDAtom] = []
    for d in pid_dicts:
        pid_atoms.append(PIDAtom(
            family_i=d["family_i"],
            family_j=d["family_j"],
            mi_i=d["mi_i"],
            mi_j=d["mi_j"],
            mi_joint=d["mi_joint"],
            mi_i_raw=d.get("mi_i_raw", d["mi_i"]),
            mi_j_raw=d.get("mi_j_raw", d["mi_j"]),
            mi_joint_raw=d.get("mi_joint_raw", d["mi_joint"]),
            redundancy=d["redundancy"],
            unique_i=d["unique_i"],
            unique_j=d["unique_j"],
            synergy=d["synergy"],
        ))
    return InformationAuditResult(
        config=config,
        pid=pid_atoms,
        h_y_plugin_bits=payload.get("h_y_plugin_bits", 0.0),
        h_y_bits=payload.get("h_y_bits", 0.0),
        o_information=payload.get("o_information", {}),
        shapley=payload.get("shapley", {}),
        subset_rows=payload.get("subset_scores", []),
        pareto=payload.get("pareto", []),
        timing=payload.get("timing", []),
        family_widths=payload.get("family_widths", {}),
        n_samples=payload.get("n_samples", 0),
        n_features_raw=payload.get("n_features_raw", 0),
        n_features_clean=payload.get("n_features_clean", 0),
        cache_dir=payload.get("cache_dir", ""),
        cache_hits=payload.get("cache_hits", 0),
        cache_misses=payload.get("cache_misses", 0),
        reference_n_steps=payload.get("reference_n_steps", config.n_steps),
        generated_at_utc=payload.get(
            "generated_at_utc", datetime.now(timezone.utc).isoformat()
        ),
    )
