"""Advanced pairwise PID via BROJA / CCS (``dit``), global and by dynamic group.

The phase-1 MMI pipeline forces :math:`U_i = 0` whenever :math:`I(F_i;Y)
\\le I(F_j;Y)`. Here each family is summarised by its leading principal
component (reused from :mod:`tsg.audit.information`), discretised by
quantile bins, and decomposed with :class:`dit.pid.measures.ibroja.PID_BROJA`
(default) or CCS. Scopes reuse the same feature matrix — no TS2G recompute.
"""

from __future__ import annotations

import logging
from collections import Counter
from dataclasses import asdict, dataclass
from itertools import combinations
from typing import Any, Literal, Sequence

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from tsg.audit.information import (
    DYNAMIC_GROUPS,
    FAMILY_ORDER,
    FAMILY_SHORT,
    family_principal_scores,
)

logger = logging.getLogger(__name__)

PidEstimator = Literal["broja", "ccs", "mmi_discrete"]
MappingScopes = dict[str, Sequence[str]]


@dataclass(frozen=True)
class AdvancedPIDConfig:
    """Configuration for discrete BROJA / CCS PID on family PCA scores.

    ``n_bins`` defaults to 3: enough resolution for unique atoms without
    emptying the joint on small macro-groups (e.g. ``chaotic_flows``, n=60).
    The hard cap is 4 even when the caller requests more.
    """

    estimator: PidEstimator = "broja"
    n_bins: int = 3
    min_samples: int = 24
    seed: int = 0
    clip_neg_atoms: bool = True
    max_bins: int = 4


@dataclass
class AdvancedPIDAtom:
    """One pairwise PID atom at a given scope (global or macro-group)."""

    scope: str
    group: str
    family_i: str
    family_j: str
    estimator: str
    n_bins: int
    n_samples: int
    redundancy: float
    unique_i: float
    unique_j: float
    synergy: float
    mi_i: float
    mi_j: float
    mi_joint: float

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


def _require_dit() -> Any:
    """Import ``dit`` lazily so the core package stays optional-free."""
    try:
        import dit
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            "Advanced PID requires the optional dependency 'dit'. "
            "Install with: pip install 'dit[broja]'  (or: uv pip install 'dit[broja]')."
        ) from exc
    return dit


def quantile_bin_codes(
    values: NDArray[np.floating],
    n_bins: int,
    *,
    seed: int = 0,
) -> NDArray[np.int_]:
    """Map a 1-D continuous series to ``0 … n_bins-1`` via empirical quantiles.

    Ties at bin edges are broken deterministically with a seeded rank jitter so
    that equal values do not collapse into a single over-populated bin.
    """
    x = np.asarray(values, dtype=float).ravel()
    if x.size == 0:
        raise ValueError("Cannot bin an empty series")
    n_bins = int(max(2, n_bins))
    if np.allclose(x, x[0]):
        return np.zeros(x.size, dtype=int)
    rng = np.random.default_rng(seed)
    jitter = rng.uniform(-1e-12, 1e-12, size=x.size)
    ranks = pd.Series(x + jitter).rank(method="average").to_numpy(dtype=float)
    # Equal-frequency bins on ranks; clip the rightmost edge into the last bin.
    edges = np.linspace(ranks.min(), ranks.max(), n_bins + 1)
    codes = np.digitize(ranks, edges[1:-1], right=False)
    return np.clip(codes, 0, n_bins - 1).astype(int)


def adaptive_n_bins(
    n_samples: int,
    requested: int,
    n_classes: int,
    *,
    max_bins: int = 4,
) -> int:
    """Shrink the bin count when the empirical joint would be too sparse.

    Targets roughly ≥2 samples per ``(bin_i, bin_j, y)`` cell:
    ``n_bins**2 * n_classes * 2 ≲ n_samples``. Macro-groups with
    ``n < 120`` (e.g. chaotic flows) are hard-capped at 3 bins.
    """
    n_samples = int(max(n_samples, 1))
    n_classes = int(max(n_classes, 1))
    max_by_cells = int(
        np.floor(np.sqrt(max(n_samples / max(2 * n_classes, 1), 1.0)))
    )
    small_group_cap = 3 if n_samples < 120 else int(max_bins)
    hard_cap = min(int(max_bins), small_group_cap)
    return int(max(2, min(int(requested), max_by_cells, hard_cap)))


def empirical_joint_distribution(
    source_i: Sequence[Any],
    source_j: Sequence[Any],
    target: Sequence[Any],
) -> Any:
    """Build a normalised ``dit.Distribution`` over ``(S_i, S_j, Y)``."""
    dit = _require_dit()
    outcomes = list(
        zip(
            (str(v) for v in source_i),
            (str(v) for v in source_j),
            (str(v) for v in target),
        )
    )
    if not outcomes:
        raise ValueError("Cannot build a joint from an empty sample")
    counts = Counter(outcomes)
    total = float(sum(counts.values()))
    pmf = {outcome: count / total for outcome, count in counts.items()}
    return dit.Distribution(pmf, rv_names=["Si", "Sj", "Y"])


def _mmi_from_joint(dist: Any) -> tuple[float, float, float, float, float, float, float]:
    """MMI atoms from a discrete joint (fallback / diagnostic)."""
    from tsg.audit.information import mmi_decomposition

    mi_i = float(dist.mutual_information(["Si"], ["Y"]))
    mi_j = float(dist.mutual_information(["Sj"], ["Y"]))
    mi_joint = float(dist.mutual_information(["Si", "Sj"], ["Y"]))
    red, u_i, u_j, syn = mmi_decomposition(mi_i, mi_j, mi_joint)
    return red, u_i, u_j, syn, mi_i, mi_j, mi_joint


def _broja_or_ccs_atoms(
    dist: Any,
    estimator: PidEstimator,
) -> tuple[float, float, float, float, float, float, float]:
    """Return ``(red, u_i, u_j, syn, I_i, I_j, I_ij)`` in bits."""
    mi_i = float(dist.mutual_information(["Si"], ["Y"]))
    mi_j = float(dist.mutual_information(["Sj"], ["Y"]))
    mi_joint = float(dist.mutual_information(["Si", "Sj"], ["Y"]))

    if estimator == "mmi_discrete":
        return _mmi_from_joint(dist)

    dit = _require_dit()
    if estimator == "broja":
        from dit.pid.measures.ibroja import PID_BROJA

        pid = PID_BROJA(dist, [["Si"], ["Sj"]], ["Y"], method="auto")
    elif estimator == "ccs":
        from dit.pid.measures.iccs import PID_CCS

        pid = PID_CCS(dist, [["Si"], ["Sj"]], ["Y"])
    else:  # pragma: no cover
        raise ValueError(f"Unknown PID estimator {estimator!r}")

    redundancy = float(pid.get_pi((("Si",), ("Sj",))))
    unique_i = float(pid.get_pi((("Si",),)))
    unique_j = float(pid.get_pi((("Sj",),)))
    synergy = float(pid.get_pi((("Si", "Sj"),)))
    _ = dit  # silence unused when imported only for side effects
    return redundancy, unique_i, unique_j, synergy, mi_i, mi_j, mi_joint


def _clip_atoms(
    redundancy: float,
    unique_i: float,
    unique_j: float,
    synergy: float,
    *,
    enabled: bool,
) -> tuple[float, float, float, float]:
    """Optionally zero out numerical noise below machine precision."""
    if not enabled:
        return redundancy, unique_i, unique_j, synergy
    atoms = [redundancy, unique_i, unique_j, synergy]
    cleaned = [0.0 if abs(v) < 1e-10 else float(v) for v in atoms]
    return cleaned[0], cleaned[1], cleaned[2], cleaned[3]


def decompose_pair_discrete(
    scores_i: NDArray[np.floating],
    scores_j: NDArray[np.floating],
    y: Sequence[Any],
    config: AdvancedPIDConfig,
    *,
    pair_seed: int = 0,
) -> tuple[float, float, float, float, float, float, float, int]:
    """Discretise two score vectors and run the configured PID estimator.

    Returns
    -------
    redundancy, unique_i, unique_j, synergy, mi_i, mi_j, mi_joint, n_bins_used
    """
    labels = np.asarray(list(y), dtype=object)
    n = int(labels.size)
    n_classes = int(len(np.unique(labels)))
    n_bins = adaptive_n_bins(
        n, config.n_bins, n_classes, max_bins=config.max_bins
    )
    codes_i = quantile_bin_codes(scores_i, n_bins, seed=config.seed + pair_seed)
    codes_j = quantile_bin_codes(scores_j, n_bins, seed=config.seed + pair_seed + 17)
    dist = empirical_joint_distribution(codes_i, codes_j, labels)
    red, u_i, u_j, syn, mi_i, mi_j, mi_ij = _broja_or_ccs_atoms(dist, config.estimator)
    red, u_i, u_j, syn = _clip_atoms(
        red, u_i, u_j, syn, enabled=config.clip_neg_atoms and config.estimator == "broja"
    )
    return red, u_i, u_j, syn, mi_i, mi_j, mi_ij, n_bins


def pairwise_pid_advanced(
    features: pd.DataFrame,
    y: pd.Series | Sequence[Any],
    *,
    config: AdvancedPIDConfig | None = None,
    scopes: MappingScopes | None = None,
    scores: pd.DataFrame | None = None,
) -> list[AdvancedPIDAtom]:
    """PID for all six family pairs, globally and inside each dynamic group.

    Parameters
    ----------
    features
        Clean descriptor matrix (columns prefixed by family). Ignored when
        ``scores`` is provided.
    y
        Regime labels aligned with ``features`` / ``scores``.
    config
        Estimator and discretisation options.
    scopes
        Mapping ``scope_name -> iterable of regime names``. Defaults to
        ``{"global": all}`` plus :data:`DYNAMIC_GROUPS`.
    scores
        Optional precomputed family PCA scores (avoids recomputing SVD).
    """
    cfg = config or AdvancedPIDConfig()
    labels = pd.Series(list(y), dtype=object)
    if scores is None:
        scores = family_principal_scores(features)
    else:
        missing = [f for f in FAMILY_ORDER if f not in scores.columns]
        if missing:
            raise ValueError(f"scores missing families {missing}")

    scope_map = scopes if scopes is not None else _default_scopes(labels)
    atoms: list[AdvancedPIDAtom] = []
    pair_index = 0
    for scope_name, members in scope_map.items():
        member_set = set(members)
        mask = labels.isin(member_set).to_numpy()
        n = int(mask.sum())
        if n < cfg.min_samples:
            logger.warning(
                "Skipping PID scope=%s group=%s (n=%d < min_samples=%d)",
                "group" if scope_name != "global" else "global",
                scope_name,
                n,
                cfg.min_samples,
            )
            continue
        y_scope = labels.to_numpy()[mask]
        logger.info(
            "PID-%s scope=%-16s n=%d estimator=%s",
            cfg.estimator,
            scope_name,
            n,
            cfg.estimator,
        )
        for left, right in combinations(FAMILY_ORDER, 2):
            pair_index += 1
            try:
                red, u_i, u_j, syn, mi_i, mi_j, mi_ij, n_bins = decompose_pair_discrete(
                    scores.loc[mask, left].to_numpy(dtype=float),
                    scores.loc[mask, right].to_numpy(dtype=float),
                    y_scope,
                    cfg,
                    pair_seed=pair_index * 101,
                )
            except Exception:
                logger.exception(
                    "PID failed for %s | %s at scope %s", left, right, scope_name
                )
                continue
            atom = AdvancedPIDAtom(
                scope="global" if scope_name == "global" else "group",
                group="" if scope_name == "global" else scope_name,
                family_i=left,
                family_j=right,
                estimator=cfg.estimator,
                n_bins=n_bins,
                n_samples=n,
                redundancy=red,
                unique_i=u_i,
                unique_j=u_j,
                synergy=syn,
                mi_i=mi_i,
                mi_j=mi_j,
                mi_joint=mi_ij,
            )
            atoms.append(atom)
            logger.info(
                "  %s red=%.4f u_%s=%.4f u_%s=%.4f syn=%.4f (bins=%d)",
                atom.pair_label,
                red,
                FAMILY_SHORT[left],
                u_i,
                FAMILY_SHORT[right],
                u_j,
                syn,
                n_bins,
            )
    return atoms


def _default_scopes(labels: pd.Series) -> MappingScopes:
    """Global panel plus the three RQ1 macro-groups."""
    scopes: MappingScopes = {"global": tuple(sorted(set(labels.astype(str))))}
    for name, members in DYNAMIC_GROUPS.items():
        scopes[name] = members
    return scopes


def atoms_to_frame(atoms: Sequence[AdvancedPIDAtom]) -> pd.DataFrame:
    """Tabular export of :class:`AdvancedPIDAtom` rows."""
    if not atoms:
        return pd.DataFrame()
    return pd.DataFrame([atom.as_dict() for atom in atoms])


@dataclass
class OneVsRestPIDAtom:
    """Bivariate PID of one family against the union of the other three."""

    family: str
    estimator: str
    n_bins: int
    n_samples: int
    redundancy: float
    unique_family: float
    unique_rest: float
    synergy: float
    mi_family: float
    mi_rest: float
    mi_joint: float

    @property
    def pair_label(self) -> str:
        return f"{FAMILY_SHORT[self.family]} | rest"

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["pair_label"] = self.pair_label
        payload["conservation_residual"] = float(
            self.mi_joint
            - (
                self.redundancy
                + self.unique_family
                + self.unique_rest
                + self.synergy
            )
        )
        return payload


def rest_principal_score(
    scores: pd.DataFrame,
    family: str,
) -> NDArray[np.floating]:
    """Leading PC of the three family scores complementary to ``family``."""
    from tsg.audit.information import first_principal_component

    if family not in FAMILY_ORDER:
        raise ValueError(f"Unknown family {family!r}")
    others = [name for name in FAMILY_ORDER if name != family]
    block = scores.loc[:, others].to_numpy(dtype=float)
    return first_principal_component(block)


def one_vs_rest_pid(
    features: pd.DataFrame,
    y: pd.Series | Sequence[Any],
    *,
    config: AdvancedPIDConfig | None = None,
    scores: pd.DataFrame | None = None,
    estimators: Sequence[PidEstimator] = ("broja", "mmi_discrete"),
) -> list[OneVsRestPIDAtom]:
    """Four bivariate PIDs :math:`X_i \\mid X_{-i}` for each TS2G family.

    ``rest`` is the leading principal component of the other three family
    PCA scores (already aligned with the panel). Both BROJA and discrete MMI
    are evaluated on the same quantile-binned joint so the unique-atom bias of
    MMI is directly comparable.
    """
    cfg = config or AdvancedPIDConfig()
    labels = pd.Series(list(y), dtype=object)
    if scores is None:
        scores = family_principal_scores(features)
    n = int(len(labels))
    if n < cfg.min_samples:
        logger.warning(
            "Skipping one-vs-rest PID (n=%d < min_samples=%d)", n, cfg.min_samples
        )
        return []

    atoms: list[OneVsRestPIDAtom] = []
    for fam_idx, family in enumerate(FAMILY_ORDER):
        family_scores = scores[family].to_numpy(dtype=float)
        rest_scores = rest_principal_score(scores, family)
        for est_idx, estimator in enumerate(estimators):
            local_cfg = AdvancedPIDConfig(
                estimator=estimator,
                n_bins=cfg.n_bins,
                min_samples=cfg.min_samples,
                seed=cfg.seed,
                clip_neg_atoms=cfg.clip_neg_atoms,
                max_bins=cfg.max_bins,
            )
            try:
                red, u_fam, u_rest, syn, mi_f, mi_r, mi_ij, n_bins = (
                    decompose_pair_discrete(
                        family_scores,
                        rest_scores,
                        labels,
                        local_cfg,
                        pair_seed=1000 + fam_idx * 50 + est_idx,
                    )
                )
            except Exception:
                logger.exception(
                    "One-vs-rest PID failed for family=%s estimator=%s",
                    family,
                    estimator,
                )
                continue
            atom = OneVsRestPIDAtom(
                family=family,
                estimator=estimator,
                n_bins=n_bins,
                n_samples=n,
                redundancy=red,
                unique_family=u_fam,
                unique_rest=u_rest,
                synergy=syn,
                mi_family=mi_f,
                mi_rest=mi_r,
                mi_joint=mi_ij,
            )
            atoms.append(atom)
            logger.info(
                "PID 1-vs-rest %s [%s] red=%.4f u_fam=%.4f u_rest=%.4f syn=%.4f",
                atom.pair_label,
                estimator,
                red,
                u_fam,
                u_rest,
                syn,
            )
    return atoms


def one_vs_rest_to_frame(atoms: Sequence[OneVsRestPIDAtom]) -> pd.DataFrame:
    """Tabular export of :class:`OneVsRestPIDAtom` rows."""
    if not atoms:
        return pd.DataFrame()
    return pd.DataFrame([atom.as_dict() for atom in atoms])
