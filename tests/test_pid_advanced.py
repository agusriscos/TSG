"""Tests for BROJA / discrete PID on family PCA scores."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

dit = pytest.importorskip("dit")
from dit.pid.measures.ibroja import PID_BROJA  # noqa: E402

from tsg.audit.pid_advanced import (
    AdvancedPIDConfig,
    adaptive_n_bins,
    decompose_pair_discrete,
    empirical_joint_distribution,
    one_vs_rest_pid,
    pairwise_pid_advanced,
    quantile_bin_codes,
)


def test_quantile_bins_are_balanced_on_uniform() -> None:
    rng = np.random.default_rng(0)
    x = rng.uniform(size=400)
    codes = quantile_bin_codes(x, n_bins=4, seed=0)
    counts = np.bincount(codes, minlength=4)
    assert codes.min() == 0 and codes.max() == 3
    assert counts.max() - counts.min() <= 2


def test_adaptive_bins_shrink_for_sparse_joints() -> None:
    # chaotic_flows-like panel: n=60, 2 classes → hard-capped at 3.
    assert adaptive_n_bins(60, requested=8, n_classes=2) == 3
    assert adaptive_n_bins(30, requested=8, n_classes=5) <= 3
    # Large panel may use up to max_bins=4.
    assert adaptive_n_bins(1000, requested=4, n_classes=3, max_bins=4) == 4
    assert adaptive_n_bins(1000, requested=8, n_classes=3, max_bins=4) == 4


def test_broja_recovers_xor_synergy() -> None:
    rng = np.random.default_rng(1)
    n = 1500
    x1 = rng.integers(0, 2, size=n)
    x2 = rng.integers(0, 2, size=n)
    y = x1 ^ x2
    dist = empirical_joint_distribution(x1, x2, y)
    pid = PID_BROJA(dist, [["Si"], ["Sj"]], ["Y"])
    synergy = float(pid.get_pi((("Si", "Sj"),)))
    redundancy = float(pid.get_pi((("Si",), ("Sj",))))
    assert synergy == pytest.approx(1.0, abs=0.05)
    assert redundancy == pytest.approx(0.0, abs=0.05)


def test_broja_allows_nonzero_unique_on_both_sides() -> None:
    """Unlike MMI, BROJA can assign unique information to both sources."""
    rng = np.random.default_rng(2)
    n = 2000
    # Si and Sj each carry partially distinct signal about a 3-class Y.
    latent = rng.integers(0, 3, size=n)
    noise_i = rng.normal(size=n)
    noise_j = rng.normal(size=n)
    scores_i = latent.astype(float) + 0.35 * noise_i
    scores_j = latent.astype(float) + 0.35 * noise_j + 0.8 * (latent == 2)
    y = [f"c{v}" for v in latent]
    cfg = AdvancedPIDConfig(estimator="broja", n_bins=4, seed=2)
    red, u_i, u_j, syn, *_rest = decompose_pair_discrete(
        scores_i, scores_j, y, cfg, pair_seed=0
    )
    assert red >= -1e-6
    # At least one unique atom should be strictly positive (MMI often zeros one).
    assert max(u_i, u_j) > 1e-4
    assert np.isfinite(syn)


def test_pairwise_pid_advanced_emits_global_and_groups() -> None:
    rng = np.random.default_rng(3)
    n = 120
    # Fake family blocks with discriminative signal.
    columns = []
    data = []
    for family in ("visibility", "proximity", "transition", "similarity"):
        for k in range(3):
            columns.append(f"{family}__m{k}")
            data.append(rng.normal(size=n))
    features = pd.DataFrame(np.column_stack(data), columns=columns)
    regimes = (
        ["white_noise"] * 20
        + ["brownian"] * 20
        + ["lorenz"] * 20
        + ["rossler"] * 20
        + ["logistic"] * 20
        + ["henon"] * 20
    )
    # Inject regime means into visibility / proximity so PID is non-trivial.
    for idx, regime in enumerate(regimes):
        features.loc[idx, "visibility__m0"] += hash(regime) % 7
        features.loc[idx, "proximity__m0"] += (hash(regime) % 5) * 0.5

    y = pd.Series(regimes)
    atoms = pairwise_pid_advanced(
        features,
        y,
        config=AdvancedPIDConfig(estimator="broja", n_bins=3, seed=3, min_samples=20),
    )
    scopes = {(a.scope, a.group) for a in atoms}
    assert ("global", "") in scopes
    assert ("group", "stochastic") in scopes
    assert ("group", "chaotic_flows") in scopes
    assert ("group", "discrete_maps") in scopes
    # structured_other absent here (no quasiperiodic / regime_switching in y).
    # Six pairs × four scopes present in this fixture.
    assert len(atoms) == 24
    assert all(a.n_bins <= 3 for a in atoms if a.n_samples < 120)
    # BROJA should not force every unique_i to zero across all pairs.
    assert any(abs(a.unique_i) > 1e-6 and abs(a.unique_j) > 1e-6 for a in atoms) or any(
        abs(a.unique_i) > 1e-6 for a in atoms
    )


def test_one_vs_rest_emits_four_families_times_two_estimators() -> None:
    rng = np.random.default_rng(4)
    n = 120
    columns = []
    data = []
    for family in ("visibility", "proximity", "transition", "similarity"):
        for k in range(3):
            columns.append(f"{family}__m{k}")
            data.append(rng.normal(size=n))
    features = pd.DataFrame(np.column_stack(data), columns=columns)
    regimes = (
        ["white_noise"] * 30
        + ["lorenz"] * 30
        + ["logistic"] * 30
        + ["henon"] * 30
    )
    for idx, regime in enumerate(regimes):
        features.loc[idx, "visibility__m0"] += hash(regime) % 7
    y = pd.Series(regimes)
    atoms = one_vs_rest_pid(
        features,
        y,
        config=AdvancedPIDConfig(estimator="broja", n_bins=3, seed=4, min_samples=20),
        estimators=("broja", "mmi_discrete"),
    )
    assert len(atoms) == 8
    families = {a.family for a in atoms}
    estimators = {a.estimator for a in atoms}
    assert families == {"visibility", "proximity", "transition", "similarity"}
    assert estimators == {"broja", "mmi_discrete"}
    for atom in atoms:
        resid = (
            atom.redundancy
            + atom.unique_family
            + atom.unique_rest
            + atom.synergy
            - atom.mi_joint
        )
        assert abs(resid) < 1e-6
