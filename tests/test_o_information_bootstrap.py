"""Tests for bootstrap CIs on Gaussian O-information."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from itertools import combinations

import pandas as pd

from tsg.audit.information import FAMILY_ORDER
from tsg.audit.o_information_bootstrap import (
    OmegaBootstrapConfig,
    bootstrap_indices,
    bootstrap_omega,
    holm_adjust,
    o_information_bootstrap_report,
    permutation_pvalue_omega,
    two_sided_pvalue_vs_zero,
)
from tsg.audit.shapley_interaction import exact_sii_pairwise
from tsg.audit.stress_eval import add_white_noise_snr, multiclass_log_loss, truncate_series


def test_bootstrap_indices_shape_and_range() -> None:
    idx = bootstrap_indices(30, 50, seed=0)
    assert idx.shape == (50, 30)
    assert idx.min() == 0
    assert idx.max() == 29


def test_pvalue_vs_zero_on_clearly_positive_sample() -> None:
    reps = np.full(200, 0.5)
    assert two_sided_pvalue_vs_zero(reps) == pytest.approx(1.0 / 201.0)


def test_pvalue_vs_zero_on_symmetric_noise() -> None:
    rng = np.random.default_rng(0)
    reps = rng.normal(0.0, 0.05, size=500)
    p = two_sided_pvalue_vs_zero(reps)
    assert 0.05 < p <= 1.0


def test_bootstrap_omega_ci_covers_independent_case() -> None:
    rng = np.random.default_rng(1)
    matrix = rng.normal(size=(80, 4))
    cfg = OmegaBootstrapConfig(n_bootstrap=200, n_permutations=80, seed=1)
    omega, lo, hi, p_boot, p_perm, reps = bootstrap_omega(matrix, cfg)
    assert lo <= omega <= hi
    assert reps.shape == (200,)
    assert 0.0 < p_perm <= 1.0
    # Independent Gaussians → Ω near 0; CI should typically contain 0.
    assert lo < 0.0 < hi or abs(omega) < 0.15


def test_bootstrap_omega_detects_redundant_copies() -> None:
    rng = np.random.default_rng(2)
    base = rng.normal(size=(100, 1))
    matrix = np.hstack([base, base, base, base])
    cfg = OmegaBootstrapConfig(n_bootstrap=150, n_permutations=100, seed=2, ridge=1e-4)
    omega, lo, hi, p_boot, p_perm, _ = bootstrap_omega(matrix, cfg)
    assert omega > 0.5
    assert lo > 0.0
    assert p_boot < 0.05
    assert p_perm < 0.05


def test_holm_adjust_is_monotone_and_bounded() -> None:
    adjusted = holm_adjust([0.01, 0.04, 0.03, 0.20])
    assert all(0.0 <= p <= 1.0 for p in adjusted)
    # Smallest raw p gets the strongest multiplier but step-down keeps order.
    assert adjusted[0] <= adjusted[2] <= adjusted[1] or adjusted[0] <= adjusted[1]


def test_permutation_pvalue_in_unit_interval() -> None:
    rng = np.random.default_rng(9)
    matrix = rng.normal(size=(60, 4))
    from tsg.audit.information import o_information_gaussian

    omega = float(o_information_gaussian(matrix, ridge=1e-3))
    p = permutation_pvalue_omega(
        matrix, omega, n_permutations=80, seed=9, ridge=1e-3
    )
    assert 0.0 < p <= 1.0


def test_report_includes_groups_and_regimes() -> None:
    rng = np.random.default_rng(3)
    n = 90
    scores = pd.DataFrame(
        rng.normal(size=(n, 4)),
        columns=list(FAMILY_ORDER),
    )
    y = pd.Series(
        ["white_noise"] * 30 + ["lorenz"] * 30 + ["logistic"] * 30
    )
    cfg = OmegaBootstrapConfig(
        n_bootstrap=40, n_permutations=30, seed=3, min_samples=8
    )
    results = o_information_bootstrap_report(scores, y, config=cfg)
    scopes = {r.scope for r in results}
    assert "global" in scopes
    assert "group" in scopes
    assert "regime" in scopes
    global_row = next(r for r in results if r.scope == "global")
    assert np.isfinite(global_row.ci_low)
    assert global_row.n_bootstrap == 40
    assert np.isfinite(global_row.p_perm)
    assert np.isfinite(global_row.p_holm)
    assert all(np.isfinite(r.p_holm) for r in results)


def test_exact_sii_zero_on_additive_game() -> None:
    # v(S) = |S| → pairwise interactions vanish.
    players = ("A", "B", "C")
    values = {
        frozenset(subset): float(len(subset))
        for size in range(4)
        for subset in combinations(players, size)
    }
    sii = exact_sii_pairwise(values, players)
    assert all(abs(v) < 1e-12 for v in sii.values())


def test_exact_sii_positive_on_synergistic_pair() -> None:
    players = ("A", "B", "C")
    values = {frozenset(): 0.0}
    for size in range(1, 4):
        for subset in combinations(players, size):
            key = frozenset(subset)
            # Bonus only when A and B are both present.
            values[key] = float(len(subset)) + (0.5 if {"A", "B"} <= key else 0.0)
    sii = exact_sii_pairwise(values, players)
    assert sii[("A", "B")] == pytest.approx(0.5, abs=1e-9)
    assert abs(sii[("A", "C")]) < 1e-9


def test_snr_noise_matches_target_approximately() -> None:
    rng = np.random.default_rng(4)
    x = rng.normal(size=20_000)
    noisy = add_white_noise_snr(x, snr_db=10.0, seed=4)
    signal_var = float(np.var(x))
    noise_var = float(np.var(noisy - x))
    snr = 10.0 * np.log10(signal_var / noise_var)
    assert snr == pytest.approx(10.0, abs=0.3)


def test_truncate_and_log_loss_helpers() -> None:
    x = np.arange(50, dtype=float)
    assert truncate_series(x, 10).shape == (10,)
    loss = multiclass_log_loss(
        ["a", "b"],
        np.array([[0.9, 0.1], [0.2, 0.8]]),
        labels=["a", "b"],
    )
    assert loss > 0.0


def test_sii_report_from_information_metrics_json() -> None:
    from pathlib import Path

    from tsg.audit.shapley_interaction import sii_report_from_metrics_json

    path = Path("outputs/reports/phase2_information/information_metrics.json")
    if not path.exists():
        pytest.skip("information_metrics.json not present")
    report = sii_report_from_metrics_json(path, n_bootstrap=200, seed=0)
    assert "random_forest" in report.by_model
    assert "logistic" in report.by_model
    rf = report.by_model["random_forest"]
    assert rf.v_empty == pytest.approx(0.012820512820512822, abs=1e-9)
    assert rf.v_full == pytest.approx(1.0, abs=1e-9)
    assert abs(sum(rf.phi.values()) + rf.v_empty - rf.v_full) < 1e-9
    assert len(rf.sii) == 6
    assert len(rf.subset_scores) == 16
    assert rf.n_bootstrap == 200
    assert set(rf.phi_std) == set(rf.phi)
    assert all(np.isfinite(v) for v in rf.phi_std.values())
    assert all(rf.phi_ci_low[f] <= rf.phi[f] <= rf.phi_ci_high[f] for f in rf.phi)


def test_mmi_broja_contrast_table() -> None:
    from pathlib import Path

    from tsg.audit.pid_advanced import AdvancedPIDAtom
    from tsg.audit.pid_contrast import (
        contrast_mmi_vs_broja,
        contrast_to_markdown,
        mmi_atoms_from_metrics_json,
    )

    path = Path("outputs/reports/phase2_information/information_metrics.json")
    if not path.exists():
        pytest.skip("information_metrics.json not present")
    mmi = mmi_atoms_from_metrics_json(path)
    # Synthetic BROJA atoms with non-zero uniques where MMI zeros one side.
    broja_atoms = []
    for _, row in mmi.iterrows():
        broja_atoms.append(
            AdvancedPIDAtom(
                scope="global",
                group="",
                family_i=row["family_i"],
                family_j=row["family_j"],
                estimator="broja",
                n_bins=3,
                n_samples=360,
                redundancy=0.5,
                unique_i=0.2,
                unique_j=0.15,
                synergy=0.1,
                mi_i=0.7,
                mi_j=0.65,
                mi_joint=0.95,
            )
        )
    from tsg.audit.pid_contrast import broja_global_frame

    contrast = contrast_mmi_vs_broja(mmi, broja_global_frame(broja_atoms))
    assert len(contrast) == 6
    assert contrast["mmi_forced_unique_zero"].all()
    assert contrast["broja_both_unique_positive"].all()
    md = contrast_to_markdown(contrast)
    assert "MMI vs BROJA" in md


def test_stress_conditions_include_snr_and_short_n() -> None:
    from tsg.audit.stress_eval import StressConfig, build_stress_conditions

    conds = build_stress_conditions(StressConfig())
    names = [c.name for c in conds]
    assert "baseline_clean" in names
    assert "snr_10db" in names
    assert "snr_5db" in names
    assert "short_n200" in names


def test_group_coverage_helper_on_balanced_panel() -> None:
    from tsg.audit.information import DYNAMIC_GROUPS
    from tsg.audit.rq1_pipeline import verify_dynamic_group_coverage

    regimes = [r for members in DYNAMIC_GROUPS.values() for r in members for _ in range(5)]
    y = pd.Series(regimes)
    report = verify_dynamic_group_coverage(y)
    assert report["ok"] is True
    assert report["n_covered"] == len(y)
    assert report["missing_regimes"] == []
