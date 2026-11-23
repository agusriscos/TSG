"""Tests for PID atoms, exact Shapley, Gaussian O-information and Pareto."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tsg.audit.config import AuditConfig
from tsg.audit.information import (
    DYNAMIC_GROUPS,
    FAMILY_ORDER,
    discrete_entropy_bits,
    exact_shapley,
    evaluate_subset_scores,
    gaussian_entropy_bits,
    majority_baseline_macro_f1,
    mmi_decomposition,
    nonempty_family_subsets,
    o_information_from_entropies,
    o_information_gaussian,
    oof_logistic_proba,
    predictive_mi_bits,
    subset_label,
)
from tsg.audit.pareto import SeriesFamilyTiming, pareto_front_mask, subset_cost_stats


def test_mmi_nonnegative_unique_and_exact_conservation() -> None:
    redundancy, unique_i, unique_j, synergy = mmi_decomposition(0.5, 0.8, 1.0)
    assert redundancy == pytest.approx(0.5)
    assert unique_i == pytest.approx(0.0)
    assert unique_j == pytest.approx(0.3)
    assert synergy == pytest.approx(0.2)
    assert redundancy + unique_i + unique_j + synergy == pytest.approx(1.0)
    assert redundancy >= 0.0
    assert unique_i >= 0.0
    assert unique_j >= 0.0


def test_mmi_synergy_may_be_negative_but_atoms_still_add_up() -> None:
    redundancy, unique_i, unique_j, synergy = mmi_decomposition(1.0, 1.0, 0.5)
    assert redundancy == pytest.approx(1.0)
    assert unique_i == pytest.approx(0.0)
    assert unique_j == pytest.approx(0.0)
    assert synergy == pytest.approx(-0.5)
    assert redundancy >= 0.0 and unique_i >= 0.0 and unique_j >= 0.0
    assert redundancy + unique_i + unique_j + synergy == pytest.approx(0.5)


def test_miller_madow_shift_only_moves_redundancy() -> None:
    base = mmi_decomposition(1.0, 0.8, 1.2)
    shifted = mmi_decomposition(1.0 + 0.066, 0.8 + 0.066, 1.2 + 0.066)
    assert shifted[1] == pytest.approx(base[1])
    assert shifted[2] == pytest.approx(base[2])
    assert shifted[3] == pytest.approx(base[3])
    assert shifted[0] == pytest.approx(base[0] + 0.066)


def test_mmi_conservation_on_random_nonnegative_inputs() -> None:
    rng = np.random.default_rng(1)
    for _ in range(30):
        mi_i, mi_j = rng.random(2)
        mi_joint = float(rng.random() + max(mi_i, mi_j))
        redundancy, unique_i, unique_j, synergy = mmi_decomposition(mi_i, mi_j, mi_joint)
        assert redundancy >= -1e-12
        assert unique_i >= -1e-12
        assert unique_j >= -1e-12
        assert redundancy + unique_i + unique_j + synergy == pytest.approx(mi_joint)


def test_discrete_entropy_of_balanced_alphabet() -> None:
    labels = np.tile(np.arange(4), 25)
    plugin = discrete_entropy_bits(labels, miller_madow=False)
    corrected = discrete_entropy_bits(labels, miller_madow=True)
    assert plugin == pytest.approx(2.0)
    assert corrected == pytest.approx(2.0 + 3.0 / (2.0 * 100.0 * np.log(2.0)))


def test_gaussian_entropy_matches_standard_normal() -> None:
    rng = np.random.default_rng(0)
    sample = rng.normal(size=(20000, 1))
    entropy = gaussian_entropy_bits(sample, ridge=0.0, bias_correction=True)
    truth = 0.5 * np.log2(2.0 * np.pi * np.e)
    assert entropy == pytest.approx(truth, abs=0.02)


def test_gaussian_entropy_bias_correction_on_small_samples() -> None:
    rng = np.random.default_rng(1)
    truth = 0.5 * np.log2(2.0 * np.pi * np.e)
    raw_error = []
    corrected_error = []
    for _ in range(300):
        sample = rng.normal(size=(30, 1))
        raw_error.append(gaussian_entropy_bits(sample, ridge=0.0, bias_correction=False) - truth)
        corrected_error.append(
            gaussian_entropy_bits(sample, ridge=0.0, bias_correction=True) - truth
        )
    assert abs(float(np.mean(corrected_error))) < abs(float(np.mean(raw_error)))
    assert abs(float(np.mean(corrected_error))) < 0.03


def test_o_information_formula_independent_and_copies() -> None:
    independent = o_information_from_entropies(4.0, [1.0, 1.0, 1.0, 1.0], [3.0, 3.0, 3.0, 3.0])
    copies = o_information_from_entropies(1.0, [1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0])
    assert independent == pytest.approx(0.0)
    assert copies == pytest.approx(2.0)


def test_o_information_of_two_variables_is_identically_zero() -> None:
    rng = np.random.default_rng(2)
    sample = rng.normal(size=(500, 2))
    sample[:, 1] *= 7.0
    assert o_information_gaussian(sample) == pytest.approx(0.0, abs=1e-9)


def test_gaussian_o_information_sign_on_known_dependence() -> None:
    rng = np.random.default_rng(3)
    independent = rng.normal(size=(8000, 4))
    assert abs(o_information_gaussian(independent, ridge=1e-6)) < 0.08

    shared = rng.normal(size=3000)
    copies = shared[:, None] + 0.01 * rng.normal(size=(3000, 4))
    assert o_information_gaussian(copies, ridge=1e-4) > 0.5


def test_exact_shapley_efficiency_on_four_players() -> None:
    weights = {name: float(2**index) for index, name in enumerate(FAMILY_ORDER)}

    def value_of(coalition: frozenset[str]) -> float:
        return float(sum(weights[player] for player in coalition))

    values: dict[frozenset[str], float] = {}
    players = FAMILY_ORDER
    for size in range(len(players) + 1):
        from itertools import combinations

        for coalition in combinations(players, size):
            values[frozenset(coalition)] = value_of(frozenset(coalition))
    phi = exact_shapley(values, players)
    for player in players:
        assert phi[player] == pytest.approx(weights[player])
    full = values[frozenset(players)]
    empty = values[frozenset()]
    assert sum(phi.values()) == pytest.approx(full - empty)
    assert abs(sum(phi.values()) - (full - empty)) < 1e-9


def test_exact_shapley_two_player_game() -> None:
    values = {
        frozenset(): 0.0,
        frozenset({"A"}): 3.0,
        frozenset({"B"}): 4.0,
        frozenset({"A", "B"}): 5.0,
    }
    phi = exact_shapley(values, ("A", "B"))
    assert phi["A"] == pytest.approx(2.0)
    assert phi["B"] == pytest.approx(3.0)
    assert phi["A"] + phi["B"] == pytest.approx(5.0)


def test_subset_catalog_has_fifteen_labels() -> None:
    subsets = nonempty_family_subsets()
    assert len(subsets) == 15
    assert len({subset_label(subset) for subset in subsets}) == 15
    assert subset_label(()) == "empty"
    assert subset_label(FAMILY_ORDER) == "full"


def test_dynamic_groups_are_disjoint() -> None:
    seen: set[str] = set()
    for members in DYNAMIC_GROUPS.values():
        assert seen.isdisjoint(members)
        seen.update(members)


def test_dynamic_groups_cover_paper_panel() -> None:
    """Quasiperiodic + regime_switching live in structured_other; 12 regimes total."""
    covered = {name for members in DYNAMIC_GROUPS.values() for name in members}
    expected = {
        "white_noise",
        "fbm_antipersistent",
        "brownian",
        "fbm_persistent",
        "arma_correlated",
        "quasiperiodic",
        "lorenz",
        "rossler",
        "henon",
        "logistic_weak",
        "logistic",
        "regime_switching",
    }
    assert covered == expected
    assert "structured_other" in DYNAMIC_GROUPS
    assert set(DYNAMIC_GROUPS["structured_other"]) == {
        "quasiperiodic",
        "regime_switching",
    }


def test_majority_baseline_on_balanced_three_classes() -> None:
    y = pd.Series(["a", "b", "c"] * 12)
    config = AuditConfig(n_folds=4, seed=0, regimes=("a", "b", "c"))
    mean, std = majority_baseline_macro_f1(y, config)
    assert mean == pytest.approx(1.0 / 6.0, abs=1e-9)
    assert std == pytest.approx(0.0, abs=1e-9)


def test_predictive_mi_ranks_informative_feature_above_noise() -> None:
    rng = np.random.default_rng(4)
    y = np.array([0, 1, 2] * 20)
    labels = pd.Series(y.astype(str))
    config = AuditConfig(n_folds=4, seed=0, regimes=("0", "1", "2"))
    h_y = discrete_entropy_bits(labels.to_numpy(), miller_madow=True)
    informative = pd.DataFrame({"visibility__M__f": y.astype(float) + 0.01 * rng.normal(size=y.size)})
    noise = pd.DataFrame({"visibility__M__f": rng.normal(size=y.size)})
    mi_info = predictive_mi_bits(oof_logistic_proba(informative, labels, config), h_y)
    mi_noise = predictive_mi_bits(oof_logistic_proba(noise, labels, config), h_y)
    assert mi_info > mi_noise + 0.4
    assert mi_noise < 0.45


def test_shapley_efficiency_of_logistic_characteristic_function() -> None:
    rng = np.random.default_rng(5)
    classes = np.array([0, 1, 2] * 12)
    matrix = rng.normal(size=(36, 8))
    matrix[:, 0] += classes * 2.0
    columns = [f"{family}__M__d{slot}" for family in FAMILY_ORDER for slot in range(2)]
    features = pd.DataFrame(matrix, columns=columns)
    labels = pd.Series(classes.astype(str))
    config = AuditConfig(n_folds=3, seed=0, regimes=("0", "1", "2"))
    evaluation = evaluate_subset_scores(features, labels, config, model_names=("logistic",))
    values = evaluation.scores["logistic"]
    assert len(values) == 16
    phi = exact_shapley(values, FAMILY_ORDER)
    full = values[frozenset(FAMILY_ORDER)]
    empty = values[frozenset()]
    assert abs(sum(phi.values()) - (full - empty)) < 1e-9


def test_pareto_front_on_a_known_set() -> None:
    cost = np.array([1.0, 2.0, 3.0, 0.5])
    score = np.array([0.5, 0.9, 0.8, 0.4])
    mask = pareto_front_mask(cost, score)
    assert list(mask) == [True, True, False, True]


def test_pareto_keeps_ties_and_drops_equal_cost_worse_score() -> None:
    ties = pareto_front_mask(np.array([1.0, 1.0]), np.array([0.5, 0.5]))
    assert list(ties) == [True, True]
    dominated = pareto_front_mask(np.array([1.0, 1.0]), np.array([0.4, 0.6]))
    assert list(dominated) == [False, True]


def test_subset_cost_adds_family_times() -> None:
    timing = SeriesFamilyTiming(
        n_steps=1000,
        dataset_ids=("a", "b", "c"),
        families=FAMILY_ORDER,
        time_ms=np.array(
            [
                [10.0, 20.0, 30.0, 40.0],
                [12.0, 18.0, 33.0, 37.0],
                [8.0, 22.0, 27.0, 43.0],
            ]
        ),
    )
    mean, std = subset_cost_stats(timing, FAMILY_ORDER)
    totals = timing.time_ms.sum(axis=1)
    assert mean == pytest.approx(float(totals.mean()))
    assert std == pytest.approx(float(totals.std(ddof=1)))
    pair_mean, pair_std = subset_cost_stats(timing, ("visibility", "similarity"))
    pair = timing.time_ms[:, 0] + timing.time_ms[:, 3]
    assert pair_mean == pytest.approx(float(pair.mean()))
    assert pair_std == pytest.approx(float(pair.std(ddof=1)))
