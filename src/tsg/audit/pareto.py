"""Compute-cost benchmark and Pareto front across TS2G family subsets."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Sequence

import numpy as np
from numpy.typing import NDArray

from tsg.audit.config import AuditConfig
from tsg.audit.ensemble import _prepare_series_for_method, load_generator_suite_config, realization_seed
from tsg.descriptors.pipeline import DescriptorPipeline
from tsg.generators.config import GeneratorSuiteConfig
from tsg.generators.registry import REGIME_LABELS, build_generator
from tsg.graphs.config import TS2GSuiteConfig
from tsg.graphs.registry import build_transformers

logger = logging.getLogger(__name__)


def timing_lengths_for_profile(profile: str | None) -> tuple[int, ...]:
    """Series lengths timed for each benchmark profile.

    ``dev`` times the same length as the cached dev panel (``N=1000``).
    ``paper`` reports scaling at ``N ∈ {500, 1000, 1500}``.
    """
    if profile == "paper":
        return (500, 1000, 1500)
    return (1000,)


def pareto_front_mask(cost: np.ndarray, score: np.ndarray) -> np.ndarray:
    """Mark points that no other point strictly dominates.

    A point dominates another when it is at least as cheap and at least as
    accurate, and strictly better in one of the two. Cost is minimized and
    score is maximized. Ties (equal cost and equal score) stay on the front.
    """
    costs = np.asarray(cost, dtype=float).reshape(-1)
    scores = np.asarray(score, dtype=float).reshape(-1)
    if costs.shape != scores.shape:
        raise ValueError(
            f"cost and score must have the same length, got {costs.shape} and {scores.shape}"
        )
    if costs.size == 0:
        return np.zeros(0, dtype=bool)
    if not (np.isfinite(costs).all() and np.isfinite(scores).all()):
        raise ValueError("cost and score must be finite")
    # dominates[i, j] is True when j dominates i.
    cheaper_or_equal = costs[None, :] <= costs[:, None]
    better_or_equal = scores[None, :] >= scores[:, None]
    strictly = (costs[None, :] < costs[:, None]) | (scores[None, :] > scores[:, None])
    dominated = (cheaper_or_equal & better_or_equal & strictly).any(axis=1)
    return ~dominated


def mean_and_sample_std(values: NDArray[np.floating]) -> tuple[float, float]:
    """Return the mean and the sample standard deviation (``ddof=1``)."""
    data = np.asarray(values, dtype=float).reshape(-1)
    if data.size == 0:
        return float("nan"), float("nan")
    if data.size == 1:
        return float(data[0]), 0.0
    return float(data.mean()), float(data.std(ddof=1))


@dataclass(frozen=True)
class SeriesFamilyTiming:
    """Per-series graph + descriptor time, in milliseconds.

    ``time_ms`` has shape ``(n_series, n_families)`` and is aligned with
    ``dataset_ids`` and ``families``.
    """

    n_steps: int
    dataset_ids: tuple[str, ...]
    families: tuple[str, ...]
    time_ms: np.ndarray

    def column(self, family: str) -> np.ndarray:
        index = self.families.index(family)
        return np.asarray(self.time_ms[:, index], dtype=float)


def subset_cost_stats(
    timing: SeriesFamilyTiming,
    families: Sequence[str],
) -> tuple[float, float]:
    """Mean and sample std of the summed family times, in ms per series."""
    if not families:
        zeros = np.zeros(len(timing.dataset_ids), dtype=float)
        return mean_and_sample_std(zeros)
    total = np.zeros(len(timing.dataset_ids), dtype=float)
    for family in families:
        total += timing.column(family)
    return mean_and_sample_std(total)


def _generate_values(
    dataset_id: str,
    *,
    n_steps: int,
    seed: int,
    generator_config: GeneratorSuiteConfig,
) -> np.ndarray:
    from dataclasses import replace

    config = replace(
        generator_config,
        global_params=replace(
            generator_config.global_params,
            n_steps=n_steps,
            seed=seed,
            n_samples=1,
        ),
    )
    series = build_generator(dataset_id, config).generate(n_steps, seed=seed)
    return np.asarray(series.values, dtype=float).ravel()


def _time_family_ms(
    values: np.ndarray,
    *,
    dataset_id: str,
    family: str,
    methods: Sequence[str],
    transformers: dict,
    pipeline: DescriptorPipeline,
    audit_config: AuditConfig,
) -> float:
    """Wall time to build every graph in ``family`` and extract its descriptors."""
    regime = REGIME_LABELS[dataset_id]
    elapsed = 0.0
    for method in methods:
        prepared = _prepare_series_for_method(values, dataset_id, method, audit_config)
        started = time.perf_counter()
        graph_result = transformers[method].transform(
            prepared,
            dataset_id,
            regime=regime,
        )
        pipeline.compute(
            graph_result.graph,
            dataset_id=dataset_id,
            method=method,
            regime=regime,
            family=graph_result.family,
        )
        elapsed += time.perf_counter() - started
    return elapsed * 1000.0


def benchmark_family_times(
    config: AuditConfig,
    *,
    families: Sequence[str],
    lengths: Sequence[int],
) -> dict[int, SeriesFamilyTiming]:
    """Time graph construction plus descriptors for each family and length.

    One realization per regime (seed of realization 0) enters the mean. A
    discarded warm-up series absorbs importer and first-call costs so they do
    not land on the first regime. Series generation itself is not timed: the
    clock covers the TS2G map and the descriptor extraction only.
    """
    family_list = tuple(families)
    missing = [name for name in family_list if name not in config.families]
    if missing:
        raise KeyError(f"Families missing from the audit config: {missing}")
    generator_config = load_generator_suite_config()
    methods = tuple(dict.fromkeys(m for name in family_list for m in config.families[name]))
    transformers = {
        transformer.method: transformer
        for transformer in build_transformers(TS2GSuiteConfig(methods=methods))
    }
    pipeline = DescriptorPipeline()
    warmup_id = "white_noise" if "white_noise" in config.regimes else config.regimes[0]
    warmup = _generate_values(
        warmup_id,
        n_steps=200,
        seed=int(config.seed),
        generator_config=generator_config,
    )
    for family in family_list:
        _time_family_ms(
            warmup,
            dataset_id=warmup_id,
            family=family,
            methods=config.families[family],
            transformers=transformers,
            pipeline=pipeline,
            audit_config=config,
        )
    logger.info("Discarded timing warm-up on %s (N=200)", warmup_id)

    measured: dict[int, SeriesFamilyTiming] = {}
    dataset_ids = tuple(config.regimes)
    for n_steps in lengths:
        rows = np.zeros((len(dataset_ids), len(family_list)), dtype=float)
        for row, dataset_id in enumerate(dataset_ids):
            seed = realization_seed(config.seed, dataset_id, 0)
            values = _generate_values(
                dataset_id,
                n_steps=int(n_steps),
                seed=seed,
                generator_config=generator_config,
            )
            logger.info(
                "Timing N=%d series %d/%d dataset=%s",
                n_steps,
                row + 1,
                len(dataset_ids),
                dataset_id,
            )
            for column, family in enumerate(family_list):
                rows[row, column] = _time_family_ms(
                    values,
                    dataset_id=dataset_id,
                    family=family,
                    methods=config.families[family],
                    transformers=transformers,
                    pipeline=pipeline,
                    audit_config=config,
                )
        measured[int(n_steps)] = SeriesFamilyTiming(
            n_steps=int(n_steps),
            dataset_ids=dataset_ids,
            families=family_list,
            time_ms=rows,
        )
        means = ", ".join(
            f"{family}={rows[:, idx].mean():.1f} ms"
            for idx, family in enumerate(family_list)
        )
        logger.info("Timing N=%d mean/series: %s", n_steps, means)
    return measured
