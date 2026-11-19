"""Multiregime dataset registry and suite orchestration."""

from __future__ import annotations

import logging
from typing import Sequence

from tsg.generators.base import GeneratedSeries, SeriesGenerator
from tsg.generators.chaotic import LogisticMap, LorenzX
from tsg.generators.config import GeneratorSuiteConfig
from tsg.generators.periodic import Quasiperiodic, SineWave
from tsg.generators.stochastic import BrownianMotion, WhiteNoise

logger = logging.getLogger(__name__)

DATASET_IDS: tuple[str, ...] = (
    "white_noise",
    "brownian",
    "sine",
    "quasiperiodic",
    "lorenz",
    "logistic",
)


def build_default_generators(
    config: GeneratorSuiteConfig | None = None,
) -> list[SeriesGenerator]:
    """Instantiate one generator per Dataset ID using Table 3.1 defaults."""
    cfg = config or GeneratorSuiteConfig()
    return [
        WhiteNoise(params=cfg.white_noise),
        BrownianMotion(params=cfg.brownian),
        SineWave(params=cfg.sine),
        Quasiperiodic(params=cfg.quasiperiodic),
        LorenzX(params=cfg.lorenz),
        LogisticMap(params=cfg.logistic),
    ]


def generate_suite(
    config: GeneratorSuiteConfig | None = None,
    *,
    generators: Sequence[SeriesGenerator] | None = None,
    n_samples: int | None = None,
) -> list[GeneratedSeries]:
    """Generate ``n_samples`` series per generator (Phase 1: ``n_samples=1``).

    Seeds are offset per sample index (``seed + sample_idx``) so Phase 2 can
    scale to N random realizations without changing the call site.
    """
    cfg = config or GeneratorSuiteConfig()
    gens = list(generators) if generators is not None else build_default_generators(cfg)
    samples = cfg.global_params.n_samples if n_samples is None else n_samples
    n_steps = cfg.global_params.n_steps
    base_seed = cfg.global_params.seed
    global_burn_in = cfg.global_params.burn_in

    series_list: list[GeneratedSeries] = []
    for sample_idx in range(samples):
        seed = base_seed + sample_idx
        for gen in gens:
            # Chaotic systems keep their own burn_in unless globally overridden (>0).
            burn_in: int | None
            if global_burn_in > 0:
                burn_in = global_burn_in
            else:
                burn_in = None
            logger.info(
                "Generating dataset_id=%s sample=%d seed=%d n_steps=%d",
                gen.dataset_id,
                sample_idx,
                seed,
                n_steps,
            )
            series_list.append(gen.generate(n_steps, seed=seed, burn_in=burn_in))
    return series_list
