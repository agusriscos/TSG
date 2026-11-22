"""Multiregime dataset registry and suite orchestration."""

from __future__ import annotations

import logging
from typing import Callable, Sequence

from tsg.generators.base import GeneratedSeries, SeriesGenerator
from tsg.generators.chaotic import HenonMap, LogisticMap, LogisticWeak, LorenzX, Rossler
from tsg.generators.config import GeneratorSuiteConfig
from tsg.generators.nonstationary import RegimeSwitching
from tsg.generators.periodic import Quasiperiodic, SineWave
from tsg.generators.stochastic import (
    ARMACorrelated,
    BrownianMotion,
    FractionalBrownianMotion,
    WhiteNoise,
)

logger = logging.getLogger(__name__)

# Phase-1 control panel (Table 3.1). Do not extend: frontier regimes live in
# CANONICAL_REGIMES so existing realization seeds stay stable.
DATASET_IDS: tuple[str, ...] = (
    "white_noise",
    "brownian",
    "sine",
    "quasiperiodic",
    "lorenz",
    "logistic",
)

# Fixed order keeps realization seeds stable. New regimes are appended.
CANONICAL_REGIMES: tuple[str, ...] = DATASET_IDS + (
    "fbm_antipersistent",
    "fbm_persistent",
    "arma_correlated",
    "rossler",
    "henon",
    "logistic_weak",
    "regime_switching",
)

REGIME_LABELS: dict[str, str] = {
    "white_noise": "stochastic",
    "brownian": "stochastic",
    "fbm_antipersistent": "stochastic",
    "fbm_persistent": "stochastic",
    "arma_correlated": "stochastic",
    "sine": "periodic",
    "quasiperiodic": "periodic",
    "lorenz": "chaotic",
    "rossler": "chaotic",
    "henon": "chaotic",
    "logistic": "chaotic",
    "logistic_weak": "chaotic",
    "regime_switching": "nonstationary",
}


def _factories() -> dict[str, Callable[[GeneratorSuiteConfig], SeriesGenerator]]:
    return {
        "white_noise": lambda cfg: WhiteNoise(params=cfg.white_noise),
        "brownian": lambda cfg: BrownianMotion(params=cfg.brownian),
        "sine": lambda cfg: SineWave(params=cfg.sine),
        "quasiperiodic": lambda cfg: Quasiperiodic(params=cfg.quasiperiodic),
        "lorenz": lambda cfg: LorenzX(params=cfg.lorenz),
        "logistic": lambda cfg: LogisticMap(params=cfg.logistic),
        "fbm_antipersistent": lambda cfg: FractionalBrownianMotion(
            params=cfg.fbm, variant="antipersistent"
        ),
        "fbm_persistent": lambda cfg: FractionalBrownianMotion(
            params=cfg.fbm, variant="persistent"
        ),
        "arma_correlated": lambda cfg: ARMACorrelated(params=cfg.arma),
        "rossler": lambda cfg: Rossler(params=cfg.rossler),
        "henon": lambda cfg: HenonMap(params=cfg.henon),
        "logistic_weak": lambda cfg: LogisticWeak(params=cfg.logistic_weak),
        "regime_switching": lambda cfg: RegimeSwitching(
            params=cfg.regime_switching,
            quasiperiodic=cfg.quasiperiodic,
            lorenz=cfg.lorenz,
            rossler=cfg.rossler,
        ),
    }


GENERATOR_FACTORIES = _factories()


def build_generator(
    dataset_id: str,
    config: GeneratorSuiteConfig | None = None,
) -> SeriesGenerator:
    """Instantiate the generator registered under ``dataset_id``.

    Parameters
    ----------
    dataset_id
        Key of :data:`CANONICAL_REGIMES`.
    config
        Suite configuration. Defaults are used when omitted, so every regime
        remains selectable without a dedicated YAML block.
    """
    cfg = config or GeneratorSuiteConfig()
    try:
        factory = GENERATOR_FACTORIES[dataset_id]
    except KeyError as exc:
        known = ", ".join(CANONICAL_REGIMES)
        raise KeyError(f"Unknown dataset_id={dataset_id!r}. Known: {known}") from exc
    return factory(cfg)


def build_default_generators(
    config: GeneratorSuiteConfig | None = None,
) -> list[SeriesGenerator]:
    """Instantiate one generator per Phase-1 Dataset ID (Table 3.1)."""
    cfg = config or GeneratorSuiteConfig()
    return [build_generator(dataset_id, cfg) for dataset_id in DATASET_IDS]


def generate_suite(
    config: GeneratorSuiteConfig | None = None,
    *,
    generators: Sequence[SeriesGenerator] | None = None,
    n_samples: int | None = None,
) -> list[GeneratedSeries]:
    """Generate ``n_samples`` series per generator (Phase 1: ``n_samples=1``).

    Seeds are offset per sample index (``seed + sample_idx``) so later phases
    can scale to N random realizations without changing the call site.
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
