"""Synthetic multiregime series generators."""

from tsg.generators.base import GeneratedSeries, Regime, SeriesGenerator
from tsg.generators.config import (
    BrownianParams,
    GeneratorSuiteConfig,
    GlobalGenerationParams,
    LogisticParams,
    LorenzParams,
    QuasiperiodicParams,
    SineParams,
    WhiteNoiseParams,
)
from tsg.generators.registry import DATASET_IDS, build_default_generators, generate_suite
from tsg.generators.stochastic import BrownianMotion, WhiteNoise
from tsg.generators.periodic import Quasiperiodic, SineWave
from tsg.generators.chaotic import LogisticMap, LorenzX

__all__ = [
    "DATASET_IDS",
    "BrownianMotion",
    "BrownianParams",
    "GeneratedSeries",
    "GeneratorSuiteConfig",
    "GlobalGenerationParams",
    "LogisticMap",
    "LogisticParams",
    "LorenzParams",
    "LorenzX",
    "Quasiperiodic",
    "QuasiperiodicParams",
    "Regime",
    "SeriesGenerator",
    "SineParams",
    "SineWave",
    "WhiteNoise",
    "WhiteNoiseParams",
    "build_default_generators",
    "generate_suite",
]
