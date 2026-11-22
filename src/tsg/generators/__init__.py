"""Synthetic multiregime series generators."""

from tsg.generators.base import GeneratedSeries, Regime, SeriesGenerator
from tsg.generators.chaotic import HenonMap, LogisticMap, LogisticWeak, LorenzX, Rossler
from tsg.generators.config import (
    ARMAParams,
    BrownianParams,
    FractionalBrownianParams,
    GeneratorSuiteConfig,
    GlobalGenerationParams,
    HenonParams,
    LogisticParams,
    LogisticWeakParams,
    LorenzParams,
    QuasiperiodicParams,
    RegimeSwitchingParams,
    RosslerParams,
    SineParams,
    WhiteNoiseParams,
)
from tsg.generators.nonstationary import RegimeSwitching
from tsg.generators.periodic import Quasiperiodic, SineWave
from tsg.generators.registry import (
    CANONICAL_REGIMES,
    DATASET_IDS,
    build_default_generators,
    build_generator,
    generate_suite,
)
from tsg.generators.stochastic import (
    ARMACorrelated,
    BrownianMotion,
    FractionalBrownianMotion,
    WhiteNoise,
)
from tsg.generators.perturbations import (
    add_white_noise_snr,
    affine_transform,
    apply_transform,
    downsample_series,
    impulsive_outliers,
    monotone_nonlinear,
    quantize_ties,
)
from tsg.generators.surrogates import (
    ft_surrogate,
    iaaft_surrogate,
    make_surrogate,
    shuffled_surrogate,
)

__all__ = [
    "ARMACorrelated",
    "ARMAParams",
    "CANONICAL_REGIMES",
    "DATASET_IDS",
    "BrownianMotion",
    "BrownianParams",
    "FractionalBrownianMotion",
    "FractionalBrownianParams",
    "GeneratedSeries",
    "GeneratorSuiteConfig",
    "GlobalGenerationParams",
    "HenonMap",
    "HenonParams",
    "LogisticMap",
    "LogisticParams",
    "LogisticWeak",
    "LogisticWeakParams",
    "LorenzParams",
    "LorenzX",
    "Quasiperiodic",
    "QuasiperiodicParams",
    "Regime",
    "RegimeSwitching",
    "RegimeSwitchingParams",
    "Rossler",
    "RosslerParams",
    "SeriesGenerator",
    "SineParams",
    "SineWave",
    "WhiteNoise",
    "WhiteNoiseParams",
    "add_white_noise_snr",
    "affine_transform",
    "apply_transform",
    "build_default_generators",
    "build_generator",
    "downsample_series",
    "ft_surrogate",
    "generate_suite",
    "iaaft_surrogate",
    "impulsive_outliers",
    "make_surrogate",
    "monotone_nonlinear",
    "quantize_ties",
    "shuffled_surrogate",
]
