"""TSG: synthetic multiregime time-series generation."""

from tsg.generators.base import GeneratedSeries, Regime, SeriesGenerator
from tsg.generators.registry import DATASET_IDS, build_default_generators, generate_suite

__all__ = [
    "DATASET_IDS",
    "GeneratedSeries",
    "Regime",
    "SeriesGenerator",
    "build_default_generators",
    "generate_suite",
]

__version__ = "0.1.0"
