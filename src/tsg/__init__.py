"""TSG: synthetic multiregime time-series generation and TS2G mapping."""

from tsg.generators.base import GeneratedSeries, Regime, SeriesGenerator
from tsg.generators.registry import DATASET_IDS, build_default_generators, generate_suite
from tsg.graphs.base import GraphResult
from tsg.graphs.registry import METHOD_IDS, build_transformers, transform_suite

__all__ = [
    "DATASET_IDS",
    "METHOD_IDS",
    "GeneratedSeries",
    "GraphResult",
    "Regime",
    "SeriesGenerator",
    "build_default_generators",
    "build_transformers",
    "generate_suite",
    "transform_suite",
]

__version__ = "0.2.0"
