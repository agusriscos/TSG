"""TSG: synthetic series, TS2G mapping, descriptor matrix D and discriminability audit."""

from tsg.generators.base import GeneratedSeries, Regime, SeriesGenerator
from tsg.generators.registry import DATASET_IDS, build_default_generators, generate_suite
from tsg.descriptors.base import DESCRIPTOR_IDS
from tsg.graphs.base import GraphResult
from tsg.graphs.registry import METHOD_IDS, build_transformers, transform_suite

__all__ = [
    "DATASET_IDS",
    "DESCRIPTOR_IDS",
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

__version__ = "0.5.0"
