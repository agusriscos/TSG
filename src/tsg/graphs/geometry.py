"""Direct-geometry visibility graphs (NVG, HVG)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import networkx as nx
import numpy as np
from numpy.typing import NDArray
from ts2vg import HorizontalVG, NaturalVG

from tsg.graphs.base import BaseGraphTransformer, Family
from tsg.graphs.config import HVGParams, NVGParams


@dataclass(frozen=True)
class NVGTransformer(BaseGraphTransformer):
    """Natural Visibility Graph (line-of-sight geometry)."""

    params: NVGParams = NVGParams()

    @property
    def method(self) -> str:
        return "NVG"

    @property
    def family(self) -> Family:
        return "Direct Geometry"

    @property
    def directed(self) -> bool:
        return False

    @property
    def weighted(self) -> bool:
        return False

    def params_dict(self) -> dict[str, Any]:
        return {}

    def expected_n_nodes(self, series_length: int) -> int | None:
        return series_length

    def _build(self, series: NDArray[np.floating]) -> nx.Graph:
        # ts2vg requires a writable buffer (pandas/.ravel views may be read-only).
        x = np.array(series, dtype=float, copy=True)
        vg = NaturalVG()
        vg.build(x)
        g = vg.as_networkx()
        if isinstance(g, nx.DiGraph):
            g = nx.Graph(g)
        return g


@dataclass(frozen=True)
class HVGTransformer(BaseGraphTransformer):
    """Horizontal Visibility Graph."""

    params: HVGParams = HVGParams()

    @property
    def method(self) -> str:
        return "HVG"

    @property
    def family(self) -> Family:
        return "Direct Geometry"

    @property
    def directed(self) -> bool:
        return False

    @property
    def weighted(self) -> bool:
        return False

    def params_dict(self) -> dict[str, Any]:
        return {}

    def expected_n_nodes(self, series_length: int) -> int | None:
        return series_length

    def _build(self, series: NDArray[np.floating]) -> nx.Graph:
        x = np.array(series, dtype=float, copy=True)
        vg = HorizontalVG()
        vg.build(x)
        g = vg.as_networkx()
        if isinstance(g, nx.DiGraph):
            g = nx.Graph(g)
        return g
