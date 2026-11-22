"""Direct-geometry visibility graphs (NVG, HVG, dHVG)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import networkx as nx
import numpy as np
from numpy.typing import NDArray
from ts2vg import HorizontalVG, NaturalVG

from tsg.graphs.base import BaseGraphTransformer, Family
from tsg.graphs.config import DHVGParams, HVGParams, NVGParams


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


@dataclass(frozen=True)
class DHVGTransformer(BaseGraphTransformer):
    """Directed Horizontal Visibility Graph (time arrow ``i → j`` for ``i < j``).

    Same horizontal visibility criterion as :class:`HVGTransformer`, but edges
    follow the temporal order and the result is an ``nx.DiGraph``. This enables
    directional descriptors (``i_dir``, ``kl_inout``) inside the visibility
    family without altering undirected HVG/NVG behaviour.
    """

    params: DHVGParams = DHVGParams()

    @property
    def method(self) -> str:
        return "dHVG"

    @property
    def family(self) -> Family:
        return "Direct Geometry"

    @property
    def directed(self) -> bool:
        return True

    @property
    def weighted(self) -> bool:
        return False

    def params_dict(self) -> dict[str, Any]:
        return {"directed": "left_to_right"}

    def expected_n_nodes(self, series_length: int) -> int | None:
        return series_length

    def _build(self, series: NDArray[np.floating]) -> nx.DiGraph:
        x = np.array(series, dtype=float, copy=True)
        vg = HorizontalVG(directed="left_to_right")
        vg.build(x)
        g = vg.as_networkx()
        if not isinstance(g, nx.DiGraph):
            g = nx.DiGraph(g)
        return g
