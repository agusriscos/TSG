"""TS2G method registry and suite orchestration."""

from __future__ import annotations

import logging
from typing import Sequence

import numpy as np
from numpy.typing import NDArray

from tsg.graphs.base import BaseGraphTransformer, GraphResult
from tsg.graphs.config import TS2GSuiteConfig
from tsg.graphs.geometry import HVGTransformer, NVGTransformer
from tsg.graphs.phase_space import KNNTransformer, RecurrenceNetworkTransformer
from tsg.graphs.similarity import CorrelationNetworkTransformer, SegmentNetworkTransformer
from tsg.graphs.transition import OPNTransformer, STNTransformer

logger = logging.getLogger(__name__)

METHOD_IDS: tuple[str, ...] = (
    "NVG",
    "HVG",
    "RN",
    "kNN",
    "STN",
    "OPN",
    "Correlation",
    "Segment",
)


def build_transformers(
    config: TS2GSuiteConfig | None = None,
) -> list[BaseGraphTransformer]:
    """Instantiate transformers for the configured method list."""
    cfg = config or TS2GSuiteConfig()
    catalog: dict[str, BaseGraphTransformer] = {
        "NVG": NVGTransformer(params=cfg.nvg),
        "HVG": HVGTransformer(params=cfg.hvg),
        "RN": RecurrenceNetworkTransformer(params=cfg.rn),
        "kNN": KNNTransformer(params=cfg.knn),
        "STN": STNTransformer(params=cfg.stn),
        "OPN": OPNTransformer(params=cfg.opn),
        "Correlation": CorrelationNetworkTransformer(params=cfg.correlation),
        "Segment": SegmentNetworkTransformer(params=cfg.segment),
    }
    missing = [m for m in cfg.methods if m not in catalog]
    if missing:
        raise KeyError(f"Unknown TS2G methods: {missing}")
    return [catalog[m] for m in cfg.methods]


def transform_suite(
    series_items: Sequence[tuple[str, str, NDArray[np.floating]]],
    config: TS2GSuiteConfig | None = None,
    *,
    transformers: Sequence[BaseGraphTransformer] | None = None,
) -> list[GraphResult]:
    """Apply every transformer to every series (Phase 1: 6 × 8 = 48 graphs).

    Parameters
    ----------
    series_items
        Iterable of ``(dataset_id, regime, values)``.
    """
    cfg = config or TS2GSuiteConfig()
    transformers = list(transformers) if transformers is not None else build_transformers(cfg)
    results: list[GraphResult] = []
    for dataset_id, regime, values in series_items:
        for tr in transformers:
            logger.info("TS2G %s ← %s", tr.method, dataset_id)
            results.append(tr.transform(values, dataset_id, regime=regime))
    return results


def expected_nodes_ok(result: GraphResult, transformer: BaseGraphTransformer) -> bool:
    """Check node-count formula from Table 3.2 (OPN: only upper bound)."""
    expected = transformer.expected_n_nodes(result.series_length)
    if expected is None:
        # OPN: n <= m!
        max_n = int(result.params.get("max_patterns", 24))
        return 0 < result.n_nodes <= max_n
    return result.n_nodes == expected
