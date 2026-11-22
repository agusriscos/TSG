"""Directional-dynamics descriptors for transition networks."""

from __future__ import annotations

import numpy as np

from tsg.descriptors.base import BaseDescriptorExtractor, PreparedGraph
from tsg.descriptors.config import DescriptorConfig


class DirectionalExtractor(BaseDescriptorExtractor):
    """``i_dir``, ``self_loop_ratio``."""

    @property
    def descriptor_ids(self) -> tuple[str, ...]:
        return ("i_dir", "self_loop_ratio")

    def extract(self, prepared: PreparedGraph, config: DescriptorConfig) -> dict[str, float]:
        del config
        if not prepared.is_directed_weighted or prepared.w_directed is None:
            return {"i_dir": 0.0, "self_loop_ratio": 0.0}

        w = np.asarray(prepared.w_directed, dtype=float)
        diff = w - w.T
        sym = w + w.T
        denom = float(np.linalg.norm(sym, ord="fro"))
        if denom == 0.0:
            i_dir = 0.0
        else:
            i_dir = float(np.linalg.norm(diff, ord="fro") / denom)

        total = float(w.sum())
        if total == 0.0:
            self_loop_ratio = 0.0
        else:
            self_loop_ratio = float(np.trace(w) / total)

        return {"i_dir": i_dir, "self_loop_ratio": self_loop_ratio}
