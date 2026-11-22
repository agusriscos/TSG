"""Configuration for descriptor extraction (Table 3.3)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import yaml

LaplacianType = Literal["combinatorial", "normalized"]
EntropyBase = Literal["e", "2"]


@dataclass(frozen=True)
class DescriptorConfig:
    """Descriptor computation settings."""

    laplacian: LaplacianType = "combinatorial"
    entropy_base: EntropyBase = "e"
    large_graph_threshold: int = 250
    eigsh_maxiter: int = 5000
    eigsh_tol: float = 1e-8
    # When set, expensive descriptors are skipped (placeholder 0.0) for SA inner loop.
    skip_descriptors: tuple[str, ...] = ()

    @classmethod
    def from_mapping(cls, data: dict[str, Any]) -> DescriptorConfig:
        fields = getattr(cls, "__dataclass_fields__", {})
        kwargs = {k: v for k, v in data.items() if k in fields}
        if "skip_descriptors" in kwargs and isinstance(kwargs["skip_descriptors"], list):
            kwargs["skip_descriptors"] = tuple(kwargs["skip_descriptors"])
        return cls(**kwargs)

    @classmethod
    def from_yaml(cls, path: Path | str) -> DescriptorConfig:
        with open(path, encoding="utf-8") as fh:
            raw = yaml.safe_load(fh) or {}
        if not isinstance(raw, dict):
            raise ValueError(f"Expected mapping in {path}, got {type(raw)!r}")
        return cls.from_mapping(raw.get("descriptors", raw))
