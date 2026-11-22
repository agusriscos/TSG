"""Typed configuration for TS2G transformers (Table 3.2 defaults)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class NVGParams:
    pass


@dataclass(frozen=True)
class HVGParams:
    pass


@dataclass(frozen=True)
class DHVGParams:
    """Directed Horizontal Visibility Graph (time arrow i → j for i < j)."""

    pass


@dataclass(frozen=True)
class RNParams:
    m: int = 3
    tau: int = 1
    w: int = 5
    eps_percentile: float = 5.0


@dataclass(frozen=True)
class KNNParams:
    m: int = 3
    tau: int = 1
    w: int = 5
    k: int = 5


@dataclass(frozen=True)
class STNParams:
    M: int = 10


@dataclass(frozen=True)
class OPNParams:
    m: int = 4
    tau: int = 1


@dataclass(frozen=True)
class CorrelationParams:
    L: int = 20
    theta_corr: float = 0.75
    use_bonferroni: bool = False
    alpha: float = 0.05


@dataclass(frozen=True)
class SegmentParams:
    L: int = 20
    eps_percentile: float = 20.0


@dataclass(frozen=True)
class TS2GSuiteConfig:
    """Full TS2G suite configuration for Phase 1 and future N-series runs."""

    methods: tuple[str, ...] = (
        "NVG",
        "HVG",
        "RN",
        "kNN",
        "STN",
        "OPN",
        "Correlation",
        "Segment",
    )
    nvg: NVGParams = field(default_factory=NVGParams)
    hvg: HVGParams = field(default_factory=HVGParams)
    dhvg: DHVGParams = field(default_factory=DHVGParams)
    rn: RNParams = field(default_factory=RNParams)
    knn: KNNParams = field(default_factory=KNNParams)
    stn: STNParams = field(default_factory=STNParams)
    opn: OPNParams = field(default_factory=OPNParams)
    correlation: CorrelationParams = field(default_factory=CorrelationParams)
    segment: SegmentParams = field(default_factory=SegmentParams)

    @classmethod
    def from_mapping(cls, data: dict[str, Any]) -> TS2GSuiteConfig:
        methods = data.get("methods")
        return cls(
            methods=tuple(methods) if methods is not None else cls().methods,
            nvg=NVGParams(**_pick(data.get("NVG", data.get("nvg", {})), NVGParams)),
            hvg=HVGParams(**_pick(data.get("HVG", data.get("hvg", {})), HVGParams)),
            dhvg=DHVGParams(
                **_pick(data.get("dHVG", data.get("dhvg", {})), DHVGParams)
            ),
            rn=RNParams(**_pick(data.get("RN", data.get("rn", {})), RNParams)),
            knn=KNNParams(**_pick(data.get("kNN", data.get("knn", {})), KNNParams)),
            stn=STNParams(**_pick(data.get("STN", data.get("stn", {})), STNParams)),
            opn=OPNParams(**_pick(data.get("OPN", data.get("opn", {})), OPNParams)),
            correlation=CorrelationParams(
                **_pick(
                    data.get("Correlation", data.get("correlation", {})),
                    CorrelationParams,
                )
            ),
            segment=SegmentParams(
                **_pick(data.get("Segment", data.get("segment", {})), SegmentParams)
            ),
        )

    @classmethod
    def from_yaml(cls, path: Path | str) -> TS2GSuiteConfig:
        with open(path, encoding="utf-8") as fh:
            raw = yaml.safe_load(fh) or {}
        if not isinstance(raw, dict):
            raise ValueError(f"Expected mapping in {path}, got {type(raw)!r}")
        return cls.from_mapping(raw)


def _pick(data: dict[str, Any], cls: type) -> dict[str, Any]:
    fields = getattr(cls, "__dataclass_fields__", {})
    return {k: v for k, v in data.items() if k in fields}
