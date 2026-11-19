"""Typed configuration for synthetic generators (Phase 1 defaults + Phase 2 ranges)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class GlobalGenerationParams:
    """Shared generation controls for the suite."""

    n_steps: int = 5000
    seed: int = 42
    burn_in: int = 0
    n_samples: int = 1


@dataclass(frozen=True)
class WhiteNoiseParams:
    sigma: float = 1.0
    # Phase-2 sampling ranges (unused in Phase 1)
    sigma_range: tuple[float, float] = (0.5, 2.0)


@dataclass(frozen=True)
class BrownianParams:
    sigma: float = 1.0
    sigma_range: tuple[float, float] = (0.5, 2.0)


@dataclass(frozen=True)
class SineParams:
    freq: float = 0.02
    amplitude: float = 1.0
    phase: float = 0.0
    jitter: bool = True
    jitter_sigma: float = 1e-3
    incommensurable: bool = False
    incommensurable_eps: float = 1e-4
    freq_range: tuple[float, float] = (0.01, 0.05)
    jitter_sigma_range: tuple[float, float] = (1e-4, 5e-3)


@dataclass(frozen=True)
class QuasiperiodicParams:
    f1: float = 0.02
    # f2 defaults to f1 / golden_ratio when None
    f2: float | None = None
    amplitude: float = 1.0
    f1_range: tuple[float, float] = (0.01, 0.05)


@dataclass(frozen=True)
class LorenzParams:
    sigma: float = 10.0
    rho: float = 28.0
    beta: float = 8.0 / 3.0
    dt: float = 0.02
    burn_in: int = 1000
    x0: float = 1.0
    y0: float = 1.0
    z0: float = 1.0
    dt_range: tuple[float, float] = (0.01, 0.03)
    burn_in_range: tuple[int, int] = (500, 2000)


@dataclass(frozen=True)
class LogisticParams:
    r: float = 4.0
    x0: float = 0.1
    burn_in: int = 200
    x0_range: tuple[float, float] = (0.05, 0.95)
    burn_in_range: tuple[int, int] = (100, 500)


@dataclass(frozen=True)
class GeneratorSuiteConfig:
    """Full suite configuration: globals + per-dataset defaults."""

    global_params: GlobalGenerationParams = field(default_factory=GlobalGenerationParams)
    white_noise: WhiteNoiseParams = field(default_factory=WhiteNoiseParams)
    brownian: BrownianParams = field(default_factory=BrownianParams)
    sine: SineParams = field(default_factory=SineParams)
    quasiperiodic: QuasiperiodicParams = field(default_factory=QuasiperiodicParams)
    lorenz: LorenzParams = field(default_factory=LorenzParams)
    logistic: LogisticParams = field(default_factory=LogisticParams)

    @classmethod
    def from_mapping(cls, data: dict[str, Any]) -> GeneratorSuiteConfig:
        """Build config from a nested mapping (e.g. loaded YAML)."""
        g = data.get("global", {})
        return cls(
            global_params=GlobalGenerationParams(**_pick(g, GlobalGenerationParams)),
            white_noise=WhiteNoiseParams(**_pick(data.get("white_noise", {}), WhiteNoiseParams)),
            brownian=BrownianParams(**_pick(data.get("brownian", {}), BrownianParams)),
            sine=SineParams(**_pick(data.get("sine", {}), SineParams)),
            quasiperiodic=QuasiperiodicParams(
                **_pick(data.get("quasiperiodic", {}), QuasiperiodicParams)
            ),
            lorenz=LorenzParams(**_pick(data.get("lorenz", {}), LorenzParams)),
            logistic=LogisticParams(**_pick(data.get("logistic", {}), LogisticParams)),
        )

    @classmethod
    def from_yaml(cls, path: Path | str) -> GeneratorSuiteConfig:
        """Load suite configuration from a YAML file."""
        with open(path, encoding="utf-8") as fh:
            raw = yaml.safe_load(fh) or {}
        if not isinstance(raw, dict):
            raise ValueError(f"Expected mapping in {path}, got {type(raw)!r}")
        return cls.from_mapping(raw)


def _pick(data: dict[str, Any], cls: type) -> dict[str, Any]:
    """Keep only keys that exist as fields on ``cls``."""
    fields = getattr(cls, "__dataclass_fields__", {})
    return {k: v for k, v in data.items() if k in fields}
