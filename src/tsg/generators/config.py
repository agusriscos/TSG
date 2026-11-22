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
    f1_range: tuple[float, float] = (0.018, 0.022)
    # Relative amplitude of the second tone vs the first (sampled per realization).
    amp2_ratio_range: tuple[float, float] = (0.85, 1.15)
    # Fractional jitter on the golden-ratio frequency relation.
    phi_jitter_frac: float = 0.05


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
    # Safe chaotic window around the classical attractor (beta stays fixed).
    sigma_range: tuple[float, float] = (9.5, 10.5)
    rho_range: tuple[float, float] = (27.0, 29.0)
    dt_range: tuple[float, float] = (0.01, 0.03)
    burn_in_range: tuple[int, int] = (500, 2000)


@dataclass(frozen=True)
class LogisticParams:
    r: float = 4.0  # Keep fixed at fully-chaotic value; never sample.
    x0: float = 0.1
    burn_in: int = 200
    x0_range: tuple[float, float] = (0.05, 0.95)
    # Short random transient so realizations start decorrelated on the attractor.
    burn_in_range: tuple[int, int] = (500, 1500)


@dataclass(frozen=True)
class FractionalBrownianParams:
    """Shared controls for anti-persistent and persistent fractional Brownian motion."""

    sigma: float = 1.0
    hurst_antipersistent_range: tuple[float, float] = (0.15, 0.25)
    hurst_persistent_range: tuple[float, float] = (0.75, 0.85)


@dataclass(frozen=True)
class ARMAParams:
    """Stationary ARMA(2, 1) with complex poles near the unit circle.

    Poles sit at ``rho * exp(±i omega)``. ``rho`` close to 1 yields an
    autocorrelation time comparable to a slowly sampled chaotic flow, while
    ``|rho| < 1`` keeps the process (strictly) stationary.
    """

    sigma: float = 1.0
    rho_range: tuple[float, float] = (0.96, 0.99)
    omega_range: tuple[float, float] = (0.05, 0.25)
    theta_range: tuple[float, float] = (0.3, 0.8)
    burn_in_range: tuple[int, int] = (800, 1600)


@dataclass(frozen=True)
class RosslerParams:
    """Classical Rössler chaotic window and RK4 step."""

    a_range: tuple[float, float] = (0.18, 0.22)
    b_range: tuple[float, float] = (0.18, 0.22)
    c_range: tuple[float, float] = (5.5, 5.9)
    dt: float = 0.05
    burn_in: int = 1000
    x0: float = 1.0
    y0: float = 1.0
    z0: float = 1.0
    ic_sigma: float = 0.5


@dataclass(frozen=True)
class HenonParams:
    """Hénon map inside the classical chaotic band (``b`` fixed)."""

    a_range: tuple[float, float] = (1.38, 1.41)
    b: float = 0.3
    burn_in: int = 500
    x0: float = 0.0
    y0: float = 0.0
    ic_sigma: float = 0.1
    # Trajectories leaving this box are treated as divergent and restarted.
    bound: float = 5.0


@dataclass(frozen=True)
class LogisticWeakParams:
    """Logistic map on the weak-chaos side of the onset (``r_∞ ≈ 3.57``)."""

    r_range: tuple[float, float] = (3.58, 3.65)
    r_fallback: float = 3.60
    x0: float = 0.1
    x0_range: tuple[float, float] = (0.05, 0.95)
    burn_in_range: tuple[int, int] = (500, 1500)
    # Finite-time Lyapunov probe used to reject periodic windows.
    lyapunov_probe: int = 2000
    max_attempts: int = 24


@dataclass(frozen=True)
class RegimeSwitchingParams:
    """Quasiperiodic → chaotic splice with a change point in the central window."""

    change_frac_range: tuple[float, float] = (0.35, 0.65)
    smooth_width_frac: float = 0.04
    min_smooth_width: int = 8
    lorenz_burn_in: int = 1000
    rossler_burn_in: int = 1000


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
    fbm: FractionalBrownianParams = field(default_factory=FractionalBrownianParams)
    arma: ARMAParams = field(default_factory=ARMAParams)
    rossler: RosslerParams = field(default_factory=RosslerParams)
    henon: HenonParams = field(default_factory=HenonParams)
    logistic_weak: LogisticWeakParams = field(default_factory=LogisticWeakParams)
    regime_switching: RegimeSwitchingParams = field(default_factory=RegimeSwitchingParams)

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
            fbm=FractionalBrownianParams(**_pick(data.get("fbm", {}), FractionalBrownianParams)),
            arma=ARMAParams(**_pick(data.get("arma", {}), ARMAParams)),
            rossler=RosslerParams(**_pick(data.get("rossler", {}), RosslerParams)),
            henon=HenonParams(**_pick(data.get("henon", {}), HenonParams)),
            logistic_weak=LogisticWeakParams(
                **_pick(data.get("logistic_weak", {}), LogisticWeakParams)
            ),
            regime_switching=RegimeSwitchingParams(
                **_pick(data.get("regime_switching", {}), RegimeSwitchingParams)
            ),
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
    """Keep only keys that exist as fields on ``cls`` (lists → tuples)."""
    fields = getattr(cls, "__dataclass_fields__", {})
    out: dict[str, Any] = {}
    for key, value in data.items():
        if key not in fields:
            continue
        out[key] = tuple(value) if isinstance(value, list) else value
    return out
