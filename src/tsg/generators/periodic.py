"""Periodic regime generators: sine and quasiperiodic superposition."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from tsg.generators.base import GeneratedSeries, Regime, SeriesGenerator
from tsg.generators.config import QuasiperiodicParams, SineParams

GOLDEN_RATIO: float = (1.0 + np.sqrt(5.0)) / 2.0


@dataclass(frozen=True)
class SineWave(SeriesGenerator):
    """Single-frequency closed orbit with optional jitter / incommensurability."""

    params: SineParams = SineParams()

    @property
    def dataset_id(self) -> str:
        return "sine"

    @property
    def regime(self) -> Regime:
        return "periodic"

    def params_dict(self) -> dict[str, Any]:
        p = self.params
        return {
            "freq": p.freq,
            "amplitude": p.amplitude,
            "phase": p.phase,
            "jitter": p.jitter,
            "jitter_sigma": p.jitter_sigma,
            "incommensurable": p.incommensurable,
            "incommensurable_eps": p.incommensurable_eps,
            "effective_freq": self._effective_freq(),
        }

    def _effective_freq(self) -> float:
        p = self.params
        if p.incommensurable:
            return p.freq * (1.0 + p.incommensurable_eps)
        return p.freq

    def generate(
        self,
        n_steps: int,
        *,
        seed: int = 0,
        burn_in: int | None = None,
    ) -> GeneratedSeries:
        del burn_in
        p = self.params
        t = np.arange(n_steps, dtype=float)
        freq = self._effective_freq()
        values = p.amplitude * np.sin(2.0 * np.pi * freq * t + p.phase)
        if p.jitter:
            rng = np.random.default_rng(seed)
            values = values + rng.normal(0.0, p.jitter_sigma, size=n_steps)
        return self._finalize(values, seed=seed, burn_in=0)


@dataclass(frozen=True)
class Quasiperiodic(SeriesGenerator):
    """Additive superposition of incommensurate frequencies via the golden ratio."""

    params: QuasiperiodicParams = QuasiperiodicParams()

    @property
    def dataset_id(self) -> str:
        return "quasiperiodic"

    @property
    def regime(self) -> Regime:
        return "periodic"

    def _f2(self) -> float:
        if self.params.f2 is None:
            return self.params.f1 / GOLDEN_RATIO
        return float(self.params.f2)

    def params_dict(self) -> dict[str, Any]:
        return {
            "f1": self.params.f1,
            "f2": self._f2(),
            "amplitude": self.params.amplitude,
            "phi": float(GOLDEN_RATIO),
        }

    def generate(
        self,
        n_steps: int,
        *,
        seed: int = 0,
        burn_in: int | None = None,
    ) -> GeneratedSeries:
        del burn_in
        p = self.params
        rng = np.random.default_rng(seed)
        # Seed-controlled phase offsets → distinct realizations of the same torus.
        phase1 = float(rng.uniform(0.0, 2.0 * np.pi))
        phase2 = float(rng.uniform(0.0, 2.0 * np.pi))
        t = np.arange(n_steps, dtype=float)
        f2 = self._f2()
        values = p.amplitude * (
            np.sin(2.0 * np.pi * p.f1 * t + phase1)
            + np.sin(2.0 * np.pi * f2 * t + phase2)
        )
        return self._finalize(values, seed=seed, burn_in=0)
