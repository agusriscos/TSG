"""Stochastic regime generators: white noise and Brownian motion."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from tsg.generators.base import GeneratedSeries, Regime, SeriesGenerator
from tsg.generators.config import BrownianParams, WhiteNoiseParams


@dataclass(frozen=True)
class WhiteNoise(SeriesGenerator):
    """Uncorrelated Gaussian control: N(0, sigma^2)."""

    params: WhiteNoiseParams = WhiteNoiseParams()

    @property
    def dataset_id(self) -> str:
        return "white_noise"

    @property
    def regime(self) -> Regime:
        return "stochastic"

    def params_dict(self) -> dict[str, Any]:
        return {"sigma": self.params.sigma}

    def generate(
        self,
        n_steps: int,
        *,
        seed: int = 0,
        burn_in: int | None = None,
    ) -> GeneratedSeries:
        del burn_in  # white noise has no transient dynamics
        rng = np.random.default_rng(seed)
        values = rng.normal(0.0, self.params.sigma, size=n_steps)
        return self._finalize(values, seed=seed, burn_in=0)


@dataclass(frozen=True)
class BrownianMotion(SeriesGenerator):
    """Non-stationary random walk: cumulative sum of N(0, sigma^2) increments."""

    params: BrownianParams = BrownianParams()

    @property
    def dataset_id(self) -> str:
        return "brownian"

    @property
    def regime(self) -> Regime:
        return "stochastic"

    def params_dict(self) -> dict[str, Any]:
        return {"sigma": self.params.sigma}

    def generate(
        self,
        n_steps: int,
        *,
        seed: int = 0,
        burn_in: int | None = None,
    ) -> GeneratedSeries:
        del burn_in
        rng = np.random.default_rng(seed)
        increments = rng.normal(0.0, self.params.sigma, size=n_steps)
        values = np.cumsum(increments)
        return self._finalize(values, seed=seed, burn_in=0)
