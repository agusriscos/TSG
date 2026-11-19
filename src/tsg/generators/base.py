"""Abstract generator interface and standardized series container."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

Regime = Literal["stochastic", "periodic", "chaotic"]


@dataclass(frozen=True)
class GeneratedSeries:
    """Standardized univariate synthetic series with provenance metadata."""

    dataset_id: str
    regime: Regime
    values: NDArray[np.floating]
    length: int
    seed: int
    params: dict[str, Any] = field(default_factory=dict)
    burn_in: int = 0
    variable: str = "Value"

    def __post_init__(self) -> None:
        values = np.asarray(self.values, dtype=float).ravel()
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "length", int(values.size))

    def to_metadata(self) -> dict[str, Any]:
        """Return serializable metadata excluding the raw values array."""
        return {
            "dataset_id": self.dataset_id,
            "regime": self.regime,
            "length": self.length,
            "seed": self.seed,
            "burn_in": self.burn_in,
            "variable": self.variable,
            "params": dict(self.params),
        }

    def to_dict(self) -> dict[str, Any]:
        """Return metadata plus values as a plain Python list."""
        payload = self.to_metadata()
        payload["values"] = self.values.tolist()
        return payload


class SeriesGenerator(ABC):
    """Abstract factory for one Dataset ID from the multiregime registry."""

    @property
    @abstractmethod
    def dataset_id(self) -> str:
        """Canonical Dataset ID (Table 3.1)."""

    @property
    @abstractmethod
    def regime(self) -> Regime:
        """Dynamical regime label."""

    @property
    def variable(self) -> str:
        """Observed variable name reported in metadata."""
        return "Value"

    @abstractmethod
    def params_dict(self) -> dict[str, Any]:
        """Exact parameters used by this generator instance."""

    @abstractmethod
    def generate(
        self,
        n_steps: int,
        *,
        seed: int = 0,
        burn_in: int | None = None,
    ) -> GeneratedSeries:
        """Generate one length-``n_steps`` series after optional burn-in.

        Parameters
        ----------
        n_steps
            Number of samples retained after burn-in.
        seed
            Seed for ``np.random.default_rng`` when the generator is stochastic
            or uses jitter.
        burn_in
            Transient steps to discard. ``None`` means use the generator-specific
            default (0 for stochastic/periodic; params.burn_in for chaotic).
        """

    def _finalize(
        self,
        values: NDArray[np.floating],
        *,
        seed: int,
        burn_in: int,
    ) -> GeneratedSeries:
        return GeneratedSeries(
            dataset_id=self.dataset_id,
            regime=self.regime,
            values=values,
            length=int(np.asarray(values).size),
            seed=seed,
            params=self.params_dict(),
            burn_in=burn_in,
            variable=self.variable,
        )


def dataclass_params(instance: Any) -> dict[str, Any]:
    """Serialize a frozen params dataclass to a plain dict."""
    return asdict(instance)
