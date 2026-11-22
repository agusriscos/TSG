"""Non-stationary generators: quasiperiodic to chaotic regime switching."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from tsg.generators.base import GeneratedSeries, Regime, SeriesGenerator
from tsg.generators.chaotic import LorenzX, Rossler
from tsg.generators.config import (
    LorenzParams,
    QuasiperiodicParams,
    RegimeSwitchingParams,
    RosslerParams,
)
from tsg.generators.periodic import Quasiperiodic

logger = logging.getLogger(__name__)


def _standardize(values: NDArray[np.floating]) -> NDArray[np.floating]:
    """Return a zero-mean, unit-variance copy (centered if the scale vanishes)."""
    x = np.asarray(values, dtype=float)
    centered = x - float(x.mean())
    scale = float(centered.std())
    if scale < 1e-12:
        return centered
    return centered / scale


@dataclass(frozen=True)
class RegimeSwitching(SeriesGenerator):
    """Splice a quasiperiodic prefix to a chaotic suffix.

    The change index is drawn uniformly from
    ``[floor(0.35 N), floor(0.65 N)]``. The chaotic piece is either Lorenz or
    Rössler (equal probability). Both pieces are standardized to zero mean and
    unit variance before they are joined, so the transition is dynamical
    rather than a jump in scale.

    Transition modes, chosen with equal probability:

    * ``abrupt`` — the suffix starting at the change index is replaced by the
      chaotic series.
    * ``smooth`` — a raised-cosine (Hann) crossfade of width
      ``max(min_smooth_width, round(smooth_width_frac * N))`` is centered on
      the change index. Samples before the window stay quasiperiodic; samples
      after it are chaotic.

    Child generators receive independent seeds spawned from the realization
    seed, so initial conditions and parameter jitter stay reproducible and
    differ across realizations.
    """

    params: RegimeSwitchingParams = RegimeSwitchingParams()
    quasiperiodic: QuasiperiodicParams = QuasiperiodicParams()
    lorenz: LorenzParams = LorenzParams()
    rossler: RosslerParams = RosslerParams()

    @property
    def dataset_id(self) -> str:
        return "regime_switching"

    @property
    def regime(self) -> Regime:
        return "nonstationary"

    def params_dict(self) -> dict[str, Any]:
        p = self.params
        return {
            "change_frac_range": list(p.change_frac_range),
            "smooth_width_frac": p.smooth_width_frac,
            "min_smooth_width": p.min_smooth_width,
            "lorenz_burn_in": p.lorenz_burn_in,
            "rossler_burn_in": p.rossler_burn_in,
            "standardized": True,
        }

    def generate(
        self,
        n_steps: int,
        *,
        seed: int = 0,
        burn_in: int | None = None,
    ) -> GeneratedSeries:
        del burn_in  # each child generator discards its own transient
        if n_steps < 1:
            raise ValueError(f"n_steps must be positive; got {n_steps}")
        p = self.params
        rng = np.random.default_rng(seed)
        seed_quasi = int(rng.integers(0, 2**31 - 1))
        seed_chaos = int(rng.integers(0, 2**31 - 1))
        source = "lorenz" if float(rng.random()) < 0.5 else "rossler"
        mode = "abrupt" if float(rng.random()) < 0.5 else "smooth"

        lo = int(np.floor(p.change_frac_range[0] * n_steps))
        hi = int(np.floor(p.change_frac_range[1] * n_steps))
        hi = min(max(hi, 0), n_steps - 1)
        lo = min(max(lo, 0), hi)
        change_index = int(rng.integers(lo, hi + 1))

        quasi = Quasiperiodic(params=self.quasiperiodic).generate(
            n_steps, seed=seed_quasi
        )
        if source == "lorenz":
            chaotic = LorenzX(params=self.lorenz).generate(
                n_steps,
                seed=seed_chaos,
                burn_in=int(p.lorenz_burn_in),
            )
        else:
            chaotic = Rossler(params=self.rossler).generate(
                n_steps,
                seed=seed_chaos,
                burn_in=int(p.rossler_burn_in),
            )

        quasi_z = _standardize(quasi.values)
        chaos_z = _standardize(chaotic.values)
        values = quasi_z.copy()
        width = 0
        if mode == "abrupt":
            values[change_index:] = chaos_z[change_index:]
        else:
            width = max(
                int(p.min_smooth_width),
                int(round(p.smooth_width_frac * n_steps)),
            )
            width = min(width, n_steps)
            start = max(0, change_index - width // 2)
            end = min(n_steps, start + width)
            if end <= start:
                values[change_index:] = chaos_z[change_index:]
                width = 0
                mode = "abrupt"
            else:
                # Raised cosine from 0 (quasiperiodic) to 1 (chaotic).
                alpha = 0.5 - 0.5 * np.cos(np.linspace(0.0, np.pi, end - start))
                values[start:end] = (1.0 - alpha) * quasi_z[start:end] + alpha * chaos_z[
                    start:end
                ]
                values[end:] = chaos_z[end:]

        if not np.isfinite(values).all():
            raise RuntimeError("regime_switching produced non-finite values")
        logger.info(
            "regime_switching seed=%d source=%s transition=%s change_index=%d/%d",
            seed,
            source,
            mode,
            change_index,
            n_steps,
        )
        used = {
            "change_index": change_index,
            "change_fraction": float(change_index / n_steps),
            "transition": mode,
            "smooth_width": int(width),
            "chaotic_source": source,
            "standardized": True,
            "seed_quasiperiodic": seed_quasi,
            "seed_chaotic": seed_chaos,
            "change_frac_range": list(p.change_frac_range),
            "quasiperiodic": dict(quasi.params),
            "chaotic": dict(chaotic.params),
        }
        return self._finalize(values, seed=seed, burn_in=0, params=used)
