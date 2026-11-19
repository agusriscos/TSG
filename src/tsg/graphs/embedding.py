"""Takens delay embedding and Theiler-window helpers."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def delay_embedding(
    values: NDArray[np.floating] | np.ndarray,
    m: int,
    tau: int,
) -> NDArray[np.floating]:
    """Return delay vectors of shape ``(N - (m-1)τ, m)``."""
    x = np.asarray(values, dtype=float).ravel()
    n = x.size
    length = n - (m - 1) * tau
    if length <= 1:
        raise ValueError(f"Series too short for embedding m={m}, tau={tau}, n={n}")
    return np.column_stack([x[i * tau : i * tau + length] for i in range(m)])


def theiler_mask(n: int, w: int) -> NDArray[np.bool_]:
    """Boolean mask of shape ``(n, n)`` that is True where ``|i-j| <= w``."""
    idx = np.arange(n)
    return np.abs(idx[:, None] - idx[None, :]) <= w
