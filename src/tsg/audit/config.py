"""Configuration for Phase-1 discriminability audit."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import yaml

BrownianMode = Literal["levels", "increments", "stl_residuals"]


@dataclass(frozen=True)
class AuditConfig:
    """Audit hyperparameters and paths."""

    n_steps: int = 5000
    n_realizations: int = 30
    seed: int = 42
    brownian_mode: BrownianMode = "increments"
    n_folds: int = 5
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
    families: dict[str, tuple[str, ...]] = field(
        default_factory=lambda: {
            "visibility": ("NVG", "HVG"),
            "transition": ("STN", "OPN"),
            "proximity": ("RN", "kNN"),
        }
    )
    critical_pairs: tuple[tuple[str, str], ...] = (
        ("logistic", "white_noise"),
        ("lorenz", "white_noise"),
        ("lorenz", "quasiperiodic"),
    )
    regimes: tuple[str, ...] = (
        "white_noise",
        "brownian",
        "sine",
        "quasiperiodic",
        "lorenz",
        "logistic",
    )
    variance_eps: float = 1e-12
    figure_dpi: int = 300

    @classmethod
    def from_mapping(cls, data: dict[str, Any]) -> AuditConfig:
        fields = getattr(cls, "__dataclass_fields__", {})
        kwargs: dict[str, Any] = {}
        for key, value in data.items():
            if key not in fields:
                continue
            if key in {"methods", "regimes"} and isinstance(value, list):
                kwargs[key] = tuple(value)
            elif key == "families" and isinstance(value, dict):
                kwargs[key] = {k: tuple(v) for k, v in value.items()}
            elif key == "critical_pairs" and isinstance(value, list):
                kwargs[key] = tuple(tuple(pair) for pair in value)
            else:
                kwargs[key] = value
        return cls(**kwargs)

    @classmethod
    def from_yaml(cls, path: Path | str) -> AuditConfig:
        with open(path, encoding="utf-8") as fh:
            raw = yaml.safe_load(fh) or {}
        if not isinstance(raw, dict):
            raise ValueError(f"Expected mapping in {path}, got {type(raw)!r}")
        return cls.from_mapping(raw.get("audit", raw))
