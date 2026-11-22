"""Configuration for Phase-1 discriminability audit."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Literal

import yaml

BrownianMode = Literal["levels", "increments", "stl_residuals"]

# CLI fallbacks. A YAML ``profiles`` entry overrides the matching name.
# Paper defaults target a single-night Q1 run (N=1500 for Bloque 3/1).
BUILTIN_PROFILES: dict[str, dict[str, int]] = {
    "dev": {"n_realizations": 10, "n_steps": 1000},
    "paper": {"n_realizations": 30, "n_steps": 1500},
}


@dataclass(frozen=True)
class AuditConfig:
    """Audit hyperparameters and paths."""

    n_steps: int = 5000
    n_realizations: int = 30
    seed: int = 42
    brownian_mode: BrownianMode = "levels"
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
            "similarity": ("Correlation", "Segment"),
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
    # Active profile name (``dev`` / ``paper``). Included in the ensemble-cache hash.
    profile: str | None = None
    profiles: dict[str, dict[str, int]] = field(default_factory=dict)

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
            elif key == "profiles" and isinstance(value, dict):
                kwargs[key] = _parse_profiles(value)
            elif key == "profile":
                kwargs[key] = None if value in (None, "", "null") else str(value)
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


def apply_profile(config: AuditConfig, profile: str) -> AuditConfig:
    """Return ``config`` with ``n_realizations`` and ``n_steps`` from ``profile``.

    YAML entries override :data:`BUILTIN_PROFILES` for the same name. The chosen
    name is stored on ``config.profile`` so the ensemble cache cannot mix runs.
    """
    if profile not in BUILTIN_PROFILES and profile not in config.profiles:
        known = sorted(set(BUILTIN_PROFILES) | set(config.profiles))
        raise ValueError(f"Unknown profile {profile!r}. Known: {known}")
    chosen = dict(BUILTIN_PROFILES.get(profile, {}))
    chosen.update(config.profiles.get(profile, {}))
    missing = [k for k in ("n_realizations", "n_steps") if k not in chosen]
    if missing:
        raise ValueError(f"Profile {profile!r} is missing keys: {missing}")
    return replace(
        config,
        n_realizations=int(chosen["n_realizations"]),
        n_steps=int(chosen["n_steps"]),
        profile=profile,
    )


def _parse_profiles(raw: dict[str, Any]) -> dict[str, dict[str, int]]:
    parsed: dict[str, dict[str, int]] = {}
    for name, body in raw.items():
        if not isinstance(body, dict):
            raise ValueError(f"Profile {name!r} must be a mapping")
        parsed[str(name)] = {
            "n_realizations": int(body["n_realizations"]),
            "n_steps": int(body["n_steps"]),
        }
    return parsed
