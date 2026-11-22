"""Atomic persistence helpers for paper-grade audit artifacts.

Writes the NPZ / CSV / JSON snapshots needed to regenerate figures and tables
without recomputing TS2G graphs.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

FIGURE_EXTENSIONS: tuple[str, ...] = ("pdf", "svg", "png")


def default_workers() -> int:
    """Return ``max(1, cpu_count - 1)`` for CLI defaults."""
    cpu = os.cpu_count() or 1
    return max(1, cpu - 1)


def resolve_lang(profile: str | None, lang: str | None) -> str:
    """Default to English when ``profile == 'paper'`` unless overridden."""
    if lang in {"en", "es"}:
        return lang
    if profile == "paper":
        return "en"
    return "en" if lang is None else str(lang)


def atomic_write_bytes(path: Path, data: bytes) -> None:
    """Write bytes via temp file + ``os.replace``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        dir=str(path.parent),
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
        os.replace(tmp_name, path)
    except Exception:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


def atomic_write_text(path: Path, text: str) -> None:
    """Write UTF-8 text atomically."""
    atomic_write_bytes(path, text.encode("utf-8"))


def atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Serialize ``payload`` to indented JSON atomically."""
    atomic_write_text(path, json.dumps(payload, indent=2, default=str))


def git_commit_hash(repo_root: Path | None = None) -> str:
    """Best-effort short git SHA; ``unknown`` if unavailable."""
    root = repo_root or Path(__file__).resolve().parents[3]
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(root),
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return out.strip() or "unknown"
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def package_versions() -> dict[str, str]:
    """Versions of the core numeric stack used in audits."""
    versions: dict[str, str] = {}
    for name in ("numpy", "scipy", "sklearn", "networkx"):
        try:
            mod = __import__(name if name != "sklearn" else "sklearn")
            versions[name if name != "sklearn" else "scikit-learn"] = getattr(
                mod, "__version__", "unknown"
            )
        except ImportError:
            versions[name if name != "sklearn" else "scikit-learn"] = "missing"
    return versions


def write_config_snapshot(
    output_dir: Path,
    *,
    block: str,
    profile: str | None,
    seed: int,
    config: Mapping[str, Any],
    extra: Mapping[str, Any] | None = None,
) -> Path:
    """Write ``config_snapshot.json`` with full reproducibility metadata."""
    payload: dict[str, Any] = {
        "block": block,
        "profile": profile,
        "seed": seed,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit_hash(),
        "versions": package_versions(),
        "config": dict(config),
    }
    if extra:
        payload.update(dict(extra))
    path = Path(output_dir) / "config_snapshot.json"
    atomic_write_json(path, payload)
    logger.info("Wrote config snapshot %s", path)
    return path


def save_features_npz(
    path: Path,
    *,
    x: pd.DataFrame | np.ndarray,
    y: Sequence[str] | pd.Series | np.ndarray,
    feature_names: Sequence[str] | None = None,
    regime_names: Sequence[str] | None = None,
    realization_ids: Sequence[int] | np.ndarray | None = None,
    transform_or_kind: Sequence[str] | None = None,
    n_steps: Sequence[int] | np.ndarray | None = None,
    snr_db: Sequence[float] | np.ndarray | None = None,
    extra: Mapping[str, Any] | None = None,
) -> Path:
    """Persist a feature matrix and aligned metadata as NPZ (atomic)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(x, pd.DataFrame):
        x_arr = x.to_numpy(dtype=float)
        names = list(feature_names) if feature_names is not None else list(x.columns)
    else:
        x_arr = np.asarray(x, dtype=float)
        if feature_names is None:
            raise ValueError("feature_names required when x is not a DataFrame")
        names = list(feature_names)
    y_arr = np.asarray(list(y), dtype=object)
    payload: dict[str, Any] = {
        "X": x_arr,
        "feature_names": np.asarray(names, dtype=object),
        "y": y_arr,
    }
    if regime_names is not None:
        payload["regime_names"] = np.asarray(list(regime_names), dtype=object)
    if realization_ids is not None:
        payload["realization_ids"] = np.asarray(realization_ids, dtype=int)
    if transform_or_kind is not None:
        payload["transform_or_kind"] = np.asarray(list(transform_or_kind), dtype=object)
    if n_steps is not None:
        payload["n_steps"] = np.asarray(n_steps, dtype=int)
    if snr_db is not None:
        # Use object so None / inf survive round-trip.
        payload["snr_db"] = np.asarray(
            [np.inf if v is None else float(v) for v in snr_db],
            dtype=float,
        )
    if extra:
        for key, value in extra.items():
            payload[str(key)] = value

    fd, tmp_name = tempfile.mkstemp(
        dir=str(path.parent),
        prefix=f".{path.name}.",
        suffix=".tmp.npz",
    )
    os.close(fd)
    try:
        np.savez_compressed(tmp_name, **payload)
        os.replace(tmp_name, path)
    except Exception:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise
    logger.info("Wrote features NPZ %s shape=%s", path, x_arr.shape)
    return path


def load_features_npz(path: Path) -> dict[str, Any]:
    """Load a features NPZ into a plain dict of arrays / lists."""
    with np.load(path, allow_pickle=True) as data:
        out: dict[str, Any] = {key: data[key] for key in data.files}
    out["feature_names"] = [str(v) for v in out["feature_names"].tolist()]
    out["y"] = [str(v) for v in out["y"].tolist()]
    if "regime_names" in out:
        out["regime_names"] = [str(v) for v in out["regime_names"].tolist()]
    if "transform_or_kind" in out:
        out["transform_or_kind"] = [str(v) for v in out["transform_or_kind"].tolist()]
    return out


def save_oof_predictions(
    path: Path,
    bundles: Mapping[str, Mapping[str, Any]],
) -> Path:
    """Save nested OOF predictions keyed by ``view__model``.

    Each bundle must provide ``y_true``, ``y_pred``, ``fold_id`` and optionally
    ``y_proba`` (n_samples × n_classes).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {}
    keys: list[str] = []
    for key, bundle in bundles.items():
        safe = str(key).replace("/", "_")
        keys.append(safe)
        payload[f"{safe}__y_true"] = np.asarray(bundle["y_true"], dtype=object)
        payload[f"{safe}__y_pred"] = np.asarray(bundle["y_pred"], dtype=object)
        payload[f"{safe}__fold_id"] = np.asarray(bundle["fold_id"], dtype=int)
        if bundle.get("y_proba") is not None:
            payload[f"{safe}__y_proba"] = np.asarray(bundle["y_proba"], dtype=float)
        if bundle.get("class_names") is not None:
            payload[f"{safe}__class_names"] = np.asarray(
                bundle["class_names"], dtype=object
            )
    payload["keys"] = np.asarray(keys, dtype=object)

    fd, tmp_name = tempfile.mkstemp(
        dir=str(path.parent),
        prefix=f".{path.name}.",
        suffix=".tmp.npz",
    )
    os.close(fd)
    try:
        np.savez_compressed(tmp_name, **payload)
        os.replace(tmp_name, path)
    except Exception:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise
    logger.info("Wrote OOF predictions %s (%d views)", path, len(keys))
    return path


def load_oof_predictions(path: Path) -> dict[str, dict[str, Any]]:
    """Inverse of :func:`save_oof_predictions`."""
    with np.load(path, allow_pickle=True) as data:
        keys = [str(k) for k in data["keys"].tolist()]
        out: dict[str, dict[str, Any]] = {}
        for key in keys:
            bundle: dict[str, Any] = {
                "y_true": [str(v) for v in data[f"{key}__y_true"].tolist()],
                "y_pred": [str(v) for v in data[f"{key}__y_pred"].tolist()],
                "fold_id": data[f"{key}__fold_id"].astype(int),
            }
            proba_key = f"{key}__y_proba"
            if proba_key in data.files:
                bundle["y_proba"] = data[proba_key]
            names_key = f"{key}__class_names"
            if names_key in data.files:
                bundle["class_names"] = [str(v) for v in data[names_key].tolist()]
            out[key] = bundle
    return out


def save_table_csv(path: Path, frame: pd.DataFrame) -> Path:
    """Write a numeric table as CSV atomically."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        dir=str(path.parent),
        prefix=f".{path.name}.",
        suffix=".tmp.csv",
    )
    os.close(fd)
    try:
        frame.to_csv(tmp_name, index=False, float_format="%.10g")
        os.replace(tmp_name, path)
    except Exception:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise
    return path


def save_figure_trio(fig: Any, stem: Path, *, dpi: int = 300) -> list[Path]:
    """Save a matplotlib figure as PDF + SVG + PNG (300 dpi by default)."""
    import matplotlib.pyplot as plt

    stem = Path(stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for ext in FIGURE_EXTENSIONS:
        target = stem.with_suffix(f".{ext}")
        fd, tmp_name = tempfile.mkstemp(
            dir=str(target.parent),
            prefix=f".{target.name}.",
            suffix=f".tmp.{ext}",
        )
        os.close(fd)
        try:
            extras: dict[str, Any] = {"format": ext, "bbox_inches": "tight"}
            if ext == "png":
                extras["dpi"] = dpi
            fig.savefig(tmp_name, facecolor="white", **extras)
            os.replace(tmp_name, target)
            written.append(target)
        except Exception:
            try:
                os.unlink(tmp_name)
            except OSError:
                pass
            raise
    plt.close(fig)
    logger.info("Wrote figure trio %s.{pdf,svg,png}", stem)
    return written


def write_metrics_alias(output_dir: Path, metrics_path: Path) -> Path:
    """Copy/symlink canonical metrics to ``metrics.json`` for resume checks."""
    alias = Path(output_dir) / "metrics.json"
    if metrics_path.resolve() == alias.resolve():
        return alias
    atomic_write_text(alias, Path(metrics_path).read_text(encoding="utf-8"))
    return alias


def meta_from_records(
    records: Sequence[Mapping[str, Any]],
) -> dict[str, list[Any]]:
    """Extract aligned metadata columns from ensemble-like records."""
    realization_ids = [int(r.get("realization", i)) for i, r in enumerate(records)]
    transform_or_kind: list[str] = []
    n_steps: list[int] = []
    snr_db: list[float] = []
    for rec in records:
        kind = rec.get("surrogate_kind", rec.get("transform", "original"))
        transform_or_kind.append(str(kind))
        n_steps.append(int(rec.get("n_steps", rec.get("series_length", -1))) )
        snr = rec.get("snr_db", np.inf)
        snr_db.append(np.inf if snr is None else float(snr))
    return {
        "realization_ids": realization_ids,
        "transform_or_kind": transform_or_kind,
        "n_steps": n_steps,
        "snr_db": snr_db,
    }
