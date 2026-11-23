"""Generate R realizations × the configured regimes and extract descriptor matrices D_r."""

from __future__ import annotations

import hashlib
import json
import logging
import pickle
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Callable

import numpy as np
from numpy.typing import NDArray

from tsg.audit.config import AuditConfig
from tsg.descriptors.config import DescriptorConfig
from tsg.descriptors.pipeline import DescriptorPipeline, results_to_matrix
from tsg.generators.config import GeneratorSuiteConfig
from tsg.generators.registry import (
    CANONICAL_REGIMES,
    REGIME_LABELS,
    build_generator,
)
from tsg.graphs.config import TS2GSuiteConfig
from tsg.graphs.registry import build_transformers

logger = logging.getLogger(__name__)

# Bump when generation / preprocessing semantics change (invalidates old caches).
# fase2_v1: frontier regimes + profile/regime list inside the cache fingerprint.
ENSEMBLE_CACHE_VERSION = "fase2_v1"

_REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_GENERATOR_SUITE_YAML = _REPO_ROOT / "configs" / "synthetic_suite_phase1.yaml"

# Integrated paths. ``brownian_mode`` rewrites STN/OPN inputs for every member,
# so fBm is treated the same way as classical Brownian motion.
INTEGRATED_TRAJECTORY_IDS: frozenset[str] = frozenset(
    {"brownian", "fbm_antipersistent", "fbm_persistent"}
)

# Re-exported so existing imports of the audit ensemble keep working.
REGIME_LABEL: dict[str, str] = REGIME_LABELS


def realization_seed(base_seed: int, dataset_id: str, realization: int) -> int:
    """Deterministic seed using the canonical regime index."""
    return int(base_seed + 1000 * CANONICAL_REGIMES.index(dataset_id) + realization)


def load_generator_suite_config(
    path: Path | str | None = None,
) -> GeneratorSuiteConfig:
    """Load the Phase-1 synthetic suite YAML (or defaults if missing)."""
    yaml_path = Path(path) if path is not None else DEFAULT_GENERATOR_SUITE_YAML
    if yaml_path.exists():
        logger.info("Loading generator suite config from %s", yaml_path)
        return GeneratorSuiteConfig.from_yaml(yaml_path)
    logger.warning(
        "Generator suite YAML not found at %s; using dataclass defaults",
        yaml_path,
    )
    return GeneratorSuiteConfig()


def ensemble_cache_fingerprint(
    audit_config: AuditConfig,
    generator_config: GeneratorSuiteConfig,
) -> str:
    """Stable subdirectory name so old caches never mix with new semantics."""
    payload = {
        "cache_version": ENSEMBLE_CACHE_VERSION,
        "n_steps": audit_config.n_steps,
        "n_realizations": audit_config.n_realizations,
        "profile": audit_config.profile,
        "regimes": list(audit_config.regimes),
        "brownian_mode": audit_config.brownian_mode,
        "methods": list(audit_config.methods),
        "seed": audit_config.seed,
        "generator": asdict(generator_config),
    }
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()[:12]
    return f"{ENSEMBLE_CACHE_VERSION}_{digest}"


def versioned_cache_dir(
    cache_dir: Path,
    audit_config: AuditConfig,
    generator_config: GeneratorSuiteConfig,
) -> Path:
    """Return ``cache_dir / <fingerprint>`` for this audit+generator config."""
    return cache_dir / ensemble_cache_fingerprint(audit_config, generator_config)


def _cache_path(cache_dir: Path, dataset_id: str, realization: int, seed: int) -> Path:
    return cache_dir / dataset_id / f"r{realization:03d}_seed{seed}.pkl"


def save_record(record: dict[str, Any], cache_dir: Path) -> Path:
    """Persist one ensemble record (includes the descriptor matrix)."""
    path = _cache_path(
        cache_dir,
        str(record["dataset_id"]),
        int(record["realization"]),
        int(record["seed"]),
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as fh:
        pickle.dump(record, fh, protocol=pickle.HIGHEST_PROTOCOL)
    return path


def load_record(
    cache_dir: Path, dataset_id: str, realization: int, seed: int
) -> dict[str, Any] | None:
    path = _cache_path(cache_dir, dataset_id, realization, seed)
    if not path.exists():
        return None
    with path.open("rb") as fh:
        return pickle.load(fh)


def _prepare_series_for_method(
    values: NDArray[np.floating],
    dataset_id: str,
    method: str,
    config: AuditConfig,
) -> NDArray[np.floating]:
    """Optionally transform integrated trajectories before transition networks.

    ``brownian_mode`` applies to every cumulative path (classical Brownian
    motion and both fBm variants). ``levels`` leaves the path unchanged.
    """
    x = np.asarray(values, dtype=float).ravel()
    if dataset_id not in INTEGRATED_TRAJECTORY_IDS:
        return x
    if method not in {"STN", "OPN"}:
        return x
    if config.brownian_mode == "levels":
        return x
    if config.brownian_mode == "increments":
        if x.size < 3:
            return x
        return np.diff(x)
    if config.brownian_mode == "stl_residuals":
        # Lightweight STL surrogate: high-pass residual via moving-average detrend.
        window = max(5, min(51, (x.size // 20) | 1))
        kernel = np.ones(window, dtype=float) / float(window)
        trend = np.convolve(x, kernel, mode="same")
        return x - trend
    raise ValueError(f"Unknown brownian_mode={config.brownian_mode!r}")


def generate_one_matrix(
    dataset_id: str,
    *,
    seed: int,
    audit_config: AuditConfig,
    generator_config: GeneratorSuiteConfig | None = None,
    ts2g_config: TS2GSuiteConfig | None = None,
    descriptor_config: DescriptorConfig | None = None,
    values_transform: Callable[[NDArray[np.floating]], NDArray[np.floating]] | None = None,
) -> dict[str, Any]:
    """Generate one series, map with all TS2G methods and return D matrix.

    Parameters
    ----------
    values_transform
        Optional map applied to the **raw** series *before* any TS2G method
        sees it (used by the RQ1 stress suite for SNR noise / truncation).
    """
    gen_cfg = generator_config or load_generator_suite_config()
    # Align series length with the audit run; keep chaotic burn-in defaults from YAML.
    gen_cfg = replace(
        gen_cfg,
        global_params=replace(
            gen_cfg.global_params,
            n_steps=audit_config.n_steps,
            seed=seed,
            n_samples=1,
        ),
    )

    generator = build_generator(dataset_id, gen_cfg)
    series = generator.generate(audit_config.n_steps, seed=seed)
    values = np.asarray(series.values, dtype=float).ravel()
    if values_transform is not None:
        values = np.asarray(values_transform(values), dtype=float).ravel()
        if values.size < 8:
            raise ValueError(
                f"Transformed series too short for TS2G (n={values.size}) "
                f"dataset={dataset_id} seed={seed}"
            )

    ts2g_cfg = ts2g_config or TS2GSuiteConfig(methods=audit_config.methods)
    transformers = {
        t.method: t
        for t in build_transformers(ts2g_cfg)
        if t.method in audit_config.methods
    }
    pipeline = DescriptorPipeline(config=descriptor_config or DescriptorConfig())
    desc_results = []
    for method in audit_config.methods:
        tr = transformers[method]
        series_m = _prepare_series_for_method(values, dataset_id, method, audit_config)
        graph_result = tr.transform(
            series_m,
            dataset_id,
            regime=REGIME_LABEL[dataset_id],
        )
        desc_results.append(
            pipeline.compute(
                graph_result.graph,
                dataset_id=dataset_id,
                method=method,
                regime=REGIME_LABEL[dataset_id],
                family=graph_result.family,
            )
        )
    matrix = results_to_matrix(desc_results, method_order=audit_config.methods)
    return {
        "dataset_id": dataset_id,
        "regime": REGIME_LABEL[dataset_id],
        "seed": seed,
        "matrix": matrix,
        "series_length": int(values.size),
        "series_params": series.params,
        "cache_version": ENSEMBLE_CACHE_VERSION,
    }


def _ensemble_job(
    dataset_id: str,
    realization: int,
    seed: int,
    *,
    audit_config: AuditConfig,
    generator_config: GeneratorSuiteConfig,
    descriptor_config: DescriptorConfig | None,
    cache: Path | None,
    force: bool,
) -> dict[str, Any]:
    """One (regime, realization) unit — safe for process-pool execution."""
    if cache is not None and not force:
        cached = load_record(cache, dataset_id, realization, seed)
        if cached is not None:
            cached.setdefault("realization", realization)
            return cached
    rec = generate_one_matrix(
        dataset_id,
        seed=seed,
        audit_config=audit_config,
        generator_config=generator_config,
        descriptor_config=descriptor_config,
    )
    rec["realization"] = realization
    if cache is not None:
        save_record(rec, cache)
    return rec


def build_ensemble_records(
    config: AuditConfig,
    *,
    descriptor_config: DescriptorConfig | None = None,
    generator_config: GeneratorSuiteConfig | None = None,
    generator_suite_yaml: Path | str | None = None,
    cache_dir: Path | str | None = None,
    force_regimes: tuple[str, ...] | None = None,
    n_workers: int = 1,
) -> list[dict[str, Any]]:
    """Build descriptor-matrix records with controlled seeds.

    Parameters
    ----------
    cache_dir
        If set, load existing records from a **versioned** subdirectory derived
        from the audit + generator fingerprint, so stale caches never mix.
    force_regimes
        Regimes that must be recomputed even if a cache entry exists.
    n_workers
        Parallel workers for graph/descriptor extraction. Each task derives a
        deterministic seed via :func:`realization_seed`, so ``workers=1`` and
        ``workers>1`` produce identical matrices.
    """
    gen_cfg = generator_config or load_generator_suite_config(generator_suite_yaml)
    cache_root = Path(cache_dir) if cache_dir is not None else None
    cache = (
        versioned_cache_dir(cache_root, config, gen_cfg)
        if cache_root is not None
        else None
    )
    if cache is not None:
        cache.mkdir(parents=True, exist_ok=True)
        logger.info("Ensemble cache directory: %s", cache)

    force = set(force_regimes or ())
    jobs: list[tuple[str, int, int, bool]] = []
    for dataset_id in config.regimes:
        for r in range(config.n_realizations):
            seed = realization_seed(config.seed, dataset_id, r)
            jobs.append((dataset_id, r, seed, dataset_id in force))

    total = len(jobs)
    workers = max(1, int(n_workers))
    logger.info(
        "Building ensemble jobs=%d workers=%d regimes=%d R=%d",
        total,
        workers,
        len(config.regimes),
        config.n_realizations,
    )

    if workers == 1:
        records: list[dict[str, Any]] = []
        for done, (dataset_id, r, seed, must_force) in enumerate(jobs, start=1):
            logger.info(
                "Ensemble %d/%d dataset=%s realization=%d seed=%d",
                done,
                total,
                dataset_id,
                r,
                seed,
            )
            records.append(
                _ensemble_job(
                    dataset_id,
                    r,
                    seed,
                    audit_config=config,
                    generator_config=gen_cfg,
                    descriptor_config=descriptor_config,
                    cache=cache,
                    force=must_force,
                )
            )
        return records

    from joblib import Parallel, delayed

    computed = Parallel(n_jobs=workers, prefer="processes")(
        delayed(_ensemble_job)(
            dataset_id,
            r,
            seed,
            audit_config=config,
            generator_config=gen_cfg,
            descriptor_config=descriptor_config,
            cache=cache,
            force=must_force,
        )
        for dataset_id, r, seed, must_force in jobs
    )
    return list(computed)
