"""Generate R realizations × 6 regimes and extract descriptor matrices D_r."""

from __future__ import annotations

import logging
import pickle
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from tsg.audit.config import AuditConfig
from tsg.descriptors.config import DescriptorConfig
from tsg.descriptors.pipeline import DescriptorPipeline, results_to_matrix
from tsg.generators.chaotic import LogisticMap, LorenzX
from tsg.generators.config import GeneratorSuiteConfig, GlobalGenerationParams
from tsg.generators.periodic import Quasiperiodic, SineWave
from tsg.generators.stochastic import BrownianMotion, WhiteNoise
from tsg.graphs.config import TS2GSuiteConfig
from tsg.graphs.registry import build_transformers

logger = logging.getLogger(__name__)

# Fixed order keeps seeds stable even when auditing a regime subset.
CANONICAL_REGIMES: tuple[str, ...] = (
    "white_noise",
    "brownian",
    "sine",
    "quasiperiodic",
    "lorenz",
    "logistic",
)

REGIME_TO_GENERATOR = {
    "white_noise": lambda cfg: WhiteNoise(params=cfg.white_noise),
    "brownian": lambda cfg: BrownianMotion(params=cfg.brownian),
    "sine": lambda cfg: SineWave(params=cfg.sine),
    "quasiperiodic": lambda cfg: Quasiperiodic(params=cfg.quasiperiodic),
    "lorenz": lambda cfg: LorenzX(params=cfg.lorenz),
    "logistic": lambda cfg: LogisticMap(params=cfg.logistic),
}

REGIME_LABEL: dict[str, str] = {
    "white_noise": "stochastic",
    "brownian": "stochastic",
    "sine": "periodic",
    "quasiperiodic": "periodic",
    "lorenz": "chaotic",
    "logistic": "chaotic",
}


def realization_seed(base_seed: int, dataset_id: str, realization: int) -> int:
    """Deterministic seed using the canonical regime index."""
    return int(base_seed + 1000 * CANONICAL_REGIMES.index(dataset_id) + realization)


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
    """Optionally transform Brownian series before transition networks."""
    x = np.asarray(values, dtype=float).ravel()
    if dataset_id != "brownian":
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
) -> dict[str, Any]:
    """Generate one series, map with all TS2G methods and return D matrix."""
    gen_cfg = generator_config or GeneratorSuiteConfig(
        global_params=GlobalGenerationParams(
            n_steps=audit_config.n_steps,
            seed=seed,
            n_samples=1,
        )
    )
    # Keep chaotic burn-ins but align length.
    if dataset_id == "lorenz":
        gen_cfg = replace(
            gen_cfg,
            lorenz=replace(gen_cfg.lorenz, burn_in=min(gen_cfg.lorenz.burn_in, 1000)),
        )
    if dataset_id == "logistic":
        gen_cfg = replace(
            gen_cfg,
            logistic=replace(gen_cfg.logistic, burn_in=min(gen_cfg.logistic.burn_in, 200)),
        )

    factory = REGIME_TO_GENERATOR[dataset_id]
    generator = factory(gen_cfg)
    series = generator.generate(audit_config.n_steps, seed=seed)
    values = series.values

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
    }


def build_ensemble_records(
    config: AuditConfig,
    *,
    descriptor_config: DescriptorConfig | None = None,
    cache_dir: Path | str | None = None,
    force_regimes: tuple[str, ...] | None = None,
) -> list[dict[str, Any]]:
    """Build descriptor-matrix records with controlled seeds.

    Parameters
    ----------
    cache_dir
        If set, load existing records from disk and only regenerate missing
        ones (or those listed in ``force_regimes``).
    force_regimes
        Regimes that must be recomputed even if a cache entry exists.
    """
    cache = Path(cache_dir) if cache_dir is not None else None
    force = set(force_regimes or ())
    records: list[dict[str, Any]] = []
    total = len(config.regimes) * config.n_realizations
    done = 0
    for dataset_id in config.regimes:
        for r in range(config.n_realizations):
            seed = realization_seed(config.seed, dataset_id, r)
            done += 1
            rec: dict[str, Any] | None = None
            if cache is not None and dataset_id not in force:
                rec = load_record(cache, dataset_id, r, seed)
                if rec is not None:
                    logger.info(
                        "Ensemble %d/%d dataset=%s realization=%d seed=%d [cache]",
                        done,
                        total,
                        dataset_id,
                        r,
                        seed,
                    )
            if rec is None:
                logger.info(
                    "Ensemble %d/%d dataset=%s realization=%d seed=%d [compute]",
                    done,
                    total,
                    dataset_id,
                    r,
                    seed,
                )
                rec = generate_one_matrix(
                    dataset_id,
                    seed=seed,
                    audit_config=config,
                    descriptor_config=descriptor_config,
                )
                rec["realization"] = r
                if cache is not None:
                    save_record(rec, cache)
            records.append(rec)
    return records
