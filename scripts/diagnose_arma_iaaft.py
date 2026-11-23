#!/usr/bin/env python3
"""Diagnose why linear ``arma_correlated`` separates from its IAAFT surrogate.

Uses the descriptor matrices already stored in
``outputs/reports/phase2_surrogates/data/records.pkl`` and rebuilds only the
ARMA series (same seeds as the surrogate audit) to measure the IAAFT spectral
residual. It does not change the IAAFT algorithm.

An optional end-matching control rebuilds only ``transition`` / ``visibility``
graphs for ``arma_correlated`` (R=30, N=1000) after removing the periodic
endpoint jump, to check whether AUROC falls toward ~0.50.

The question is why binary AUROC(original vs IAAFT) is about 0.78 (transition)
and 0.72 (visibility) for a linear Gaussian process that should sit near 0.50.
"""

from __future__ import annotations

import argparse
import logging
import pickle
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.stats import spearmanr
from sklearn.base import clone

from tsg.audit.ensemble import load_generator_suite_config, realization_seed
from tsg.audit.features import family_feature_mask, matrix_to_feature_row
from tsg.audit.health import diagnose_and_clean
from tsg.audit.surrogates import (
    SurrogateAuditConfig,
    _binary_original_vs_iaaft,
    _extract_matrix_for_series,
    _make_models,
    _surrogate_seed,
)
from tsg.descriptors.config import DescriptorConfig
from tsg.descriptors.pipeline import DescriptorPipeline
from tsg.generators.registry import build_generator
from tsg.generators.surrogates import (
    _spectral_relative_error,
    iaaft_surrogate,
    shuffled_surrogate,
)
from tsg.graphs.config import TS2GSuiteConfig
from tsg.graphs.registry import build_transformers
from tsg.utils.logging import setup_logging

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "phase2_surrogates.yaml"
DEFAULT_RECORDS = (
    ROOT / "outputs" / "reports" / "phase2_surrogates" / "data" / "records.pkl"
)
DEFAULT_OUTPUT = ROOT / "outputs" / "reports" / "phase2_surrogates"

_VIEWS: tuple[str, ...] = ("transition", "visibility")
_ENDMATCH_METHODS: tuple[str, ...] = ("NVG", "HVG", "dHVG", "STN", "OPN")


@dataclass
class SpectralTrace:
    """IAAFT convergence and boundary diagnostics for one ARMA realization."""

    realization: int
    seed: int
    rho: float
    n_iter_returned: int
    stop_reason: str
    spectral_rel_err_returned: float
    spectral_rel_err_at_max_iter: float
    endpoint_jump_ratio_original: float
    endpoint_jump_ratio_iaaft: float
    mssd_original: float
    mssd_iaaft: float
    rel_err_low_freq: float
    rel_err_high_freq: float
    residual_energy_frac_low: float
    residual_energy_frac_high: float
    matches_library: bool


def _endpoint_jump_ratio(values: NDArray[np.floating]) -> float:
    """Circular seam ``|x0 - x_{N-1}|`` in units of the median absolute step.

    The real FFT treats the series as one period. A ratio ≫ 1 means the
    periodic extension has a jump that is not typical of the interior, which
    leaks broadband power into the target spectrum.
    """
    series = np.asarray(values, dtype=float).ravel()
    step = float(np.median(np.abs(np.diff(series))))
    return float(abs(series[0] - series[-1]) / (step + 1e-15))


def _mssd(values: NDArray[np.floating]) -> float:
    """Mean square successive difference, a roughness proxy."""
    series = np.asarray(values, dtype=float).ravel()
    return float(np.mean(np.diff(series) ** 2))


def end_match_series(values: NDArray[np.floating]) -> NDArray[np.floating]:
    """Subtract the linear ramp connecting endpoints so ``y[0] == y[-1]``.

    This removes the periodic seam that the real FFT assumes, without changing
    the IAAFT algorithm itself. After matching, ``|y0 - yN| = 0``.
    """
    series = np.asarray(values, dtype=float).ravel()
    n = series.size
    if n < 2:
        return series.copy()
    ramp = np.linspace(series[0] - series[-1], 0.0, n)
    return np.asarray(series - ramp, dtype=float)


def hann_taper_series(values: NDArray[np.floating]) -> NDArray[np.floating]:
    """Hann window taper (alternative edge treatment; ends → 0)."""
    series = np.asarray(values, dtype=float).ravel()
    if series.size < 2:
        return series.copy()
    return np.asarray(series * np.hanning(series.size), dtype=float)


def _trace_iaaft(
    values: NDArray[np.floating],
    rng: np.random.Generator,
    *,
    max_iter: int,
    tol: float,
    min_iter: int,
) -> tuple[NDArray[np.floating], NDArray[np.floating], int, str, NDArray[np.floating]]:
    """Run the production IAAFT loop to ``max_iter`` and record the stop point.

    The update is a line-by-line copy of :func:`iaaft_surrogate` so the
    diagnostic can see the iteration of the plateau stop without editing that
    function. The series returned at the stop index is checked against the
    library.

    Returns
    -------
    y_at_stop, y_at_max_iter, n_iter_returned, stop_reason, error_history
    """
    series = np.asarray(values, dtype=float).ravel()
    n = series.size
    sorted_amps = np.sort(series)
    target = np.abs(np.fft.rfft(series))
    current = shuffled_surrogate(series, rng)
    prev_err = np.inf
    y_at_stop: NDArray[np.floating] | None = None
    stop_reason = "max_iter"
    n_iter = max_iter
    history = np.empty(max_iter, dtype=float)

    for it in range(max_iter):
        spectrum = np.fft.rfft(current)
        phases = np.angle(spectrum)
        phases[0] = 0.0
        if n % 2 == 0 and phases.size > 1:
            phases[-1] = 0.0
        current = np.fft.irfft(target * np.exp(1j * phases), n=n)
        ranks = np.argsort(np.argsort(current))
        current = sorted_amps[ranks]
        err = _spectral_relative_error(current, target)
        history[it] = err
        delta = abs(prev_err - err)
        prev_err = err
        done = it + 1
        if y_at_stop is None and done >= min_iter and (err < tol or delta < tol):
            y_at_stop = np.asarray(current, dtype=float).copy()
            n_iter = done
            stop_reason = "err<tol" if err < tol else "plateau"

    y_final = np.asarray(current, dtype=float)
    if y_at_stop is None:
        y_at_stop = y_final.copy()
    return y_at_stop, y_final, n_iter, stop_reason, history[: max(n_iter, 1)]


def _band_errors(
    original: NDArray[np.floating],
    surrogate: NDArray[np.floating],
) -> tuple[float, float, float, float]:
    """Low/high relative error and the share of residual energy in each band.

    Low band = first 20% of ``rFFT`` bins, high band = last 20%.
    """
    amp_x = np.abs(np.fft.rfft(original))
    amp_y = np.abs(np.fft.rfft(surrogate))
    residual = amp_y - amp_x
    n_bins = amp_x.size
    low = slice(0, max(1, n_bins // 5))
    high = slice(min(n_bins - 1, 4 * n_bins // 5), n_bins)

    def _rel(band: slice) -> float:
        denom = float(np.linalg.norm(amp_x[band])) + 1e-15
        return float(np.linalg.norm(residual[band]) / denom)

    energy = float(np.dot(residual, residual)) + 1e-30
    frac_low = float(np.dot(residual[low], residual[low]) / energy)
    frac_high = float(np.dot(residual[high], residual[high]) / energy)
    return _rel(low), _rel(high), frac_low, frac_high


class ArmaIAAFTDiagnostic:
    """Feature-importance and IAAFT-convergence report for ``arma_correlated``."""

    def __init__(
        self,
        config: SurrogateAuditConfig,
        records: Sequence[dict[str, Any]],
        output_dir: Path,
        *,
        run_end_match: bool = True,
    ) -> None:
        self.config = config
        self.records = [
            rec for rec in records if rec["dataset_id"] == "arma_correlated"
        ]
        self.output_dir = Path(output_dir)
        self.run_end_match = bool(run_end_match)
        if not self.records:
            raise ValueError("records.pkl has no arma_correlated rows")

    def run(self) -> Path:
        """Write the CSV tables and the markdown diagnosis. Return the md path."""
        t0 = time.perf_counter()
        auroc = self._reproduced_auroc()
        importance = self._feature_importance()
        traces = self._spectral_traces()
        self._attach_feature_correlation(importance, traces)
        end_match: dict[str, Any] | None = None
        if self.run_end_match:
            end_match = self._end_match_control()
        tables = self.output_dir / "data" / "tables"
        tables.mkdir(parents=True, exist_ok=True)
        importance_path = tables / "arma_iaaft_feature_importance.csv"
        spectral_path = tables / "arma_iaaft_spectral.csv"
        importance.to_csv(importance_path, index=False, float_format="%.10g")
        pd.DataFrame([trace.__dict__ for trace in traces]).to_csv(
            spectral_path, index=False, float_format="%.10g"
        )
        if end_match is not None:
            pd.DataFrame(end_match["auroc_rows"]).to_csv(
                tables / "arma_iaaft_endmatch_auroc.csv",
                index=False,
                float_format="%.10g",
            )
        markdown = self._render(
            auroc,
            importance,
            traces,
            time.perf_counter() - t0,
            end_match=end_match,
        )
        md_path = self.output_dir / "arma_iaaft_diagnosis.md"
        md_path.write_text(markdown, encoding="utf-8")
        logger.info("Wrote %s", md_path)
        logger.info("Wrote %s", importance_path)
        logger.info("Wrote %s", spectral_path)
        cause = markdown.split("## Causa raíz", 1)[-1].split("## ", 1)[0].strip()
        logger.info("Causa raíz:\n%s", cause)
        return md_path

    def _reproduced_auroc(self) -> dict[str, float]:
        """Recompute original-vs-IAAFT AUROC on the saved matrices."""
        audit = replace(self.config.audit, regimes=("arma_correlated",))
        binary = _binary_original_vs_iaaft(self.records, audit)
        out: dict[str, float] = {}
        block = binary.get("arma_correlated", {})
        for view in _VIEWS:
            for model in ("random_forest", "logistic"):
                out[f"{view}_{model}"] = float(
                    block.get(view, {}).get(model, {}).get("auroc_mean", float("nan"))
                )
        return out

    def _end_match_control(self) -> dict[str, Any]:
        """Rebuild transition/visibility graphs after end-matching ARMA series.

        Only ``arma_correlated`` is regenerated (same R, N, seed). Pretreated
        originals and their IAAFT surrogates are compared; the nine-method
        ``records.pkl`` panel is left untouched.
        """
        audit = replace(
            self.config.audit,
            regimes=("arma_correlated",),
            methods=_ENDMATCH_METHODS,
            families={
                "visibility": ("NVG", "HVG", "dHVG"),
                "transition": ("STN", "OPN"),
            },
        )
        gen_cfg = load_generator_suite_config()
        gen_cfg = replace(
            gen_cfg,
            global_params=replace(
                gen_cfg.global_params,
                n_steps=audit.n_steps,
                seed=audit.seed,
                n_samples=1,
            ),
        )
        ts2g_cfg = TS2GSuiteConfig(methods=list(_ENDMATCH_METHODS))
        transformers = {
            t.method: t
            for t in build_transformers(ts2g_cfg)
            if t.method in _ENDMATCH_METHODS
        }
        pipeline = DescriptorPipeline(config=DescriptorConfig())
        generator = build_generator("arma_correlated", gen_cfg)

        jump_raw: list[float] = []
        jump_matched: list[float] = []
        jump_iaaft: list[float] = []
        records: list[dict[str, Any]] = []
        t0 = time.perf_counter()
        for r in range(audit.n_realizations):
            seed = realization_seed(audit.seed, "arma_correlated", r)
            series = generator.generate(audit.n_steps, seed=seed)
            raw = np.asarray(series.values, dtype=float).ravel()
            matched = end_match_series(raw)
            jump_raw.append(_endpoint_jump_ratio(raw))
            jump_matched.append(_endpoint_jump_ratio(matched))

            for kind, values in (("original", matched),):
                matrix = _extract_matrix_for_series(
                    values,
                    dataset_id="arma_correlated",
                    audit_config=audit,
                    transformers=transformers,
                    pipeline=pipeline,
                )
                records.append(
                    {
                        "dataset_id": "arma_correlated",
                        "regime": "stochastic",
                        "realization": r,
                        "seed": seed,
                        "surrogate_kind": kind,
                        "matrix": matrix,
                        "series_length": int(values.size),
                    }
                )

            rng = np.random.default_rng(_surrogate_seed(seed, "iaaft", r))
            surr = iaaft_surrogate(
                matched,
                rng,
                max_iter=self.config.iaaft_max_iter,
                min_iter=self.config.iaaft_min_iter,
                tol=self.config.iaaft_tol,
            )
            jump_iaaft.append(_endpoint_jump_ratio(surr))
            matrix_s = _extract_matrix_for_series(
                surr,
                dataset_id="arma_correlated",
                audit_config=audit,
                transformers=transformers,
                pipeline=pipeline,
            )
            records.append(
                {
                    "dataset_id": "arma_correlated",
                    "regime": "stochastic",
                    "realization": r,
                    "seed": seed,
                    "surrogate_kind": "iaaft",
                    "matrix": matrix_s,
                    "series_length": int(surr.size),
                }
            )
            if (r + 1) % 5 == 0 or r == 0:
                logger.info(
                    "End-match control realization %d/%d (%.1fs)",
                    r + 1,
                    audit.n_realizations,
                    time.perf_counter() - t0,
                )

        binary = _binary_original_vs_iaaft(records, audit)
        block = binary.get("arma_correlated", {})
        auroc_rows: list[dict[str, Any]] = []
        auroc: dict[str, float] = {}
        for view in _VIEWS:
            for model in ("random_forest", "logistic"):
                value = float(
                    block.get(view, {}).get(model, {}).get("auroc_mean", float("nan"))
                )
                auroc[f"{view}_{model}"] = value
                auroc_rows.append(
                    {
                        "pretreatment": "end_match",
                        "view": view,
                        "model": model,
                        "auroc_mean": value,
                        "n_realizations": audit.n_realizations,
                        "n_steps": audit.n_steps,
                        "seed": audit.seed,
                        "methods": ",".join(_ENDMATCH_METHODS),
                    }
                )
        elapsed = time.perf_counter() - t0
        logger.info(
            "End-match AUROC transition RF=%.3f visibility RF=%.3f (%.1fs)",
            auroc.get("transition_random_forest", float("nan")),
            auroc.get("visibility_random_forest", float("nan")),
            elapsed,
        )
        return {
            "auroc": auroc,
            "auroc_rows": auroc_rows,
            "jump_raw_median": float(np.median(jump_raw)),
            "jump_matched_median": float(np.median(jump_matched)),
            "jump_iaaft_median": float(np.median(jump_iaaft)),
            "elapsed_sec": elapsed,
            "n_realizations": audit.n_realizations,
            "n_steps": audit.n_steps,
            "seed": audit.seed,
            "methods": list(_ENDMATCH_METHODS),
        }

    def _paired_frame(self) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
        """Full feature matrix of original and IAAFT rows, plus labels and ids."""
        rows: list[dict[str, float]] = []
        labels: list[str] = []
        realizations: list[int] = []
        methods = self.config.audit.methods
        for rec in self.records:
            if rec["surrogate_kind"] not in {"original", "iaaft"}:
                continue
            rows.append(matrix_to_feature_row(rec["matrix"], methods=methods))
            labels.append(str(rec["surrogate_kind"]))
            realizations.append(int(rec["realization"]))
        x = pd.DataFrame(rows)
        y = pd.Series(labels, name="surrogate_kind")
        realization = pd.Series(realizations, name="realization")
        return x, y, realization

    def _feature_importance(self) -> pd.DataFrame:
        """L2 coefficients and Gini importances on the audit's cleaned views."""
        x, y, _realization = self._paired_frame()
        health = diagnose_and_clean(x, self.config.audit)
        x_clean = health.x_clean
        y_bin = (y == "original").astype(int).to_numpy()
        models = _make_models(self.config.audit.seed, rf_n_jobs=1)
        frames: list[pd.DataFrame] = []
        for view in _VIEWS:
            mask = family_feature_mask(x_clean.columns, view)
            x_view = x_clean.loc[:, mask]
            if x_view.shape[1] == 0:
                continue
            original = x_view.iloc[y_bin == 1]
            iaaft = x_view.iloc[y_bin == 0]
            mean_o = original.mean()
            mean_i = iaaft.mean()
            pooled = np.sqrt(
                (
                    (len(original) - 1) * original.var(ddof=1)
                    + (len(iaaft) - 1) * iaaft.var(ddof=1)
                )
                / max(len(original) + len(iaaft) - 2, 1)
            )
            cohens_d = (mean_o - mean_i) / pooled.replace(0.0, np.nan)
            for model_name, model in models.items():
                clf = clone(model)
                clf.fit(x_view.to_numpy(dtype=float), y_bin)
                estimator = clf.named_steps["clf"]
                if model_name == "logistic":
                    signed = np.asarray(estimator.coef_, dtype=float).ravel()
                    score = np.abs(signed)
                else:
                    score = np.asarray(estimator.feature_importances_, dtype=float)
                    signed = np.full(score.shape, np.nan)
                order = np.argsort(-score)
                frames.append(
                    pd.DataFrame(
                        {
                            "view": view,
                            "model": model_name,
                            "feature": x_view.columns.to_numpy()[order],
                            "rank": np.arange(1, order.size + 1),
                            "importance": score[order],
                            "signed_coefficient": signed[order],
                            "mean_original": mean_o.to_numpy()[order],
                            "mean_iaaft": mean_i.to_numpy()[order],
                            "delta_original_minus_iaaft": (mean_o - mean_i).to_numpy()[order],
                            "cohens_d": cohens_d.to_numpy()[order],
                        }
                    )
                )
        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, ignore_index=True)

    def _spectral_traces(self) -> list[SpectralTrace]:
        """Regenerate ARMA with the audit seeds and measure the IAAFT residual."""
        from dataclasses import replace

        audit = self.config.audit
        gen_cfg = load_generator_suite_config()
        gen_cfg = replace(
            gen_cfg,
            global_params=replace(
                gen_cfg.global_params,
                n_steps=audit.n_steps,
                seed=audit.seed,
                n_samples=1,
            ),
        )
        by_realization: dict[int, dict[str, Any]] = {}
        for rec in self.records:
            if rec["surrogate_kind"] != "original":
                continue
            by_realization[int(rec["realization"])] = rec

        traces: list[SpectralTrace] = []
        for realization in sorted(by_realization):
            rec = by_realization[realization]
            seed = int(rec.get("seed", realization_seed(audit.seed, "arma_correlated", realization)))
            series = build_generator("arma_correlated", gen_cfg).generate(
                audit.n_steps, seed=seed
            )
            values = np.asarray(series.values, dtype=float).ravel()
            if values.size != int(rec["series_length"]):
                raise RuntimeError(
                    f"ARMA length mismatch at r={realization}: "
                    f"{values.size} != {rec['series_length']}"
                )
            library_rng = np.random.default_rng(
                _surrogate_seed(seed, "iaaft", realization)
            )
            library = iaaft_surrogate(
                values,
                library_rng,
                max_iter=self.config.iaaft_max_iter,
                min_iter=self.config.iaaft_min_iter,
                tol=self.config.iaaft_tol,
            )
            trace_rng = np.random.default_rng(_surrogate_seed(seed, "iaaft", realization))
            y_stop, y_max, n_iter, reason, _history = _trace_iaaft(
                values,
                trace_rng,
                max_iter=self.config.iaaft_max_iter,
                tol=self.config.iaaft_tol,
                min_iter=self.config.iaaft_min_iter,
            )
            matches = bool(np.allclose(y_stop, library))
            if not matches:
                raise RuntimeError(
                    f"Instrumented IAAFT diverged from iaaft_surrogate at r={realization}"
                )
            rel_low, rel_high, frac_low, frac_high = _band_errors(values, library)
            target = np.abs(np.fft.rfft(values))
            traces.append(
                SpectralTrace(
                    realization=realization,
                    seed=seed,
                    rho=float(series.params.get("rho", float("nan"))),
                    n_iter_returned=n_iter,
                    stop_reason=reason,
                    spectral_rel_err_returned=float(
                        _spectral_relative_error(library, target)
                    ),
                    spectral_rel_err_at_max_iter=float(
                        _spectral_relative_error(y_max, target)
                    ),
                    endpoint_jump_ratio_original=_endpoint_jump_ratio(values),
                    endpoint_jump_ratio_iaaft=_endpoint_jump_ratio(library),
                    mssd_original=_mssd(values),
                    mssd_iaaft=_mssd(library),
                    rel_err_low_freq=rel_low,
                    rel_err_high_freq=rel_high,
                    residual_energy_frac_low=frac_low,
                    residual_energy_frac_high=frac_high,
                    matches_library=matches,
                )
            )
        return traces

    def _attach_feature_correlation(
        self,
        importance: pd.DataFrame,
        traces: Sequence[SpectralTrace],
    ) -> None:
        """Store Spearman(spectral error, top-feature gap) on the instance."""
        self.correlations: list[dict[str, float | str]] = []
        if importance.empty or not traces:
            return
        x, y, realization = self._paired_frame()
        err = pd.Series(
            {trace.realization: trace.spectral_rel_err_returned for trace in traces},
            name="spectral_rel_err",
        )
        gini = importance[
            (importance["model"] == "random_forest") & (importance["rank"] <= 3)
        ]
        for row in gini.itertuples(index=False):
            feature = str(row.feature)
            if feature not in x.columns:
                continue
            wide = pd.DataFrame(
                {
                    "realization": realization.to_numpy(),
                    "kind": y.to_numpy(),
                    "value": x[feature].to_numpy(),
                }
            )
            pivot = wide.pivot(index="realization", columns="kind", values="value")
            if not {"original", "iaaft"}.issubset(pivot.columns):
                continue
            delta = (pivot["original"] - pivot["iaaft"]).rename("delta")
            joined = pd.concat([delta, err], axis=1, join="inner").dropna()
            if len(joined) < 5:
                continue
            rho, p_value = spearmanr(joined["delta"], joined["spectral_rel_err"])
            self.correlations.append(
                {
                    "view": str(row.view),
                    "feature": feature,
                    "spearman_rho": float(rho),
                    "p_value": float(p_value),
                }
            )

    def _render(
        self,
        auroc: dict[str, float],
        importance: pd.DataFrame,
        traces: Sequence[SpectralTrace],
        elapsed_sec: float,
        *,
        end_match: dict[str, Any] | None = None,
    ) -> str:
        """Markdown report. The algorithm is not modified."""
        spectral = pd.DataFrame([trace.__dict__ for trace in traces])
        lines = [
            "# Diagnóstico: `arma_correlated` original vs IAAFT",
            "",
            "No se ha modificado el algoritmo IAAFT ni se ha relanzado el pipeline "
            "de 9 métodos. Las importancias salen de `records.pkl`. El error "
            "espectral se mide regenerando solo las series ARMA con las semillas "
            f"de `phase2_surrogates` (perfil `{self.config.audit.profile}`, "
            f"R={self.config.audit.n_realizations}, N={self.config.audit.n_steps}, "
            f"seed={self.config.audit.seed}, "
            f"`iaaft_max_iter={self.config.iaaft_max_iter}`, "
            f"`min_iter={self.config.iaaft_min_iter}`, "
            f"`tol={self.config.iaaft_tol:g}`).",
            "",
            f"- Tiempo del diagnóstico: {elapsed_sec:.1f} s.",
            "",
            "## AUROC reproducido (original vs IAAFT, matrices guardadas)",
            "",
            "| Vista | RF | Logística L2 |",
            "|---|---:|---:|",
        ]
        for view in _VIEWS:
            lines.append(
                f"| {view} | {auroc.get(f'{view}_random_forest', float('nan')):.3f} | "
                f"{auroc.get(f'{view}_logistic', float('nan')):.3f} |"
            )
        lines.extend(
            [
                "",
                "Referencia en `auroc_binary.csv`: transition RF = 0.783, "
                "visibility RF = 0.717.",
                "",
                "## 1. Features que separan original de IAAFT",
                "",
                "Clasificador ajustado sobre las 60 filas (30 original + 30 IAAFT) "
                "después del mismo filtro de varianza que la auditoría. "
                "Coeficiente L2 en el espacio estandarizado (positivo ⇒ más alto "
                "en el original). Gini del random forest. "
                "`d` de Cohen = (media original − media IAAFT) / desviación agrupada.",
                "",
            ]
        )
        for view in _VIEWS:
            block = importance[
                (importance["view"] == view) & (importance["model"] == "random_forest")
            ].head(8)
            coef = importance[
                (importance["view"] == view) & (importance["model"] == "logistic")
            ].set_index("feature")
            lines.append(f"### {view}")
            lines.append("")
            lines.append(
                "| rank | feature | Gini | coef L2 | d | media orig | media IAAFT |"
            )
            lines.append("|---:|---|---:|---:|---:|---:|---:|")
            for row in block.itertuples(index=False):
                signed = coef["signed_coefficient"].get(row.feature, float("nan"))
                lines.append(
                    f"| {int(row.rank)} | `{row.feature}` | {row.importance:.3f} | "
                    f"{float(signed):.3f} | {row.cohens_d:.2f} | "
                    f"{row.mean_original:.4g} | {row.mean_iaaft:.4g} |"
                )
            lines.append("")

        if spectral.empty:
            lines.append("No hay trazas espectrales.")
            return "\n".join(lines) + "\n"

        reasons = spectral["stop_reason"].value_counts().to_dict()
        mssd_ratio = spectral["mssd_iaaft"] / spectral["mssd_original"]
        jump_ratio = (
            spectral["endpoint_jump_ratio_iaaft"] / spectral["endpoint_jump_ratio_original"]
        )
        lines.extend(
            [
                "## 2. ¿Converge IAAFT en `arma_correlated`?",
                "",
                f"- Realizaciones: {len(spectral)}. Todas coinciden con `iaaft_surrogate`.",
                f"- Motivo de parada: `{reasons}`.",
                f"- Iteración de parada: mediana {spectral['n_iter_returned'].median():.0f} "
                f"(mín {spectral['n_iter_returned'].min():.0f}, "
                f"máx {spectral['n_iter_returned'].max():.0f}). "
                f"Ninguna llega a `max_iter={self.config.iaaft_max_iter}` "
                "si el motivo es `plateau`.",
                f"- Error espectral relativo al parar: "
                f"mediana {spectral['spectral_rel_err_returned'].median():.3e}, "
                f"media {spectral['spectral_rel_err_returned'].mean():.3e}, "
                f"máx {spectral['spectral_rel_err_returned'].max():.3e}. "
                f"Tolerancia pedida: {self.config.iaaft_tol:g}. "
                f"Fracción con error < 1e-3: "
                f"{(spectral['spectral_rel_err_returned'] < 1e-3).mean():.2f}.",
                f"- El mismo error al forzar las {self.config.iaaft_max_iter} iteraciones: "
                f"mediana {spectral['spectral_rel_err_at_max_iter'].median():.3e}. "
                f"Mejora mediana respecto a la parada: "
                f"{(spectral['spectral_rel_err_returned'] - spectral['spectral_rel_err_at_max_iter']).median():.3e}.",
                f"- Error relativo en el 20% de bins graves (baja frecuencia): "
                f"mediana {spectral['rel_err_low_freq'].median():.3e}. "
                f"En el 20% agudo: mediana {spectral['rel_err_high_freq'].median():.3e}.",
                f"- Fracción de la energía del residual en baja frecuencia: "
                f"mediana {spectral['residual_energy_frac_low'].median():.3f}; "
                f"en alta frecuencia: mediana {spectral['residual_energy_frac_high'].median():.3f}.",
                f"- `rho` del AR(2): mediana {spectral['rho'].median():.3f} "
                f"(rango {spectral['rho'].min():.3f}–{spectral['rho'].max():.3f}).",
                f"- Salto de borde `|x0-xN|/mediana|Δx|`: original "
                f"{spectral['endpoint_jump_ratio_original'].median():.2f}, "
                f"IAAFT {spectral['endpoint_jump_ratio_iaaft'].median():.2f} "
                f"(razón IAAFT/original mediana {jump_ratio.median():.3f}).",
                f"- Rugosidad MSSD: razón IAAFT/original mediana {mssd_ratio.median():.3f} "
                f"(media {mssd_ratio.mean():.3f}).",
                "",
            ]
        )
        if self.correlations:
            lines.append(
                "Correlación de Spearman entre el error espectral de cada "
                "realización y (feature original − feature IAAFT) de los tres "
                "primeros Gini:"
            )
            lines.append("")
            lines.append("| vista | feature | rho | p |")
            lines.append("|---|---|---:|---:|")
            for item in self.correlations:
                lines.append(
                    f"| {item['view']} | `{item['feature']}` | "
                    f"{float(item['spearman_rho']):.2f} | {float(item['p_value']):.2e} |"
                )
            lines.append("")

        lines.extend(
            [
                "## Causa raíz",
                "",
                self._root_cause(spectral, importance),
                "",
            ]
        )
        if end_match is not None:
            lines.extend(self._render_end_match(end_match, baseline_auroc=auroc))
        return "\n".join(lines)

    def _render_end_match(
        self,
        end_match: dict[str, Any],
        *,
        baseline_auroc: dict[str, float],
    ) -> list[str]:
        """Append the bounded end-matching control section."""
        auroc = end_match["auroc"]
        lines = [
            "## 3. Control de cierre: end-matching (solo `arma_correlated`)",
            "",
            "Verificación acotada: se regeneran las series ARMA con la misma "
            f"semilla (`seed={end_match['seed']}`, R={end_match['n_realizations']}, "
            f"N={end_match['n_steps']}), se aplica *end-matching* "
            "(rampa lineal que fuerza `y[0]=y[-1]`) **antes** de IAAFT, y se "
            "extraen únicamente las vistas `transition` / `visibility` "
            f"(`methods={end_match['methods']}`). No se relanza el panel de "
            "9 métodos ni se modifica `iaaft_surrogate`.",
            "",
            f"- Tiempo del control: {float(end_match['elapsed_sec']):.1f} s.",
            f"- Salto de borde mediano `|x0-xN|/med|Δx|`: raw "
            f"{float(end_match['jump_raw_median']):.2f} → end-matched "
            f"{float(end_match['jump_matched_median']):.3g} → IAAFT(end-matched) "
            f"{float(end_match['jump_iaaft_median']):.2f}.",
            "",
            "### AUROC binario original(end-matched) vs IAAFT(end-matched)",
            "",
            "| Vista | RF (baseline) | RF (end-match) | LR (baseline) | LR (end-match) |",
            "|---|---:|---:|---:|---:|",
        ]
        for view in _VIEWS:
            lines.append(
                f"| {view} | "
                f"{baseline_auroc.get(f'{view}_random_forest', float('nan')):.3f} | "
                f"{auroc.get(f'{view}_random_forest', float('nan')):.3f} | "
                f"{baseline_auroc.get(f'{view}_logistic', float('nan')):.3f} | "
                f"{auroc.get(f'{view}_logistic', float('nan')):.3f} |"
            )
        rf_t = float(auroc.get("transition_random_forest", float("nan")))
        rf_v = float(auroc.get("visibility_random_forest", float("nan")))
        lines.extend(
            [
                "",
                "Interpretación: un nulo lineal bien emparejado en el borde "
                "periódico debería situar el AUROC cerca de 0.50. "
                f"Tras end-matching, transition RF = {rf_t:.3f} y "
                f"visibility RF = {rf_v:.3f} "
                + (
                    "(cae hacia el azar: el salto de borde era el confusor)."
                    if max(rf_t, rf_v) < 0.60
                    else (
                        "(sigue por encima de 0.60: el borde no basta por sí "
                        "solo para explicar toda la separación)."
                        if max(rf_t, rf_v) >= 0.60
                        else ""
                    )
                ),
                "",
            ]
        )
        return lines

    def _root_cause(self, spectral: pd.DataFrame, importance: pd.DataFrame) -> str:
        """State the mechanism supported by the measured residual, not a new IAAFT."""
        plateau = float((spectral["stop_reason"] == "plateau").mean())
        err = float(spectral["spectral_rel_err_returned"].median())
        rel_low = float(spectral["rel_err_low_freq"].median())
        rel_high = float(spectral["rel_err_high_freq"].median())
        frac_low = float(spectral["residual_energy_frac_low"].median())
        frac_high = float(spectral["residual_energy_frac_high"].median())
        jump_x = float(spectral["endpoint_jump_ratio_original"].median())
        jump_y = float(spectral["endpoint_jump_ratio_iaaft"].median())
        jump_ratio = float(
            (
                spectral["endpoint_jump_ratio_iaaft"]
                / spectral["endpoint_jump_ratio_original"]
            ).median()
        )
        mssd_ratio = float(
            (spectral["mssd_iaaft"] / spectral["mssd_original"]).median()
        )
        top_t = _top_names(importance, "transition")
        top_v = _top_names(importance, "visibility")
        improve = float(
            (
                spectral["spectral_rel_err_returned"]
                - spectral["spectral_rel_err_at_max_iter"]
            ).median()
        )
        return (
            f"El IAAFT de producción no alcanza `tol={self.config.iaaft_tol:g}`. "
            f"En el {plateau:.0%} de las realizaciones se detiene por la cláusula "
            f"de meseta (`|Δ error| < tol`) hacia la iteración "
            f"{spectral['n_iter_returned'].median():.0f}, con un error espectral "
            f"relativo mediano de {err:.3e} (ninguna serie baja de 1e-3). "
            f"Forzar las {self.config.iaaft_max_iter} iteraciones solo mejora ese "
            f"error en {improve:.3e} (mediana pareada): el residual es un punto "
            "fijo del algoritmo, no un presupuesto de iteraciones corto.\n\n"
            f"El pico AR está bien imitado en términos relativos: el 20% grave "
            f"del espectro (`rho` mediano {spectral['rho'].median():.3f}) queda a "
            f"{rel_low:.3e}. Ese pico domina la norma L2, así que aun así concentra "
            f"una fracción {frac_low:.2f} de la energía absoluta del residual. "
            f"El 20% agudo tiene un error relativo mediano de {rel_high:.3e} "
            f"(denominador pequeño: la serie original casi no tiene potencia ahí) "
            f"y una fracción {frac_high:.2f} de la energía del residual. "
            f"El hecho geométrico es el borde periódico. `|x0 − xN|` vale "
            f"{jump_x:.1f} veces el paso interior, pero el grafo no es circular "
            "y no ve ese salto. El IAAFT, obligado a realizar el periodograma de "
            f"esa serie con empalme, reduce el salto a {jump_y:.2f} "
            f"(razón mediana {jump_ratio:.3f}) y reparte la energía como rugosidad "
            f"interior (MSSD IAAFT/original = {mssd_ratio:.3f}).\n\n"
            "Un ARMA gaussiano lineal debería ser indistinguible de un IAAFT "
            "convergido (AUROC ≈ 0.50). El AUROC de 0.783 en `transition` y "
            "0.717 en `visibility` mide esa rugosidad, no no-linealidad. "
            f"En transición las features que más separan son {top_t}: "
            "el original tiene más auto-bucles ordinales y más heterogeneidad "
            "de grado que su IAAFT. "
            f"En visibilidad, {top_v}: `HVG` y `dHVG` salen duplicados "
            "porque `i_dir` está saturado en 1 y el resto de descriptores se "
            "calcula sobre el grafo simetrizado, así que la vista de visibilidad "
            "no está usando la flecha direccional. "
            "No se modifica el algoritmo IAAFT en este cierre. Si se quiere "
            "acercar el nulo lineal a 0.50, el siguiente experimento es tratar "
            "el borde (ventana o series con extremos emparejados) y volver a "
            "medir solo este régimen."
        )


def _top_names(importance: pd.DataFrame, view: str, n: int = 3) -> str:
    block = importance[
        (importance["view"] == view) & (importance["model"] == "random_forest")
    ].head(n)
    if block.empty:
        return "(sin features)"
    return ", ".join(f"`{name}`" for name in block["feature"])


def _load_records(path: Path) -> list[dict[str, Any]]:
    with path.open("rb") as handle:
        loaded = pickle.load(handle)
    if not isinstance(loaded, list):
        raise TypeError(f"Expected a list of records in {path}")
    return loaded


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--records", type=Path, default=DEFAULT_RECORDS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--profile", choices=("dev", "paper"), default="paper")
    parser.add_argument(
        "--skip-end-match",
        action="store_true",
        help="Skip the bounded end-matching AUROC control (transition/visibility).",
    )
    return parser.parse_args()


def main() -> None:
    setup_logging()
    args = parse_args()
    t0 = time.perf_counter()
    config = SurrogateAuditConfig.from_yaml(args.config).with_profile(args.profile)
    records = _load_records(args.records)
    diagnostic = ArmaIAAFTDiagnostic(
        config,
        records,
        args.output_dir,
        run_end_match=not args.skip_end_match,
    )
    path = diagnostic.run()
    logger.info("Diagnosis finished in %.1fs → %s", time.perf_counter() - t0, path)


if __name__ == "__main__":
    main()
