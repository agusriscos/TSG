"""Automatic Go/No-Go traffic-light verdict for Phase-1 audit."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

from tsg.audit.classification import ClassificationResult
from tsg.audit.config import AuditConfig


class Verdict(str, Enum):
    GREEN = "GREEN"
    YELLOW = "YELLOW"
    RED = "RED"


@dataclass
class VerdictResult:
    verdict: Verdict
    reasons: list[str]
    recommendations: list[str]
    metrics_snapshot: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "verdict": self.verdict.value,
            "reasons": self.reasons,
            "recommendations": self.recommendations,
            "metrics_snapshot": self.metrics_snapshot,
        }


def decide_verdict(
    classification: ClassificationResult,
    config: AuditConfig,
) -> VerdictResult:
    """Apply Table semáforo rules on full-matrix RF metrics."""
    full_rf = classification.full_rf()
    full_lin = classification.full_linear()
    if full_rf is None:
        return VerdictResult(
            verdict=Verdict.RED,
            reasons=["No hay métricas RandomForest para la matriz completa."],
            recommendations=["Re-ejecutar la auditoría con la vista full."],
            metrics_snapshot={},
        )

    macro = full_rf.macro_f1_mean
    f1_by = full_rf.f1_by_class
    critical = classification.critical_pair_confusion_rate
    min_f1 = min(f1_by.values()) if f1_by else 0.0
    max_crit = max(critical.values()) if critical else 0.0

    linear_gap = 0.0
    if full_lin is not None:
        linear_gap = macro - full_lin.macro_f1_mean

    snapshot = {
        "macro_f1_rf": macro,
        "macro_f1_linear": None if full_lin is None else full_lin.macro_f1_mean,
        "min_f1_by_regime": min_f1,
        "f1_by_regime": f1_by,
        "critical_pair_confusion_rate": critical,
        "linear_nonlinear_gap": linear_gap,
    }

    reasons: list[str] = []
    recommendations: list[str] = []

    # RED conditions
    red_pairs = [k for k, v in critical.items() if v >= 0.10]
    red_regimes = [k for k, v in f1_by.items() if v < 0.85]
    if red_pairs or red_regimes:
        if red_pairs:
            reasons.append(
                "Confusión ≥ 10% en pares críticos: " + ", ".join(red_pairs)
            )
        if red_regimes:
            reasons.append(
                "F1 < 0.85 en regímenes: " + ", ".join(f"{k}={f1_by[k]:.3f}" for k in red_regimes)
            )
        recommendations.extend(
            [
                "Enriquecer D en Fase 2 con invariantes de orden superior.",
                "Valorar motivos HVG m=4–5, asimetría dirigida, plano Fisher–Shannon,",
                "entropía de transición y/o más estructura OPTN/STN.",
            ]
        )
        return VerdictResult(
            verdict=Verdict.RED,
            reasons=reasons,
            recommendations=recommendations,
            metrics_snapshot=snapshot,
        )

    # GREEN conditions
    green_crit_ok = all(v < 0.05 for v in critical.values()) if critical else True
    green_f1_ok = all(v >= 0.90 for v in f1_by.values()) if f1_by else False
    if macro >= 0.95 and green_f1_ok and green_crit_ok:
        reasons.append(
            f"Macro-F1={macro:.3f} ≥ 0.95, F1≥0.90 en todos los regímenes "
            "y confusión <5% en pares críticos."
        )
        return VerdictResult(
            verdict=Verdict.GREEN,
            reasons=reasons,
            recommendations=[
                "Matriz D discriminativa: alcance de investigación cubierto hasta la auditoría."
            ],
            metrics_snapshot=snapshot,
        )

    # YELLOW otherwise if still reasonably separable
    if macro >= 0.90:
        fragile = [k for k, v in critical.items() if v >= 0.05]
        if fragile:
            reasons.append(
                "Suficiencia frágil: confusión ≥5% en " + ", ".join(fragile)
            )
        if linear_gap > 0.05:
            reasons.append(
                f"Gap lineal vs no lineal = {linear_gap:.3f} "
                "(separabilidad frágil: revisar interacciones entre descriptores)."
            )
        if not reasons:
            reasons.append(
                f"Macro-F1={macro:.3f} ≥ 0.90 pero no se cumplen todos los umbrales verdes."
            )
        recommendations.append(
            "Revisar pares frágiles y valorar enriquecer la matriz D."
        )
        return VerdictResult(
            verdict=Verdict.YELLOW,
            reasons=reasons,
            recommendations=recommendations,
            metrics_snapshot=snapshot,
        )

    reasons.append(f"Macro-F1={macro:.3f} < 0.90: discriminabilidad insuficiente.")
    recommendations.append("Enriquecer obligatoriamente la matriz D antes de ampliar el alcance.")
    return VerdictResult(
        verdict=Verdict.RED,
        reasons=reasons,
        recommendations=recommendations,
        metrics_snapshot=snapshot,
    )
