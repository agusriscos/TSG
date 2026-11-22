"""Publication-facing plot strings (English / Spanish)."""

from __future__ import annotations

from typing import Mapping

Lang = str

_STRINGS: dict[str, dict[str, str]] = {
    "en": {
        "pca_panel_a": "A) PCA projection (~95% ellipses)",
        "pca_panel_b1": "B1) Top-8 loadings on PC1 ({pc1_var:.1f}% var.)",
        "pca_panel_b2": "B2) Top-8 loadings on PC2 ({pc2_var:.1f}% var.)",
        "pca_family_legend": "family",
        "pca_suptitle": "PCA: class geometry (A) and descriptors driving each axis (B)",
        "confusion_suptitle": "Normalized confusion matrices (%) — family ablation (5 views)",
        "fingerprint_title": "Mean standardized regime fingerprint (z-score)",
        "ablation_title": "Family ablation",
        "ablation_f1_label": "Macro-F1 (RF)",
        "ablation_silhouette": "Global silhouette",
        "ablation_per_regime": "Per-regime F1 (full matrix · RF)",
        "ablation_threshold": "threshold 0.90",
        "critical_pairs_suptitle": (
            "Critical pairs on degree/cohesion descriptors (mean NVG/HVG/RN/kNN)"
        ),
        "importance_no_data": "No importances",
        "importance_family": "family",
        "importance_suptitle": (
            "Top-15 descriptors (full view) — Gini + L2 norm\n"
            "(Permutation importance omitted: Macro-F1=1 + collinearity → Δscore≈0)"
        ),
        "surrogate_summary_suptitle": "Surrogate audit — summary by TS2G family",
        "surrogate_delta_ylabel": r"$\Delta$ Macro-F1",
        "surrogate_multi_title": "Surrogate-type multiclass Macro-F1  [RF]",
        "surrogate_auroc_title": (
            "Binary AUROC original vs IAAFT (RF) by regime × family"
        ),
        "invariance_matrix_title": "Invariance matrix (descriptor stability / OOD F1)",
        "phase_maps_families_title": r"Phase maps: Macro-F1 RF over $N \times$ SNR",
        "phase_topo_vs_spec_title": (
            r"Topological vs spectral failure modes vs SNR ($N$ fixed)"
        ),
        "pid_title": "Partial information decomposition (MMI atoms)",
        "shapley_title": "Exact Shapley importance by family",
        "pareto_title": r"Cost–accuracy Pareto frontier (time vs Macro-F1)",
    },
    "es": {
        "pca_panel_a": "A) Proyección PCA (elipses ~95%)",
        "pca_panel_b1": "B1) Top-8 loadings en PC1 ({pc1_var:.1f}% var.)",
        "pca_panel_b2": "B2) Top-8 loadings en PC2 ({pc2_var:.1f}% var.)",
        "pca_family_legend": "familia",
        "pca_suptitle": "PCA: geometría de clases (A) y descriptores que empujan cada eje (B)",
        "confusion_suptitle": (
            "Matrices de confusión normalizadas (%) — ablación por familia (5 vistas)"
        ),
        "fingerprint_title": "Huella media estandarizada por régimen (z-score)",
        "ablation_title": "Ablación por familia",
        "ablation_f1_label": "Macro-F1 (RF)",
        "ablation_silhouette": "Silhouette global",
        "ablation_per_regime": "F1 por régimen (matriz completa · RF)",
        "ablation_threshold": "umbral 0.90",
        "critical_pairs_suptitle": (
            "Pares críticos en descriptores de grado/cohesión (media NVG/HVG/RN/kNN)"
        ),
        "importance_no_data": "Sin importancias",
        "importance_family": "familia",
        "importance_suptitle": (
            "Top-15 descriptores (vista full) — Gini + norma L2\n"
            "(Permutation Importance omitida: Macro-F1=1 + colinealidad → Δscore≈0)"
        ),
        "surrogate_summary_suptitle": "Auditoría de subrogados — resumen por familia TS2G",
        "surrogate_delta_ylabel": "Δ Macro-F1",
        "surrogate_multi_title": "Macro-F1 multiclase surrogate-type  [RF]",
        "surrogate_auroc_title": (
            "AUROC binario original vs IAAFT (RF) por régimen × familia"
        ),
        "invariance_matrix_title": "Matriz de invarianzas (estabilidad / F1 OOD)",
        "phase_maps_families_title": r"Mapas de fase: Macro-F1 RF en $N \times$ SNR",
        "phase_topo_vs_spec_title": (
            r"Fallo topológico vs espectral frente a SNR ($N$ fijo)"
        ),
        "pid_title": "Descomposición de información parcial (átomos MMI)",
        "shapley_title": "Importancia Shapley exacta por familia",
        "pareto_title": r"Frontera de Pareto coste–exactitud (tiempo vs Macro-F1)",
    },
}


def t(key: str, lang: Lang = "en", **kwargs: object) -> str:
    """Translate ``key`` for ``lang``, with optional ``str.format`` kwargs."""
    table = _STRINGS.get(lang, _STRINGS["en"])
    template = table.get(key) or _STRINGS["en"].get(key, key)
    if kwargs:
        return template.format(**kwargs)
    return template


def available_keys() -> Mapping[str, Mapping[str, str]]:
    """Return the full translation table (read-only intent)."""
    return _STRINGS
