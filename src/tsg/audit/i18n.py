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
        "critical_pairs_family_suptitle": (
            "Critical pairs by TS2G family — confusion (A) vs geometric separability (B)"
        ),
        "critical_pairs_family_panel_a": (
            "A) Confusion rate (%) — logistic (RF in parentheses if > 0)"
        ),
        "critical_pairs_family_panel_b": (
            "B) Pair silhouette (Fisher ratio annotated)"
        ),
        "critical_pairs_family_xlabel": "TS2G family view",
        "critical_pairs_family_ylabel": "Critical pair",
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
        "surrogate_dhvg_suptitle": (
            "dHVG temporal irreversibility — original vs surrogates"
        ),
        "surrogate_dhvg_kl_title": (
            r"Smoothed $\mathrm{KL}(P(k^{\mathrm{in}})\,\|\,P(k^{\mathrm{out}}))$"
        ),
        "surrogate_dhvg_idir_title": r"Directional asymmetry $i_{\mathrm{dir}}$",
        "surrogate_dhvg_ylabel_kl": (
            r"$\mathrm{KL}(P(k^{\mathrm{in}})\,\|\,P(k^{\mathrm{out}}))$"
        ),
        "surrogate_dhvg_ylabel_idir": r"$i_{\mathrm{dir}}$",
        "surrogate_dhvg_xlabel": "regime",
        "surrogate_dhvg_idir_saturated": (
            r"$i_{\mathrm{dir}}=1$ on every dHVG: all edges follow the time arrow, "
            "so Frobenius asymmetry is saturated."
        ),
        "invariance_matrix_title": "Invariance matrix (descriptor stability / OOD F1)",
        "phase_maps_families_title": r"Phase maps: Macro-F1 RF over $N \times$ SNR",
        "phase_topo_vs_spec_title": (
            r"Topological vs spectral failure modes vs SNR ($N$ fixed)"
        ),
        "phase_disconnection_by_family_title": (
            r"Algebraic disconnection by family "
            r"($\lambda_2(G)\leq 10^{-12}$ or non-finite)"
        ),
        "phase_disconnection_rate_label": "Disconnection rate",
        "pid_title": "Partial information decomposition (MMI atoms)",
        "pid_ylabel": "Information (bits)",
        "pid_xlabel": "Family pair (1st | 2nd)",
        "pid_redundancy": "Redundancy",
        "pid_unique_i": "Unique (1st family)",
        "pid_unique_j": "Unique (2nd family)",
        "pid_synergy": "Synergy",
        "pid_synergy_negative": "Negative synergy",
        "shapley_title": "Exact Shapley importance by family",
        "shapley_ylabel": "Shapley value (Macro-F1)",
        "shapley_xlabel": "TS2G family",
        "model_random_forest": "Random Forest",
        "model_logistic": "Logistic L2",
        "pareto_title": r"Cost–accuracy Pareto frontier (time vs Macro-F1)",
        "pareto_xlabel": "Time per series (ms, N={n_steps})",
        "pareto_n_families_one": "{size} family",
        "pareto_n_families_many": "{size} families",
        "pareto_front": "Pareto front",
        "pareto_front_legend": "Pareto front",
        "invariance_error_title": "Normalized descriptor error (median)",
        "invariance_delta_title": (
            r"$\Delta$ Macro-F1 RF OOD (perturbed − clean)"
        ),
        "invariance_method_error_title": "A) Normalized descriptor error (median)",
        "invariance_method_delta_title": (
            r"B) $\Delta$ Macro-F1 RF OOD (perturbed − clean)"
        ),
        "invariance_matrix_by_method_title": (
            "Invariance matrix by method (descriptor stability / OOD F1)"
        ),
        "rq1_pid_title": (
            r"Pairwise PID ($I_{\mathrm{broja}}$) by dynamic scope"
        ),
        "rq1_scope_global": "global",
        "rq1_omega_suptitle": (
            r"O-information $\Omega$ with bootstrap 95% CI"
        ),
        "rq1_omega_groups_title": "Macro-groups",
        "rq1_omega_regimes_title": "Regimes",
        "rq1_omega_xlabel": r"$\Omega$ (bits)",
        "rq1_omega_nonsig": r"CI contains 0",
        "rq1_sii_title": "Pairwise Shapley Interaction Index (SII)",
        "rq1_sii_ylabel": "SII (Macro-F1)",
        "rq1_sii_baseline_title": r"Baseline $v(\emptyset)$ vs full set",
        "rq1_shapley_phi_title": r"Shapley $\phi_i$ with 95% CI",
        "rq1_shapley_phi_ylabel": r"Shapley $\phi$ (Macro-F1)",
        "rq1_one_vs_rest_title": r"PID one-vs-rest ($X_i \mid X_{-i}$)",
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
        "critical_pairs_family_suptitle": (
            "Pares críticos por familia TS2G — confusión (A) vs separabilidad geométrica (B)"
        ),
        "critical_pairs_family_panel_a": (
            "A) Tasa de confusión (%) — logística (RF entre paréntesis si > 0)"
        ),
        "critical_pairs_family_panel_b": (
            "B) Silhouette del par (Fisher ratio anotado)"
        ),
        "critical_pairs_family_xlabel": "Vista por familia TS2G",
        "critical_pairs_family_ylabel": "Par crítico",
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
        "surrogate_dhvg_suptitle": (
            "Irreversibilidad temporal dHVG — original vs subrogados"
        ),
        "surrogate_dhvg_kl_title": (
            r"KL suavizada $\mathrm{KL}(P(k^{\mathrm{in}})\,\|\,P(k^{\mathrm{out}}))$"
        ),
        "surrogate_dhvg_idir_title": r"Asimetría direccional $i_{\mathrm{dir}}$",
        "surrogate_dhvg_ylabel_kl": (
            r"$\mathrm{KL}(P(k^{\mathrm{in}})\,\|\,P(k^{\mathrm{out}}))$"
        ),
        "surrogate_dhvg_ylabel_idir": r"$i_{\mathrm{dir}}$",
        "surrogate_dhvg_xlabel": "régimen",
        "surrogate_dhvg_idir_saturated": (
            r"$i_{\mathrm{dir}}=1$ en todo dHVG: todas las aristas siguen la flecha "
            "temporal y la asimetría de Frobenius está saturada."
        ),
        "invariance_matrix_title": "Matriz de invarianzas (estabilidad / F1 OOD)",
        "phase_maps_families_title": r"Mapas de fase: Macro-F1 RF en $N \times$ SNR",
        "phase_topo_vs_spec_title": (
            r"Fallo topológico vs espectral frente a SNR ($N$ fijo)"
        ),
        "phase_disconnection_by_family_title": (
            r"Desconexión algebraica por familia "
            r"($\lambda_2(G)\leq 10^{-12}$ o no finito)"
        ),
        "phase_disconnection_rate_label": "Tasa de desconexión",
        "pid_title": "Descomposición de información parcial (átomos MMI)",
        "pid_ylabel": "Información (bits)",
        "pid_xlabel": "Par de familias (1ª | 2ª)",
        "pid_redundancy": "Redundancia",
        "pid_unique_i": "Única (1ª familia)",
        "pid_unique_j": "Única (2ª familia)",
        "pid_synergy": "Sinergia",
        "pid_synergy_negative": "Sinergia negativa",
        "shapley_title": "Importancia Shapley exacta por familia",
        "shapley_ylabel": "Valor de Shapley (Macro-F1)",
        "shapley_xlabel": "Familia TS2G",
        "model_random_forest": "Random Forest",
        "model_logistic": "Logística L2",
        "pareto_title": r"Frontera de Pareto coste–exactitud (tiempo vs Macro-F1)",
        "pareto_xlabel": "Tiempo por serie (ms, N={n_steps})",
        "pareto_n_families_one": "{size} familia",
        "pareto_n_families_many": "{size} familias",
        "pareto_front": "Frontera",
        "pareto_front_legend": "frontera",
        "invariance_error_title": "Error normalizado de descriptores (mediana)",
        "invariance_delta_title": (
            r"$\Delta$ Macro-F1 RF OOD (perturbado $-$ clean)"
        ),
        "invariance_method_error_title": "A) Error normalizado de descriptores (mediana)",
        "invariance_method_delta_title": (
            r"B) $\Delta$ Macro-F1 RF OOD (perturbado $-$ clean)"
        ),
        "invariance_matrix_by_method_title": (
            "Matriz de invarianzas por método (estabilidad / F1 OOD)"
        ),
        "rq1_pid_title": (
            r"PID por pares ($I_{\mathrm{broja}}$) según alcance dinámico"
        ),
        "rq1_scope_global": "global",
        "rq1_omega_suptitle": (
            r"O-información $\Omega$ con IC bootstrap al 95%"
        ),
        "rq1_omega_groups_title": "Macro-grupos",
        "rq1_omega_regimes_title": "Regímenes",
        "rq1_omega_xlabel": r"$\Omega$ (bits)",
        "rq1_omega_nonsig": r"IC contiene 0",
        "rq1_sii_title": "Índice de interacción de Shapley (SII) por pares",
        "rq1_sii_ylabel": "SII (Macro-F1)",
        "rq1_sii_baseline_title": r"Línea base $v(\emptyset)$ vs conjunto completo",
        "rq1_shapley_phi_title": r"Shapley $\phi_i$ con IC 95%",
        "rq1_shapley_phi_ylabel": r"Shapley $\phi$ (Macro-F1)",
        "rq1_one_vs_rest_title": r"PID uno-frente-al-resto ($X_i \mid X_{-i}$)",
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
