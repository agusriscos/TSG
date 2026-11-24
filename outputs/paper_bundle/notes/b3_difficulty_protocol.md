# B3: protocolo de dificultad no saturado

Mejoras de experimentación empaquetadas en este bundle (sin tocar el ensemble limpio N=1500, R=30 ni la caché B1):

1. **Curva N × SNR** (`b3_difficulty_curve`, fig. `b3_hard_08`): rejilla a priori sobre los 12 regímenes duros; el headline se fija en N=500, SNR=20 dB (Macro-F1 RF ≈ 0.88, IC sin incluir 1), fuera del techo del panel limpio.
2. **Ablación y pares críticos en el punto operativo** (`b3_ablation`, `b3_critical_pairs_binary_hard`, fig. `b3_hard_09`); el panel limpio queda archivado en `b3_ablation_n1500_clean` y `b3_proper_scores_n1500_clean`.
3. **dHVG**: la figura de irreversibilidad muestra solo la KL suavizada in/out; se omite `i_dir` porque satura en 1 en todo dHVG.
4. **CLI**: `run_phase1_audit.py --difficulty-sweep` (opcional `--difficulty-force`); fingerprint de caché con SNR para no colisionar con el digest limpio histórico.

Artefactos clave: `tables/b3_headline_operating_point.csv`, `tables/b3_difficulty_curve.csv`, `figures/b3_hard_08_*`, `figures/b3_hard_09_*`.
