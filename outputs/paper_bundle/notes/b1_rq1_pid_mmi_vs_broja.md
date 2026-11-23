## PID contrast: MMI vs BROJA (global pairs)

| pair | mmi_redundancy | broja_redundancy | mmi_unique_i | broja_unique_i | mmi_unique_j | broja_unique_j | mmi_synergy | broja_synergy | mmi_forced_unique_zero | broja_both_unique_positive |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| vis|prox | 2.9245 | 0.4808 | 0.0000 | 0.7704 | 0.0294 | 0.7092 | 0.3294 | 0.0354 | True | True |
| vis|trans | 2.8220 | 0.8400 | 0.1025 | 0.4112 | 0.0000 | 0.5187 | 0.3707 | 0.0103 | True | True |
| vis|sim | 2.8329 | 0.4079 | 0.0916 | 0.8433 | 0.0000 | 0.7589 | 0.3034 | 0.0094 | True | True |
| prox|trans | 2.8220 | 0.5906 | 0.1319 | 0.5994 | 0.0000 | 0.7680 | 0.3018 | 0.0123 | True | True |
| prox|sim | 2.8329 | 0.2645 | 0.1210 | 0.9255 | 0.0000 | 0.9023 | 0.2223 | 0.0553 | True | True |
| trans|sim | 2.8220 | 0.3855 | 0.0000 | 0.9732 | 0.0109 | 0.7814 | 0.2804 | 0.0317 | True | True |


Notes: `mmi_forced_unique_zero` marks pairs where MMI set at least one unique atom to exactly 0. `broja_both_unique_positive` marks pairs where BROJA assigns strictly positive unique information to both families. BROJA atoms come from quantile-binned family PCA scores; MMI atoms come from the predictive-MI pipeline — absolute bit scales are not directly comparable, but the unique-atom pattern is.
