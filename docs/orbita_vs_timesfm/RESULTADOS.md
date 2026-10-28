# RESULTADOS — exp/orbita-vs-timesfm

Generado automáticamente desde analysis.json. Protocolo: docs/orbita_vs_timesfm/PREREG.md. 0 consultas a TEST. timesfm Version: 3.0.2, checkpoint google/timesfm-2.5-200m-pytorch, CPU.
Latentes: reconstruidos de los embeddings Gemma cacheados de dinlat; verificación automática PERS/AR2 idénticos a res_*.json de dinlat.

## DEV
Seeds con resultado: 16/16

| Contraste G=log(MSE_B/MSE_A) | media | IC95 |
|---|---|---|
| H1 ORB_SESGO vs TIMESFM (ho) | +0.3296 | [+0.2979, +0.3615] |
| H1b ORB_LIBRE vs TIMESFM (ho) | +0.0419 | [+0.0039, +0.0796] |
| H2 TIMESFM vs PERS (ho) | +0.3548 | [+0.3341, +0.3748] |
| H3 ORB_LIBRE vs TIMESFM (tf) | -0.0178 | [-0.1010, +0.0595] |
| S ORB_SESGO vs TIMESFM (tf) | -0.1385 | [-0.1775, -0.0971] |
| S TIMESFM vs AR2 (ho) | -0.1067 | [-0.1430, -0.0714] |
| S TIMESFM vs VAR2_OLS (ho) | -0.0901 | [-0.1287, -0.0532] |
| S AR2 vs VAR2_OLS (ho) | +0.0166 | [+0.0133, +0.0198] |
| S TIMESFM vs AR2 (tf) | -0.0470 | [-0.0780, -0.0152] |
| S TIMESFM vs VAR2_OLS (tf) | -0.0308 | [-0.0634, +0.0026] |
| S AR2 vs VAR2_OLS (tf) | +0.0162 | [+0.0139, +0.0187] |
| S TIMESFM vs PERS (tf) | +0.2779 | [+0.2503, +0.3038] |
| S ORB_SESGO vs TIMESFM (ho, geométricas) | +0.1985 | [+0.1700, +0.2292] |
| S ORB_LIBRE vs TIMESFM (ho, geométricas) | -0.4207 | [-0.4995, -0.3515] |
| S AR2 vs TIMESFM (ho, geométricas) | -0.2443 | [-0.2925, -0.1979] |
| S PERS vs TIMESFM (ho, geométricas) | +0.1023 | [+0.0885, +0.1163] |
| S ORB_SESGO vs TIMESFM (ho, no geométricas) | +0.3688 | [+0.3291, +0.4081] |
| S ORB_LIBRE vs TIMESFM (ho, no geométricas) | +0.2236 | [+0.1759, +0.2696] |
| S AR2 vs TIMESFM (ho, no geométricas) | +0.2294 | [+0.1871, +0.2743] |
| S PERS vs TIMESFM (ho, no geométricas) | -0.4507 | [-0.4747, -0.4264] |
| S h3 ORB_SESGO vs TIMESFM (ho) | +0.4121 | [+0.3667, +0.4522] |
| S h3 ORB_LIBRE vs TIMESFM (ho) | +0.0365 | [-0.0239, +0.0876] |
| S h3 AR2 vs TIMESFM (ho) | +0.1854 | [+0.1611, +0.2146] |
| S h3 PERS vs TIMESFM (ho) | -0.2078 | [-0.2292, -0.1864] |
| C TIMESFM vs PERS (C1_permutado) | +0.3267 | [+0.3183, +0.3352] |
| C ORB_LIBRE vs TIMESFM (C1_permutado) | +0.2305 | [+0.1967, +0.2555] |
| C TIMESFM vs PERS (C2_barajado) | +0.3111 | [+0.3004, +0.3230] |
| C ORB_LIBRE vs TIMESFM (C2_barajado) | +0.0109 | [-0.0125, +0.0334] |

Criterios: H1=sí, H1b=no, H2=sí, H3=no, TIMESFM_mejor_que_ORB_SESGO=no, TIMESFM_mejor_que_ORB_LIBRE=no
Gate DEV→CONFIRM: pasa → CONFIRM

## CONFIRM
Seeds con resultado: 0/16

| Contraste G=log(MSE_B/MSE_A) | media | IC95 |
|---|---|---|
| H1 ORB_SESGO vs TIMESFM (ho) | +nan | [+nan, +nan] |
| H1b ORB_LIBRE vs TIMESFM (ho) | +nan | [+nan, +nan] |
| H2 TIMESFM vs PERS (ho) | +nan | [+nan, +nan] |
| H3 ORB_LIBRE vs TIMESFM (tf) | +nan | [+nan, +nan] |
| S ORB_SESGO vs TIMESFM (tf) | +nan | [+nan, +nan] |
| S TIMESFM vs AR2 (ho) | +nan | [+nan, +nan] |
| S TIMESFM vs VAR2_OLS (ho) | +nan | [+nan, +nan] |
| S AR2 vs VAR2_OLS (ho) | +nan | [+nan, +nan] |
| S TIMESFM vs AR2 (tf) | +nan | [+nan, +nan] |
| S TIMESFM vs VAR2_OLS (tf) | +nan | [+nan, +nan] |
| S AR2 vs VAR2_OLS (tf) | +nan | [+nan, +nan] |
| S TIMESFM vs PERS (tf) | +nan | [+nan, +nan] |
| S ORB_SESGO vs TIMESFM (ho, geométricas) | +nan | [+nan, +nan] |
| S ORB_LIBRE vs TIMESFM (ho, geométricas) | +nan | [+nan, +nan] |
| S AR2 vs TIMESFM (ho, geométricas) | +nan | [+nan, +nan] |
| S PERS vs TIMESFM (ho, geométricas) | +nan | [+nan, +nan] |
| S ORB_SESGO vs TIMESFM (ho, no geométricas) | +nan | [+nan, +nan] |
| S ORB_LIBRE vs TIMESFM (ho, no geométricas) | +nan | [+nan, +nan] |
| S AR2 vs TIMESFM (ho, no geométricas) | +nan | [+nan, +nan] |
| S PERS vs TIMESFM (ho, no geométricas) | +nan | [+nan, +nan] |
| S h3 ORB_SESGO vs TIMESFM (ho) | +nan | [+nan, +nan] |
| S h3 ORB_LIBRE vs TIMESFM (ho) | +nan | [+nan, +nan] |
| S h3 AR2 vs TIMESFM (ho) | +nan | [+nan, +nan] |
| S h3 PERS vs TIMESFM (ho) | +nan | [+nan, +nan] |
| C TIMESFM vs PERS (C1_permutado) | +nan | [+nan, +nan] |
| C ORB_LIBRE vs TIMESFM (C1_permutado) | +nan | [+nan, +nan] |
| C TIMESFM vs PERS (C2_barajado) | +nan | [+nan, +nan] |
| C ORB_LIBRE vs TIMESFM (C2_barajado) | +nan | [+nan, +nan] |

Criterios: H1=no, H1b=no, H2=no, H3=no, TIMESFM_mejor_que_ORB_SESGO=no, TIMESFM_mejor_que_ORB_LIBRE=no
