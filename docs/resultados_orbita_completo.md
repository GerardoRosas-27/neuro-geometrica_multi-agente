# Resultados completos — ÓRBITA vs v4 E3 vs MLP M3 (DEV + CONFIRM + benchmark de inferencia)

**Rama:** `exp/orbita` · **Pre-registro:** `docs/preregistro_orbita.md` (`f5d7781`, antes de DEV/CONFIRM) · **Resumen y veredicto:** `docs/resultados_orbita.md` · **Fecha:** 2026-10-14.
Fuente de todos los números: `artifacts/orbita/{dev,confirm}/metrics_0x*.csv`, `parity_0x*.json`, `paired_stats.md` (ÓRBITA) y `artifacts/v4/distill_e/{dev,confirm}/metrics.csv` (v4 E3 y M3, artefactos publicados de §16, no re-entrenados). Tablas por semilla generadas con `python3 orbita/per_seed_tables.py dev|confirm`; medias e IC95 con `python3 orbita/analyze_orbita.py dev|confirm`. Código `orbita/` de entrenamiento/evaluación sin cambios desde `f5d7781` (sólo se añadió después el subcomando `bench` y scripts de tablas).

## 1. Medias por fase

| Fase | Brazo | params | acc 1 paso | rel_err₁ | gate E33 | región mediana | h1 | h2 | h4 | h8 | h16 | h32 | h64 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| DEV | v4 E3 | 2590 | 0.9834 | 0.0090 | 13/16 | 8 | 0.992 | 0.985 | 0.925 | 0.632 | 0.264 | 0.050 | 0.005 |
| DEV | MLP M3 | 2590 | 0.9802 | 0.0093 | 12/16 | 8 | 0.992 | 0.986 | 0.939 | 0.616 | 0.230 | 0.055 | 0.010 |
| DEV | **ÓRBITA** | 2602 | 0.9783 | 0.0082 | **16/16** | 8 | 0.984 | 0.984 | 0.942 | 0.706 | 0.313 | 0.079 | 0.017 |
| CONFIRM | v4 E3 | 2590 | 0.9864 | 0.0089 | 9/16 | 8 | 0.991 | 0.978 | 0.869 | 0.551 | 0.217 | 0.044 | 0.008 |
| CONFIRM | MLP M3 | 2590 | 0.9837 | 0.0093 | 10/16 | 8 | 0.994 | 0.981 | 0.878 | 0.558 | 0.210 | 0.055 | 0.011 |
| CONFIRM | **ÓRBITA** | 2602 | 0.9819 | 0.0081 | **16/16** | 8 | 0.988 | 0.982 | 0.912 | 0.687 | 0.323 | 0.073 | 0.008 |

## 2. Comparaciones pareadas (IC95 bootstrap, 10 000 remuestreos, semilla 12345; "a favor" = diferencia > 0)

| Fase | Comparación | acc h16 | región | acc h8 | acc 1 paso |
|---|---|---|---|---|---|
| DEV | ÓRBITA − E3 | +0.0488 [−0.0029, +0.1065] (8/16) | +1.25 [+0.25, +2.50] (4/16) | +0.0742 [+0.0195, +0.1328] (11/16) | −0.0051 [−0.0127, +0.0024] (7/16) |
| DEV | ÓRBITA − M3 | +0.0830 [+0.0224, +0.1416] (12/16) | +1.50 [+0.50, +2.75] (5/16) | +0.0898 [+0.0225, +0.1562] (12/16) | −0.0020 [−0.0120, +0.0085] (8/16) |
| CONFIRM | ÓRBITA − E3 | +0.1065 [+0.0664, +0.1504] (15/16) | +1.75 [+0.75, +2.75] (7/16) | +0.1357 [+0.0752, +0.1963] (14/16) | −0.0044 [−0.0108, +0.0017] (6/16) |
| CONFIRM | ÓRBITA − M3 | +0.1133 [+0.0615, +0.1641] (14/16) | +1.50 [+0.50, +2.50] (6/16) | +0.1289 [+0.0498, +0.2041] (14/16) | −0.0017 [−0.0098, +0.0068] (5/16) |

## 3. Auditoría (E35/limpieza/fugas/proceso nuevo/paridad)
- ÓRBITA: estado CLEAN, leakage = 0 y 0 queries (CDT, RQM, tabla, NN, atractor, memoria directa) en las 32 seeds; flags de auditoría incluyen los targets de la CDT y todos los hits del auditor sobre el pool de replay.
- Proceso nuevo (`orbita fresh`): `fresh_ok=true` (|Δacc| < 1e-12, 0 queries) en 32/32.
- Paridad Rust↔PyTorch (f64, 16 sondas por seed): máximo 4.9e-15 (DEV) y 4.1e-15 (CONFIRM).
- v4 E3 / M3: CLEAN, leakage 0, 0 queries según los artefactos de §16; proceso nuevo de E3 registrado en las notas del gate (`fresh_E3`). M3 no tiene evaluación en proceso nuevo en §16.

## 4. Tablas por semilla
### DEV — por semilla

| seed | modelo | gate E33 | región | acc1 | rel1 | h1 | h2 | h4 | h8 | h16 | estado | leak | queries | proceso nuevo | paridad Rust↔Py |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0xA400 | ÓRBITA | PASS | 8 | 1.0000 | 0.0067 | 1.000 | 1.000 | 1.000 | 0.578 | 0.266 | CLEAN | 0 | 0 | true | 3.2e-15 |
| 0xA400 | v4 E3 | PASS | 8 | 0.9922 | 0.0078 | 1.000 | 1.000 | 0.953 | 0.641 | 0.328 | CLEAN | 0 | 0 | - | — |
| 0xA400 | MLP M3 | PASS | 8 | 0.9570 | 0.0100 | 1.000 | 1.000 | 0.953 | 0.625 | 0.344 | CLEAN | 0 | 0 | — | — |
| 0xA401 | ÓRBITA | PASS | 8 | 0.9805 | 0.0083 | 0.984 | 1.000 | 0.953 | 0.547 | 0.219 | CLEAN | 0 | 0 | true | 2.7e-15 |
| 0xA401 | v4 E3 | FAIL | 4 | 0.9766 | 0.0107 | 0.984 | 0.984 | 0.922 | 0.375 | 0.203 | CLEAN | 0 | 0 | - | — |
| 0xA401 | MLP M3 | FAIL | 4 | 0.9727 | 0.0114 | 0.984 | 0.969 | 0.906 | 0.406 | 0.234 | CLEAN | 0 | 0 | — | — |
| 0xA402 | ÓRBITA | PASS | 8 | 0.9766 | 0.0083 | 1.000 | 1.000 | 0.969 | 0.781 | 0.344 | CLEAN | 0 | 0 | true | 2.0e-15 |
| 0xA402 | v4 E3 | FAIL | 4 | 0.9883 | 0.0090 | 1.000 | 1.000 | 0.828 | 0.422 | 0.141 | CLEAN | 0 | 0 | - | — |
| 0xA402 | MLP M3 | PASS | 8 | 0.9688 | 0.0101 | 1.000 | 1.000 | 0.953 | 0.703 | 0.172 | CLEAN | 0 | 0 | — | — |
| 0xA403 | ÓRBITA | PASS | 8 | 0.9805 | 0.0078 | 1.000 | 0.984 | 0.953 | 0.578 | 0.188 | CLEAN | 0 | 0 | true | 2.4e-15 |
| 0xA403 | v4 E3 | PASS | 8 | 0.9805 | 0.0103 | 0.953 | 0.953 | 0.922 | 0.656 | 0.172 | CLEAN | 0 | 0 | - | — |
| 0xA403 | MLP M3 | FAIL | 4 | 0.9570 | 0.0126 | 0.953 | 0.938 | 0.906 | 0.438 | 0.062 | CLEAN | 0 | 0 | — | — |
| 0xA404 | ÓRBITA | PASS | 8 | 0.9922 | 0.0089 | 1.000 | 1.000 | 0.906 | 0.656 | 0.203 | CLEAN | 0 | 0 | true | 2.7e-15 |
| 0xA404 | v4 E3 | PASS | 8 | 0.9883 | 0.0089 | 1.000 | 1.000 | 0.922 | 0.594 | 0.297 | CLEAN | 0 | 0 | - | — |
| 0xA404 | MLP M3 | PASS | 8 | 0.9883 | 0.0085 | 1.000 | 1.000 | 0.938 | 0.672 | 0.266 | CLEAN | 0 | 0 | — | — |
| 0xA405 | ÓRBITA | PASS | 8 | 0.9648 | 0.0094 | 0.906 | 0.969 | 0.797 | 0.500 | 0.109 | CLEAN | 0 | 0 | true | 1.9e-15 |
| 0xA405 | v4 E3 | FAIL | 4 | 0.9883 | 0.0091 | 0.969 | 0.938 | 0.781 | 0.438 | 0.141 | CLEAN | 0 | 0 | - | — |
| 0xA405 | MLP M3 | PASS | 8 | 0.9883 | 0.0085 | 0.969 | 0.969 | 0.938 | 0.547 | 0.094 | CLEAN | 0 | 0 | — | — |
| 0xA406 | ÓRBITA | PASS | 8 | 0.9609 | 0.0088 | 1.000 | 1.000 | 1.000 | 0.953 | 0.359 | CLEAN | 0 | 0 | true | 3.2e-15 |
| 0xA406 | v4 E3 | PASS | 8 | 0.9883 | 0.0081 | 1.000 | 1.000 | 1.000 | 0.875 | 0.391 | CLEAN | 0 | 0 | - | — |
| 0xA406 | MLP M3 | PASS | 8 | 0.9727 | 0.0095 | 1.000 | 1.000 | 0.984 | 0.875 | 0.234 | CLEAN | 0 | 0 | — | — |
| 0xA407 | ÓRBITA | PASS | 8 | 0.9688 | 0.0085 | 1.000 | 1.000 | 0.984 | 0.766 | 0.297 | CLEAN | 0 | 0 | true | 1.7e-15 |
| 0xA407 | v4 E3 | PASS | 8 | 0.9844 | 0.0091 | 1.000 | 1.000 | 0.969 | 0.625 | 0.125 | CLEAN | 0 | 0 | - | — |
| 0xA407 | MLP M3 | FAIL | 4 | 0.9961 | 0.0085 | 1.000 | 0.984 | 0.938 | 0.406 | 0.141 | CLEAN | 0 | 0 | — | — |
| 0xA408 | ÓRBITA | PASS | 8 | 0.9844 | 0.0074 | 1.000 | 1.000 | 1.000 | 0.703 | 0.344 | CLEAN | 0 | 0 | true | 2.8e-15 |
| 0xA408 | v4 E3 | PASS | 8 | 0.9766 | 0.0094 | 0.984 | 0.984 | 0.922 | 0.531 | 0.266 | CLEAN | 0 | 0 | - | — |
| 0xA408 | MLP M3 | FAIL | 4 | 0.9805 | 0.0095 | 0.984 | 0.984 | 0.953 | 0.453 | 0.203 | CLEAN | 0 | 0 | — | — |
| 0xA409 | ÓRBITA | PASS | 8 | 0.9922 | 0.0064 | 1.000 | 1.000 | 0.953 | 0.781 | 0.484 | CLEAN | 0 | 0 | true | 3.8e-15 |
| 0xA409 | v4 E3 | PASS | 8 | 0.9961 | 0.0080 | 1.000 | 1.000 | 0.969 | 0.688 | 0.188 | CLEAN | 0 | 0 | - | — |
| 0xA409 | MLP M3 | PASS | 8 | 0.9961 | 0.0069 | 1.000 | 1.000 | 1.000 | 0.609 | 0.250 | CLEAN | 0 | 0 | — | — |
| 0xA40A | ÓRBITA | PASS | 8 | 0.9844 | 0.0079 | 1.000 | 1.000 | 1.000 | 0.734 | 0.375 | CLEAN | 0 | 0 | true | 2.4e-15 |
| 0xA40A | v4 E3 | PASS | 8 | 1.0000 | 0.0074 | 1.000 | 1.000 | 0.969 | 0.562 | 0.203 | CLEAN | 0 | 0 | - | — |
| 0xA40A | MLP M3 | PASS | 8 | 0.9922 | 0.0073 | 1.000 | 1.000 | 0.984 | 0.672 | 0.125 | CLEAN | 0 | 0 | — | — |
| 0xA40B | ÓRBITA | PASS | 8 | 0.9883 | 0.0078 | 1.000 | 1.000 | 0.984 | 0.766 | 0.203 | CLEAN | 0 | 0 | true | 4.9e-15 |
| 0xA40B | v4 E3 | PASS | 8 | 0.9648 | 0.0095 | 1.000 | 1.000 | 0.984 | 0.688 | 0.250 | CLEAN | 0 | 0 | - | — |
| 0xA40B | MLP M3 | PASS | 8 | 0.9688 | 0.0104 | 1.000 | 1.000 | 0.984 | 0.609 | 0.172 | CLEAN | 0 | 0 | — | — |
| 0xA40C | ÓRBITA | PASS | 8 | 0.9414 | 0.0110 | 1.000 | 0.984 | 0.984 | 0.797 | 0.328 | CLEAN | 0 | 0 | true | 2.5e-15 |
| 0xA40C | v4 E3 | PASS | 8 | 0.9648 | 0.0108 | 1.000 | 0.984 | 0.938 | 0.797 | 0.328 | CLEAN | 0 | 0 | - | — |
| 0xA40C | MLP M3 | PASS | 8 | 0.9805 | 0.0108 | 1.000 | 0.969 | 0.859 | 0.531 | 0.172 | CLEAN | 0 | 0 | — | — |
| 0xA40D | ÓRBITA | PASS | 16 | 1.0000 | 0.0057 | 1.000 | 1.000 | 0.891 | 0.719 | 0.547 | CLEAN | 0 | 0 | true | 2.0e-15 |
| 0xA40D | v4 E3 | PASS | 8 | 0.9961 | 0.0075 | 1.000 | 1.000 | 0.859 | 0.734 | 0.375 | CLEAN | 0 | 0 | - | — |
| 0xA40D | MLP M3 | PASS | 8 | 0.9883 | 0.0080 | 1.000 | 1.000 | 0.875 | 0.688 | 0.312 | CLEAN | 0 | 0 | — | — |
| 0xA40E | ÓRBITA | PASS | 8 | 0.9883 | 0.0070 | 0.984 | 0.984 | 0.969 | 0.875 | 0.453 | CLEAN | 0 | 0 | true | 3.1e-15 |
| 0xA40E | v4 E3 | PASS | 8 | 0.9727 | 0.0095 | 1.000 | 0.984 | 0.953 | 0.781 | 0.469 | CLEAN | 0 | 0 | - | — |
| 0xA40E | MLP M3 | PASS | 8 | 1.0000 | 0.0080 | 1.000 | 0.984 | 0.953 | 0.844 | 0.422 | CLEAN | 0 | 0 | — | — |
| 0xA40F | ÓRBITA | PASS | 8 | 0.9492 | 0.0106 | 0.875 | 0.828 | 0.734 | 0.562 | 0.281 | CLEAN | 0 | 0 | true | 3.0e-15 |
| 0xA40F | v4 E3 | PASS | 8 | 0.9766 | 0.0086 | 0.984 | 0.938 | 0.906 | 0.703 | 0.344 | CLEAN | 0 | 0 | - | — |
| 0xA40F | MLP M3 | PASS | 8 | 0.9766 | 0.0081 | 0.984 | 0.984 | 0.906 | 0.781 | 0.469 | CLEAN | 0 | 0 | — | — |

### CONFIRM — por semilla

| seed | modelo | gate E33 | región | acc1 | rel1 | h1 | h2 | h4 | h8 | h16 | estado | leak | queries | proceso nuevo | paridad Rust↔Py |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0xB400 | ÓRBITA | PASS | 8 | 0.9648 | 0.0094 | 0.969 | 0.969 | 0.938 | 0.812 | 0.312 | CLEAN | 0 | 0 | true | 2.5e-15 |
| 0xB400 | v4 E3 | PASS | 8 | 0.9883 | 0.0085 | 1.000 | 0.984 | 0.953 | 0.656 | 0.266 | CLEAN | 0 | 0 | acc=0.9883,q=0 | — |
| 0xB400 | MLP M3 | PASS | 8 | 0.9922 | 0.0085 | 1.000 | 0.984 | 0.984 | 0.656 | 0.219 | CLEAN | 0 | 0 | — | — |
| 0xB401 | ÓRBITA | PASS | 8 | 0.9961 | 0.0070 | 1.000 | 1.000 | 0.922 | 0.688 | 0.328 | CLEAN | 0 | 0 | true | 4.1e-15 |
| 0xB401 | v4 E3 | FAIL | 4 | 1.0000 | 0.0093 | 1.000 | 1.000 | 0.891 | 0.422 | 0.156 | CLEAN | 0 | 0 | acc=1.0000,q=0 | — |
| 0xB401 | MLP M3 | PASS | 8 | 0.9883 | 0.0099 | 1.000 | 1.000 | 0.891 | 0.625 | 0.219 | CLEAN | 0 | 0 | — | — |
| 0xB402 | ÓRBITA | PASS | 8 | 0.9883 | 0.0070 | 1.000 | 1.000 | 0.891 | 0.609 | 0.172 | CLEAN | 0 | 0 | true | 2.4e-15 |
| 0xB402 | v4 E3 | FAIL | 4 | 0.9805 | 0.0091 | 1.000 | 1.000 | 0.875 | 0.469 | 0.156 | CLEAN | 0 | 0 | acc=0.9805,q=0 | — |
| 0xB402 | MLP M3 | FAIL | 4 | 1.0000 | 0.0077 | 1.000 | 1.000 | 0.891 | 0.406 | 0.078 | CLEAN | 0 | 0 | — | — |
| 0xB403 | ÓRBITA | PASS | 8 | 0.9844 | 0.0084 | 0.984 | 0.984 | 0.922 | 0.625 | 0.328 | CLEAN | 0 | 0 | true | 2.1e-15 |
| 0xB403 | v4 E3 | PASS | 8 | 0.9844 | 0.0092 | 0.984 | 0.984 | 0.891 | 0.562 | 0.250 | CLEAN | 0 | 0 | acc=0.9844,q=0 | — |
| 0xB403 | MLP M3 | PASS | 8 | 0.9883 | 0.0093 | 0.984 | 0.922 | 0.797 | 0.609 | 0.297 | CLEAN | 0 | 0 | — | — |
| 0xB404 | ÓRBITA | PASS | 8 | 0.9844 | 0.0075 | 0.984 | 0.969 | 0.938 | 0.734 | 0.219 | CLEAN | 0 | 0 | true | 1.9e-15 |
| 0xB404 | v4 E3 | FAIL | 4 | 0.9961 | 0.0079 | 1.000 | 0.969 | 0.844 | 0.469 | 0.172 | CLEAN | 0 | 0 | acc=0.9961,q=0 | — |
| 0xB404 | MLP M3 | FAIL | 4 | 0.9883 | 0.0092 | 1.000 | 0.969 | 0.812 | 0.391 | 0.062 | CLEAN | 0 | 0 | — | — |
| 0xB405 | ÓRBITA | PASS | 8 | 0.9766 | 0.0085 | 1.000 | 0.984 | 0.922 | 0.766 | 0.469 | CLEAN | 0 | 0 | true | 2.4e-15 |
| 0xB405 | v4 E3 | PASS | 8 | 0.9883 | 0.0094 | 0.969 | 0.953 | 0.906 | 0.656 | 0.234 | CLEAN | 0 | 0 | acc=0.9883,q=0 | — |
| 0xB405 | MLP M3 | FAIL | 4 | 0.9766 | 0.0119 | 0.969 | 0.969 | 0.797 | 0.484 | 0.203 | CLEAN | 0 | 0 | — | — |
| 0xB406 | ÓRBITA | PASS | 8 | 0.9883 | 0.0087 | 1.000 | 0.953 | 0.875 | 0.500 | 0.094 | CLEAN | 0 | 0 | true | 2.4e-15 |
| 0xB406 | v4 E3 | FAIL | 4 | 0.9961 | 0.0081 | 1.000 | 1.000 | 0.875 | 0.312 | 0.062 | CLEAN | 0 | 0 | acc=0.9961,q=0 | — |
| 0xB406 | MLP M3 | PASS | 8 | 0.9961 | 0.0085 | 1.000 | 1.000 | 0.906 | 0.656 | 0.219 | CLEAN | 0 | 0 | — | — |
| 0xB407 | ÓRBITA | PASS | 8 | 0.9688 | 0.0090 | 1.000 | 1.000 | 0.984 | 0.719 | 0.469 | CLEAN | 0 | 0 | true | 4.1e-15 |
| 0xB407 | v4 E3 | PASS | 8 | 1.0000 | 0.0071 | 1.000 | 1.000 | 0.828 | 0.594 | 0.375 | CLEAN | 0 | 0 | acc=1.0000,q=0 | — |
| 0xB407 | MLP M3 | PASS | 8 | 0.9961 | 0.0071 | 1.000 | 0.969 | 0.938 | 0.594 | 0.359 | CLEAN | 0 | 0 | — | — |
| 0xB408 | ÓRBITA | PASS | 8 | 0.9805 | 0.0077 | 1.000 | 1.000 | 0.984 | 0.859 | 0.453 | CLEAN | 0 | 0 | true | 2.3e-15 |
| 0xB408 | v4 E3 | FAIL | 4 | 0.9805 | 0.0090 | 0.969 | 0.906 | 0.797 | 0.453 | 0.188 | CLEAN | 0 | 0 | acc=0.9805,q=0 | — |
| 0xB408 | MLP M3 | FAIL | 4 | 0.9688 | 0.0109 | 0.984 | 0.969 | 0.891 | 0.453 | 0.188 | CLEAN | 0 | 0 | — | — |
| 0xB409 | ÓRBITA | PASS | 8 | 0.9883 | 0.0073 | 1.000 | 0.984 | 0.922 | 0.828 | 0.391 | CLEAN | 0 | 0 | true | 2.0e-15 |
| 0xB409 | v4 E3 | PASS | 8 | 0.9805 | 0.0096 | 1.000 | 0.969 | 0.938 | 0.719 | 0.281 | CLEAN | 0 | 0 | acc=0.9805,q=0 | — |
| 0xB409 | MLP M3 | PASS | 8 | 0.9492 | 0.0113 | 1.000 | 0.984 | 0.938 | 0.703 | 0.188 | CLEAN | 0 | 0 | — | — |
| 0xB40A | ÓRBITA | PASS | 8 | 0.9883 | 0.0079 | 1.000 | 1.000 | 0.812 | 0.516 | 0.234 | CLEAN | 0 | 0 | true | 1.6e-15 |
| 0xB40A | v4 E3 | FAIL | 4 | 0.9805 | 0.0103 | 0.953 | 0.906 | 0.641 | 0.391 | 0.234 | CLEAN | 0 | 0 | acc=0.9805,q=0 | — |
| 0xB40A | MLP M3 | FAIL | 4 | 0.9883 | 0.0095 | 0.984 | 0.969 | 0.719 | 0.406 | 0.156 | CLEAN | 0 | 0 | — | — |
| 0xB40B | ÓRBITA | PASS | 8 | 0.9727 | 0.0082 | 1.000 | 1.000 | 0.953 | 0.672 | 0.359 | CLEAN | 0 | 0 | true | 1.7e-15 |
| 0xB40B | v4 E3 | FAIL | 4 | 0.9688 | 0.0099 | 1.000 | 1.000 | 0.859 | 0.438 | 0.094 | CLEAN | 0 | 0 | acc=0.9688,q=0 | — |
| 0xB40B | MLP M3 | FAIL | 4 | 0.9727 | 0.0099 | 1.000 | 1.000 | 0.906 | 0.344 | 0.109 | CLEAN | 0 | 0 | — | — |
| 0xB40C | ÓRBITA | PASS | 8 | 0.9570 | 0.0105 | 0.875 | 0.875 | 0.766 | 0.531 | 0.219 | CLEAN | 0 | 0 | true | 1.8e-15 |
| 0xB40C | v4 E3 | PASS | 8 | 0.9805 | 0.0093 | 1.000 | 0.984 | 0.938 | 0.641 | 0.203 | CLEAN | 0 | 0 | acc=0.9805,q=0 | — |
| 0xB40C | MLP M3 | PASS | 8 | 0.9844 | 0.0087 | 1.000 | 1.000 | 0.938 | 0.734 | 0.266 | CLEAN | 0 | 0 | — | — |
| 0xB40D | ÓRBITA | PASS | 8 | 1.0000 | 0.0070 | 1.000 | 1.000 | 0.875 | 0.734 | 0.438 | CLEAN | 0 | 0 | true | 2.6e-15 |
| 0xB40D | v4 E3 | PASS | 8 | 0.9922 | 0.0080 | 1.000 | 1.000 | 0.844 | 0.688 | 0.297 | CLEAN | 0 | 0 | acc=0.9922,q=0 | — |
| 0xB40D | MLP M3 | PASS | 8 | 0.9883 | 0.0081 | 1.000 | 1.000 | 0.891 | 0.672 | 0.375 | CLEAN | 0 | 0 | — | — |
| 0xB40E | ÓRBITA | PASS | 8 | 0.9805 | 0.0078 | 1.000 | 1.000 | 0.906 | 0.719 | 0.438 | CLEAN | 0 | 0 | true | 2.4e-15 |
| 0xB40E | v4 E3 | PASS | 8 | 0.9883 | 0.0079 | 1.000 | 1.000 | 0.953 | 0.797 | 0.375 | CLEAN | 0 | 0 | acc=0.9883,q=0 | — |
| 0xB40E | MLP M3 | PASS | 8 | 0.9844 | 0.0090 | 0.984 | 0.984 | 0.812 | 0.594 | 0.234 | CLEAN | 0 | 0 | — | — |
| 0xB40F | ÓRBITA | PASS | 8 | 0.9922 | 0.0071 | 1.000 | 1.000 | 0.984 | 0.672 | 0.250 | CLEAN | 0 | 0 | true | 2.2e-15 |
| 0xB40F | v4 E3 | PASS | 8 | 0.9766 | 0.0095 | 0.984 | 0.984 | 0.875 | 0.547 | 0.125 | CLEAN | 0 | 0 | acc=0.9766,q=0 | — |
| 0xB40F | MLP M3 | PASS | 8 | 0.9766 | 0.0095 | 1.000 | 0.984 | 0.938 | 0.594 | 0.188 | CLEAN | 0 | 0 | — | — |

(h32/h64 por semilla en los CSV; "proceso nuevo" de E3 = acc y queries del evaluador en proceso nuevo de v4.)

## 5. Veredicto pre-registrado
**Empate / no concluyente frente a v4 E3 → no se propone v6.** Gate E33 PASS en ambas fases (16/16, 16/16) y acc 1 paso dentro de 0.02, pero el criterio principal (IC95 de ÓRBITA − E3 en acc h16 > 0 en DEV **y** CONFIRM) falla en DEV ([−0.0029, +0.1065]); en CONFIRM sí se cumple. v4 tampoco gana. Secundario: contra el MLP M3 el IC95 de h16 es > 0 en ambas fases.

## 6. Desviaciones declaradas y pendientes
- ÓRBITA-v1 **no implementa** el verificador por paso ni la subdivisión adaptativa del diseño (`docs/arquitectura_optimizada.md` §3.4); tampoco Sobolev, Muon ni codificador EMA. Multi-horizonte sólo en coordenadas xy (λ_H = 0.01). 2602 parámetros (+12 vs 2590).
- Replay materializado en un pool de 100 000 muestras por seed (v4 lo genera en línea), muestreado ∝ exp(−r_f/τ).
- Ablación λ_H = 0 en DEV/CONFIRM: **pendiente** (no corrida).
- Incidencias de infraestructura: reinicios del box durante DEV (reanudado desde checkpoint determinista) y un primer lanzamiento de CONFIRM fallido por ausencia del binario (sin entrenar; relanzado con el mismo código).

## 7. Benchmark de inferencia (2026-10-14)
Comando: `orbita/target/release/orbita bench --seed 0xB400 --weights artifacts/orbita/confirm/weights_0xB400.json --e3 artifacts/v4/distill_e/confirm/checkpoints/dphi_DIST_E_E3_0xB400.json --reps N`, bajo `taskset` a un núcleo. Salida literal de las 6 corridas: `artifacts/orbita/bench/bench_runs_2026-10-14.txt`.

Corrida más limpia (6.ª, `taskset -c 6`, 20 000 reps, carga 1-min ≈5):

| Modelo | params | media µs/paso (b1) | p50 µs | p95 µs | pasos/s b1 | pasos/s b1024 | rollout 16 media / p95 µs |
|---|---|---|---|---|---|---|---|
| **ÓRBITA** | 2602 | **10.76** | **9.99** | **11.40** | **92 948** | **125 387** | **138.3 / 166.4** |
| v4 E3 | 2590 | 24.43 | 23.99 | 29.46 | 40 929 | 40 975 | 340.4 / 390.0 |
| MLP M3 | 2590 | 27.64 | 27.01 | 34.26 | 36 177 | 34 914 | 342.5 / 384.4 |

p50 en las 6 corridas: ÓRBITA 8.02–9.99 µs, E3 20.17–24.43 µs, M3 26.65–27.59 µs. Medias y p95 de las otras corridas están infladas por la carga.

Protocolo: inferencia Rust f64 por `Predictor::predict` (misma ruta que `evaluate`/E33), 1 hilo fijado con `taskset`, 2000 pasos de warmup, latencia por llamada con `Instant` sobre 10 000 o 20 000 pasos batch 1 (un objeto de 6 puntos) sobre estados TEST de 0xB400; "batch 1024" = 1024 estados × 20 pasadas (los tres modelos procesan secuencialmente, no hay vectorización por lotes); rollout = 16 pasos free-run × 2000. Carga: el box no estaba libre (job v5 de otro agente, ≈560 % CPU; carga 1-min 10.3 → 5.1 durante las corridas). **M3 se midió con pesos sin entrenar** de la misma arquitectura (`Dphi::gated_h` no residual, H=136), porque §16 no guardó checkpoint de M3; el coste no depende de los valores. ÓRBITA sin verificador/subdivisión (no implementados), así que no hay medida "con verificador". Resultado: ÓRBITA es la más rápida (≈2.4× E3 y ≈2.7× M3 en p50).

## 8. Pesos y datos publicados
Ver `artifacts/orbita/release/README.md` (pesos de las 32 seeds en binario f64 sin pérdida + datos de entrenamiento + comando de regeneración del pool de replay).
