# Auditoría de Evidencia — Frecuencia de Ondas de Riemann v1 (`exp/riemann-wave-inference`)

**Estado global:** `AUDIT PASS`

| Comprobación | Estado | Detalle |
|---|---|---|
| SHA-256 código fuente == FROZEN.json == CONFIRM/run.json | `PASS` | `6ee81050faf1e80e` |
| 32 semillas completas (16 DEV + 16 CONFIRM, 960 filas totales) | `PASS` | `rows=960, missing=0` |
| Cero NaN / Inf en todas las métricas | `PASS` | `non_finite=0` |
| Replay determinista bit-a-bit en 0xB700 | `PASS` | `286 celdas exactas` |
| Tests unitarios debug y release 100% PASS | `PASS` | `tests.log & tests_release.log` |
| Aislamiento estricto (solo riemann_wave/ y docs/riemann_wave_v1/) | `PASS` | `240 archivos en rama` |

## Integración con aplicación principal (`main`)
- Checks del crate raíz (`Paso0`, `known_acc16`, `E33`, `aura-engine`, `orbita`): **NO EVALUADOS** (rama aislada; ningún archivo de `main` fue modificado).
