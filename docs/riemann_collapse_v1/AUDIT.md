# Auditoría de Evidencia — Colapso de Ondas Pasado–Futuro v1 (`exp/riemann-wave-collapse`)

**Estado global:** `AUDIT PASS`

| Comprobación | Estado | Detalle |
|---|---|---|
| SHA-256 código fuente == FROZEN.json == CONFIRM/run.json | `PASS` | `a0277590be2c2c65` |
| 32 semillas completas (16 DEV + 16 CONFIRM, 960 filas totales) | `PASS` | `rows=960, missing=0` |
| Cero NaN / Inf en todas las métricas | `PASS` | `non_finite=0` |
| Replay determinista bit-a-bit en 0xB800 | `PASS` | `277 celdas exactas` |
| Tests unitarios debug y release 100% PASS | `PASS` | `tests.log & tests_release.log` |
| Aislamiento estricto (solo riemann_collapse/ y docs/riemann_collapse_v1/) | `PASS` | `209 archivos en rama` |

## Integración con aplicación principal (`main`)
- Checks del crate raíz (`Paso0`, `known_acc16`, `E33`, `aura-engine`, `orbita`): **NO EVALUADOS** (rama aislada; ningún archivo de `main` fue modificado).
