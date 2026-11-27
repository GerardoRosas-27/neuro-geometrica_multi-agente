# Auditoría Científica y Técnica — `exp/wave-collision-autocal`

**Veredicto final:** **`AUDIT PASS`**
**Fecha UTC:** 2026-11-27T03:50:57.998946+00:00
**Rama:** `exp/wave-collision-autocal` · **HEAD:** `4a21cab1d0b31060e2b8f1c19c31fa16e098dc19`

## Verificaciones Ejecutadas

| Comprobación | Estado | Detalle |
|---|---|---|
| FROZEN.json SHA-256 | PASS | 13/13 hashes exact |
| DEV CSVs (16 seeds) | PASS | 64/64 CSVs valid, 0 NaN/Inf, 100% frozen checksums ok |
| CONFIRM CSVs (16 seeds) | PASS | 64/64 CSVs valid, 0 NaN/Inf, 100% frozen checksums ok |
| Bit-exact replay 0xF200 | PASS | 5 files matched=True |
| Rust unit tests (release) | PASS | tests_release.log: ok (0 failed) |
| SVG Figures (4) | PASS | 4/4 present |
| ORBITA / CDT-RQM / aura-engine untouched | PASS | 0 diffs |
| Strict branch isolation | PASS | branch=exp/wave-collision-autocal, 157 touched files inside allowed scope |

## Alcance y Aislamiento

- Experimento autónomo de `768` ondas colisionantes auto-calibrables (`wave_collision/`), sin matriz densa `W_pair` y sin conexión a ÓRBITA (`orbita/`), CDT-RQM (`src/`), `aura-engine/` ni `Cargo.toml` raíz.
- Cero ceros de Riemann y cero primos en brazos adaptativos (`#[should_panic]` verificado); `256` ceros exactos de Riemann `< 1e-10` en el brazo de control.
- Invarianza bit a bit IEEE-754 de las `3,328` frecuencias y fases base en todas las semillas de transferencia.
