# Auditoría Científica y Técnica — `exp/adaptive-spectral-collapse`

**Veredicto final:** **`AUDIT PASS`**
**Fecha UTC:** 2026-11-24T08:26:45.954494+00:00
**Rama:** `exp/adaptive-spectral-collapse` · **HEAD:** `3d7d95a1d3a6cd55522b65ffe692f0308bde04dd`

## Verificaciones Ejecutadas

| Comprobación | Estado | Detalle |
|---|---|---|
| FROZEN.json SHA-256 | PASS | 15/15 hashes exact |
| DEV CSVs (16 seeds) | PASS | 64/64 CSVs valid, 0 NaN/Inf |
| CONFIRM CSVs (16 seeds) | PASS | 64/64 CSVs valid, 0 NaN/Inf |
| Bit-exact replay 0xD100 | PASS | 6 files matched=True |
| Rust unit tests (debug) | PASS | tests.log: ok (0 failed) |
| Rust unit tests (release) | PASS | tests_release.log: ok (0 failed) |
| SVG Figures (4) | PASS | 4/4 present |
| Strict branch isolation | PASS | branch=exp/adaptive-spectral-collapse, 162 touched files all inside allowed directories |

## Alcance y Checks de la App Raíz / ÓRBITA

- Experimento estrictamente aislado en `adaptive_spectral_collapse/` y `docs/adaptive_spectral_collapse_v1/`.
- Ninguna rama experimental fue fusionada con `main` ni con ÓRBITA.
- Checks de la aplicación raíz en `main` (`Paso0`, `known_acc16`, `E33`, `aura-engine`, `orbita`): **NO EVALUADOS**.
