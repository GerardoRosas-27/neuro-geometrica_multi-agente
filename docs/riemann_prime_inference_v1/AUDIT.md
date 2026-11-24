# Auditoría Científica y Técnica — `exp/riemann-prime-inference`

**Veredicto final:** **`AUDIT PASS`**
**Fecha UTC:** 2026-11-24T06:53:45.923318+00:00
**Rama:** `exp/riemann-prime-inference` · **HEAD:** `f4410810cf6bd4f654c4592b9aff9491cef982ff`

## Verificaciones Ejecutadas

| Comprobación | Estado | Detalle |
|---|---|---|
| FROZEN.json SHA-256 | PASS | 13/13 hashes exact |
| DEV CSVs (16 seeds) | PASS | 64/64 CSVs valid, 0 NaN/Inf |
| CONFIRM CSVs (16 seeds) | PASS | 64/64 CSVs valid, 0 NaN/Inf |
| Bit-exact replay 0xB900 | PASS | 6 files matched=True |
| Rust unit tests (debug) | PASS | tests.log: ok (0 failed) |
| Rust unit tests (release) | PASS | tests_release.log: ok (0 failed) |
| SVG Figures (4) | PASS | 4/4 present |
| Strict branch isolation | PASS | branch=exp/riemann-prime-inference, 160 touched files all inside allowed directories |

## Alcance y Checks de la App Raíz

- Experimento estrictamente aislado en `riemann_prime_inference/` y `docs/riemann_prime_inference_v1/`.
- Checks de la aplicación raíz en `main` (`Paso0`, `known_acc16`, `E33`, `aura-engine`, `orbita`): **NO EVALUADOS**.
