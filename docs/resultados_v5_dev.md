# Resultados v5 DEV — campo predictivo

**Rama:** `exp/predictive-field-v5`  
**Fecha corrida:** 2026-10-14  
**SHA corrida:** `a0dbb169` (tip rama al documentar: `19bf9ac`)  
**Artefactos:** `artifacts/v5/dev/`

## Veredicto

`R_P2_gated` **4/16 PASS** → **FAIL** (pre-registro exige ≥ 9/16).  
`R_E3_v4` (misma fase, mismas seeds) **13/16**.  
P2 no supera a E3. **P3 no aplica.**

| Gate | PASS | FAIL | n |
|---|---|---|---|
| R_E3_v4 | 13 | 3 | 16 |
| R_P2_gated | 4 | 12 | 16 |
| R_P2_gated_eqc | 0 | 16 | 16 |
| R_P2_mlp | 4 | 12 | 16 |
| R_P2_shuffled | 0 | 16 | 16 |

Seeds PASS en `R_P2_gated`: `0xA401`, `0xA403`, `0xA404`, `0xA40B`.

CONFIRM incompleto (`artifacts/v5/confirm/`, sin summary). Detalle por seed en `artifacts/v5/dev/summary.md`.
