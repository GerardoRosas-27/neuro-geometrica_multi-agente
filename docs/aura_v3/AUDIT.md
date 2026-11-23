# Auditoría de Integridad Científica — AURA v3 Genesis (`exp/aura-v3-genesis`)

**Veredicto de auditoría automatizada (`freeze_and_verify_v3.py audit`):** **`AUDIT-v3 PASS`**

## 1. Integridad Criptográfica y Congelado Pre-`CONFIRM`

| Verificación | Valor / Hash | Estado |
|---|---|---|
| Commit de prerregistro (`PLAN.md` inmutable) | `fece4e6` | PASS (`git diff fece4e6..HEAD -- docs/aura_v3/PLAN.md` vacío) |
| Commit de congelado pre-`CONFIRM` | `77c7a6e` | PASS (anterior a la ejecución `CONFIRM`) |
| SHA-256 en `docs/aura_v3/FROZEN.json` (22 archivos `.rs`) | `cc90b95faa8a1885e308b81270a94babcb57b624a7323599edb8b38f0f366171` | PASS |
| SHA-256 en `docs/aura_v3/raw/DEV/run.json` | `cc90b95faa8a1885e308b81270a94babcb57b624a7323599edb8b38f0f366171` | PASS (match exacto) |
| SHA-256 en `docs/aura_v3/raw/CONFIRM/run.json` | `cc90b95faa8a1885e308b81270a94babcb57b624a7323599edb8b38f0f366171` | PASS (match exacto, `--skip-build`) |
| SHA-256 del árbol fuente actual (`aura-engine/crates/**/*.rs`) | `cc90b95faa8a1885e308b81270a94babcb57b624a7323599edb8b38f0f366171` | PASS (0 ediciones post-`CONFIRM`) |

## 2. Completitud de Datos Crudos y Sanidad Numérica

| Tabla consolidada | Filas esperadas (`DEV` + `CONFIRM`, 32 semillas) | Filas observadas | Valores `NaN`/`Inf` | Estado |
|---|---:|---:|---:|---|
| `docs/aura_v3/metrics_g1.csv` | 64 (32 × 2 brazos) | 64 | 0 | PASS |
| `docs/aura_v3/metrics_g2.csv` | 160 (32 × 5 brazos) | 160 | 0 | PASS |
| `docs/aura_v3/metrics_g3.csv` | 96 (32 × 3 brazos) | 96 | 0 | PASS |
| `docs/aura_v3/metrics_g4.csv` | 96 (32 × 3 brazos) | 96 | 0 | PASS |
| `docs/aura_v3/metrics_g5_fabric.csv` | 64 (32 × 2 brazos) | 64 | 0 | PASS |
| `docs/aura_v3/metrics_g5_bytes.csv` | 128 (32 × 4 brazos) | 128 | 0 | PASS |

## 3. Verificación de Replay Bit-Exacto (`0xB600`)

Se re-ejecutó la semilla `0xB600` con el binario congelado en un directorio temporal independiente y se comparó campo por campo con `docs/aura_v3/raw/CONFIRM/0xB600/` (`docs/aura_v3/replay.json`):

| Archivo | Campos no-temporales comparados | Diferencias | Estado |
|---|---:|---:|---|
| `g1.csv` | 18 | 0 | EXACT |
| `g2.csv` | 30 | 0 | EXACT |
| `g3.csv` | 33 | 0 | EXACT |
| `g4.csv` | 15 | 0 | EXACT |
| `g5_fabric.csv` | 14 | 0 | EXACT |
| `g5_bytes.csv` | 40 | 0 | EXACT |
| `g5_samples.txt` | archivo completo | 0 | EXACT |
| **Total** | **150 campos + muestras** | **0** | **`all_identical_excluding_timing = true`** |

(Las únicas métricas excluidas de igualdad bit-a-bit son los temporizadores de reloj de pared `throughput_tps_2048` y `train_bytes_per_sec`, dependientes de la planificación del sistema operativo.)

## 4. Pruebas Unitarias, Compilación y Controles Anti-Leakage

- **Unit tests (`docs/aura_v3/tests_release.log` y `docs/aura_v3/tests.log`):** **15/15 PASS** en `release` y **15/15 PASS** en `debug` (incluyendo los 11 tests previos de AURA v2 + 3 tests unitarios de `aura-genesis` + 1 smoke test end-to-end `end_to_end_smoke_aura_v3_genesis_g1_to_g4_exact`), con **0 warnings** del compilador `rustc 1.99.1`.
- **Cero fuga de datos en Subrutina 4 (Prenatal):**
  - En `G4`, las ondas retinianas y espasmos motores se generan 100% con física sintética (`retinal_wave_frame`, `simulate_fetal_body_echo`), sin ninguna barra visual de prueba ni ensayo postnatal. El control negativo `white_noise_prenatal` aplica permutación espacial píxel-a-píxel sobre las mismas ondas (`generate_shuffled_noise_frames`), igualando histograma y energía por frame al 100% y destruyendo solo la coherencia espacio-temporal.
  - En `G5b`, el currículo prenatal (`generate_prenatal_byte_waves`, 3,072 bytes) usa exclusivamente sílabas consonante-vocal sintéticas uniformes sobre `a..z`/`A..Z`, dígitos uniformes `0..9` y delimitadores balanceados — **0 bytes** de Markdown, Rust ni ningún archivo del repositorio — y el log-prior escalar en `l2.bias` asigna exactamente el mismo valor a las 26 letras minúsculas, a las 26 mayúsculas y a los 10 dígitos.
- **Retrocompatibilidad con AURA v2:** `myelin_delay_enabled` permanece `false` por defecto en `FabricConfig`, y todos los tests y benchmarks B1–B6 de v2 compilan y pasan intactos.
- **Aislamiento de ramas:** `main` (`34f04b2`), `exp/aura-v2-training` (`c8d5cc0`) y `docs/aura_v2/` permanecen completamente intactos.
