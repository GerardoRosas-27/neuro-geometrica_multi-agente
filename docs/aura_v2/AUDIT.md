# Auditoría de Integridad Experimental — AURA v2 (`exp/aura-v2-training`)

Ejecutada mediante `python3 aura-engine/freeze_and_verify.py audit` y verificación manual del historial Git y artefactos crudos.

## 1. Resumen del Veredicto de Auditoría

> **VEREDICTO: `AUDIT PASS` — Cero anomalías de integridad, cero fugas de datos, cero ediciones post-freeze.**

| Comprobación | Resultado | Evidencia |
| :--- | :---: | :--- |
| Preregistro previo al código | **PASS** | `docs/aura_v2/PLAN.md` y `docs/aura_v2/PROPUESTA.md` fijados en commit `02b6348` antes de crear `aura-engine/`; `PLAN.md` intacto desde `02b6348`. |
| Separación estricta `DEV` vs `CONFIRM` | **PASS** | `DEV`: `0xA500–0xA50F` (16 semillas); `CONFIRM`: `0xB500–0xB50F` (16 semillas disjuntas frescas). |
| Freeze SHA-256 previo a `CONFIRM` | **PASS** | `docs/aura_v2/FROZEN.json` fijado en commit `a48994f` antes de ejecutar `CONFIRM` con `--skip-build`. |
| Identidad criptográfica del código (16 `.rs`) | **PASS** | SHA-256 = `d3f710b5372dadeecac478386316acbd0c5e61b80c9b058aa298ac56f134914d` idéntico en `FROZEN.json`, `raw/DEV/run.json`, `raw/CONFIRM/run.json` y árbol actual. |
| Conteo exacto de filas consolidadas (`DEV + CONFIRM`) | **PASS** | `b1`: 32 (32×1), `b2`: 160 (32×5), `b3`: 128 (32×4), `b4`: 96 (32×3), `b5`: 128 (32×4), `b6`: 160 (32×5). |
| Integridad numérica (0 NaN / 0 Inf) | **PASS** | 0 valores `NaN`, `Inf` o `-Inf` en las 704 filas de los 6 CSV consolidados (`metrics_b1.csv`–`metrics_b6.csv`). |
| Replay determinista bit-a-bit (`0xB500`) | **PASS** | `docs/aura_v2/replay.json`: `all_identical_excluding_timing = true` en los 250 campos no-temporales (`b1`: 7, `b2`: 60, `b3`: 36, `b4`: 24, `b5`: 68, `b6`: 55) + `b6_samples.txt` idéntico byte a byte. |
| Pruebas unitarias (debug + release) | **PASS** | 11/11 PASS, 0 warnings en `docs/aura_v2/tests.log` y `docs/aura_v2/tests_release.log`. |
| Cero dependencias externas en Rust | **PASS** | `aura-engine/Cargo.toml` y `aura-engine/Cargo.lock`: únicamente los 8 crates internos del workspace (`std` + intrínsecos `std::arch::x86_64`). |
| Aislamiento de rama y código ajeno | **PASS** | Rama `exp/aura-v2-training` desde `main@34f04b2`; cero modificaciones en `main`, `exp/aura-neuromorphic-rust`, `orbita/`, `gemma_campo/`, `soft_bellman*`, `bio_quantum_engine/`. |
| Transparencia de calibraciones `DEV` | **PASS** | Calibración de B3 (48D en 64D manteniendo 66.7% solapamiento) y nota editorial de conteo (17 criterios `H1a–H6c`) documentadas explícitamente en `TRAZABILIDAD.md` y `RESULTADOS.md`. |

## 2. Verificación de Replay (`docs/aura_v2/replay.json`)

```json
{
  "seed": "0xB500",
  "all_identical_excluding_timing": true,
  "files": {
    "b1.csv": { "identical": true, "fields_checked": 7, "diffs": [] },
    "b2.csv": { "identical": true, "fields_checked": 60, "diffs": [] },
    "b3.csv": { "identical": true, "fields_checked": 36, "diffs": [] },
    "b4.csv": { "identical": true, "fields_checked": 24, "diffs": [] },
    "b5.csv": { "identical": true, "fields_checked": 68, "diffs": [] },
    "b6.csv": { "identical": true, "fields_checked": 55, "diffs": [] },
    "b6_samples.txt": { "identical": true }
  }
}
```

## 3. Verificación de Paridad de Salida de Microbenchmark Aislado (`B1`)

`docs/aura_v2/bench_b1_standalone.log` (`aura-v2 b1 0xB500`):
- `max_abs_diff = 0` (paridad entera exacta entre `TernaryMatrix::project_lut4_simd`, `NaiveBitplaneMatrix` y `DenseF32Matrix`).
- `mem_ratio_1024 = 16.0x` (`262,144 B` bit-planes 2-bit vs `4,194,304 B` f32).
- `1024×1024 B=16 d=67%`: `290.79 µs` (`lut4_tiled`) vs `13,957.36 µs` (`naive`, **48.00×**) y `10,793.02 µs` (`f32`, **37.12×**).
- `CorticalFabric 2048`: `262,030 tps` (`tiled+AVX2`) vs `57,232 tps` (`scatter`, **4.58×**).
- `FastWeight 256 NLMS`: `55,210 ups` (`nlms_avx2`, error `2,682`) vs `32,802 ups` (`naive`, error `32,346,466,825`, **1.68×**).
