# Trazabilidad Experimental — AURA v2 + CorticalFabric v2 (`exp/aura-v2-training`)

## 1. Cadena de Commits y Disciplina Científica

| Fase | Commit | Descripción | Estado |
| :--- | :--- | :--- | :--- |
| Base `main` | `34f04b2` | Estado limpio de `main` (sin modificar `orbita/`, `gemma_campo/`, `soft_bellman*`, `bio_quantum_engine/`) | Intacto |
| Preregistro | `02b6348` | Commit previo al código con `docs/aura_v2/PROPUESTA.md` y `docs/aura_v2/PLAN.md` (17 criterios `H1a–H6c`) | Congelado (no editado) |
| Implementación + DEV + Freeze | *(este commit)* | Workspace 8 crates `aura-engine/`, 11/11 tests (debug + release, 0 warnings), 16 semillas `DEV` (`0xA500–0xA50F`, 17/17 PASS) y `docs/aura_v2/FROZEN.json` | Congelado antes de `CONFIRM` |
| Confirmación (`CONFIRM`) | *(pendiente siguiente commit)* | Ejecución única del binario congelado en 16 semillas frescas (`0xB500–0xB50F`), replay bit-a-bit `0xB500` y auditoría automatizada | Pendiente |

---

## 2. Entorno de Hardware y Compilación

Tomado de `docs/aura_v2/environment.json`:

- **CPU:** `Intel(R) Xeon(R) Platinum 8481C` (2 vCPUs, instrucciones AVX2 + AVX-512 + VNNI + VPOPCNTDQ; sin GPU).
- **SO / Kernel:** `Linux-6.12.105+-x86_64-with-glibc2.36`
- **Rust / Cargo:** `rustc 1.99.1 (58cf8f9c3 2026-10-13)` / `cargo 1.99.1 (5f94df478 2026-08-27)`
- **Python:** `3.11.2` (`numpy` solo para bootstrap CI95% en `summarize.py`).
- **Dependencias externas en Rust (`aura-engine/`):** **0 crates externos** (`std` + intrínsecos `std::arch::x86_64` exclusivamente).
- **Perfil de compilación:** `release` (`opt-level = 3`, `lto = "thin"`, `codegen-units = 1`).

---

## 3. Huellas Criptográficas (SHA-256)

### 3.1 Código fuente congelado (`docs/aura_v2/FROZEN.json`)

- **SHA-256 combinado (16 archivos `.rs`):** `d3f710b5372dadeecac478386316acbd0c5e61b80c9b058aa298ac56f134914d`
- **Coincidencia con `docs/aura_v2/raw/DEV/run.json`:** Exacta (`d3f710b5372dadeecac478386316acbd0c5e61b80c9b058aa298ac56f134914d`, 148.73 s para 16 semillas).

### 3.2 Corpus reales deterministas de B6 (archivos intactos de `main`)

| Archivo | Rol en B6 | Bytes | SHA-256 |
| :--- | :--- | ---: | :--- |
| `README.md` | Pre-entrenamiento / eval held-out Markdown (split disjunto 80/20 en bloques de 1 KiB) | 11,889 | `4228034c62240a9b2bf67114e8ddd9c984fb37321a4eba25e931555b339d70b4` |
| `docs/RESULTADOS_CONSOLIDADOS.md` | Pre-entrenamiento / eval held-out Markdown (24,576 B train / 6,144 B eval) | 31,769 | `d8ee8507ad7349d462272c9177f2030476183b8bb2e6dd47c03c92a69ee58c42` |
| `orbita/src/model.rs` | Dominio no visto Rust (shift online: 6,144 B totales combinados) | 6,933 | `5e0eea0adeafc52a899ab2cd5ba9400100b841b63ba2420ad34ddce1f956458e` |
| `orbita/src/fam.rs` | Dominio no visto Rust (shift online) | 8,106 | `da67bd52b8ee65207e4e460dc213869b617023ad875ff30ff6cd5bd7739264d2` |

---

## 4. Registro Transparente de Calibraciones en Fase `DEV` (Previas al Freeze)

Todas las modificaciones y calibraciones se realizaron **estrictamente durante `DEV` (`0xA500–0xA50F`)** antes de generar `FROZEN.json` y antes de ejecutar `CONFIRM`. El archivo preregistrado `docs/aura_v2/PLAN.md` (commit `02b6348`) **no fue modificado**.

1. **Nota editorial sobre conteo en `PLAN.md`:** El subtítulo de `PLAN.md` menciona "18 criterios" mientras que la lista detallada `H1a`–`H6c` enumera exactamente **17 criterios** (`H1a–c`, `H2a–c`, `H3a–c`, `H4a–b`, `H5a–c`, `H6a–c`). `summarize.py` evalúa los 17 criterios preregistrados de forma íntegra.
2. **Calibración de dimensionalidad en B3 (`bench_b3.rs`):** `PLAN.md` §4.3 esbozó originalmente una tarea en 96D con `N = 256`, régimen estadísticamente subdeterminado ($\alpha = N/D = 2.67$) donde incluso clasificadores densos en precisión infinita se saturan cerca de ~78%. Durante `DEV`, B3 se calibró a **48 dimensiones activas dentro de 64D** (`N_TRAIN = 512`, `N_TEST = 256`, maestro no lineal de 2 capas con escalas heterogéneas), preservando exactamente el **66.7% de solapamiento** entre Tarea A y Tarea B (32 dimensiones compartidas + 16 exclusivas cada una). Esto permite aislar limpiamente el techo de ~62% del ternario sin escala (`v1_naive_ternary = 61.91%`) frente a BitNet b1.58 (`72.36%`).
3. **B1 — Vectorización incondicional en `CorticalFabric` (`aura-cortex`):** Inicialmente `step_neurons_avx2` se desactivaba cuando `ne > 0` (saltando a la ruta escalar en ticks con ruido/sorpresa, `0.87×`). Al vectorizar también la modulación de noradrenalina en entero AVX2, el speedup del tejido tiled+AVX2 pasó a **5.00×**.
4. **B1 — Recíproco Q8 exacto en NLMS AVX2 (`aura-math`):** El redondeo inicial a potencia de 2 en `FastWeightMemory::update_online_nlms` estancaba la contracción en claves de escala pequeña (`scale = 2`). Se sustituyó por un paso recíproco escalado en Q8 (`(err << 20) / denom`) completamente vectorizado en AVX2, logrando contracción exacta en escalas `{2, 25, 120}`, **1.70×** más rápido que el baseline ingenuo y reduciendo el error explosivo de `3.23e10` a `2,682`.
5. **B2 — Clamping sensorial y criterio de equilibrio en `CorticalFabric` (`aura-cortex`):** El early-exit inicial (`tol_pct = 8%`) frente a la fuga de membrana (`12.5%`) lograba solo `2.40×` en ticks. Con clamping sensorial explícito (`clamp_sensory_byte`), preservación de `v_apical` en warm-start y parada por equilibrio (`fe <= ema_fe_slow + ema_fe_slow / 16`), se obtuvo **3.23×** menos ticks (`4.95` vs `16.0`) con energía libre inferior incluso a `cold_k16`.
6. **B5 — Regla monotónica BLT de pre-pico y compuerta invariante a escala (`aura-perception`, `aura-world`, `aura-runtime`):**
   - Se eliminó `end_pos % period` y `tail_ctx` cruzado entre palabras en `EntropyPatcher` (rompían invariancia traslacional en palabras aperiódicas de longitud variable).
   - El umbral anterior (`ent - prev_ema >= 1.4 && ent >= 3.2`) dejaba de detectar fronteras de palabra en cuanto el modelo causal de bytes aprendía la gramática (donde la entropía de bifurcación es ~1.4 bits frente a ~0.39 bits intra-palabra). Se implementó la regla monotónica BLT real (`ent - prev_ent >= 0.75 && ent >= 1.10`, `max_patch_len = 12`).
   - Se hizo invariante a escala la compuerta de recuperación hipocampal en `AuraAgent` (`dist * 4 < prev_norm * 3`) y se igualó la tasa de aprendizaje en regímenes estructurados entre `v1` y `v2` manteniendo únicamente la supresión de aprendizaje en estática de TV (`struct_gate < 0.18`, Propuesta §3.3). Esto redujo `p2_err` en `DEV` de `1.0145` a `0.8526` (vs `v1 = 0.9286`), preservando `static_hipp_writes = 0.0` y `ood_epi = 574.1 vs static_epi = 51.7`.
7. **B6 — Remoción de `rmsnorm_inplace` sin ganancia entrenable (`aura-decoder`, `aura-hippocampus`):** Añadir RMSNorm sin parámetro $\gamma$ antes de `ScaledTernaryLayer` desacopló la escala por fila $\alpha_r$ y degradó `eval_md_bpb` de `4.08` a `5.75` (y `shift_rust_bpb` a `11.35`). Al removerlo, `eval_md_bpb` bajó a **3.7600 BPB** (superando `bigram = 4.0428`) y `shift_rust_bpb` bajó a **5.2497 BPB** (vs `frozen = 7.8024`).

---

## 5. Pruebas Unitarias y Verificación `DEV`

- `docs/aura_v2/tests.log` (debug): **11/11 PASS**, 0 warnings.
- `docs/aura_v2/tests_release.log` (release): **11/11 PASS**, 0 warnings.
- `docs/aura_v2/summary_dev.json` (`DEV`, 16 semillas `0xA500–0xA50F`): **17/17 PASS**.

---

## 6. Ejecución `CONFIRM` y Replay

*(Se completará inmediatamente después de ejecutar `CONFIRM` una única vez sobre el binario congelado.)*
