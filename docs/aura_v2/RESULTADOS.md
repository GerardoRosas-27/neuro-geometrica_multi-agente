# Resultados Experimentales — AURA v2 + CorticalFabric v2 (`exp/aura-v2-training`)

**Arquitectura construida desde cero en Rust puro (8 crates, 0 dependencias externas) resolviendo los 6 cuellos de botella de entrenamiento de la propuesta *Arquitectura Cognitiva Neuromórfica en Rust* (`docs/aura_v2/PROPUESTA.md`).**

- **Protocolo preregistrado:** `docs/aura_v2/PLAN.md` (commit `02b6348`, intacto).
- **Código congelado:** `docs/aura_v2/FROZEN.json` (commit `a48994f`, SHA-256 `d3f710b5372dadeecac478386316acbd0c5e61b80c9b058aa298ac56f134914d`).
- **Validación:** 16 semillas `DEV` (`0xA500–0xA50F`) + ejecución única congelada en 16 semillas frescas `CONFIRM` (`0xB500–0xB50F`), bootstrap pareado 95% (4,000 remuestreos), replay determinista bit-a-bit en `0xB500` y auditoría `AUDIT PASS`.
- **Veredicto global:** **17/17 PASS en `DEV`** y **17/17 PASS en `CONFIRM`**.

---

## 1. Resumen Ejecutivo: Antes vs Después por Cuello de Botella (`CONFIRM`, N = 16 semillas)

| Cuello de Botella | Problema Original (Propuesta / v1) | Solución AURA v2 (desde cero) | Métrica Antes (`Baseline / v1`) | Métrica Después (`AURA v2`, `CONFIRM` N=16) | Mejora (`CI95%` bootstrap pareado) | Veredicto |
| :--- | :--- | :--- | :--- | :--- | :--- | :---: |
| **B1. Desajuste con silicio + scatter** | Bit-planes 2-bit con `tzcnt` escalar y scatter CSR lento en CPU; Hebbiano sin normalizar diverge | LUT-4 AVX2 (`vpshufb`) + tiling 32 filas (`B=16`) + CSR por minicolumnas + `step_neurons_avx2` + NLMS exacto Q8 AVX2 | `12,941.5 µs` (naive) / `10,798.8 µs` (`f32`); `56.8k tps` scatter; error FW `3.23e10` | **`291.9 µs`** (`diff=0`, `16.0×` RAM); **`280.4k tps`**; error FW **`2,682`** | **`44.37×`** `[43.70, 44.94]` vs naive; **`37.02×`** `[36.49, 37.47]` vs `f32`; **`4.93×`** CSR; **`1.64×`** NLMS | **3/3 PASS** (`H1a–c`) |
| **B2. Settling $K=16$ + recurrencia secuencial + paradoja NE** | Relajación en frío $K=16$ lenta; recurrencia $O(L)$; neuromodulación v1 empeoraba FE (+30%) | Warm-start top-down + early-exit por equilibrio + NE/ACh gated por sorpresa + **Blelloch Parallel Scan** $O(\log L)$ fwd/bwd | `16.0` ticks (`32.7 ms`); `no_neuromod` FE `10,316.7` | **`4.97` ticks** (`8.8 ms`); FE **`4,443.0`** (≈ `cold16 = 4,423.7`, < `cold4 = 4,601.0`, < `frozen = 5,524.9`) | **`3.22×`** `[3.17, 3.28]` ticks; FE ratio vs `no_nm` **`0.432`** `[0.419, 0.447]`; NE $\Delta$ **`+1.31`** `[+0.93, +1.67]`; scan diff `1.79e-7 / 4.77e-7` | **3/3 PASS** (`H2a–c`) |
| **B3. Techo 2-bit (~62%) + olvido catastrófico** | Ternario ingenuo sin escala estancado en ~62%; sin replay olvida Tarea A (`29.6 pp`); bit-flips excesivos | **BitNet b1.58** (`ScaledTernaryLayer` + $\alpha_r$ + STE + AdamW + AbsMean + cosine LR) + `SleepConsolidator` v2 (replay + histéresis Fisher) | `v1`: **`61.87%`** A+B; `no_replay`: olvido **`29.62 pp`** (`69.00%`); `no_hyst`: **`102,081`** flips | **`72.57%`** A+B (`acc_A_pre = 84.30%`); olvido **`11.63 pp`**; **`32,873`** flips | **`+10.69 pp`** `[+10.25, +11.19]` vs v1; olvido **`-17.99 pp`** vs `no_replay`; **`3.11×` menos flips** y **`-2.80 pp`** olvido vs `no_hyst` | **3/3 PASS** (`H3a–c`) |
| **B4. Crédito a 5,120 bytes sin BPTT** | Trazas locales y SSM decaen exponencialmente al cruzar 2–5 KiB de distractores | Corteza multiescala + `HippocampusGraph` (HNSW L1 $O(\log N)$) gated por saliencia + inyección fast-weight + dopamina retroactiva | `local_trace`: **`5.08%`**; `ssm_only`: **`3.52%`** en $D = 5,120\text{ B}$ | **`100.00%`** en $D \in \{128, 512, 2048, 5120\}\text{ B}$ con **`0.621%`** almacenamiento | **`+94.92 pp`** `[+92.58, +96.88]` vs `local`; **`+96.48 pp`** `[+94.14, +98.44]` vs `ssm`; store `< 5%` | **2/2 PASS** (`H4a–b`) |
| **B5. Trampa de estática TV + desalineación + colapso** | Ensamble crudo premiaba ruido blanco; corte post-byte perdía fase en palabras aperiódicas; colapso sin VICReg | Descomposición **aleatoria vs epistémica reducible** (learning progress) + **BLT monotónico pre-pico** + **VICReg** | `v1`: `static_epi = 731.8`, `p2_err = 0.9297`, `p1_err = 0.9457`; `no_vicreg`: `erank = 15.44` | `ood_epi` **`580.9`** vs `static_epi` **`51.9`** (**`0.0`** writes estática vs `98.5` OOD); `p2_err` **`0.8517`** (`comp = 5.94×`); `erank` **`20.15`** | `ood - static` **`+529.0`** `[+487.7, +573.1]`; `p2_err` **`-0.0780`** vs v1 (`p1` `-0.2880`); `erank` **`+4.71`** `[+4.32, +5.06]` | **3/3 PASS** (`H5a–c`) |
| **B6. Entrenamiento real E2E en bytes + decodificador motor** | Sin decodificador latente→bytes ni evaluación en texto real/shift de dominio en v1 | `AuraByteModel` (0 tokens BPE, 12 KiB 2-bit BitNet b1.58, Blelloch SSM, adaptación online FW + Hipocampo) sobre Markdown + Rust reales del repo | Markdown BPB: `v1_unscaled` **`6.8734`**, `bigram` **`4.0428`**; Rust congelado: **`7.7848`** | Markdown held-out: **`3.7546 BPB`**; Rust shift online: **`5.2601 BPB`** (supera `bigram = 6.4045`); `comp = 1.99×`, `erank = 33.21`, **`11,995 B/s`** | **`-3.119 BPB`** vs v1; **`-0.288 BPB`** `[-0.314, -0.260]` vs bigram; **`-2.525 BPB`** `[-2.602, -2.448]` online vs congelado | **3/3 PASS** (`H6a–c`) |

---

## 2. Resultados Detallados por Benchmark (`DEV` vs `CONFIRM`)

### B1 — Silicio: LUT-4 AVX2 Zero-Mul, Tiling CSR y NLMS Exacto (`aura-math`, `aura-cortex`)

| Operación / Métrica | `DEV` (N=16) | `CONFIRM` (N=16) | `CI95%` (`CONFIRM`) |
| :--- | ---: | ---: | :--- |
| Paridad exacta entera (`max_abs_diff`) | `0` | **`0`** | Exacto (0 errores) |
| Compresión RAM (`1024×1024`, 2-bit vs `f32`) | `16.00×` | **`16.00×`** | `262,144 B` vs `4,194,304 B` |
| `256×256` (`d=67%`, `B=1`): Naive / `f32` / **LUT-4 AVX2** | `35.23 / 35.74 / 1.69 µs` | **`35.38 / 35.90 / 1.66 µs`** | `21.3×` vs naive |
| `1024×1024` (`d=30%`, `B=1`): Naive / `f32` / **LUT-4 AVX2** | `522.8 / 676.2 / 18.43 µs` | **`519.7 / 673.1 / 17.62 µs`** | `29.5×` vs naive |
| `1024×1024` (`d=67%`, `B=1`): Naive / `f32` / **LUT-4 AVX2** | `817.7 / 671.4 / 18.21 µs` | **`818.1 / 673.3 / 17.81 µs`** | `45.9×` vs naive |
| `1024×1024` (`d=67%`, `B=16`): Naive / `f32` / **LUT-4 Tiled** | `13,056 / 10,825 / 290.5 µs` | **`12,941 / 10,799 / 291.9 µs`** | **`44.37×`** `[43.70, 44.94]` vs naive; **`37.02×`** `[36.49, 37.47]` vs `f32` |
| `CorticalFabric 2048`: Scatter vs **Tiled CSR + AVX2** | `56,661 vs 283,001 tps` (`5.00×`) | **`56,845 vs 280,444 tps`** | **`4.93×`** `[4.81, 5.04]` |
| `FastWeight 256`: Naive Hebbian vs **NLMS Q8 AVX2** | `33,022 vs 55,384 ups` (`1.70×`) | **`33,733 vs 55,143 ups`** | **`1.64×`** `[1.61, 1.66]`; error `2,682` vs `32,346,466,825` |

### B2 — Settling Amortizado, Neuromodulación y Blelloch Parallel Prefix Scan (`aura-cortex`)

| Brazo (`CONFIRM`, N=16) | Ticks / ventana | Tiempo (`ms`) | Energía Libre tardía (`late_fe`) | Tasa disparo (%) | $\Delta$ NE post-shift |
| :--- | ---: | ---: | ---: | ---: | ---: |
| **`amortized_early_exit` (AURA v2)** | **`4.97`** | **`8.76`** | **`4,443.0`** | `7.27%` | **`+1.31`** `[+0.93, +1.67]` |
| `naive_cold_k16` (referencia $K=16$) | `16.00` | `32.66` | `4,423.7` | `7.34%` | `+1.89` |
| `cold_k4_truncated` (truncado $K=4$) | `4.00` | `6.06` | `4,601.0` | `6.54%` | `+4.09` |
| `frozen_control` (sin plasticidad) | `5.39` | `3.14` | `5,524.9` | `7.48%` | `-0.03` |
| `no_neuromod_control` (sin NE/ACh) | `5.62` | `13.68` | `10,316.7` | `8.57%` | `+0.37` |

- **Speedup en ticks vs `cold_k16`:** **`3.22×`** `[3.17, 3.28]` (`DEV`: `3.23×`).
- **Energía libre vs controles:** `amort - cold4 = -158.0` `[-249.5, -64.7]`; `amort - frozen = -1,081.9` `[-1,227.4, -934.2]`; `amort / no_neuromod = 0.432` `[0.419, 0.447]` (resuelve completamente la paradoja E4 de v1).
- **Exactitud del Blelloch Parallel Associative Scan ($L=256, D=64$):** `max_diff_fwd = 1.79e-7`, `max_diff_bwd = 4.77e-7` (umbral `< 1e-5`).

### B3 — Ruptura del Techo Ternario 2-Bit (BitNet b1.58) y Sueño Anti-Olvido (`aura-hippocampus`, `aura-math`)

| Brazo (`CONFIRM`, N=16) | `acc_A` pre (%) | `acc_A` post (%) | Olvido A (`pp`) | `acc_B` post (%) | **Combinada A+B (%)** | Bit-Flips (Fase B) | Densidad no-cero |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `v1_naive_ternary` (sin escala) | `64.97` | `61.27` | `3.71`* | `62.48` | `61.87` | `4,633` | `81.0%` |
| `v2_bitnet_ste_no_replay` | `84.30` | `54.68` | `29.62` | `83.32` | `69.00` | `58,569` | `53.6%` |
| `v2_bitnet_ste_no_hysteresis` | `84.30` | `69.88` | `14.43` | `75.26` | `72.57` | `102,081` | `51.6%` |
| **`v2_aura_sleep_full` (AURA v2)** | **`84.30`** | **`72.67`** | **`11.63`** | **`72.46`** | **`72.57`** | **`32,873`** | **`52.3%`** |

*\*Nota honesta: `v1_naive_ternary` tiene olvido numérico menor (`3.71 pp`) únicamente porque fracasa al aprender Tarea A inicialmente (`64.97%` frente a `84.30%` de BitNet b1.58).*

- **Ganancia sobre techo v1 (`H3a`):** **`+10.69 pp`** `[+10.25, +11.19]` (`DEV`: `+10.45 pp`).
- **Efecto del replay hipocampal (`H3b`):** olvido **`-17.99 pp`** `[-18.77, -17.23]` y combinada **`+3.57 pp`** `[+3.21, +3.93]`.
- **Efecto de histéresis por importancia Fisher (`H3c`):** **`-69,207` flips** (`3.11×` menos churning) y **`-2.80 pp`** `[-3.38, -2.25]` menos olvido en Tarea A manteniendo idéntica precisión combinada (`72.57%`).

### B4 — Crédito a Largo Horizonte de 5,120 Bytes sin Backprop (`aura-hippocampus`, `aura-cortex`)

| Brazo (`CONFIRM`, N=16) | `D = 128 B` | `D = 512 B` | `D = 2,048 B` | **`D = 5,120 B`** | Almacenamiento hipocampal |
| :--- | ---: | ---: | ---: | ---: | ---: |
| `local_trace_only` (traza local) | `7.03%` | `6.64%` | `6.25%` | `5.08%` | `0.000%` |
| `ssm_only_no_hipp` (SSM + FW sin hipocampo) | `97.27%` | `85.94%` | `9.77%` | `3.52%` | `0.000%` |
| **`aura_v2_hipp_credit` (AURA v2)** | **`100.00%`** | **`100.00%`** | **`100.00%`** | **`100.00%`** | **`0.621%`** (`< 5%`) |

- **Ventaja en $D = 5,120\text{ B}$:** **`+94.92 pp`** `[+92.58, +96.88]` vs `local_trace` y **`+96.48 pp`** `[+94.14, +98.44]` vs `ssm_only` (que cae en picada entre `512 B` y `2,048 B` por decaimiento exponencial de memoria de trabajo).

### B5 — Trampa de Estática TV, Patching BLT Pre-Pico y VICReg (`aura-perception`, `aura-world`, `aura-metacog`)

| Brazo (`CONFIRM`, N=16) | Compresión (`pred/static`) | `p1_err` | **`p2_err`** | `p4a_err` | `erank` | `fam_epi` | `static_epi` | `ood_epi` | Writes (`static / OOD`) |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `aura_v1_baseline` (post-corte + crudo) | `11.47×` | `0.9457` | `0.9297` | `0.8529` | `20.21` | `2,495.4` | `731.8` | `1,658.3` | `0.0 / 79.3` |
| `fixed_patcher_4` (control sin compresión) | `1.00×` | `0.4260` | `0.6350` | `0.4759` | `14.43` | `619.2` | `70.2` | `597.5` | `0.0 / 99.1` |
| `no_vicreg` (ablación VICReg) | `5.94×` | `0.6364` | `0.8343` | `0.5926` | `15.44` | `163.2` | `13.1` | `143.6` | `0.0 / 35.9` |
| **`aura_v2_full` (AURA v2)** | **`5.94×`** | **`0.6576`** | **`0.8517`** | **`0.5898`** | **`20.15`** | **`446.7`** | **`51.9`** | **`580.9`** | **`0.0 / 98.5`** |

- **Solución a la trampa de estática de TV (`H5a`):** En `aura_v2_full`, `ood_reducible_epi (580.9) > familiar_reducible_epi (446.7) >> static_reducible_epi (51.9)` (**11.2×** separación OOD vs estática; diferencia pareada `+529.0` `[+487.7, +573.1]`), con **`0.0`** escrituras hipocampales en ruido blanco frente a **`98.5`** en OOD estructurado (`static_aleatoric = 10,758` vs `ood_aleatoric = 3,152`).
- **Alineación BLT monotónica (`H5b`):** Compresión **`5.94×`** `[5.75, 6.14]` con reducción significativa del error frente a `aura_v1_baseline` en las tres fases estructuradas: `p1_err` **`-0.2880`** `[-0.3287, -0.2462]`, `p2_err` **`-0.0780`** `[-0.0959, -0.0620]` y `p4a_err` **`-0.2631`**.
- **Anti-colapso VICReg (`H5c`):** `erank` sube **`+4.71`** `[+4.32, +5.06]` frente a `no_vicreg` (`20.15` vs `15.44`).

### B6 — Entrenamiento End-to-End en Bytes Reales + Decodificador Motor + Shift Online Markdown → Rust (`aura-decoder`)

Corpus real determinista de `main`: `24,576 B` train Markdown / `6,144 B` held-out Markdown (bloques disjuntos de 1 KiB de `README.md` + `docs/RESULTADOS_CONSOLIDADOS.md`) → shift online sobre `6,144 B` de código Rust no visto (`orbita/src/model.rs` + `orbita/src/fam.rs`).

| Brazo (`CONFIRM`, N=16) | `train_init` BPB | `train_final` BPB | **`eval_md` BPB** | **`shift_rust` BPB** | Compresión | `erank` | Pesos 2-bit / RAM | Velocidad (`B/s`) |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `uniform_baseline` | `8.0000` | `8.0000` | `8.0000` | `8.0000` | `1.00×` | — | `0 B` | — |
| `bigram_byte_baseline` | `3.8618` | `3.8618` | `4.0428` | `6.4045` | `1.00×` | — | `262,144 B` | — |
| `v1_unscaled_ternary` | `7.7523` | `6.8113` | `6.8734` | `6.4769` | `1.99×` | `0.13` | `12,288 B` | `42,818` |
| `aura_v2_frozen_infer` (pesos lentos congelados) | `4.8255` | `3.3574` | `3.7546` | `7.7848` | `1.99×` | `33.21` | `12,288 B` | `11,995` |
| **`aura_v2_online_full` (lentos congelados + FW + Hipocampo)** | **`4.8255`** | **`3.3574`** | **`3.7546`** | **`5.2601`** | **`1.99×`** | **`33.21`** | **`12,288 B`** | **`11,995`** |

- **Generalización real en Markdown held-out (`H6a`):** **`3.7546 BPB`**, superando a `v1_unscaled_ternary` por **`-3.1188 BPB`** `[-3.1444, -3.0902]` y al baseline de bigramas de 256 KiB por **`-0.2882 BPB`** `[-0.3138, -0.2595]` usando solo **12 KiB** de pesos 2-bit (**21.3×** menos memoria que la tabla de bigramas y **16.1×** menos que `f32`).
- **Adaptación online sin reentrenar pesos lentos (`H6b`):** Al cambiar bruscamente de Markdown español a código Rust no visto, activar Fast Weights + Hipocampo reduce el error de `7.7848` a **`5.2601 BPB`** (**`-2.5246 BPB`** `[-2.6020, -2.4483]`), superando incluso al baseline de bigramas en `1.14 BPB`.
- **Muestras cualitativas textuales (sin editar, `CONFIRM/0xB500/b6_samples.txt`):**
  ```text
  PROMPT_MD: ## Resultados de 
  GEN_MD:     en Gentolidodizado ****************************************************************************
  PROMPT_RS: pub fn 
  GEN_RS:    se **** (nestíanacora sostoracontos [`docs/rafion S$ Dafisado = + Mwron GPUtrata ** Elualadod_a
  ```
  El decodificador motor de 12 KiB (0 tokens BPE, entrenado ~2 s en CPU sobre 24 KiB de texto) captura morfología de bytes, sintaxis Markdown (``[`docs/...``, `**`, `=`, `+`) y patrones en español, pero **no genera prosa coherente ni código Rust compilable** (véase §4).

---

## 3. Veredicto Completo de los 17 Criterios Preregistrados (`PLAN.md`)

| ID | Criterio Preregistrado | `DEV` (N=16) | `CONFIRM` (N=16) |
| :--- | :--- | :---: | :---: |
| `H1a` | Paridad exacta `max_abs_diff == 0` y `16.0×` compresión RAM | PASS | **PASS** |
| `H1b` | LUT-4 AVX2 `B=16` speedup `> 3×` vs naive y `> 1.5×` vs `f32` (`44.37×` / `37.02×`) | PASS | **PASS** |
| `H1c` | CSR tiled speedup `> 1×` (`4.93×`), NLMS speedup `> 1×` (`1.64×`) y menor error explosivo | PASS | **PASS** |
| `H2a` | Settling amortizado `>= 2.5×` menos ticks (`3.22×`) y menor FE que `cold_k4` y `frozen` | PASS | **PASS** |
| `H2b` | Estallido de NE ante shift (`+1.31 > 0`) y FE ratio vs `no_neuromod <= 1.05` (`0.432`) | PASS | **PASS** |
| `H2c` | Blelloch Parallel Scan paridad fwd y bwd `< 1e-5` (`1.79e-7` / `4.77e-7`) | PASS | **PASS** |
| `H3a` | BitNet b1.58 supera techo v1 en `> 10 pp` (`+10.69 pp`) | PASS | **PASS** |
| `H3b` | Replay hipocampal reduce olvido (`-17.99 pp`) y sube precisión combinada (`+3.57 pp`) | PASS | **PASS** |
| `H3c` | Histéresis por importancia reduce bit-flips (`3.11×`) y reduce olvido (`-2.80 pp`) | PASS | **PASS** |
| `H4a` | Memoria a 5,120 B supera `local_trace` (`> +40 pp`: `+94.92 pp`) y `ssm_only` (`> +15 pp`: `+96.48 pp`) | PASS | **PASS** |
| `H4b` | Almacenamiento hipocampal sublineal `< 5%` (`0.621%`) | PASS | **PASS** |
| `H5a` | `ood_reducible_epi > static_reducible_epi` (`+529.0`) y `static_writes < ood_writes` (`0.0 vs 98.5`) | PASS | **PASS** |
| `H5b` | Compresión `>= 2.5×` (`5.94×`) y `p2_err` menor que `aura_v1_baseline` (`-0.0780`) | PASS | **PASS** |
| `H5c` | VICReg eleva rango efectivo vs `no_vicreg` (`+4.71`) | PASS | **PASS** |
| `H6a` | `eval_md_bpb` supera `v1_unscaled` (`-3.12 BPB`) y `bigram` (`-0.29 BPB`) | PASS | **PASS** |
| `H6b` | `shift_rust_bpb` online supera inferencia congelada (`-2.52 BPB`) | PASS | **PASS** |
| `H6c` | Compresión en texto real `> 1.5×` (`1.99×`) y rango efectivo saludable `> 8.0` (`33.21`) | PASS | **PASS** |

---

## 4. Limitaciones, Matices y Controles Negativos (Honestidad Científica)

1. **B5 — Compromiso compresión vs error L1 frente a `fixed_patcher_4`:** Aunque `aura_v2_full` reduce drásticamente el error frente a `aura_v1_baseline` (`-30.5%` en P1, `-8.4%` en P2, `-30.9%` en P4A) manteniendo `5.94×` de compresión y `erank = 20.15`, el control fijo `fixed_patcher_4` obtiene menor error L1 bruto (`0.6350` vs `0.8517` en P2) al renunciar por completo a la compresión adaptativa (`1.00×`, `erank = 14.43`). La compresión por entropía variable sigue introduciendo una penalización de varianza sobre parches de longitud heterogénea que requiere un modelo del mundo con acondicionamiento explícito por duración de parche en futuras iteraciones.
2. **B2 — Paridad (no mejora) frente a `naive_cold_k16` en `CONFIRM`:** Mientras en `DEV` el settling amortizado quedó ligeramente por debajo de `cold_k16` (`4,362.8` vs `4,388.7`), en `CONFIRM` quedó estadísticamente empatado (`4,443.0` vs `4,423.7`, diferencia `+19.3` `CI95 [-90.1, +129.7]`, $+0.4\%$). El beneficio demostrado es reducir ticks **3.22×** (y tiempo real **3.73×**) sin pérdida estadística de precisión frente a $K=16$.
3. **B3 — Olvido mitigado, no eliminado, y calibración de dimensionalidad `DEV`:** El olvido de Tarea A baja de `29.62 pp` a `11.63 pp`, pero no es cero. Adicionalmente, la dimensionalidad se calibró en `DEV` de 96D subdeterminado a 48D activas en 64D (manteniendo el 66.7% exacto de solapamiento) para separar el techo ternario de la subdeterminación muestral (ver `TRAZABILIDAD.md` §4.2).
4. **B4 — Detección de saliencia estructural sintáctica:** La puerta de escritura hipocampal en B4 utiliza una señal estructural explícita (`@Kx=Vyy`) sobre un stream con 5,120 bytes de distractores; detectar automáticamente qué hechos arbitrarios de lenguaje natural merecen consolidación episódica sin marcadores sintácticos sigue siendo un problema abierto.
5. **B6 — Fluidez limitada de un modelo compacto de 12 KiB:** Un decodificador de 12 KiB entrenado sobre 24 KiB de Markdown en ~2 s de CPU produce mejoras sólidas en BPB frente a bigramas y adaptación online real ante cambio de dominio, pero su generación cruda (`GEN_MD` / `GEN_RS`) muestra pseudo-palabras y bucles de repetición (`****`). Tampoco es un reemplazo de Transformers/LLMs de frontera ni ha sido evaluado en los benchmarks de ÓRBITA Paso 0 / E33 ni en GPU (WGSL/CUDA).
6. **B6 — Nota sobre `v1_unscaled_ternary` en shift a Rust:** `v1_unscaled_ternary` obtuvo `6.48 BPB` en Rust vs `7.78 BPB` de `aura_v2_frozen_infer` porque el modelo v1 nunca aprendió bien Markdown (`6.87 BPB`, distribución casi uniforme con `erank = 0.13`, menos sobre-especializado), mientras que `aura_v2_frozen_infer` sí se especializó en Markdown (`3.75 BPB`). Al activar el bucle online completo (`aura_v2_online_full`), AURA v2 supera a todos los baselines en Rust (**`5.26 BPB`**).
