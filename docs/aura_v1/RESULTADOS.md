# AURA + CorticalFabric v1 — Resultados Experimentales Reales (DEV + CONFIRM)

**Fecha de cierre:** 2026-11-20 (UTC-8) / 2026-11-21 (UTC).
**Rama:** `exp/aura-neuromorphic-rust` (aislada de `main`; no modifica `orbita/`, `gemma_campo/`, `soft_bellman*` ni `agentic_web`).
**Fuente implementada:** [`PROPUESTA.md`](PROPUESTA.md) (`https://gemini.google.com/share/bb4ada4c377a`, enlace corto `https://share.gemini.google/PlhXdw38cTyQ`).

---

## 1. Veredicto Global

**13/13 criterios prerregistrados (`H1a`, `H1b`, `H1c`, `H1d`, `H2a×2`, `H2b`, `H3a×2`, `H3b`, `H4a`, `H4b`, `E5`) PASAN en CONFIRM** sobre 16 semillas frescas independientes (`0xB400–0xB40F`) ejecutadas una sola vez con binario congelado, **con cuatro matices y negativos exploratorios importantes documentados con total honestidad en §7**.

Este experimento construye y verifica en Rust puro (workspace modular de 7 crates `aura-engine/`, cero dependencias externas, cero tokens BPE, cero multiplicaciones de matrices flotantes en inferencia ternaria y en el núcleo event-driven) tanto la arquitectura **AURA** (*Adaptive Unified Recurrent-Associative Core*) como el núcleo neuromórfico **`CorticalFabric`** ($U(1)$ + Energía Libre de Laplace + álgebra tropical + 3 factores + NE/ACh/DA). **No es un LLM de lenguaje natural fluido ni reemplaza a BQ10/ATLAS-FW/Transformers**; valida las Etapas 1 y 2 propuestas en §5 del documento fuente sobre bancos controlados cerrados.

---

## 2. Trazabilidad y Protocolo Científico

- **Prerregistro ANTES de implementación y DEV:** commit `fa31fd1e44777822704fa9e11ac22c9b05b2078b` ([`PLAN.md`](PLAN.md), [`PROPUESTA.md`](PROPUESTA.md)).
- **Implementación + DEV + Congelación ANTES de CONFIRM:** commit `0b950d070ce19fab52d8313fc0d8e3dc068d8776` ([`FROZEN.json`](FROZEN.json)).
- **Huella SHA-256 verificada de todos los fuentes `.rs`:** `1e9b7f0f2eafe4b5d7e8281f88867aaa3d0f14c7a7e62ac14da5e869bb97ad64` (idéntica en `DEV/run.json`, `FROZEN.json` y `CONFIRM/run.json`; cero cambios después de DEV).
- **Semillas:** `DEV` = 16 semillas independientes `0xA400..=0xA40F`; `CONFIRM` = 16 semillas frescas independientes `0xB400..=0xB40F`.
- **Estadística:** Unidad estadística = las 16 semillas independientes por fase (contrastes pareados dentro de cada semilla sobre los mismos flujos/entornos exactos). Bootstrap pareado de 10,000 remuestreos, semilla `12345`, $\alpha = 0.05$, cuantiles interpolados (`aura-engine/summarize.py`).
- **Pruebas unitarias y determinismo:** 15/15 pruebas unitarias en `debug` ([`tests.log`](tests.log)) y `release` ([`tests_release.log`](tests_release.log)); replay independiente de `CONFIRM/0xB400` **100% idéntico bit a bit** en los 5 archivos de métricas tras excluir únicamente columnas de reloj ([`replay.json`](replay.json)).
- **Tiempo de pared real:** DEV (16 semillas × 4 experimentos + E5, 2 workers, `nice -n 10`) = **40.38 s**; CONFIRM = **40.42 s** (CPU Intel Xeon Platinum 8481C, 2 vCPUs, `rustc 1.100.0`).

---

## 3. Experimento E1 (Etapa 1A): Flujo Crudo de Bytes Sin Tokens, Cambio de Regla, Hipocampo y Anti-Colapso VICReg

Flujo continuo de 2,400 bytes crudos por semilla en 4 fases (Régimen A → cambio abrupto a Régimen B → estática aleatoria de TV → retorno a Régimen A + Régimen C OOD inédito). Todos los brazos se evalúan pareados sobre exactamente el mismo flujo por semilla.

### Tabla E1 — CONFIRM (`0xB400–0xB40F`, medias sobre 16 semillas)

| Brazo | Parches totales | Ratio compresión (pred / alta entropía) | Error L1 norm Fase 1 | Error L1 norm post-cambio Fase 2 (↓) | Error L1 norm retorno Fase 4A (↓) | `effective_rank` (§6) | Norma latente media | Epistémica familiar / OOD |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **`aura_full`** | 945.4 | **4.074×** (`4.11` vs `1.01` B/parche) | 0.5185 | **0.9353** | **0.5344** | **27.39** | **21.14** | 1,666 / **2,453** |
| `no_fast_weights` | 945.4 | 4.074× | 0.5185 | 1.1249 | 0.5964 | 28.26 | 20.76 | 1,666 / 4,969 |
| `no_hippocampus` | 945.4 | 4.074× | 0.5185 | 0.9353 | 0.7157 | 27.63 | 21.09 | 1,666 / 2,347 |
| `fixed_patcher` (ventana 4) | 600.0 | 1.000× (`4.00` vs `4.00` B/parche) | 0.3908* | 0.7204* | 0.4663* | 15.89 | 23.77 | 1,321 / 1,724 |
| `no_vicreg` | 945.4 | 4.074× | 0.5076 | 0.9300 | 0.5403 | 24.14 | 16.53 | 1,249 / 2,164 |
| `persistence` | 945.4 | 4.074× | 1.5977 | 1.3650 | 1.5092 | 31.75 | 19.56 | 0 / 0 |

### Contrastes Pareados E1 (DEV vs CONFIRM, bootstrap 10,000)

| Hipótesis / Contraste | DEV Δ [IC95%] | CONFIRM Δ [IC95%] | Veredicto |
|---|---|---|---|
| **H1a (Primario):** Fase 2 `aura_full − no_fast_weights` (↓) | −0.1922 [−0.2199, −0.1668] | **−0.1896 [−0.2101, −0.1684]** | **PASA** |
| **H1a ref:** Fase 2 `aura_full − persistence` (↓) | −0.4763 [−0.5162, −0.4440] | **−0.4297 [−0.4517, −0.4095]** | **PASA** |
| **H1b (Recall hipocampal):** Fase 4A `aura_full − no_hippocampus` (↓) | −0.1848 [−0.2106, −0.1548] | **−0.1813 [−0.2070, −0.1552]** | **PASA** |
| **H1c (Anti-colapso rango):** `erank` `aura_full − no_vicreg` (↑) | +3.19 [+2.98, +3.42] | **+3.25 [+3.03, +3.47]** | **PASA** |
| **H1c (Anti-colapso norma):** `mean_latent_norm` `aura_full − no_vicreg` (↑) | +4.49 [+4.35, +4.65] | **+4.62 [+4.43, +4.84]** | **PASA** |
| **H1d (Compresión sin tokens):** `aura_full − fixed_patcher` (↑) | +3.12 [+3.02, +3.25] | **+3.07 [+2.99, +3.18]** | **PASA** |
| **H1e (Epistémica OOD vs familiar):** `ood − familiar` en `aura_full` | +757.4 [+670.7, +839.7] | **+786.5 [+710.5, +862.9]** | **PASA** |
| **H1e (Coherencia direccional):** `shift − static` en `aura_full` | +1.50 pp [+0.30, +2.59] | **+1.12 pp [+0.19, +2.03]** | **PASA (margen modesto)** |

\* *Matiz honesto sobre `fixed_patcher`: ver §7.1 (alineación de fase con tramas exactas de 12 bytes = 3×4).*

---

## 4. Experimento E2 (Etapa 1B): Mundo Activo 2D con Cambio de Ley Física y Control Metacognitivo Dual

Mundo discreto $6 \times 6$ codificado en tramas sensoriales de 256D: Fase 1 bajo Ley $L_1$ (24 episodios), Fase 2 tras permutación abrupta de acciones $L_1 \to L_2$ (24 episodios), y Fase 3 con 8 estados familiares + 8 estados trampa OOD inéditos donde una acción ciega errónea penaliza `-10` y `AbstainAndAsk` cuesta `-1`.

### Tabla E2 — CONFIRM (`0xB400–0xB40F`, medias sobre 16 semillas)

| Brazo | Retorno Fase 1 | Éxito Fase 1 | Retorno post-cambio Fase 2 | Éxito Fase 2 (↑) | Retorno OOD Fase 3 | Retorno Combinado F2+F3 (↑) | Abstención Trampa OOD | Abstención Familiar | Fracción Sistema 1 (F1) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **`aura_full`** | 10.43 | 90.6% | **+5.32** | **82.8%** | **+1.89** | **+7.21** | **100.0%** | **0.8%** | 27.7% |
| `system1_only` | 10.61 | 89.8% | −29.09 | 0.0% | −2.56 | −31.64 | 0.0% | 0.0% | 100.0% |
| `no_fast_weights` | 10.43 | 90.6% | −18.09 | 0.0% | −0.89 | −18.97 | 100.0% | 84.4% | 27.7% |
| `random_control` | −19.47 | 2.3% | −19.87 | 1.0% | −2.46 | −22.33 | 0.0% | 0.0% | 100.0% |

### Contrastes Pareados E2 (DEV vs CONFIRM)

| Hipótesis / Contraste | DEV Δ [IC95%] | CONFIRM Δ [IC95%] | Veredicto |
|---|---|---|---|
| **H2a (Primario vs Sistema 1):** `combined` `aura_full − system1_only` (↑) | +41.11 [+38.42, +43.85] | **+38.86 [+36.32, +40.89]** | **PASA** |
| **H2a (Primario vs congelado):** `combined` `aura_full − no_fast_weights` (↑) | +25.95 [+22.70, +29.04] | **+26.18 [+24.01, +27.97]** | **PASA** |
| **H2a (Éxito post-cambio):** `p2_succ` `aura_full − system1_only` (↑) | +81.25 pp [+72.66, +89.32] | **+82.81 pp [+76.30, +88.28]** | **PASA** |
| **H2b (Calibración Abstain):** `trap − familiar` en `aura_full` (↑) | +99.22 pp [+97.66, +100.0] | **+99.22 pp [+97.66, +100.0]** | **PASA** |

---

## 5. Experimento E3 (Etapa 2): Validación del Ciclo de Sueño (Anti-Olvido en Planos de Bits Ternarios)

Tarea A (`dims [0..96)`) consolidada primero en `TernaryMatrix` (`16×256`, 2 bits/peso) y 96 episodios comprimidos guardados en `HippocampusGraph`. Luego se expone **únicamente** Tarea B (`dims [32..128)`, **66.7% de solapamiento de entrada** = interferencia real). Evaluación final sobre conjuntos disjuntos `Test_A` (128) y `Test_B` (128) usando `project_zero_mul` puro con **`FastWeightMemory = 0`**. Todos los brazos parten del mismo checkpoint exacto post-Tarea A.

### Tabla E3 — CONFIRM (`0xB400–0xB40F`, medias sobre 16 semillas)

| Brazo | `acc_A_pre` | `acc_A_post` (↑) | Olvido `A_pre − A_post` (↓) | `acc_B_post` (↑) | Media Combinada A+B (↑) | Conmutaciones de bit (`weight_flips`) | Densidad no nula (`nonzero_frac`) |
|---|---:|---:|---:|---:|---:|---:|---:|
| **`aura_sleep_full` (Replay + Histéresis)** | 66.15% | **61.65%** | **4.50 pp** | 63.47% | **62.56%** | **27,787** | **24.5%** |
| `ablation_no_replay` (Sin Replay, con Histéresis) | 66.15% | 55.05% | 11.11 pp | **66.46%** | 60.75% | 26,474 | 25.2% |
| `ablation_no_hysteresis` (Con Replay, sin Histéresis) | 66.15% | 53.12% | 13.03 pp | 52.55% | 52.83% | 426,043 | 50.0% |
| `ablation_naive` (Sin Replay ni Histéresis) | 66.15% | 54.22% | 11.93 pp | 53.96% | 54.09% | 418,591 | 43.9% |

### Contrastes Pareados E3 (DEV vs CONFIRM)

| Hipótesis / Contraste | DEV Δ [IC95%] | CONFIRM Δ [IC95%] | Veredicto |
|---|---|---|---|
| **H3a (Primario combinado):** `aura_sleep_full − ablation_no_replay` (↑) | +1.78 pp [+1.37, +2.19] | **+1.81 pp [+1.21, +2.37]** | **PASA** |
| **H3a (Primario anti-olvido):** `forgetting_A` `full − no_replay` (↓) | −6.42 pp [−7.21, −5.69] | **−6.60 pp [−7.46, −5.78]** | **PASA** (olvido reducido 59.5%) |
| **H3b (Contribución histéresis):** `combined` `full − no_hysteresis` (↑) | +9.59 pp [+9.12, +10.02] | **+9.73 pp [+9.21, +10.28]** | **PASA** (**15.3× menos flips**) |
| **H3b (Anti-olvido histéresis):** `forgetting_A` `full − no_hysteresis` (↓) | −8.26 pp [−9.09, −7.35] | **−8.53 pp [−9.39, −7.63]** | **PASA** |

---

## 6. Experimento E4 (Parte 1) y Verificación E5: `CorticalFabric` Event-Driven y Planos de Bits Zero-Mul

### Tabla E4 — `CorticalFabric` CONFIRM (`0xB400–0xB40F`, 256 neuronas Gamma/Beta/Theta, 1,200 ticks, 0 floats en `step()`)

| Brazo | FE tardía media (↓) | FE sensorial tardía (↓) | Tasa disparo cortical (%) | NE pre-cambio | NE post-cambio | Δ NE al cambio (↑) | Throughput (ticks/s) |
|---|---:|---:|---:|---:|---:|---:|---:|
| **`cortical_full`** | **9,326.0** | **3,198.0** | **6.01%** | 3.06 | **9.46** | **+6.40** | 101,083 |
| `cortical_frozen` (sin plasticidad 3-factores) | 11,094.2 | 4,160.0 | 6.56% | 6.23 | 7.37 | +1.14 | 125,680 |
| `cortical_no_phase_gate` (sin galga $U(1)$) | 10,300.5 | 2,982.0 | 6.45% | 2.39 | 8.98 | +6.58 | 122,598 |
| `cortical_no_neuromod` (NE/ACh/DA fijos)* | 7,161.2* | 2,000.2* | 4.27%* | 3.48 | 11.19 | +7.71 | 137,203 |

### Contrastes Pareados E4 (DEV vs CONFIRM)

| Hipótesis / Contraste | DEV Δ [IC95%] | CONFIRM Δ [IC95%] | Veredicto |
|---|---|---|---|
| **H4a (Primario):** `fe_late` `cortical_full − cortical_frozen` (↓) | −2,370.4 [−2,701.4, −2,023.7] | **−1,768.3 [−2,184.8, −1,353.7]** | **PASA (−15.9% FE)** |
| **H4a sensorial:** `sensory_fe` `cortical_full − cortical_frozen` (↓) | −1,185.7 [−1,339.9, −1,018.4] | **−962.0 [−1,117.9, −799.0]** | **PASA (−23.1% FE)** |
| **H4b (Pico NE):** `ne_shift_delta` en `cortical_full` (↑) | +7.29 [+3.93, +10.02] | **+6.40 [+3.44, +9.10]** | **PASA** |
| **H4c (Compuerta $U(1)$):** `fe_late` `full − no_phase_gate` (↓) | −1,236.4 [−1,493.4, −986.9] | **−974.6 [−1,185.7, −747.9]** | **PASA (−9.5% FE extra)** |
| **H4c (Neuromodulación — negativo exploratorio):** `full − no_neuromod` | +1,839.6 [+1,568.9, +2,119.7] | **+2,164.8 [+1,875.3, +2,454.6]** | **NEGATIVO (ver §7.4)** |

### Verificación de Ingeniería E5 (`TernaryMatrix::project_zero_mul`, 30% densidad, CPU Xeon)

| Métrica | DEV (16 semillas) | CONFIRM (16 semillas) | Bench dedicado `0xB400` (`bench_e5_standalone.log`) |
|---|---:|---:|---:|
| Error máximo absoluto vs matvec `i32` | **0** (exacto) | **0** (exacto) | **0** (exacto) |
| Reducción de memoria (`f32 / packed 2-bit`) | **16×** (`256 KiB` vs `4 MiB`) | **16×** (`256 KiB` vs `4 MiB`) | **16×** (`262,144 B` vs `4,194,304 B`) |
| Latencia `256×256`: `zero-mul` vs `f32` denso | **11.97 µs** vs 35.37 µs (**2.95×**) | **12.33 µs** vs 35.43 µs (**2.87×**) | **12.56 µs** vs 36.01 µs (**2.87×**) |
| Latencia `1024×1024`: `zero-mul` vs `f32` denso | **538.86 µs** vs 705.38 µs (**1.31×**) | **513.89 µs** vs 686.14 µs (**1.34×**) | **516.42 µs** vs 668.83 µs (**1.30×**) |

---

## 7. Lectura Crítica Honesta: Negativos, Matices y Límites Reales

1. **Alineación de fase de `fixed_patcher` en E1:** Aunque el *Entropy Patcher* logra **4.07×** de compresión adaptativa (4.11 B/parche predecible vs 1.01 B/parche aleatorio) y mayor riqueza representacional (`effective_rank 27.39` vs `15.89`), el control `fixed_patcher` de ventana fija 4 obtiene menor error L1 en Fase 1 y Fase 2 (`0.391 / 0.720` vs `0.519 / 0.935`) porque las tramas sintéticas tienen longitud exacta 12 (= 3×4), sincronizando artificialmente los cortes fijos con el protocolo. Esto confirma que la segmentación por entropía introduce variabilidad de frontera que dificulta la predicción a un paso cuando el flujo tiene periodicidad fija.
2. **Problema de la estática de TV en E1 (`static_epistemic` vs `ood_epistemic`):** Aunque la coherencia direccional separa el cambio de regla de la estática aleatoria (+1.12 pp, IC95 > 0) y `ood_epistemic` supera claramente a `familiar_epistemic` (+786.5), `static_epistemic` (`2,956`) sigue siendo mayor que `ood_epistemic` (`2,453`) porque en solo 300 bytes de ruido uniforme cada parche de 1 byte produce una clave nueva nunca repetida (las 4 cabezas no alcanzan a promediar la distribución aleatoria en una ventana corta).
3. **Techo de capacidad absoluta de pesos ternarios 2-bit en E3:** Aunque el replay hipocampal y la histéresis reducen el olvido un **59.5%** y evitan el colapso por conmutación de bits (15.3× menos flips), la precisión absoluta en held-out sobre dos tareas binarias solapadas de 96D queda en **62.56%** (`acc_A_pre = 66.15%`). Los planos de bits ternarios sin multiplicación son muy compactos (16× menos RAM) y rápidos, pero con reglas lineales solapadas de alta dimensión presentan menor precisión absoluta que pesos continuos en punto flotante.
4. **Negativo exploratorio en E4 (`cortical_full` vs `cortical_no_neuromod`):** Desactivar la modulación NE/ACh/DA sobre el drive (`cortical_no_neuromod`) logra una Energía Libre bruta menor (`7,161` vs `9,326`) a costa de reducir la actividad cortical de `6.01%` a `4.27%`. El bucle neuromodulador introduce un costo real de exploración (jitter de fase + mayor excitabilidad basal) que solo se compensa en tareas con cambios de distribución activos como E2, no en reducción pasiva de energía estacionaria.
5. **Alcance estricto:** Benchmarks sintéticos cerrados en CPU (2 vCPUs sin GPU física). No se afirma comprensión de lenguaje natural, consciencia ni reemplazo de BQ10, ATLAS-FW o LLMs modernos. Paso 0 `1985bdbad5eee41f`, `known_acc16` y `E33` de ÓRBITA **NO fueron evaluados** en esta rama.

---

## 8. Evidencia y Reproducción

- Código fuente (7 crates): [`aura-engine/`](../../aura-engine/)
- Prerregistro y propuesta: [`PLAN.md`](PLAN.md), [`PROPUESTA.md`](PROPUESTA.md), [`TRAZABILIDAD.md`](TRAZABILIDAD.md)
- Congelación y auditoría: [`FROZEN.json`](FROZEN.json), [`AUDIT.md`](AUDIT.md), [`replay.json`](replay.json), [`environment.json`](environment.json)
- Pruebas y métricas: [`tests.log`](tests.log), [`tests_release.log`](tests_release.log), [`summary.json`](summary.json), [`analysis.log`](analysis.log), [`metrics_e1.csv`](metrics_e1.csv)–[`metrics_e4.csv`](metrics_e4.csv), [`bench_e5_standalone.log`](bench_e5_standalone.log), y evidencia cruda por semilla en `raw/{DEV,CONFIRM}/`.
