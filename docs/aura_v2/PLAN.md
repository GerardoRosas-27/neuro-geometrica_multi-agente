# AURA v2 + CorticalFabric v2 — Plan Prerregistrado: Construcción desde Cero y Resolución de Cuellos de Botella de Entrenamiento

**Rama:** `exp/aura-v2-training` (creada limpia desde `main` @ `34f04b2`; no modifica `main`, `orbita/`, `gemma_campo/`, `soft_bellman*`, `bio_quantum_engine/` ni `exp/aura-neuromorphic-rust`).
**Fuente arquitectónica:** [`PROPUESTA.md`](PROPUESTA.md) (`https://share.gemini.google/bQhzFL1d08xX` → `https://gemini.google.com/share/f7b77dc7945c`).
**Fecha de prerregistro:** 2026-11-22 (antes de escribir código fuente o ejecutar `DEV`).

---

## 1. Objetivo Científico y de Ingeniería

Construir **desde cero** en Rust puro (cero dependencias externas, cero tokens BPE, inferencia de pesos lentos 100% en planos de bits ternarios 2-bit sin multiplicaciones de matrices) la arquitectura **AURA + CorticalFabric** descrita en `PROPUESTA.md`, **identificando y resolviendo los 6 cuellos de botella reales de entrenamiento** señalados en la propia propuesta (§4 Parte 1, §5–§6 Parte 2) y validando cada solución mediante ablaciones pareadas **Antes (cuello de botella ingenuo) vs. Después (solución v2)** sobre 16 semillas independientes en `DEV` y 16 semillas frescas congeladas en `CONFIRM`.

---

## 2. Mapa de los 6 Cuellos de Botella de Entrenamiento y sus Soluciones

| ID | Cuello de Botella (Propuesta Original) | Causa Raíz | Solución Arquitectónica en `aura-engine` (v2) |
|---|---|---|---|
| **B1** | **Silicon Mismatch & Memory-Bound Scatter** (Parte 1 §4.1, Parte 2 §4) | 1) `while pos != 0 { trailing_zeros() }` falla predicción de ramas a densidad 67% y recarga pesos en cada muestra; 2) scatter aleatorio `v_basal[dst]` rompe caché L1; 3) `FastWeightMemory::update_online` aloca `Vec` en el bucle caliente y carece de normalización NLMS. | 1) **Kernel Zero-Mul LUT-4 + AVX2 (`vpshufb`) + Tiling L1 por mini-lotes ($B=16$)** sobre los mismos planos de bits `pos_planes`/`neg_planes` (cero `IMUL`/`FMUL`, cero ramas); 2) **CSR particionado por baldosas de minicolumnas (64 neuronas)** + actualización SoA sin ramas; 3) **NLMS normalizado zero-alloc con AVX2**. |
| **B2** | **Costo de Relajación Temporal ($K=5..30$) y Secuencialidad SSM** (Parte 1 §4.2) | 1) Reiniciar `CorticalFabric` en frío cada ventana exige $K=16$ micro-pasos por dato; 2) recurrencia SSM secuencial $O(L)$ frena el entrenamiento por lotes; 3) ruido NE constante penaliza FE estacionaria. | 1) **Inicialización Predictiva Amortizada (warm-start top-down) + Early-Exit adaptativo por convergencia de FE** ($\Delta\mathcal{F}/\mathcal{F} < \epsilon$); 2) **Parallel Associative Prefix Scan (Blelloch $O(\log L)$)** exacto para forward y gradientes del SSM selectivo; 3) **NE/ACh con compuerta de sorpresa/volatilidad** (quiescente en estado estacionario). |
| **B3** | **Techo de Capacidad 2-Bit y Olvido/Oscilación en Entrenamiento Ternario** (Parte 2 §4–§5) | Pesos `{-1,0,+1}` sin escala por fila ($\alpha_r$), sin *Straight-Through Estimator* (STE), sin umbral adaptativo *AbsMean* ni protección de importancia: capacidad atascada en ~62% o conmutación masiva de bits (*thrashing*). | **Capa Ternaria Escalada estilo BitNet b1.58 (`ScaledTernaryMatrix`)**: inferencia 100% en planos de bits 2-bit zero-mul + 1 escala escalar por fila $\alpha_r$; entrenamiento con **STE + umbral dinámico AbsMean + AdamW sobre `shadow_w`** + **Consolidación de Sueño con Replay Hipocampal e Histéresis ponderada por importancia**. |
| **B4** | **Asignación de Crédito a Horizontes Largos (5,000 bytes) sin Backprop** (Parte 1 §4.3) | Traza única `e -= e >> 3` (vida media ~5 pasos) queda ciega a dependencias de 128–5,120 bytes (cerrar llave de código o recordar una premisa inicial). | **Trazas multi-escala** + **Enlace Episódico Hipocampal guiado por saliencia (`HippocampusGraph` $O(\log N)$)** que captura aperturas/premisas en un disparo, recupera en $D = 5,120$ bytes inyectando en `FastWeightMemory`, y cierra el crédito con dopamina diferida + replay. |
| **B5** | **Trampa de la Estática de TV, Inestabilidad Latente y Desalineación de Frontera** (Parte 2 §6.1) | 1) Desacuerdo bruto del ensamble $K=4$ dispara falsamente sobre ruido blanco (estática de TV); 2) parches de longitud variable sin fase de frontera pierden contra ventanas fijas en tramas periódicas; 3) colapso dimensional sin VICReg. | 1) **Descomposición Heteroscedástica Aleatoria + Progreso de Aprendizaje Epistémico** (*Learning Progress / Compression Progress*) + compuerta de entropía para rechazar estática de TV; 2) **`EntropyPatcher v2` consciente de longitud/fase y contexto de frontera**; 3) **VICReg completo** (piso de varianza + decorrelación de covarianza) con monitoreo continuo de `effective_rank`. |
| **B6** | **Entrenamiento Real Fin-a-Fin Sin Tokens + Decodificador Motor Latente-a-Byte** (Parte 2 §5–§6.2) | Sin decodificador motor entrenable ni bucle mini-lote sobre texto/código real, el núcleo latente no traduce pensamientos a bytes ni demuestra mejora real en bits-por-byte (BPB). | **Decodificador Motor Ternario Escalado (`aura-decoder`, 256 bytes, 0 tokens)** acoplado al núcleo AURA v2, entrenado fin-a-fin sobre **corpus real del repositorio** (Markdown técnico español/inglés de `main`) y evaluado bajo **cambio de dominio online real (Markdown → código Rust de `main`)**. |

---

## 3. Estructura del Workspace (8 Crates Rust, 0 Dependencias Externas)

```text
aura-engine/
├── Cargo.toml
├── README.md
├── run.py                  # Orquestación determinista DEV/CONFIRM + verificación SHA-256
├── summarize.py            # Bootstrap pareado (10,000 remuestreos, seed 12345, alpha=0.05)
├── freeze_and_verify.py    # Congelación FROZEN.json + auditoría + replay bit-exacto
└── crates/
    ├── aura-math/          # TernaryMatrix (LUT-4 AVX2 + tiling B=16), ScaledTernaryMatrix (BitNet b1.58 STE + AdamW + EWC/histéresis), FastWeightMemory (NLMS zero-alloc AVX2), LUTs, effective_rank
    ├── aura-perception/    # EntropyPatcher v2 (consciente de fase/longitud) + FixedPatcher + delta-modulación
    ├── aura-cortex/        # SelectiveSsm (Blelloch parallel prefix scan + paridad secuencial), DualCorticalCore, CorticalFabric v2 (CSR en baldosas, warm-start early-exit, NE/ACh con compuerta)
    ├── aura-world/         # EnsembleWorldModel K=4 JEPA + EMA + VICReg + cabeza aleatoria heteroscedástica + Learning Progress epistémico
    ├── aura-hippocampus/   # HippocampusGraph L1 HNSW + memoria de enlace a 5,120 bytes + SleepConsolidator v2
    ├── aura-metacog/       # MetacogController Sistema 1 / Sistema 2 (EFE H=2) / AbstainAndAsk + compuerta Learning Progress
    ├── aura-decoder/       # Decodificador motor latente-a-byte (0 tokens, 256 bytes, pesos ternarios escalados 2-bit + contexto sub-parche + adaptador online FastWeight)
    └── aura-runtime/       # AuraAgent v2 + entrenador fin-a-fin + CLI `aura-v2` (B1–B6)
```

---

## 4. Protocolo Experimental, Semillas y Criterios Prerregistrados de Éxito

- **Semillas `DEV` (desarrollo, calibración y depuración):** 16 semillas independientes `0xA500..=0xA50F`.
- **Congelación obligatoria:** Tras cerrar `DEV`, se registra el SHA-256 de todos los archivos `.rs` en `docs/aura_v2/FROZEN.json` y se realiza commit git **antes** de ejecutar `CONFIRM`.
- **Semillas `CONFIRM` (validación ciega única):** 16 semillas frescas independientes `0xB500..=0xB50F`, ejecutadas **una sola vez** con el binario congelado.
- **Verificación de determinismo:** Replay independiente de `CONFIRM/0xB500` con paridad **bit a bit exacta** (`all_identical_excluding_timing: true`).
- **Estadística:** Unidad estadística = 16 semillas independientes. Contrastes pareados dentro de cada semilla sobre exactamente los mismos datos/flujos. Bootstrap pareado de 10,000 remuestreos (semilla `12345`, $\alpha = 0.05$, IC95% percentiles interpolados). Un criterio pasa si el IC95% completo favorece la dirección prerregistrada (y cumple los umbrales numéricos cuando se indican).

### Lista Completa de Hipótesis Prerregistradas (18 criterios en B1–B6)

1. **B1 — Silicon Mismatch & Zero-Mul (`aura-math` + `aura-cortex`):**
   - **H1a (Exactitud y Memoria):** `max_abs_diff == 0` para `lut4_avx2_tiled` vs referencia exacta `i32` en todas las formas/densidades (30% y 67%, $B=1$ y $B=16$), con reducción de memoria exactamente **16×** (`f32` vs planos de bits 2-bit).
   - **H1b (Aceleración Real a Densidad BitNet 67%, `1024×1024`, $B=16$):** `lut4_avx2_tiled` supera a `naive_bitscan` (**speedup IC95 > 3.0×**) y supera a `dense_f32` (**speedup IC95 > 1.5×**).
   - **H1c (CSR en Baldosas + NLMS Zero-Alloc):** `tiled_zero_alloc_avx2` supera en throughput (ticks/s y updates/s) a `naive_scatter_alloc` (`IC95 > 1.0×`) y el error NLMS contrae monotónicamente donde el update sin normalizar diverge en claves de norma alta.

2. **B2 — Relajación Temporal & Parallel Scan (`aura-cortex`):**
   - **H2a (Aceleración de Settling sin Pérdida de FE):** `amortized_early_exit` reduce los micro-pasos por ventana en **≥ 2.5×** respecto a `naive_cold_k16` mientras mantiene FE tardía menor o igual (`Δ_FE <= 0` o no peor que `naive_cold_k16`, y estrictamente mejor que `cold_k4_truncated` y `frozen_control`, `IC95 < 0`).
   - **H2b (Resolución de Paradoja Neuromoduladora + Detección de Cambio):** `amortized_early_exit` dispara $\Delta\text{NE} > 0$ ante el cambio de régimen (`IC95 > 0`) **sin** sufrir degradación severa de FE estacionaria frente a `no_neuromod_control` (`FE_full <= FE_no_neuromod * 1.05`).
   - **H2c (Paridad Exacta del Parallel Prefix Scan):** Error máximo absoluto entre Blelloch parallel scan $O(\log L)$ y recurrencia secuencial $O(L)$ es `< 1e-5` tanto en estados forward como en gradientes backward ($L \in \{64, 256, 1024\}$).

3. **B3 — Techo de Capacidad Ternaria 2-Bit & Sueño Anti-Olvido (`aura-math` + `aura-hippocampus`):**
   - **H3a (Ruptura del Techo v1 de 62.6%):** En dos tareas solapadas 66.7% (96D) evaluadas en held-out con inferencia 100% 2-bit zero-mul y `FastWeightMemory = 0`, `v2_aura_sleep_full` supera a `v1_naive_ternary` en media combinada A+B (`IC95 > +10 pp`, rompiendo ampliamente el techo del 62.6%).
   - **H3b (Anti-Olvido por Replay):** `v2_aura_sleep_full` reduce `forgetting_A` y mejora la media combinada A+B frente a `v2_bitnet_ste_no_replay` (`IC95` favorable en ambas).
   - **H3c (Estabilidad por Histéresis de Importancia):** `v2_aura_sleep_full` reduce significativamente `weight_flips` y mejora la retención frente a `v2_bitnet_ste_no_hysteresis` (`IC95` favorable).

4. **B4 — Asignación de Crédito a 5,120 Bytes sin Backprop (`aura-hippocampus` + `aura-cortex`):**
   - **H4a (Resolución Exacta a $D = 5,120$ bytes):** En recuperación de llaves/premisas tras 5,120 bytes de distractores con recompensa diferida al final del episodio, `aura_v2_hipp_credit` supera ampliamente a `local_trace_only` y a `ssm_only_no_hipp` (`IC95 > +40 pp` vs `local_trace_only` y `IC95 > +15 pp` vs `ssm_only_no_hipp`).
   - **H4b (Compresión Episódica $O(\log N)$):** El número de entradas almacenadas en hipocampo por episodio de 5,120 bytes es estrictamente sublineal (`< 5%` de los bytes del flujo, sin KV-cache $O(L)$).

5. **B5 — Estabilidad Latente, Estática de TV y Parcheo Consciente de Fase (`aura-perception` + `aura-world`):**
   - **H5a (Resolución de la Trampa de Estática de TV):** En `aura_v2_full`, la curiosidad epistémica reducible en **OOD estructurado** supera estrictamente a la de **estática de TV** (`ood_reducible_epi − static_reducible_epi > 0`, `IC95 > 0`), invirtiendo el fallo de v1, y las escrituras hipocampales en estática son estrictamente menores que en OOD estructurado (`IC95 < 0`).
   - **H5b (Resolución de Desalineación de Frontera):** `aura_v2_full` mantiene compresión adaptativa **≥ 2.5×** (predecible vs aleatorio) y reduce el error de predicción L1 frente a `aura_v1_baseline` (`IC95 < 0`), acercándose o superando a `fixed_patcher_4` en flujo mixto (periódico + aperiódico).
   - **H5c (Anti-Colapso VICReg):** `effective_rank` en `aura_v2_full` supera estrictamente a `no_vicreg` (`IC95 > 0`).

6. **B6 — Entrenamiento Fin-a-Fin sobre Texto Real + Cambio de Dominio a Código Rust (`aura-decoder` + `aura-runtime`):**
   - **H6a (Entrenamiento Real 2-Bit > Ternario Ingenuo y > Bigrama en Held-Out Markdown):** Sobre texto real Markdown (español/inglés) de `main`, `aura_v2` entrenado fin-a-fin con pesos 2-bit escalados (sin tokens BPE) logra menor **BPB** (bits-por-byte) en test held-out que `v1_unscaled_ternary` y que el baseline estadístico `bigram_byte_baseline` (`IC95 < 0` en ambos).
   - **H6b (Adaptación Online Real: Markdown → Código Rust):** Al cambiar abruptamente a código Rust real no visto durante el pre-entrenamiento, `aura_v2_online_full` (Pesos Rápidos + Hipocampo activos) logra menor **BPB online** que `aura_v2_frozen_infer` (`IC95 < 0`).
   - **H6c (Compresión Real y Salud Latente):** `EntropyPatcher v2` logra compresión real sobre Markdown frente a bytes aleatorios (`> 1.5×`) y mantiene `effective_rank` sin colapso durante todo el entrenamiento.

---

## 5. Alcance Honesto y Límites Explícitos

- Todo se ejecuta en **CPU (Intel Xeon Platinum 8481C, 2 vCPUs, AVX2/AVX-512, 7.8 GiB RAM, sin GPU física)**.
- El modelo de B6 es un prototipo compacto sub-millón de parámetros (~150K–300K pesos ternarios, ~40–80 KiB empaquetados en 2 bits) entrenado en segundos por semilla sobre texto/código real del repositorio para demostrar matemáticamente y empíricamente el entrenamiento fin-a-fin sin tokens y la eliminación de los 6 cuellos de botella.
- **No** se afirma fluidez de LLM frontera de miles de millones de parámetros, consciencia ni reemplazo de BQ10/ATLAS-FW/Transformers. Paso 0 `1985bdbad5eee41f`, `known_acc16` y `E33` de ÓRBITA no se evalúan en esta rama.
- Cualquier criterio que falle o cualquier matiz negativo en `CONFIRM` se reportará con total transparencia en `docs/aura_v2/RESULTADOS.md`.
