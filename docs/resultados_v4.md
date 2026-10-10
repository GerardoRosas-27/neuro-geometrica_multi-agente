# Resultados v4 — E31–E44 (DEV + CONFIRM)

**Rama:** `exp/field-autonomy-v4` (no mergeada) · **Fecha:** 2026-10-09
**Código:** `5892867` (harness), `5c558fe` (fmt). DEV corrido en `5c558fe`, CONFIRM en `456c77d` (sólo añade artefactos DEV; mismo código).
**Pre-registro:** `docs/preregistro_v4.md` (commiteado antes de DEV). **Lock hash:** `9578f730…4647` (idéntico en DEV y CONFIRM).
**Seeds:** DEV `0xA400–0xA40F`, CONFIRM `0xB400–0xB40F` (16+16). Sin retune entre fases.
**Artefactos:** `artifacts/v4/{dev,confirm}/` → `config.json`, `dataset_manifest.jsonl` (16 manifests CLEAN), `provenance.jsonl.gz` (una línea por predicción E35 A/B, con `provenance_hash`), `metrics.csv`, `summary.md`, `paired_stats.txt`, `e32_trace.csv`, `checkpoints/` (Dφ_B bit-exact + CDT serializado). Diagnóstico E23: `artifacts/v4/E23_diag/trace.csv`. Teacher post-hoc: `artifacts/v4/teacher_diag/`.

Reproducir: `cargo run --release --bin run_field_autonomy_v4 -- run --phase dev|confirm` · `python3 scripts/v4_analyze.py artifacts/v4/<phase>/metrics.csv`.

## 1. Tabla de gates (PASS por seed / 16)

| Gate | DEV | CONFIRM | Estado |
|---|---|---|---|
| E31 rule learning | 16 | 16 | **PASS** |
| E32 stability (σ_max) | 16 | 16 | **PASS** (diagnóstico: ver §4) |
| E33 long rollout | 0 | 0 | **FAIL** |
| E34 static/dynamic | 5 | 6 | **PARTIAL** (Dφ ≈ MLP de igual capacidad) |
| E35 CDT experience | 11 | 10 | **PASS** (con la salvedad crítica de §3) |
| E36 consolidation ablation | 10 | 13 | **PASS** |
| E37 delete-after-learning | 11 | 10 | **PASS** |
| E38 persistence | 11 | 10 | **PASS** |
| E39 transfer | 13 | 10 | **PASS** |
| E40 continual learning | 16 | 16 | **PASS** |
| E41 causal intervention | 16 | 16 | **PASS** |
| E42 scaling | 4 | 5 | **FAIL/PARTIAL** por gate estricto (todas las N por seed); IC pareado >0 en las 5 N y ambas fases |
| E43 language | — | — | **NOT_RUN** (sin periferia LLM en este harness) |
| E44 LLM swap | — | — | **NOT_RUN** |

Leakage: 0 predicciones LEAKED y 0 queries de memoria (cdt/rqm/table/nn/attractor/direct) en TEST de E35/E36/E39/E42, en ambas fases. 32/32 manifests `CLEAN`.

## 2. E35 — A vs B (accuracy TEST, rel_err<5 %)

| Fase | mean A | mean B | B−A | mediana | IC95 bootstrap | d_z | seeds B>A | PASS (≥+0.05) |
|---|---|---|---|---|---|---|---|---|
| DEV | 0.2314 | 0.3386 | +0.1072 | +0.1328 | [+0.053, +0.163] | 0.92 | 12/16 | 11/16 |
| CONFIRM | 0.2480 | 0.3774 | +0.1294 | +0.1622 | [+0.074, +0.184] | 1.10 | 14/16 | 10/16 |

Contadores auditados durante TEST de B (todas las seeds, ambas fases): `cdt_queries=0, rqm_queries=0, table_queries=0, nn_queries=0, attractor_queries=0, direct_memory_queries=0, decoder_lookup=0, leaked=0, target_seen_cdt=false, target_equivalent_seen=false` (cada target de replay se comparó contra TEST por hash canónico y bola de corrección 5 %; 0 hits). Contadores thread-local, instrumentados en el store CDT (que sí registra sus lecturas durante la consolidación: p.ej. 192 lecturas por seed fuera de TEST).
Partición DEV (in-dist, sin hueco): B−A = +0.075 (DEV seeds) / +0.065 (CONFIRM seeds), IC inferior ≈ 0 → la ganancia se concentra en el hueco de interpolación de parámetros.
E37: proceso nuevo cargando sólo el checkpoint: |pre−post| = 0.0 en 32/32; queries=0. E38: P1 (CDT cargado, sin consultar) = P0 exacto; TEST' nuevo: P0 0.342/0.373 vs A 0.238/0.255; P3 (lookup CDT, control positivo) ≈ 0.012–0.015 → el lookup **no** explica el rendimiento.

## 3. Salvedad crítica (lectura honesta de E35)

Diagnóstico post-hoc (no gate, `examples/v4_teacher_diag.rs`): la `LearningSignal` usada directamente como predictor ("teacher") acierta **0.829 (DEV) / 0.831 (CONFIRM)**, muy por encima de B (0.34/0.38). La consolidación es una regresión estructurada (regla afín puntual con coeficientes cuadráticos en p) cuya clase de hipótesis coincide con 6 de 8 familias del generador. Por tanto:

- Lo que E35 demuestra: información consolidada fuera del hot path **se transfiere persistentemente a los pesos de Dφ** y mejora TEST sin ninguna ruta de recuperación (borrada, proceso nuevo, contadores en 0), y E36 muestra que depende del contenido (B_raw, corrupto, irrelevante, sin topología, orden barajado no la reproducen; B_raw−A ≈ +0.03/+0.02, IC incluye 0).
- Lo que **no** demuestra: que el campo descubra la regularidad por sí mismo. El sesgo inductivo lo aporta el diseño de `consolidate`; Dφ sólo destila ~40 % de lo que el teacher ya sabe. Es destilación desde un modelo de reglas consolidado, no aprendizaje emergente.
- Precisión absoluta baja: Dφ_A ≈ 0.23–0.25 y B ≈ 0.34–0.38 con el umbral 5 %.

## 4. E31–E34 (Dφ solo)

- E31: Dφ_A 0.231/0.248 vs STATIC 0.04, LINEAR 0, RANDOM 0, SHUFFLED 0, NN_LOOKUP 0.012/0.015; gana ≥3 familias en 32/32 seeds.
- E32: σ_max(J_Dφ) dentro del 10 % del operador verdadero en ≥3 familias en todas las seeds, pero **sesgo sistemático expansivo**: en isometrías σ_Dφ ≈ 1.05–1.14 vs 1.00; σ>1 en ~98 % de los pasos del rollout.
- E33: free-run iterable acc h1 0.13 → h2 0.01 → h4 ≈ 0; rel_err 0.08 (h1) → 0.49 (h8) → 1.9 (h64). Región de estabilidad < 1 paso con el umbral pre-registrado. Traza (rotate): norm ratio 1.00→1.11/1.19, energía 1.01→1.25/1.41, manifold distance 0.036→0.42/0.45, error de un paso sobre la trayectoria verdadera constante ≈ 0.065. El error es de un paso (≈6.5 %) y se compone con σ>1 y deriva fuera de la órbita rígida.
- E34: Dφ residual 0.231/0.248 vs MLP_DIRECT (962 params, mismos datos/seed/pasos) 0.217/0.232: diferencia < margen en la mayoría de seeds; ambos 0 en h4. No hay ventaja específica de "dinámica" a igual capacidad.

## 5. Diagnóstico E23 (modelo v3.7, LONG lock, seeds 0xA300–0xA30F)

`diagnose_e23` reentrena exactamente como `run_e23` y mide por paso (medianas entre seeds):

| step | cos target | σ_max(J) | ‖tanh(Wψ+b)‖ pre-proyección | manifold dist | err 1-paso sobre la trayectoria verdadera | crecimiento de perturbación ε=0.01 | ‖x_true‖ |
|---|---|---|---|---|---|---|---|
| 1 | 0.995 | 3.52 | 0.80 | 0.004 | 0.0055 | 1.19 | 6.9 |
| 8 | 0.885 | 2.77 | 0.87 | 0.037 | 0.0044 | 2.69 | 8.2 |
| 16 | 0.751 | 2.78 | 0.94 | 0.128 | 0.0034 | 3.44 | 10.5 |
| 32 | 0.684 | 2.64 | 0.97 | 0.174 | 0.0034 | 1.30 | 16.1 |
| 64 | 0.739 | 2.61 | 0.97 | 0.148 | 0.0029 | 0.51 | 28.4 |

Hallazgos: (1) el error de un paso en la trayectoria verdadera es minúsculo (≤0.6 %) → **más epochs/dyn_updates no atacan la causa**; (2) σ_max(J) ≈ 2.6–3.5 sostenido → los errores pequeños se amplifican (crecimiento de perturbación ×3.4 a h16); (3) el estado sale de la imagen del encoder (manifold distance 0.004→0.17); (4) la preactivación satura (0.80→0.97) y el target casi no se mueve en el espacio del campo cuando |x|>15 (cos entre targets consecutivos → 1.000), por lo que en h32–h64 la perturbación se contrae y la trayectoria queda en una región saturada lejos del target. Remedios candidatos (no aplicados en este ciclo): re-proyección al manifold del encoder por paso, restricción espectral en el espacio tangente, parametrización residual, o un encoder no saturante para trayectorias largas.

## 6. Criterio de éxito del handoff

"Experiencia consolidada cambia la dinámica de forma persistente y esa dinámica genera respuestas nuevas tras retirar la memoria": **cumplido operativamente** bajo este protocolo (E35/E37/E38 PASS en DEV y CONFIRM, leakage 0, contadores 0, proceso nuevo) **pero con la salvedad de §3**: el contenido de la consolidación es una regresión diseñada a mano que coincide con el generador, Dφ por sí solo no supera a un MLP de igual capacidad (E34) y no es estable en horizontes largos (E33). E42 no alcanza el gate estricto. E43/E44 sin ejecutar. No se reclama nada más allá de este benchmark sintético.

## 7. E45 — Liquid vs Thermo (pre-registrado en plan §24 / preregistro §7)

Gate (max(L1,L2) ≥ T1−0.05, queries=0): **FAIL** — DEV 0/16, CONFIRM 0/16. Leakage y queries = 0 para los cuatro cerebros; rollback L2 exacto 32/32. L3 y composición A→B→C: NOT_RUN.

## dev
| brain | acc TEST | set1 tras p2 | olvido | set2 | retención episodios borrados | consolidación ms | params cambiados | bytes | latencia µs | updates | queries |
|---|---|---|---|---|---|---|---|---|---|---|---|
| L1_liquid_gated | 0.067 | 0.000 | 0.086 | 0.133 | 0.437 | 2137.4 | 962 | 7696 | 9.45 | 78213 | 0 |
| L2_liquid_lowrank | 0.000 | 0.000 | 0.086 | 0.001 | 0.003 | 1827.0 | 972 | 9824 | 10.04 | 64000 | 0 |
| T1_thermo_cdt | 0.808 | 0.788 | 0.000 | 0.828 | 0.891 | 0.5 | 384 | 3072 | 0.12 | 96 | 0 |
| H_liquid_inf_cdt_cons | 0.213 | 0.236 | 0.150 | 0.190 | 0.344 | 1841.4 | 962 | 7696 | 9.44 | 128000 | 0 |
## confirm
| brain | acc TEST | set1 tras p2 | olvido | set2 | retención episodios borrados | consolidación ms | params cambiados | bytes | latencia µs | updates | queries |
|---|---|---|---|---|---|---|---|---|---|---|---|
| L1_liquid_gated | 0.061 | 0.000 | 0.070 | 0.122 | 0.461 | 2151.4 | 962 | 7696 | 9.41 | 79972 | 0 |
| L2_liquid_lowrank | 0.001 | 0.000 | 0.070 | 0.003 | 0.003 | 1830.1 | 972 | 9824 | 10.39 | 64000 | 0 |
| T1_thermo_cdt | 0.802 | 0.795 | 0.000 | 0.808 | 0.891 | 0.6 | 384 | 3072 | 0.12 | 96 | 0 |
| H_liquid_inf_cdt_cons | 0.238 | 0.246 | 0.082 | 0.230 | 0.355 | 1826.3 | 962 | 7696 | 9.63 | 128000 | 0 |

Lectura: con 96 experiencias y presupuesto igual de updates, el sustrato líquido puro (L1 gated, L2 low-rank) no consolida: L1 0.06–0.07 y olvida set1 por completo tras fase 2; la memoria rápida L2 (UVᵀ en capa de salida, sin gating por contexto) destruye set1 y no aprende set2 (rollback la revierte exactamente). T1 (regla CDT consolidada) gana en todo: accuracy 0.80, olvido 0, ~4000× menos tiempo de consolidación, 0.12 µs/query, 384 números. El híbrido (inferencia líquida + consolidación CDT) queda en medio (0.21–0.24). Misma salvedad que §3: la consolidación CDT es una regresión cuya clase de hipótesis coincide con el generador, así que la comparación favorece estructuralmente a T1; no es prueba general de que lo termodinámico supere a lo líquido. Analogía cognitiva: en este benchmark el "largo plazo" útil vive en la regla consolidada, no en la dinámica.

## 8. Seguimiento 1 — consolidación genérica (RFF kernel ridge, sin estructura del generador)

Pre-registro §8. Tuning sólo en seeds nuevas `0xC400–0xC407` (partición DEV, accuracy del teacher): D=1600, ℓ=1.4, λ=1e-8·n. Mismas seeds DEV/CONFIRM y gates.

| | DEV | CONFIRM |
|---|---|---|
| Teacher genérico (consolidado como predictor, TEST) | 0.894 | 0.899 |
| E35 A / B | 0.231 / 0.360 | 0.248 / 0.389 |
| E35 B−A [IC95], d_z | +0.128 [+0.085,+0.170], 1.45 | +0.141 [+0.095,+0.187], 1.44 |
| E35 seeds PASS | **14/16** | **12/16** |
| E35 partición DEV (in-dist) B−A [IC95] | +0.095 [+0.053,+0.136] | +0.089 [+0.041,+0.133] |
| E36 seeds PASS | 10/16 | 12/16 |
| E36 C − B_raw [IC95] | +0.096 [+0.038,+0.154] | +0.125 [+0.063,+0.184] |
| Leakage / queries en TEST | 0 / 0 | 0 / 0 |

E35 y E36 siguen PASS con una consolidación que no conoce la estructura afín; la ganancia ahora también es significativa en la partición in-dist. La salvedad cambia de forma, no desaparece: Dφ (0.36–0.39) sigue muy por debajo del teacher genérico (0.89–0.90) — destila ~40 % de la regularidad consolidada.

**E45 con consolidación genérica** (DEV/CONF): T1 0.482/0.504, H 0.174/0.172, L1 0.067/0.061, L2 0.000/0.001 → gate FAIL 0/16, 0/16. T1 genérico pierde su ventaja de coste (5.1 s de consolidación, 24 090 números, 245 µs/query) y ahora olvida (0.17), pero sigue ganando en accuracy. Tabla completa: `artifacts/v4/generic/e45/table.md`.

**Corrección E45 (rule)**: H en fase 2 usaba episodios de fase 1 ya borrados; corregido (pseudo-experiencias desde la señal de fase 1). Re-corrida (`artifacts/v4/e45_fixed/`): H 0.238/0.251 (antes 0.213/0.238); el resto idéntico; gate sigue FAIL 0/16, 0/16. Sustituye a la tabla de §7.

## 9. Seguimiento 2 — estabilidad (E33S en v4; variantes E23 en v3.7)

Pre-registro §9. **E33S (v4, Dφ de E35-A, rotate+reflect, accuracy / rel_err media):**

| Variante | h1 | h2 | h4 | h8 | h64 rel_err | PASS DEV | PASS CONF |
|---|---|---|---|---|---|---|---|
| V0 baseline (residual) | 0.127 / 0.126 | 0.005 / 0.010 | 0.000 / 0.001 | 0 / 0 | 1.52 / 1.65 | 0/16 | 0/16 |
| V1 re-proyección manifold | 0.291 / 0.279 | 0.077 / 0.068 | 0.005 / 0.010 | 0 / 0.002 | 1.49 / 1.62 | 0/16 | 0/16 |
| V2 límite espectral | 0.151 / 0.170 | 0.004 / 0.009 | 0 / 0 | 0 / 0 | 1.47 / 1.60 | 0/16 | 0/16 |
| V3 V1+V2 | 0.263 / 0.253 | 0.053 / 0.052 | 0.008 / 0.007 | 0 / 0.001 | 1.45 / 1.53 | 0/16 | 0/16 |

Lectura: en v4 el cuello no es la amplificación sino el **error de un paso** (rel_err ≈ 0.08–0.10 en h1, por encima del umbral 5 %); ningún operador de rollout puede convertir eso en una región estable ≥8. V1 duplica la accuracy en h1–h2 (prior de órbita rígida, declarado), V2 reduce la deriva de norma pero añade sesgo. E33 sigue **FAIL**.

**E23 v3.7 (diagnóstico, cos medio por horizonte, 16 seeds 0xA300–0xA30F):**

| Variante | h1 | h4 | h8 | h16 | h32 | h64 |
|---|---|---|---|---|---|---|
| V0 baseline (reproduce LONG v3.7) | 0.979 | 0.853 | 0.684 | 0.469 | 0.325 | 0.448 |
| V1 re-proyección al manifold del encoder | 0.983 | 0.933 | 0.892 | 0.791 | **0.763** | **0.751** |
| V2 límite espectral | 0.994 | 0.955 | **0.907** | **0.836** | 0.608 | 0.388 |
| V3 V1+V2 | 0.994 | 0.948 | 0.879 | 0.774 | 0.621 | 0.480 |

Confirma el diagnóstico de §5: en v3 (error de un paso minúsculo) la re-proyección al manifold es la palanca que arregla h32/h64 (+0.44/+0.30 de cos) y el límite espectral arregla h4–h16. Encoder no saturante: NOT_RUN (requiere re-entrenar encoder/Dφ v3). Sólo diagnóstico, sin gate pre-registrado sobre v3.

## 10. Seguimiento 3 — E45 completo (L2g, L3, composición, iteración adaptativa)

Pre-registro §10. Medias de 16 seeds; DEV / CONFIRM. `artifacts/v4/e45_full_{rule,generic}/`.

| acc TEST | L1 | L2 | **L2g** | **L3** | T1 | H |
|---|---|---|---|---|---|---|
| consolidación rule | 0.067 / 0.061 | 0.001 / 0.001 | 0.062 / 0.050 | 0.009 / 0.010 | 0.808 / 0.802 | 0.238 / 0.251 |
| consolidación genérica | 0.067 / 0.061 | 0.001 / 0.001 | 0.062 / 0.050 | 0.009 / 0.010 | 0.482 / 0.504 | 0.174 / 0.172 |

**Composición** (rotate(p1,0) → scale(p2,p2) encadenados vs compose(p); acc encadenada / directa, rule): T1 1.000/0.961 (0.994/0.973); H 0.049/0.164 (0.066/0.191); L1 0/0.11; L2g ≈0; L3 0; STATIC 0.006. Genérica: T1 0.18/0.57 (0.23/0.59), H 0.02–0.05.

**Iteración adaptativa** (sub-pasos p/n, parada si cambio <1 %): nunca se detiene antes de n=8 en ningún tercil de dificultad; accuracy n=1 → adaptativa: H 0.316 → 0.001 (rule), L1 0 → 0. Iterar Dφ acumula su error de un paso (consistente con §9).

| Gate (PASS seeds) | rule DEV | rule CONF | gen DEV | gen CONF |
|---|---|---|---|---|
| E45b max(L1,L2,L2g,L3) ≥ T1−0.05 | 0/16 | 0/16 | 0/16 | 0/16 |
| E45c mejor líquido encadenado ≥ STATIC+0.05 | 3/16 | 7/16 | 2/16 | 3/16 |
| E45a ganancia adaptativa ≥ 0.05 | 0/16 | 0/16 | 0/16 | 0/16 |

Los tres gates **FAIL**. L2g elimina el olvido por construcción pero no aprende nuevas familias con 12 ejemplos/familia; L3 (energía) falla como se anticipó (las rotaciones no son flujos de gradiente). Sólo el consolidado T1 compone leyes no vistas (el rule por estructura; el genérico parcialmente). Analogía cognitiva: la memoria rápida (L2/L2g) no generaliza; la dinámica consolidada (H) generaliza parcialmente; la ley explícita consolidada (T1) es la que transfiere.

## 11. Seguimiento 4 — E43/E44 con periferia LLM congelada

Pre-registro §11. Periferia A = Gemma-2-2B-it Q3_K_L (loader existente, denso); B = Qwen2.5-1.5B-instruct Q4_K_M (loader candle vendorizado con `forward_hidden`); R = encoder aleatorio congelado. Embeddings cacheados (`artifacts/v4/lang/*.json.gz`, sha256 en `artifacts/v4/E43_E44/*/summary.md`). Decoder de vocabulario cerrado (azar ≈ 1/40–1/60). Accuracy top-1 en conceptos novedosos, media de 16 seeds (DEV / CONF):

| | es held-out | en | fr | ja |
|---|---|---|---|---|
| E43 Dφ (Gemma) | 0.194 / 0.167 | 0.042 / 0.056 | 0.064 / 0.086 | 0.077 / 0.058 |
| E43 OFFSET lineal (Gemma) | 0.319 / 0.264 | 0.069 / 0.090 | 0.177 / 0.214 | 0.030 / 0.010 |
| E43 Dφ (encoder aleatorio) | 0.076 / 0.181 | 0.035 / 0.076 | 0.062 / 0.141 | 0.054 / 0.042 |
| STATIC | 0 | 0 | 0 | 0 |
| E44 Dφ_A ← Qwen alineado | 0.049 / 0.118 | 0.056 / 0.049 | 0.023 / 0.056 | 0.019 / 0.032 |
| E44 Dφ_A ← aleatorio alineado | 0.069 / 0.056 | 0.076 / 0.083 | 0.073 / 0.049 | 0.077 / 0.030 |
| E44 Dφ_A ← Gemma alineado (techo) | 0.153 / 0.153 | 0.028 / 0.049 | 0.065 / 0.057 | 0.049 / 0.040 |

| Gate | DEV | CONF |
|---|---|---|
| E43 (≥2 de en/fr/ja superan max(STATIC, R)+0.05) | 2/16 FAIL | 3/16 FAIL |
| E44 (Qwen ≥ R+0.05 y ≥ 0.5·Gemma) | 3/16 FAIL | 3/16 FAIL |

**E43/E44 FAIL.** Con ~21 pares españoles, Dφ aprende algo en español (0.17–0.19, por encima del aleatorio en DEV) pero no transfiere a en/fr/ja más allá del encoder aleatorio, y queda por debajo del control lineal OFFSET. El cambio de LLM con alineación por 96 anclas no conserva la dinámica. Limitaciones: vocabulario pequeño, ja-plural artificial (々), pooling medio del hidden final; no se exploraron capas intermedias (no pre-registrado).

## 12. Resumen final (estado de todos los experimentos, conclusiones, pendientes)

Estado a 2026-10-10, tip de código `1da52e7`. DEV / CONFIRM, 16+16 seeds, sin retune.

| Exp | Qué mide | Estado | Número clave (DEV / CONF) |
|---|---|---|---|
| E31 | Dφ aprende reglas | PASS 16/16 | acc 0.231 / 0.248 (STATIC 0.04) |
| E32 | estabilidad σ_max | PASS (sesgo expansivo σ≈1.05–1.14) | 16/16 |
| E33 | rollout largo | **FAIL** | h1 0.13 → h4 ≈0 |
| E33S (seg. 2) | re-proyección / límite espectral | **FAIL** | mejor V1 h1 0.29, h4 ≤0.01 |
| E34 | dinámica vs MLP igual capacidad | PARTIAL | Dφ 0.231/0.248 vs MLP 0.217/0.232 |
| E35 | CDT → consolidación → Dφ (sin memoria) | PASS 11/16, 10/16 | A 0.231/0.248 → B 0.339/0.377 |
| E35 genérico (seg. 1) | consolidación RFF sin estructura | PASS 14/16, 12/16 | B 0.360/0.389; teacher 0.894/0.899 |
| E36 | ablación de contenido | PASS (rule y genérico) | 10/13 · 10/12 seeds |
| E37 / E38 | borrado / persistencia | PASS | |pre−post| = 0 en 32/32 |
| E39 / E40 / E41 | transfer / continual / causal | PASS | 13/10 · 16/16 · 16/16 |
| E42 | scaling | FAIL/PARTIAL (gate estricto) | IC>0 en las 5 N |
| E43 / E44 (seg. 4) | periferia LLM congelada / swap | **FAIL** | E43 2/16, 3/16; E44 3/16, 3/16 |
| E45 (+seg. 1, 3) | líquido vs termo (CDT) | **FAIL** para líquido (0/16 todos los gates) | T1 0.808/0.802 (rule), 0.482/0.504 (gen); mejor líquido ≤0.07 |
| E23 v3.7 diag (seg. 2) | re-proyección al manifold | diagnóstico (sin gate) | cos h64 0.448 → 0.751 |

**Conclusiones sólidas.** (1) La consolidación fuera del hot path transfiere información de forma persistente a los pesos de Dφ (E35–E38, leakage/queries 0, proceso nuevo), y sobrevive a un teacher genérico sin estructura del generador (seg. 1). (2) La regla/teacher consolidado (T1/CDT) domina a los sustratos líquidos en accuracy, olvido, composición y coste (E45 completo). (3) En v3 la re-proyección al manifold arregla horizontes largos; en v4 no, porque el cuello es el error de un paso (~8 %).

**Lo que no está demostrado.** Que el campo descubra regularidades solo (B es destilación de un teacher; Dφ captura ~40 %); ventaja de "dinámica" sobre MLP de igual capacidad (E34); rollout largo (E33); lenguaje real / swap de LLM (E43/E44).

**Pendientes / abiertos.** (a) Subir la captura de Dφ del teacher consolidado (capacidad, pérdida de destilación, más pasos de replay). (b) Reducir el error de un paso de Dφ por debajo del 5 % antes de volver a atacar E33 (después: re-proyección V1). (c) Entrada lingüística: probar Ψ de 64-D de Gemma (bottleneck E45 de la rama token-free) como entrada a Dφ en lugar del pooling medio crudo. (d) E42 estricto. Comparativa y hoja de ruta: `docs/comparativa_v4_vs_token_free.md`.

---

# Plan Dφ / teacher distillation (`docs/plan_v4_dphi_teacher_distillation.md`)

Pre-registro: `docs/preregistro_v4.md` §12+. Código: `src/v4_distill.rs`, análisis `scripts/v4_distill_analyze.py`. Artefactos: `artifacts/v4/distill/{smoke,dev,confirm}/` (`metrics.csv`, `summary.md`, `paired_stats.txt`, `config.json`, `checkpoints/`). Seeds DEV `0xA400–0xA40F`, CONFIRM `0xB400–0xB40F`, sin retune.

## Fase A — techo de destilación (`DIST_A`)

Corrido en `86a750a` (git_dirty=false), DEV y CONFIRM, 16+16 seeds, ~32 s cada fase. Presupuesto 8000×16 en todas las condiciones salvo D4 (4×).

| Condición (TEST, acc rel_err<5 %) | DEV acc | DEV rel_err₁ | CONF acc | CONF rel_err₁ | teacher→modelo rel (DEV/CONF) | σ_max med |
|---|---|---|---|---|---|---|
| T teacher directo (post-hoc) | 0.829 | 0.027 | 0.831 | 0.025 | — | — |
| A sólo TRAIN | 0.231 | 0.079 | 0.248 | 0.078 | 0.081/0.080 | 1.12 |
| B ruta de consolidación (E35-B) | 0.339 | 0.071 | 0.377 | 0.069 | 0.066/0.064 | 1.10 |
| **D1 destilación directa del teacher** | **0.334** | 0.073 | **0.367** | 0.069 | 0.065/0.063 | 1.09 |
| M1 MLP directo (962) destilado | 0.318 | 0.073 | 0.334 | 0.071 | 0.065/0.064 | 1.09 |
| D4 destilación, 4× pasos (diagnóstico) | 0.610 | 0.057 | 0.591 | 0.057 | 0.045/0.046 | 1.07 |

Pareadas (IC95 bootstrap):
- **D1 − B: DEV −0.004 [−0.030, +0.019] (8/16); CONF −0.010 [−0.044, +0.025] (5/16).**
- D1 − A: +0.103 [+0.051, +0.154] / +0.119 [+0.063, +0.176].
- D1 − M1 (control MLP igual capacidad): +0.016 [−0.033, +0.069] / +0.033 [−0.012, +0.073] → sin ventaja demostrada de Dφ residual.
- D4 − D1: +0.275 [+0.198, +0.350] (16/16) / +0.224 [+0.170, +0.275] (15/16).

Integridad: leakage=0, queries=0 en todas las condiciones; 0 hits del auditor en las etiquetas del teacher; proceso nuevo (`eval-checkpoint`) bit-exacto con queries=0 en 32/32.

**Gate Fase A: FAIL** (gate por seed 3/16 y 3/16; IC de D1−B incluye 0 en ambas fases). Destilar directamente del teacher no supera a la ruta de consolidación existente.

Lectura pre-registrada (§12, caso ii): D1 ≈ B y ambos muy lejos del teacher (captura ≈ 40–44 %), con error teacher→Dφ ≈ 6.5 %. La ruta de consolidación **no** es el cuello (dar etiquetas perfectas del teacher en todo el espacio no ayuda). El cuello está en la capacidad/optimización de Dφ-962 con el presupuesto bloqueado: con 4× pasos la misma arquitectura llega a 0.59–0.61 (captura ≈ 72 %), es decir, la representación de 962 parámetros puede acercarse al teacher pero el presupuesto de 8000 pasos la deja sub-entrenada. Según lo pre-registrado, se continúa a la Fase B (barrido de capacidad), que es la prueba que el plan prescribe para este caso. Nota honesta: D4 no es una condición comparable (usa 4× cómputo); es diagnóstico.

## Fase B — barrido de capacidad (`DIST_B`)

Corrido en `03335f9` (git_dirty=false). Ruta D1, presupuesto fijo 8000×16, mismas seeds/init. Memoria = params×8 B×4.

| Modelo | params | memoria | wall entren. (s) | DEV acc | DEV rel_err₁ | CONF acc | CONF rel_err₁ | σ_max med (D/C) |
|---|---|---|---|---|---|---|---|---|
| Dφ H64 | 962 | 30.8 KB | 2.1 | 0.334 | 0.073 | 0.367 | 0.069 | 1.09/1.10 |
| Dφ H136 | 2042 | 65.3 KB | 4.0 | **0.399** | 0.070 | **0.414** | 0.067 | 1.08/1.09 |
| Dφ H272 | 4082 | 130.6 KB | 7.7 | 0.363 | 0.073 | 0.341 | 0.073 | 1.08/1.09 |
| Dφ H544 | 8162 | 261.2 KB | 14.5 | 0.323 | 0.078 | 0.319 | 0.077 | 1.08/1.09 |
| MLP H64 / H136 / H272 / H544 | igual | igual | igual | 0.318 / 0.371 / 0.372 / 0.300 | — | 0.334 / 0.421 / 0.357 / 0.337 | — | ≈1.08 |
| B consolidación H64 | 962 | | | 0.339 | 0.071 | 0.377 | 0.069 | 1.10 |
| Teacher | (384 números) | | | 0.829 | 0.027 | 0.831 | 0.025 | — |

Pareadas vs H64: H136 **DEV +0.065 [+0.0005, +0.126]**, **CONF +0.046 [−0.024, +0.112]**; H272 +0.029 / −0.026 (IC incluye 0); H544 −0.011 / −0.048. Dφ vs MLP de igual H: todas las diferencias con IC que incluye 0 (−0.018…+0.033). Leakage=0, queries=0.

**Gate Fase B: FAIL.** H* = 136 se eligió en DEV (único con IC>0, apenas), pero CONFIRM no reproduce (IC incluye 0). Con presupuesto fijo, más capacidad no ayuda de forma reproducible y H≥272 empeora (más lento de optimizar con los mismos 8000 pasos); el error teacher→Dφ se queda en 6–7 % en todas las H. Ninguna ventaja de la dinámica residual frente al MLP de igual capacidad. Conclusión: con este presupuesto el cuello no es el número de parámetros sino la optimización (coherente con D4 de la Fase A). Según lo pre-registrado (§13), la Fase C se aplica a H=64.

## Fase C — optimización explícita de un paso (`DIST_C`, H=64)

Corrido en `5d08d64` (git_dirty=false). Receta única pre-registrada: etiquetas de la ruta de consolidación (TRAIN real + replay del teacher auditado), 32000 pasos × batch 32 (8× cómputo del lock), Adam 3e-3 → 3e-5 coseno, pérdida relativa.

| Condición | DEV acc | DEV rel_err₁ (IC95) | CONF acc | CONF rel_err₁ (IC95) | teacher→modelo | σ_max med | seeds rel<5 % |
|---|---|---|---|---|---|---|---|
| C0 lock (referencia) | 0.339 | 0.0705 | 0.377 | 0.0690 | 0.066/0.064 | 1.10 | 0/16 |
| **C1 Dφ receta one-step** | **0.808** | **0.0374** [0.0358, 0.0390] | **0.814** | **0.0367** [0.0356, 0.0378] | 0.030/0.029 | 1.06 | **16/16, 16/16** |
| M1 MLP igual capacidad, misma receta | 0.808 | 0.0369 | 0.812 | 0.0364 | 0.030/0.029 | 1.06 | 16/16 |

C1−C0: +0.469 [+0.423, +0.514] / +0.436 [+0.394, +0.481] (16/16). C1−M1: −0.000 [−0.007, +0.007] / +0.001 [−0.006, +0.008]. Teacher 0.829/0.831 → captura de Dφ ≈ 97–98 % (objetivo fuerte del plan ≥0.70: alcanzado). Leakage=0, queries=0, proceso nuevo bit-exacto 32/32.

Desglose rel_err₁ C1 (DEV; CONF casi idéntico): translate 0.020, rotate 0.021, **reflect 0.101**, scale 0.022, shear 0.023, affine 0.023, compose 0.028, **radial_warp 0.061**; in_dist 0.039, interp_hole 0.036. Coseno del desplazamiento ≥0.94 salvo radial_warp (0.73). Distancia al manifold (órbita de similitud) alta en reflect (0.98: la métrica no incluye espejo) y en scale/shear (no son similitudes).

**Gate Fase C: PASS** (rel_err₁ medio < 5 % en DEV y CONFIRM, 16/16 seeds cada una, sin lookup ni memoria).

Lectura honesta: (1) la mejora viene de optimización (8× pasos, LR decreciente, pérdida relativa), no de capacidad ni de la ruta; cuesta 8× cómputo (~16 s por modelo). (2) El MLP no residual de igual capacidad logra lo mismo: no hay ventaja atribuible a la "dinámica". (3) Dos familias siguen por encima del 5 % (reflect 10 %, radial_warp 6 %); el promedio pasa por las otras seis. (4) Sigue siendo destilación de un teacher con clase de hipótesis diseñada (no emergente).
