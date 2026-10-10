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
