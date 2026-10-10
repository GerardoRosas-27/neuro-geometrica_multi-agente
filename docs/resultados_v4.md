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
