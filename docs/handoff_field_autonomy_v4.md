# Handoff — Field Autonomy v4

**Branch:** `exp/field-autonomy-v4`  
**Base:** `main`  
**Updated:** 2026-10-09

## Estado

Esta rama contiene únicamente la especificación v4. Fue creada directamente desde `main` para iniciar un ciclo limpio. El Clean-Room v3.7 ya está integrado en `main` mediante PR #29; no asumir que los textos históricos que dicen «no mergeado» reflejan el estado actual.

Documentación principal:

- `docs/plan_autonomia_campo_v4.md` — diseño científico detallado.
- `docs/protocolo_v4_agente.md` — protocolo operativo para implementación.
- `docs/handoff_field_autonomy_v4.md` — este documento.

## Qué debe resolver v4

V3 ya aportó evidencia de que Dφ puede aprender reglas/composición bajo FIELD_ONLY, pero E23 fue débil y la cadena central experiencia→CDT→Dφ quedó abierta.

La pregunta prioritaria es:

```text
experiencias TRAIN
      ↓
CDT consolidation
      ↓
learning signal
      ↓
adaptación Dφ
      ↓
DELETE CDT/RAM/RQM/retrieval
      ↓
X_new → Dφ → Y_new nunca almacenado
```

No confundir con:

```text
X_new → CDT → Y_new
```

La segunda es lookup/memoria como respuesta y no demuestra la hipótesis arquitectónica.

## Prioridad

### P0 — E31–E34

Implementar primero:

- generador limpio TRAIN/DEV/TEST;
- manifest y hashes canónicos;
- auditoría de provenance;
- controles paired;
- free-run;
- diagnóstico Jacobiano.

La razón es que E35 no puede interpretarse si no sabemos cuánto generaliza Dφ sin experiencia consolidada.

### P1 — E35–E38

Este es el núcleo científico: determinar si CDT produce aprendizaje persistente en Dφ, no recuperación.

### P2 — E39–E41

Transferencia, continual learning y causalidad.

### P3 — E42–E44

Escala, lenguaje y cambio de LLM.

## E35 — diseño recomendado

Dos sistemas idénticos y paired:

### A — control

```text
TRAIN → Dφ_A
```

### B — tratamiento

```text
TRAIN experiences
→ CDT consolidation
→ learning_signal
→ Dφ_B adaptation
→ DELETE CDT/RAM/RQM/indexes
```

Mismo TEST sellado para A y B.

La evidencia buscada es:

```text
performance(B) > performance(A)
```

simultáneamente con:

```text
CDT queries = 0
RQM queries = 0
lookup queries = 0
leakage = 0
```

Si B no supera A, se reporta como resultado negativo; no se retoca hasta obtener PASS sin documentar el cambio de protocolo.

## Gate de seguridad contra leakage

Cada predicción debe poder responder:

```text
target_seen_training
target_seen_dev
target_seen_cdt
target_seen_rqm
target_seen_table
target_seen_nn
target_seen_attractor
target_equivalent_seen
cdt_queries
rqm_queries
table_queries
nn_queries
attractor_queries
direct_memory_queries
decoder_lookup
```

Una sola ruta directa al target marca `LEAKED`.

## Diagnóstico E23

No subir epochs/dyn_updates a ciegas. Medir primero:

- Jacobian spectral norm;
- norm drift;
- manifold drift;
- error por step;
- sensibilidad a perturbaciones;
- energía.

Después decidir si procede regularización, residual parameterization, spectral constraint, learned step size o energy shaping.

## Reglas de implementación

- No copiar pesos/datasets/checkpoints de E11–E17/v3.
- No añadir rutas RQM ocultas para rescatar E22/E35.
- No consultar CDT durante TEST y llamarlo «memoria como experiencia».
- No seleccionar seeds favorables.
- No eliminar seeds fallidas sin causa registrada.
- No cambiar decoder después de observar TEST.
- No confundir SMOKE con evidencia científica.
- No cerrar un gate solo por accuracy 100 %.

## Primer trabajo del agente

```bash
git status
git log --oneline -20
cargo test --all-targets
```

Después implementar infraestructura de dataset + manifest + provenance antes de E31/E35.

## Criterio de cierre

Mantener una tabla explícita para E31–E44 con `NOT_RUN`, `SMOKE_ONLY`, `PASS`, `PARTIAL`, `FAIL`, `LEAKED` o `DATASET_INVALID`.

Solo cambiar de estado con resultados realmente ejecutados y artefactos reproducibles.

**Regla final:** v4 no busca simplemente que el campo acierte más. Busca demostrar que la experiencia consolidada cambia la dinámica de forma persistente y que esa dinámica genera respuestas nuevas después de que la memoria de experiencias ha sido retirada.

## Estado tras el primer ciclo v4 (2026-10-09)

Ver `docs/resultados_v4.md` (DEV `0xA400–0xA40F` + CONFIRM `0xB400–0xB40F`, pre-registro `docs/preregistro_v4.md`).

| Gate | Estado |
|---|---|
| E31 | PASS (16/16, 16/16) |
| E32 | PASS (16/16, 16/16) — sesgo expansivo σ≈1.08 |
| E33 | FAIL (0/16, 0/16) |
| E34 | PARTIAL (5/16, 6/16) |
| E35 | PASS (11/16, 10/16; IC95 B−A >0) — B destila un teacher consolidado de acc 0.83 |
| E36 | PASS (10/16, 13/16) |
| E37 | PASS (11/16, 10/16) |
| E38 | PASS (11/16, 10/16) |
| E39 | PASS (13/16, 10/16) |
| E40 | PASS (16/16, 16/16) |
| E41 | PASS (16/16, 16/16) |
| E42 | FAIL/PARTIAL (4/16, 5/16) por gate estricto |
| E43 | NOT_RUN |
| E44 | NOT_RUN |
| E45 | FAIL (0/16, 0/16): T1 CDT 0.80 >> L1 0.06, L2 0.00, H 0.22 |

Pendiente: consolidación menos diseñada a mano (aprendida/genérica), estabilidad multi-paso (E33), E43/E44 con periferia LLM.
