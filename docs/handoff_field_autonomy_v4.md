# Handoff — Field Autonomy v4

**Branch:** `exp/field-autonomy-v4`  
**Base:** `main`  
**Created:** 2026-10-06

## Estado inicial

Esta rama fue creada directamente desde `main` para evitar contaminar el nuevo ciclo con pesos/datasets de ramas experimentales previas.

Documentación inicial:

- `docs/plan_autonomia_campo_v4.md`
- `docs/protocolo_v4_agente.md`
- `docs/handoff_field_autonomy_v4.md`

## Qué dejó v3

La confirmación v3 cerró en mayoría POS+STRONG los gates E18/E21/E22/E24, pero dejó abiertos E23 y especialmente la cadena experiencia→CDT→Dφ, además de forgetting, causalidad, persistencia y transferencia. El cierre v3 explícitamente no fue mergeado a `main`.

Por tanto v4 no debe comenzar suponiendo que existe autonomía del campo.

## Objetivo de esta rama

Obtener evidencia causal y limpia de esta cadena:

```text
experiencias
    ↓
consolidación CDT
    ↓
regularidad / learning signal
    ↓
adaptación de Dφ
    ↓
CDT/RQM/RAM eliminados
    ↓
X_new
    ↓
Dφ
    ↓
Y_new nunca almacenado
```

El punto crítico es que CDT debe influir en **cómo aprende la dinámica**, no convertirse en un diccionario `X_new→Y_new`.

## Prioridad absoluta

### P0 — E31–E34

Antes de tocar E35, implementar:

- dataset generator limpio;
- manifest/hash;
- provenance;
- controles paired;
- free-run;
- diagnóstico Jacobiano.

La razón es que E35 no puede interpretarse si no sabemos si Dφ ya generaliza por sí mismo o si simplemente tenemos un baseline dinámico débil/fuerte.

### P1 — E35–E38

Estos experimentos responden la pregunta central:

> ¿Puede una experiencia consolidada producir una modificación persistente de la dinámica que luego resuelva un estado nuevo sin consultar la experiencia durante la inferencia?

### P2 — E39–E41

Transferencia, continual learning y causalidad.

### P3 — E42–E44

Scaling, lenguaje y cambio de LLM.

## Gate de seguridad contra leakage

No ejecutar una suite científica si el auditor no puede contestar, por cada predicción:

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

Una sola ruta directa al target debe marcar la corrida `LEAKED`.

## Diseño recomendado de E35

Usar dos modelos idénticos:

### Control A — no experience

```text
TRAIN → Dφ_A
```

### Tratamiento B — consolidated experience

```text
TRAIN experiences
→ CDT consolidation
→ learning signal
→ Dφ_B adaptation
→ DELETE CDT/RAM/RQM
```

Ambos deben recibir el mismo TEST sellado.

El resultado interesante sería:

```text
performance(B) > performance(A)
```

mientras:

```text
CDT queries = 0
RQM queries = 0
lookup queries = 0
```

y el target no exista en ninguna memoria.

Si B no supera A, la hipótesis central queda sin evidencia positiva bajo ese protocolo. Eso es un resultado válido y debe documentarse.

## Error conceptual que v4 debe evitar

No confundir:

### Memoria como respuesta

```text
X_new → CDT → Y_new
```

con:

### Memoria como experiencia

```text
E_train → CDT → regularidad → Dφ adaptado
X_new → Dφ → Y_new
```

Solo la segunda respalda la hipótesis arquitectónica.

## Diagnóstico E23

No intentar simplemente aumentar `dyn_epochs` o `dyn_updates` para arreglar h32/h64.

Primero medir:

- Jacobian spectral norm;
- norm drift;
- manifold drift;
- error por step;
- sensibilidad a perturbación.

Después decidir si conviene:

- regularización de estabilidad;
- residual parameterization;
- spectral constraint;
- learned step size;
- attractor/energy shaping.

La elección debe salir de los datos, no al revés.

## Criterio para cerrar v4

No cerrar la rama porque un experimento aislado dé 100%.

Cerrar únicamente con una tabla explícita:

| Gate | Estado | Evidencia | Limitación |
|---|---|---|---|
| E31 rule learning | NOT_RUN | — | — |
| E32 stability | NOT_RUN | — | — |
| E33 long rollout | NOT_RUN | — | — |
| E34 static/dynamic | NOT_RUN | — | — |
| E35 CDT experience | NOT_RUN | — | — |
| E36 consolidation ablation | NOT_RUN | — | — |
| E37 delete-after-learning | NOT_RUN | — | — |
| E38 persistence | NOT_RUN | — | — |
| E39 transfer | NOT_RUN | — | — |
| E40 continual | NOT_RUN | — | — |
| E41 causal | NOT_RUN | — | — |
| E42 scaling | NOT_RUN | — | — |
| E43 language | NOT_RUN | — | — |
| E44 LLM swap | NOT_RUN | — | — |

El siguiente agente debe actualizar esta tabla solo con resultados realmente ejecutados.

## Primer comando recomendado

Inspeccionar el repo y después implementar la infraestructura, no los experimentos finales.

```bash
git status
git log --oneline -20
cargo test --all-targets
```

Después crear primero el generador de datasets/manifests y el auditor de provenance.

## Regla final

**La prioridad de v4 no es hacer que el campo “acierte más”. Es hacer imposible confundir recuperación, interpolación trivial o memorización con aprendizaje dinámico persistente.**
