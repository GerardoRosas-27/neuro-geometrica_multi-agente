# Protocolo de implementación — Token-Free Liquid Field

## Regla principal

Esta rama es un laboratorio aislado. No reutilizar checkpoints, pesos entrenados, datasets ni resultados favorables de `exp/field-autonomy-v4` o `exp/field-autonomy-next` para declarar resultados positivos.

La arquitectura puede reutilizar código genérico, pero todo artefacto entrenable debe tener provenance y checksum.

## Arquitectura objetivo

```text
                 ┌─────────────────────┐
texto ──────────►│ periférico/encoder  │
                 └──────────┬──────────┘
                            │ h
                            ▼
                 ┌─────────────────────┐
                 │    FieldEncoder      │
                 │ compression/bottleneck│
                 └──────────┬──────────┘
                            │ Ψ₀
                            ▼
                 ┌─────────────────────┐
                 │    Liquid Core       │
                 │ Fθ(Ψ,c)              │
                 └──────────┬──────────┘
                            │ Ψ₁...Ψₙ
                            ▼
                 ┌─────────────────────┐
                 │ field decoder       │
                 └──────────┬──────────┘
                            ▼
                         respuesta

   sleep / consolidation (fuera del hot path)

 experiencias ─► plasticidad/consolidación ─► Δθ / memoria de campo
```

## Contrato del núcleo

El Liquid Core debe aceptar únicamente:

- `Psi`;
- contexto de campo permitido;
- parámetros internos;
- estado dinámico.

No debe aceptar:

- token IDs;
- tokenizer;
- logits del LLM;
- texto;
- lookup de respuestas;
- índice directo input→target.

## Primera implementación

Crear módulos aislados, por ejemplo:

- `src/token_free_field.rs`
- `src/bin/run_token_free_field.rs`
- `src/token_free_dataset.rs`
- `src/token_free_metrics.rs`
- `src/token_free_provenance.rs`
- `src/token_free_controls.rs`

No modificar la arquitectura principal para ejecutar los primeros experimentos.

## Orden obligatorio

1. E45 bottleneck.
2. E46 comparación de dinámicas.
3. E47 sparse.
4. E48 adaptive stopping.
5. E49 coarse-to-fine.
6. E50 geometric beam.
7. E51 associative memory.
8. E52 plasticity.
9. E53 consolidation benchmark.
10. E54 LLM kill test.
11. E55 encoder swap.
12. E56 scaling.
13. E57 head-to-head v4.

No saltar a E52/E57 para evitar optimizar una arquitectura que todavía no tiene una interfaz de campo validada.

## Dataset

Debe existir un manifest con:

- versión;
- generación;
- seed;
- SHA-256 de cada split;
- familias de transformación;
- tamaño;
- dimensionalidad;
- provenance.

Splits mínimos:

`TRAIN / DEV / TEST / OOD / COMPOSITION / LONG_HORIZON`.

Los targets de TEST no pueden aparecer literalmente ni como equivalentes canónicos en TRAIN/DEV.

## Métricas obligatorias

### Calidad

- accuracy;
- OOD accuracy;
- composition accuracy;
- long-horizon accuracy;
- abstention si aplica.

### Dinámica

- `||Psi||`;
- `||DeltaPsi||`;
- cosine;
- energy;
- manifold distance;
- transition error;
- Jacobian spectral norm;
- perturbation amplification.

### Eficiencia

- p50/p95/p99 µs/query;
- pasos promedio;
- nodos activos;
- memoria residente;
- operaciones aproximadas;
- coste de entrenamiento;
- coste de consolidación.

### Dependencia lingüística

- token IDs vistos por el core;
- llamadas al LLM después del encoder;
- llamadas al tokenizer después del encoder;
- decoder lookups;
- equivalentes lingüísticos almacenados.

## Estados

Usar únicamente:

`NOT_RUN`, `SMOKE_ONLY`, `PASS`, `PARTIAL`, `FAIL`, `LEAKED`, `DATASET_INVALID`.

## Criterio de descubrimiento fuerte

El resultado más importante sería demostrar simultáneamente:

`Ψ₀ → dinámica → estado nuevo → respuesta`

con:

- cero acceso a tokens en el core;
- cero acceso al LLM durante rollout;
- target nunca almacenado;
- composición no memorizada;
- OOD positivo;
- ventaja de coste respecto a baseline equivalente;
- repetición en múltiples semillas.

Para consolidación, el gate es más fuerte:

`experiencias → consolidación → cambio persistente en Dθ → borrar experiencias → resolver nuevos estados`.

Eso sería evidencia de aprendizaje persistente del sustrato, no simple recuperación.

## Anti-autoengaño

No declarar “cognición” por velocidad.

No declarar “memoria” por nearest-neighbor.

No declarar “aprendizaje” por fitting del dataset.

No declarar “autonomía” si RQM/CDT/tabla/decoder recupera la respuesta.

No declarar “token-free” si el LLM sigue siendo consultado después de Ψ₀.

No comparar latencias de implementaciones con hardware, batch, compilación o dimensiones diferentes sin reportarlo.

## Cierre de la rama

La rama estará lista para comparación con v4 cuando E45–E56 tengan un reporte reproducible y E57 pueda ejecutarse sobre un dataset sellado común.

La decisión de merge será posterior a la evidencia, no parte del objetivo experimental.
