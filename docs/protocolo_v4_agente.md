# Protocolo de implementación — v4

**Rama:** `exp/field-autonomy-v4`

Este documento convierte `plan_autonomia_campo_v4.md` en una lista operativa para el siguiente agente.

## 1. Antes de programar

1. Confirmar branch `exp/field-autonomy-v4`.
2. Leer:
   - `docs/plan_autonomia_campo_v4.md`
   - `docs/plan_autonomia_campo_v3.md`
   - `docs/cierre_exp_field_autonomy_next_v3.md`
   - `docs/etapa_2_autonomia_campo_experimentos.md`
3. Inspeccionar `src/field_autonomy_stage2_v2.rs` y módulos de FieldEncoder/Dφ/CDT/RQM.
4. Confirmar SHA de `main` que sirve de base.
5. No traer archivos de `exp/field-autonomy-next` salvo documentación de referencia. Si se copia lógica, documentar por qué y verificar que no se copian pesos/datasets.

## 2. Primera implementación: infraestructura común

Crear, si no existe un equivalente limpio:

```text
src/field_autonomy_v4.rs
src/bin/run_field_autonomy_v4.rs
src/v4_dataset.rs
src/v4_provenance.rs
src/v4_controls.rs
src/v4_metrics.rs
```

Los nombres pueden adaptarse al estilo existente del repo, pero la separación conceptual debe mantenerse.

### `v4_dataset`

Responsabilidades:

- generar TRAIN/DEV/TEST;
- producir familias estructurales;
- evitar duplicados exactos;
- detectar equivalentes canónicos;
- emitir manifest con hash;
- sellar TEST antes de entrenar.

### `v4_provenance`

Responsabilidades:

- registrar todas las fuentes consultadas;
- marcar retrieval ON/OFF;
- marcar si target/equivalente apareció;
- generar `LEAKED` / `CLEAN`;
- hash del provenance por predicción.

### `v4_controls`

Implementar interfaces comunes para:

- static;
- linear;
- MLP;
- nearest-neighbor;
- random;
- simple dynamic/recurrent control;
- Dφ.

### `v4_metrics`

Mínimo:

- accuracy/error;
- cosine;
- energy;
- manifold distance;
- stability;
- norm drift;
- horizon error;
- parameter count;
- training examples;
- compute budget;
- latency.

## 3. E31 primero

Construir un benchmark donde una misma regla tenga muchas instancias independientes.

Importante: no usar solo A→B→C. Las instancias deben tener variación continua/discreta suficiente para que un lookup puntual sea inútil.

Ejemplo de estructura:

```text
rule_id
family_id
instance_id
parameter_vector
input_state
context
expected_state
canonical_target_hash
partition
```

El `canonical_target_hash` sirve para impedir que TEST aparezca de forma equivalente en TRAIN/DEV/CDT.

## 4. E32 Jacobian/stability

Implementar instrumentación sin cambiar la función científica de Dφ.

Registrar al menos:

```text
step
norm_z
norm_delta
cosine_target
energy
manifold_distance
transition_error
jacobian_sigma_max
perturbation_norm
```

Si la estimación exacta del Jacobiano es demasiado cara, implementar primero una estimación por power iteration/JVP o finite differences documentada.

No modificar Dφ para forzar `sigma < 1` hasta tener resultados diagnósticos.

## 5. E33 free-run

No permitir que el evaluator reinyecte el target verdadero después del primer step.

Pseudoflujo:

```text
z0 = Encoder(x0)
for t in 1..H:
    z = Dphi(z, context)
    record(z)
```

El target se usa únicamente para calcular métricas.

## 6. E34 paired controls

Crear una función de evaluación que reciba el mismo `RunContext` para todos los modelos.

El reporte debe mostrar:

```text
model
seed
parameter_count
train_examples
compute_budget
accuracy
cosine
energy
```

No comparar un Dφ de gran capacidad contra un static diminuto y llamar a la diferencia “dinámica”.

## 7. E35 — diseño de pipeline

Separar físicamente tres funciones:

```text
collect_experience(train) -> experiences
consolidate(experiences) -> learning_signal
adapt_dynamics(Dphi, learning_signal) -> Dphi_adapted
```

Luego:

```text
Dphi_adapted = adapt_dynamics(...)
DROP CDT
DROP RAM
DROP RQM
DROP retrieval indexes
TEST(Dphi_adapted)
```

El test debe fallar si alguna API de retrieval se invoca.

### Test crítico

Agregar un guard global:

```text
assert cdt_queries == 0
assert rqm_queries == 0
assert table_queries == 0
assert nn_queries == 0
assert attractor_queries == 0
```

para la condición científica E35.

## 8. E36

No basta con comparar “CDT sí/no”. Las condiciones deben conservar:

- misma cantidad de experiencias;
- mismo presupuesto de adaptación;
- misma arquitectura Dφ;
- mismo seed cuando sea paired.

CDT corrupto e irrelevante deben tener tamaño comparable para evitar confundir información con capacidad.

## 9. E37

Después de `adapt_dynamics`, serializar solo los artefactos permitidos. Ejecutar una fase de destrucción que elimine deliberadamente:

- CDT;
- RAM;
- RQM;
- índices de retrieval.

Luego crear un nuevo proceso/evaluator que no tenga acceso a esos objetos.

La prueba es especialmente importante: el mismo proceso no debe conservar accidentalmente referencias en memoria.

## 10. E38

Usar subprocess o proceso independiente si es posible. El evaluator de persistencia debe recibir explícitamente rutas a checkpoints y no el objeto Python/Rust de entrenamiento.

Comparar antes/después de restart con los mismos TEST y con TEST nuevo.

## 11. E39/E40/E41

Implementar después de E35–E38.

E39: familias nuevas.  
E40: secuencia de reglas + forgetting/transfer.  
E41: intervención + random control + rollback.

## 12. E42–E44

Solo después de los gates centrales.

E42 debe conservar el mismo protocolo a diferentes N.  
E43 debe separar representación lingüística de dinámica.  
E44 debe medir explícitamente cuánto cambia el espacio del encoder al cambiar de LLM.

## 13. Formato de resultados

Cada experimento debe producir una fila por:

`experiment × seed × condition × partition`.

CSV mínimo:

```text
experiment,seed,condition,partition,n_examples,params,compute,
accuracy,cosine,energy,manifold_distance,stability,
leakage,cdt_queries,rqm_queries,table_queries,nn_queries,
attractor_queries,direct_memory_queries,status
```

## 14. Estados válidos

- `CLEAN`
- `LEAKED`
- `DATASET_INVALID`
- `NOT_RUN`
- `SMOKE_ONLY`
- `PASS`
- `PARTIAL`
- `FAIL`

Nunca convertir `SMOKE_ONLY` en evidencia científica.

## 15. Criterio de reporte

Cada `summary.md` debe contener:

1. git SHA;
2. dataset manifest hash;
3. seeds;
4. hyperparameter lock;
5. conditions;
6. leakage summary;
7. metrics;
8. controls;
9. failures;
10. interpretación limitada al experimento.

## 16. Qué NO hacer

- No aumentar epochs para rescatar un resultado antes de diagnosticar.
- No añadir una ruta RQM oculta para mejorar E22/E35.
- No usar CDT retrieval durante TEST y llamarlo “memoria como experiencia”.
- No seleccionar seeds favorables.
- No descartar seeds malas sin causa registrada.
- No cambiar decoder después de ver TEST.
- No convertir un benchmark pequeño en una afirmación general sobre cognición.

## 17. Orden de commits sugerido

```text
1. docs: add v4 plan/protocol
2. feat: add v4 dataset manifests
3. feat: add v4 provenance guard
4. feat: add E31 rule suite
5. feat: add E32 dynamics diagnostics
6. feat: add E33 free-run evaluator
7. feat: add E34 paired controls
8. feat: add E35 CDT experience pipeline
9. feat: add E36/E37 ablations
10. feat: add E38 persistence
11. feat: add E39-E41 transfer/causal
12. feat: add E42-E44 scaling/LLM tests
13. docs: stamp results and close/open gates
```

Cada commit de resultados debe indicar si los números son smoke, DEV o CONFIRM.

## 18. E45 (Liquid vs Thermo) — protocolo

Módulo `src/v4_e45.rs`; runner `run_field_autonomy_v4 run --phase dev|confirm --exps E45 --out artifacts/v4/e45`. Mismas seeds DEV/CONFIRM v4, mismo lock (`steps` totales repartidos 50/50 por fase). Pre-registro en `plan_autonomia_campo_v4.md §24` y `preregistro_v4.md §7`, commiteado antes de correr.
