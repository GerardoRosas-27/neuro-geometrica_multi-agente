# Plan experimental — autonomía del campo v4

**Rama:** `exp/field-autonomy-v4`  
**Base:** `main`  
**Fecha:** 2026-10-09  
**Estado:** especificación detallada para implementación por otro agente

> v4 no intenta rescatar E23 con más epochs ni añadir otro mecanismo de lookup. Su objetivo es separar recuperación, interpolación estática, dinámica aprendida y aprendizaje persistente a partir de experiencia consolidada.

## 0. Estado de la base

El Clean-Room v3.7 **sí está integrado en `main`** mediante PR #29 (`c59b9fe`, 2026-09-26). Algunos documentos históricos conservan frases antiguas como «no mergeado»; no deben interpretarse como estado actual.

La base científica de v4 es `main`. No reutilizar pesos, datasets, checkpoints ni artefactos entrenados de E11–E17/v3 como entrenamiento.

## 1. Pregunta científica central

E18/E21/E22/E24 de v3 aportaron evidencia de que Dφ aprende reglas, generaliza parámetros, compone transformaciones y supera controles estáticos bajo el clean-room definido. La pregunta decisiva de v4 es:

> **¿Puede una experiencia previa, consolidada por CDT, modificar persistentemente la dinámica Dφ de modo que, después de eliminar CDT/RQM/RAM y cualquier retrieval, Dφ resuelva estados nuevos cuya respuesta jamás estuvo almacenada?**

Hipótesis H1:

```text
experiencias TRAIN
      ↓
consolidación CDT
      ↓
regularidad / learning signal
      ↓
adaptación de Dφ
      ↓
ELIMINAR CDT + RAM + RQM + retrieval
      ↓
X_new → Dφ → Y_new jamás almacenado
```

Hipótesis rivales:

```text
H0a: X_new → lookup → Y_new
H0b: X_new → NN/interpolación estática → Y_new
H0c: X_new → Dφ preexistente → Y_new
H0d: decoder/encoder contiene implícitamente la respuesta
H0e: el beneficio es regularización genérica y no información procedente de CDT
```

v4 debe distinguirlas experimentalmente.

## 2. Principios obligatorios

1. TRAIN/DEV/TEST generados y sellados antes de CONFIRM.
2. No reutilizar pesos, checkpoints, prototypes, tablas, CDT, RQM ni datasets de E11–E17/v3 como entrenamiento.
3. TEST debe auditarse mediante hashes exactos y hashes canónicos/equivalentes.
4. Benchmark principal FIELD_ONLY.
5. En TEST científico E35–E41: `cdt_queries=0`, `rqm_queries=0`, `table_queries=0`, `nn_queries=0`, `attractor_queries=0`, `direct_memory_queries=0`.
6. No teacher forcing durante free-run.
7. Controles con mismo dataset, seed y presupuesto razonable.
8. No retune después de observar TEST.
9. Decoder independiente cuando sea posible.
10. Una ruta directa al target/equivalente marca `LEAKED`.
11. No seleccionar seeds favorables.
12. No claims de AGI/consciencia/cognición general.

## 3. Arquitectura experimental

```text
Periferia/encoder → representación z → Dφ → estado z' → decoder independiente → respuesta
```

Fuera del hot path de TEST:

```text
experiencias → CDT consolidation → learning_signal → adapt(Dφ)
```

CDT no puede ser un diccionario de respuestas. El `learning_signal` puede ser una distribución, regla, estadística, matriz, parámetro, estructura relacional, gradiente o representación agregada, pero nunca `x_test → y_test`.

## 4. Batería

### V4-A — dinámica intrínseca
- **E31:** rule-learning 2.0 multi-familia.
- **E32:** estabilidad local/Jacobiano.
- **E33:** free-run largo.
- **E34:** static vs dynamic emparejado.

### V4-B — memoria como experiencia
- **E35:** CDT → learning signal → adaptación Dφ → delete CDT → TEST.
- **E36:** ablación causal de consolidación.
- **E37:** delete-after-learning/amnesia.
- **E38:** persistencia tras restart.

### V4-C — transferencia
- **E39:** transferencia entre familias.
- **E40:** continual learning/forgetting.
- **E41:** intervención causal + rollback.

### V4-D — límites
- **E42:** N=8/16/32/64/128.
- **E43:** transferencia lingüística ampliada.
- **E44:** cambio de LLM/encoder.

Orden: E31→E34, luego E35→E38, luego E39→E41 y finalmente E42→E44. Si E35 no supera su control, E42–E44 no se presentan como confirmación de la hipótesis central.

# 5. E31 — Rule Learning 2.0

## Objetivo
Probar que Dφ aprende una regularidad reutilizable y no una trayectoria/tabla.

## Familias mínimas

Traslación 2D/3D, rotación, reflexión, escala, shear, afín, composición de transformaciones y una familia suave/no lineal.

Cada familia debe tener múltiples objetos, parámetros continuos y contextos. Un único A→B→C no es suficiente.

## Split

TRAIN: objetos/parámetros base. DEV: distintos. TEST: objetos, parámetros o composiciones nunca vistas. Además de hash exacto, usar `canonical_target_hash` para equivalentes algebraicos.

## Controles

STATIC, LINEAR, MLP de capacidad emparejada, Dφ, nearest-neighbor, random dynamics y shuffled-label.

## Gate

Dφ supera STATIC y controles simples en al menos 3 familias, mayoría de seeds, leakage=0. Reportar effect size además de accuracy.

# 6. E32 — estabilidad de Dφ

E23 v3 fue débil en horizontes largos. Antes de modificar Dφ hay que medir por qué.

Registrar por paso:

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

Estimar `σ_max(J)` mediante JVP/power iteration o finite differences documentadas. Perturbaciones: ε={0.001,0.01,0.05,0.10,0.20}.

No imponer contractividad antes de medirla. `σ_max(J)>1` sostenido puede explicar amplificación de error; `σ_max(J)<1` localmente favorece contracción, pero no es por sí mismo prueba de buen aprendizaje.

# 7. E33 — free-run h1…h64

Entrenar one-step y evaluar sin reinyectar targets: `H={1,2,4,8,16,32,64}`.

Separar reglas vistas, objetos nuevos, parámetros nuevos y perturbación inicial. Reportar error, cosine, energía, norma, manifold distance y divergencia por horizonte.

Gate: región de estabilidad reproducible. Si h32/h64 fallan, conservar el diagnóstico; no ocultarlo con teacher forcing.

# 8. E34 — static vs dynamic emparejado

Mismo dataset, seed, inicialización cuando sea comparable, optimizer, ejemplos y presupuesto de compute. Parámetros dentro de un rango previamente definido.

```text
STATIC(x) = Decoder(Encoder(x))
DYNAMIC(x,c) = Decoder(Dφ(Encoder(x),c))
```

Añadir MLP de capacidad comparable. La pregunta es qué comportamiento aparece específicamente por dinámica y no por más capacidad.

# 9. E35 — GATE CENTRAL: CDT como experiencia

Crear dos sistemas idénticos desde la misma inicialización:

### A — control

```text
TRAIN → Dφ_A
```

### B — experiencia consolidada

```text
TRAIN experiences
→ CDT consolidation
→ learning_signal
→ adapt Dφ_B
→ DROP CDT/RAM/RQM/retrieval
```

Ambos reciben el mismo TEST sellado.

## Prohibiciones

El learning signal no puede contener target TEST, equivalente canónico, índice input→target, prototype directo, ruta RQM, tabla de respuestas ni embedding almacenado que permita recuperar directamente el target.

## Información permitida

CDT puede producir parámetros de regla, estadísticas agregadas, matriz de transición, distribución, topología, correlaciones, incertidumbre o señales de actualización de Dφ, siempre que la auditoría demuestre que no codifican respuestas TEST directamente.

## Protocolo

```text
1. generar TRAIN/DEV/TEST
2. sellar manifest
3. entrenar control A
4. recolectar experiencias TRAIN
5. consolidar B en CDT
6. producir learning_signal
7. adaptar Dφ_B
8. serializar Dφ_B
9. destruir CDT/RAM/RQM/indexes
10. iniciar evaluator limpio
11. cargar solo artefactos permitidos
12. ejecutar TEST
13. verificar queries=0
14. comparar A vs B
```

## Gate

```text
performance(B) > performance(A)
leakage = 0
cdt_queries = 0
rqm_queries = 0
lookup_queries = 0
target_seen_cdt = false
target_equivalent_seen = false
```

La comparación debe ser paired por seed y acompañada de effect size/intervalo. Si B no supera A, la hipótesis queda sin evidencia positiva bajo este protocolo.

# 10. E36 — ablación de consolidación

Condiciones: A sin experiencia; B experiencia cruda; C consolidada; D CDT corrupto; E CDT irrelevante de igual tamaño; F estadísticas sin topología; G orden barajado.

Mantener cantidad de experiencias y presupuesto posterior. Si C solo mejora retrieval, es evidencia negativa para memoria como experiencia.

# 11. E37 — delete-after-learning

```text
experiencia → CDT → adapt Dφ → save Dφ
                                  ↓
                         DELETE CDT/RAM/RQM
                                  ↓
                             TEST limpio
```

Ejecutar en proceso nuevo. Gate: rendimiento post-delete cercano al pre-delete y por encima del control sin experiencia.

# 12. E38 — persistencia tras restart

Guardar encoder, Dφ, decoder, CDT, manifest y metadata por separado.

P0=encoder+Dφ+decoder con retrieval OFF.  
P1=encoder+Dφ+decoder+CDT con retrieval OFF.  
P2=Dφ+decoder cuando el protocolo lo permita.  
P3=CDT retrieval como control positivo de lookup.

La evidencia de aprendizaje persistente debe aparecer en P0/P2, no depender de P3.

# 13. E39 — transferencia

Aprender regla en familia A y probar la misma regularidad en B estructuralmente nueva. Ejemplo: círculos→triángulos bajo la misma transformación.

Gate: B consolidado supera no-consolidado y static, sin target B en CDT.

# 14. E40 — continual learning

Secuencia `R1→R2→R3→R4`, consolidando/adaptando tras cada bloque. Medir accuracy por regla, forgetting, forward/backward transfer, interference, tamaño CDT y updates. Añadir CDT corrupto/irrelevante.

# 15. E41 — intervención causal

Guardar checkpoint baseline. Intervenir componentes estructurados de Dφ y, como control, componentes aleatorios de igual magnitud. Medir cambio en trayectoria, predicción, energía y estabilidad. Hacer rollback exacto.

Gate: intervención estructurada reproducible, control aleatorio con distribución distinta y rollback recupera baseline.

# 16. E42 — scaling

`N={8,16,32,64,128}`. No compensar únicamente con epochs. Reportar capacidad efectiva, accuracy, margin, energy gap, samples-to-learn, compute, memoria CDT y estabilidad.

# 17. E43 — transferencia lingüística

LLM congelado. Entrenar en español y evaluar inglés/francés/japonés, con plural, género, paráfrasis, contexto, oraciones y distractores.

Separar: invariancia del encoder, dinámica y decoder lingüístico. No usar E43 como evidencia general si falla E31 no lingüístico.

# 18. E44 — cambio de LLM

Entrenar con periferia A y evaluar con B mediante alineamiento explícito y predefinido. Comparar mismo LLM, LLM cambiado y encoder aleatorio controlado.

La hipótesis fuerte requiere conservar estructura del campo frente a un cambio razonable de periferia.

# 19. Estadística

Mínimo 16 seeds DEV + 16 CONFIRM. Mismos seeds entre condiciones paired. Reportar media, mediana, dispersión, bootstrap CI y effect size. No publicar solo el mejor seed.

# 20. Auditoría de información

Cada predicción registra:

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
provenance_hash
```

Una ruta directa al target marca `LEAKED` y excluye la corrida científica.

# 21. Artefactos obligatorios

```text
artifacts/v4/<experiment>/<run>/
  config.json
  dataset_manifest.json
  provenance.jsonl
  metrics.csv
  summary.md
  checkpoints/
  plots/
```

`config.json`: git SHA, branch, seed, hiperparámetros, tamaños y budget. `dataset_manifest.json`: hashes/particiones. `provenance.jsonl`: camino de información por predicción.

# 22. Gates iniciales

| Gate | Estado inicial |
|---|---|
| E31 rule learning | NOT_RUN |
| E32 stability | NOT_RUN |
| E33 long rollout | NOT_RUN |
| E34 static/dynamic | NOT_RUN |
| E35 CDT experience | NOT_RUN |
| E36 consolidation ablation | NOT_RUN |
| E37 delete-after-learning | NOT_RUN |
| E38 persistence | NOT_RUN |
| E39 transfer | NOT_RUN |
| E40 continual learning | NOT_RUN |
| E41 causal intervention | NOT_RUN |
| E42 scaling | NOT_RUN |
| E43 language | NOT_RUN |
| E44 LLM swap | NOT_RUN |

Solo un resultado reproducible y auditado puede cambiar `NOT_RUN` a `PASS`.

## 23. Orden de implementación

1. dataset + manifests;
2. provenance/auditoría;
3. E31;
4. E32;
5. E33;
6. E34;
7. E35;
8. E36;
9. E37;
10. E38;
11. E39–E41;
12. E42;
13. E43–E44.

**Regla v4:** no hacer que el campo acierte más a cualquier precio. Hacer imposible confundir recuperación, interpolación trivial, capacidad del decoder o memorización con aprendizaje dinámico persistente.
