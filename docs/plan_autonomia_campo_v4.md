# Plan experimental — autonomía del campo v4

**Rama:** `exp/field-autonomy-v4`  
**Base:** `main`  
**Fecha:** 2026-10-06  
**Estado:** especificación para implementación por otro agente

> v4 no pretende añadir más complejidad al campo por defecto. Reorganiza la siguiente batería alrededor de las preguntas que v3 dejó abiertas: si Dφ aprende reglas reales, si CDT puede actuar como experiencia en vez de lookup, si la dinámica es estable a largo horizonte, y si la capacidad persiste después de eliminar el estado episódico.

## 1. Punto de partida científico

El ciclo v3 confirmó en seeds de confirmación los gates E18/E21/E22/E24, mientras E23, E25, E26, E27, E28, E29 y E30 permanecieron abiertos. La evidencia de v3 no debe reinterpretarse como prueba de autonomía completa.

La hipótesis central de v4 es:

`experiencias → consolidación CDT → regularidad/learning signal → adaptación de Dφ → dinámica sobre X_new → Y_new jamás almacenado`

La hipótesis rival es:

`X_new → recuperación/vecindad/tabla/CDT/RQM → Y_new`

o bien:

`X_new → interpolación trivial del encoder/decoder → Y_new`

v4 debe diseñarse para separar estas explicaciones.

## 2. Principios obligatorios

1. **Clean-room:** TRAIN/DEV/TEST generados y sellados antes de la confirmación.
2. **No reutilización:** no usar pesos, checkpoints, prototypes, tablas, CDT, RQM ni datasets de E11–E17/v3 como entrenamiento.
3. **FIELD_ONLY:** el benchmark científico principal tiene RQM, CDT retrieval, tablas, NN y attractor bank desactivados durante TEST.
4. **Provenance:** cada predicción debe registrar el camino de información.
5. **No teacher forcing en pruebas dinámicas.**
6. **Controles fuertes:** static, linear, MLP, nearest-neighbor, random y simple recurrent/dynamic control.
7. **Presupuesto emparejado:** comparar modelos con parámetros, ejemplos y compute documentados.
8. **No retune sobre TEST.** Si se cambia arquitectura/hiperparámetros después de observar TEST, se invalida TEST y se genera otro split.
9. **Independent decoder:** cuando el experimento lo permita, el decoder no puede codificar respuestas de TEST ni compartir lookup con CDT/RQM.
10. **No claims de AGI/consciencia.** El resultado máximo depende de los gates realmente cerrados.

## 3. Nueva jerarquía de experimentos

### Fase V4-A — fundamentos de dinámica

- **E31:** rule-learning suite 2.0, multi-familia.
- **E32:** estabilidad dinámica/Jacobiano.
- **E33:** long-horizon free-run con perturbaciones.
- **E34:** static-vs-dynamic paired benchmark con capacidad igualada.

### Fase V4-B — CDT como experiencia

- **E35:** CDT-as-experience causal pipeline.
- **E36:** ablación de consolidación.
- **E37:** delete-after-learning / memory amnesia.
- **E38:** persistence after restart.

### Fase V4-C — transferencia y aprendizaje continuo

- **E39:** transfer de regla entre familias.
- **E40:** continual learning + forgetting.
- **E41:** causal intervention y rollback.

### Fase V4-D — escalamiento y dependencia lingüística

- **E42:** scaling N=8/16/32/64/128.
- **E43:** cross-lingual semantic transfer.
- **E44:** LLM replacement / encoder swap.

No ejecutar E42–E44 como prioridad si E35 no supera el control de no-consolidación.

---

# 4. E31 — Rule Learning 2.0

## Objetivo
Demostrar que Dφ aprende una transformación/regla y no una colección de trayectorias.

## Familias

A. traslación 2D/3D  
B. rotación  
C. reflexión  
D. escala  
E. shear  
F. afín  
G. composición R2(R1(x))  
H. perturbación suave/no lineal

Cada familia debe tener múltiples instancias, objetos y parámetros.

## Split estructural

TRAIN: reglas y objetos base.  
DEV: parámetros/objetos diferentes de TRAIN.  
TEST: familia de objetos y combinaciones nunca observadas; targets y equivalentes no almacenados.

## Controles

- static encoder + decoder;
- linear predictor;
- MLP predictor;
- Dφ;
- nearest-neighbor;
- shuffled-label control;
- random dynamics control.

## Gate
Dφ debe superar al static y a los controles simples en al menos 3 familias y en la mayoría de seeds, con leakage=0. No se acepta una sola familia simbólica como prueba general.

---

# 5. E32 — estabilidad local de Dφ

El fallo E23 de v3 puede ser acumulación de error. Antes de añadir más capacidad, medir la geometría local.

Para una transición `z' = Dφ(z,c)` registrar por step:

- `||z||`;
- `Δz`;
- cosine con target;
- energía;
- distancia al manifold;
- error de transición;
- estimación de `σ_max(J)` del Jacobiano;
- sensibilidad a `ε`.

Probar `ε ∈ {0.001, 0.01, 0.05, 0.10, 0.20}`.

Interpretación:

- `σ_max(J) < 1` localmente favorece contracción/estabilidad;
- `σ_max(J) > 1` de forma sostenida puede explicar amplificación de error;
- no imponer artificialmente contractividad: medir primero.

Entregable: curvas por horizon y seed, no solo accuracy final.

---

# 6. E33 — free-run largo sin teacher forcing

Entrenar one-step. Evaluar exclusivamente free-run:

`1,2,4,8,16,32,64` steps.

Separar:

1. rollout sobre reglas vistas;
2. rollout sobre objetos nuevos;
3. rollout con parámetros nuevos;
4. rollout con perturbación inicial.

Registrar error acumulado, cosine, energía, norm drift, manifold distance y divergencia.

Gate: demostrar una región de estabilidad reproducible y no solo un buen primer paso. Si h32/h64 fallan, conservar el diagnóstico y no maquillar la métrica mediante teacher forcing.

---

# 7. E34 — static vs dynamic estrictamente emparejado

Usar exactamente el mismo:

- dataset;
- seed;
- encoder initialization;
- decoder initialization cuando sea comparable;
- número de parámetros dentro de un rango documentado;
- optimizer;
- training examples;
- compute budget.

Comparar:

`STATIC(x)=Decoder(Encoder(x))`

contra

`DYNAMIC(x,c)=Decoder(Dφ(Encoder(x),c))`.

Añadir un MLP con capacidad aproximadamente equivalente.

La pregunta no es si Dφ gana a un baseline débil, sino qué capacidad aparece específicamente por dinámica iterativa.

---

# 8. E35 — EXPERIMENTO CENTRAL: CDT como experiencia

Este es el gate principal de v4.

## Fase A — Experience acquisition

Generar experiencias `E_i=(x_i,c_i,y_i,context_i)` en TRAIN.

No guardar respuestas TEST.

## Fase B — Consolidación

CDT recibe solo experiencias TRAIN y debe producir un artefacto de consolidación. Este artefacto puede contener:

- estadísticas agregadas;
- estructura de transición;
- parámetros/reglas estimadas;
- correlaciones;
- distribución;
- topología;
- confianza/incertidumbre.

No puede contener un índice `x_test → y_test`.

## Fase C — Adaptación

Usar el resultado de consolidación para entrenar/adaptar Dφ.

La adaptación debe ocurrir **antes** de TEST.

## Fase D — eliminación del CDT

Antes de TEST:

- desconectar CDT;
- vaciar RAM episódica;
- desactivar RQM;
- desactivar table/NN/attractor retrieval.

La predicción debe ser:

`X_new → Encoder → Dφ → independent decoder → Y_new`.

## Gate E35

El modelo con experiencias consolidadas debe superar significativamente al modelo idéntico sin experiencias, mientras el CDT ya no está disponible durante TEST y `target_seen_* = false`.

Este es el experimento que puede distinguir **memoria como experiencia** de **memoria como respuesta**.

---

# 9. E36 — ablación causal de consolidación

Comparar al menos:

A. sin experiencia;
B. experiencia cruda sin consolidar;
C. experiencia consolidada;
D. CDT corrupto;
E. CDT irrelevante pero del mismo tamaño;
F. estadísticas agregadas sin topología;
G. consolidación con orden de experiencias barajado.

Misma cantidad de experiencia y mismo presupuesto posterior.

Hipótesis esperada:

`C > B > A` en generalización, pero C debe producir targets nunca almacenados.

Si C solo mejora retrieval, el resultado es negativo para la hipótesis central.

---

# 10. E37 — delete-after-learning / amnesia experimental

Secuencia:

1. aprender experiencias;
2. consolidar CDT;
3. adaptar Dφ;
4. guardar Dφ;
5. borrar CDT;
6. borrar RAM;
7. borrar RQM;
8. ejecutar TEST.

Condición adicional: reconstruir el proceso en otro proceso limpio cargando únicamente los pesos/adaptaciones permitidos.

Gate: rendimiento post-delete cercano al pre-delete. Si cae a baseline, CDT no produjo aprendizaje persistente suficiente.

---

# 11. E38 — persistencia real

Guardar artefactos separados:

- encoder checkpoint;
- Dφ checkpoint;
- decoder checkpoint;
- CDT artifact;
- manifest;
- metadata de versión.

Reiniciar proceso y evaluar:

P0 encoder+Dφ+decoder;  
P1 encoder+Dφ+decoder+CDT pero retrieval OFF;  
P2 Dφ+decoder;  
P3 CDT retrieval control.

La ruta científica es P0/P2. P3 es únicamente control de lookup.

---

# 12. E39 — transferencia entre familias

Aprender una regla sobre familia A, consolidar y adaptar Dφ. Evaluar la misma regularidad sobre familia B estructuralmente nueva.

Ejemplo:

TRAIN: círculos bajo T.  
TEST: triángulos bajo T.

La salida de B no puede existir en CDT ni en cualquier lookup.

Gate: mejora sobre no-consolidado y static.

---

# 13. E40 — continual learning

Secuencia de reglas:

`R1 → consolidate → adapt → R2 → consolidate → adapt → ... → R4`.

Después evaluar R1–R4.

Métricas:

- accuracy por regla;
- forgetting;
- forward transfer;
- backward transfer;
- interference;
- tamaño CDT;
- número de updates.

Añadir una condición CDT corrupto y una irrelevante para demostrar que el beneficio no es solo regularización genérica.

---

# 14. E41 — intervención causal

Guardar checkpoint baseline.

Intervenir:

A. componentes estructurados de Dφ;  
B. componentes aleatorios de igual magnitud.

Medir cambio en predicciones y dinámica.

Después rollback exacto.

Gate:

- intervención estructurada produce efecto reproducible;
- control aleatorio produce distribución distinta;
- rollback recupera baseline.

---

# 15. E42 — scaling

Escalar conceptos/reglas: `N=8,16,32,64,128`.

No aumentar únicamente epochs. Mantener protocolo y reportar:

- capacidad efectiva;
- accuracy;
- margin;
- energy gap;
- samples-to-learn;
- compute;
- memoria CDT;
- estabilidad.

Comparar contra Hopfield/Hebb o controles disponibles en el repositorio cuando sea técnicamente comparable.

---

# 16. E43 — transferencia lingüística

Mantener el LLM congelado.

Entrenar con español y evaluar equivalentes en inglés/francés/japonés, pero aumentar el benchmark respecto E12:

- singular/plural;
- género;
- paráfrasis;
- contexto completo;
- oraciones;
- distractores.

Separar claramente:

1. invariancia del encoder;
2. aprendizaje de la dinámica;
3. recuperación lingüística del decoder.

No usar esto como evidencia de cognición si falla el benchmark no lingüístico.

---

# 17. E44 — LLM swap

Entrenar el campo con un encoder/periferia congelado A y evaluar con periferia B bajo un alineamiento explícito y documentado.

Comparar:

- mismo LLM;
- LLM cambiado;
- encoder aleatorio controlado.

La tesis fuerte requiere que la estructura aprendida por el campo sobreviva a un cambio de periferia razonable.

---

# 18. Protocolo estadístico

Mínimo:

- 16 seeds DEV;
- 16 seeds CONFIRM;
- intervalos de confianza bootstrap;
- mediana y media;
- distribución completa por seed;
- effect size frente a controles;
- paired tests cuando corresponda.

No resumir únicamente con el mejor seed.

Para E35–E41, usar exactamente los mismos seeds entre condiciones cuando sea posible.

---

# 19. Auditoría de información

Cada predicción debe registrar:

`target_seen_training`  
`target_seen_dev`  
`target_seen_cdt`  
`target_seen_rqm`  
`target_seen_table`  
`target_seen_nn`  
`target_seen_attractor`  
`target_equivalent_seen`  
`cdt_queries`  
`rqm_queries`  
`table_queries`  
`nn_queries`  
`attractor_queries`  
`direct_memory_queries`  
`decoder_lookup`  
`provenance_hash`

Cualquier camino directo al target marca `LEAKED` y excluye la fila.

---

# 20. Artefactos obligatorios por experimento

Cada ejecución debe producir:

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

`config.json` debe incluir git SHA, branch, seed, hyperparameters, model sizes y budget.

`dataset_manifest.json` debe contener hashes y particiones.

`provenance.jsonl` debe permitir reconstruir el camino de información de cada predicción.

---

# 21. Orden de implementación recomendado

1. E31 + auditoría común.
2. E32 + instrumentación Jacobiana.
3. E33.
4. E34.
5. E35.
6. E36.
7. E37.
8. E38.
9. E39/E40.
10. E41.
11. E42.
12. E43/E44.

**Regla:** si E35 falla, no interpretar E42–E44 como evidencia de experiencia consolidada. Si E33 falla, no ocultar el problema aumentando capacidad sin diagnóstico.

---

# 22. Criterio de éxito de v4

No existe un único PASS global. Se deben cerrar gates independientes.

### Gate cognitivo principal

E35 + E36 + E37:

`experiencia → consolidación → adaptación Dφ → CDT eliminado → target nuevo`

con leakage=0, múltiples seeds, controles y mejora estadísticamente clara.

### Gate dinámico

E32 + E33:

estabilidad y generalización de rollout a horizonte largo, con diagnóstico de Jacobiano.

### Gate causal

E41:

intervención estructurada ≠ intervención aleatoria + rollback.

### Gate persistente

E38:

capacidad sobrevive restart y eliminación de memoria episódica.

### Gate transferencia

E39 + E43/E44:

regularidad transferible más allá de la familia/periferia usada para aprender.

Solo si varios gates convergen se podrá afirmar que existe evidencia fuerte de un **sustrato externo que aprende dinámicas y consolida experiencia**. Incluso entonces no se sigue que exista consciencia, subjetividad o AGI.

---

# 23. Regla para el agente implementador

El agente que continúe esta rama debe:

- leer primero este documento y `docs/plan_autonomia_campo_v3.md`;
- inspeccionar `main` antes de tocar código;
- no copiar checkpoints de ramas experimentales;
- implementar primero infraestructura de auditoría/provenance;
- separar benchmark, modelo, controles y reportería;
- ejecutar smoke tests antes de suites completas;
- no reportar resultados no ejecutados;
- no modificar TEST después de observarlo;
- registrar cada commit relevante en el handoff v4.
