# Motor Unificado CDT–Spin–RQM–EPR–Cognición

## 1. Resumen

El repositorio conserva el motor CDT-RQM-EPR histórico como compatibilidad y
añade un orquestador unificado cuya arquitectura vigente es:

```text
CDT simplicial pyrochlore
  -> líquido de espines XXZ
  -> RQM de amplitud/fase
  -> EPR predictivo
  -> capa cognitiva relacional
  -> gate de conocimiento consolidado
```

El resultado histórico del currículo legacy se conserva en este documento. La
nueva hipótesis evaluada es más limitada: la simetría guía cómo se comparte y
consolida aprendizaje, mientras la composición cognitiva ocurre en una capa
relacional más abstracta. No se afirma conciencia ni cognición general.

## 2. Arquitectura

```text
boundary + observador
  -> scoring relacional nativo
  -> pulsos térmicos CDT
  -> sincronización EPR opcional
  -> selección por energía libre
  -> sueño A (consolidación)
  -> sueño B (propuestas prospectivas)
  -> limpieza de residuo R/S/Z
  -> fluctuación geométrica débil
  -> puentes EPR verificados
  -> consolidación final + gate
```

Módulos del crate:

```text
src/native_thermodynamic_cdt.rs
src/native_thermo_rqm_epr.rs
src/native_thermodynamic_engine.rs
src/entanglement.rs
src/relational_field.rs   # ObserverId
src/residue_budget.rs
src/residue_vacuum_fluctuation.rs
src/residue_vacuum_bridge.rs
src/plasticity_controller.rs
```

## 3. Checkpoints

```text
data/native_thermo_clean.cdt_native
data/native_curriculum_5phase.cdt_native
data/native_massive_thermo.cdt_native
data/native_prototype_bridge_thermo.cdt_native
data/native_gemma_distilled_thermo.cdt_native
```

## 4. Inferencia Nativa

La consulta nativa combina:

```text
score = relational_score + thermal_score_gain * thermal_activation
```

La dinámica térmica actualiza estado, amplitud y fase locales. La poda por
energía libre elimina candidatos con alta fuga o alta energía residual, protegiendo
los nodos esperados.

## 5. Sueños A Y B

**Sueño A (consolidativo):** rejuega recuerdos, atenúa distractores y remotes
competidores. Reduce fuga sin destruir precisión.

**Sueño B (prospectivo):** genera futuros por puentes A→B→C de alto prior,
reserva cupo EPR reemplazando enlaces de menor utilidad, entrena con success bajo
y vuelve a aplicar Sueño A. Objetivo: bajar fuga (sobre todo fase 5) y crecer EPR útil sin perder
`broad_5phase_pass`.

```text
Sueño A: preservar accuracy, reducir leakage
Sueño B: reasignar EPR selectivo + consolidar
Controlador: limpiar residuo -> fluctuar localmente -> proponer -> consolidar -> gate
```

## 6. Currículo De 5 Fases

1. Cross-lingual (Gemma como periferia)
2. Composición multi-hop
3. Razonamiento fuerte con poda
4. Aprendizaje continuo / retención base + stream
5. Comparación contra baselines graph/lexical

Comando de evaluación amplia:

```powershell
$env:GEMMA_MODEL="gemma2:2b"; $env:CURRICULUM_EVAL_USE_GEMMA="true"; cargo run --release --bin native_curriculum_broad_evaluation
```

Checkpoint histórico evaluado (antes del entrenamiento prolongado):

```text
data/native_curriculum_5phase.cdt_native
cycle=113017
nodes=640
relations=6842
epr_links=620
```

## 7. Resultados De La Evaluación Amplia

```text
phase1_cross_lingual:      accuracy=100.0% leakage=6.6%
phase2_compositional:      accuracy=100.0% leakage=3.7%
phase3_strong_reasoning:   accuracy=100.0% leakage=0.6%
phase4_base_retention:     accuracy=100.0% leakage=11.7%
phase4_stream_retention:   accuracy=100.0% leakage=0.0%
phase5_native:             accuracy=100.0% leakage=8.1%
phase5_graph_baseline:     accuracy=0.0%   leakage=50.0%
phase5_lexical_baseline:   accuracy=0.0%   leakage=87.5%
decision=broad_5phase_pass
```

Estos valores son históricos y no deben confundirse con el checkpoint vigente.

Interpretación:

```text
Fase 1 pasa: transferencia multilingüe
Fase 2 pasa: composición multi-hop
Fase 3 pasa: razonamiento con poda
Fase 4 pasa: retención base y stream
Fase 5 pasa: ventaja clara sobre baselines
```

## 8. Estado Vigente Y Reparación De Retención

El entrenamiento prolongado llegó a `cycle=116347`, pero la evaluación amplia detectó
una regresión que el loop interno no veía:

```text
phase4_base_retention: accuracy=75.0% leakage=13.5%
phase5_native:         accuracy=100.0% leakage=9.5%
decision=broad_5phase_needs_tuning
```

La causa inmediata era una divergencia entre currículo y evaluación: el trainer omitía
`rain_story -> wet_ground` en fase 4 base y omitía el cuarto caso de fase 5. Tras alinear
los casos, ejecutar un ciclo reparador y activar el controlador de plasticidad:

```text
checkpoint=data/native_curriculum_5phase.cdt_native
cycle=116350
nodes=4480
relations=8192
epr_links=5292
phase1_cross_lingual:    accuracy=100.0% leakage=6.2%
phase2_compositional:    accuracy=100.0% leakage=4.7%
phase3_strong_reasoning: accuracy=100.0% leakage=0.0%
phase4_base_retention:   accuracy=100.0% leakage=0.0%
phase4_stream_retention: accuracy=100.0% leakage=0.0%
phase5_native:           accuracy=100.0% leakage=3.3%
decision=broad_5phase_pass
```

La evaluación fue repetida con y sin la periferia Gemma en el ciclo 116349. En ese ciclo,
Sueño B volvió a aceptar propuestas (`accepted=2`) y creó un enlace EPR neto (`+1`) sin
romper retención. El ciclo 116350 alineó también los fixtures faltantes de fases 2 y 3,
pasó el gate estricto, amplió el sustrato y redujo la fuga de fase 5 a 3.3%.

## 9. Residuo Como Presupuesto De Plasticidad

El residuo es una variable computacional, no una afirmación de equivalencia física:

```text
Z = suma local de pesos
S = -sum(p ln p)
R = 1 - p_elegido
```

El controlador mantiene cuatro etapas separadas:

1. **Limpieza:** atenúa caminos perdedores y fuga usando `R`.
2. **Fluctuación:** alquila una señal débil de activación/fase en regiones ambiguas.
3. **Propuesta:** detecta A→B→C con coherencia relacional y geométrica.
4. **Consolidación:** Sueño A y gate transaccional deciden si el cambio persiste.

EPR usa una cuota prospectiva por nodo. Cuando la capacidad está saturada, desactiva el
enlace de menor utilidad estimada (`coherence × (1-entropy) × (1-heat)`) antes de probar
un puente. Un candidato rechazado vuelve al estado original.

La ruta fue consolidada en `run_native_thermo_engine` y en el trainer de cinco fases.
Los bins A/B utilizados durante el desarrollo fueron eliminados; sus algoritmos y tests
permanecen como componentes del motor.

## 10. Comparación Antes Y Después

```text
Métrica                    ciclo 116347     ciclo 116350
phase1 leakage                 6.4%             6.2%
phase2 leakage                 4.2%             4.7%
phase3 leakage                 0.3%             0.0%
phase4 base accuracy          75.0%           100.0%
phase4 base leakage           13.5%             0.0%
phase4 stream accuracy       100.0%           100.0%
phase5 accuracy              100.0%           100.0%
phase5 leakage                 9.5%             3.3%
nodos                          4160             4480
relaciones                     7504             8192
enlaces EPR                    5028             5292
decisión               needs_tuning broad_5phase_pass
```

La mejora principal no es solo crecimiento: recupera el caso base omitido, reduce fuga
en razonamiento y fase 5, conserva todas las precisiones y vuelve transaccional el
aprendizaje continuo. El ligero aumento de fuga en fase 2 (`+0.5 pp`) permanece dentro
del gate y debe seguir monitorizándose.

La ejecución del motor consolidado sobre `data/native_thermo_clean.cdt_native` produjo:

```text
antes:
  accuracy=45.8% leakage=58.0% margin=5.524 relations=3878 epr=640
Sueño A:
  accepted=7 accuracy=100.0% leakage=0.1% margin=385.290 epr=652
PlasticityController:
  accepted=true cleanup=true fluctuation=true bridge=true
después del pipeline completo:
  accuracy=100.0% leakage=0.1% margin=904.396 relations=5990 epr=655
decision=native_thermo_stable_pass
```

La plasticidad mantuvo precisión y fuga, y elevó el margen posterior sin requerir
crecimiento EPR indiscriminado. La compactación elimina slots inactivos durante sueño,
reduciendo memoria e índices clonados sin eliminar enlaces activos.

## 11. Actualización EPR Y Sustrato En Tiempo Real

El motor expone dos APIs de escritura online:

```rust
train_observed_transition_realtime(...)
query_and_learn_realtime(...)
```

`RealtimeUpdateConfig` limita por evento:

```text
max_relation_updates
max_epr_observations
max_epr_evictions
epr_reserve_slots
max_window_nodes
thermal_microsteps
min_success
```

La ruta anterior ejecutaba un paso térmico global `O(nodos + aristas)` después de cada
observación. La ruta realtime modifica un máximo de pares y evoluciona solo el vecindario
causa/efecto. EPR mantiene grados activos en caché `O(1)`, reutiliza buffers de
sincronización y limita reemplazos por evento para evitar churn.

Benchmark release, 500 actualizaciones, 4480 nodos (rango de ejecuciones validadas):

```text
actualización global:
  mean=506-528 us
actualización realtime local:
  mean=44-47 us  throughput=21432-22567 ops/s
consulta realtime + EPR:
  mean=139.6-162.1 us  throughput=6169-7164 ops/s
reporte CDT completo:
  mean=33-39 us
query RQM solo relacional:
  mean=8-13 us  throughput=77651-124131 ops/s
speedup actualización=11-12x

nuevo enlace EPR:
  latency=9-11 us created=1 evicted=0 active=true

calidad global:
  accuracy=100.0% leakage=11.00%
calidad realtime:
  accuracy=100.0% leakage=10.65%
decision=realtime_optimization_pass
```

“Tiempo real” significa latencia acotada y presupuestada bajo un único escritor `&mut`;
no implica garantías hard-real-time del sistema operativo ni escrituras concurrentes
lock-free.

Cuellos de botella adicionales eliminados:

```text
observables CDT:
  seis recorridos de arrays -> una reducción fusionada
  secuencial <8192 nodos, Rayon para sustratos mayores
vecindario CDT:
  Vec::contains O(ventana²) -> marcas generacionales O(1)
scheduling térmico:
  scan schedule × impacted -> ranking precompilado + top-K determinista
scoring relacional:
  búsqueda lineal por candidato -> acumulador HashMap reutilizable
  sort completo de fan-out -> selección parcial top-K
fase de relaciones:
  scan de todas las relaciones -> lookup O(1)
EPR:
  grado por scan -> contador O(1)
  HashSets por sync -> buffers reutilizables
  summary O(enlaces) -> contador rápido cuando diagnostics=false
Sueño B:
  doble observe_correlation -> una observación con beneficio acumulado equivalente
probes de residuo:
  un clone completo por lección -> un clone reutilizado por ciclo
slots EPR inactivos:
  acumulación indefinida -> compactación durante sueño
```

Los clones completos requeridos para rollback de sueño, plasticidad y gates permanecen
fuera de la ruta realtime. Se conservan deliberadamente porque garantizan transacciones
seguras; sustituirlos requiere snapshots diferenciales con una validación separada.

## 12. Herramientas Conservadas

```powershell
cargo run --release --bin native_thermodynamic_engine
cargo run --release --bin native_thermo_clean_trainer
cargo run --release --bin native_curriculum_5phase_trainer
cargo run --release --bin native_curriculum_broad_evaluation
cargo run --release --bin native_massive_dataset_trainer
cargo run --release --bin native_massive_checkpoint_evaluation
cargo run --release --bin native_scientific_benchmark
cargo run --release --bin native_phase2_compositional_trainer
cargo run --release --bin native_phase3_strong_reasoning
cargo run --release --bin native_phase4_continual_learning
cargo run --release --bin native_phase5_baseline_comparison
cargo run --release --bin native_gemma_phase1_trainer
cargo run --release --bin native_realtime_benchmark
```

## 13. Gate Y Persistencia

El trainer ejecuta un gate determinista equivalente al núcleo de las cinco fases antes
de guardar. Exige retención base, precisión composicional y fuga de fase 5 dentro del
presupuesto. El guardado usa:

```text
serializar -> archivo temporal -> backup .bak -> commit
si falla -> rollback
```

Plasticidad y gate están activos por defecto. Para diagnósticos aislados pueden
desactivarse con `CURRICULUM_PLASTICITY=false` y
`CURRICULUM_CHECKPOINT_GATE=false`, respectivamente.

## 14. Decisión

```text
Esta decisión describe el cierre histórico del currículo CDT-RQM-EPR.
Desde la sección 15, el legacy permanece como baseline de compatibilidad y el
motor unificado es la entrada de producción para la nueva hipótesis.
```

Criterio de estabilidad vigente:

```text
accuracy alta
leakage baja
EPR creciente sin romper retención
ventaja sobre baselines
broad_5phase_pass
native_thermo_stable_pass
realtime_optimization_pass
```

## 15. Hipótesis De Cognición Operacional Y Simetría

### 15.1 Hipótesis

Se separan dos afirmaciones:

**H1 — cognición operacional emergente.** Una capa abstracta muestra una
capacidad nueva si compone `A→C` después de aprender únicamente `A→B` y `B→C`,
sin que exista una relación directa `A→C`.

**H2 — simetría como guía de consolidación.** La simetría no crea contenido
cognitivo. Determina qué relaciones equivalentes reciben transferencia y
participa en el gate que decide si una relación se convierte en conocimiento
persistente.

Estas hipótesis son computacionales y falsables dentro del modelo. No implican
conciencia, comprensión humana ni emergencia ontológica.

### 15.2 Arquitectura Evaluada

```text
CDT pyrochlore regular
  -> estado cuántico XXZ entrelazado
  -> RQM: amplitud, fase, coherencia, error
  -> EPR: enlaces por utilidad predictiva
  -> capa cognitiva: composición y abstención
  -> ConsolidatedKnowledge
```

El gate exige:

```text
topología CDT regular
AND entropía/testigo de entrelazamiento
AND relación RQM consolidada
AND error predictivo bajo
```

### 15.3 Protocolo Causal

`emergent_cognition_training::run_emergent_cognition_cycle` ejecuta 32 ensayos:

1. **Control vacío:** simetría perfecta sin relaciones debe abstenerse.
2. **Primera regla:** se entrena `A→B`; todavía no debe aparecer `A→C`.
3. **Segunda regla:** se entrena `B→C`; la capa cognitiva debe producir
   `A→B→C`.
4. **Control de almacenamiento:** se verifica que RQM no contiene una relación
   directa `A→C`.
5. **Órbita:** una relación no observada debe consolidarse sólo con transferencia
   simétrica activa.
6. **Ablación:** con confianza de simetría cero, la órbita no debe aparecer.
7. **Lesión CDT:** retirar un enlace físico debe bloquear conocimiento nuevo.
8. **Reparación:** restaurar el enlace debe volver a habilitar consolidación.
9. **OOD:** una señal sin relaciones debe causar abstención.

Comando reproducible:

```powershell
cargo test --release causal_cycle_supports_operational_emergent_cognition --lib -- --nocapture
```

### 15.4 Resultados

```text
ensayos                                             32
simetría sin relaciones -> abstención              100%
composición ausente tras una regla                  100%
composición A→B→C tras dos reglas                    100%
relación directa A→C ausente                        100%
transferencia de órbita con simetría                100%
órbita ausente sin transferencia                    100%
consolidación con CDT intacto                       100%
lesión CDT bloquea consolidación                    100%
reparación restaura consolidación                   100%
abstención OOD                                      100%

conocimiento medio:
  etapa 0                                             0
  después de A→B                                      1
  después de B→C                                      2
  final                                               5

decision=evidence_pass
```

### 15.5 Interpretación

Los resultados constituyen evidencia interna de una capacidad composicional que
no está almacenada como arista directa. En ese sentido operacional y limitado,
la conducta cognitiva aparece en la capa de composición.

La ablación y la lesión muestran que la simetría tiene un papel causal en la
transferencia y consolidación, pero el control vacío demuestra que no es
suficiente para producir contenido. Por tanto:

```text
simetría = sesgo y gate de aprendizaje
RQM/EPR = sustrato relacional
cognición operacional = composición/selección sobre relaciones
```

## 16. Comparación Con El Motor Legacy

En 24 ensayos pareados:

```text
métrica                         unificado    legacy
asociación directa                100%        100%
composición A→B→C                 100%        100%
retención                         100%        100%
abstención OOD                    100%        100%
transferencia de órbita           100%          0%
gate ante lesión CDT              100%          0%
tiempo total                    464.9 ms      1.15 ms
```

El motor unificado amplía generalización estructural y seguridad de
consolidación sin perder las capacidades compartidas. No mejora latencia: el
legacy, evaluado sin pasos térmicos, es aproximadamente 403 veces más rápido.
La comparación temporal no iguala física ni funcionalidad y no debe interpretarse
como benchmark de algoritmos equivalentes.

### 16.1 Checkpoint prolongado frente a legacy

El checkpoint durable de `221,600,000` ejemplos fue evaluado sin continuar el
entrenamiento:

```text
métrica                         entrenado    legacy
asociación directa                 100%        100%
composición A→B→C                  100%        100%
abstención OOD                     100%        100%
transferencia de órbita            100%          0%
cobertura conocimiento core        100%         N/D

relaciones                       12,400          72
conocimiento consolidado         12,400     sin gate
EPR                                 616          72
latencia 3,200 consultas        181.397 ms    1.364 ms
```

Decisión:

```text
FUNCTIONALLY_SUPERIOR_BUT_OVERCONSOLIDATED_AND_SLOWER
```

La selectividad `knowledge/relations = 100%` indica saturación: el gate terminó
aceptando todas las relaciones del espacio sintético. La prioridad deja de ser
añadir ejemplos repetidos y pasa a ser poda por utilidad holdout, indexación de
consulta y ampliación de la distribución.

## 17. Estado De La Evidencia

La suite completa contiene 81 pruebas unitarias y causales. Los resultados
respaldan capacidades operacionales dentro de fixtures sintéticos controlados.
Para afirmar cognición más general faltan tareas sensoriales complejas, ambientes
temporales, múltiples semillas y distribuciones, aprendizaje de simetrías no
proporcionadas y comparación con modelos de capacidad equivalente.

## 18. Entrenamiento Sintético Prolongado

Se añadió `native_unified_infinite_trainer`, un proceso reanudable por batches
con checkpoint completo del estado cuántico, RQM, EPR, conocimiento y RNG.

La corrida de validación corta produjo:

```text
primera evidencia composicional   batch 14
ejemplos                           2,800
composición                        100%
órbitas                            100%
OOD                                100%
relación directa compuesta ausente 100%
decision=EMERGENT_COGNITION_SUSTAINED
```

Al escalar sin enfriamiento, la capa relacional conservó composición y órbitas,
pero el testigo de entrelazamiento cayó y el gate global rechazó consolidación.
Este resultado refuta que la estabilidad relacional sea suficiente.

Se incorporó homeostasis cuántica adaptativa. Tras reanudar desde checkpoint:

```text
ejemplos              2,202,000
relaciones            12,400
composición           100%
órbitas               100%
OOD                    100%
enlaces entrelazados  7–8
gate                  true
```

La sesión fue detenida manualmente después de aproximadamente 3 h 10 min:

```text
último checkpoint durable          batch 1,108,000
ejemplos checkpoint                221,600,000
última validación observada        batch 1,108,200
ejemplos observados                221,640,000
composición                        100%
órbitas                            100%
OOD                                100%
relación directa compuesta ausente 100%
conocimiento consolidado           12,400
relaciones                         12,400
EPR                                602
entropía de spin                   0.692869
enlaces entrelazados               6
gate                               true
validaciones consecutivas          10,974
```

La tasa de gate posterior a homeostasis fue `99.9909%`; hubo un único fallo
transitorio registrado por pérdida total del testigo de entrelazamiento. El
checkpoint permite reanudar desde 221.6 millones de ejemplos. Los 200 batches
posteriores observados no estaban aún comprometidos y se descartan al reanudar.
