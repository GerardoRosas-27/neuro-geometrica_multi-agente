# Etapa 2 — Autonomía del campo: dinámica, reglas y generación de estados no almacenados

## Objetivo científico

E11/E12 ya dan evidencia de que un FieldEncoder entrenable puede modificar la geometría y reducir variación lingüística usando estados reales de Gemma GGUF. E13/E15 muestran el cuello de botella: representar conceptos no implica aprender relaciones ni composición. E16/E17 muestran una señal prometedora de dinámica, pero E17 aún tiene contaminación por RQM composition.

La pregunta central de esta etapa es:

> ¿Puede el campo aprender una regla o dinámica relacional que produzca estados correctos que nunca fueron almacenados, sin depender de RQM, CDT, tablas, attractor banks ni memoria directa?

Separar estrictamente:
1. representación: FieldEncoder;
2. dinámica: D_phi;
3. memoria: CDT/RQM/engramas.

---

## Diagnóstico E11–E17

### E11 — FieldEncoder
Resultado positivo pero variable. Hay semillas fuertes y una débil. La mejora causal indica que el encoder modifica la geometría, pero no demuestra semántica general.

Mejorar con 8–16 semillas, baseline Gemma raw, proyector aleatorio, embeddings sin entrenamiento, bootstrap/CI y hashes de dataset/checkpoint.

### E12 — Invariancia lingüística
Es el resultado más limpio: con Gemma GGUF real y entrenamiento español, 8/8 obtienen seen=1, unseen=1 y OOD=1, con margins aproximados 0.69–0.75.

Interpretación correcta: evidencia de invariancia lingüística en este benchmark; no prueba una semántica independiente del lenguaje en general.

Ampliar a sinónimos, plural/género, diminutivos, paráfrasis, frases completas, ruido ortográfico, conceptos cercanos y más idiomas.

### E13 — Relaciones
Es el cuello de botella. Tabla funciona, RQM funciona parcialmente y static/dynamic siguen bajos.

Conclusión: no seguir priorizando el encoder. Hay que enseñar y medir reglas relacionales.

### E14 — Continual learning
8/8 PASS, pero usa fixed prototypes + micro-rehearsal + local biases. Aún no demuestra plasticidad natural del campo.

Hacer ablaciones de cada estabilizador y una variante sin todos.

### E15 — Estructura sin etiquetas
PARTIAL/FAIL. Las distancias OK/BAD son demasiado parecidas y structure suele 0/3.

Es un resultado negativo útil: geometría no equivale a composicionalidad. Sustituir el benchmark pequeño por reglas generativas.

### E16 — Dynamic Field
Varias semillas muestran cos_B=1, rollB=1, rollC=1 y rollD=1, pero existen semillas débiles y a veces static es comparable.

Necesita un control pareado static vs dynamic.

### E17 — Estado nunca almacenado
Es una señal importante: targets ausentes de train/CDT/attractor/direct RQM, con distancias pequeñas. Pero audit_rqm_compose sigue activo.

No llamarlo todavía field-only.

---

# Arquitectura experimental obligatoria

Implementar modos explícitos:

### MODE_A_STATIC
Gemma -> FieldEncoder -> estado -> decoder/probe.

### MODE_B_DYNAMIC
Gemma -> FieldEncoder -> D_phi rollout -> decoder/probe.

### MODE_C_DYNAMIC_RQM
Dynamic field + RQM.

### MODE_D_DYNAMIC_CDT
Dynamic field + CDT.

### MODE_E_FULL
Arquitectura completa.

### MODE_F_FIELD_ONLY
Una vez codificado el input:
FieldEncoder -> D_phi -> rollout -> decoder.

En FIELD_ONLY:
- RQM OFF
- CDT OFF
- attractor bank OFF
- direct memory OFF
- lookup tables OFF
- nearest-neighbor retrieval OFF
- no target vectors hard-coded

Cada JSON debe registrar:
mode, seed, dataset_hash, model_hash, field_checkpoint_hash, rqm_enabled, cdt_enabled, attractor_bank_enabled, direct_memory_enabled, lookup_enabled, nn_retrieval_enabled, train_targets, test_targets, target_seen_in_training, target_seen_in_memory, target_seen_in_rqm, target_seen_in_cdt, target_seen_in_attractor_bank, steps, accuracy, cosine_to_target, energy, stability, abstention y runtime.

Crear un leakage audit que marque LEAKED/INVALID y haga fallar la ejecución si aparece una condición prohibida.

---

# E18 — Rule Learning / Transformation Generalization

## Pregunta
¿El campo aprende una regla abstracta o memoriza trayectorias?

## Dataset
Generar vectores 2D/3D con identidad de instancia separada de identidad de regla.

Reglas:
- H1 translation: (x,y) -> (x+dx,y+dy)
- H2 rotation: (x,y) -> R(theta)(x,y)
- H3 reflection
- H4 scaling
- H5 affine: x' = Ax+b
- H6 rotation + translation
- H7 reflection + translation

## Entrenamiento
Para una regla, entrenar varios puntos y ocultar otros puntos de la misma regla. La prueba debe aplicar la regla a una instancia nueva, no recuperar una trayectoria.

## Test
1. regla vista / punto nuevo;
2. parámetro de regla no visto;
3. composición no vista;
4. extrapolación fuera del rango;
5. transformación inversa.

## Controles
table, nearest-neighbor, static field, D_phi, RQM direct, RQM compose, random dynamics y linear dynamics.

## Métricas
endpoint cosine, error euclídeo, relative transformation error, rule consistency, multi-step error, energy, stability y abstention.

## PASS
Dynamic field debe generalizar a puntos no vistos y superar static, nearest-neighbor y random dynamics sin RQM/CDT.

---

# E19 — RQM-free State Generation

## Pregunta
¿Puede D_phi producir un estado nunca almacenado usando únicamente dinámica?

## Protocolo
Entrenar A -> B y B -> C. Nunca entrenar A -> C.

En test:
A --D_phi--> B --D_phi--> C.

Probar 1, 2, 3, 4 y 8 pasos.

C no puede aparecer:
- como target de entrenamiento;
- en memoria;
- RQM;
- CDT;
- attractor bank;
- prototipo;
- selección de hiperparámetros.

Comparar contra static, random dynamics, table y nearest-neighbor.

El resultado fuerte sería generar C correctamente con todos los mecanismos externos desactivados.

---

# E20 — Static vs Dynamic controlado

Para cada seed:
1. inicializar un FieldEncoder común;
2. clonar pesos exactamente;
3. rama STATIC;
4. rama DYNAMIC;
5. mismo dataset;
6. mismo batch order;
7. mismo número de epochs;
8. presupuesto comparable.

Comparar seen, unseen, OOD, rule generalization, 1/2/4/8-step, energy y stability.

Reportar mean, median, std, paired delta dynamic-static y bootstrap 95% CI.

Pregunta directa:
¿La dinámica aporta capacidad que una geometría estática no puede producir?

---

# E21 — Separación FieldEncoder / Field Dynamics

Entrenar Gemma -> FieldEncoder -> D_phi. Congelar ambos.

Probar nuevo input lingüístico, mismo concepto, misma regla y nueva instancia.

Guardar:
z_input
z_after_1
z_after_2
z_target
decoder_output

Separar:
A. error de encoding;
B. error de dinámica;
C. error de decoding.

Esto evita atribuir a D_phi un fallo que pertenece al encoder.

---

# E22 — LLM swap / independencia de periferia

Entrenar con Gemma 2 y guardar únicamente FieldEncoder, D_phi y decoder/probe independiente.

Cambiar la periferia lingüística. Puede usarse otro modelo local o una transformación controlada del hidden state.

No permitir fine-tuning del campo durante test.

Comparar Gemma original -> field contra periferia alternativa -> field.

Si hace falta adapter por dimensionalidad, el adapter debe estar predefinido o evaluarse por separado; no permitir que el adapter vuelva a aprender la tarea.

---

# E23 — Long-horizon rollout

Entrenar solo 1-step. Evaluar:
1, 2, 4, 8, 16 y 32 pasos.

Medir error por paso, cosine, energy, norm, stability, diversidad de estados y distancia a attractores incorrectos.

Añadir perturbaciones epsilon:
0.01, 0.05, 0.10, 0.20.

Medir recuperación de la trayectoria correcta. Esto conecta dinámica con basin/attractor sin asumir que toda recuperación es memoria.

---

# E24 — Causal intervention sobre el campo

1. entrenar;
2. guardar checkpoint;
3. identificar subespacio/nodos activos sin usar test;
4. intervenir con zero, noise o phase perturbation;
5. ejecutar mismos inputs;
6. restaurar checkpoint.

Comparar baseline, intervention y rollback.

Resultado causal fuerte:
- intervención produce cambio específico y reproducible en la regla;
- rollback recupera comportamiento.

No seleccionar la intervención usando el conjunto de test.

---

# E25 — Continual Relational Learning real

Fase A: regla R1.
Fase B: regla R2.
Fase C: regla R3.

Después de cada fase evaluar todas las reglas.

Variantes:
A full;
B sin rehearsal;
C sin local bias;
D sin prototypes;
E sin todos los estabilizadores.

Métricas:
accuracy, forgetting R1/R2, forward transfer, backward transfer, energy drift y attractor drift.

---

# E26 — Scaling

Escalar conceptos:
8, 16, 32, 64, 128.

Relaciones:
2, 4, 8, 16, 32.

Registrar accuracy, OOD, rule generalization, parámetros, training time, inference time, memory y estadísticas del paisaje energético.

La pregunta es si la propiedad escala o desaparece.

---

# E27 — Compositional algebra

Entrenar transformaciones individuales:
T1 = translation
T2 = rotation

Probar T2(T1(x)) sin mostrar la composición directa.

Después:
T3 = reflection
T4 = scaling

Probar composiciones de longitud 3 y 4.

Comparar D_phi contra RQM compose, table y static field.

El target compuesto no puede aparecer como ejemplo directo.

---

# E28 — Energy-based dynamics

Hacer explícita una energía E(z,c) y dinámica z_{t+1}=D_phi(z_t,c).

Medir energía inicial, energía por paso, energía target, energía de distractores, basin depth y barrier height.

No imponer que la energía siempre disminuya: primero comprobar si la dinámica aprendida presenta descenso, barreras o trayectorias no monotónicas.

Objetivo: distinguir dinámica energética real de una red recurrente que simplemente aproxima una transformación.

---

# E29 — Counterfactual field

Entrenar una regla R y cambiar una sola variable latente.

Ejemplo:
translation(dx=2,dy=0)
vs
translation(dx=2,dy=1)

El campo debe producir trayectorias diferentes de manera sistemática.

Medir:
Delta input -> Delta trajectory.

Evaluar sensibilidad, composicionalidad y posible linealidad/no-linealidad.

---

# E30 — Persistencia real

1. entrenar;
2. guardar checkpoint;
3. terminar proceso;
4. reiniciar;
5. cargar solo FieldEncoder + D_phi;
6. repetir test;
7. comparar bitácora y métricas.

Después repetir con RQM/CDT/memoria auxiliar eliminados.

PASS: el comportamiento permanece después del restart y el checkpoint del campo es suficiente para reproducirlo.

---

# Controles obligatorios

Agregar a los experimentos principales:

1. Random field.
2. Static field.
3. Random dynamics.
4. Linear dynamics.
5. Nearest neighbor.
6. Table.
7. RQM only.
8. LLM raw.

No atribuir al campo una capacidad que también aparece en una baseline trivial.

---

# Protocolo estadístico

Mínimo:
- 8 seeds durante desarrollo;
- 16 seeds para resultado principal.

Para cada métrica:
- mean;
- median;
- std;
- min/max;
- bootstrap 95% CI;
- paired delta cuando corresponda.

Clasificación:
STRONG_POSITIVE
POSITIVE
PARTIAL
NULL
NEGATIVE
LEAKED
INVALID

LEAKED e INVALID se excluyen de promedios científicos.

---

# Reproducibilidad

Cada experimento debe producir:
results.json
results.csv
config.json
dataset.json
dataset_hash
model_hash
checkpoint_hash
seed
git_commit
rustc_version
hardware
runtime
repro_command

Un resultado debe poder repetirse sin depender del estado de un proceso anterior.

---

# Qué NO hacer

No aumentar porcentajes mediante complejidad que no demuestre una propiedad nueva.

Evitar:
- más RQM para arreglar E13;
- más tablas;
- prototypes que codifiquen respuestas;
- transformaciones hard-coded;
- target vectors para seleccionar parámetros;
- almacenar estados de test;
- attractor banks construidos con targets;
- usar test para elegir hiperparámetros;
- llamar aprendizaje dinámico a interpolación estática;
- llamar memoria a una tabla;
- llamar cognición a una clasificación.

---

# Orden recomendado

## Fase A — Fundamento
1. leakage audit universal;
2. modos STATIC/DYNAMIC/FIELD_ONLY;
3. controles;
4. reproducibilidad.

## Fase B — Cuello de botella
5. E18 Rule Learning;
6. E20 Static vs Dynamic;
7. E19 RQM-free.

## Fase C — Dinámica
8. E23 Long-horizon;
9. E27 Compositional algebra;
10. E28 Energy dynamics.

## Fase D — Independencia
11. E21 Encoder/Dynamics separation;
12. E22 LLM swap;
13. E30 persistence.

## Fase E — Plasticidad y escala
14. E25 continual relational learning;
15. E26 scaling;
16. E24 causal intervention;
17. E29 counterfactuals.

---

# Resultado que realmente cambiaría el estado del proyecto

La evidencia fuerte de esta etapa no es obtener otro 100%.

Sería demostrar conjuntamente:

1. FieldEncoder aprende representación estable.
2. D_phi aprende una regla y no una tabla.
3. La regla generaliza a instancias no vistas.
4. D_phi genera estados nunca almacenados.
5. RQM OFF.
6. CDT OFF.
7. attractor bank OFF.
8. direct memory OFF.
9. static field queda por debajo en el benchmark dinámico.
10. la propiedad sobrevive restart.
11. el checkpoint del campo reproduce la propiedad.
12. una intervención causal altera el comportamiento esperado.
13. la capacidad escala.
14. la capacidad no depende exclusivamente de Gemma.

Una conclusión defendible sería:

> Existe evidencia experimental de un sustrato externo entrenable que aprende representaciones y dinámicas relacionales persistentes independientemente de los pesos del LLM, incluyendo generación de estados no almacenados.

No saltar desde aquí directamente a AGI, vida o conciencia. Esas serían hipótesis posteriores.
