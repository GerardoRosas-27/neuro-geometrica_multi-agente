# Etapa 2 — Campo autónomo con aprendizaje por experiencia consolidada

## 1. Pregunta central de toda la etapa

Los experimentos E11–E17 no deben tratarse como pruebas independientes. En conjunto forman una hipótesis arquitectónica:

Gemma / periferia lingüística → FieldEncoder → campo dinámico D_phi → estados y transiciones

y, después de múltiples experiencias:

experiencias → consolidación CDT → conocimiento/regularidades consolidadas → nueva experiencia → adaptación del campo

La pregunta que debe responder esta etapa es:

> ¿Puede el campo aprender una regla a partir de experiencias anteriores, consolidarlas en CDT y posteriormente aplicar esa regla a un estado completamente nuevo que nunca vio, sin que una tabla, RQM, CDT, attractor bank o nearest-neighbor le entregue directamente la respuesta?

Esta formulación es más fuerte que preguntar simplemente si el campo puede recuperar un estado.

El objetivo es distinguir:
1. memoria: recordar una experiencia concreta;
2. representación: codificar un estado en una geometría útil;
3. regla: aprender una transformación general;
4. dinámica: aplicar esa transformación a un estado nuevo;
5. consolidación: extraer regularidades de experiencias anteriores y hacerlas disponibles para aprendizaje posterior.

La CDT puede participar como memoria de experiencias consolidadas, pero no debe funcionar como una tabla de respuestas.

---

# 2. Interpretación conjunta de E11–E17

## E11 — Representación entrenable

El FieldEncoder modifica la geometría y muestra delta causal positivo en varias semillas.

Esto apoya:

El campo puede aprender una representación.

Todavía no demuestra que pueda aprender reglas.

## E12 — Invariancia lingüística

Con Gemma GGUF real, entrenamiento en español y holdouts lingüísticos, los 8 seeds alcanzan seen/unseen/OOD = 1.0.

Esto apoya:

El campo puede aprender una geometría relativamente independiente de la superficie lingüística en el benchmark.

No demuestra todavía independencia general del LLM.

## E13 — Relaciones

Los controles de tabla/RQM superan claramente a static/dynamic.

Esto revela el cuello de botella:

El campo sabe representar conceptos, pero todavía no demuestra que pueda descubrir y reutilizar reglas relacionales.

## E14 — Consolidación/retención

La retención es positiva, pero fixed prototypes, micro-rehearsal y local biases pueden estar aportando parte del resultado.

Por eso hay que separar memoria consolidada de mecanismos auxiliares de estabilidad.

## E15 — Composición

El resultado PARTIAL/FAIL es importante:

Una geometría útil no genera automáticamente composicionalidad.

## E16 — Dinámica

Varias semillas muestran rollouts correctos, pero todavía hay semillas débiles y casos donde static es comparable.

Esto sugiere que existe una señal de dinámica aprendida, pero no está suficientemente aislada.

## E17 — Estados no almacenados

El target no aparece directamente en train/CDT/attractor bank/direct RQM, pero RQM composition todavía participa.

Por tanto:

E17 es una señal de generación de estado no almacenado, pero todavía no es una demostración limpia de autonomía del campo.

---

# 3. Cambio conceptual de esta etapa

No hacer únicamente:

entrenar campo → probar campo

Hacer un ciclo de aprendizaje:

EXPERIENCIA 1
    ↓
FieldEncoder
    ↓
campo dinámico
    ↓
resultado
    ↓
experiencia registrada
    ↓
CDT
    ↓
CONSOLIDACIÓN
    ↓
regularidades reutilizables
    ↓
nuevo episodio
    ↓
FieldEncoder
    ↓
D_phi
    ↓
estado completamente nuevo

La hipótesis es que CDT no tiene que almacenar la respuesta futura.

Debe almacenar experiencias/regularidades consolidadas que modifican el aprendizaje o el contexto del campo.

La distinción crítica es:

### Memoria prohibida como respuesta

nuevo_estado X
      ↓
CDT
      ↓
respuesta Y

Esto sería recuperación.

### Memoria permitida como experiencia

experiencias anteriores
      ↓
CDT
      ↓
señal/estructura consolidada
      ↓
D_phi aprende la regla
      ↓
nuevo estado X
      ↓
D_phi(X)
      ↓
Y

Aquí Y nunca debe existir en CDT antes del test.

---

# 4. Arquitectura experimental

Implementar explícitamente cinco modos.

## MODE_0_RAW

Gemma → output

Baseline lingüístico.

## MODE_1_STATIC_FIELD

Gemma → FieldEncoder → z

Sin dinámica.

## MODE_2_DYNAMIC_FIELD

Gemma → FieldEncoder → D_phi(z,c)

Sin CDT/RQM.

## MODE_3_DYNAMIC_PLUS_CDT_TRAINING

Experiencias
    ↓
CDT consolidation
    ↓
señal de experiencia consolidada
    ↓
entrenamiento/adaptación de D_phi
    ↓
nuevo estado
    ↓
D_phi

CDT participa en el aprendizaje histórico, pero no puede devolver el target durante la evaluación.

## MODE_4_FULL

Arquitectura completa, incluyendo RQM/CDT cuando corresponda.

El resultado principal debe provenir de MODE_2 y MODE_3, no solamente de MODE_4.

---

# 5. Definición de CDT como experiencia y no como respuesta

El agente debe implementar una interfaz conceptual equivalente a:

ConsolidatedExperience {
    context_signature
    state_before
    action_or_relation
    state_after
    confidence
    energy
    repetition_count
}

Pero durante un test de generalización:

- state_after del target no puede haber sido observado para el nuevo estado;
- no se puede hacer lookup de un estado equivalente;
- no se puede buscar el vecino más cercano;
- no se puede recorrer RQM hasta encontrar el target;
- no se puede usar CDT para devolver directamente el target.

CDT puede aportar:

- regularidades;
- estadísticas;
- parámetros de contexto;
- patrones de transición;
- experiencia agregada;
- señal de confianza;
- estructura consolidada.

El experimento debe registrar exactamente qué información sale de CDT.

---

# 6. E18 — Aprendizaje de regla a partir de experiencias

## Pregunta

¿Puede el campo descubrir una regla común a partir de múltiples experiencias y posteriormente aplicarla a una instancia que nunca vio?

## Dataset

Usar transformaciones matemáticas porque permiten conocer la respuesta correcta sin almacenarla.

### Regla R1 — Translation

(x,y) → (x+dx,y+dy)

### Regla R2 — Rotation

x → R(theta)x

### Regla R3 — Reflection

### Regla R4 — Scaling

### Regla R5 — Affine

x' = Ax+b

### Regla R6 — composición

R2(R1(x))

---

# 7. E18A — Aprendizaje directo sin CDT

Entrenar únicamente:

FieldEncoder + D_phi

Experiencias:

A1 → B1
A2 → B2
A3 → B3
A4 → B4

Test:

A5 → ?

A5 jamás apareció durante entrenamiento.

Objetivo:

D_phi(A5) ≈ B5

Este es el baseline fundamental.

---

# 8. E18B — Aprendizaje mediante consolidación CDT

Ahora separar entrenamiento por episodios.

### Episodio 1

A1 → B1

### Episodio 2

A2 → B2

### Episodio 3

A3 → B3

Cada experiencia pasa por consolidación.

Después de varias experiencias:

CDT
 ↓
regularidad consolidada
 ↓
actualización/adaptación del campo

Finalmente:

A_new → ?

La instancia A_new nunca fue vista.

## PASS

El campo produce B_new correcto sin que B_new exista en CDT.

---

# 9. E18C — Prueba de que CDT aporta aprendizaje y no recuperación

Comparar cuatro condiciones:

### C0 — No experiencia previa

Campo no recibe experiencias consolidadas.

### C1 — Experiencias sin consolidar

Se presentan experiencias durante entrenamiento pero no pasan por CDT.

### C2 — Experiencias consolidadas

Las mismas experiencias pasan por CDT.

### C3 — CDT corrupto

Se introduce ruido controlado en la memoria consolidada.

Comparar:

accuracy
rule_generalization
energy
stability
training_steps

Si C2 mejora respecto a C0/C1 y sigue generando correctamente estados nunca almacenados, existe evidencia de que la consolidación aporta aprendizaje.

---

# 10. E19 — Prueba definitiva de no-lookup

Construir una auditoría automática.

Para cada target de test:

target_seen_training = false
target_seen_CDT = false
target_seen_RQM = false
target_seen_attractor = false
target_seen_table = false
target_seen_NN = false

Además:

target_equivalent_seen = false

porque no basta con esconder el vector exacto si existe una copia equivalente.

Durante inferencia:

CDT query count
RQM query count
table query count
NN query count
attractor query count

debe ser cero en el experimento field-only.

---

# 11. E20 — Regla frente a trayectoria

Este experimento es fundamental.

Construir dos datasets.

## Dataset A — Trayectoria

A → B → C → D

## Dataset B — Regla

Muchos pares independientes:

A1 → B1
A2 → B2
A3 → B3
...

Después preguntar por:

A_new → ?

Si solamente aprende trayectoria, fallará.

Si aprende la regla, debería generalizar.

---

# 12. E21 — Regla abstracta + parámetro nuevo

Entrenar:

translation(dx=2,dy=1)
translation(dx=2,dy=1)
translation(dx=2,dy=1)

con diferentes estados iniciales.

Test:

translation(dx=2,dy=1)

sobre un estado completamente nuevo.

Después aumentar dificultad:

Entrenar parámetros:

dx ∈ {-2,-1,0,1,2}

Test:

dx = 3

Esto prueba extrapolación del parámetro de la regla.

---

# 13. E22 — Composición no observada

Entrenar por separado:

T1(x)
T2(x)

Nunca entrenar:

T2(T1(x))

Test:

D_phi(T2(T1(x)))

La composición debe generarse.

Comparar:

- D_phi;
- RQM compose;
- table;
- nearest-neighbor;
- static field.

La variante D_phi debe funcionar sin RQM.

---

# 14. E23 — Long-horizon

Entrenar solamente transiciones de un paso.

Probar:

1
2
4
8
16
32

pasos.

Medir:

- error;
- cosine;
- energía;
- estabilidad;
- norm;
- desviación acumulada.

Después añadir perturbación:

epsilon = 0.01
0.05
0.10
0.20

Pregunta:

¿La dinámica mantiene la regla o colapsa después de varios pasos?

---

# 15. E24 — Static vs Dynamic

Usar exactamente la misma inicialización del FieldEncoder.

STATIC:
z = FieldEncoder(x)

DYNAMIC:
z' = D_phi(z,c)

Misma seed, dataset, batch order y presupuesto.

Comparar generalización a estados nuevos.

El objetivo no es que dynamic tenga mejor accuracy en todo. El objetivo es identificar qué comportamiento solamente aparece cuando existe dinámica.

---

# 16. E25 — CDT como memoria de experiencia, no como memoria de respuesta

Este es el experimento que más conecta con la arquitectura propuesta.

### Fase 1

Campo aprende experiencias:

E1
E2
E3
E4

### Consolidación

E1..E4 → CDT

### Fase 2

Aparece una nueva familia de estados:

N1
N2
N3

No aparecen sus targets.

CDT puede aportar las regularidades aprendidas de E1..E4.

El campo debe inferir:

N1 → ?
N2 → ?
N3 → ?

## Hipótesis

La experiencia consolidada permite que D_phi aprenda más rápido o generalice mejor, pero la respuesta final es generada por la dinámica del campo.

Esto es mucho más interesante que simplemente demostrar que CDT puede recuperar información.

---

# 17. E26 — Ablación de experiencia

Comparar:

A — sin experiencia previa
B — experiencia reciente sin consolidar
C — experiencia consolidada en CDT
D — CDT parcialmente corrupto
E — CDT con experiencias irrelevantes

Medir:

- velocidad de aprendizaje;
- muestras necesarias;
- generalización;
- error;
- estabilidad.

Una propiedad especialmente interesante sería:

C > B > A

en generalización, sin que C tenga acceso directo a los targets de test.

---

# 18. E27 — Transferencia de regla

Aprender una regla en una familia:

familia A

Consolidarla.

Después presentar:

familia B

con estados que nunca aparecieron.

La pregunta:

¿La CDT permite que el campo transfiera una regularidad abstracta a una nueva instancia?

Esto empieza a separar memoria episódica de conocimiento estructural.

---

# 19. E28 — Catastrophic forgetting con conocimiento consolidado

Aprender:

R1

Consolidar.

Aprender:

R2

Consolidar.

Aprender:

R3

Después probar:

R1
R2
R3

Variantes:

1. sin CDT;
2. CDT activo;
3. CDT corrupto;
4. CDT con experiencias irrelevantes.

Medir forgetting y transferencia.

---

# 20. E29 — Intervención causal

Guardar checkpoint del campo.

Intervenir solamente sobre componentes asociados a la dinámica aprendida.

Ejecutar:

baseline
intervention
rollback

Una intervención válida debe cambiar el comportamiento de forma reproducible.

El rollback debe restaurarlo.

---

# 21. E30 — Persistencia después de apagar todo

Después de consolidar:

1. terminar proceso;
2. reiniciar;
3. cargar checkpoint;
4. no restaurar estado temporal;
5. ejecutar nueva experiencia;
6. probar nuevo estado.

Separar:

checkpoint FieldEncoder
checkpoint D_phi
CDT
RAM episódica
RQM

Repetir progresivamente quitando cada componente.

La pregunta final es:

¿Qué conocimiento permanece realmente en el campo?

---

# 22. Controles obligatorios

Cada resultado principal debe compararse contra:

1. Gemma raw;
2. random FieldEncoder;
3. static field;
4. dynamic field aleatorio;
5. linear dynamics;
6. table;
7. nearest-neighbor;
8. RQM only;
9. CDT retrieval only;
10. dynamic + CDT;
11. dynamic sin CDT.

Especialmente importante:

CDT retrieval only ≠ dynamic field.

Si CDT retrieval resuelve el benchmark, eso demuestra memoria, no generación dinámica.

---

# 23. Métricas nuevas

Además de accuracy:

### Rule generalization

Porcentaje de estados nuevos correctamente transformados.

### Novel-state generation

Porcentaje de targets nunca almacenados que son generados correctamente.

### Experience gain

accuracy_with_CDT - accuracy_without_CDT

### Sample efficiency

Número de experiencias necesarias para alcanzar un umbral.

### Rule retention

Capacidad de seguir aplicando la regla después de aprender otras reglas.

### Transfer

Rendimiento sobre una nueva familia de estados.

### Rollout stability

Error después de 1/2/4/8/16/32 pasos.

### Leakage score

Debe ser exactamente 0 para el experimento field-only.

---

# 24. Auditoría de información

No basta con registrar flags.

Implementar una auditoría de procedencia:

target_origin

y registrar si el target pudo influir en:

- entrenamiento;
- selección de hiperparámetros;
- construcción de CDT;
- construcción de RQM;
- construcción de attractors;
- decoder;
- normalización;
- thresholds.

También registrar:

information_path

para cada predicción:

input
→ encoder
→ field
→ CDT context
→ D_phi
→ output

Si aparece:

input
→ CDT
→ target

el experimento se marca LEAKED.

---

# 25. Protocolo estadístico

Desarrollo:
8 seeds.

Resultados principales:
16 seeds.

Para cada comparación:

- mean;
- median;
- std;
- min/max;
- bootstrap 95% CI;
- paired delta;
- effect size;
- curva de aprendizaje.

No utilizar solamente PASS/FAIL.

Estados:

STRONG_POSITIVE
POSITIVE
PARTIAL
NULL
NEGATIVE
LEAKED
INVALID

---

# 26. Qué NO hacer

No convertir CDT en una tabla disfrazada.

No almacenar el target de test.

No usar RQM para resolver el target y después atribuirlo a D_phi.

No construir attractor banks con estados de test.

No usar nearest-neighbor para elegir el resultado.

No ajustar hiperparámetros mirando el conjunto de test.

No introducir el target como prototype.

No aumentar RQM para resolver E13.

No interpretar 100% de accuracy como prueba de regla.

La propiedad buscada es:

misma regla + estado nuevo → respuesta nueva generada por la dinámica.

---

# 27. Orden de ejecución

## Fase A — Aislamiento

1. leakage/provenance audit;
2. FIELD_ONLY;
3. static vs dynamic;
4. RQM/CDT OFF.

## Fase B — Regla

5. E18A directo;
6. E18B con consolidación;
7. E18C CDT vs no-CDT;
8. E20 trayectoria vs regla;
9. E21 parámetro nuevo.

## Fase C — Composición

10. E22 composición no observada;
11. E23 long-horizon.

## Fase D — Memoria como experiencia

12. E25 CDT como experiencia;
13. E26 ablation;
14. E27 transferencia.

## Fase E — Persistencia

15. E28 continual learning;
16. E29 causal intervention;
17. E30 restart/persistence.

---

# 28. Resultado decisivo

El resultado más fuerte de toda esta etapa sería obtener:

Experiencias anteriores
        ↓
      CDT
        ↓
regularidad consolidada
        ↓
     D_phi
        ↓
estado nuevo nunca visto
        ↓
respuesta nueva nunca almacenada

mientras simultáneamente:

RQM = OFF
lookup = OFF
nearest_neighbor = OFF
attractor_bank = OFF
direct_memory = OFF
target leakage = 0

Y demostrar que:

Dynamic + consolidated experience
        >
Dynamic without experience
        >
Static field
        >
random controls

en generalización de reglas, no solamente en recuperación.

Si esto ocurre de forma reproducible, la interpretación defendible sería:

> El sustrato externo puede adquirir regularidades a partir de experiencias consolidadas y utilizar esas regularidades para transformar estados nuevos que nunca fueron almacenados como respuestas.

Eso sería evidencia mucho más fuerte de aprendizaje externo al LLM.

No demostraría por sí solo AGI, conciencia o vida. Demostraría algo más concreto y experimentalmente importante: aprendizaje persistente de una dinámica/regla fuera de los pesos del LLM, con memoria consolidada actuando como experiencia y no como tabla de respuestas.
