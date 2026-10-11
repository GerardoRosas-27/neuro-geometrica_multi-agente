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


---

# 29. Protocolo Clean-Room v2 — ejecución desde cero

Esta sección **prevalece sobre cualquier instrucción anterior del documento** para la ejecución E18–E30.

## 29.1 Separación absoluta respecto de E11–E17

E11–E17 quedan exclusivamente como resultados históricos. Para E18–E30 no se permite reutilizar:

- checkpoints de FieldEncoder;
- checkpoints de D_phi;
- prototipos;
- attractor banks;
- tablas;
- CDT/engrams;
- RQM entrenado;
- normalizaciones;
- thresholds;
- hiperparámetros elegidos por observar E11–E17;
- ejemplos holdout de E13/E15/E17;
- seeds seleccionadas por rendimiento.

La única excepción es el código de infraestructura que no contiene estado aprendido.

La periferia Gemma GGUF permanece congelada y puede reutilizarse como encoder lingüístico, pero el estado del campo debe inicializarse desde cero.

Estado inicial obligatorio:

    FieldEncoder = random initialization
    D_phi        = random initialization
    CDT          = empty
    RQM          = disabled
    attractor bank = empty
    NN memory    = disabled
    lookup table = disabled

---

## 29.2 Dataset completamente nuevo

Crear un generador independiente para Stage 2. No copiar los datasets de E11–E17.

Separar:

    TRAIN
    DEV
    TEST

El TEST se genera y se sella antes del entrenamiento.

Cada manifest debe registrar:

    dataset_version
    generator_version
    generator_commit
    seed
    sha256
    rule_family
    dimension
    parameter_range

Las seeds de desarrollo y confirmación deben ser diferentes.

Propuesta:

    development: 0xA300–0xA307
    confirmation: 0xB300–0xB30F

El contenido completo del TEST no puede entrar al pipeline de entrenamiento.

---

## 29.3 Anti-contaminación matemática

No basta con comprobar que el vector exacto no apareció.

Antes de entrenar, el generador debe verificar:

    exact_duplicate
    near_duplicate
    equivalent_state
    same_orbit
    equivalent_transformation
    equivalent_target

Para reglas con simetrías, las equivalencias matemáticas también cuentan como contaminación.

Si el auditor encuentra una coincidencia:

    DATASET_INVALID

El experimento no se ejecuta.

---

## 29.4 Lock de hiperparámetros

El flujo obligatorio es:

    TRAIN → DEV → LOCK → TEST

DEV puede utilizarse para seleccionar:

- learning rate;
- epochs;
- arquitectura;
- regularización;
- thresholds;
- tamaño del campo;
- λ;
- criterio de parada.

Después de mirar TEST no se permite modificar ninguno.

Si se modifica una decisión después de observar TEST:

    TEST_INVALIDATED

y se debe generar un nuevo TEST sellado.

---

# 30. E18 reforzado — aprender una regla, no memorizar pares

Usar transformaciones continuas conocidas:

    y = x + b
    y = R(theta)x
    y = Mx
    y = s*x
    y = A*x+b
    y = T2(T1(x))

Variar dimensiones, estados iniciales, orientación, escala y parámetros.

El test debe contener estados que no aparezcan en TRAIN y cuyos targets tampoco aparezcan.

Medir además de accuracy:

- MSE;
- cosine;
- error relativo;
- energía;
- estabilidad;
- distancia al manifold;
- error por dimensión.

La condición científica no es simplemente acertar B. Es que D_phi aprenda una función que transforme X_new en B_new.

---

# 31. E18C reforzado — demostrar que CDT aporta aprendizaje

Comparar las mismas experiencias bajo:

    C0 = sin experiencia
    C1 = experiencia sin consolidar
    C2 = experiencia consolidada en CDT
    C3 = CDT parcialmente corrupto
    C4 = CDT irrelevante
    C5 = CDT de otra regla

Calcular:

    experience_gain = accuracy_C2 - accuracy_C0

pero también:

    sample_efficiency_gain
    novel_generation_gain
    stability_gain

La prueba no es válida si C2 mejora únicamente porque puede recuperar un ejemplo cercano al target.

Añadir una condición:

    C6 = CDT con estadísticas agregadas solamente

Si C6 conserva parte importante de la mejora sin guardar episodios concretos, existe evidencia adicional de que CDT funciona como experiencia/regularidad y no como respuesta.

---

# 32. E19 reforzado — auditoría de procedencia

Cada predicción debe producir un registro:

    target_seen_training
    target_seen_dev
    target_seen_cdt
    target_seen_rqm
    target_seen_attractor
    target_seen_table
    target_seen_nn
    target_equivalent_seen

y contadores:

    cdt_queries
    rqm_queries
    table_queries
    nn_queries
    attractor_queries

Además:

    target_in_normalization
    target_in_decoder
    target_in_threshold_selection
    target_in_hyperparameter_selection

Para FIELD_ONLY:

    CDT queries       = 0
    RQM queries       = 0
    table queries     = 0
    NN queries        = 0
    attractor queries = 0
    leakage_score     = 0

Para DYNAMIC+CDT, CDT puede participar durante el aprendizaje histórico, pero no puede devolver el target durante evaluación.

---

# 33. E20 reforzado — regla contra trayectoria

Usar tres conjuntos:

### Trayectoria

    A → B → C → D

### Regla

    A1 → B1
    A2 → B2
    A3 → B3
    A4 → B4
    ...

### Anti-memorization

Cambiar simultáneamente:

- escala;
- orientación;
- magnitud;
- distribución;
- orden;
- estados iniciales.

Si el campo aprendió una regla, el tercer conjunto no debe destruir el comportamiento.

---

# 34. E21 reforzado — extrapolación

No limitarse a valores vistos.

Ejemplos:

    TRAIN dx = -2,-1,0,1,2
    TEST  dx = 3

    TRAIN theta = -30,-15,0,15,30
    TEST  theta = 45

    TRAIN scale = 0.5,0.75,1,1.25,1.5
    TEST  scale = 2

Separar claramente:

    interpolation
    extrapolation

porque son evidencias distintas.

---

# 35. E22 reforzado — composición sin RQM

Entrenar:

    T1(x)
    T2(x)

Nunca entrenar:

    T2(T1(x))

Evaluar:

    D_phi(T2(T1(x)))

La comparación debe incluir:

- D_phi;
- STATIC;
- linear dynamics;
- RQM compose;
- table;
- nearest-neighbor.

La evidencia de autonomía solo puede venir de la variante:

    D_phi + RQM OFF

---

# 36. E23 reforzado — rollout sin teacher forcing

Entrenar únicamente un paso.

Evaluar:

    1, 2, 4, 8, 16, 32, 64

Durante el rollout está prohibido insertar el estado verdadero intermedio.

Debe utilizarse:

    x1 = D_phi(x0)
    x2 = D_phi(x1)
    x3 = D_phi(x2)

y así sucesivamente.

Medir:

- error acumulado;
- cosine;
- energía;
- norm;
- estabilidad;
- recuperación ante perturbación.

Esto elimina una fuente importante de falsos positivos en dinámica.

---

# 37. E24 reforzado — STATIC vs DYNAMIC pareado

Usar la misma:

- seed;
- inicialización;
- dataset;
- batch order;
- normalización;
- presupuesto;
- número de pasos.

Comparar:

    STATIC:
    z = Encoder(x)

    DYNAMIC:
    z' = D_phi(z,c)

Reportar:

    delta = dynamic - static

con mean, median, std, bootstrap 95% CI y effect size.

La pregunta es qué capacidad aparece específicamente al introducir D_phi, no cuál obtiene mayor accuracy absoluta.

---

# 38. E25 — experimento central

El ciclo completo debe ser:

    experiencias E1..E4
            ↓
          CDT
            ↓
    regularidad consolidada
            ↓
      adaptación de D_phi
            ↓
        N_new
            ↓
         D_phi
            ↓
         Y_new

La respuesta Y_new debe cumplir:

    Y_new ∉ TRAIN
    Y_new ∉ DEV
    Y_new ∉ CDT
    Y_new ∉ RQM
    Y_new ∉ attractor bank
    Y_new ∉ nearest-neighbor memory
    Y_new ∉ table

La hipótesis principal es:

    Dynamic + consolidated experience
             >
    Dynamic without experience

en generalización, sample efficiency y estabilidad.

---

# 39. E26 — ablation fuerte de experiencia

Comparar:

    A = sin experiencia
    B = experiencia reciente sin consolidar
    C = experiencia consolidada
    D = CDT corrupto
    E = CDT irrelevante
    F = CDT de regla incorrecta
    G = CDT con estadísticas agregadas

La condición G es un control especialmente importante para separar:

    experiencia/regularidad

de:

    memoria episódica/lookup

---

# 40. E27 — transferencia de regla

Aprender en familia A.

Consolidar.

Probar familia B con estados nunca vistos.

Crear niveles:

    B1 = mismos espacios, nuevos valores
    B2 = nueva distribución
    B3 = nueva escala/orientación

Los targets de B nunca pueden aparecer durante consolidación.

---

# 41. E28 — continual learning

Ejecutar:

    R1 → consolidate
    R2 → consolidate
    R3 → consolidate
    R4 → consolidate

Después probar todas.

Comparar:

1. sin CDT;
2. CDT activo;
3. CDT corrupto;
4. replay;
5. CDT + replay mínimo.

Medir:

- forgetting;
- forward transfer;
- backward transfer;
- sample efficiency;
- novel-state generation.

No atribuir automáticamente una mejora a CDT si replay o prototypes explican el resultado.

---

# 42. E29 — intervención causal mejorada

Guardar checkpoint antes de intervenir.

Comparar:

    baseline
    targeted intervention
    random intervention
    rollback

La intervención específica debe producir un cambio reproducible mayor que una perturbación aleatoria de igual magnitud.

El rollback debe recuperar el comportamiento original.

Esto permite distinguir correlación de dependencia causal.

---

# 43. E30 — persistencia después de reinicio

Después de consolidar:

1. guardar FieldEncoder;
2. guardar D_phi;
3. guardar CDT;
4. terminar proceso;
5. borrar RAM episódica;
6. reiniciar;
7. cargar checkpoints;
8. probar estado nuevo.

Ablaciones:

    P0 = FieldEncoder + D_phi + CDT
    P1 = FieldEncoder + D_phi
    P2 = D_phi
    P3 = CDT
    P4 = CDT retrieval

Esto identifica dónde persiste realmente la información.

---

# 44. El decoder no puede salvar el experimento

Para los benchmarks matemáticos E18–E30 usar un decoder determinista independiente.

Gemma debe actuar como periferia:

    lenguaje → representación de campo

No:

    lenguaje → campo → Gemma inventa la respuesta

La versión lingüística se ejecuta como benchmark separado después de validar el benchmark matemático.

---

# 45. Escalamiento

Cada prueba tendrá niveles:

    Level 1 = 2D
    Level 2 = 4D
    Level 3 = 8D
    Level 4 = 16D
    Level 5 = composición
    Level 6 = long-horizon
    Level 7 = extrapolación + perturbación

No declarar robustez a partir de Level 1.

---

# 46. Controles obligatorios finales

Ejecutar:

1. Gemma raw;
2. random FieldEncoder;
3. random D_phi;
4. STATIC;
5. linear dynamics;
6. polynomial baseline;
7. MLP baseline;
8. table;
9. nearest-neighbor;
10. RQM only;
11. CDT retrieval;
12. DYNAMIC sin CDT;
13. DYNAMIC + CDT.

La pregunta es si la dinámica aprendida aporta algo que un baseline más simple no explica.

---

# 47. Protocolo estadístico

Desarrollo:

    8 seeds

Confirmación:

    16 seeds nuevas

Las seeds de confirmación no participan en selección de arquitectura.

Reportar:

- mean;
- median;
- std;
- min/max;
- bootstrap 95% CI;
- paired delta;
- effect size;
- learning curves;
- resultado por regla;
- resultado por dificultad.

Estados:

    STRONG_POSITIVE
    POSITIVE
    PARTIAL
    NULL
    NEGATIVE
    LEAKED
    INVALID

LEAKED e INVALID no entran en promedios.

---

# 48. Preregistro

Antes de la corrida principal crear:

    docs/stage2_preregistered_protocol.md

Debe fijar:

- hipótesis;
- datasets;
- seeds;
- arquitectura;
- hiperparámetros;
- métricas;
- controles;
- criterios PASS/FAIL;
- exclusiones.

Registrar hashes de:

    protocol
    dataset
    generator
    model
    code commit

No cambiar el protocolo después de observar TEST.

---

# 49. Artefactos reproducibles

Cada experimento debe generar como mínimo:

    results/stage2/E18/
      train_manifest.json
      dev_manifest.json
      test_manifest.json
      metrics.csv
      metrics.json
      provenance.json
      audit.json
      config.json
      summary.md

Y cada resultado debe incluir:

    dataset_hash
    protocol_hash
    code_commit
    model_hash
    seed

---

# 50. Orden definitivo

## Fase 0 — Clean room

1. nuevo generador;
2. nuevos train/dev/test;
3. hashes;
4. anti-duplicados;
5. provenance;
6. checkpoints vacíos;
7. comprobar ausencia de artefactos E11–E17.

## Fase 1 — Dinámica

8. E18A;
9. E20;
10. E21;
11. E24.

## Fase 2 — Composición

12. E22;
13. E23.

## Fase 3 — Experiencia consolidada

14. E18B;
15. E18C;
16. E25;
17. E26;
18. E27.

## Fase 4 — Persistencia

19. E28;
20. E29;
21. E30.

## Fase 5 — Confirmación

22. 16 seeds nuevas;
23. bootstrap CI;
24. paired analysis;
25. auditoría final;
26. informe reproducible.

---

# 51. Criterio de éxito más fuerte

El resultado decisivo sería:

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

con:

    RQM = OFF
    lookup = OFF
    NN = OFF
    attractor bank = OFF
    direct memory = OFF
    leakage = 0

y además:

    Dynamic + consolidated experience
                 >
    Dynamic without experience
                 >
    Static field
                 >
    random controls

en generalización de reglas, no simplemente en recuperación.

La interpretación defendible sería:

> El sustrato externo puede adquirir regularidades a partir de experiencias consolidadas y utilizar una dinámica aprendida para transformar estados nuevos que nunca fueron almacenados como respuestas.

Esto no demostraría por sí solo AGI, conciencia o vida. Sí sería evidencia mucho más fuerte de aprendizaje persistente de una dinámica fuera de los pesos del LLM.

---

# 52. Principio final

La prioridad no es conseguir PASS.

La prioridad es construir una prueba donde, si aparece PASS, sea difícil explicarlo por:

- contaminación;
- lookup;
- nearest-neighbor;
- tabla;
- RQM;
- attractor bank;
- decoder lingüístico;
- estado temporal;
- selección manual de hiperparámetros.

**La etapa debe empezar desde cero y permitir que los datos, no los resultados E11–E17, determinen si el campo realmente aprendió una regla.**
