# Plan de cierre experimental — autonomía del campo v3

Rama: `exp/field-autonomy-next`
Base: `main`
Fecha: 2026-09-23

> **Cierre del ciclo v3.7 + confirmación:** ver
> [`cierre_exp_field_autonomy_next_v3.md`](cierre_exp_field_autonomy_next_v3.md)
> (cierre docs `44c1910`; tip evidencia `33a35db`). Este plan sigue siendo la especificación;
> el cierre resume qué quedó cerrado/abierto **sin merge a main**.

## 1. Objetivo

Esta rama parte deliberadamente de `main`, no de las ramas experimentales anteriores. Su objetivo es construir una segunda generación de experimentos que determine qué capacidad cognitiva permanece en el sustrato externo cuando se eliminan progresivamente los mecanismos que pueden convertirlo en un sistema de recuperación de respuestas.

La hipótesis que se quiere poner a prueba es:

> Experiencias previas pueden consolidarse en CDT como regularidades/experiencia y modificar el aprendizaje de una dinámica del campo Dφ; posteriormente Dφ puede transformar estados nuevos y producir estados de salida que nunca fueron almacenados como respuestas.

No se considera evidencia suficiente que el sistema simplemente encuentre un vecino, consulte una tabla, recupere una respuesta de CDT/RQM o reproduzca una trayectoria.

## 2. Qué aprendimos de E11–E17

### Señales positivas

- E11: el FieldEncoder entrenable modifica causalmente la geometría.
- E12: con Gemma 2 GGUF congelado, entrenamiento únicamente en español y holdouts cross-lingüísticos, 8/8 semillas obtuvieron 1.0 en seen/unseen/OOD. El espacio raw de Gemma no daba la separación deseada; el campo la aprendió.
- E14: los atractores pueden conservarse bajo aprendizaje incremental en el protocolo probado.
- E15 endurecido: composición factorial + Dφ produjo en smoke 3/3 de estructura y ~0.80 de ranking, con mayor energía y distancia de manifold para combinaciones incompatibles.
- E16: Dφ puede aprender transiciones, aunque en algunas semillas el resultado no supera claramente al campo estático.
- E17: aparecen estados nunca almacenados directamente, pero la auditoría todavía detecta composición RQM.

### Problemas que deben cerrarse

1. E13 todavía tiene generalización relacional débil: el campo memoriza/representa bien lo visto, pero la transferencia a entidades nuevas es insuficiente.
2. E15 necesita revalidación multi-seed y OOD más difícil.
3. E16 necesita comparación estricta y emparejada contra estático, NN, RQM y modelos dinámicos simples.
4. E17 necesita RQM=OFF, CDT=OFF y todos los mecanismos de lookup=OFF.
5. Falta demostrar la cadena central: experiencia → consolidación CDT → regularidad → aprendizaje/adaptación de Dφ → respuesta nueva.
6. Los benchmarks actuales son pequeños; aumentar cantidad, diversidad y dificultad es necesario.
7. Hace falta separar claramente aprendizaje de una regla, memorización de una trayectoria y recuperación de una respuesta.

## 3. Principio de entrenamiento desde cero

Esta rama debe tener un protocolo clean-room independiente de E11–E17.

### Prohibiciones

- No reutilizar pesos del FieldEncoder entrenado en E11–E17.
- No reutilizar Dφ entrenado previamente.
- No usar prototypes construidos a partir del TEST.
- No ajustar hiperparámetros mirando TEST.
- No construir CDT con ejemplos de TEST.
- No construir RQM con ejemplos de TEST.
- No almacenar targets de TEST en tablas, NN, attractor banks, caches o fixtures.
- No seleccionar seeds, thresholds o arquitecturas después de observar resultados TEST.
- No usar el target de evaluación para seleccionar decoder, normalización o temperatura.

### Tres particiones

Cada benchmark debe generar:

- TRAIN: aprendizaje.
- DEV: selección de hiperparámetros y early stopping.
- TEST: sellado antes del entrenamiento.

Además se generarán familias estructuralmente nuevas para TEST.

El generador debe producir un manifiesto con hash de cada ejemplo y de cada conjunto.

## 4. Dataset mucho más amplio y variado

El entrenamiento anterior es demasiado pequeño para decidir si Dφ aprendió una regla general. La siguiente etapa debe aumentar la variedad, no simplemente repetir más ejemplos equivalentes.

### 4.1 Transformaciones geométricas

Generar familias:

- traslación 2D y 3D;
- rotación;
- reflexión;
- escala;
- shear;
- transformación afín;
- composición de transformaciones;
- inversión;
- transformaciones no lineales suaves;
- ruido y perturbaciones.

Ejemplo:

TRAIN:
- múltiples puntos y figuras bajo la misma transformación.

TEST:
- puntos y figuras nunca vistos bajo la misma regla.

Después:

TRAIN:
- reglas con parámetros variados.

TEST:
- parámetros nuevos.

### 4.2 Variación de contenido

No utilizar solamente símbolos A/B/C.

Cada regla debe probarse sobre:

- posiciones nuevas;
- magnitudes nuevas;
- familias nuevas;
- combinaciones nuevas;
- permutaciones;
- ruido;
- estados parcialmente corruptos;
- estados fuera de la distribución de entrenamiento.

Esto permite distinguir regla de memorización.

### 4.3 Relaciones abstractas

Expandir E13 a relaciones:

- is-a;
- parte-de;
- causa;
- antes/después;
- contiene;
- equivalente;
- opuesto;
- atributo;
- función;
- jerarquía.

No entrenar únicamente pares cue→label. Introducir suficientes ejemplos para que la regla tenga una estructura estadística real.

### 4.4 Composición

Entrenar:

R1 y R2 por separado.

Nunca entrenar:

R2(R1(x)).

Test:

R2(R1(x_new)).

La salida compuesta debe ser producida por Dφ sin consultar RQM.

## 5. E18 — aprendizaje de reglas

Crear un benchmark unificado de rule learning.

Para cada regla:

1. entrenar múltiples instancias;
2. generar instancias nuevas;
3. ocultar estados objetivo;
4. comprobar que el target equivalente tampoco aparece en otra forma;
5. ejecutar Dφ;
6. comparar contra controles.

Métricas:

- error absoluto;
- cosine;
- distancia de manifold;
- energía;
- estabilidad;
- norma;
- abstención;
- tiempo/steps;
- número de ejemplos necesarios.

Resultado esperado para considerar una señal fuerte:

Dφ debe generalizar mejor que static field, NN y modelos aleatorios en TEST estructuralmente nuevo.

## 6. E19 — prueba formal de no-lookup

Cada ejecución debe producir un reporte de provenance.

Campos obligatorios:

- target_seen_training
- target_seen_dev
- target_seen_cdt
- target_seen_rqm
- target_seen_table
- target_seen_nn
- target_seen_attractor
- target_equivalent_seen
- cdt_queries
- rqm_queries
- table_queries
- nn_queries
- attractor_queries
- direct_memory_queries

Para el modo FIELD_ONLY:

Todos los mecanismos de recuperación deben ser cero.

Si alguno participa en la producción del target:

`LEAKED`

y la corrida queda excluida de los resultados científicos.

No basta con demostrar que el target exacto no aparece. También se debe impedir que exista una representación equivalente que permita reconstruirlo directamente.

## 7. E20 — regla contra trayectoria

Entrenar muchas trayectorias independientes:

A1→B1→C1
A2→B2→C2
A3→B3→C3
...

Después:

A_new→?

La pregunta es si el sistema aprendió:

`f(A)=B`

o solamente:

`A1→B1→C1`.

El dataset debe contener suficientes trayectorias para que memorizar una sola ruta sea insuficiente.

## 8. E21 — extrapolación de parámetros

Primera fase:

TRAIN:
- dx = -2,-1,0,1,2

TEST:
- dx = 3

Segunda fase:

TRAIN:
- transformaciones con múltiples parámetros.

TEST:
- combinación de parámetros no observada.

Esto prueba si el campo aprende la estructura paramétrica o solo interpola ejemplos.

## 9. E22 — composición sin RQM

Entrenar R1 y R2 por separado.

TEST:

`R2(R1(x_new))`

En este modo:

- RQM OFF;
- CDT retrieval OFF;
- table OFF;
- NN OFF;
- attractor bank OFF.

Comparar:

1. Dφ;
2. static field;
3. MLP;
4. modelo lineal;
5. polinomial;
6. RQM;
7. NN.

El resultado principal debe ser el contraste Dφ vs controles, no el resultado del sistema completo.

## 10. E23 — horizonte largo

Entrenar únicamente one-step.

Evaluar:

- 1;
- 2;
- 4;
- 8;
- 16;
- 32 pasos.

Medir:

- error acumulado;
- cosine;
- energía;
- estabilidad;
- drift de norma;
- divergencia de trayectoria.

Añadir perturbaciones:

ε = 0.01, 0.05, 0.10, 0.20.

Un sistema dinámico útil debe mostrar una región de estabilidad medible, no únicamente acertar un paso.

## 11. E24 — comparación estático vs dinámico

El control debe ser extremadamente estricto.

Misma:

- inicialización del encoder;
- seed;
- dataset;
- presupuesto;
- número de updates;
- batch;
- optimizador cuando sea comparable.

Comparar:

STATIC:
`z = Encoder(x)`

DYNAMIC:
`z' = Dφ(z, context)`

La pregunta es:

> ¿Qué comportamiento aparece únicamente cuando existe una dinámica aprendida?

Esto evita atribuir a Dφ una mejora que en realidad proviene del encoder.

## 12. E25 — experimento central: CDT como experiencia

Este es el experimento prioritario.

### Fase A — experiencias

El sistema observa:

E1, E2, E3, E4...

Cada experiencia contiene estado, contexto, transición y resultado.

### Fase B — consolidación

CDT consolida las experiencias.

Pero CDT NO debe guardar:

`input_nuevo → target_nuevo`.

Debe guardar una representación de experiencia/regularidad:

- estadísticas;
- estructura;
- distribución;
- transición agregada;
- parámetros de dinámica;
- correlaciones;
- confianza;
- estructura topológica.

### Fase C — aprendizaje/adaptación

La información consolidada se utiliza para modificar/adaptar Dφ.

### Fase D — prueba

Se introduce:

X_new

donde Y_new:

- nunca estuvo en TRAIN;
- nunca estuvo en CDT;
- nunca estuvo en RQM;
- nunca estuvo en tabla;
- nunca estuvo en NN;
- nunca estuvo en attractor bank.

El sistema debe producir:

`Dφ(X_new) → Y_new`.

La métrica principal es la generación correcta de Y_new.

## 13. E26 — ablación de experiencia

Comparar:

A. sin experiencias previas;
B. experiencias recientes sin consolidación;
C. experiencias consolidadas en CDT;
D. CDT parcialmente corrupto;
E. CDT irrelevante;
F. CDT reducido a estadísticas agregadas.

Medir:

- muestras necesarias;
- pasos de entrenamiento;
- accuracy;
- generalización;
- transferencia;
- estabilidad;
- forgetting.

La hipótesis fuerte sería:

C > B > A

en generalización, pero C debe seguir generando respuestas nuevas.

Si C únicamente permite recuperar respuestas, la hipótesis queda falsada para ese protocolo.

## 14. E27 — transferencia entre familias

Aprender una regla en familia A.

Consolidarla.

Aplicarla a familia B completamente nueva.

Ejemplo:

TRAIN:
círculos A → transformación T.

TEST:
triángulos B → transformación T.

La salida no debe haber aparecido durante consolidación.

Esto prueba transferencia de regularidad y no solo transferencia de representación.

## 15. E28 — aprendizaje continuo

Secuencia:

R1 → consolidar
R2 → consolidar
R3 → consolidar
R4 → consolidar

Después evaluar todas.

Comparar:

- sin CDT;
- CDT activo;
- CDT corrupto;
- CDT irrelevante.

Medir simultáneamente:

- aprendizaje de R4;
- retención R1;
- transferencia R1→R4;
- interferencia.

## 16. E29 — intervención causal

Guardar checkpoint.

Ejecutar:

1. baseline;
2. intervención en parámetros/componentes de Dφ;
3. rollback.

Además:

4. intervención aleatoria de igual magnitud.

Resultado esperado:

- intervención estructurada cambia comportamiento de manera reproducible;
- intervención aleatoria produce un patrón significativamente distinto;
- rollback recupera el comportamiento.

Esto es importante porque demostraría que la dinámica aprendida no es decorativa.

## 17. E30 — persistencia real

Después de entrenar/consolidar:

1. cerrar proceso;
2. reiniciar;
3. cargar checkpoint;
4. eliminar RAM episódica;
5. ejecutar TEST nuevo.

Separar artefactos:

- FieldEncoder;
- Dφ;
- CDT;
- RAM episódica;
- RQM.

Ablaciones:

P0 Encoder + Dφ + CDT
P1 Encoder + Dφ
P2 Dφ
P3 CDT
P4 CDT retrieval.

La persistencia debe sobrevivir a restart sin depender de estado transitorio.

## 18. Nueva arquitectura recomendada

No aumentar complejidad indiscriminadamente.

Separar explícitamente:

`Periphery → Encoder → Field State → Dynamics → Decoder`

y:

`Experience → Consolidation → CDT → Learning Signal`

CDT no debería estar en el hot path como una base de respuestas.

La ruta científica principal debe ser:

`experiencia`
→ `CDT`
→ `regularidad consolidada`
→ `adaptación de Dφ`
→ `estado nuevo`
→ `respuesta nueva`

La ruta:

`estado nuevo → CDT → respuesta`

debe existir solamente como control de retrieval y nunca como evidencia principal.

## 19. Entrenamiento más extenso: sí, pero no simplemente más epochs

Aumentar entrenamiento por sí solo puede empeorar la validez.

La prioridad es:

1. más familias;
2. más instancias independientes;
3. más parámetros;
4. más transformaciones;
5. más composiciones;
6. más ruido;
7. más OOD;
8. más seeds.

Después aumentar epochs si las curvas de TRAIN/DEV muestran underfitting.

Usar learning curves para distinguir:

- underfitting;
- overfitting;
- memorización;
- verdadera generalización.

## 20. Escalado progresivo

Cada experimento debe ejecutarse en:

- N=8;
- N=16;
- N=32;
- N=64;
- N=128 cuando sea viable.

La dificultad debe crecer con el número de conceptos y relaciones.

Un sistema que solo funciona en N=8 no debe interpretarse como arquitectura general.

## 21. Seeds y estadística

Desarrollo:

- 8 seeds.

Resultado final:

- 16 seeds.

Reportar:

- mean;
- median;
- std;
- min/max;
- bootstrap 95% CI;
- paired delta;
- effect size;
- learning curves.

No declarar PASS por una única corrida.

Los estados:

- STRONG_POSITIVE
- POSITIVE
- PARTIAL
- NULL
- NEGATIVE
- LEAKED
- INVALID

deben ser automáticos.

## 22. Controles obligatorios

Toda afirmación de aprendizaje dinámico debe compararse contra:

1. Gemma raw;
2. random encoder;
3. static field;
4. random Dφ;
5. linear dynamics;
6. polynomial dynamics;
7. MLP;
8. nearest neighbor;
9. table;
10. RQM;
11. CDT retrieval;
12. dynamic without CDT;
13. dynamic + consolidated CDT.

El objetivo no es ganar contra todos en toda métrica. Es identificar qué capacidad aparece específicamente con el componente estudiado.

## 23. Nuevos controles especialmente importantes

### Control A — mayor capacidad

Un MLP suficientemente grande debe recibir el mismo presupuesto de entrenamiento.

Si Dφ solo gana porque el baseline es demasiado pequeño, el resultado no es concluyente.

### Control B — parámetro count

Reportar número de parámetros de:

- encoder;
- Dφ;
- baselines.

### Control C — compute budget

Misma cantidad aproximada de:

- ejemplos;
- updates;
- FLOPs cuando sea viable.

### Control D — decoder independiente

El decoder no puede esconder información del target.

Entrenar/seleccionar el decoder únicamente en TRAIN/DEV y congelarlo antes de TEST.

### Control E — shuffled labels

Romper las correspondencias manteniendo distribución y dificultad.

Si el campo sigue produciendo resultados similares, probablemente está explotando un artefacto.

## 24. Benchmark lingüístico ampliado

E12 debe evolucionar de perro/gato a familias semánticas:

Animales:
- perro, gato, lobo, zorro, caballo, águila...

Objetos:
- coche, bicicleta, avión...

Acciones:
- correr, saltar, comer...

Propiedades:
- grande, pequeño, rápido...

Relaciones:

- animal;
- mamífero;
- ser-vivo;
- objeto;
- acción;
- propiedad.

Usar español como entrenamiento y reservar familias completas para cross-lingual/OOD.

Idiomas:

- inglés;
- francés;
- alemán;
- japonés;
- portugués;
- variantes/paráfrasis.

No limitar el benchmark a traducciones palabra-a-palabra.

## 25. Composición lingüística

Entrenar:

- perro corre;
- perro come;
- gato corre;
- gato come.

Evaluar:

- perro come;
- gato corre;
- perro salta;
- gato salta.

Pero después escalar a:

- sujeto nuevo;
- verbo nuevo;
- propiedad nueva;
- relación nueva;
- combinaciones múltiples.

El objetivo es comprobar composición estructural y no clasificación.

## 26. Prueba crítica de dependencia del LLM

Una vez entrenado el campo:

1. congelar checkpoint;
2. cambiar Gemma por otro modelo;
3. cambiar tokenizer;
4. cambiar idioma del prompt;
5. mantener encoder/field lo más estable posible.

Preguntar si la estructura conceptual consolidada sobrevive.

Esto es fundamental para la hipótesis de que el LLM es una periferia lingüística y no el verdadero almacén cognitivo.

## 27. Criterio de éxito fuerte

La siguiente afirmación solo debe hacerse si se cumplen simultáneamente:

- target nunca almacenado;
- target equivalente nunca almacenado;
- leakage = 0;
- RQM OFF;
- CDT retrieval OFF durante TEST;
- table OFF;
- NN OFF;
- attractor bank OFF;
- decoder independiente;
- múltiples seeds;
- múltiples familias;
- mejor que static;
- mejor que random;
- mejor que controles dinámicos simples;
- persistencia tras restart;
- intervención causal reproducible;
- transferencia a familia nueva.

Entonces sería razonable concluir:

> El sustrato externo aprendió una regularidad de experiencias y puede utilizar su dinámica aprendida para transformar estados nuevos sin recuperar directamente una respuesta almacenada.

## 28. Qué NO demostraría todavía

Incluso un resultado positivo no demostraría por sí mismo:

- consciencia;
- subjetividad;
- AGI;
- vida biológica;
- experiencia fenomenológica.

Sí podría aportar evidencia de:

- memoria externa persistente;
- aprendizaje de dinámica;
- generalización estructural;
- transferencia;
- consolidación;
- generación de estados nuevos;
- separación entre periferia lingüística y sustrato cognitivo.

## 29. Orden de implementación

### Fase 0
Clean-room + generadores + manifests + hashes + provenance.

### Fase 1
Controles y baselines.

### Fase 2
E18/E19/E20.

### Fase 3
E21/E22/E23.

### Fase 4
E24.

### Fase 5
E25 — CDT como experiencia.

### Fase 6
E26/E27/E28.

### Fase 7
E29/E30.

### Fase 8
Cross-lingual ampliado y cambio de LLM.

### Fase 9
16 seeds + escalado N=8→128.

## 30. Resultado que buscamos

El experimento definitivo no es:

`LLM + memoria = respuesta correcta`

sino:

`experiencias`
→ `consolidación`
→ `regularidad`
→ `dinámica del campo`
→ `estado jamás observado`
→ `respuesta jamás almacenada`

Si el sistema falla ahí, el proyecto debe identificar exactamente dónde falla.

Si funciona, la siguiente pregunta pasa a ser cuánto de esa capacidad escala cuando aumentamos:

- diversidad;
- número de conceptos;
- número de reglas;
- profundidad composicional;
- horizonte temporal;
- ruido;
- cambio de lenguaje;
- cambio de LLM.

## 31. Artefactos obligatorios

Cada experimento debe producir:

- manifest TRAIN/DEV/TEST;
- hashes;
- configuración congelada;
- seed;
- checkpoint;
- metrics.json;
- metrics.csv;
- provenance.json;
- leakage_report.json;
- learning_curve.csv;
- stdout/stderr;
- versión de Rust;
- commit SHA;
- hash del GGUF;
- número de parámetros;
- presupuesto de entrenamiento.

Ningún resultado debe considerarse reproducible si faltan estos artefactos.

## 32. Regla de la rama

Esta rama comienza desde `main`.

Las ramas experimentales anteriores son evidencia histórica, no pesos ni datos reutilizables.

El objetivo es obtener una segunda generación de evidencia limpia y comparable que responda la pregunta central del proyecto sin contaminación retrospectiva.
