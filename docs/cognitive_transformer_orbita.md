# Cognitive Transformer + ÓRBITA

**Rama de investigación:** `exp/cognitive-transformer-orbita`  
**Estado:** hipótesis arquitectónica / diseño experimental.  
**Relación:** extensión conceptual de ÓRBITA, sin modificar su implementación ni sus resultados publicados.

---

## 1. Idea central

La hipótesis es construir un sistema con mecanismos tipo **Transformer y atención**, pero sin utilizar tokens lingüísticos como unidad fundamental de entrenamiento.

La diferencia clave no es eliminar la operación matemática de atención, sino cambiar aquello sobre lo que opera y el objetivo de aprendizaje:

```text
Transformer convencional

texto → tokens → embeddings → atención → MLP → logits → siguiente token
```

frente a:

```text
Cognitive Transformer

estado_t → atención sobre estados/memoria → dinámica → estado_{t+1}
```

El sistema no aprendería principalmente una distribución de siguiente token. Aprendería cómo evolucionan estados internos no lingüísticos y cómo distintas partes de un estado o memoria se relacionan entre sí.

La hipótesis es especialmente interesante al combinar tres mecanismos con funciones distintas:

```text
LLM              = periferia lingüística
Cognitive        = atención/relaciones sobre estados
ÓRBITA           = dinámica y evolución del estado
CDT-RQM          = memoria persistente y consolidación
```

El LLM deja de ser el lugar donde necesariamente reside toda la cognición aprendida y pasa a ser una interfaz entre lenguaje y un sustrato cognitivo externo.

---

# 2. Motivación

Un Transformer obtiene gran parte de su potencia de la atención: puede seleccionar dinámicamente qué partes de su contexto son relevantes para producir una representación.

Sin embargo, en un LLM convencional, el contexto está organizado alrededor de una secuencia de tokens y el objetivo principal suele estar ligado a la predicción lingüística.

La pregunta de esta rama es:

> **¿Qué ocurre si conservamos atención, pero sustituimos los tokens lingüísticos por entidades de estado y entrenamos el sistema para predecir evolución cognitiva en lugar de texto?**

Esto permitiría separar conceptualmente:

1. representación lingüística;
2. representación del estado del mundo o de una tarea;
3. atención relacional;
4. dinámica temporal;
5. memoria persistente;
6. generación lingüística.

No se afirma que esta separación produzca AGI ni conciencia. Es una hipótesis de arquitectura que debe someterse a experimentación.

---

# 3. Arquitectura propuesta

La arquitectura inicial propuesta es:

```text
                    ┌──────────────────────┐
                    │   LLM lingüístico    │
                    │      periférico      │
                    └──────────┬───────────┘
                               │
                        encoder / bridge
                               │
                               ▼
                    ┌──────────────────────┐
                    │  Estado latente z_t  │
                    └──────────┬───────────┘
                               │
                 ┌─────────────┴─────────────┐
                 │                           │
                 ▼                           ▼
       ┌──────────────────┐        ┌──────────────────┐
       │      ÓRBITA      │        │     CDT-RQM      │
       │ dinámica local   │        │ memoria externa  │
       └────────┬─────────┘        └────────┬─────────┘
                │                           │
                │ Q                         │ K,V
                │                           │
                └────────────┬──────────────┘
                             ▼
                    ┌──────────────────┐
                    │    Attention     │
                    │ estado ↔ memoria │
                    └────────┬─────────┘
                             │
                          Δz / contexto
                             │
                             ▼
                    ┌──────────────────┐
                    │      ÓRBITA      │
                    │ evolución z_t+1  │
                    └────────┬─────────┘
                             │
                     prediction error
                             │
                             ▼
                    ┌──────────────────┐
                    │    CDT-RQM       │
                    │ consolidación    │
                    └──────────────────┘
                             │
                             ▼
                    decoder lingüístico
                             │
                             ▼
                           texto
```

Una formulación compacta sería:

```text
z_t = Encoder(LLM(x_t))
M_t = Retrieve(CDT-RQM, z_t)
Q_t = f_orbita(z_t)
K_t,V_t = EncodeMemory(M_t)
A_t = MultiHeadAttention(Q_t, K_t, V_t)
z'_t = z_t + A_t
z_{t+1} = ÓRBITA(z'_t, context_t)
error_t = ||z_{t+1} - z^*_{t+1}||
CDT-RQM ← Consolidate(z_t, z_{t+1}, error_t)
y_t = Decoder(z_{t+1})
```

La forma exacta de esta ecuación queda deliberadamente abierta para experimentación.

---

# 4. Qué significa "sin tokens"

La expresión "sin entrenar tokens" debe interpretarse con precisión.

Un Transformer necesita vectores/slots sobre los cuales realizar atención. Por tanto, probablemente existirán unidades internas que técnicamente podrían llamarse tokens, pero **no serán tokens lingüísticos**.

Para evitar confusión, esta investigación utiliza preferentemente los términos:

- `latent state slots`;
- `state nodes`;
- `concept nodes`;
- `cognitive entities`.

Estos elementos podrían representar:

- objetos;
- conceptos;
- variables de estado;
- relaciones;
- acciones;
- estados físicos;
- recuerdos;
- hipótesis;
- agentes;
- regiones del CDT-RQM.

La diferencia fundamental es que estos estados no se entrenan para predecir la siguiente palabra.

---

# 5. El problema crítico: un único vector no basta

Si toda la cognición se representa como un único vector `z`, aplicar atención sobre ese único vector puede reducirse a una transformación equivalente a una red neuronal convencional.

La ventaja relacional de un Transformer aparece cuando existen múltiples entidades sobre las que la atención puede seleccionar relaciones.

Por ello, una implementación más fuerte debería representar el estado como un conjunto estructurado:

```text
z_t = {s_1, s_2, ..., s_N}
```

donde cada `s_i` es un estado latente o nodo cognitivo.

Entonces la atención puede calcular relaciones:

```text
Attention(s_i, {s_1 ... s_N, m_1 ... m_K})
```

y determinar qué entidades son relevantes para la evolución de `s_i`.

Esto abre una conexión natural con CDT-RQM, donde la memoria podría devolver nodos y relaciones en lugar de un vector indiferenciado.

---

# 6. Atención como mecanismo cognitivo no lingüístico

La hipótesis experimental no es que "la atención es cognición".

La hipótesis más limitada es:

> **Un mecanismo de atención aplicado sobre estados no lingüísticos puede aprender relaciones dinámicas útiles para predecir la evolución de un sistema sin utilizar predicción de tokens.**

La atención tendría una función de selección relacional:

```text
estado actual
     │
     ├── ¿qué partes del estado son relevantes?
     │
     ├── ¿qué recuerdos tienen relación causal?
     │
     ├── ¿qué entidades deben influir en la predicción?
     │
     └── ¿qué información debe incorporarse antes de evolucionar?
```

ÓRBITA tendría una función diferente:

```text
atención → selecciona/infiere influencia contextual
ÓRBITA   → evoluciona el estado bajo una dinámica estructurada
```

Esta separación debe comprobarse mediante ablaciones.

---

# 7. ÓRBITA como operador dinámico

La implementación actual de ÓRBITA no debe reinterpretarse como una matriz exponencial literal.

En el código actual, el núcleo utiliza una transformación estructurada mediante:

- transformación de Cayley para `Q`;
- bloques rotacionales;
- escalas expansivas/contractivas;
- traslación `β`;
- corrector residual.

Conceptualmente, el núcleo representa una familia de dinámicas latentes condicionadas por contexto.

Por tanto, la arquitectura híbrida puede usar ÓRBITA como **operador de transición**:

```text
z'_t = Attention(z_t, Memory_t)
z_{t+1} = ÓRBITA(z'_t, c_t)
```

La atención no reemplaza necesariamente la dinámica. Puede actuar como mecanismo para modificar el estado antes de que la dinámica estructurada lo propague.

---

# 8. Atención + ÓRBITA

La primera arquitectura experimental mínima debería ser:

```text
                ┌─────────────┐
                │ estado z_t  │
                └──────┬──────┘
                       │
                       ▼
                ┌─────────────┐
                │   Queries   │
                └──────┬──────┘
                       │
             ┌─────────┴─────────┐
             │                   │
             ▼                   ▼
          Memoria             estado
          K,V                  contextual
             │                   │
             └─────────┬─────────┘
                       ▼
                  Attention
                       │
                       ▼
                  Δz_t / z'_t
                       │
                       ▼
                    ÓRBITA
                       │
                       ▼
                  z_{t+1}
```

Una formulación sencilla:

\[
Q_t = W_Q f(z_t)
\]

\[
K_t = W_K M_t,
\qquad
V_t = W_V M_t
\]

\[
A_t = \operatorname{softmax}\left(\frac{Q_tK_t^T}{\sqrt d}\right)V_t
\]

\[
z'_t = z_t + W_OA_t
\]

\[
z_{t+1}=\operatorname{ÓRBITA}(z'_t,c_t)
\]

La formulación debe considerarse un punto de partida, no una arquitectura final.

---

# 9. CDT-RQM como memoria para la atención

CDT-RQM no debería actuar simplemente como una base de datos de textos.

La hipótesis de esta arquitectura es que puede almacenar estructuras que luego sean recuperables como estados:

```text
CDT-RQM
   │
   ├── nodos de estado
   ├── relaciones
   ├── atractores
   ├── asociaciones
   ├── transiciones consolidadas
   └── contexto de experiencias anteriores
```

El módulo de retrieval podría devolver:

```text
M_t = {
    m_1,
    m_2,
    ...,
    m_k
}
```

Cada `m_i` sería una representación latente, potencialmente acompañada por metadatos estructurales.

La atención determinaría cuáles son relevantes para el estado actual.

Esto crea una separación importante:

```text
CDT-RQM = almacena
Attention = selecciona relacionando
ÓRBITA = evoluciona
```

---

# 10. Objetivo de entrenamiento

El sistema no debe entrenarse inicialmente para predecir tokens.

El objetivo principal sería predicción de estados:

\[
L_{1}=||\hat z_{t+1}-z_{t+1}^{*}||^2
\]

Para evaluar consistencia temporal:

\[
L_{future}=\sum_{h\in H}\gamma^h
||\hat z_{t+h}-z_{t+h}^{*}||^2
\]

con:

```text
H = {1, 2, 4, 8}
```

como punto de partida, siguiendo la lógica experimental ya utilizada en ÓRBITA.

Posteriormente pueden probarse horizontes no entrenados:

```text
3, 6, 12, 24, 48, 96, 192, 256
```

La hipótesis fuerte sería que la arquitectura aprende una dinámica que generaliza en el tiempo, no que memoriza puntos de supervisión.

---

# 11. Cómo obtener estados sin convertir el sistema en otro LLM

El encoder debe ser tratado como un problema de investigación independiente.

Una primera opción sería utilizar una representación interna del LLM como señal semántica y aprender una proyección:

```text
texto
  ↓
LLM
  ↓
hidden state
  ↓
encoder/projection
  ↓
z_t
```

Pero existe un riesgo importante: si `z_t` simplemente copia los embeddings o hidden states del LLM, el supuesto sustrato independiente puede ser superficial.

Por ello se deberían evaluar objetivos que obliguen al estado a capturar propiedades funcionales y no sólo similitud lingüística.

Ejemplos:

1. predicción del siguiente estado;
2. predicción multi-horizonte;
3. equivalencia entre descripciones lingüísticas diferentes del mismo estado;
4. separación de estados que tienen consecuencias diferentes;
5. consistencia bajo paráfrasis;
6. recuperación de memoria relevante;
7. predicción de acciones o resultados.

---

# 12. Un criterio especialmente importante: equivalencia semántica

Dos expresiones diferentes deberían poder producir estados próximos si describen la misma situación:

```text
"El objeto cayó al suelo."
"El objeto se desplomó."
```

Idealmente:

\[
d(z_1,z_2) \approx 0
\]

mientras que estados con consecuencias distintas deberían permanecer separables.

Esto permitiría que el espacio cognitivo sea parcialmente independiente del lenguaje.

Sin embargo, no debe imponerse una equivalencia absoluta: dos frases lingüísticamente similares pueden implicar estados causales diferentes dependiendo del contexto.

Por ello, el criterio correcto es **equivalencia funcional**, no sólo similitud semántica superficial.

---

# 13. Riesgo de convertirlo en un Transformer convencional disfrazado

Esta es una de las objeciones principales.

Si se hace:

```text
hidden state → Transformer → MLP → prediction
```

y únicamente se cambia el nombre de los vectores, el experimento no demostraría una arquitectura cognitiva distinta.

Para que la hipótesis sea falsable, deben mantenerse separadas las funciones:

```text
LLM
  └── lenguaje

Encoder
  └── interfaz lenguaje → estado

Attention
  └── relaciones/selección contextual

ÓRBITA
  └── dinámica temporal estructurada

CDT-RQM
  └── persistencia/consolidación

Decoder
  └── estado → lenguaje
```

Además, deben existir ablaciones que demuestren qué componente aporta qué propiedad.

---

# 14. Ablaciones mínimas

La primera matriz experimental debería incluir:

| Modelo | Atención | ÓRBITA | CDT-RQM | Objetivo |
|---|---:|---:|---:|---|
| MLP | — | — | — | baseline |
| Attention | ✓ | — | — | medir atención sola |
| ÓRBITA | — | ✓ | — | baseline dinámico |
| Attention + ÓRBITA | ✓ | ✓ | — | interacción principal |
| Attention + memoria | ✓ | — | ✓ | medir retrieval |
| Attention + ÓRBITA + memoria | ✓ | ✓ | ✓ | arquitectura completa |

Para memoria, añadir además:

```text
memoria aleatoria
memoria congelada
memoria entrenable
memoria consolidada CDT-RQM
```

Esto evitaría atribuir una mejora de retrieval genérico a CDT-RQM específicamente.

---

# 15. Benchmark inicial recomendado

No comenzar con lenguaje abierto ni con un benchmark enorme.

El primer experimento debe ser controlado y causal.

### Fase 1 — dinámica sintética

Generar sistemas donde exista una transición conocida:

```text
z_t → z_{t+1}
```

con:

- rotaciones;
- contracciones/expansiones;
- traslaciones;
- composiciones;
- perturbaciones;
- interacciones entre entidades.

### Fase 2 — estados relacionales

Representar múltiples entidades:

```text
{A, B, C, D, ...}
```

con relaciones que cambian en el tiempo.

### Fase 3 — memoria

Introducir información que debe recuperarse para predecir correctamente un estado futuro.

### Fase 4 — lenguaje

Sólo después conectar el LLM como interfaz.

Este orden permite determinar si la arquitectura funciona antes de introducir la enorme complejidad del lenguaje.

---

# 16. Experimento mínimo de alto valor

La primera prueba de esta rama debería comparar únicamente cuatro sistemas:

```text
1. MLP
2. ÓRBITA
3. Attention
4. Attention + ÓRBITA
```

Todos con presupuesto paramétrico y cómputo registrados.

Entrenamiento:

```text
z_t → z_{t+1}
```

Evaluación:

```text
h1
h2
h4
h8
h16
h32
h64
h128
h256
```

Y adicionalmente:

```text
h3
h6
h12
h24
h48
h96
h192
```

Las métricas deben incluir:

- error de estado;
- error relativo;
- accuracy si existe una clasificación asociada;
- norma del estado;
- estabilidad;
- colapso;
- divergencia temporal;
- coste computacional.

El criterio principal no debe ser únicamente `h1`.

---

# 17. Hipótesis falsables

### H1 — atención no lingüística

Un mecanismo de atención sobre estados supera a un baseline MLP de capacidad comparable en tareas donde las relaciones entre entidades son relevantes.

### H2 — dinámica estructurada

ÓRBITA conserva mejor la estabilidad multi-horizonte que una arquitectura sin restricciones dinámicas, especialmente fuera de los horizontes usados durante entrenamiento.

### H3 — complementariedad

Attention + ÓRBITA supera de forma reproducible a Attention sola y ÓRBITA sola en tareas relacionales dinámicas.

### H4 — memoria

La incorporación de memoria estructurada mejora predicción únicamente cuando el estado futuro depende de información que no está contenida en el estado local actual.

### H5 — independencia lingüística

Un estado latente entrenado con objetivos funcionales puede conservar información predictiva cuando la descripción lingüística de una misma situación cambia.

Estas hipótesis pueden ser refutadas.

---

# 18. Qué resultado sería realmente interesante

El resultado más importante no sería simplemente:

```text
Attention + ÓRBITA tiene menor loss.
```

Sería observar simultáneamente:

1. mejor predicción a largo plazo;
2. menor degradación fuera de los horizontes entrenados;
3. robustez a paráfrasis o cambios de lenguaje;
4. recuperación de memoria estructural relevante;
5. mejora que desaparece cuando se elimina el mecanismo correspondiente;
6. estabilidad reproducible en DEV y CONFIRM.

En ese caso habría evidencia de que la separación entre atención, dinámica y memoria tiene valor funcional.

---

# 19. Qué NO demostraría el experimento

Incluso un resultado positivo no demostraría por sí mismo:

- AGI;
- conciencia;
- vida artificial;
- comprensión humana;
- que el sistema sea físicamente equivalente al cerebro;
- que CDT-RQM sea una memoria universal;
- que ÓRBITA represente una ley física real.

Demostraría, como máximo, que una determinada arquitectura de estados + atención + dinámica estructurada + memoria puede aprender determinadas regularidades de evolución con propiedades que los baselines no reproducen igual de bien.

---

# 20. Hipótesis arquitectónica de mayor nivel

Si las pruebas anteriores fueran positivas, la arquitectura podría expresarse como:

```text
             LANGUAGE
                │
                ▼
        ┌───────────────┐
        │ LLM peripheral│
        └───────┬───────┘
                │
          semantic encoder
                │
                ▼
       ┌───────────────────┐
       │ latent state space│
       └─────────┬─────────┘
                 │
          ┌──────┴──────┐
          │             │
          ▼             ▼
     CDT-RQM         current state
     memory              │
          │              │
          └──────┬───────┘
                 ▼
             ATTENTION
                 │
             Δ context
                 │
                 ▼
              ÓRBITA
                 │
                 ▼
            future state
                 │
          ┌──────┴───────┐
          │              │
          ▼              ▼
       prediction      decoder
          │              │
          ▼              ▼
       learning       language
          │
          ▼
       CDT-RQM
      consolidation
```

La arquitectura propone una división funcional:

> **La atención determina qué relaciones importan; ÓRBITA determina cómo evoluciona el estado; CDT-RQM determina qué permanece; el LLM determina cómo se expresa y recibe lenguaje.**

Esta frase es una hipótesis de diseño, no una conclusión científica.

---

# 21. Relación con la investigación ÓRBITA existente

Esta rama no sustituye `exp/orbita`.

La rama actual de ÓRBITA ya proporciona un punto de partida experimental para estudiar dinámica estructurada y extrapolación temporal.

La nueva rama debe reutilizar, cuando sea metodológicamente válido:

- dataset sellado;
- filosofía de preregistro;
- DEV/CONFIRM;
- bootstrap pareado;
- auditoría de leakage;
- comparación de horizontes;
- pruebas de extrapolación;
- paridad Rust/PyTorch cuando corresponda.

Sin embargo, no se deben transferir resultados de ÓRBITA automáticamente al nuevo sistema. Cada arquitectura debe obtener sus propios resultados.

---

# 22. Ruta de implementación propuesta

## Paso 1 — estado sintético

Implementar un dataset donde cada ejemplo sea:

```text
state_t → state_{t+1}
```

sin lenguaje.

## Paso 2 — ÓRBITA como baseline

Reutilizar la lógica actual de ÓRBITA adaptándola a estados de mayor dimensionalidad o a múltiples slots.

## Paso 3 — atención

Añadir Multi-Head Attention sobre slots de estado.

## Paso 4 — híbrido

Implementar:

```text
Attention → ÓRBITA
```

sin CDT-RQM.

## Paso 5 — memoria

Añadir retrieval de estados estructurados.

## Paso 6 — consolidación

Conectar error de predicción con actualización/consolidación de CDT-RQM.

## Paso 7 — LLM

Conectar una periferia lingüística mediante encoder/decoder.

## Paso 8 — evaluación causal

Ejecutar la matriz completa de ablaciones.

---

# 23. Riesgos técnicos principales

### Riesgo 1 — colapso del estado

Todos los estados podrían converger a representaciones similares.

**Control:** diversidad, reconstrucción funcional y tareas con consecuencias diferentes.

### Riesgo 2 — atención como MLP sofisticado

Si existe un único vector global, la atención podría aportar poco.

**Control:** utilizar múltiples slots/nodos y tareas relacionales.

### Riesgo 3 — encoder dependiente del lenguaje

El espacio latente podría ser sólo una copia comprimida del LLM.

**Control:** equivalencia funcional entre múltiples formulaciones lingüísticas.

### Riesgo 4 — sobreajuste temporal

El modelo podría memorizar `h1,h2,h4,h8`.

**Control:** horizontes no potencia de dos y extrapolación a `h128/h256`.

### Riesgo 5 — memoria trivial

El sistema podría beneficiarse de retrieval sin que CDT-RQM aporte una propiedad especial.

**Control:** comparar contra memorias alternativas.

### Riesgo 6 — ventaja por capacidad

Una arquitectura con atención puede tener más parámetros.

**Control:** registrar parámetros, FLOPs y tiempo; incluir controles de capacidad cuando sea viable.

### Riesgo 7 — dinámica artificialmente restringida

ÓRBITA podría ganar únicamente porque el benchmark fue construido demasiado cerca de sus inductive biases.

**Control:** incluir tareas donde la dinámica no coincida con la estructura de ÓRBITA.

---

# 24. Criterio de éxito de esta rama

No se considerará exitosa porque el sistema sea conceptualmente elegante.

El criterio mínimo será:

```text
Attention + ÓRBITA
        >
Attention
        > o ≈
ÓRBITA
        >
baseline
```

en al menos una familia de tareas relacionales dinámicas, con:

- mejora reproducible;
- ventaja multi-horizonte;
- extrapolación medible;
- ablaciones consistentes;
- ausencia de leakage;
- coste computacional documentado.

Si no se cumple, la hipótesis deberá modificarse o descartarse.

---

# 25. Próximo experimento recomendado

El primer experimento de implementación debe ser deliberadamente pequeño:

```text
Input: conjunto de estados latentes

                 ┌─────────────┐
state_t ────────►│  Attention  │
                 └──────┬──────┘
                        │
                        ▼
                 ┌─────────────┐
                 │   ÓRBITA    │
                 └──────┬──────┘
                        │
                        ▼
                  state_{t+1}
```

Comparar contra:

```text
MLP
ÓRBITA
Attention
Attention + ÓRBITA
```

No conectar todavía el LLM ni CDT-RQM.

La razón es metodológica: primero debe demostrarse que la combinación de **atención sobre estados + dinámica ÓRBITA** aporta una propiedad medible por sí misma. Después se incorpora memoria y finalmente lenguaje.

---

# 26. Conclusión provisional

La idea no debe formularse como "un Transformer sin tokens" en el sentido de eliminar los vectores discretizados internamente.

La formulación técnicamente más precisa es:

> **Un Transformer cognitivo de estados: un mecanismo de atención que opera sobre entidades latentes no lingüísticas, acoplado a una dinámica estructurada ÓRBITA y, posteriormente, a una memoria persistente CDT-RQM, con un LLM utilizado únicamente como periferia lingüística.**

La pregunta científica central es entonces:

> **¿Puede una arquitectura que separa atención relacional, dinámica temporal, memoria persistente y lenguaje aprender y extrapolar estructuras cognitivas de forma más estable que un predictor convencional de estados de capacidad comparable?**

Ésta es una hipótesis experimentalmente falsable.

El siguiente paso no es afirmar que ya existe una nueva forma de cognición, sino construir el benchmark mínimo que permita descubrir si esta separación funcional produce una ventaja real.
