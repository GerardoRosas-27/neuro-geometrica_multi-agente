# Experimentos 11–17 — campo entrenable con LLM como periferia

**Rama:** exp/liquid-inference-experiments-8-9-10  
**Base experimental:** resultados E8–E10 de esta rama + arquitectura campo/líquido/CDT/RQM existente.  
**Propósito:** definir el siguiente ciclo para comprobar si el sustrato de campo aprende representaciones y dinámica propias, manteniendo al LLM congelado como periferia lingüística.

---

## 0. Qué demostraron realmente E8–E10

Los resultados actuales son útiles, pero deben interpretarse con precisión.

### E8 — invariancia de representación

Tres semillas reportan:

- acc_seen = 1.0
- acc_unseen = 1.0
- acc_ood = 0.50, 0.625 y 0.50
- 3–4 abstenciones/rechazos OOD por semilla
- clasificación correcta: PARTIAL_PASS

La ejecución declara que la periferia fue gemma-shaped-lexicon porque no estaba disponible el GGUF. Por tanto, E8 todavía **no demuestra invariancia cross-lingual real con Gemma**.

Además, E8 usa ConceptAttractorBank y matching por coseno. Esto evita el colapso del projector, pero introduce un riesgo metodológico: el banco puede convertir parte de la tarea en clasificación/nearest-neighbor de fingerprints.

### E9 — composición relacional

Las tres semillas obtienen:

- acc_seen = 1.0
- acc_unseen = 1.0
- acc_ood = 1.0
- margen top-1/top-2 ≈ 0.841
- control liquid-only compose = 0/3
- composición mediante infer_compose y caminata RQM

Esto demuestra una forma de **composición relacional implementada**, pero no debe atribuirse todavía a una dinámica autónoma del campo: RQM participa en los saltos de composición.

### E10 — predicción futura

Las tres semillas obtienen 100 % en seen/unseen/OOD y margen ≈ 0.845.

Está correctamente clasificado como PASS_PARTIAL porque la predicción se apoya en compose + wave. Falta demostrar que una dinámica aprendida por el campo produzca estados nuevos sin que composición relacional o una tabla de relaciones ya conocida haga el trabajo.

### Pregunta que queda abierta

> ¿El campo está aprendiendo una representación y una dinámica propias, o está organizando/recuperando información que ya contiene la periferia lingüística y RQM?

E11–E17 deben responder exactamente esa pregunta.

---

# 1. Arquitectura objetivo

La arquitectura experimental debe separar tres aprendizajes:

    texto
      │
      ▼
    LLM congelado
      │ hidden states
      ▼
    FieldEncoder entrenable
      │
      ▼
    Campo Ψ
      ├── representación
      ├── dinámica
      └── memoria/consolidación
      │
      ├── Dynamic Field
      ├── Thermo CDT
      └── decoder lingüístico

Formalmente:

- Encoder: Eθ(hLLM) → Ψ
- Dinámica: Dφ(Ψt, contexto) → Ψt+1
- Memoria: M(Ψ) → Ψ*

**E11–E15** deben aislar principalmente el encoder/campo.  
**E16–E17** deben atacar la dinámica.  
CDT debe funcionar como memoria y no como sustituto oculto de la dinámica.

---

# 2. Regla fundamental: LLM congelado

Durante E11–E17:

- no actualizar pesos del LLM;
- no hacer fine-tuning;
- no usar next-token loss para entrenar el campo;
- no almacenar token IDs dentro de FieldState;
- no permitir que el campo consulte directamente tokens;
- separar trainable/frozen en el reporte;
- guardar checksum/versionado del encoder lingüístico.

La variable experimental debe ser el campo, no Gemma.

---

# 3. E11 — Entrenamiento del campo con LLM congelado

## Pregunta

¿Puede un campo entrenable aprender una representación útil a partir de estados producidos por un LLM congelado, sin entrenar el LLM?

## Hipótesis

Eθ(h) debería aprender una geometría donde ejemplos semánticamente relacionados se acerquen y ejemplos incompatibles se separen.

No se busca reconstruir texto. Se busca aprender estructura en Ψ.

## Diseño

Dataset mínimo:

- animales: perro, gato, lobo;
- objetos: mesa, silla, coche;
- propiedades: frío, caliente, grande, pequeño;
- acciones: correr, comer, dormir.

Para cada concepto:

1. obtener hidden states del LLM congelado;
2. proyectar al campo;
3. entrenar únicamente FieldEncoder;
4. medir geometría antes/después.

## Objetivos

Usar como mínimo:

- L_cluster: agrupa conceptos de la misma familia;
- L_margin: separa familias distintas;
- opcional L_energy: penaliza estados inestables.

Por ejemplo:

    L = L_cluster + λ L_margin + β L_energy

## Controles

1. hidden state sin projector entrenado;
2. projector aleatorio;
3. projector entrenado;
4. embedding/vector DB;
5. campo sin dinámica;
6. campo con dinámica.

## Métricas

- intra-class cosine;
- inter-class cosine;
- margen;
- k-NN accuracy;
- silhouette;
- energía;
- estabilidad tras perturbación;
- pasos hasta convergencia.

## Criterio

No basta con mejorar clasificación. El campo debe conservar estructura ante:

- cambios de redacción;
- cambios de orden;
- ruido;
- paráfrasis;
- cambio de idioma cuando se use GGUF real.

---

# 4. E12 — concepto independiente del lenguaje

## Pregunta

¿perro, dog, chien y 犬 pueden converger al mismo atractor sin enseñar explícitamente las cuatro etiquetas al campo?

## Protocolo

### Train

Entrenar solamente:

    perro → C

### Holdout

No entrenar:

- dog
- chien
- 犬

### Test

Medir:

- d(Ψperro, Ψdog)
- d(Ψperro, Ψchien)
- d(Ψperro, Ψ犬)

y compararlas contra:

- d(Ψperro, Ψgato)

## Requisito crítico

Este experimento debe usar **Gemma GGUF real**, no gemma-shaped-lexicon.

El E8 anterior queda como baseline histórico, pero no como evidencia cross-lingual.

## Extensión fuerte

Añadir:

- paráfrasis;
- singular/plural;
- género;
- diminutivos;
- contexto;
- frases completas.

Ejemplos:

- "el perro"
- "un perro grande"
- "mi perro corre"
- "dog"
- "a large dog"
- "chien qui court"

El objetivo no es que los hidden states sean idénticos, sino que el **campo aprenda a eliminar variaciones lingüísticas irrelevantes**.

## Control indispensable

Comparar contra:

    cos(h_perro, h_dog)

sin pasar por el campo.

Si Gemma ya separa correctamente los conceptos, una mejora del campo podría ser trivial.

---

# 5. E13 — el campo aprende relaciones, no solo conceptos

## Pregunta

¿Puede el campo descubrir una estructura relacional a partir de múltiples ejemplos?

## Dataset

Entrenar:

    perro → animal
    gato → animal
    perro → mamífero
    gato → mamífero
    águila → ave

No entrenar:

    lobo → animal
    lobo → mamífero
    águila → ser-vivo

## Comparación

A. Tabla directa: cue → label  
B. RQM: cue → relation → label  
C. Campo estático: Ψ(cue) → Ψ(label)  
D. Campo dinámico: Ψt → Dφ(Ψt) → Ψt+1

La comparación C vs D es crítica: un campo estático puede ser solamente un embedding sofisticado.

## Métricas

- relaciones vistas;
- relaciones no vistas;
- composición;
- OOD;
- energía;
- margen;
- abstención;
- pasos dinámicos.

---

# 6. E14 — aprendizaje incremental y olvido catastrófico

## Pregunta

¿Puede el campo aprender nuevos conceptos sin destruir los anteriores?

## Protocolo

Entrenar secuencialmente:

    Fase A → A
    Fase B → B
    Fase C → C
    Fase D → D

Después de cada fase:

- consultar A;
- consultar B;
- consultar el concepto nuevo;
- medir perturbación de los atractores anteriores.

## Matriz

| Fase | Entrenado | Evaluar |
|---|---|---|
| 1 | A | A |
| 2 | B | A+B |
| 3 | C | A+B+C |
| 4 | D | A+B+C+D |

## Métrica principal

    ΔR_A = R_A(post) - R_A(pre)

y análogamente para B/C/D.

No aceptar solamente accuracy final = 100 %. Registrar la trayectoria completa.

## Prueba adversarial

Después de aprender D:

- invertir etiquetas de D;
- introducir ruido;
- consolidar;
- volver a medir A/B/C.

Esto prueba si la plasticidad local contamina atractores anteriores.

---

# 7. E15 — aprendizaje semántico sin etiquetas explícitas

## Pregunta

¿Puede el campo detectar regularidades estructurales sin recibir una etiqueta de clase?

## Dataset

    El perro corre.
    El perro come.
    El perro ladra.

    El gato corre.
    El gato come.
    El gato maúlla.

No enseñar:

    El perro maúlla.
    El gato ladra.

## Objetivo

El campo debe construir una representación donde:

- perro comparte estructura con gato;
- correr y comer son relaciones compartidas;
- ladrar y maullar son relaciones específicas.

## Prueba

Consultar:

- perro + corre;
- gato + corre;
- perro + maúlla;
- gato + ladra.

No exigir simplemente verdadero/falso. Medir si combinaciones no observadas aparecen como estados menos estables.

## Métricas

- energía;
- score;
- estabilidad;
- margen;
- distancia al manifold aprendido;
- abstención.

Esta prueba es más informativa que accuracy binaria.

---

# 8. E16 — aprendizaje de la dinámica del campo

Este es uno de los experimentos más importantes.

## Pregunta

¿El campo aprende una **ley de evolución**, en lugar de almacenar estados?

## Modelo

    Ψ(t+1) = Dφ(Ψ(t), contexto)

y no simplemente:

    Ψ(x) = embedding(x)

## Entrenamiento

    A → B
    B → C
    C → D

El campo recibe estados y aprende la transformación.

## Test

Consultar:

    A → ?

y realizar rollout:

    A → B → C → D

También probar saltos de dos o más pasos.

## Control crítico

Comparar contra:

1. tabla A→B;
2. embedding nearest-neighbor;
3. RQM directo;
4. RQM con composición;
5. campo estático;
6. campo dinámico entrenable.

El resultado fuerte es que D produzca estados no almacenados explícitamente como respuestas directas.

---

# 9. E17 — predicción de estados nunca observados

## Objetivo

Llevar E16 a un régimen donde el estado final no aparezca durante entrenamiento.

## Dataset geométrico

Entrenar:

    A → B
    B → C

pero no:

    A → C

Consultar:

    A → C

### Variante 2D

Aprender separadamente:

- traslación;
- rotación;
- reflexión.

### Variante composición

Aprender:

    T

y:

    R

pero no:

    R(T(x))

Probar la composición.

## OOD real

El estado objetivo debe estar ausente de:

- train;
- memoria directa;
- RQM;
- attractor bank;
- CDT.

Si el target aparece en cualquiera de ellos, el caso queda invalidado como unseen.

## Métricas

- distancia al estado objetivo;
- energía;
- top-1;
- top-2;
- margen;
- abstención;
- error de rollout;
- número de pasos;
- estabilidad;
- sensibilidad al ruido.

## Resultado fuerte

El sistema produce un estado cercano al objetivo aunque ese estado concreto nunca haya sido almacenado.

---

# 10. Mejora principal: separar almacenamiento de dinámica

El problema más importante detectado en E8–E10 es que RQM puede explicar una parte importante de E9/E10.

Introducir tres modos explícitos:

    STATIC_FIELD
    DYNAMIC_FIELD
    FIELD_PLUS_CDT

### STATIC_FIELD

    h → Ψ

### DYNAMIC_FIELD

    h → Ψt → Dφ → Ψt+1

### FIELD_PLUS_CDT

    h → Ψt → Dφ → CDT

La comparación debe usar el mismo dataset, semillas y particiones.

---

# 11. Mejora: prohibir fugas de información

Cada experimento debe tener una auditoría de contaminación.

Registrar:

    target_in_training
    target_in_rqm
    target_in_cdt
    target_in_attractor_bank
    target_in_decoder_memory
    target_seen_as_exact_vector

Si cualquiera es verdadero para un target declarado unseen, el caso queda invalidado.

Esto es especialmente importante para E16/E17.

---

# 12. Mejora: baseline de identidad del LLM

El campo podría parecer aprender cuando simplemente copia estructura ya presente en Gemma.

Cada benchmark debe tener:

### Baseline LLM

    h_LLM → classifier / nearest-neighbor

### Baseline campo

    h_LLM → random projector → field

### Campo entrenado

    h_LLM → FieldEncoderθ → field

### Campo dinámico

    h_LLM → FieldEncoderθ → Dφ → field'

La ganancia debe atribuirse al componente que cambia.

---

# 13. Mejora: el campo debe poder sobrevivir sin el LLM

Esta es una prueba especialmente importante.

### Entrenamiento

    LLM → FieldEncoder → Ψ

### Guardar

    field_checkpoint

### Reiniciar

Cargar únicamente:

- topología;
- pesos del campo;
- atractores;
- parámetros dinámicos;
- memoria CDT.

### Consulta

Usar una nueva periferia lingüística para codificar el concepto.

Si el conocimiento desaparece al cambiar el LLM, el campo no es realmente independiente de la periferia.

---

# 14. Mejora: decoder independiente

Para evitar que el decoder memorice respuestas:

- separar banco lingüístico de memoria cognitiva;
- el campo devuelve un estado abstracto;
- el decoder recibe únicamente ese estado;
- registrar si el texto producido estaba explícitamente en el dataset.

Prueba:

    perro → Ψ_perro

Cambiar el decoder y comprobar que el concepto persiste aunque cambie la superficie lingüística.

---

# 15. Mejora: dinámica como paisaje, no como tabla

El objetivo de E16/E17 debe ser aprender parámetros del paisaje:

    Fφ(Ψ)

y no una función lookup.

La dinámica puede expresarse como:

    Ψt+1 = Ψt - η∇Fφ(Ψt) + ξ

o mediante una formulación fasorial equivalente.

Registrar:

- energía inicial;
- energía por paso;
- energía final;
- número de mínimos;
- basin de atracción;
- distancia entre atractores;
- estabilidad ante ruido.

La hipótesis científica cambia de:

> "el campo guarda vectores"

a:

> "el aprendizaje modifica el paisaje dinámico que transforma y recupera estados".

---

# 16. Mejora: pruebas de causalidad

No basta observar que después del entrenamiento funciona.

Cada experimento debe tener:

### Pre
Snapshot del campo.

### Intervención
Entrenamiento de un concepto o relación.

### Post
Misma consulta, mismas semillas y mismo solver.

### Ablación
Eliminar la modificación concreta.

### Recuperación
Comprobar si desaparece el efecto.

Ejemplo:

    train A→B
    ↓
    A→? = B

    rollback del campo
    ↓
    A→? ≠ B

Esto permite demostrar que el aprendizaje está en el sustrato.

---

# 17. Mejora: multisemilla y tamaños crecientes

E8–E10 usan tres semillas. Para E11–E17:

- mínimo: 8 semillas;
- medio: 16 semillas;
- escalado: N = 8, 16, 32 y 64 conceptos.

Registrar media, desviación estándar, p50/p95 y distribución completa.

No aceptar una sola corrida como evidencia.

---

# 18. Métricas comunes obligatorias

Cada experimento debe producir JSON/CSV con:

    experiment
    commit
    seed
    hardware
    rust_version

    n_concepts
    train_examples
    unseen_examples
    ood_examples

    encoder_type
    encoder_frozen
    field_trainable
    dynamics_trainable
    cdt_enabled
    rqm_enabled
    decoder_type

    accuracy_seen
    accuracy_unseen
    accuracy_ood

    energy_initial
    energy_final
    energy_delta

    top1
    top2
    margin
    abstentions

    field_steps
    rollout_steps
    convergence_steps

    mean_us
    p50_us
    p95_us
    p99_us

    target_in_training
    target_in_rqm
    target_in_cdt
    target_in_attractor_bank

    forgetting_A
    forgetting_B
    forgetting_C

    checkpoint_hash
    field_hash
    llm_hash

---

# 19. Orden recomendado de implementación

No implementar E11–E17 simultáneamente.

## Fase 1 — E11

Construir TrainableFieldEncoder y demostrar que el campo aprende una geometría mejor que un projector fijo.

## Fase 2 — E12

Conectar **Gemma GGUF real** y repetir E8.

Este experimento es obligatorio antes de afirmar invariancia lingüística.

## Fase 3 — E13

Introducir relaciones sin RQM como explicación principal.

## Fase 4 — E14

Aprendizaje incremental + forgetting curve.

## Fase 5 — E15

Composición semántica sin etiquetas explícitas.

## Fase 6 — E16

Entrenar la dinámica:

    Dφ(Ψt) → Ψt+1

## Fase 7 — E17

Extrapolación a estados nunca observados.

---

# 20. Criterios de interpretación

## Resultado débil

El campo mejora clasificación, pero:

- necesita un banco de conceptos;
- copia embeddings;
- RQM contiene los targets;
- el LLM ya resolvía la tarea.

Interpretación: **mejora de representación**, no dinámica cognitiva.

## Resultado intermedio

El campo aprende conceptos y relaciones, conserva memoria y generaliza a combinaciones no vistas.

Interpretación: **sustrato de representación + memoria generalizable**.

## Resultado fuerte

Además:

- aprende dinámica;
- predice estados no almacenados;
- sobrevive al cambio de LLM;
- no requiere RQM para generar el estado;
- mantiene el conocimiento después de reiniciar;
- supera controles estáticos;
- la intervención sobre parámetros del campo cambia causalmente el comportamiento.

Interpretación: evidencia de un **sustrato dinámico de aprendizaje independiente de la periferia lingüística**.

No debe denominarse automáticamente consciencia, AGI o vida.

---

# 21. Arquitectura final propuesta

    ┌─────────────────────┐
    │ LLM congelado       │
    │ lenguaje/contexto   │
    └──────────┬──────────┘
               │ hidden state
               ▼
    ┌─────────────────────┐
    │ FieldEncoder θ      │
    │ entrenable          │
    └──────────┬──────────┘
               │
               ▼
         ┌──────────────┐
         │ Campo Ψ      │
         │              │
         │ representación
         │ memoria      │
         │ dinámica     │
         └──────┬───────┘
                │
        ┌───────┴────────┐
        ▼                ▼
   Dynamic Field     Thermo CDT
   predicción        consolidación
        │                │
        └───────┬────────┘
                ▼
         estado abstracto
                │
                ▼
        decoder lingüístico
                │
                ▼
               texto

La propiedad que E11–E17 intentan validar es:

    LLM = interfaz lingüística
    Campo = aprendizaje + dinámica + memoria

pero el protocolo debe estar diseñado para poder **refutar** esa separación.

---

# 22. Pregunta decisiva del ciclo

La pregunta final de E11–E17 no es:

> "¿el campo tiene 100 % de accuracy?"

La pregunta correcta es:

> **¿Qué parte del comportamiento aprendido permanece cuando eliminamos progresivamente el LLM, RQM, tablas, attractor bank y memoria directa, dejando únicamente la dinámica entrenada del campo?**

Ese es el experimento decisivo para la arquitectura propuesta.

## Regla de publicación

- No mezclar resultados E8–E10 con E11–E17.
- Mantener E8–E10 como baseline.
- Reportar resultados negativos.
- No usar "cognición" como etiqueta experimental primaria.
- Toda generalización debe tener un conjunto unseen/OOD explícito.
- Todo estado declarado nuevo debe pasar auditoría anti-leakage.
- Toda ventaja debe compararse contra el mismo hidden state del LLM y contra un baseline no dinámico.
