# Token-Free Liquid Field — nueva línea experimental

## Propósito

Esta rama investiga una hipótesis deliberadamente separada de la línea `field-autonomy-v4`:

> Después de una etapa de codificación semántica, el razonamiento/inferencia puede realizarse sobre un estado de campo compacto y continuo, sin volver a consultar tokens, logits ni el LLM durante el hot path.

La rama **no modifica `main` ni la versión 4**. Su objetivo es producir evidencia independiente que en el futuro pueda enfrentarse experimentalmente contra `main` + v4.

## Rama

`exp/token-free-liquid-field`

Base inicial: `main`.

## Hipótesis central

Arquitectura propuesta:

`texto → periférico lingüístico → FieldEncoder → Ψ₀ → Liquid Core → Ψ₁ → Ψ₂ → ... → decoder`

Durante inferencia:

- el LLM/periférico no vuelve a intervenir;
- no entran token IDs al Liquid Core;
- no se generan probabilidades por token para explorar futuros;
- el núcleo opera sobre Ψ y contexto de campo;
- la decodificación ocurre al final.

Importante: esto **no** significa que el LLM deje de procesar tokens. Si se usa un LLM como encoder, los tokens siguen entrando al LLM. Lo que desaparece del núcleo cognitivo es el acceso a tokens después de producir Ψ₀.

## Base teórica

### 1. Estado latente como interfaz

El FieldEncoder debe transformar una representación lingüística de alta dimensión en un estado `Ψ ∈ R^N` de dimensión pequeña o moderada. El objetivo no es conservar información lexical completa, sino conservar las variables necesarias para la dinámica.

`Ψ₀ = E_φ(h_LLM(x))`

Donde `h_LLM` es una activación congelada y `E_φ` aprende una representación de campo.

### 2. Dinámica líquida

El núcleo se modela como una dinámica recurrente:

`Ψ(t+1) = Ψ(t) + Δt · F_θ(Ψ(t), c(t))`

Una variante continua puede expresarse como:

`dΨ/dt = F_θ(Ψ,c,t)`

El carácter líquido debe significar que la dinámica puede adaptar su evolución al estado, contexto o incertidumbre, no simplemente que se ejecuta un MLP repetido.

Liquid Time-Constant Networks son una referencia teórica relevante porque utilizan dinámicas continuas con constantes de tiempo dependientes del estado. Mamba/SSM es una referencia de ingeniería para estudiar estados recurrentes selectivos y procesamiento eficiente, aunque no debe asumirse que su arquitectura sea automáticamente la mejor para este proyecto.

### 3. Predicción de estados, no de tokens

El objetivo de cada paso es predecir el siguiente estado de campo:

`Ψ_t → Ψ̂_{t+1}`

La probabilidad lingüística solo aparece cuando el sistema necesita producir una respuesta externa. Esto permite estudiar si una gran parte del cálculo actualmente asociado a generación autoregresiva puede reemplazarse por evolución en un espacio latente compacto.

### 4. Dinámica dispersa

Una hipótesis adicional es que no todos los nodos deben actualizarse en cada paso:

`A_t = TopK(|Ψ_t|, k)`

`Ψ_{t+1} = Ψ_t + M(A_t) ⊙ F_θ(Ψ_t)`

La hipótesis debe probarse contra el modelo denso. La sparsidad no debe introducirse solo para obtener velocidad; debe demostrar que mantiene calidad y estabilidad.

### 5. Memoria asociativa como componente opcional

Hopfield moderno y memorias asociativas densas muestran que los estados distribuidos pueden almacenar patrones de forma asociativa y que la capacidad puede crecer fuertemente con la dimensión. Esta línea puede inspirar una memoria de campo, pero el benchmark debe distinguir memoria asociativa de simple nearest-neighbor/lookup.

### 6. El LLM como periférico, no como motor cognitivo

La tesis fuerte que se desea evaluar es:

`lenguaje → representación → dinámica → representación → lenguaje`

No:

`lenguaje → Transformer → token siguiente → Transformer → token siguiente...`

La comparación debe ser empírica. No se presupone que el campo sea superior al LLM.

## Qué NO se afirma todavía

Esta rama no demuestra por sí misma:

- conciencia;
- vida artificial;
- inteligencia general;
- causalidad física del campo;
- superioridad frente a Transformers;
- que los tokens sean intrínsecamente innecesarios;
- que un campo compacto pueda representar todo el conocimiento lingüístico.

Solo investiga si una representación de campo puede convertirse en el dominio principal de inferencia después de una codificación inicial.

## Principio experimental

Cada experimento debe poder responder:

1. ¿Qué información entra al núcleo?
2. ¿Qué información está prohibida durante el hot path?
3. ¿Cuántos estados actualiza?
4. ¿Cuántos pasos ejecuta?
5. ¿Qué precisión conserva?
6. ¿Cuánto cuesta en µs/query?
7. ¿Qué ocurre fuera de distribución?
8. ¿Qué información lingüística permanece en Ψ?
9. ¿Puede cambiarse el LLM encoder sin destruir la dinámica?
10. ¿La velocidad procede de una representación compacta o de memorizar respuestas?

## Familias experimentales

- E45: bottleneck token-free.
- E46: dinámica líquida vs MLP recurrente vs SSM.
- E47: inferencia dispersa.
- E48: inferencia adaptativa y early stopping.
- E49: predicción multiescala/coarse-to-fine.
- E50: trayectorias múltiples / beam geométrico.
- E51: memoria asociativa de campo.
- E52: consolidación online en el campo.
- E53: separación inferencia/consolidación.
- E54: eliminación completa del LLM después de Ψ₀.
- E55: sustitución del LLM encoder.
- E56: escalabilidad del campo.
- E57: benchmark integral contra v4.

La numeración continúa desde v4 para evitar confundir resultados.

## Criterios generales de aceptación

Un experimento puede ser `PASS` únicamente si:

- tiene dataset y manifest reproducibles;
- tiene train/dev/test separados;
- registra provenance;
- no consulta información prohibida;
- tiene al menos un control adecuado;
- reporta latencia p50/p95/p99;
- reporta precisión y OOD;
- reporta pasos de dinámica y actividad del campo;
- no usa el decoder como tabla de respuestas;
- no reinyecta el target durante rollout;
- permite repetir el resultado con semillas independientes.

Un resultado de velocidad sin control de exactitud será `PARTIAL`, no `PASS`.

## Descubrimientos que cambiarían la hipótesis

### Descubrimiento A — bottleneck suficiente
Si un campo de 32–128 dimensiones conserva las propiedades semánticas necesarias y permite inferencia sin pérdida relevante, se valida el bottleneck como interfaz cognitiva candidata.

### Descubrimiento B — dinámica aprende reglas
Si `D_θ` compone transformaciones y generaliza a estados nunca observados sin RQM/tabla/NN/attractor lookup, se fortalece la hipótesis de que la dinámica es portadora de conocimiento operacional.

### Descubrimiento C — sparse field
Si actualizar solo una fracción de nodos mantiene calidad y reduce significativamente el coste, la sparsidad se convierte en una vía principal de escalabilidad.

### Descubrimiento D — inferencia adaptativa
Si la mayoría de consultas converge en pocos pasos y las difíciles reciben más cómputo, puede sustituirse el número fijo de iteraciones por un presupuesto dinámico basado en estabilidad/incertidumbre.

### Descubrimiento E — consolidación plástica
Si experiencias pueden modificar `D_θ` o un estado plástico auxiliar y, tras borrar episodios y memoria explícita, el sistema conserva una mejora causal sobre el control sin experiencia, aparece evidencia del tipo:

`experiencia → consolidación → modificación persistente → nueva inferencia`

Esto conecta esta rama con la pregunta central de v4, pero debe mantenerse experimentalmente separado.

## Comparación futura con v4

La comparación final deberá ser posterior a la finalización de v4 y utilizar un dataset sellado común.

Comparar como mínimo:

- v4 Dynamic Field;
- Token-Free Liquid Field;
- Token-Free Liquid + sparse;
- Token-Free Liquid + consolidación;
- baseline LLM;
- baseline MLP recurrente;
- baseline SSM.

Métricas:

`accuracy, OOD, composition, long-horizon stability, µs/query, p95, active_nodes, rollout_steps, memory_bytes, training_cost, consolidation_cost, persistence, transfer`.

La pregunta final no será simplemente “¿cuál tiene mayor accuracy?”, sino:

> ¿Qué arquitectura obtiene mejor relación entre capacidad de generalización, persistencia del aprendizaje y coste computacional, sin esconder memoria de respuestas dentro del decoder o de mecanismos de lookup?
