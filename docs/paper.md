# Arquitectura Neuro-Geométrica Multi-Agente

## Complejos Simpliciales Guiados por Energía Libre con Renderizado Lingüístico Periférico

### Resumen

Los modelos de lenguaje de gran escala (LLMs) han demostrado una capacidad notable para interpolar patrones lingüísticos, pero su arquitectura dominante mezcla en una misma tubería tres funciones que en sistemas biológicos suelen estar separadas: percepción, estabilización conceptual y expresión simbólica. Esta mezcla obliga a resolver razonamiento abstracto, coherencia física, memoria episódica y sintaxis mediante álgebra lineal densa, atención cuadrática y retropropagación global. El resultado es un régimen de cómputo intensivo con altos costos energéticos, latencia elevada y fragilidad semántica ante tareas que exigen anclaje espacial o causal.

Este documento propone el **Sistema Neuro-Geométrico de Agentes (SNGA)**, una arquitectura híbrida experimental en la que la cognición abstracta se modela como relajación mecánica de una malla topológica descentralizada, mientras que los LLMs permanecen como interfaces lingüísticas periféricas. La tesis central no es reemplazar los LLMs, sino desacoplar el lenguaje del núcleo de memoria e inferencia: SNGA almacena y evoca estados conceptuales mediante complejos simpliciales esparsos; los LLMs traducen entre lenguaje humano y activaciones geométricas internas. El núcleo cognitivo no es un vector denso, sino un complejo simplicial formado por agentes binarios, aristas asíncronas y símplices de orden superior. Cada agente minimiza una energía libre local derivada de la tensión geométrica con sus vecinos.

La tesis se presenta como una hipótesis de arquitectura, no como una demostración de AGI. El repositorio acompaña la propuesta con un prototipo íntegro en Rust. La implementación actual incluye una red binaria event-driven, una malla simplicial 2D/3D, reglas de relajación elástica local, memoria episódica, atención dinámica, predicción causal, planificación de rutas, un demostrador multimodal sintético y un motor gráfico basado en `macroquad`.

## 1. Introducción

La inteligencia artificial contemporánea suele tratar el lenguaje como el medio universal del pensamiento. En los LLMs, el razonamiento aparece como una trayectoria dentro de un espacio latente de alta dimensión entrenado para predicción de tokens. Este enfoque ha escalado con éxito, pero introduce una dependencia fuerte en multiplicaciones matriciales masivas, memoria de activaciones, sincronización global y optimización por retropropagación. En términos energéticos, la red paga por activar una gran fracción de sus parámetros incluso cuando el problema requiere solo una pequeña región conceptual.

SNGA parte de una hipótesis distinta: el lenguaje no es el sustrato primario de la cognición, sino una interfaz periférica. La representación abstracta se define como una geometría dinámica, semejante a un mapa conceptual. El razonamiento no consiste necesariamente en recorrer tokens, sino en deformar y estabilizar una estructura espacial sometida a restricciones locales. Bajo esta hipótesis, el LLM conserva un papel fundamental: actúa como codificador y decodificador lingüístico, pero no como único depósito de memoria conceptual.

La inspiración neurobiológica procede de la separación funcional entre sistemas de mapeo espacial/conceptual, como la corteza entorrinal y las células de rejilla, y sistemas lingüísticos especializados, como las áreas de Broca y Wernicke. En esta analogía, el núcleo SNGA opera como un tejido de navegación conceptual, mientras que el LLM actúa como traductor entre texto humano y estados de la malla. La arquitectura resultante es explícitamente híbrida:

```text
entrada humana / sensorial
        |
        v
LLM o encoder periférico
        |
        v
spikes binarios multimodales
        |
        v
SNGA: memoria e inferencia geométrica esparsa
        |
        v
estado conceptual estabilizado
        |
        v
adaptador / LLM decodificador
        |
        v
salida lingüística, visual o motora
```

## 2. Marco Teórico

### 2.1 Complejos Simpliciales

Un grafo clásico representa relaciones binarias mediante vértices y aristas. Un complejo simplicial extiende esta idea incorporando relaciones de orden superior:

- 0-símplices: vértices o agentes.
- 1-símplices: aristas entre pares de agentes.
- 2-símplices: triángulos que codifican coherencia entre ternas.
- 3-símplices: tetraedros para restricciones volumétricas en implementaciones 3D.

En SNGA, un concepto no se almacena como un vector denso fijo. Se aproxima como una región estable dentro de un complejo:

```text
G = (V, E, S)
```

donde `V` es el conjunto de agentes binarios, `E` el conjunto de canales asíncronos y `S` el conjunto de símplices que preservan estructura de orden superior. En el prototipo Rust, la visualización principal usa triángulos 2D, pero el núcleo ya permite experimentar con tetraedros y profundidad 3D.

### 2.2 Energía Libre Local

El Principio de Energía Libre de Friston puede interpretarse computacionalmente como una tendencia a reducir sorpresa variacional. SNGA traduce esta idea a geometría: una entrada produce sorpresa cuando deforma la malla de forma incompatible con sus longitudes y áreas esperadas.

Para una arista entre agentes `i` y `j`, se define una energía elástica:

```text
F_ij = w_ij (||x_i - x_j|| - l_ij)^2
```

donde `x_i` y `x_j` son posiciones, `l_ij` es la longitud de reposo y `w_ij` la rigidez conceptual de la relación. Para un triángulo `s = (i, j, k)`, se añade una penalización de área:

```text
F_s = beta (A(x_i, x_j, x_k) - A_s^0)^2
```

La energía libre total aproximada del sistema es:

```text
F = sum(F_ij) + sum(F_s)
```

El aprendizaje no actualiza pesos por retropropagación global. Cada agente aplica una regla local:

```text
Delta x_i = -alpha * grad_x_i F_i
```

En el prototipo, el gradiente se implementa como fuerzas de resorte sobre aristas y fuerzas de preservación de área sobre triángulos. La activación binaria aumenta temporalmente la rigidez local, simulando atención event-driven sin atención densa.

### 2.3 Codificación Esparsa Predictiva

Los agentes tienen un estado binario de activación y un escalar de sorpresa. Una señal lingüística periférica se convierte en ráfagas discretas que activan vértices específicos. La propagación ocurre por colas de `Spike`, no por pasos matriciales densos. Esto aproxima una red de picos:

```text
spike = { source, target, ttl }
```

Cada pico posee un tiempo de vida finito. Si un agente supera un umbral de sorpresa, propaga nuevos picos hacia vecinos. La computación se concentra en la región activa de la malla.

## 3. Arquitectura SNGA

### 3.1 Núcleo de Simulación

El núcleo implementado en Rust se organiza en capas separadas:

- `geometry.rs`: álgebra vectorial mínima para posiciones, distancias y fuerzas.
- `mesh_engine.rs`: motor matemático/topológico que genera la malla 2D/3D, aristas, triángulos y tetraedros.
- `simplicial.rs`: capa neuronal que consume la topología del motor y añade agentes, spikes, memoria, aprendizaje, oscilaciones y planificación.
- `render.rs`: motor gráfico 2D para visualizar la red y sus métricas.

La estructura principal es `SimplicialNetwork`. Contiene:

- `agents`: vértices binarios con posición, velocidad, activación y sorpresa.
- `edges`: restricciones elásticas de distancia.
- `simplices`: triángulos con área objetivo.
- `tetrahedra`: símplices 3D con volumen objetivo.
- `semantic_cells`: celdas asociativas de orden superior que agrupan patrones compuestos.
- `focus_edges`: rutas de representantes conceptuales que promueven nodos esperados al ranking estricto.
- `spikes`: cola asíncrona de eventos.
- `config`: parámetros físicos y topológicos.

El prototipo genera la topología mediante `SimplicialMeshEngine`. En 2D, cada celda rectangular se divide en dos triángulos, creando una malla con interacciones binarias y ternarias. En 3D, el motor añade capas de profundidad, aristas verticales y tetraedros para que la capa neuronal opere sobre un complejo volumétrico. Esta separación permite optimizar o reemplazar el motor matemático sin mezclarlo con memoria, atención u oscilaciones.

### 3.2 Ciclo de Inferencia

Cada frame ejecuta el siguiente ciclo:

```text
1. Propagar spikes activos.
2. Activar agentes destino y acumular sorpresa.
3. Calcular fuerzas locales sobre aristas.
4. Calcular correcciones por área en símplices.
5. Integrar velocidad y posición con amortiguamiento.
6. Decaer sorpresa y apagar agentes en reposo.
```

El estado final tras varios pasos no es una secuencia de tokens, sino una configuración geométrica estabilizada. Esta configuración puede interpretarse como un atractor conceptual.

### 3.3 Arquitectura Híbrida: SNGA como Núcleo y LLM como Interfaz

SNGA propone una separación funcional entre núcleo cognitivo y periferia lingüística. El objetivo no es eliminar los LLMs, sino especializarlos. En lugar de usar un LLM monolítico como memoria, razonador, simulador físico y generador textual al mismo tiempo, la arquitectura divide el sistema en cuatro etapas:

**Codificación periférica de entrada.** Un LLM o encoder externo transforma texto, imagen o sonido en impulsos discretos. En el prototipo, `MultimodalDemo` usa una proyección determinista separada por modalidad: lenguaje, visión y audio ocupan bandas distintas de la malla. Esta función no pretende sustituir a un encoder semántico real; actúa como sustituto mínimo para demostrar el mecanismo de inyección, coactivación y evocación.

**Núcleo neuro-geométrico.** La malla propaga los impulsos y minimiza energía libre local hasta alcanzar una configuración estable. Este núcleo funciona como memoria asociativa, espacio de grounding y posible motor de inferencia geométrica.

**Adaptador de lectura.** Un módulo futuro observaría distancias, curvaturas, regiones activas y caminos geodésicos para producir embeddings condicionantes. Esta capa es el puente entre un estado topológico discreto y el espacio continuo que un LLM puede consumir.

**Renderizador lingüístico de salida.** Un LLM decodificador generaría lenguaje a partir del paisaje geométrico estacionario. En el código actual, el renderizado es visual: muestra agentes activos, aristas excitadas, símplices, energía libre y una proyección simple de los agentes con mayor sorpresa.

Esta división permite que el LLM haga lo que mejor sabe hacer: interpretar y producir lenguaje. SNGA asume la tarea complementaria: almacenar asociaciones multimodales, limitar la activación a regiones relevantes y ofrecer un estado conceptual estable que pueda condicionar al LLM.

La integración periférica con un LLM pequeño se implementa como capa opcional, no como parte de la memoria del núcleo. El módulo `linguistic_engine.rs` define un adaptador para Ollama/Gemma (`gemma2:2b` por defecto) y el binario `snga_gemma_bridge` demuestra el flujo:

```text
prompt humano
  -> activación SNGA / proyección geométrica
  -> intención y resumen geométrico
  -> Gemma periférico como renderizador lingüístico
  -> respuesta en lenguaje natural
```

Si Gemma/Ollama no está disponible, el sistema usa un fallback simbólico de SNGA. Esto preserva la tesis central: el LLM no almacena la memoria conceptual; solo verbaliza el estado geométrico producido por la red.

### 3.4 Aprendizaje Multimodal Inicial

La primera demostración implementada prueba una versión mínima de grounding multimodal. El sistema define conceptos sintéticos como `manzana` y `roca`. Cada concepto tiene rasgos separados por modalidad:

```text
manzana:
  lenguaje = [manzana, fruta, dulce]
  vision   = [redonda, roja, verde, brillante]
  audio    = [crujiente, mordida]
```

Durante el entrenamiento, los patrones de lenguaje, visión y audio se inyectan simultáneamente como picos binarios. La red aplica una regla local de coactivación:

```text
si i y j se activan juntos:
  aumentar rigidez w_ij
  reducir ligeramente la longitud de reposo l_ij
```

Esto no crea semántica humana completa. Lo que demuestra es el mecanismo básico: estímulos de modalidades distintas pueden quedar ligados en una vecindad topológica compartida. Después del entrenamiento, activar solo la entrada lingüística de `manzana` tiende a reactivar parte de la región multimodal asociada.

### 3.5 Inhibición Lateral y Control de Cascadas

Una red de picos sin inhibición tiende a activar demasiadas regiones, análogo a una crisis epiléptica computacional. Para evitarlo, el prototipo introduce tres compuertas:

```text
1. Propagación solo por aristas asociativas aprendidas.
2. Presupuesto máximo de spikes por paso.
3. Inhibición lateral top-k: solo sobreviven los N agentes con mayor sorpresa.
```

La inhibición no elimina la memoria; limita la difusión global. Esto permite que una evocación active su vecindad conceptual sin contaminar toda la malla.

### 3.6 Plasticidad, Ritmos, Replay y Causalidad

La versión actual del núcleo añade mecanismos biomiméticos adicionales:

- **Crecimiento estructural:** si dos agentes se coactivan y no existe arista asociativa, la red crea una nueva conexión.
- **Consolidación:** conexiones reforzadas repetidamente se marcan como consolidadas y olvidan más lento.
- **Olvido y poda:** aristas no consolidadas pierden peso con el tiempo y pueden quedar inactivas.
- **Poda áurea por utilidad:** refuerzos opcionales pueden exigirse solo cuando la utilidad supera `1/phi ~= 0.618`, evitando consolidar asociaciones de baja calidad.
- **Inhibición local:** además del presupuesto global top-k, existe inhibición por vecindad geométrica base para evitar hiperactivación local.
- **Ritmos temporales:** el umbral de activación puede oscilar periódicamente para simular ventanas de excitabilidad.
- **Memoria episódica y replay:** patrones recientes se almacenan como episodios y pueden reinyectarse durante fases de replay para reforzar trazas.
- **Causalidad predictiva:** el sistema aprende transiciones dirigidas `causa -> efecto` y puede predecir agentes esperados desde un patrón causa.
- **Celdas semánticas de orden superior:** patrones coactivados se consolidan en `SemanticCell`, una celda poligonal/hiperrelacional con vértices, aristas, peso, edad y payload compacto.
- **Aprendizaje por error predictivo:** cuando una predicción no contiene agentes esperados, la red refuerza transiciones correctivas, crea celdas compuestas y registra representantes de foco.
- **Representantes de foco:** rutas `focus` persistentes seleccionan nodos esperados dentro de una región amplia, permitiendo que el conocimiento pase de recall difuso a top-k estricto.

Estos mecanismos no sustituyen todavía a encoders reales de visión/audio/texto. Esos módulos se mantienen explícitamente como periferia futura. El objetivo actual es fortalecer el núcleo SNGA para que pueda recibir dichos encoders cuando estén disponibles.

### 3.7 Operadores de Razonamiento Topológico

Para pasar de asociación a razonamiento, el núcleo incorpora operadores que actúan sobre la malla sin recurrir a multiplicación matricial densa:

- **Implicación causal dirigida:** una relación `A -> B` se almacena como transición orientada entre agentes.
- **Inferencia transitiva:** cadenas `A -> B -> C` pueden consultarse como predicción `A -> C`, aunque el atajo no haya sido entrenado.
- **Contradicción energética:** relaciones incompatibles aumentan la energía libre cuando se coactivan.
- **Selección por inhibición:** rutas competidoras se limitan por presupuestos de activación y spikes.
- **Optimización por flujo/evaporación:** rutas candidatas compiten; las rutas predictivas reciben depósito de conductancia y las rutas débiles se evaporan.

En este marco, la lógica no aparece como reglas simbólicas externas, sino como dinámica de rutas, tensiones y estabilización topológica.

La optimización de rutas se inspira en sistemas tipo *Physarum*: primero se permite una nube de caminos posibles y luego se refuerzan únicamente los caminos que llegan a estados esperados con menor costo. Las conexiones no usadas o menos predictivas pierden conductancia. Esto transforma una inferencia difusa con alto recall pero baja precisión en una ruta preferente de menor energía.

### 3.8 Geometría 3D, Hiperbólica y Símplex de Orden Superior

El prototipo conserva renderizado 2D para visualización, pero el núcleo ya soporta coordenada de profundidad, distancia 3D opcional, curvatura hiperbólica aproximada y símplices tetraédricos (`Simplex3`). Esto permite experimentar con volúmenes conceptuales y no solo con superficies triangulares.

La distancia entre agentes puede operar en modo euclidiano 3D o aplicar una deformación hiperbólica controlada por `hyperbolic_curvature`. Esta extensión es relevante para jerarquías conceptuales, donde la geometría hiperbólica suele representar árboles y taxonomías con menor distorsión que un plano euclidiano.

### 3.9 Oscilaciones Funcionales y Modos Globales

La versión actual incorpora una capa opcional de oscilaciones funcionales inspiradas en bandas neurofisiológicas. No simula campos físicos; modela el papel computacional de los ritmos como moduladores globales y regionales de la malla:

```text
Delta -> replay y consolidación lenta
Theta -> memoria episódica y secuencias
Alpha -> inhibición y filtrado
Beta  -> mantenimiento de objetivo/plan
Gamma -> propagación local rápida
```

La red puede operar en tres modos globales:

```text
Exploration  = mayor excitabilidad y búsqueda
Focus        = estabilización de objetivo y rutas activas
SleepReplay  = replay episódico y consolidación sin entrada externa
```

La malla se divide internamente en regiones ligeras de agentes. Cada región adopta dinámicamente una banda dominante según su sorpresa, actividad y relación con el objetivo atencional. Esto permite coordinación global sin conectar todos los agentes con todos: las regiones no intercambian un campo físico, sino que ajustan umbrales, replay, propagación e inhibición según fase.

En el motor, las oscilaciones modulan:

- Umbral efectivo de activación.
- Peso de propagación de spikes.
- Fuerza del replay episódico.
- Prioridad de regiones alineadas con objetivo.
- Inhibición de regiones no relevantes.

Esta capa está desactivada por defecto para conservar compatibilidad con los experimentos base y se activa explícitamente con `enable_neural_oscillations()`.

### 3.10 Celdas Semánticas, Detectores Locales y Foco de Atractor

La actualización más reciente añade una capa explícita para pasar de asociaciones por pares a conocimiento compuesto. La red ya no aprende solo aristas `i-j` o transiciones causales `A -> B`; también puede consolidar regiones de orden superior:

```text
input lingüístico
  -> detectores locales de rasgo/intención/frame
  -> SemanticCell compuesta
  -> concepto / control / plan
  -> representantes de foco para ranking top-k
```

Las `SemanticCell` funcionan como caras/celdas asociativas dinámicas. No son matrices ni tensores densos: son listas de vértices y aristas con índices inversos `agent_to_cells`. Si una consulta toca varios vértices de una celda, la celda resuena y distribuye score al resto de sus vértices. Esto modela una idea cercana a detectores V1/V2: rasgos locales simples se combinan en una configuración compuesta estable.

El adaptador semántico-ejecutivo entrena detectores pequeños y reutilizables:

```text
feature_intent   -> intención lingüística
feature_control  -> tarea de control semántico
feature_frame    -> marco de respuesta / memoria de trabajo
feature_keyword  -> rasgos léxicos relevantes
```

Estos detectores conectan la entrada textual con conceptos internos, control ejecutivo y planificador. Cuando la evaluación predice una región amplia correcta pero no logra incluir el concepto esperado en el top-k estricto, `learn_from_prediction_error` crea rutas correctivas. La corrección tiene tres efectos:

1. Refuerza transiciones dirigidas hacia los agentes esperados que faltaron.
2. Crea o fortalece una `SemanticCell` que une entrada y objetivo.
3. Registra `focus_edges` persistentes desde representantes de entrada hacia representantes del objetivo.

En inferencia, los `focus_edges` no sustituyen a la dinámica causal. Se aplican como una etapa final de promoción solo en inferencia transitiva/exact-hop, no en `predict_next_pattern`. Esta restricción fue importante: una versión más agresiva contaminaba la verificación de salida. La versión estable promueve como máximo `512` representantes por consulta y conserva intacta la ruta de verificación.

## 4. Complejidad y Eficiencia

En atención densa, la interacción entre tokens escala como:

```text
O(N^2)
```

En SNGA, el costo dominante por paso depende de aristas y símplices activos o evaluados:

```text
O(E + S)
```

En hardware neuromórfico o FPGA, esta complejidad puede volverse event-driven real:

```text
O(E_activos + S_activos)
```

La diferencia arquitectónica es importante. Un transformer procesa capas completas incluso cuando solo una parte de la información es relevante. SNGA permite reposo nulo: agentes no excitados pueden permanecer sin cómputo hasta recibir un pico local.

### 4.1 Comparación Teórica Frente a Transformers

SNGA no debe interpretarse como "un transformer sin matrices". La diferencia central es más profunda: en un transformer, el lenguaje suele operar como sustrato principal del cómputo; en SNGA, el lenguaje es una interfaz periférica que activa y lee un núcleo geométrico. La ruta conceptual es:

```text
entrada lingüística
  -> intención abstracta
  -> rutas geométricas activas
  -> minimización de energía / contradicción / causalidad
  -> estado conceptual estabilizado
  -> salida lingüística
```

En un transformer, buena parte del razonamiento queda distribuida en operaciones densas de atención y MLP sobre tokens. En SNGA, la hipótesis es que el entendimiento emerge de relaciones topológicas: rutas causales, jerarquías, tensión por contradicción, replay, inhibición y selección de caminos de baja energía. Por tanto, al escalar la red, el costo de inferencia no debería depender del número total de nodos, sino del subgrafo activo necesario para resolver la tarea.

Esta diferencia permite formular una ventaja potencial:

```text
Transformer: costo asociado al procesamiento denso de secuencias y capas.
SNGA: costo asociado a rutas activas, spikes y regiones geométricas relevantes.
```

Los experimentos actuales no demuestran superioridad general frente a LLMs. Sí muestran que SNGA puede resolver memoria asociativa, inferencia transitiva, contradicción energética, selección de rutas e intención lingüística de dominio pequeño sin recurrir a multiplicación matricial densa. La tesis fuerte es que, con mayor escala y mejores periféricos sensoriales, el núcleo SNGA podría ofrecer una forma más eficiente de grounding y razonamiento, mientras el lenguaje permanece como mecanismo de comunicación y no como centro del pensamiento.

## 5. Implementación Rust

El repositorio incluye una versión inicial funcional:

```text
cargo run
```

El motor abre una ventana con una malla triangulada. Los colores codifican estado:

- Nodos claros: agentes en reposo.
- Nodos naranjas: agentes activados.
- Aristas azules: canales con actividad local.
- Triángulos tenues: símplices de coherencia.

Controles:

- `Espacio`: pausar o reanudar.
- `Click izquierdo`: inyectar estímulo en el agente más cercano.
- `M`: entrenar coactivaciones multimodales sintéticas.
- `L`: evocar `manzana` desde su patrón lingüístico.
- `O`: evocar `roca` desde su patrón lingüístico.
- `T`: inyectar patrón textual de ejemplo.
- `R`: reiniciar la malla.
- `+` / `-`: zoom.
- Flechas: mover cámara.

## 6. Resultados Iniciales de Validación

Se añadió un experimento sin ventana (`cargo run --bin experiment`) para medir si la malla aprende asociaciones multimodales. La prueba usa 8 conceptos sintéticos (`manzana`, `roca`, `lluvia`, `fuego`, `perro`, `tambor`, `cafe`, `bicicleta`) con rasgos separados de lenguaje, visión y audio.

El protocolo evita medir activación residual:

```text
1. Inicializar una malla limpia.
2. Medir evocación sin entrenamiento.
3. Entrenar por coactivación multimodal durante 6 épocas.
4. Limpiar toda actividad dinámica.
5. Inyectar solo el patrón lingüístico.
6. Medir recuperación de rasgos sensoriales y fuga a distractores.
```

Resultados de referencia:

```text
antes:
  recall_medio    = 0.0%
  precision_media = 0.0%
  fuga_media      = 0.0%

despues:
  recall_medio    = 100.0%
  precision_media = 68.2%
  fuga_media      = 10.9%
```

Estos resultados muestran que la red puede funcionar como memoria asociativa topológica inicial: una entrada lingüística reactiva rasgos sensoriales aprendidos mediante coactivación. Sin embargo, la precisión imperfecta y la fuga residual demuestran que todavía no hay razonamiento general ni grounding robusto. El siguiente reto es introducir inhibición competitiva, control causal y separación geométrica de conceptos cercanos.

Se añadió además `large_experiment`, una validación sintética con 10,000 conceptos, 180,000 agentes y control inhibitorio estricto:

```text
conceptos              = 10000
epocas                 = 3
muestras_eval          = 100
max_active_agents      = 32
max_spikes_per_step    = 128

recall_medio           = 100.0%
precision_media        = 55.1%
fuga_media             = 0.017%
activos_max_observado  = 32
```

El resultado indica que la malla puede almacenar miles de asociaciones sintéticas y evocarlas sin colapso global. La baja fuga porcentual y el límite de activación muestran que la inhibición controla la expansión. La precisión media todavía no es perfecta porque el sistema usa codificación sintética por hashing y no aprende todavía fronteras semánticas reales. Aun así, el resultado es compatible con la hipótesis de una memoria asociativa esparsa y evolutiva.

Estos datos no permiten afirmar que SNGA sea superior a un LLM completo. Sí permiten una afirmación más acotada y alineada con la tesis híbrida: para almacenamiento y evocación de asociaciones multimodales discretas, una red geométrica esparsa puede servir como núcleo de memoria más eficiente que activar una red densa de lenguaje. En la validación actual se usa una fracción fija y pequeña de agentes activos (`32/180000`, aproximadamente `0.018%`) durante la evocación. El LLM, en esta visión, no desaparece; se acopla a SNGA para traducir entre símbolos humanos y estados geométricos.

Finalmente, `advanced_experiment` valida los mecanismos biomiméticos extendidos:

```text
tetrahedra             = 374
episodios              = 8
aristas_causales       = 50
aristas_consolidadas   = 20
prediccion A->B        = 100.0% precision / 100.0% recall
prediccion B->C        = 100.0% precision / 100.0% recall
```

El experimento muestra consolidación de trazas repetidas, poda/olvido de huellas transitorias, replay episódico, causalidad dirigida y geometría tetraédrica activa. Esta evidencia sigue siendo sintética, pero amplía el argumento: SNGA puede modelarse no solo como memoria asociativa, sino como un tejido plástico con dinámica temporal y capacidad predictiva inicial.

Para verificar que el aprendizaje no consiste únicamente en cambiar pesos, sino también en deformar la geometría de la malla, se añadió `geometry_learning_experiment`. Este experimento mide distancias internas del concepto, distancia hacia distractores, energía libre y aristas asociativas antes y después del entrenamiento:

```text
before:
  intra_distance      = 137.988
  distractor_distance = 199.521
  compactness         = 0.692
  free_energy         = 16581.990
  associative_edges   = 0

after:
  intra_distance      = 108.401
  distractor_distance = 188.358
  compactness         = 0.576
  free_energy         = 1495.597
  associative_edges   = 21
  mean_weight         = 3.028
```

La distancia interna del concepto se redujo `21.44%`, la relación de compactación mejoró `16.79%` y la energía libre cayó `90.98%`. Esto apoya directamente la tesis geométrica: al aprender, la red no solo almacena asociaciones en pesos; también compacta regiones conceptuales y modifica el paisaje físico del complejo.

Como consecuencia, el sustrato aprendido debe persistir. `persistent_substrate_experiment` valida que la geometría deformada, las aristas aprendidas y los pesos pueden guardarse y cargarse en una nueva instancia de red con la misma topología:

```text
save:
  agents = 1200
  edges  = 4022
  causal = 0

load:
  agents = 1200
  edges  = 4022
  causal = 0

geometry:
  trained_distance = 112.753
  loaded_distance  = 112.753
  delta            = 0.000000

recall_after_load = 100.0%
```

Esto confirma que el aprendizaje geométrico no se pierde al cerrar el proceso: el sistema conserva posiciones, profundidad, pesos, longitudes de reposo, aristas asociativas y causalidad. En ejecución, el visor opcional mantiene un buffer serializado y permite autoguardado/guardado manual del sustrato.

Se evaluaron además tres variantes inspiradas en fractalidad orgánica y proporción áurea:

```text
FibonacciLayout  = distribución espacial áurea de agentes
GoldenLearning   = escalado áureo del learning-rate
GoldenPruning    = refuerzo solo si utilidad >= 1/phi
```

La prueba comparativa (`golden_fractal_experiment`) mostró que la distribución Fibonacci y el escalado áureo de pesos no superaron al baseline:

```text
Baseline        score = 0.775; leakage = 100.0%
FibonacciLayout score = 0.775; leakage = 100.0%
GoldenLearning  score = 0.775; leakage = 100.0%
GoldenPruning   score = 1.000; leakage = 0.0%
```

La variante útil fue **GoldenPruning**: mantuvo recall, causalidad y predicción lingüística, pero eliminó la fuga asociativa en el escenario de prueba. Por eso el proyecto conserva la proporción áurea solo como umbral de utilidad para poda/refuerzo selectivo, no como geometría fractal rígida. La lectura experimental es que la forma fractal espacial no mejora por sí misma el aprendizaje; lo que sí ayuda es impedir que asociaciones de baja utilidad se consoliden.

También se validó la capa de oscilaciones funcionales (`oscillatory_modes_experiment`). La prueba compara una red base contra una red con oscilaciones activadas. El escenario está diseñado para medir consolidación por replay: la red recibe episodios, no refuerzo asociativo explícito. La red sin oscilaciones no consolida esos episodios; la red con Delta/SleepReplay sí los refuerza durante reposo:

```text
baseline:
  target_recall   = 0.0%
  sequence_recall = 100.0%
  replay_edges    = 0
  score           = 0.500

oscillatory:
  target_recall   = 100.0%
  sequence_recall = 100.0%
  replay_edges    = 81
  score           = 0.959
```

El resultado apoya la hipótesis de que los ritmos funcionales pueden servir como medio global de coordinación sin campo magnético físico: Delta habilita consolidación, Beta/Gamma organizan regiones enfocadas y Alpha actúa como filtro.

La separación entre motor matemático y capa neuronal se validó con `mesh_engine_validation`:

```text
mesh:
  nodes        = 576
  edges        = 1947
  triangles    = 990
  tetrahedra   = 330
  depth_layers = 3

neural:
  recall       = 100.0%
  oscillations = true
  mode         = Focus
```

Esto confirma que el motor 3D puede construir la topología de forma independiente y que la capa neuronal oscilatoria puede usarla sin perder recuperación de memoria.

`reasoning_experiment` valida razonamiento topológico inicial mediante datos sintéticos donde las respuestas correctas no fueron entrenadas directamente:

```text
directo fuego->ruptura      = 0.0% recall
transitivo fuego->ruptura   = 100.0% recall
directo perro->animal       = 0.0% recall
transitivo perro->animal    = 100.0% recall
contradiccion frio/caliente = tension 25.000; delta energia 100.000
```

La lectura es importante: el sistema no memorizó el atajo `fuego -> ruptura` ni `perro -> animal`; los recuperó recorriendo rutas causales/jerárquicas dentro de la malla. Además, la coactivación de estados incompatibles (`frio` y `caliente`) elevó la energía libre, proporcionando una forma geométrica de contradicción.

`reasoning_benchmark` escala esta prueba a miles de estructuras sintéticas y compara inferencia amplia contra rutas optimizadas por flujo/evaporación:

```text
causal_chains     = 5000
hierarchy_chains  = 3000
contradictions    = 3000

causal:
  broad_recall        = 100.0%
  broad_precision     = 4.5%
  optimized_recall    = 96.6%
  optimized_precision = 96.7%

jerarquia:
  broad_recall        = 100.0%
  broad_precision     = 11.7%
  optimized_recall    = 100.0%
  optimized_precision = 100.0%

contradiccion:
  tension_media       = 6.250
  delta_energia_medio = 25.000
```

Estos resultados sugieren que la red puede pasar de "encontrar muchas rutas posibles" a "consolidar rutas útiles". El mecanismo no usa multiplicación matricial densa; opera sobre rutas, pesos locales, evaporación y energía libre.

Como experimento temporal, `language_experiment` implementa un tokenizador de palabras y firmas contextuales n-grama que actúan como entrada/salida lingüística provisional para SNGA. El objetivo no es reemplazar al LLM periférico futuro, sino probar si la malla puede aprender regularidades discretas de secuencia:

```text
vocab                 = 36
context_window        = 2
eval_next_token top1  = 42.9%
eval_next_token top3  = 59.5%
eval_next_token top5  = 81.0%
```

La lectura es limitada pero útil: SNGA aprende transiciones lingüísticas locales y puede generar secuencias gramaticalmente simples dentro del dominio sintético. Sin embargo, no muestra todavía comprensión semántica abierta ni capacidades comparables a transformers. Este resultado refuerza la decisión arquitectónica de mantener el LLM como interfaz lingüística periférica en versiones futuras.

Se añadió una segunda variante con **memoria de trabajo pre-lingüística**. En esta modalidad, antes de generar palabras, la red recibe una huella abstracta de la idea a expresar: determinante, sujeto, acción, objeto y lugar. Esta huella no es un LLM; es un estado topológico interno que organiza la intención antes de renderizarla en tokens. Con esta memoria de trabajo, el mismo experimento obtiene:

```text
train_sentences               = 3840
vocab                         = 64
eval_next_token top1          = 27.1%
eval_next_token top3          = 52.9%
eval_next_token top5          = 65.7%
eval_with_working_memory top1 = 97.1%
eval_with_working_memory top3 = 98.6%
eval_with_working_memory top5 = 100.0%
```

La diferencia entre ambas pruebas es significativa. Sin memoria de trabajo, SNGA aprende regularidades locales pero tiende a producir frases genéricas. Con memoria de trabajo, la red dispone de un estado abstracto organizado y puede verbalizarlo de forma consistente incluso con frases más largas, adjetivos, adverbios y conectores causales/temporales. Esto apoya la hipótesis biológica del paper: el lenguaje funciona mejor como renderizador de una idea ya estructurada que como único sustrato del pensamiento.

Un benchmark lingüístico escalado (`scaled_language_benchmark`) amplía el corpus a 19,220 frases sintéticas, vocabulario de 75 tokens y una malla de 92,400 nodos:

```text
eval_baseline_long top1       = 69.0%
eval_baseline_long top3       = 82.1%
eval_baseline_long top5       = 85.7%
eval_with_working_memory top1 = 100.0%
eval_with_working_memory top3 = 100.0%
eval_with_working_memory top5 = 100.0%
dialogue_coherence score      = 100.0% (10/10 casos)
internal_language_probe       = ok
```

La métrica `dialogue_coherence` evalúa si la respuesta contiene los conceptos clave esperados para intenciones como energía, memoria, lenguaje, razonamiento, GPU, matrices e inhibición. El resultado indica que SNGA puede sostener comunicación coherente en un dominio pequeño cuando la respuesta está guiada por memoria de trabajo, memoria episódica, predicción de patrón y planificación local. Sin embargo, no demuestra lenguaje abierto general ni reemplaza a un LLM: valida una ruta experimental para usar SNGA como núcleo pre-lingüístico y renderizador simbólico limitado.

Para evaluar la tesis híbrida con un LLM periférico real, se añadió `snga_llm_peripheral_benchmark`. La prueba usa códigos privados (`xq17`, `v9k2`, `p3lm`) que no pertenecen al conocimiento general del modelo lingüístico. SNGA aprende internamente qué significan y aprende también una cadena causal privada. Luego se comparan tres condiciones:

```text
SNGA inference     = memoria/inferencia interna de la malla
Gemma only         = LLM periférico sin memoria privada SNGA
SNGA + Gemma       = SNGA infiere; Gemma solo verbaliza
```

Resultado con `gemma2:2b` vía Ollama:

```text
snga_inference = 100.0%
gemma_only     = 0.0%
snga_plus_gemma= 100.0%
```

La lectura es acotada pero importante: no demuestra superioridad general frente a LLMs masivos, pero sí demuestra una clase de ventaja arquitectónica. Cuando la respuesta depende de memoria privada aprendida durante la vida del sistema, SNGA puede actuar como núcleo persistente de memoria/razonamiento y el LLM pequeño puede quedar reducido al papel de renderizador lingüístico.

Un segundo benchmark (`autonomous_language_benchmark`) elimina el plan manual explícito. La red aprende rutas `prompt -> intención abstracta -> respuesta` y usa un filtrado semántico simple del prompt para enfocar contenido sobre palabras funcionales. En una versión ampliada con 16 intenciones, vocabulario de 148 tokens y 186,000 nodos, obtiene:

```text
intent_accuracy     = 89.6%
response_coherence  = 89.6%
```

Esto indica que SNGA puede empezar a internalizar la memoria de trabajo: no solo verbaliza una idea dada, sino que infiere la intención abstracta desde la entrada del usuario dentro de un dominio pequeño ampliado. El resultado sigue lejos de un LLM general y todavía falla en algunas paráfrasis ambiguas, pero reduce la dependencia del plan externo y acerca el sistema a una arquitectura de conversación autónoma centrada en el núcleo geométrico.

### 6.1 Sustrato Fractal, Currículo Lingüístico y Compresión Validada

Después de los experimentos lingüísticos iniciales se evaluó una variante más cercana a la hipótesis biológica del proyecto: reemplazar la rejilla uniforme por un sustrato fractal jerárquico, organizar el lenguaje por regiones de escala y entrenar el núcleo con un maestro lingüístico externo solo durante la fase de datos. En esta modalidad, Gemma/Ollama no actúa como memoria ni como motor de conversación en inferencia; su papel es generar lotes de entrenamiento y exámenes curriculares. El estado aprendido queda dentro de SNGA.

La topología fractal se genera mediante contracción jerárquica:

```text
s = branches^(-1 / D)
D ~= 2.65
```

El generador `SimplicialMeshEngine::fractal_3d` permite fijar el número objetivo de nodos (`target_nodes`) y construir una malla multi-escala. La primera comparación entre una grilla 3D y una malla fractal estática mostró que la fractal podía conservar recall y precisión con menos nodos/aristas en una prueba pequeña:

```text
Grid3d:
  nodes      = 1080
  edges      = 3706
  tetrahedra = 493
  recall     = 100.0%
  precision  = 100.0%
  leakage    = 0.0%

Fractal3d:
  nodes      = 781
  edges      = 2454
  tetrahedra = 468
  recall     = 100.0%
  precision  = 100.0%
  leakage    = 0.0%
```

Al transferir el estado lingüístico escalado previo (`snga_scaled_gemma_language.snga`) a la nueva malla fractal, la carga directa falló porque el estado persistido guardaba un conteo fijo de agentes. Para preservar la nueva geometría se añadió una carga de memoria (`load_persistent_memory_state`) que importa aristas y causalidad sin sobrescribir posiciones. Con una malla fractal de `5760` nodos, la memoria escalada se transfirió sin perder predicciones:

```text
grid_learned:
  topics    = 12/12
  questions = 6/6
  relations = 5/5
  conf      = 3.078 promedio

fractal_learned:
  topics    = 12/12
  questions = 6/6
  relations = 5/5
  conf      = 3.078 promedio

fractal_baseline:
  topics    = 0/12
  questions = 0/6
  relations = 0/5
```

La transferencia inicial mantuvo conocimiento pero elevó la energía geométrica. Se implementó entonces un recocido local de longitudes de reposo (`anneal_active_edge_rest_lengths`) combinado con replay de prompts lingüísticos. En seis épocas, la energía bajó aproximadamente 45 veces sin degradar cobertura ni confianza:

```text
before:
  coverage = 23/23
  conf     = 3.078
  energy   = 173445952.0

after:
  coverage = 23/23
  conf     = 3.078
  energy   = 3777787.5
```

Posteriormente se aplicó compresión validada. El criterio fue conservador: se podan aristas asociativas de baja utilidad por tandas, se recalcula la firma top-k de un conjunto de pruebas y solo se acepta la poda si la firma permanece idéntica. Con este procedimiento, el estado fractal comprimido preservó exactamente las 23 predicciones top-k evaluadas y redujo fuertemente tamaño y aristas:

```text
grid_original:
  edges       = 609202
  associative = 590013
  file        = 36643830 bytes

fractal_compressed:
  edges       = 302110
  associative = 270648
  file        = 21420425 bytes

knowledge:
  cases              = 23
  avg_overlap        = 100.0%
  exact_topk_matches = 23/23
```

La siguiente etapa fue entrenar lingüística española en un currículo jerárquico con Gemma como maestro de datos:

```text
letras -> silabas -> palabras -> uniones_de_palabras -> oraciones
       -> gramatica_basica -> espanol_medio
```

El binario `fractal_gemma_curriculum_trainer` genera lotes con Gemma, entrena SNGA, genera exámenes de etapa y permite avanzar si el maestro considera suficiente la señal de la red. El entrenamiento se reanuda desde progreso persistente:

```text
batches = 113
lessons = 3613
stage   = oraciones
```

Para imitar la organización cortical distribuida del lenguaje, se añadió una codificación regional compatible. La firma antigua se conserva para no olvidar el conocimiento previo, pero los nuevos patrones agregan componentes por región:

```text
0-20%   letras, grafemas, fonemas
20-40%  silabas y combinaciones letra-sonido
40-65%  palabras y raices
65-85%  frases, oraciones y roles gramaticales
85-100% significado, causa, intencion y contexto
```

La validación posterior mostró aprendizaje por etapas. La red entrenada supera al baseline fractal comprimido por margen amplio en confianza y overlap:

```text
letras:
  trained  conf = 60.969  overlap = 4.9%
  baseline conf = 2.346   overlap = 1.0%

silabas:
  trained  conf = 108.193 overlap = 14.5%
  baseline conf = 3.483   overlap = 0.7%

palabras/significado:
  trained  conf = 224.077 overlap = 11.8%
  baseline conf = 4.923   overlap = 1.6%

frases:
  trained  conf = 216.000 overlap = 19.5%
  baseline conf = 4.998   overlap = 0.8%

oraciones:
  trained  conf = 417.000 overlap = 15.4%
  baseline conf = 8.041   overlap = 1.5%

semantica_media:
  trained  conf = 286.526 overlap = 4.8%
  baseline conf = 8.130   overlap = 1.1%
```

Algunos ejemplos concretos del probe amplio (`fractal_curriculum_broad_probe`) son:

```text
input    = "p a"
expected = "pa"
trained overlap  = 19.7%
baseline overlap = 0.0%

input    = "c a s a"
expected = "casa lugar para vivir"
trained overlap  = 13.9%
baseline overlap = 0.9%

input    = "nino corre"
expected = "sujeto y verbo"
trained overlap  = 19.5%
baseline overlap = 0.3%

input    = "el nino come pan"
expected = "sujeto verbo objeto"
trained overlap  = 16.1%
baseline overlap = 2.0%
```

También se probó un razonamiento relacional controlado:

```text
Juan es padre de Ana.
Ana es madre de Luis.
=> Juan es abuelo de Luis.
```

El resultado fue mixto:

```text
reasoning_family_path:
  juan -> ana -> luis
  overlap_luis = 0.0%

reasoning_family_conclusion:
  juan_padre_ana + ana_madre_luis => juan_abuelo_luis
  overlap = 16.7%
```

La lectura es que SNGA no demuestra todavía inferencia transitiva lingüística robusta en lenguaje abierto; sin embargo, sí muestra una señal parcial cuando se le proporciona una estructura relacional explícita. Esto separa dos capacidades: la red ya aprende rutas lingüísticas y asociaciones semánticas simples, pero necesita entrenamiento específico en reglas relacionales para generalizar razonamientos familiares, jerárquicos o lógicos.

El crecimiento de conexiones fue un problema práctico. Durante el entrenamiento curricular, las aristas causales llegaron a varios millones. La poda causal validada resultó mucho más difícil que la poda asociativa: en un estado con `1,356,872` aristas causales solo pudieron eliminarse `654` sin cambiar la firma de validación. En cambio, la poda asociativa permitió grandes reducciones. Una compresión máxima sobre el currículo reciente redujo 800,000 aristas asociativas y preservó el conocimiento validado:

```text
before:
  edges       = 1288157
  associative = 1264807
  causal      = 3698798
  file        = 145948033 bytes

after:
  edges       = 488157
  associative = 464807
  causal      = 3698798
  file        = 104750496 bytes
  knowledge   = preserved
```

Finalmente se amplió el sustrato fractal de `5760` a `11520` nodos. La expansión conserva las rutas aprendidas usando `SNGA_PATTERN_NODES=5760` para las firmas heredadas, mientras deja espacio regional nuevo para aprendizaje posterior:

```text
expanded:
  agents       = 11520
  edges        = 506036
  causal_edges = 3699245
  energy       = 0.0

validation:
  letras              = 2/2
  silabas             = 2/2
  palabras            = 2/2
  uniones_de_palabras = 2/2
  oraciones           = 2/2
  gramatica_basica    = 1/1
  espanol_medio       = 1/1
```

La experiencia con el chat SNGA-tokenizador mostró una limitación adicional. El estado entrenado sí contenía conocimiento, pero el chat original no lo aprovechaba bien porque usaba una lista fija de respuestas, una función `infer_topic` demasiado estrecha y una codificación distinta de la usada por el currículo. Al alinear el chat con la codificación jerárquica/regional y evitar que `--once` guardara automáticamente, las respuestas mejoraron en consultas simples:

```text
hola
-> Hola. Soy SNGA funcionando con tokenizador y memoria en la malla fractal.

que es casa
-> Casa es una palabra que nombra un lugar para vivir.

que es miedo
-> Miedo es una emocion que aparece ante peligro, amenaza o incertidumbre.

que es un saludo
-> Un saludo es una frase social breve para iniciar contacto, como hola.
```

Estos resultados deben interpretarse con cautela. La red muestra aprendizaje lingüístico estructural y señales iniciales de significado, pero todavía no posee generación abierta robusta. El sistema actual funciona mejor como memoria geométrica y selector de respuestas simbólicas que como generador autónomo de lenguaje natural. La dirección prometedora no es hacer que SNGA imite directamente a un LLM, sino usar la malla como sustrato persistente y causal, con una interfaz lingüística cada vez más alineada con sus regiones internas.

### 6.8 Sustrato Semántico-Ejecutivo con Celdas y Representantes de Foco

Después del currículo lingüístico fractal se construyó un sustrato semántico-ejecutivo de mayor escala. El objetivo fue separar regiones funcionales análogas a un sistema cortical simplificado:

```text
semantic_hub_atl        -> integración conceptual abstracta
concept_binder          -> unión de rasgos dispersos
semantic_control        -> selección contextual y desambiguación
executive_logic_dlpfc   -> reglas, restricciones y alternativas
working_memory          -> mantenimiento de metas y contexto
planner                 -> pasos de respuesta / plan
control_gate            -> inhibición de significados no pertinentes
visual/auditory/somatic/linguistic/episodic slots -> periferia futura
```

El estado base semántico-ejecutivo contiene `98,304` agentes distribuidos en 12 regiones. Sobre este sustrato se entrenó un adaptador lingüístico con Gemma como maestro/fallback de datos, pero la memoria resultante queda persistida dentro de SNGA. El probe final evalúa cinco tareas pequeñas:

```text
"que es una manzana roja"
"planea cena vegetariana sin carne"
"banco en el parque"
"si llueve que pasa con el suelo"
"explica tu plan antes de responder"
```

La línea base entrenada antes de introducir celdas semánticas y representantes de foco mostraba buena verificación y verbalización, pero no lograba concentrar concepto ni frame en el top-k estricto:

```text
baseline_adapter:
  input_to_concept_hits      = 0/5
  input_to_concept_wide_hits = 4/5
  input_to_frame_hits        = 0/5
  frame_to_verbal_hits       = 5/5
  output_to_verification     = 5/5
  confidence                 = 630211477504.000
```

La introducción progresiva de celdas y detectores produjo una mejora acumulativa:

```text
solo SemanticCell:
  cells      = 40
  confidence = 648475770880.000

detectores + error predictivo, 2 lecciones:
  cells      = 94
  confidence = 695367630848.000

detectores + error predictivo, 5 lecciones:
  cells                  = 232
  input_to_concept_wide  = 5/5
  confidence             = 801984806912.000
```

El hallazgo importante fue que la red ya estaba llegando a la región semántica correcta, pero el conocimiento quedaba distribuido fuera del top-k estricto. Es decir, había recall amplio pero faltaba una selección de representantes. Para resolverlo se añadió una memoria de foco persistente (`focus_edges`) entrenada por error predictivo. La versión final guardada en `data/snga_fractal_semantic_executive_gemma_adapter.snga` contiene:

```text
trained_network:
  nodes        = 98304
  edges        = 5670449
  associative  = 5376075
  causal       = 2748979
  cells        = 241
  focus        = 5812
  energy       = 122422520.0
```

El probe oficial posterior a la optimización obtuvo:

```text
trained_summary:
  input_to_concept_hits       = 5/5
  input_to_concept_wide_hits  = 5/5
  input_to_frame_hits         = 5/5
  frame_to_verbal_hits        = 5/5
  output_to_verification_hits = 5/5
  input_to_concept_overlap    = 100.0%
  input_to_frame_overlap      = 100.0%
  frame_to_verbal_overlap     = 14.7%
  output_to_verify_overlap    = 100.0%
  confidence                  = 13111338729472.000
```

La comparación es significativa para la hipótesis del paper. Los mecanismos de celdas y detectores mejoraron recall amplio; los representantes de foco con promoción controlada convirtieron ese recall en ranking estricto sin perder verificación. Una variante inicial de foco demasiado agresiva obtuvo `input_to_concept=5/5` e `input_to_frame=5/5`, pero degradó `output_to_verification` a `2/5`; la versión estable limita la promoción a inferencia transitiva/exact-hop y deja `predict_next_pattern` sin contaminación. Esta separación sugiere una división funcional útil:

```text
SemanticCell      -> agrupación asociativa de alto orden
FeatureDetector   -> ancla local reusable de rasgo/intención/frame
focus_edges       -> selección explícita de representantes top-k
predict_next      -> verificación/salida sin promoción de foco
```

El resultado no demuestra comprensión lingüística abierta. Sí muestra una propiedad nueva del sustrato: puede aprender una región semántica amplia y luego aprender representantes discretos para consultarla con precisión, manteniendo separadas memoria conceptual, inferencia de foco y verificación de salida.

## 7. Viabilidad hacia AGI

SNGA no demuestra AGI por sí mismo. Su valor en esta dirección es que separa tres funciones que los LLMs actuales tienden a mezclar: representación conceptual persistente, inferencia dinámica y renderizado lingüístico. Esta separación podría ser relevante para AGI si el núcleo geométrico demuestra cuatro propiedades:

1. **Grounding multimodal verificable.** Un concepto debe quedar ligado a visión, audio, tacto, acción y lenguaje sin depender solo de correlaciones textuales.
2. **Aprendizaje continuo local.** La red debe incorporar conceptos nuevos sin reentrenar todo el sistema ni destruir atractores previos.
3. **Restricciones físicas y causales.** El motor geométrico debe poder penalizar respuestas incompatibles con relaciones espaciales, temporales o causales aprendidas.
4. **Interfaz independiente del lenguaje.** La misma configuración abstracta debería poder renderizarse como texto, imagen, acción robótica o consulta estructurada.

Por tanto, el camino hacia AGI se formula como una hipótesis experimental: si un núcleo geométrico esparso puede aprender atractores multimodales estables y guiar módulos periféricos especializados, entonces podría reducir parte de la dependencia actual en modelos monolíticos de lenguaje. La afirmación requiere evidencia empírica comparativa; no debe presentarse como conclusión cerrada.

Con los resultados actuales, la evaluación de viabilidad queda así:

- **Viable:** memoria asociativa multimodal, propagación esparsa, aprendizaje estructural local, celdas semánticas de alto orden, detectores locales de rasgo/intención/frame, representantes de foco top-k, poda áurea por utilidad, oscilaciones funcionales, control de cascadas por inhibición, replay episódico sintético, causalidad dirigida inicial, inferencia transitiva, contradicción energética, optimización de rutas por flujo/evaporación y geometría 3D/tetraédrica.
- **No demostrado:** lenguaje natural abierto, planificación larga, transferencia fuera de distribución, grounding con sensores reales y superioridad general frente a LLMs.
- **Hipótesis fuerte siguiente:** combinar SNGA con encoders reales y un LLM periférico podría reducir costo en tareas donde el LLM hoy funciona como memoria semántica, dejando al LLM como traductor, narrador y adaptador lingüístico.

## 8. Viabilidad de Hardware

La arquitectura SNGA es compatible con hardware donde la localidad física importa:

- FPGAs con regiones dedicadas a submallas.
- Procesadores neuromórficos con comunicación por spikes.
- NoC con micro-paquetes asíncronos.
- CPU multinúcleo para simulación y prototipado.

En el estado actual del proyecto, la ruta prioritaria no es introducir GPU ni campos globales. La mejora que sí produjo resultados medibles fue reforzar el núcleo CPU con memoria episódica, atención dinámica, predicción de patrones, rollouts internos y planificación sobre rutas causales. Por tanto, la optimización práctica inmediata debe concentrarse en:

1. Reducir asignaciones temporales.
2. Mantener buffers reutilizables.
3. Limitar spikes y agentes activos.
4. Medir costo por subgrafo activo, no por tamaño total de la red.
5. Evitar dependencias de hardware que no estén disponibles en el entorno real.

## 9. Limitaciones del Prototipo

La versión actual es una demostración de mecanismo, no un modelo entrenado. Sus principales limitaciones son:

- La codificación textual es determinista pero no semántica.
- La visualización principal sigue siendo 2D; la geometría 3D/hiperbólica existe como soporte experimental, no como validación completa de escala.
- El aprendizaje por coactivación es local y simple; todavía no separa causalidad de coincidencia.
- No existe aún decodificador LLM periférico.
- No hay persistencia de memoria episódica en disco.
- El crecimiento topológico existe solo como refuerzo/creación de aristas, no como neurogénesis estructural completa.
- La validación actual usa datos sintéticos; todavía no prueba visión, audio o lenguaje reales.
- La fuga residual entre conceptos indica falta de mecanismos de inhibición y desambiguación causal.
- En el currículo lingüístico fractal, las aristas causales crecen con rapidez y son difíciles de podar sin alterar firmas predictivas; esto sugiere que hace falta limitar causalidad por región, por etapa o por top-k durante el entrenamiento, no solo comprimir después.
- El chat SNGA-tokenizador sigue siendo un renderizador simbólico de respuestas candidatas; aunque ya usa la codificación jerárquica/regional, no genera lenguaje abierto desde la malla con la flexibilidad de un LLM.
- Las celdas semánticas y los `focus_edges` resuelven el ranking estricto en el probe semántico-ejecutivo actual, pero añaden nuevas estructuras persistentes que deben ser reguladas para evitar crecimiento excesivo, sobreajuste al conjunto de validación o contaminación de rutas de verificación.
- Los resultados `5/5` del sustrato semántico-ejecutivo se obtienen en cinco casos sintéticos/controlados. No prueban generalización lingüística abierta; prueban que la arquitectura puede convertir recall amplio en representantes top-k cuando el dominio está bien anclado.

Estas limitaciones son deliberadas: el objetivo inicial es aislar el principio operativo de relajación local y visualizarlo con claridad.

## 10. Ruta de Investigación

Los siguientes pasos técnicos son:

1. Sustituir la proyección sintética por encoders reales: CLIP/ViT para visión, encoder de audio y LLM pequeño para lenguaje.
2. Implementar crecimiento topológico completo: creación, poda y consolidación de aristas/símplices según coactivación y predicción.
3. Añadir persistencia de memoria episódica y snapshots del mundo interno.
4. Fortalecer atención dinámica basada en sorpresa predictiva, objetivo y contexto.
5. Mejorar el planificador multi-paso sobre rutas causales y contradicciones.
6. Evaluar acoplamientos más finos entre Delta/Theta/Alpha/Beta/Gamma y tareas cognitivas.
7. Entrenar un adaptador de lectura que observe regiones activas, distancias y rutas causales.
8. Medir energía, latencia y sparsity frente a una línea base transformer pequeña.
9. Evaluar tareas pequeñas de grounding: recuperación de rasgos, consistencia física simple y aprendizaje incremental.
10. Evaluar replay episódico con secuencias temporales largas y benchmarks causales.
11. Convertir la optimización de rutas en un mecanismo no supervisado basado solo en reducción de energía libre y estabilidad del atractor.
12. Mantener el currículo lingüístico fractal con regiones por escala, pero controlar la creación de causalidad durante el aprendizaje: presupuestos por región, consolidación solo tras exámenes y poda causal validada por firmas top-k.
13. Desarrollar un decodificador SNGA-tokenizador más expresivo que lea patrones regionales y no dependa únicamente de respuestas simbólicas predefinidas.
14. Regular `SemanticCell` y `focus_edges` con presupuestos por región, consolidación por repetición y poda validada, para conservar el beneficio de ranking estricto sin crecimiento no controlado.
15. Evaluar el sustrato semántico-ejecutivo con paráfrasis no vistas y tareas fuera de las cinco frases de validación actuales.

## 11. Conclusión

SNGA plantea un cambio de énfasis: de predicción lingüística densa como única arquitectura cognitiva a una arquitectura híbrida donde la memoria e inferencia abstracta ocurren en una malla geométrica esparsa y el lenguaje se resuelve en módulos periféricos especializados. El sistema no elimina los LLMs, sino que los reubica como interfaces de entrada/salida. El núcleo cognitivo se modela como un complejo simplicial que minimiza tensión local, permitiendo una forma de inferencia más cercana a navegación conceptual que a multiplicación matricial global.

El prototipo Rust de este repositorio materializa la primera pieza de esa hipótesis: una red binaria de agentes, una malla simplicial, propagación por eventos, memoria episódica, atención dinámica, predicción causal, planificación local y relajación elástica observable en tiempo real.
