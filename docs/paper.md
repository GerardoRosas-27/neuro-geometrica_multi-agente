# SNGA -> CDT-RQM

## Consolidación Causal Relacional de Memoria Neuro-Geométrica

### Resumen

Los modelos de lenguaje de gran escala (LLMs) han demostrado una capacidad notable para interpolar patrones lingüísticos, pero su arquitectura dominante mezcla en una misma tubería tres funciones que conviene separar: aprendizaje de asociaciones, consolidación causal y expresión simbólica. Esta mezcla obliga a resolver razonamiento abstracto, memoria episódica y sintaxis mediante álgebra lineal densa, atención cuadrática y retropropagación global. El resultado es un régimen de cómputo intensivo con altos costos energéticos, latencia elevada y fragilidad semántica ante tareas que exigen anclaje causal o separación contextual.

Este documento consolida la arquitectura del proyecto en una ruta única: **entrenar en SNGA, migrar a CDT-RQM, aplicar annealing Graphity y ejecutar inferencia en CDT-RQM consolidado**. SNGA queda como etapa de entrenamiento neuro-geométrico: aprende memoria causal y asociaciones binarias en una malla simplicial. CDT-RQM queda como sustrato final de ejecución: cristaliza esas rutas en hardware causal foliado, separa futuros relativos mediante un software relacional RQM y comprime la geometría con Graphity sin perder memoria.

La tesis actual ya no es "SNGA como sustrato final", sino **SNGA como entrenador y CDT-RQM como sustrato consolidado**. Los resultados internos muestran que CDT-RQM conserva el accuracy de SNGA, reduce fuga semántica y comprime fuertemente la geometría activa. La afirmación sigue siendo experimental: no demuestra AGI ni física fundamental, pero sí establece una ruta computacional medible para pasar de memoria aprendida a hardware causal compacto.

La ruta canónica del repositorio es:

```text
1. Entrenar en SNGA.
2. Migrar conocimiento causal a CDT-RQM.
3. Aplicar annealing Graphity validado por memoria.
4. Ejecutar inferencia en CDT-RQM consolidado.
```

El comando principal es:

```text
cargo run --bin snga_to_cdt_rqm_consolidate
```

El estado consolidado resultante queda en:

```text
data/cdt_rqm_consolidated_from_snga.cdt_rqm
```

## 1. Introducción

La inteligencia artificial contemporánea suele tratar el lenguaje como el medio universal del pensamiento. En los LLMs, el razonamiento aparece como una trayectoria dentro de un espacio latente de alta dimensión entrenado para predicción de tokens. Este enfoque ha escalado con éxito, pero introduce una dependencia fuerte en multiplicaciones matriciales masivas, memoria de activaciones, sincronización global y optimización por retropropagación. En términos energéticos, la red paga por activar una gran fracción de sus parámetros incluso cuando el problema requiere solo una pequeña región conceptual.

La arquitectura consolidada parte de una hipótesis distinta: el lenguaje no es el sustrato primario de la cognición, sino una interfaz periférica. La representación abstracta se aprende primero como memoria causal neuro-geométrica en SNGA, pero no se conserva ahí como forma final. Una vez aprendida, se migra a CDT-RQM, donde el conocimiento queda separado en dos capas: hardware causal foliado y software relacional por observador.

La inspiración neurobiológica procede de la separación funcional entre sistemas de aprendizaje, consolidación y expresión. En esta analogía, SNGA opera como etapa de adquisición de memoria causal; CDT-RQM opera como sustrato consolidado de ejecución; el LLM o encoder periférico queda como traductor entre lenguaje humano y activaciones internas. La arquitectura resultante es:

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
SNGA: entrenamiento neuro-geométrico causal
        |
        v
migración causal + destilación RQM
        |
        v
CDT-RQM: hardware causal + software relacional
        |
        v
annealing Graphity validado por memoria
        |
        v
estado consolidado compacto
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

## 3. Arquitectura Consolidada SNGA -> CDT-RQM

La arquitectura final del repositorio queda reducida a una ruta principal:

```text
SNGA aprende -> CDT-RQM consolida -> Graphity comprime -> CDT-RQM ejecuta
```

Las demás variantes del repositorio quedan como antecedentes experimentales o herramientas de comparación. La ruta de producción experimental es:

```text
cargo run --bin snga_to_cdt_rqm_consolidate
cargo run --bin snga_vs_cdt_rqm_consolidated_tests
cargo run --bin snga_vs_cdt_rqm_profile
```

La responsabilidad de cada capa es:

```text
SNGA:
  aprender asociaciones y causalidad binaria
  generar causal_edges_snapshot()
  actuar como profesor/entrenador

CDT hardware:
  imponer foliación temporal
  conservar solo rutas causales válidas
  medir acción Regge combinatoria

RQM software:
  separar futuros por observador
  reducir fuga semántica/contextual
  recuperar candidatos relativos a una frontera local

Graphity:
  enfriar el grafo
  podar aristas no necesarias
  aceptar cambios solo si no degradan memoria
```

En esta estructura, SNGA ya no es el sustrato final de ejecución; es la fase de adquisición. El conocimiento consolidado vive en `CdtRqmUniverseSubstrate`.

## 3.1 SNGA Como Entrenador

### 3.1.1 Núcleo de Simulación Histórico

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

### 3.11 RQF-SNGA: Sustrato Simplicial de Campo Relacional

La nueva hipótesis de trabajo extiende SNGA con un sustrato inspirado en principios relacionales de la mecánica cuántica, sin afirmar que el sistema sea cuántico físico ni que la cuántica implique conciencia. La tesis arquitectónica es más precisa: si en una descripción relacional las propiedades no pertenecen de forma absoluta a los objetos, sino que aparecen en interacciones, entonces una arquitectura cognitiva puede evitar representar el significado como estado intrínseco de un nodo. El significado puede emerger como coherencia local entre relaciones observadas.

Esta capa se denomina provisionalmente **RQF-SNGA** (*Relational Quantum-Field Simplicial SNGA*) o **Sustrato Simplicial de Campo Relacional**. La unidad básica ya no es solo el agente ni la arista, sino el estado de una relación respecto a un observador:

```text
psi_ij^O = A_ij^O * exp(i * phi_ij^O)
```

donde `i` y `j` son agentes, `O` es el observador/contexto, `A` es amplitud relacional, `phi` es fase contextual, `|psi|^2` funciona como probabilidad de activación y la fase permite interferencia constructiva o destructiva. En código, esta idea se implementa inicialmente en `relational_field.rs` mediante relaciones con:

```text
RelationalState {
  amplitude,
  phase,
  coherence,
  uncertainty,
  last_observed_tick
}
```

El punto conceptual es que un nodo como `banco` no almacena un significado absoluto. Para un observador financiero puede resonar con `dinero`, `credito` e `institucion`; para un observador de parque puede resonar con `sentarse`, `madera` y `parque`. Ambas descripciones pueden coexistir porque pertenecen a marcos relacionales distintos. La operación de inferencia se interpreta como colapso local computacional, no como colapso físico:

```text
entrada + observador O
  -> relaciones psi_ij^O vecinas
  -> interferencia de fases compatibles
  -> supresión de fases incompatibles
  -> patrón estabilizado relativo a O
```

RQF-SNGA conserva el sustrato geométrico de SNGA, pero añade una capa de campo encima de aristas y celdas. La energía libre puede ampliarse con términos relacionales:

```text
F_total =
  F_geometrica
  + F_interferencia
  + F_incertidumbre
  + F_coherencia_simplex
```

Un término simple de interferencia puede definirse como:

```text
F_interferencia = sum A_ij^O * (1 - cos(phi_ij^O - phi_O))
```

Para un triángulo semántico `(i, j, k)`, los símplices miden si las fases cierran de forma coherente:

```text
C_ijk^O = cos(phi_ij^O + phi_jk^O + phi_ki^O)
```

Si el ciclo de fases cierra, la celda tiene baja tensión. Si no cierra, aparece contradicción, ambigüedad o necesidad de otro observador contextual. Esta formulación encaja naturalmente con los complejos simpliciales existentes: los triángulos y tetraedros dejan de ser solo restricciones de área/volumen y pasan a ser detectores de coherencia relacional.

El aprendizaje sigue siendo local:

```text
si una relación predice correctamente:
  aumentar amplitud
  alinear fase con el observador
  subir coherencia
  reducir incertidumbre

si una relación falla:
  reducir amplitud
  desplazar fase
  bajar coherencia
  aumentar incertidumbre
```

Esta regla mantiene la filosofía del proyecto: no hay retropropagación global ni tabla densa de atención. El conocimiento se estabiliza como patrón de relaciones observables. El prototipo inicial incluye un experimento ejecutable con:

```text
cargo run --bin relational_field_substrate_experiment
```

El objetivo de ese experimento no es demostrar ventaja general, sino validar una propiedad mínima: el mismo nodo puede colapsar hacia significados distintos según el observador relacional, y los símplices pueden medir coherencia de fase entre relaciones compatibles o incompatibles.

Para comparar esta hipótesis contra el sustrato SNGA previo, se añadió un benchmark más amplio:

```text
cargo run --bin relational_field_comparison_benchmark
```

La prueba entrena 12 palabras ambiguas con 24 marcos de significado. Cada palabra comparte el mismo nodo conceptual de entrada, pero debe resolverse de forma distinta según el observador/contexto. El benchmark compara tres condiciones:

```text
RQF-SNGA observer_relational = campo relacional con observador explícito
legacy_snga_cue_only        = sustrato SNGA previo usando solo la clave ambigua
legacy_snga_cue_and_context = sustrato SNGA previo usando clave + contexto explícito
```

Esta comparación no mide lenguaje general ni razonamiento abierto. Mide una propiedad específica: separación de significados competidores cuando la misma clave semántica debe colapsar hacia marcos incompatibles.

### 3.12 Integración Híbrida: SNGA Dice Qué Rutas Existen, RQF Dice Qué Rutas Son Reales Para El Observador

La conclusión de diseño posterior al benchmark es no reemplazar `SimplicialNetwork`, sino integrar `RelationalFieldSubstrate` como capa opcional de modulación contextual. La separación de responsabilidades queda así:

```text
SNGA:
  memoria estructural
  rutas causales
  aristas y símplices
  relajación geométrica
  atractores
  replay e inhibición

RQF:
  observador/contexto
  fase relacional
  amplitud contextual
  coherencia e incertidumbre
  interferencia constructiva/destructiva
  reducción de fuga semántica
```

En código, `SimplicialNetwork` ahora puede contener un campo relacional opcional:

```text
SimplicialNetwork {
  agents,
  edges,
  simplices,
  tetrahedra,
  semantic_cells,
  spikes,
  causal_edges,
  focus_edges,
  episodes,
  relational_field: Option<RelationalFieldSubstrate>
}
```

Cuando la capa RQF no está activada, la red conserva el comportamiento anterior. Cuando se activa, la propagación por aristas sigue usando el sustrato geométrico existente, pero cada spike recibe una compuerta relacional:

```text
spike_score =
  edge_weight
  * attention_weight
  * oscillatory_weight
  * relational_spike_weight
```

donde:

```text
relational_spike_weight ~= |psi_ij^O|^2 * max(0, cos(phi_ij^O - phi_O))
```

La decisión arquitectónica es importante: RQF no crea las rutas físicas de la malla; las filtra según el observador. Por eso el lema operativo de la versión híbrida es:

```text
SNGA dice qué rutas existen.
RQF dice qué rutas son reales para este observador.
```

El comando de entrenamiento del sustrato híbrido es:

```text
cargo run --bin snga_hybrid_rqf_relational_trainer
```

Por defecto escribe:

```text
data/snga_hybrid_rqf_relational.snga
data/snga_hybrid_rqf_relational.rqf
```

También acepta variables de entorno:

```text
SNGA_HYBRID_RQF_OUTPUT
SNGA_HYBRID_RQF_FIELD_OUTPUT
SNGA_HYBRID_RQF_EPOCHS
```

### 3.13 Sustrato CDT-Graphity: Geometría Causal Dinámica Para Reconfiguración Topológica

Se añadió un sustrato experimental independiente inspirado en **Triangulaciones Dinámicas Causales (CDT)** y **Quantum Graphity**. No pretende simular gravedad cuántica física de forma literal. Traduce tres ideas matemáticas a reglas computacionales para redes simpliciales:

```text
CDT:
  imponer foliación temporal estricta
  separar aristas espaciales y temporales
  prohibir ciclos causales hacia atrás
  aceptar solo tetraedros tipo (3,1) o (2,2)

Quantum Graphity:
  iniciar desde un grafo caliente/sobre-conectado
  enfriar mediante reducción de sorpresa/FEP
  podar enlaces inestables
  cristalizar una geometría local más esparsa

Regge discreto:
  medir curvatura como déficit combinatorio alrededor de aristas
  penalizar zonas topológicas con demasiados o muy pocos tetraedros incidentes
```

El módulo `cdt_graphity.rs` define un sustrato nuevo desde cero:

```text
CdtGraphitySubstrate {
  nodes: foliados por slice temporal
  edges: Spatial | Temporal
  tetrahedra: T31 | T22
  temperature
  regge_action
}
```

Las aristas espaciales conectan nodos dentro de la misma rebanada temporal. Las aristas temporales conectan únicamente `t -> t+1`; internamente se orientan siempre hacia adelante. Esto convierte la causalidad en una restricción estructural, no en una preferencia de entrenamiento. Si una conexión violaría la foliación, simplemente no se crea.

La energía usada en el paso de aprendizaje combina error predictivo, temperatura y una Acción de Regge combinatoria:

```text
F ~= prediction_error + lambda * I_R + temperature_bias
```

En vez de calcular ángulos diedros continuos, el prototipo usa una aproximación discreta:

```text
I_R = sum_edges length(edge) * abs(target_incident_tetrahedra - incident_tetrahedra(edge))
```

Esto preserva la intuición física: demasiada o muy poca densidad simplicial alrededor de una arista equivale a curvatura/inestabilidad. Cuando el error predictivo sube, la temperatura sube y Graphity permite romper/proponer enlaces. Cuando el error baja, el sistema se enfría y conserva las rutas estables.

El experimento ejecutable es:

```text
cargo run --bin cdt_graphity_substrate_experiment
```

Su lectura esperada es que el sustrato pase de un grafo caliente y sobre-conectado a una geometría causal más local, manteniendo cero violaciones de CDT.

### 3.14 CDT-RQM: Hardware Causal y Software Relacional

La siguiente composición implementa la intuición de dos capas:

```text
CDT-Graphity = hardware topológico causal
RQM/RQF      = software relacional ejecutándose sobre fronteras locales
```

En esta versión, CDT-Graphity conserva las restricciones de bajo nivel: foliación temporal, aristas espaciales/temporales, temperatura, poda, propuesta de enlaces y Acción de Regge combinatoria. La capa RQM no modifica directamente esas reglas físicas. En cambio, observa una frontera local activa, colapsa candidatos relativos al observador y entrega al hardware una expectativa de futuro:

```text
frontera activa + observador O
  -> RQM: collapse relativo psi_ij^O
  -> candidatos esperados
  -> CDT: step(expected_next)
  -> FEP/Regge/Graphity reconfiguran el hardware
```

El módulo `cdt_rqm.rs` materializa esta composición:

```text
CdtRqmUniverseSubstrate {
  hardware: CdtGraphitySubstrate,
  software: RelationalFieldSubstrate
}
```

Esto permite simular una separación análoga a:

```text
hardware = geometría causal del universo informacional
software = reglas relacionales de actualización local
```

La interpretación sigue siendo computacional, no una afirmación cosmológica literal. El valor técnico está en que el hardware impone restricciones causales que el software no puede violar, mientras que RQM ayuda a seleccionar qué futuro local debe proponerse al sustrato cuando existe ambigüedad o aprendizaje incompleto.

El experimento ejecutable es:

```text
cargo run --bin cdt_rqm_universe_substrate_experiment
```

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

También se añadió un benchmark específico para el nuevo sustrato relacional (`relational_field_comparison_benchmark`). La prueba usa 12 términos ambiguos (`banco`, `planta`, `raton`, `cola`, `carta`, `vela`, `sierra`, `copa`, `radio`, `red`, `llave`, `corriente`) y 24 marcos semánticos incompatibles. El objetivo es comparar si el significado se separa mejor cuando el marco de referencia se modela como observador relacional en vez de como otro patrón de entrada dentro del sustrato previo.

Resultado de referencia:

```text
RQF-SNGA comparison benchmark
cases=12 frames=24 training_epochs=18 symbols=108 rqf_relations=264

rqf_observer_relational:
  accuracy              = 100.0%
  purity                = 100.0%
  leakage               = 0.0%
  simplex_coherence     = 1.000
  incompatible_tension  = 0.754

legacy_snga_cue_only:
  accuracy              = 0.0%
  purity                = 50.0%
  leakage               = 50.0%

legacy_snga_cue_and_context:
  accuracy              = 100.0%
  purity                = 66.7%
  leakage               = 33.3%
```

La lectura es acotada pero relevante. El sustrato previo con contexto explícito puede elegir el marco correcto, pero conserva fuga hacia el significado competidor porque la clave ambigua refuerza ambos sentidos en el mismo espacio asociativo. El sustrato RQF-SNGA separa los estados por observador: conserva el acierto y reduce la fuga a cero en este escenario sintético. Además, los símplices compatibles cierran fase con coherencia máxima, mientras que los incompatibles acumulan tensión. Esto sugiere una mejora real para tareas donde el problema central no es recordar más asociaciones, sino mantener realidades semánticas relativas sin mezclarlas prematuramente.

Después de integrar RQF dentro de `SimplicialNetwork`, se añadió `hybrid_relational_simplicial_benchmark`. Esta prueba compara cuatro condiciones sobre el mismo conjunto de 12 ambigüedades:

```text
1. SNGA puro
2. RQF puro
3. SNGA + contexto explícito
4. SNGA + RQF integrado
```

Resultado de referencia:

```text
SNGA + RQF hybrid relational benchmark
cases=12 frames=24 epochs=18 symbols=108 hybrid_relations=5472

1_snga_puro:
  accuracy              = 0.0%
  purity                = 50.0%
  leakage               = 50.0%
  active_agents         = 28.0
  spikes                = 288.0
  energy                = 1461905.125
  stability             = 0.515

2_rqf_puro:
  accuracy              = 100.0%
  purity                = 100.0%
  leakage               = 0.0%
  active_agents         = 3.0
  spikes                = 0.0
  energy                = 0.000
  stability             = 1.000
  incompatible_tension  = 0.754

3_snga_contexto_explicito:
  accuracy              = 0.0%
  purity                = 50.0%
  leakage               = 50.0%
  active_agents         = 32.0
  spikes                = 288.0
  energy                = 1460082.500
  stability             = 0.516

4_snga_rqf_integrado:
  accuracy              = 100.0%
  purity                = 100.0%
  leakage               = 0.0%
  active_agents         = 24.0
  spikes                = 288.0
  energy                = 1459284.375
  stability             = 0.511
  incompatible_tension  = 0.754
```

La condición `RQF puro` resuelve la ambigüedad con máxima limpieza, pero no usa cuerpo geométrico, spikes ni atractores físicos. La condición `SNGA+RQF integrado` es más importante para la tesis del proyecto: conserva la malla y la propagación por eventos, pero reduce la fuga semántica a cero y usa menos agentes activos que la condición con contexto explícito. La estabilidad aún no mejora; queda como métrica a optimizar en siguientes iteraciones mediante acoplamiento más fino entre fase relacional, inhibición y relajación geométrica.

El nuevo sustrato CDT-Graphity se validó con `cdt_graphity_substrate_experiment`. La prueba inicia una malla causal sobre-conectada, entrena rutas de predicción entre rebanadas temporales y mide si el sistema se enfría sin violar la foliación CDT:

```text
CDT + Graphity substrate experiment

initial:
  free_energy           = 37.528
  regge                 = 2498.500
  temp                  = 0.945
  active_edges          = 544
  spatial               = 255
  temporal              = 289
  tetrahedra            = 66
  causality_violations  = 0

final:
  free_energy           = 27.901
  regge                 = 1859.750
  pred_error            = 0.000
  temp                  = 0.097
  active_nodes          = 22
  active_edges          = 355
  spatial               = 71
  temporal              = 284
  tetrahedra            = 52
  causality_violations  = 0

prediction_score        = 83.3%
edge_reduction          = 34.7%
temperature_drop        = 89.8%
```

La lectura es que la foliación causal se preserva durante toda la reconfiguración y Graphity sí produce una transición de fase computacional: el sustrato reduce aristas activas, baja temperatura, disminuye la acción Regge combinatoria y mantiene rutas temporales predictivas. A diferencia de SNGA/RQF, este sustrato no se centra todavía en semántica; se centra en **cómo debe mutar la geometría** cuando el FEP exige reconfiguración.

La capa compuesta CDT-RQM se validó con `cdt_rqm_universe_substrate_experiment`. La prueba compara el hardware CDT solo contra el hardware con software RQM observando fronteras locales:

```text
epoch=2:
  hardware_score = 88.9%
  universe_score = 100.0%
  relations      = 26
  temp           = 0.674

epoch=8:
  hardware_score = 100.0%
  universe_score = 100.0%
  relations      = 26
  temp           = 0.243

hardware_only:
  score                 = 100.0%
  temp_initial          = 0.945
  temp_final            = 0.276
  edges_initial         = 280
  edges_final           = 266
  regge_final           = 1267.750
  causality_violations  = 0

cdt_rqm_universe:
  score                 = 100.0%
  temp_initial          = 0.945
  temp_final            = 0.230
  edges_initial         = 280
  edges_final           = 277
  regge_final           = 1306.000
  relations             = 26
  causality_violations  = 0
```

La lectura es que RQM actúa como software relacional: propone futuros observados antes de que el hardware haya cristalizado por completo, alcanzando 100% en época 2 mientras el hardware solo todavía está en 88.9%. Al final ambos aprenden la tarea, pero CDT-RQM mantiene una temperatura ligeramente menor y conserva cero violaciones de causalidad. Esto apoya la separación de capas: el hardware causal limita lo físicamente permitido; el software relacional acelera la selección de expectativas locales.

Para comparar CDT-RQM contra el sustrato SNGA anterior, se añadió:

```text
cargo run --bin cdt_rqm_vs_snga_benchmark
```

La prueba usa ambigüedad causal: el mismo patrón de entrada puede tener dos futuros distintos según observador/contexto. SNGA anterior aprende transiciones causales planas (`cue -> futuro`) y una variante con contexto explícito (`cue + contexto`). CDT-RQM usa la misma frontera causal como hardware CDT, pero RQM separa los futuros relativos por observador.

Resultado de referencia:

```text
CDT-RQM vs previous SNGA benchmark
cases=4 frames=8 epochs=8 cdt_rqm_relations=72

early_snga_cue_only:
  accuracy = 25.0%
  purity   = 50.0%
  leakage  = 50.0%

early_snga_context:
  accuracy = 87.5%
  purity   = 53.6%
  leakage  = 46.4%

early_cdt_rqm:
  accuracy = 100.0%
  purity   = 91.8%
  leakage  = 8.2%

final_snga_cue_only:
  accuracy = 25.0%
  purity   = 50.0%
  leakage  = 50.0%

final_snga_context:
  accuracy = 100.0%
  purity   = 53.6%
  leakage  = 46.4%

final_cdt_rqm:
  accuracy = 100.0%
  purity   = 91.8%
  leakage  = 8.2%

cdt_rqm_hardware:
  temp                  = 0.663
  active_edges          = 415
  regge                 = 2357.750
  causality_violations  = 0
```

El hallazgo interesante no es solo que CDT-RQM acierte. SNGA con contexto explícito también alcanza 100% al final. La diferencia está en la **pureza del futuro seleccionado**: SNGA conserva casi la mitad de la masa predictiva en el futuro competidor, porque el mismo `cue` fue entrenado hacia ambos destinos. CDT-RQM mantiene ambos futuros en el mismo hardware causal, pero los separa por observador relacional. El resultado es una reducción de fuga de `46.4%` a `8.2%`, con cero violaciones de causalidad CDT. Esta es la primera evidencia interna de que el enfoque hardware/software puede ser mejor que el SNGA anterior en tareas donde el problema principal no es memorizar rutas, sino evitar que futuros incompatibles se mezclen.

Para evaluar si el conocimiento ya aprendido por SNGA puede migrarse al nuevo sustrato, se añadió:

```text
cargo run --bin cdt_rqm_migration_benchmark
```

La prueba entrena primero un SNGA causal binario completo. Después exporta sus aristas causales mediante `causal_edges_snapshot()` y las usa de dos formas:

```text
1. Como priors temporales válidos en el hardware CDT, cuando respetan t -> t+1.
2. Como relaciones RQM destiladas por observador, para inicializar el software relacional.
```

Resultado de referencia:

```text
CDT-RQM migration benchmark
lessons=6
snga_epochs=12
cdt_rqm_fewshot_epochs=0
snga_causal_edges=54
migrated_temporal_edges=47
migrated_rqm_relations=108

snga_previous_full:
  accuracy = 100.0%
  purity   = 91.7%
  leakage  = 8.3%

cdt_rqm_scratch_fewshot:
  accuracy = 0.0%
  purity   = 0.0%
  leakage  = 100.0%

cdt_rqm_migrated_fewshot:
  accuracy = 100.0%
  purity   = 91.7%
  leakage  = 8.3%

scratch_hardware:
  temp                  = 0.945
  active_edges          = 879
  regge                 = 4169.500
  causality_violations  = 0

migrated_hardware:
  temp                  = 0.945
  active_edges          = 926
  regge                 = 4463.250
  causality_violations  = 0
```

El hallazgo aquí es distinto al benchmark anterior. CDT-RQM migrado no supera todavía a SNGA completo en precisión o fuga; reproduce su comportamiento sin entrenamiento adicional. Eso es importante porque demuestra compatibilidad de representación: el conocimiento causal aprendido por el sustrato anterior puede cristalizarse como hardware CDT y software RQM. El CDT-RQM desde cero, sin entrenamiento, no predice nada; el CDT-RQM migrado arranca inmediatamente con `100%` de acierto y `8.3%` de fuga. La migración funciona como bootstrapping de un nuevo universo causal desde una memoria SNGA ya entrenada.

La fase posterior a esa migración es **annealing Graphity validado por memoria**:

```text
cargo run --bin cdt_rqm_annealing_benchmark
```

La regla es conservadora. Se proponen pasos de enfriamiento/poda sobre el hardware CDT, pero solo se aceptan si preservan la memoria validada por RQM:

```text
aceptar paso si:
  accuracy_final >= accuracy_previa
  leakage_final  <= leakage_previa
  y reduce Regge o reduce aristas activas
```

Resultado de referencia:

```text
CDT-RQM post-migration Graphity annealing benchmark
lessons=6
anneal_attempts=32
accepted=3
migrated_temporal_edges=45
relations=108

memory:
  accuracy 100.0% -> 100.0%
  leakage    8.3% ->   8.3%

geometry:
  regge        4447.250 -> 2519.250
  active_edges      949 -> 464
  causality_violations = 0
```

Este es el primer resultado donde la migración no solo preserva conocimiento, sino que además permite una mejora estructural clara. El sistema conserva la memoria causal migrada, mantiene la fuga constante y reduce casi a la mitad las aristas activas, bajando fuertemente la acción Regge combinatoria. En términos de la metáfora hardware/software: SNGA entrega una memoria causal; CDT-RQM la cristaliza; Graphity la enfría hasta una geometría más compacta sin romper la computación relacional que corre encima.

Para revalidar el hallazgo en una prueba más amplia, se añadió:

```text
cargo run --bin cdt_rqm_extended_validation
```

Esta prueba usa 20 lecciones binarias, 4 observadores relacionales y una SNGA entrenada durante 16 épocas como fuente. Después migra el conocimiento causal a CDT-RQM y aplica annealing validado por memoria:

```text
CDT-RQM extended migrated validation
lessons=20
observers=4
snga_epochs=16
anneal_attempts=64
accepted=3

migration:
  snga_causal_edges       = 164
  snga_associative_edges  = 284
  migrated_temporal_edges = 144
  rqm_relations           = 328

snga_trained:
  accuracy = 100.0%
  purity   = 94.0%
  leakage  = 6.0%

cdt_rqm_migrated_before_anneal:
  accuracy = 100.0%
  purity   = 96.4%
  leakage  = 3.6%

cdt_rqm_migrated_after_anneal:
  accuracy = 100.0%
  purity   = 96.4%
  leakage  = 3.6%

geometry:
  regge        15474.250 -> 710.000
  active_edges      3161 -> 164
  compression       94.8%
  causality_violations = 0

memory_delta_vs_snga:
  accuracy = +0.0%
  purity   = +2.4%
  leakage  = -2.4%
```

Este resultado revalida el hallazgo con mayor escala interna. CDT-RQM no solo conserva la memoria de SNGA: en esta tarea obtiene menor fuga y mayor pureza, y luego comprime agresivamente la geometría causal sin perder rendimiento. La compresión final deja casi solo las rutas temporales útiles, mientras la foliación CDT permanece intacta. La parte más interesante es que el annealing no fue guiado por un objetivo simbólico global, sino por una restricción local de memoria: aceptar poda solo si el software RQM sigue recuperando los futuros correctos.

La etapa siguiente convierte esta ruta en una consolidación persistente. El comando:

```text
cargo run --bin snga_to_cdt_rqm_consolidate
```

ejecuta el flujo completo:

```text
1. Entrenar SNGA causal binario.
2. Exportar causal_edges_snapshot().
3. Migrar aristas causales válidas al hardware CDT.
4. Destilar predicciones SNGA al software RQM.
5. Aplicar annealing Graphity validado por memoria.
6. Guardar el estado consolidado CDT-RQM.
```

El estado queda en:

```text
data/cdt_rqm_consolidated_from_snga.cdt_rqm
```

Resultado de referencia:

```text
SNGA -> CDT-RQM consolidation
saved=true
lessons=20
snga_epochs=16
relations=328
migrated_temporal_edges=148

anneal:
  attempts=64
  accepted=2
  accuracy 100.0% -> 100.0%
  leakage    3.6% ->   3.6%
  regge  15386.000 -> 710.000
  edges       3165 -> 164
  causality_violations=0
```

Para comparar directamente la SNGA entrenada contra el CDT-RQM consolidado se añadió:

```text
cargo run --bin snga_vs_cdt_rqm_consolidated_tests
```

La suite comprueba cuatro propiedades:

```text
1. Paridad de memoria: CDT-RQM no pierde accuracy frente a SNGA.
2. Fuga no peor: CDT-RQM no mezcla más futuros que SNGA.
3. Compresión Graphity: baja Regge/aristas activas.
4. Causalidad CDT: no aparecen violaciones de foliación.
```

Resultado:

```text
SNGA vs consolidated CDT-RQM tests

snga:
  accuracy = 100.0%
  purity   = 94.0%
  leakage  = 6.0%

cdt_rqm:
  accuracy = 100.0%
  purity   = 96.4%
  leakage  = 3.6%
  relations = 328

anneal:
  accepted = 2
  regge    = 15696.750 -> 710.000
  edges    = 3209 -> 164
  causality_violations = 0

test_memory_parity       = PASS
test_leakage_not_worse   = PASS
test_graphity_compression= PASS
test_cdt_causality       = PASS
suite                    = PASS
```

Este resultado establece una diferencia arquitectónica clara. SNGA sigue siendo útil como etapa de entrenamiento y memoria causal inicial. CDT-RQM funciona como sustrato consolidado: conserva la memoria, separa mejor los futuros relativos, y comprime la geometría causal en un hardware foliado más pequeño.

Finalmente se añadió un perfil comparativo estructural:

```text
cargo run --bin snga_vs_cdt_rqm_profile
```

Este benchmark no solo mide acierto. Compara peso estructural, optimización, conocimiento retenido, aristas, nodos activos y eficiencia por arista activa:

```text
SNGA vs CDT-RQM trained substrate profile
lessons=20
snga_epochs=16
anneal_attempts=64

snga_memory:
  accuracy = 100.0%
  purity   = 94.0%
  leakage  = 6.0%
  margin   = 8.400

cdt_rqm_memory:
  accuracy = 100.0%
  purity   = 96.4%
  leakage  = 3.6%
  margin   = 1.783

snga_structure:
  total_nodes       = 1024
  active_nodes_avg  = 17.2
  total_edges       = 3600
  active_edges      = 3600
  associative_edges = 284
  causal_edges      = 164
  semantic_cells    = 20
  free_energy_avg   = 7385.874
  knowledge_mass    = 3456.823

cdt_rqm_structure:
  total_nodes       = 256
  active_nodes_avg  = 10.9
  total_edges       = 3168
  active_edges      = 334
  spatial_edges     = 42
  temporal_edges    = 292
  rqm_relations     = 328
  regge             = 1670.000
  temperature       = 0.844
  knowledge_mass    = 553.434

optimization:
  cdt_edges_before  = 3168
  cdt_edges_after   = 334
  edge_compression  = 89.5%
  regge_before      = 15452.250
  regge_after       = 1670.000
  regge_reduction   = 89.2%
  anneal_accepted   = 3

efficiency:
  snga_accuracy_per_active_edge = 0.000278
  cdt_accuracy_per_active_edge  = 0.002994
  causality_violations          = 0
```

La lectura de perfil es más fuerte que una comparación de accuracy. Ambos sustratos alcanzan 100%, pero CDT-RQM lo hace con menos nodos totales, menos nodos activos promedio, mucha menos geometría activa y menor fuga. La eficiencia por arista activa aumenta aproximadamente un orden de magnitud. Esta métrica sugiere que la ventaja principal de CDT-RQM no es “saber más” que SNGA, sino **consolidar el mismo conocimiento en un hardware causal mucho más compacto y menos contaminado**.

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

## 7. Viabilidad de la Ruta Consolidada

La arquitectura consolidada no demuestra AGI. Su valor actual es más concreto: separa entrenamiento, consolidación, compresión e inferencia. La hipótesis útil ya no es que SNGA sea por sí mismo el núcleo final, sino que SNGA puede actuar como maestro causal y CDT-RQM como sustrato final más eficiente.

La ruta viable queda así:

```text
SNGA:
  aprende memoria causal inicial
  produce rutas candidatas

CDT-RQM:
  consolida rutas válidas
  separa futuros por observador
  aplica restricciones causales de hardware
  reduce geometría mediante Graphity

LLM/periféricos:
  traducen lenguaje/sensores a activaciones
  renderizan el estado consolidado
```

Con los resultados actuales, la evaluación queda:

- **Consolidado:** entrenamiento SNGA, migración causal, destilación RQM, annealing Graphity, persistencia CDT-RQM y comparación SNGA vs CDT-RQM.
- **Demostrado en pruebas sintéticas:** paridad de accuracy, menor fuga, compresión fuerte de aristas activas y cero violaciones de foliación CDT.
- **No demostrado:** lenguaje natural abierto, observadores emergentes desde datos reales, grounding sensorial real, planificación larga y superioridad general frente a LLMs.
- **Hipótesis siguiente:** usar `data/cdt_rqm_consolidated_from_snga.cdt_rqm` como sustrato primario y medir si la ventaja estructural se mantiene en dominios más abiertos.

## 8. Viabilidad Computacional

El perfil comparativo indica que CDT-RQM consolidado es más económico de ejecutar que SNGA entrenado en la tarea evaluada:

```text
SNGA:
  total_nodes      = 1024
  active_nodes_avg = 17.2
  active_edges     = 3600
  leakage          = 6.0%

CDT-RQM:
  total_nodes      = 256
  active_nodes_avg = 10.9
  active_edges     = 334
  leakage          = 3.6%
```

La conclusión práctica es:

```text
SNGA    = mejor como fase de entrenamiento.
CDT-RQM = mejor como fase consolidada de inferencia y mantenimiento.
```

En hardware futuro, CDT-RQM es más compatible con ejecución esparsa porque conserva solo rutas temporales útiles y separa explícitamente hardware causal de software relacional.

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
- El benchmark RQF-SNGA demuestra ventaja en ambigüedad sintética con observadores definidos a mano. La versión híbrida ya reduce fuga dentro de `SimplicialNetwork`, pero todavía no mejora estabilidad del atractor y no prueba que los observadores puedan emerger solos desde datos reales ni que la mejora se conserve en lenguaje abierto, percepción multimodal o tareas con ruido adversarial.
- El sustrato CDT-Graphity valida foliación causal y enfriamiento topológico en un experimento pequeño. Todavía no está acoplado al núcleo semántico SNGA/RQF, y su Acción de Regge es una aproximación combinatoria de densidad simplicial, no una medición geométrica continua de ángulos diedros.
- La capa CDT-RQM demuestra una separación inicial hardware/software, pero usa observadores y lecciones sintéticas. Falta evaluar si el software relacional sigue acelerando la predicción cuando los observadores emergen de datos reales, cuando hay ruido y cuando la frontera activa contiene muchos distractores.
- La comparación `cdt_rqm_vs_snga_benchmark` muestra menor fuga que SNGA anterior en ambigüedad causal controlada. No prueba todavía superioridad general: el conjunto es pequeño, los observadores están definidos manualmente y los patrones se construyen para aislar el problema de mezcla entre futuros incompatibles.
- La consolidación SNGA -> CDT-RQM preserva conocimiento causal útil, permite arranque en frío y guarda un estado persistente. El annealing posterior reduce acción Regge y aristas activas sin perder memoria en pruebas sintéticas pequeñas y medianas, pero falta probarlo con migración incremental, ruido, tareas no binarias y dominios semánticos abiertos.

Estas limitaciones son deliberadas: el objetivo inicial es aislar el principio operativo de relajación local y visualizarlo con claridad.

## 10. Ruta de Investigación Consolidada

Los siguientes pasos técnicos quedan reducidos a la ruta principal:

1. Usar `data/cdt_rqm_consolidated_from_snga.cdt_rqm` como sustrato primario de inferencia.
2. Añadir carga/lectura completa del estado consolidado, no solo guardado.
3. Ejecutar nuevos benchmarks directamente sobre CDT-RQM consolidado.
4. Sustituir patrones sintéticos por encoders reales de texto/visión/audio.
5. Aprender observadores RQM desde datos en vez de declararlos manualmente.
6. Medir latencia, memoria, energía y sparsity frente a SNGA y frente a una línea base transformer pequeña.
7. Mantener SNGA como fase de entrenamiento y CDT-RQM como fase de producción experimental.

## 11. Conclusión

El resultado consolidado del proyecto es una arquitectura de dos fases. SNGA aprende; CDT-RQM conserva, separa y comprime. Esta decisión elimina la ambigüedad arquitectónica anterior: SNGA ya no se presenta como sustrato final, sino como entrenador causal. El sustrato final es CDT-RQM consolidado.

La ruta canónica queda:

```text
Entrenar en SNGA
  -> migrar a CDT-RQM
  -> aplicar annealing Graphity
  -> ejecutar inferencia en CDT-RQM consolidado
```

Los resultados actuales muestran que CDT-RQM puede igualar el accuracy de SNGA, reducir fuga y usar una geometría activa mucho más compacta. La conclusión técnica es clara: para este proyecto, la dirección principal ya no es expandir SNGA indefinidamente, sino usar SNGA para enseñar y CDT-RQM para ejecutar.
