# Arquitectura Neuro-Geométrica Multi-Agente

## Complejos Simpliciales Guiados por Energía Libre con Renderizado Lingüístico Periférico

### Resumen

Los modelos de lenguaje de gran escala (LLMs) han demostrado una capacidad notable para interpolar patrones lingüísticos, pero su arquitectura dominante mezcla en una misma tubería tres funciones que en sistemas biológicos suelen estar separadas: percepción, estabilización conceptual y expresión simbólica. Esta mezcla obliga a resolver razonamiento abstracto, coherencia física, memoria episódica y sintaxis mediante álgebra lineal densa, atención cuadrática y retropropagación global. El resultado es un régimen de cómputo intensivo con altos costos energéticos, latencia elevada y fragilidad semántica ante tareas que exigen anclaje espacial o causal.

Este documento propone el **Sistema Neuro-Geométrico de Agentes (SNGA)**, una arquitectura híbrida experimental en la que la cognición abstracta se modela como relajación mecánica de una malla topológica descentralizada, mientras que los LLMs permanecen como interfaces lingüísticas periféricas. La tesis central no es reemplazar los LLMs, sino desacoplar el lenguaje del núcleo de memoria e inferencia: SNGA almacena y evoca estados conceptuales mediante complejos simpliciales esparsos; los LLMs traducen entre lenguaje humano y activaciones geométricas internas. El núcleo cognitivo no es un vector denso, sino un complejo simplicial formado por agentes binarios, aristas asíncronas y símplices de orden superior. Cada agente minimiza una energía libre local derivada de la tensión geométrica con sus vecinos.

La tesis se presenta como una hipótesis de arquitectura, no como una demostración de AGI. El repositorio acompaña la propuesta con un prototipo íntegro en Rust. La implementación incluye una red binaria event-driven, una malla simplicial 2D, una regla de relajación elástica local, un demostrador multimodal sintético y un motor gráfico basado en `macroquad` para observar la estabilización de la red en tiempo real.

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

donde `V` es el conjunto de agentes binarios, `E` el conjunto de canales asíncronos y `S` el conjunto de símplices que preservan estructura de orden superior. En el prototipo Rust, `S` se limita a triángulos 2D para facilitar visualización, pero la formulación se extiende naturalmente a tetraedros.

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

El núcleo implementado en Rust se organiza en tres capas:

- `geometry.rs`: álgebra vectorial mínima para posiciones, distancias y fuerzas.
- `simplicial.rs`: agentes, aristas, triángulos, picos y dinámica de relajación.
- `render.rs`: motor gráfico 2D para visualizar la red y sus métricas.

La estructura principal es `SimplicialNetwork`. Contiene:

- `agents`: vértices binarios con posición, velocidad, activación y sorpresa.
- `edges`: restricciones elásticas de distancia.
- `simplices`: triángulos con área objetivo.
- `spikes`: cola asíncrona de eventos.
- `config`: parámetros físicos y topológicos.

El prototipo genera una rejilla triangulada. Cada celda rectangular se divide en dos triángulos, creando una malla con interacciones binarias y ternarias.

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

## 7. Viabilidad hacia AGI

SNGA no demuestra AGI por sí mismo. Su valor en esta dirección es que separa tres funciones que los LLMs actuales tienden a mezclar: representación conceptual persistente, inferencia dinámica y renderizado lingüístico. Esta separación podría ser relevante para AGI si el núcleo geométrico demuestra cuatro propiedades:

1. **Grounding multimodal verificable.** Un concepto debe quedar ligado a visión, audio, tacto, acción y lenguaje sin depender solo de correlaciones textuales.
2. **Aprendizaje continuo local.** La red debe incorporar conceptos nuevos sin reentrenar todo el sistema ni destruir atractores previos.
3. **Restricciones físicas y causales.** El motor geométrico debe poder penalizar respuestas incompatibles con relaciones espaciales, temporales o causales aprendidas.
4. **Interfaz independiente del lenguaje.** La misma configuración abstracta debería poder renderizarse como texto, imagen, acción robótica o consulta estructurada.

Por tanto, el camino hacia AGI se formula como una hipótesis experimental: si un núcleo geométrico esparso puede aprender atractores multimodales estables y guiar módulos periféricos especializados, entonces podría reducir parte de la dependencia actual en modelos monolíticos de lenguaje. La afirmación requiere evidencia empírica comparativa; no debe presentarse como conclusión cerrada.

Con los resultados actuales, la evaluación de viabilidad queda así:

- **Viable:** memoria asociativa multimodal, propagación esparsa, aprendizaje estructural local, control de cascadas por inhibición.
- **No demostrado:** razonamiento causal, lenguaje natural abierto, planificación, transferencia fuera de distribución y superioridad general frente a LLMs.
- **Hipótesis fuerte siguiente:** combinar SNGA con encoders reales y un LLM periférico podría reducir costo en tareas donde el LLM hoy funciona como memoria semántica, dejando al LLM como traductor, narrador y adaptador lingüístico.

## 8. Viabilidad de Hardware

La arquitectura SNGA es especialmente compatible con hardware donde la localidad física importa:

- FPGAs con regiones dedicadas a submallas.
- Procesadores neuromórficos con comunicación por spikes.
- NoC con micro-paquetes asíncronos.
- Simuladores físicos en GPU cuando se prioriza visualización o prototipado.

Una implementación futura debería particionar la malla en sectores, asignar cada sector a un núcleo y comunicar solo eventos de frontera. Esto reduciría sincronización global y permitiría escalado espacial.

## 9. Limitaciones del Prototipo

La versión actual es una demostración de mecanismo, no un modelo entrenado. Sus principales limitaciones son:

- La codificación textual es determinista pero no semántica.
- El complejo es 2D, no hiperbólico ni 3D.
- El aprendizaje por coactivación es local y simple; todavía no separa causalidad de coincidencia.
- No existe aún decodificador LLM periférico.
- No hay persistencia de memoria episódica en disco.
- El crecimiento topológico existe solo como refuerzo/creación de aristas, no como neurogénesis estructural completa.
- La validación actual usa datos sintéticos; todavía no prueba visión, audio o lenguaje reales.
- La fuga residual entre conceptos indica falta de mecanismos de inhibición y desambiguación causal.

Estas limitaciones son deliberadas: el objetivo inicial es aislar el principio operativo de relajación local y visualizarlo con claridad.

## 10. Ruta de Investigación

Los siguientes pasos técnicos son:

1. Sustituir la proyección sintética por encoders reales: CLIP/ViT para visión, encoder de audio y LLM pequeño para lenguaje.
2. Implementar crecimiento topológico completo: creación, poda y consolidación de aristas/símplices según coactivación y predicción.
3. Añadir geometría hiperbólica para jerarquías conceptuales.
4. Incorporar símplices 3D para restricciones volumétricas.
5. Entrenar un adaptador cross-attention que lea matrices de distancia estabilizadas.
6. Medir energía, latencia y sparsity frente a una línea base transformer.
7. Evaluar tareas pequeñas de grounding: recuperación de rasgos, consistencia física simple y aprendizaje incremental.
8. Añadir inhibición lateral y normalización de energía para reducir fuga asociativa.

## 11. Conclusión

SNGA plantea un cambio de énfasis: de predicción lingüística densa como única arquitectura cognitiva a una arquitectura híbrida donde la memoria e inferencia abstracta ocurren en una malla geométrica esparsa y el lenguaje se resuelve en módulos periféricos especializados. El sistema no elimina los LLMs, sino que los reubica como interfaces de entrada/salida. El núcleo cognitivo se modela como un complejo simplicial que minimiza tensión local, permitiendo una forma de inferencia más cercana a navegación conceptual que a multiplicación matricial global.

El prototipo Rust de este repositorio materializa la primera pieza de esa hipótesis: una red binaria de agentes, una malla simplicial, propagación por eventos y relajación elástica observable en tiempo real.
