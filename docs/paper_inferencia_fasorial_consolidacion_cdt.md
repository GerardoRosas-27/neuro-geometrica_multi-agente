# Inferencia por descarte termodinámico y consolidación CDT:
## una arquitectura cognitiva con fasores, atractores y un modelo lingüístico periférico

**Estado del manuscrito:** preprint técnico, versión 0.1  
**Fecha:** 26 de julio de 2026  
**Implementación de referencia:** `cdt_rqm_epr`, Rust

---

## Resumen

Se propone una arquitectura cognitiva de dos tiempos en la que la inferencia
ocurre mediante relajación de un campo termodinámico de fasores y el aprendizaje
duradero ocurre mediante consolidación selectiva sobre un sustrato de
triangulaciones dinámicas causales (CDT). La hipótesis central es que una
respuesta puede emerger sin enumerar explícitamente todas las combinaciones:
las configuraciones incompatibles interfieren, pierden amplitud o son
descartadas por el descenso de energía libre, mientras que las configuraciones
coherentes forman mínimos estables. Estos mínimos actúan como atractores
inferenciales. Tras una verificación independiente, sólo los atractores útiles
se transfieren al sustrato CDT como cambios locales de nodos, fases y aristas.

En este enfoque, un modelo de lenguaje grande no constituye la memoria ni el
motor principal de razonamiento. Sus pesos permanecen congelados y el modelo
actúa como periférico lingüístico: traduce lenguaje a una representación
operativa y expresa en lenguaje el resultado calculado por el sustrato. La
memoria se divide en una capa rápida, volátil y maleable y una capa consolidada,
protegida y versionada. El sistema no garantiza literalmente “cero olvido”;
busca olvido catastrófico acotado mediante revalidación, repetición durante
sueño, decaimiento selectivo, checkpoints y rollback transaccional.

La implementación actual acumuló 115.876 ciclos wake y 28.969 ciclos sleep. De
los ciclos wake, 115.785 fueron verificados (99,921 %) y 91 rechazados. Las
115.785 soluciones aceptadas fueron revalidadas y consolidadas sin rechazos
registrados durante sleep. En el último ciclo persistido, la energía libre
descendió de -3,474102 a -18,544695 y el residuo final fue
9,541141 × 10⁻⁴. Por otra parte, un experimento preliminar de apagado de capas
en Gemma 2 produjo 0 rutas dispersas verificadas y activó fallback completo en
2 de 2 consultas. Este resultado negativo muestra que la arquitectura de
seguridad funciona, pero también que un Transformer denso no se vuelve disperso
por omisión directa de bloques: serán necesarios distillation, adapters o
entrenamiento explícito de rutas.

**Palabras clave:** computación termodinámica, fasores, energía libre,
atractores, CDT, consolidación, memoria de dos velocidades, inferencia
analógica, modelos de lenguaje.

---

## 1. Introducción

Los sistemas de aprendizaje profundo convencionales almacenan conocimiento,
procedimientos y capacidad lingüística dentro de una misma matriz de pesos. Esa
integración ofrece gran capacidad estadística, pero presenta tres problemas:

1. cada consulta activa una fracción grande del modelo;
2. modificar los pesos para aprender información nueva puede interferir con
   conocimiento previo;
3. el proceso lingüístico queda mezclado con memoria, inferencia y control.

Este trabajo estudia una separación funcional:

- **fasores para inferencia wake:** exploran implícitamente un paisaje de
  posibilidades y convergen hacia mínimos de energía libre;
- **CDT para memoria consolidada:** conserva sólo soluciones verificadas como
  modificaciones estructurales locales;
- **modelo lingüístico periférico:** compila peticiones y verbaliza respuestas,
  pero no es la fuente final de verdad ni el depósito principal de memoria.

La propuesta no presupone consciencia ni cognición general. Presenta un
mecanismo computacional medible y una hipótesis física falsable sobre cómo la
dinámica de descarte puede reducir el costo de búsqueda.

---

## 2. Hipótesis de descarte termodinámico

### 2.1 Formulación

Sea un problema con un espacio combinatorio \(\Omega\). En lugar de evaluar cada
\(\omega \in \Omega\) secuencialmente, se codifican restricciones y estímulos en
un campo complejo de fasores:

\[
z_i = r_i e^{\mathrm{i}\phi_i}, \qquad i=1,\ldots,N.
\]

La evolución minimiza una energía libre efectiva:

\[
F(\mathbf z,T)
= E_{\mathrm{acoplamiento}}
+ E_{\mathrm{radial}}
+ E_{\mathrm{confinamiento}}
+ E_{\mathrm{estímulo}}
- T S(\mathbf z).
\]

Una forma implementada del término de acoplamiento es:

\[
E_{\mathrm{acoplamiento}}
= \frac{J}{2}\sum_{(i,j)\in\mathcal E} w_{ij}
\left|z_i-e^{\mathrm{i}\theta_{ij}}z_j\right|^2.
\]

Las configuraciones que violan restricciones generan desacoplamiento de fase,
gradientes mayores o energía superior. Durante la relajación dejan de dominar
el estado. Las configuraciones mutuamente consistentes interfieren de forma
constructiva y forman cuencas de atracción.

### 2.2 Significado de “millones de posibilidades”

La hipótesis no afirma que la implementación digital actual enumere millones de
respuestas en memoria. Afirma que un campo físico puede representar de manera
implícita un espacio combinatorio mucho mayor que su número de variables. Por
ejemplo, \(N\) decisiones binarias inducen \(2^N\) configuraciones discretas,
mientras que un sistema de \(N\) fasores posee además grados de libertad
continuos de fase y amplitud.

La relajación no “prueba” cada configuración por separado. La dinámica local
integra simultáneamente restricciones superpuestas. El término *descarte*
describe la pérdida de competitividad de modos de alta energía, no la ejecución
de millones de ramas digitales independientes.

### 2.3 Hipótesis principal

> Si las restricciones de una tarea se codifican en un funcional de energía
> cuya geometría preserva la solución, la interferencia y la relajación
> eliminarán modos incompatibles y concentrarán la dinámica en atractores de
> baja energía libre. Un verificador externo podrá distinguir atractores útiles
> de mínimos espurios antes de consolidarlos.

---

## 3. Arquitectura propuesta

```text
Lenguaje o señal
      │
      ▼
Compilador periférico (LLM congelado o parser determinista)
      │ receta operacional + restricciones
      ▼
Motor fasorial termodinámico
      ├─ superpone restricciones
      ├─ descarta modos de alta energía
      └─ converge hacia uno o varios atractores
      │
      ▼
Verificador
      ├─ válido y estable ───────────────┐
      └─ inválido → rechazo/fallback     │
                                        ▼
                              Búfer de memoria rápida
                                        │
                              lleno o ciclo de sueño
                                        ▼
                        Revalidación y consolidación CDT
                                        │
                       memoria protegida y versionada
                                        │
                                        ▼
                          Recuperación para inferencia
                                        │
                                        ▼
                         LLM verbaliza el resultado
```

### 3.1 Inferencia wake

Durante wake, el sistema compila únicamente un conjunto de trabajo disperso.
El motor fasorial minimiza \(F\) mediante gradiente precondicionado y búsqueda
de línea de Armijo. Un paso sólo se acepta si reduce la energía. La salida
incluye:

- energía inicial y final;
- residuo del gradiente;
- coherencia de fase;
- número de iteraciones y evaluaciones;
- bandera de convergencia;
- solución operacional verificable.

La inferencia no modifica inmediatamente la memoria consolidada.

### 3.2 Memoria rápida

Las soluciones verificadas entran en un búfer mutable. Este búfer permite:

- corrección rápida;
- acumulación de evidencia;
- asociación entre contexto, atractor y resultado;
- descarte sin dañar conocimiento previo.

Se vacía cuando alcanza su capacidad o cuando comienza un ciclo de sueño.

### 3.3 Consolidación CDT

CDT actúa como sustrato estructural de largo plazo. Durante sleep:

1. se vuelve a resolver cada experiencia pendiente;
2. se rechazan resultados inestables;
3. se calculan deltas locales de nodos y aristas;
4. se aplican cambios con una tasa de aprendizaje limitada;
5. se crea un checkpoint transaccional;
6. ante un error se restaura el estado anterior.

La estructura consolidada almacena amplitud, fase, estado térmico, peso,
estabilidad y relaciones topológicas. El objetivo no es copiar una respuesta,
sino deformar el paisaje para que una experiencia futura converja más
fácilmente al atractor correcto.

### 3.4 LLM como periférico lingüístico

El LLM puede cumplir dos funciones restringidas:

1. **entrada:** convertir lenguaje ambiguo en variables, restricciones y una
   receta tipada;
2. **salida:** convertir una solución estructurada y verificada en una
   explicación legible.

La solución numérica o lógica procede del motor externo. Cuando el compilador
lingüístico produce una receta inválida, se usa un parser determinista o se
solicita aclaración. Los pesos del LLM permanecen congelados.

---

## 4. Aprendizaje con olvido acotado

“Aprender sin olvidar” debe entenderse como un objetivo operacional, no como una
garantía matemática absoluta. La arquitectura reduce interferencia mediante:

- separación entre memoria rápida y consolidada;
- actualizaciones locales dispersas;
- revalidación antes de cada consolidación;
- replay de experiencias durante sueño;
- protección de rutas de alta utilidad;
- decaimiento sólo para rutas poco útiles;
- checkpoints versionados;
- commit o rollback transaccional;
- pruebas periódicas de retención.

Una afirmación fuerte de ausencia de olvido requeriría demostrar:

\[
\Delta R = R_{\mathrm{después}} - R_{\mathrm{antes}} \geq -\epsilon
\]

para un conjunto retenido e independiente de tareas, después de aprender nuevas
tareas. La ejecución actual todavía no proporciona ese experimento longitudinal
controlado; por tanto, el manuscrito no afirma olvido cero.

---

## 5. Implementación de referencia

La implementación está escrita en Rust y emplea:

- `NativePhasorThermodynamicEngine` para minimización de energía libre;
- `NativeMultiOperatorCore` para compilar conjuntos de trabajo dispersos;
- `NativeThermoCdtSubstrate` como sustrato consolidado;
- `native_phasor_infinite_trainer` para ciclos wake/sleep;
- `OperatorDeltaSnapshot` para persistencia compacta;
- `native_gemma2_adaptive_chat` para telemetría y rutas de activación seguras;
- un gate spin exacto pequeño durante consolidación, fuera de la ruta por token.

El minimizador fasorial usa:

- warm start topológico;
- precondicionador de Jacobi;
- búsqueda de línea de Armijo;
- tolerancia de energía;
- tolerancia del residuo;
- aceptación monotónica de pasos.

El trainer infinito genera recetas fasoriales por concepto, resuelve cada
receta durante wake, acumula las verificadas y las revalida durante sleep. El
checkpoint persiste por separado estado global y deltas estructurales.

---

## 6. Metodología y métricas

### 6.1 Fuente de datos

Las métricas observadas proceden de:

- `data/native_phasor_infinite_training/latest.state.json`;
- `data/native_gemma2_adaptive/adaptive-state.json`;
- ejecuciones reales del binario adaptativo con Gemma 2 2B GGUF;
- suite automatizada de librería y del binario adaptativo.

El checkpoint fasorial fue leído en el ciclo 115.876. Las dos consultas Gemma 2
se ejecutaron el 26 de julio de 2026 sobre CPU. No se realizó aún una comparación
energética con hardware analógico.

### 6.2 Métricas

**Aceptación wake**

\[
A_\mathrm{wake}
= \frac{N_\mathrm{aceptado}}{N_\mathrm{wake}}.
\]

**Descenso de energía libre**

\[
\Delta F = F_\mathrm{final}-F_\mathrm{inicial}.
\]

**Residuo**

\[
r = \sqrt{\frac{1}{N}\sum_i
\left\|\frac{\partial F}{\partial z_i}\right\|^2}.
\]

**Retención sleep**

\[
R_\mathrm{sleep}
= \frac{N_\mathrm{revalidado}}{
N_\mathrm{revalidado}+N_\mathrm{rechazado}}.
\]

**Tasa de fallback lingüístico**

\[
B_\mathrm{fallback}
= \frac{N_\mathrm{fallback}}{N_\mathrm{consultas}}.
\]

---

## 7. Resultados

### 7.1 Entrenamiento fasorial wake/sleep

| Métrica | Resultado observado |
|---|---:|
| Ciclos totales | 115.876 |
| Ciclos wake | 115.876 |
| Ciclos sleep | 28.969 |
| Wake aceptados | 115.785 |
| Wake rechazados | 91 |
| Tasa de aceptación wake | 99,921 % |
| Experiencias consolidadas | 115.785 |
| Rechazos durante sleep | 0 |
| Retención de pendientes revalidados | 100 % |
| Última energía inicial | -3,474102 |
| Última energía final | -18,544695 |
| Descenso absoluto \(\Delta F\) | -15,070593 |
| Mejor energía final observada | -24,404709 |
| Último residuo | 9,541141 × 10⁻⁴ |

El último ciclo redujo la energía libre en 15,070593 unidades. La magnitud de la
energía final fue 5,337 veces la magnitud inicial. Esta comparación describe
descenso numérico dentro del funcional implementado; no equivale a una medición
de julios ni a eficiencia física.

La tasa sleep de 100 % se refiere únicamente a experiencias que ya habían
superado el verificador wake y fueron revalidadas por el mismo sistema. No es
una medición independiente de generalización ni demuestra ausencia de olvido.

### 7.2 Enrutamiento adaptativo de Gemma 2

| Métrica | Resultado observado |
|---|---:|
| Consultas instrumentadas | 2 |
| Capas del modelo | 26 |
| Rutas dispersas verificadas | 0 |
| Fallbacks completos | 2 |
| Tasa de fallback | 100 % |
| Capas ejecutadas tras verificación | 26/26 |
| Velocidad de decodificación CPU observada | 2,04–2,68 tokens/s |
| Pruebas de librería aprobadas | 113 |
| Pruebas del binario adaptativo aprobadas | 2 |
| Fallos en las pruebas dirigidas | 0 |

Al omitir incluso una capa candidata cambió el token principal o no se alcanzó
el umbral de similitud de logits. El verificador limpió la KV cache y repitió la
inferencia con el modelo completo. Es un resultado negativo importante:
demuestra el fallback, no una mejora de eficiencia.

### 7.3 Qué demuestran y qué no demuestran los resultados

Los resultados apoyan que:

- el minimizador reduce la energía libre implementada;
- el sistema puede sostener ciclos wake/sleep y persistir deltas;
- la consolidación está condicionada a verificación;
- el fallback evita aceptar rutas Transformer degradadas;
- el entrenamiento observado es estable durante más de \(10^5\) ciclos.

Los resultados todavía no demuestran que:

- el atractor encontrado sea correcto para problemas abiertos;
- el sistema generalice fuera de la distribución;
- no exista olvido catastrófico;
- la dinámica digital supere a GPU o CPU en energía;
- un chip físico procese literalmente millones de hipótesis útiles;
- el sistema posea consciencia o cognición general.

---

## 8. Predicciones falsables

La hipótesis puede evaluarse con las siguientes predicciones:

1. **Descenso y exactitud.** Para problemas con mínimo conocido, un menor
   residuo y una menor energía final deben correlacionarse con mayor exactitud.
2. **Formación de atractores.** Tras consolidar una solución, cues parciales
   deben converger hacia ella desde una cuenca más amplia que antes.
3. **Interferencia acotada.** Aprender una tarea nueva no debe reducir la
   retención de tareas antiguas más allá de un \(\epsilon\) predefinido.
4. **Ventaja del sueño.** Consolidar sólo experiencias revalidadas debe superar
   a consolidar todas las experiencias en retención y rechazo OOD.
5. **Ventaja física.** En hardware termodinámico, energía por solución y
   latencia deben crecer más lentamente que en una enumeración digital
   equivalente.
6. **Periferia lingüística.** Sustituir el LLM por otro compilador compatible no
   debe cambiar la solución estructural cuando ambos generan la misma receta.

Una sola de estas predicciones puede fallar sin invalidar todo el software, pero
sí restringiría la interpretación cognitiva o física del enfoque.

---

## 9. Ruta hacia un chip termodinámico

### 9.1 Objetivo

La versión física reemplazaría la integración numérica de fasores por una red
analógica de osciladores, resonadores, circuitos de fase, memristores u otro
sustrato capaz de relajar naturalmente un funcional equivalente a \(F\).

Cada variable se representaría por fase y amplitud; las aristas codificarían
acoplamientos y desfases. Después de aplicar una señal de entrada, la propia
dinámica física evolucionaría hacia un atractor. Sensores leerían el estado y
un verificador digital decidiría si consolidarlo.

### 9.2 Paralelismo físico

La ventaja teórica no sería ejecutar millones de instrucciones gratuitamente,
sino dejar que muchas restricciones actúen al mismo tiempo en el sustrato. El
tiempo de relajación podría depender de la geometría del paisaje y no del número
explícito de combinaciones.

Una demostración convincente debe medir:

- tiempo hasta convergencia;
- energía suministrada al chip;
- energía de lectura y escritura;
- costo de programación de acoplamientos;
- robustez frente a ruido y deriva;
- tasa de mínimos espurios;
- calidad frente a CPU, GPU y solvers especializados.

### 9.3 Por qué no es “procesamiento gratuito”

La naturaleza realiza la relajación, pero ningún dispositivo físico es
energéticamente gratuito. Preparar el estado, mantener acoplamientos, disipar
calor, leer la solución y borrar información tiene costo. El principio de
Landauer establece un límite para operaciones lógicamente irreversibles.

La formulación científicamente defendible es:

> Un chip termodinámico podría externalizar parte de la búsqueda a una dinámica
> física masivamente paralela y reducir energía o latencia respecto de una
> simulación digital, sin eliminar el costo termodinámico de computar.

---

## 10. Limitaciones

1. Las métricas wake/sleep proceden de un trainer sintético y no de un benchmark
   cognitivo abierto.
2. La retención reportada no usa un conjunto histórico independiente.
3. La energía libre es una magnitud del modelo, no energía física medida.
4. Los atractores pueden ser mínimos locales espurios.
5. El LLM periférico todavía puede introducir errores al compilar lenguaje.
6. La arquitectura fasorial/CDT y el chat Gemma 2 adaptativo existen como
   componentes implementados, pero falta un benchmark integral extremo a
   extremo con corpus independiente.
7. No se ha fabricado ni simulado a nivel de circuito un chip termodinámico.
8. No se ha demostrado ventaja asintótica ni energética frente a hardware
   convencional.

---

## 11. Experimentos futuros

### Fase A: validación algorítmica

- problemas SAT, QUBO, planificación y recuperación asociativa con óptimo
  conocido;
- comparación contra búsqueda exhaustiva, simulated annealing y gradiente;
- ablación de fase, amplitud, entropía, CDT y sueño;
- medición de cuencas de atracción y mínimos espurios.

### Fase B: aprendizaje continuo

- secuencias de tareas permutadas;
- retención antes y después de cada consolidación;
- comparación con replay, EWC y adapters;
- rollback automático si \(\Delta R < -\epsilon\).

### Fase C: rutas dispersas del Transformer

- distillation de la ejecución completa;
- adapters residuales para compensar capas omitidas;
- entrenamiento de máscaras por tarea;
- evaluación de KL, top-k, exactitud final, latencia y consumo;
- habilitación del apagado sólo después de superar gates de calidad.

### Fase D: prototipo físico

- red pequeña de osciladores acoplados;
- calibración del funcional físico contra el digital;
- medición en julios por solución;
- escalado de 16 a \(10^3\), \(10^5\) y eventualmente \(10^6\) grados de
  libertad efectivos;
- comparación de tiempo de relajación y tasa de error.

---

## 12. Conclusión

Este trabajo propone separar inferencia, consolidación y lenguaje. Los fasores
realizan búsqueda por relajación y descarte de modos incompatibles; CDT conserva
únicamente atractores verificados; el LLM traduce entre lenguaje y
representaciones operativas. La memoria de dos velocidades permite plasticidad
rápida sin escribir inmediatamente sobre conocimiento protegido.

La evidencia actual demuestra descenso de energía, persistencia wake/sleep y
fallback seguro, pero no demuestra todavía ausencia de olvido, ventaja
energética ni procesamiento físico masivo. Precisamente por ello, la propuesta
se expresa como una hipótesis experimental: si un sustrato termodinámico físico
puede implementar el mismo paisaje con relajación paralela, podría convertir
una búsqueda digital costosa en evolución física eficiente. La oportunidad no
es obtener cómputo gratuito, sino utilizar de forma medible la dinámica de la
naturaleza como parte del computador.

---

## Referencias

1. J. J. Hopfield, “Neural networks and physical systems with emergent
   collective computational abilities”, *Proceedings of the National Academy
   of Sciences*, 1982.
2. S. Kirkpatrick, C. D. Gelatt y M. P. Vecchi, “Optimization by Simulated
   Annealing”, *Science*, 1983.
3. Y. Kuramoto, *Chemical Oscillations, Waves, and Turbulence*,
   Springer, 1984.
4. R. Landauer, “Irreversibility and Heat Generation in the Computing
   Process”, *IBM Journal of Research and Development*, 1961.
5. K. Friston, “The free-energy principle: a unified brain theory?”,
   *Nature Reviews Neuroscience*, 2010.
6. J. L. McClelland, B. L. McNaughton y R. C. O’Reilly, “Why there are
   complementary learning systems in the hippocampus and neocortex”,
   *Psychological Review*, 1995.
7. J. Snider, “Self-organized computation with unreliable, memristive
   nanodevices”, *Nanotechnology*, 2007.
8. G. E. Hinton, O. Vinyals y J. Dean, “Distilling the Knowledge in a Neural
   Network”, arXiv:1503.02531, 2015.

