# Inferencia por descarte termodinámico y consolidación CDT:
## una arquitectura cognitiva con fasores, atractores y un modelo lingüístico periférico

**Estado del manuscrito:** preprint técnico, versión 0.6
**Fecha:** 31 de agosto de 2026
**Implementación de referencia:** `cdt_rqm_epr`, Rust

**Resultado principal.** Una consolidación CDT de un patrón verificado deforma
el paisaje fasorial y amplía de forma causal la cuenca de recuperación. El
protocolo reproducible es `consolidation_basin_experiment`; el comando es
`cargo run --release --bin native_consolidation_basin_experiment`. El resto de
este manuscrito describe infraestructura, evidencia de sesión o trabajo en
curso. `docs/paper.md` es bitácora histórica, no este resultado.

**Periferia lingüística (opción C).** Gemma 2 2B no forma parte del claim de
este preprint: no se presenta como motor de razonamiento ni como resultado.
El chat circadiano es una demo de ingeniería. Las secciones 3.4, 3.5, 7.2 y
7.10 describen esa infraestructura; no alimentan la tesis de cuenca.

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

Fuera del claim de este preprint, el crate incluye un chat Gemma 2 como demo
de ingeniería. Un modelo de lenguaje grande no constituye la memoria ni el
motor de la tesis de cuenca. La
memoria se divide en una capa rápida, volátil y maleable y una capa consolidada,
protegida y versionada. El sistema no garantiza literalmente “cero olvido”;
busca olvido catastrófico acotado mediante revalidación, repetición durante
sueño, decaimiento selectivo, checkpoints y rollback transaccional.

La implementación actual acumuló 115.876 ciclos wake y 28.969 ciclos sleep. De
los ciclos wake, 115.785 fueron verificados (99,921 %) y 91 rechazados. Las
115.785 soluciones aceptadas fueron revalidadas y consolidadas sin rechazos
registrados durante sleep. En el último ciclo persistido, la energía libre
descendió de -3,474102 a -18,544695 y el residuo final fue
9,541141 × 10⁻⁴.

Además, un experimento causal pareado de 32 nodos y 144 inferencias por
condición midió directamente el paisaje antes y después de consolidar un patrón
verificado. Antes del sueño, ninguna de las entradas con 10–40 % de corrupción
alcanzó el umbral de recuperación; después del sueño, las 144 lo alcanzaron,
con exactitud media 1,0. La corrupción crítica medida pasó de 0 a 0,40 y las
iteraciones medias bajaron en todos los niveles evaluados. El resultado se
repitió en una prueba automatizada con ocho semillas fijas independientes.
Esto demuestra deformación y ampliación de cuenca dentro del fixture controlado;
no demuestra generalización conceptual ni ventaja física.

(Infraestructura, fuera del claim.) Un experimento preliminar de apagado de
capas en Gemma 2 produjo 0 rutas dispersas verificadas. No se usa como
evidencia del resultado principal.

Finalmente, se implementó un ciclo extremo a extremo donde Gemma 2 recibe una
única cue parcial, propone fronteras futuras, y el motor postselecciona por
energía libre antes de aplicar Handshake, atención y consolidación CDT. En un
benchmark sintético pareado, el recuerdo del híbrido futuro alcanzó 0,761,
0,808 y 0,645 para 64, 128 y 256 nodos, frente a 0,505, 0,520 y 0,506 con
interferencia sin frontera. Una ejecución real continua de Gemma 2 completó 98
ciclos: 83 pasaron el gate wake, 70 se consolidaron y todos se fusionaron en un
único atractor con confianza 0,987824. Entre los primeros y últimos 20 ciclos
parseables, el residuo medio descendió de 8,236 × 10⁻³ a 4,696 × 10⁻³ y la
coherencia Handshake aumentó de 0,604 a 0,886. Es evidencia del mecanismo
algorítmico bajo una entrada repetida, no de retrocausalidad física ni de
generalización semántica.

**Palabras clave:** computación termodinámica, fasores, energía libre,
atractores, CDT, consolidación, memoria de dos velocidades, ciclo circadiano,
inferencia analógica, postselección, Handshake, modelos de lenguaje.

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
- **(fuera del claim)** un chat Gemma 2 puede verbalizar; no es el motor ni
  el resultado de este preprint.

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
                              Búfer anillo (vigilia, sin flush)
                                        │
                              ciclo de sueño
                                        ▼
                        Replay, máscaras, revalidación y CTP/CDT
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

Se vacía, en la arquitectura genérica, cuando alcanza su capacidad o cuando
comienza un ciclo de sueño. En la ruta circadiana Gemma 2 el búfer es un anillo:
al saturarse expulsa la experiencia más antigua y **no** consolida rutas
durante vigilia. La consolidación, el descubrimiento de máscaras y el
entrenamiento CTP ocurren únicamente al iniciar sueño.

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

### 3.4 LLM como periférico lingüístico (infraestructura, fuera del claim)

El LLM puede cumplir dos funciones restringidas:

1. **entrada:** convertir lenguaje ambiguo en variables, restricciones y una
   receta tipada;
2. **salida:** convertir una solución estructurada y verificada en una
   explicación legible.

La solución numérica o lógica procede del motor externo. Cuando el compilador
lingüístico produce una receta inválida, se usa un parser determinista o se
solicita aclaración. Los pesos del LLM permanecen congelados.

### 3.5 Ciclo circadiano Gemma 2 + CTP (infraestructura, fuera del claim)

La implementación de chat unifica tres tiempos que antes se mezclaban en cada
turno. El binario de referencia es `native_gemma2_circadian_chat`.

```text
Vigilia
  historial + mensaje
        │
        ▼
  plan_wake_prefill  ── un solo forward Transformer
        │                 KV incremental si el prompt extiende el prefijo
        ▼
  embeddings W_emb (cola de thermo_window; sufijo si hay cache)
        │
        ▼
  un paso CTP  →  logits_from_hidden(Φ)  →  sesgo de vocabulario
        │
        ▼
  decode Gemma:  ℓ' = ℓ + α · RMS(ℓ)/RMS(b) · b
        │         α = wake_blend (0,25 por defecto)
        ▼
  observe() al anillo  +  journal de vigilia
        │
        ▼
Sueño  (/sueño o al salir)
  replay ≤ 8 prompts
        ├─ calidad ≥ 0,50 y máscara sparse → memoria de trabajo
        ├─ calidad ≥ 0,92 + spin-gate → ruta lenta verificada
        └─ basura / máscara de todas las capas → descarte
  train CTP sobre el dataset de sueño
  persistir adaptive + thermo.cdt
```

La vigilia **no** sustituye el decode de Gemma por `hybrid.generate()` y **no**
ejecuta un segundo prefill Transformer para consultar el CTP. El sesgo se
calcula una vez por turno y se aplica en cada paso de muestreo. Las máscaras
sparse descubiertas en sueño se usan en el primer turno o cuando la KV no
puede extenderse; cambiar de máscara a mitad de conversación invalidaría la
cache.

`/limpiar` vacía historial, KV y la ventana de embeddings (`reset_context`).
No borra pesos CTP, `sleep_cycles` ni rutas adaptativas persistidas.

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
- `native_gemma2_circadian_chat` para vigilia de un prefill + sesgo CTP y
  sueño como único sitio de aprendizaje;
- `native_gemma2_adaptive_chat` para telemetría adaptativa sin híbrido CTP;
- `gemma2_thermo_hybrid_llm` para embeddings, paso CTP y mezcla de logits;
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
- una ejecución registrada del binario adaptativo con Gemma 2 2B GGUF
  (26 de julio de 2026; evidencia histórica de verificación en vigilia);
- pruebas de librería del ciclo circadiano (19 de agosto de 2026), incluida una
  prueba GGUF ignorada de sesgo CTP en vigilia;
- suite automatizada de librería y del binario adaptativo.

El checkpoint fasorial fue leído en el ciclo 115.876. Las dos consultas Gemma 2
se ejecutaron el 26 de julio de 2026 sobre CPU. No se realizó aún una comparación
energética con hardware analógico. El JSON fasorial sí está versionado en este
checkout. `data/native_gemma2_adaptive/adaptive-state.json` no está versionado,
por lo que las dos consultas Gemma se clasifican como evidencia de sesión y no
pueden reproducirse sin el GGUF y el estado externo correspondiente.

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

### 7.2 Enrutamiento adaptativo de Gemma 2 (infraestructura, fuera del claim)

#### 7.2.1 Evidencia histórica: verificación en vigilia

Las dos consultas del 26 de julio de 2026 verificaban máscaras candidatas
durante el chat: un prefill completo, comparación de logits y, si fallaba,
limpieza de KV y repetición con todas las capas.

| Métrica | Resultado observado |
|---|---:|
| Consultas instrumentadas | 2 |
| Capas del modelo | 26 |
| Rutas dispersas verificadas | 0 |
| Fallbacks completos | 2 |
| Tasa de fallback | 100 % |
| Capas ejecutadas tras verificación | 26/26 |
| Velocidad de decodificación CPU observada | 2,04–2,68 tokens/s |
| Pruebas de librería aprobadas | 121 |
| Pruebas del binario adaptativo aprobadas | 2 |
| Fallos en las pruebas dirigidas | 0 |

El conteo de 121 corresponde a `cargo test --lib` después de incorporar los
experimentos de cuenca, generalización limitada, validación adversarial y
selección de familias. Al omitir incluso una capa candidata cambió el token
principal o no se alcanzó
el umbral de similitud de logits. El verificador limpió la KV cache y repitió la
inferencia con el modelo completo. Es un resultado negativo importante:
demuestra el fallback, no una mejora de eficiencia. Motivó desplazar el
aprendizaje al sueño (sección 7.10).

### 7.3 Qué demuestran y qué no demuestran los resultados

Los resultados apoyan que:

- el minimizador reduce la energía libre implementada;
- el sistema puede sostener ciclos wake/sleep y persistir deltas;
- la consolidación está condicionada a verificación;
- el entrenamiento observado es estable durante más de \(10^5\) ciclos;
- una consolidación aceptada puede modificar las fases de arista CDT y ampliar
  de forma reproducible la cuenca del patrón consolidado en un fixture sintético.

La periferia Gemma/CTP no se lista como resultado de este preprint.

Los resultados todavía no demuestran que:

- el atractor encontrado sea correcto para problemas abiertos;
- el sistema generalice fuera de la distribución;
- no exista olvido catastrófico;
- la dinámica digital supere a GPU o CPU en energía;
- un chip físico procese literalmente millones de hipótesis útiles;
- el sistema posea consciencia o cognición general;
- el sesgo CTP en vigilia mejore respuestas abiertas, ni que las máscaras
  descubiertas en sueño reduzcan capas durante un decode incremental.

### 7.4 Experimento causal de deformación del paisaje

Se añadió un protocolo explícito para evaluar la predicción central de este
trabajo. La implementación reproducible está en
`consolidation_basin_experiment::run_consolidation_basin_experiment` y el gate
ejecutable es:

```powershell
cargo test --lib consolidation_basin_experiment -- --nocapture
cargo run --release --bin native_consolidation_basin_experiment
```

El protocolo separa las siguientes fases:

1. se crea un CDT de 32 nodos y se conserva un snapshot **pre**;
2. se presenta una configuración binaria completa y verificada en memoria rápida;
3. wake sólo la deja pendiente, sin modificar el snapshot CDT;
4. sleep la revalida y consolida 128 fases de arista;
5. se generan exactamente los mismos cues, semillas y jitter para los snapshots
   pre y post;
6. se minimiza la misma energía con el mismo solver y presupuesto;
7. se mide recuperación con corrupción de 10, 20, 25, 30, 35 y 40 %.

La fase de adquisición usa acoplamiento cero de forma deliberada: aísla el
efecto causal de escribir un patrón ya verificado, sin atribuir al solver la
capacidad adicional de descubrir ese patrón. Durante la evaluación el
acoplamiento es idéntico y no nulo en pre y post. Por tanto, la variable
experimental que cambia es el paisaje persistido por sleep.

**Métrica de exactitud.** La exactitud reportada es la **exactitud directa**:
la fracción de nodos cuyo signo recuperado coincide con el objetivo. El
funcional es simétrico ante un flip global \(Z_2\), pero el cue conserva la
convención de signo mayoritaria, así que una recuperación genuina debe
respetarla; un estado completamente invertido cuenta como fallo. El reporte
incluye además `mean_gauge_invariant_accuracy` (que cuenta el flip global
como acierto) únicamente como diagnóstico: una brecha entre ambas métricas
indicaría convergencia al atractor con la convención invertida, y ninguna
decisión del gate la consume. Versiones anteriores del protocolo usaban la
variante invariante como métrica principal; su piso de 0,5 bajo azar inflaba
los valores pre y se corrigió en la versión 0.3.

Resultados de la corrida release de referencia:

| Corrupción | Éxito pre | Éxito post | Exactitud pre | Exactitud post | Iteraciones pre | Iteraciones post |
|---:|---:|---:|---:|---:|---:|---:|
| 10 % | 0/24 | 24/24 | 0,497 | 1,000 | 76,8 | 26,1 |
| 20 % | 0/24 | 24/24 | 0,499 | 1,000 | 80,2 | 27,6 |
| 25 % | 0/24 | 24/24 | 0,500 | 1,000 | 79,8 | 26,3 |
| 30 % | 0/24 | 24/24 | 0,508 | 1,000 | 77,2 | 34,3 |
| 35 % | 0/24 | 24/24 | 0,500 | 1,000 | 72,6 | 35,1 |
| 40 % | 0/24 | 24/24 | 0,500 | 1,000 | 83,9 | 45,4 |

La exactitud pre queda en el nivel del azar sin piso artificial, y la
diagnóstica gauge-invariante post coincide con la directa (1,000 en los seis
niveles): no hubo flips globales de gauge en la recuperación.

La energía final media post quedó entre \(2,29\times10^{-5}\) y
\(3,00\times10^{-5}\), frente a 0,451–1,805 pre. Estas unidades pertenecen al
funcional implementado y no son julios.

El gate exige simultáneamente:

\[
\rho_\mathrm{crítica,post} > \rho_\mathrm{crítica,pre},
\qquad
\overline{\Delta P_\mathrm{éxito}} \geq 0,10,
\]

y ausencia de caída de exactitud media en cualquiera de los niveles. En la
corrida de referencia:

```text
rho_critica_pre=0.00
rho_critica_post=0.40
ganancia_media_probabilidad_exito=1.00
decision=basin_expansion_pass
```

La prueba multisemilla repite el gate con ocho semillas deterministas. Su
objetivo es detectar dependencia accidental del grafo o del patrón; no sustituye
un intervalo de confianza sobre una distribución externa.

El 100 % de recuperación post-sueño del patrón **inyectado** es el techo
esperado del protocolo: el patrón se escribe ya verificado. No se cita como
evidencia de cognición. El reporte de cada nivel incluye ahora la desviación
típica de la exactitud directa entre ensayos (`std_accuracy`).

La tarea discriminante es un holdout que no comparte semilla de desarrollo,
usa 48 nodos, corrupción 20/45/55 %, ruido de grafo (15 % de aristas, ±0,40 rad)
y un segundo patrón **nunca consolidado**:

```powershell
cargo test --release --lib scientific_holdout -- --nocapture
```

La métrica cognitiva principal publicada para este holdout es la tasa de
éxito del patrón no inyectado (media ± desviación entre niveles). El test
exige que deje de ser 1,0. Los números concretos salen de esa corrida; no se
fijan a mano en este manuscrito.

### 7.4.1 Baselines externos (mismo fixture)

Sobre 32 nodos, las mismas cues y 300 iteraciones se comparan cuatro métodos.
Hopfield clásico y Hopfield moderno son ajenos al crate. El brazo fasorial
pre-sueño es la relajación sin consolidación CDT. Hebb escribe fases ±π en
las mismas aristas, sin gate de sueño. El comando canónico imprime la tabla
(tiempo de pared, energía del **modelo**, recuperación, saturación). Esas
unidades no son julios.

```powershell
cargo run --release --bin native_consolidation_basin_experiment
cargo test --release --lib scientific_basin_baselines -- --nocapture
```

### 7.5 Tres niveles de afirmación

Para evitar mezclar implementación, algoritmo y física, el estado de la
evidencia se clasifica así:

**Nivel 1 — evidencia interna reproducible.** La suite y los binarios demuestran
dentro de la implementación: relajación, descenso de la energía del modelo,
inferencia fasorial, memoria rápida, replay, consolidación, gates, rollback,
persistencia, entrenamiento prolongado y separación operativa entre lenguaje e
inferencia. El experimento de la sección 7.4 añade evidencia causal de que una
consolidación puede deformar el paisaje y ampliar una cuenca medida.

**Nivel 2 — hipótesis algorítmicas no resueltas.** Todavía requieren benchmarks
externos: generalización de atractores, representaciones conceptuales no
inyectadas, reducción de interferencia frente a baselines equivalentes,
superioridad sobre métodos convencionales y escalabilidad eficiente.

**Nivel 3 — hipótesis físicas no probadas.** Un dispositivo físico tendría que
implementar un funcional equivalente, converger con menor energía medida,
aprovechar paralelismo físico, mantener estabilidad bajo ruido y permitir
consolidación útil. Ningún resultado digital de este manuscrito demuestra esas
propiedades.

La secuencia de validación defendible es:

```text
algoritmo -> ventaja computacional -> escalabilidad -> mapa físico
```

### 7.6 Memoria, variación, composición y transferencia

Se añadió un protocolo cognitivo de cuatro niveles sobre el motor unificado:

1. **Memoria exacta:** cuatro relaciones \(A_i\rightarrow B_i\).
2. **Variación no vista:** las mismas relaciones se consultan con desfases de
   contexto que no aparecieron durante entrenamiento.
3. **Composición:** se aprende \(A\rightarrow B\) y \(B\rightarrow C\), y se
   exige inferir \(A\rightarrow B\rightarrow C\) sin almacenar el atajo
   \(A\rightarrow C\).
4. **Transferencia estructural:** una relación observada se transfiere a tres
   pares isomórficos de nodos nuevos. Una ablación con confianza de simetría cero
   debe impedir la transferencia.

```powershell
cargo test --lib cognitive_generalization_benchmark -- --nocapture
cargo run --release --bin native_cognitive_generalization_benchmark
```

En 24 ensayos:

```text
memoria exacta                         100%
variaciones no vistas                  100%
composición sin atajo directo          100%
ausencia del atajo                     100%
transferencia isomórfica               100%
transferencia ausente sin simetría     100%
abstención OOD                         100%
decision=limited_structural_generalization_pass
```

El nivel 2 prueba variaciones de fase, no entradas sensoriales complejas. En el
nivel 4 la órbita isomórfica se proporciona explícitamente. Por tanto, el
resultado demuestra recuperación robusta, composición y transferencia
estructural limitada; todavía no demuestra descubrimiento autónomo de
regularidades o simetrías.

### 7.7 Selección ambigua y descubrimiento de simetría

Para reducir la facilidad y determinismo de los fixtures anteriores se añadió
un protocolo adversarial:

- ramificación \(A\rightarrow B\rightarrow C\) frente a \(A\rightarrow D\);
- selección de rama según fase de contexto;
- consultas cercanas al punto equidistante, donde el sistema debe abstenerse;
- energía efectiva de cada hipótesis, definida como \(-\ln(score)\);
- topologías de 4, 8 y 12 espines;
- número variable de exposiciones;
- patrones dispersos de 8, 12 y 16 canales;
- tres ejemplos estructurales ruidosos y un outlier contradictorio;
- transferencia a un patrón heldout;
- control con dos estructuras incompatibles.

```powershell
cargo test --lib advanced_cognitive_validation -- --nocapture
cargo run --release --bin native_advanced_cognitive_validation
```

Resultados sobre 36 ensayos:

```text
selección de rama                         100%
selección de trayectoria A→B→C            100%
abstención ante ambigüedad                100%
orden correcto de energía efectiva        100%
margen medio seleccionado                 0,73957
margen medio ambiguo                      0,01653
descubrimiento de transformación          100%
transferencia a patrón heldout            100%
rechazo de estructura conflictiva         100%
decision=adversarial_selection_and_limited_symmetry_discovery_pass
```

La nueva API `query_with_ambiguity` no fuerza una respuesta cuando las dos
mejores hipótesis tienen margen insuficiente. El descubridor no recibe la
órbita ni el desplazamiento correcto: compara transformaciones candidatas,
selecciona la de menor error mediano y tolera un outlier.

La familia de simetrías candidatas —traslaciones cíclicas de canales— sí está
predefinida. En consecuencia, se demuestra descubrimiento autónomo limitado del
elemento de simetría dentro de un grupo conocido, no descubrimiento irrestricto
de cualquier invariante.

### 7.8 Descubrimiento de familia y complejidad mínima

El siguiente benchmark amplía el espacio de hipótesis a cinco familias:

```text
H1 = traslaciones 2D
H2 = rotaciones
H3 = reflexiones
H4 = permutaciones aprendidas
H5 = composiciones rotación/reflexión + traslación
```

Cada hipótesis recibe una energía:

\[
\mathcal E(H)=\operatorname{mediana}_i
\operatorname{MSE}(H(x_i),y_i)+\lambda\,C(H),
\]

donde \(C(H)\) es la complejidad descriptiva. La mediana protege contra un
outlier y el segundo término penaliza una permutación memorizadora cuando una
regla geométrica simple explica los mismos datos.

```powershell
cargo test --lib transformation_family_discovery -- --nocapture
cargo run --release --bin native_transformation_family_discovery
```

Resultados de 50 ensayos, diez por familia:

```text
identificación global de familia          100%
traslación                                100%
rotación                                  100%
reflexión                                 100%
permutación                               100%
composición                               100%
parámetros/mapping                        100%
transferencia heldout                     100%
robustez a ruido + outlier                100%
preferencia por complejidad mínima        100%
abstención con evidencia ambigua          100%
error robusto medio                       4,2485e-6
ventaja MDL media sobre memorización      1,1050e-3
margen energético medio                   8,1648e-2
decision=family_parameter_mdl_discovery_pass
```

El sistema ya no recibe la familia concreta ni sus parámetros. Descubre ambas
seleccionando entre familias candidatas y puede identificar una composición.
La limitación restante es metaestructural: el catálogo de cinco familias y el
lenguaje de composiciones todavía fueron definidos por el investigador.

### 7.9 Futuros propuestos por Gemma y postselección por energía libre

Se añadió un entrenador en el que el modelo lingüístico deja de ser sólo un
compilador periférico y asume un papel generativo acotado: recibe una única cue
presente y propone varios estados parciales futuros. Gemma no puede escribir en
CDT ni declarar una respuesta correcta. Cada propuesta se ejecuta en un clon
idéntico del motor, se minimiza el mismo funcional y sólo sobrevive el candidato
con menor energía libre normalizada. El candidato ganador todavía debe superar:

```text
gate wake -> revalidación sleep -> estabilidad -> ΔF_store -> CDT
```

Esta separación evita que una alucinación textual se convierta directamente en
memoria. El generador sólo amplía el conjunto de hipótesis; Armijo conserva la
autoridad sobre el descenso, Handshake y atención modulan la dirección, y el
filtro variacional decide la persistencia.

#### 7.9.1 Validación sintética pareada

El primer experimento utilizó ocho prototipos balanceados como vocabulario
preentrenado de un sustituto controlado de Gemma. En cada episodio el prior
recibió sólo 30 % de los nodos, con 8 % de corrupción, y generó cuatro futuros.
No recibió el índice del patrón activo ni la etiqueta de evaluación. Se
compararon tres brazos:

1. interferencia: cue como condición inicial y Armijo, sin frontera futura;
2. futuro + Armijo: propuestas y postselección por \(F\), sin moduladores;
3. futuro adaptativo: postselección, Handshake y atención.

El presupuesto máximo se igualó: la interferencia recibió tantas iteraciones
como la suma máxima de los cuatro candidatos futuros. La evaluación reconstruyó
un motor desde el CDT entrenado y presentó cues nuevas sin meta ni estímulo.

| Nodos | Interferencia | Futuro + Armijo | Futuro adaptativo | Mejora absoluta adaptativa |
|---:|---:|---:|---:|---:|
| 64 | 0,505371 | 0,710449 | **0,761475** | +0,256104 |
| 128 | 0,520264 | 0,776978 | **0,807739** | +0,287475 |
| 256 | 0,505737 | 0,602539 | **0,645264** | +0,139527 |

El híbrido futuro ganó en las tres escalas. Sin embargo, no fue gratuito. Las
evaluaciones de \(F\) fueron 36.960 frente a 11.571 en 64 nodos, 44.908 frente
a 16.501 en 128 y 56.373 frente a 21.823 en 256. El costo observado fue
2,58–3,19 veces mayor en evaluaciones, porque el baseline convergió antes de
agotar su presupuesto mientras el sistema futuro evaluó varios candidatos.
Por tanto, el resultado demuestra una ventaja de calidad bajo presupuesto
máximo pareado, no todavía una ventaja de eficiencia.

También apareció un control negativo útil: interferencia aceptó y consolidó
muchos episodios pero su recuerdo permaneció cerca de azar. El número bruto de
consolidaciones no es evidencia de aprendizaje; el endpoint primario debe ser
recuerdo fuera de muestra.

#### 7.9.2 Ejecución con Gemma 2 real

Se ejecutó `native_gemma2_future_infinite_trainer` sobre Gemma 2 cuantizado en
CPU. La entrada permaneció fija:

```text
0:+,3:-,8:+,13:-,21:+,29:-
```

La ejecución se reanudó desde checkpoints, completó 98 ciclos y terminó mediante
Ctrl+C con guardado atómico y código de salida 0. El estado final fue:

| Métrica | Resultado |
|---|---:|
| ciclos | 98 |
| futuros parseados | 212 |
| ciclos sin futuro parseable | 3 (3,06 %) |
| gates wake aprobados | 83 (84,69 %) |
| consolidaciones sleep | 70 (71,43 %) |
| consolidación condicionada al gate | 84,34 % |
| rechazos por \(\Delta F_{\text{store}}\) | 0 |
| evaluaciones de energía | 48.848 |
| igniciones atencionales | 720 |
| tokens Gemma | 11.234 |
| tiempo de decode acumulado | 1.562,21 s |
| throughput agregado | 7,19 tokens/s |

Los 70 episodios no crearon 70 recuerdos independientes. Se fusionaron en un
único atractor, como corresponde a la repetición de la misma evidencia:

```text
atractores CDT       1
reconsolidaciones    70
confianza final      0,987824
F almacenada         -6,114373
```

La comparación entre los primeros y últimos 20 ciclos parseables mostró:

| Métrica media | Primeros 20 | Últimos 20 |
|---|---:|---:|
| residuo | 0,008236 | **0,004696** |
| coherencia de fase | 0,945587 | **0,983317** |
| coherencia Handshake | 0,604373 | **0,885597** |
| \(\Phi\) | 0,085021 | 0,065038 |
| tasa de gate | 70 % | **100 %** |
| tasa de consolidación | 35 % | **100 %** |

El residuo medio cayó aproximadamente 43 % y Handshake aumentó 0,281 puntos.
La reducción tardía de \(\Phi\) no implica necesariamente pérdida de
integración: al estabilizarse la cuenca, la distribución residual requiere
menos focalización para converger. El último ciclo terminó con residuo
0,004811, Handshake 0,934026 y \(\Phi=0,041182\).

La mejor \(F\) observada fue -177,73335, pero las energías entre ciclos no son
directamente comparables porque Gemma produjo fronteras de cardinalidad y
confianza diferentes. Las tendencias defendibles son residuo, coherencia,
aceptación y recuerdo bajo evaluación común.

#### 7.9.3 Interpretación limitada

El experimento real demuestra que:

- Gemma puede producir futuros estructurados desde una cue sin recibir una
  etiqueta;
- el parser rechaza nodos inválidos y salidas no estructuradas;
- una propuesta con residuo superior al gate no se consolida;
- el mismo estado puede reanudarse, reconsolidarse y converger hacia una cuenca
  estable.

No demuestra que el futuro generado sea semánticamente correcto, que exista
retrocausalidad física ni que el atractor generalice a entradas distintas. La
entrada fue repetida y no existe todavía un conjunto holdout semántico para
esta corrida. El benchmark sintético sí mide recuerdo ciego, pero utiliza un
prior controlado con prototipos conocidos. Ambas evidencias deben mantenerse
separadas.

### 7.10 Ciclo circadiano: un prefill, aprendizaje en sueño, sesgo CTP en vigilia (fuera del claim)

El 19 de agosto de 2026 se implementó la separación de tiempos descrita en
§3.5. No sustituye el experimento histórico de §7.2.1: lo corrige como
política de ejecución.

**Fase 1 — vigilia barata.** `plan_wake_prefill` programa como máximo un
forward Transformer. Si el prompt nuevo extiende exactamente los tokens
cacheados y la máscara no cambia, sólo se calcula el sufijo. `observe()`
escribe en un anillo y no llama a `flush_fast_memory`. Una prueba GGUF
(`incremental_wake_prefill_is_faster_on_gemma2_gguf`) confirmó que el segundo
turno prefilla sólo el sufijo.

**Fase 2 — sueño como único sitio de aprendizaje.** Hasta 8 prompts (búfer
más reciente, luego journal no entrenado) se rejuegan con forward completo.
Calidad ≥ 0,50 y máscara sparse se retienen en memoria de trabajo; ≥ 0,92 más
spin-gate se enlazan como rutas lentas. Después se entrena el CTP sobre el
dataset de sueño. La prueba GGUF `sleep_discovers_masks_on_gemma2_gguf`
completó un replay. Esto no implica que existan rutas dispersas útiles en
sesiones reales: el umbral 0,92 sigue siendo estricto.

**Fase 3 — sesgo CTP sin segundo Transformer.** Tras el prefill, el híbrido
ingiere embeddings (`W_emb·√d`), ejecuta un paso termodinámico y proyecta Φ
al vocabulario con `logits_from_hidden`. El mismo vector se mezcla en cada
paso de decode:

\[
\ell'_i = \ell_i + \alpha\,\frac{\mathrm{RMS}(\ell)}{\mathrm{RMS}(b)}\,b_i,
\qquad \alpha=\texttt{wake\_blend}\in[0,1].
\]

`α = 0` deja los logits intactos. Los `thermo.cdt` antiguos sin el campo
deserializan `α = 0,25`. `/limpiar` llama a `reset_context` y conserva
`sleep_cycles`.

| Prueba | Resultado |
|---|---|
| `mix_logits` con `α = 0` | logits Gemma inalterados |
| `mix_logits` longitudes distintas | 0 mezclas |
| `mix_logits` con `α = 1` y sesgo concentrado | puede cambiar el top-1 |
| `reset_context` | conserva `sleep_cycles` y `tokens_processed` |
| `wake_blend` ausente en JSON | valor por defecto 0,25 |
| `wake_bias_from_embeddings_on_gemma2_gguf` | sesgo finito del vocabulario, \( \|\Phi\| > 0 \), mezcla cambia logits; 26 s |

Estas pruebas demuestran que el mecanismo está cableado y que el sesgo es
numéricamente no nulo sobre Gemma 2 2B GGUF. **No** demuestran mejora de
calidad lingüística, ahorro de capas en decode ni superioridad frente al chat
Gemma denso. El decode de hasta 256 tokens en CPU Q4 sigue siendo el coste
dominante; el prefill incremental reduce el TTFT del segundo turno, no el
tiempo por token generado.

---

## 8. Predicciones falsables

La hipótesis puede evaluarse con las siguientes predicciones:

1. **Descenso y exactitud.** Para problemas con mínimo conocido, un menor
   residuo y una menor energía final deben correlacionarse con mayor exactitud.
2. **Formación de atractores.** Tras consolidar una solución, cues parciales
   deben converger hacia ella desde una cuenca más amplia que antes. Esta
   predicción pasó el fixture causal de la sección 7.4; permanece pendiente en
   distribuciones externas y tareas conceptuales.
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
6. Ya existe una ruta extrema a extremo Gemma 2 → futuros → energía libre →
   CDT, pero la corrida real repite una sola entrada y carece de corpus
   semántico independiente.
7. No se ha fabricado ni simulado a nivel de circuito un chip termodinámico.
8. No se ha demostrado ventaja asintótica ni energética frente a hardware
   convencional.
9. El experimento de cuenca consolida una configuración proporcionada y
   verificada; no demuestra que el sistema descubra autónomamente conceptos.
10. La corrida de referencia usa un patrón, un tamaño y un conjunto finito de
    corrupciones. La repetición en ocho semillas reduce fragilidad del fixture,
    pero no establece generalización fuera de distribución.
11. El prior sintético de futuros conoce un codebook de prototipos. No recibe la
    etiqueta activa, pero su alta tasa top-1 representa un generador competente
    y no prueba que Gemma alcance esa calidad en tareas abiertas.
12. Las energías de ciclos con fronteras distintas no son comparables de forma
    absoluta; deben normalizarse por cardinalidad o evaluarse bajo una frontera
    común antes de inferir progreso a partir de \(F\).
13. El ciclo circadiano está cubierto por pruebas de política y una prueba GGUF
    de sesgo; no hay evaluación ciega de calidad de chat ni ablación de
    `wake_blend`.

---

## 11. Experimentos futuros

### Fase A: validación algorítmica

- problemas SAT, QUBO, planificación y recuperación asociativa con óptimo
  conocido;
- comparación contra búsqueda exhaustiva, simulated annealing y gradiente;
- ablación de fase, amplitud, entropía, CDT y sueño;
- extensión de la medición de cuencas ya implementada a múltiples tamaños,
  familias de patrones, baselines y mínimos espurios.

### Fase B: aprendizaje continuo

- secuencias de tareas permutadas;
- retención antes y después de cada consolidación;
- comparación con replay, EWC y adapters;
- rollback automático si \(\Delta R < -\epsilon\).

### Fase C: rutas dispersas del Transformer

- distillation de la ejecución completa;
- adapters residuales para compensar capas omitidas;
- el sueño ya rejuega prompts y propone máscaras; falta medir KL, top-k y
  exactitud **después** de aplicar esas máscaras en un cache miss;
- ablación de `wake_blend` (0, 0,25, 1) sobre un conjunto de prompts fijos;
- habilitación del apagado en decode incremental sólo después de superar
  gates de calidad, porque cambiar la máscara rompe la KV cache.

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
rápida sin escribir inmediatamente sobre conocimiento protegido. En Gemma 2,
esa separación es circadiana: un prefill y un sesgo CTP de día; replay,
máscaras y entrenamiento del núcleo de noche.

La evidencia actual demuestra descenso de energía, persistencia wake/sleep,
fallback seguro y, en un experimento causal sintético, que consolidar un patrón
verificado deforma el paisaje para aumentar su cuenca de recuperación y reducir
las iteraciones. La nueva postselección futura mejora el recuerdo frente a
interferencia en tres escalas sintéticas, y la corrida con Gemma 2 demuestra que
propuestas lingüísticas imperfectas pueden ser filtradas y reconsolidadas hasta
formar una cuenca estable. El ciclo circadiano demuestra que el CTP puede
proyectarse al vocabulario de Gemma sin un segundo Transformer; no demuestra
todavía que esa proyección mejore el lenguaje, ni generalización conceptual,
ausencia de olvido, ventaja sobre baselines de capacidad equivalente, ventaja
energética, retrocausalidad ni procesamiento físico masivo. Precisamente por
ello, la propuesta se expresa como una hipótesis experimental: si un sustrato
termodinámico físico puede implementar el mismo paisaje con relajación paralela,
podría convertir una búsqueda digital costosa en evolución física eficiente. La
oportunidad no es obtener cómputo gratuito, sino utilizar de forma medible la
dinámica de la naturaleza como parte del computador.

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

