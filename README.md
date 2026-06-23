# SNGA: Sistema Neuro-Geometrico de Agentes

Prototipo Rust para explorar una arquitectura neuro-geométrica multi-agente basada en complejos simpliciales, propagación binaria por eventos y minimización local de energía libre.

El paper técnico ampliado está en [`docs/paper.md`](docs/paper.md).

Tesis central: SNGA no busca ser "un transformer sin matrices". El lenguaje es una interfaz periférica; el núcleo de memoria y razonamiento vive en una malla geométrica esparsa que activa rutas, minimiza tensión y selecciona caminos útiles.

## Ejecutar

Requisitos:

- Rust estable.
- Cargo.

Comando:

```powershell
cargo run
```

Experimento sin ventana para medir aprendizaje multimodal sintético:

```powershell
cargo run --bin experiment
```

Experimento grande con miles de conceptos e inhibición lateral:

```powershell
cargo run --bin large_experiment
```

Experimento avanzado con plasticidad rica, replay, causalidad y geometría 3D:

```powershell
cargo run --bin advanced_experiment
```

Experimento de razonamiento topológico sin atajos entrenados:

```powershell
cargo run --bin reasoning_experiment
```

Experimento temporal de tokenización/lenguaje sin LLM:

```powershell
cargo run --bin language_experiment
```

Benchmark lingüístico escalado con memoria de trabajo:

```powershell
cargo run --bin scaled_language_benchmark
```

Benchmark de lenguaje autónomo, donde SNGA infiere la intención abstracta desde el prompt:

```powershell
cargo run --bin autonomous_language_benchmark
```

## Qué Simula

El programa abre una ventana con una malla triangulada. Cada vértice es un agente binario; cada arista es una restricción elástica; cada triángulo es un símplice de coherencia de orden superior.

La inferencia ocurre en ciclos:

```text
spikes -> activacion binaria -> tension geometrica -> relajacion -> atractor estable
```

La demo inicial añade un entrenamiento multimodal sintético. Los conceptos `manzana` y `roca` se codifican como rasgos separados de lenguaje, visión y audio. Al pulsar `M`, esos rasgos se coactivan y refuerzan conexiones locales. Luego `L` u `O` evocan el concepto desde el canal lingüístico para observar si reaparece parte de su vecindad multimodal.

El binario `experiment` compara evocación antes/después del entrenamiento. La métrica principal es cuántos nodos sensoriales objetivo se reactivan cuando solo se inyecta el patrón de lenguaje. También reporta precisión y fuga hacia rasgos de otros conceptos.

Resultado de referencia con 8 conceptos sintéticos y 6 épocas:

```text
resumen_antes:   recall_medio=0.0% precision_media=0.0% fuga_media=0.0%
resumen_despues: recall_medio=100.0% precision_media=68.2% fuga_media=10.9%
```

Esto muestra aprendizaje asociativo en la malla, pero no demuestra razonamiento general. La fuga residual indica que hacen falta mejores mecanismos de inhibición, separación semántica y control causal.

Resultado de referencia con `large_experiment` a escala mayor:

```text
conceptos=10000
nodos=180000
inhibicion=max_active:32 max_spikes:128 decay:0.02

resumen:
  recall_medio=100.0%
  precision_media=55.1%
  fuga_media=0.017%
  activos_max_observado=32
```

La inhibición top-k evita cascadas globales: aunque la red entrena 10,000 asociaciones, la actividad máxima observada queda limitada a 32 agentes. Esto sugiere viabilidad como memoria asociativa esparsa y evolutiva. No prueba que sea mejor que un LLM general, pero sí muestra una ruta más eficiente para almacenamiento/evocación de asociaciones multimodales.

Resultado de referencia con `advanced_experiment`:

```text
tetrahedra=374
antes:  aristas_activas=2227 asociativas=30 consolidadas=20 episodios=8 causal=50
despues: aristas_activas=2217 asociativas=20 consolidadas=20 episodios=8 causal=50
prediccion A->B: precision=100.0% recall=100.0%
prediccion B->C: precision=100.0% recall=100.0%
```

Ese experimento activa mecanismos que no usan sensores reales todavía: poda/olvido de huellas transitorias, consolidación de conexiones repetidas, replay episódico, predicción causal simple y símplices 3D/tetraédricos.

Resultado de referencia con `reasoning_experiment`:

```text
directo fuego->ruptura: recall=0.0%
transitivo fuego->ruptura: recall=100.0%
directo perro->animal: recall=0.0%
transitivo perro->animal: recall=100.0%
contradiccion frio/caliente: tension=25.000 delta_energia=100.000
```

Esto valida razonamiento topológico inicial: la red infiere rutas no entrenadas directamente y detecta contradicciones como aumento de energía libre.

Resultado de referencia con `reasoning_benchmark` y optimización de rutas tipo flujo/evaporación:

```text
causal_chains=5000
hierarchy_chains=3000
contradictions=3000

causal:
  broad_recall=100.0%
  broad_precision=4.5%
  optimized_recall=96.6%
  optimized_precision=96.7%

jerarquia:
  broad_recall=100.0%
  broad_precision=11.7%
  optimized_recall=100.0%
  optimized_precision=100.0%

contradiccion:
  tension_media=6.250
  delta_energia_medio=25.000
```

La optimización funciona como una dinámica tipo *Physarum*: explora muchas rutas, evapora rutas débiles y refuerza rutas que reducen sorpresa/llegan al objetivo. El resultado conserva casi todo el recall y aumenta fuertemente la precisión.

Resultado de referencia con `language_experiment`:

```text
train_sentences=3840
vocab=64
context_window=2
eval_next_token:
  top1=27.1%
  top3=52.9%
  top5=65.7%

eval_with_working_memory:
  top1=97.1%
  top3=98.6%
  top5=100.0%
```

Este experimento usa un tokenizador temporal de palabras y firmas contextuales n-grama compatibles con SNGA. Muestra indicios de aprendizaje lingüístico local, pero todavía no se acerca a un transformer: aprende transiciones y patrones simples, no semántica abierta.

La variante con memoria de trabajo añade una huella abstracta previa de la idea a verbalizar (`determinante/sujeto/accion/objeto/lugar`). Con esa estructura, la red puede ordenar la salida antes de hablarla y verbalizar frases sintéticas completas.

Resultado de referencia con `scaled_language_benchmark`:

```text
train_sentences=19220
vocab=75
nodes=92400

eval_with_working_memory:
  top1=69.0%
  top3=82.1%
  top5=85.7%

dialogue_coherence:
  cases=10
  coherent=10
  score=100.0%
```

Esto muestra comunicacion coherente en un dominio pequeño y controlado usando memoria de trabajo abstracta. No equivale a un LLM general, pero valida que SNGA puede renderizar respuestas linguisticas consistentes cuando la idea ya esta organizada.

Resultado de referencia con `autonomous_language_benchmark`:

```text
intents=16
vocab=148
nodes=186000
intent_accuracy=89.6%
response_coherence=89.6%
```

Este benchmark ya no recibe el plan abstracto manualmente: usa un filtrado semántico simple del prompt y rutas SNGA para inferir la intención interna antes de responder.

## Controles

- `Espacio`: pausar o reanudar.
- `Click izquierdo`: estimular el agente más cercano.
- `M`: entrenar conceptos multimodales sintéticos.
- `L`: evocar `manzana` desde lenguaje.
- `O`: evocar `roca` desde lenguaje.
- `T`: inyectar un patrón textual de ejemplo.
- `R`: reiniciar la red.
- `+` / `-`: zoom.
- Flechas: mover cámara.

## Estructura

- `src/geometry.rs`: vectores 2D y operaciones físicas básicas.
- `src/multimodal.rs`: encoder sintético de lenguaje, visión y audio para la prueba de grounding.
- `src/simplicial.rs`: agentes, aristas, símplices, spikes y energía libre.
- `src/render.rs`: motor gráfico 2D con `macroquad`.
- `src/main.rs`: bucle principal, entradas y ejecución de la simulación.
- `src/bin/advanced_experiment.rs`: validación de plasticidad avanzada, replay, causalidad y geometría 3D.
- `src/bin/reasoning_experiment.rs`: validación de inferencia transitiva y contradicción energética.
- `src/bin/language_experiment.rs`: tokenizador temporal y prueba de predicción de siguiente token.
- `docs/paper.md`: descripción académica y técnica de la arquitectura.

## Estado del Prototipo

Esta es una primera versión experimental. Todavía no incluye encoder semántico real, geometría hiperbólica, símplices 3D ni adaptador LLM de salida. El objetivo actual es demostrar la dinámica central: computación esparsa por eventos, coactivación multimodal inicial y aprendizaje como relajación geométrica local.
