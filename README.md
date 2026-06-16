# SNGA: Sistema Neuro-Geometrico de Agentes

Prototipo Rust para explorar una arquitectura neuro-geométrica multi-agente basada en complejos simpliciales, propagación binaria por eventos y minimización local de energía libre.

El paper técnico ampliado está en [`docs/paper.md`](docs/paper.md).

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
- `docs/paper.md`: descripción académica y técnica de la arquitectura.

## Estado del Prototipo

Esta es una primera versión experimental. Todavía no incluye encoder semántico real, geometría hiperbólica, símplices 3D ni adaptador LLM de salida. El objetivo actual es demostrar la dinámica central: computación esparsa por eventos, coactivación multimodal inicial y aprendizaje como relajación geométrica local.
