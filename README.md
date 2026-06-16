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

## Qué Simula

El programa abre una ventana con una malla triangulada. Cada vértice es un agente binario; cada arista es una restricción elástica; cada triángulo es un símplice de coherencia de orden superior.

La inferencia ocurre en ciclos:

```text
spikes -> activacion binaria -> tension geometrica -> relajacion -> atractor estable
```

## Controles

- `Espacio`: pausar o reanudar.
- `Click izquierdo`: estimular el agente más cercano.
- `T`: inyectar un patrón textual de ejemplo.
- `R`: reiniciar la red.
- `+` / `-`: zoom.
- Flechas: mover cámara.

## Estructura

- `src/geometry.rs`: vectores 2D y operaciones físicas básicas.
- `src/simplicial.rs`: agentes, aristas, símplices, spikes y energía libre.
- `src/render.rs`: motor gráfico 2D con `macroquad`.
- `src/main.rs`: bucle principal, entradas y ejecución de la simulación.
- `docs/paper.md`: descripción académica y técnica de la arquitectura.

## Estado del Prototipo

Esta es una primera versión experimental. Todavía no incluye encoder semántico real, geometría hiperbólica, símplices 3D ni adaptador LLM de salida. El objetivo actual es demostrar la dinámica central: computación esparsa por eventos y aprendizaje como relajación geométrica local.
