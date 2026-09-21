# Bench: wave_predict_core vs RQM estilo `main`

**Fecha:** 2026-09-21 · release · `cargo test --release --lib wave_vs_rqm_bench -- --nocapture`

## Protocolo

| | |
|--|--|
| Tarea | observación → elegir continuación correcta entre 8 candidatos |
| Wave | interferencia analítica pasado×futuro (`WavePredictCore`) |
| Main | `NativeThermoRqmEprSubstrate` cue→label (sin spin) |
| Infer | 80 loops × 8 = **640** queries/brazo |
| Train RQM | 6 repeticiones × 8 pares |

## Resultados

| brazo | train_ms | infer_ms | µs/q | p50 µs | p99 µs | accuracy |
|-------|---------:|---------:|-----:|-------:|-------:|---------:|
| main_rqm_direct | 0.894 | 0.781 | **1.220** | 1.156 | 1.189 | **1.000** |
| wave_predict_core | 0.000 | 0.177 | **0.276** | 0.237 | 0.444 | **1.000** |

**Speedup inferencia wave/main: ×4.42**

## Veredicto

Para **inferencia** en esta tarea (predicción por continuación / interferencia):  
**`wave_predict_core` es más óptimo** — misma accuracy, ~4.4× más rápido, sin train.

RQM sigue siendo el motor adecuado cuando hace falta **aprendizaje relacional arbitrario** (órbitas, EPR, cues no geométricos). Para la máquina pasado×futuro por ondas, el core de interferencia gana.
