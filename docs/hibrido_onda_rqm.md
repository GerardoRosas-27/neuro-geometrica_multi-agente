# Híbrido: líquido de ondas (nuevo) + RQM (entrenado)

**Rama:** `exp/hibrido-onda-rqm` (aislada desde `main`, no apilada en fluido-3d)  
**Módulos:** `wave_predict_core.rs`, `hybrid_wave_rqm_infer.rs`

## Diseño

| Situación | Motor | Por qué |
|-----------|-------|---------|
| Cue **nuevo** (frío) | `WavePredictCore` | interferencia pasado×futuro, ~0.3 µs, sin train previo |
| Cue **ya entrenado** | `NativeThermoRqmEprSubstrate` | relaciones consolidadas en RQM de `main` |

Tras una predicción por ondas con `score ≥ distill_min_score`, se **destila** `cue→label` en RQM (`auto_distill`).

```rust
let mut h = HybridWaveRqm::new(8);
let cold = h.infer(3);           // WaveNew → destila
let warm = h.infer(3);           // RqmTrained
let only_new = h.infer_new(1);   // fuerza ondas
let only_rqm = h.infer_trained(3); // Some si ya destilado
```

## Resultados (`cargo test --release --lib hybrid_wave_rqm_infer`)

| Test | Resultado |
|------|-----------|
| frío→onda luego cálido→RQM | pred 3→3, distilled |
| pretrain identidad | 8/8 por RQM |
| bench | cold wave **0.30 µs** acc 1.0 · warm RQM **1.50 µs** acc 1.0 |

5/5 + wave_predict 5/5 OK.

## Nota

El camino warm (RQM) es más lento que el cold (ondas) en µs/query — es esperado: RQM aporta memoria relacional consolidada, no latencia mínima. El híbrido usa ondas para **descubrir** y RQM para **recordar**.
