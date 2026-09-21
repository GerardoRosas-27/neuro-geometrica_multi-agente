# Eval completa: train + infer + CDT (campo-híbrido) vs RQM `main`

**Rama:** `exp/campo-hibrido-onda` · `src/field_hybrid_full_eval.rs`  
**Fecha:** 2026-09-21 · `cargo test --release --lib field_hybrid_full_eval -- --nocapture`

## Protocolo

1. **Train encoder** (60 pasos) → campo fasorial  
2. **Train lingüístico** (80 pasos, sonda Gemma-shaped congelada)  
3. **run_cycle** sustrato (handshake / recon)  
4. **Híbrido:** destilar conceptos (onda→RQM) + cold (wave+CDT) + warm (RQM+CDT)  
5. **Main:** solo `NativeThermoRqmEprSubstrate` cue→label (sin campo)

## Resultados

### Entrenamiento campo

| Métrica | Antes | Después |
|---------|------:|--------:|
| Encoder recon | 1.369 | 2.319 |
| Encoder same / diff sim | — | **0.809 / 0.534** |
| Lingu cluster | 0.556 | **0.667** |
| Lingu decode | 1.000 | **1.000** |
| Field cycle handshake | — | **0.951** |
| Field cycle recon | — | **0.287** |

### Inferencia + consolidación

| Brazo | acc | µs/q | notas |
|-------|----:|-----:|-------|
| Hybrid **cold** (wave+CDT) | **1.000** | **0.85** | handshake medio 1.000 |
| Hybrid **warm** (rqm+CDT) | **1.000** | 9.37 | incluye write+handshake+Hebb |
| **Main RQM** only | **1.000** | **1.26** | sin consolidación de campo |

`tokens_in_field = false` en todo el ciclo híbrido.

### Comparación

- Exactitud empatada (1.0) en los tres caminos de inferencia.
- **Main gana latencia** en warm (1.26 µs vs 9.37 µs) porque no consolida CDT.
- **Híbrido cold** (0.85 µs) es competitivo y además escribe fasores + handshake.
- El valor del híbrido no es batir µs de RQM solo, sino **descubrir con ondas → recordar con RQM → consolidar en CDT geométrico sin tokens**.

```bash
cargo test --release --lib field_hybrid_full_eval -- --nocapture
```
