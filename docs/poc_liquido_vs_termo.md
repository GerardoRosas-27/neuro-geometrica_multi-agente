# POC: líquido (ondas) vs termo CDT — núcleo de inferencia

**Rama:** `exp/campo-hibrido-onda` · `src/poc_liquid_vs_thermo.rs`  
**Fecha:** 2026-09-21 · `cargo test --release --lib poc_liquid_vs_thermo -- --nocapture`

## Pregunta

Como **NÚCLEO de inferencia**, ¿qué es más eficiente?

| Brazo | Motor |
|-------|--------|
| Líquido | `WavePredictCore::predict_from_observation` (interferencia analítica, sin rejilla NLS) |
| Termo | `NativeThermoCdtSubstrate` (motor fasorial termodinámico CDT) |

**Tarea idéntica:** N=8 continuación identidad (obs → predecir el mismo contenido entre 8 candidatos).

## Números (release, una corrida del test)

| Brazo | µs/query (media) | Exactitud | Notas |
|-------|------------------|-----------|--------|
| **WavePredictCore (líquido)** | **0.2405** | **1.0000** | 640 queries; sin train |
| **NativeThermoCdt (small POC)** | **62.3546** | **1.0000** | 2×48, deg 3×1, T=0.2, seed `0x70C0`, 8 `step()`; train_once ≈ 0.480 ms |
| NativeThermoCdt (mid ~4×80) | 213.8543 | 1.0000 | ballpark “motor completo”; 64 queries; train_once ≈ 1.655 ms |

Relación de latencia (small): líquido ≈ **259×** más rápido que termo small  
(62.35 / 0.24 ≈ 259).

## Cómo se midió el termo

- Conceptos → conjuntos **disjuntos** de nodos.
- Train (una vez): por concepto, soft-reset → `inject_pilot_pattern` (amp=1.2, fase=`concept·τ/N`) → 8× `step()` → plantilla = concat(amp, cos φ, sin φ) de nodos propios.
- Infer: soft-reset → piloto del cue → 8× `step()` → similitud coseno de la firma del cue vs todas las plantillas → argmax.

## Veredicto

**Para eficiencia de núcleo de inferencia puro → gana el líquido (WavePredictCore).**  
Thermo CDT es para **memoria aprendida durable / dinámica**, no el core de query más rápido.

El líquido resuelve la continuación por interferencia analítica en fracciones de microsegundo. El CDT termodinámico alcanza la misma exactitud en esta tarea de identidad, pero cada query paga varios `step()` sobre el grafo — útil como sustrato de consolidación y memoria, no como motor de latencia mínima.
