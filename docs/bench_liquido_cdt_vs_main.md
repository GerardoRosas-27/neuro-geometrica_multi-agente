# Bench: LiquidCdt (NEW) vs Main RQM

**Rama:** `exp/campo-hibrido-onda` · **Módulo:** `src/liquid_cdt_vs_main_bench.rs`  
**Fecha:** 2026-09-21 (America/Mexico_City) ·  
`cargo test --release --lib liquid_cdt_vs_main -- --nocapture`

## Protocolo

- **N = 8** conceptos; cue ≠ label (`CUE_BASE+i → target`).
- Warmup **200** queries; bench **2000** queries/brazo (misma cantidad).
- **NEW:** `LiquidCdtSystem` — inferencia siempre líquida; buffer + `sleep_consolidate` → engramas CDT; RQM sólo en sueño.
- **MAIN:** `NativeThermoRqmEprSubstrate` — `train_observed_transition` × 6 epochs + `query` en hot path.
- Tareas: identidad `i→i` y mapa desplazado `i→(i+1)%N`.

## Resultados (release, run `compare_liquid_cdt_vs_main_detailed`)

### Inferencia — identidad

| Brazo | mean µs/q | p50 | p95 | p99 | acc | qps ≈ |
|-------|----------:|----:|----:|----:|----:|------:|
| **NEW LiquidCdt** | **0.3301** | 0.3200 | 0.3830 | 0.4330 | **1.0000** | **3 029 619** |
| MAIN RQM | 1.5662 | 1.4810 | 2.1750 | 2.5760 | **1.0000** | 638 489 |

NEW ≈ **4.7×** más rápido en mean (y ~5× en throughput).

### Inferencia — mapa desplazado

| Brazo | mean µs/q | acc |
|-------|----------:|----:|
| NEW LiquidCdt | 0.3375 | **0.0000** (sesgo identidad del núcleo de ondas) |
| MAIN RQM | 1.5309 | **1.0000** |

### Train / sueño

| Brazo | batch A (0..3) | batch B (4..7) | nota |
|-------|---------------:|---------------:|------|
| NEW sleep_consolidate | **0.471 ms** | **0.368 ms** | sin train relacional para infer; RQM sólo en sueño |
| MAIN train (4 rel × 6 ep) | 0.565 ms | 0.584 ms | train identidad completo ≈ 1.234 ms |

### Memoria / olvido

| Brazo | retenidos tras A+B | recall |
|-------|-------------------:|--------|
| NEW engrams CDT | **8** | **8/8** soft-recall |
| MAIN relations | **8** | **8/8** query |

### Pureza del hot path (NEW)

| Contador durante infer (2000 q) | Valor |
|--------------------------------|------:|
| `rqm_api_calls` | **0** |
| Δ `cdt_tick` | **0** |
| Infer buffer vacío (mean µs) | 0.2927 |
| Infer **después** de sueño (mean µs) | **0.2851** (sigue líquido-rápido) |

MAIN ejecuta RQM en **cada** query (2000 llamadas en el bench).

## Tabla veredicto

| Aspecto | Ganador | Detalle |
|---------|---------|---------|
| Latencia infer identidad | **NEW** | 0.33 vs 1.57 µs/q |
| Hot-path sin RQM / sin ticks CDT | **NEW** | rqm_infer=0, Δtick=0 |
| Memoria sueño-desacoplada | **NEW** | 8 engramas; post-sleep ~0.29 µs |
| Exactitud identidad | empate | 1.0 / 1.0 |
| Mapa arbitrario (shifted) | **MAIN** | 0.0 vs 1.0 |
| Primer disparo relacional sin sueño | **MAIN** | train inmediato |

## Mejoras

La arquitectura dual **NEW** mejora el núcleo de inferencia: latencia identidad ~5× menor que RQM main, throughput ~3M q/s, y el camino caliente **no** llama a RQM ni avanza el sustrato CDT. La memoria de largo plazo se consolida en sueño (lotes 0..3 y 4..7 → 8 engramas sin olvido) sin contaminar la latencia de query (infer post-sueño sigue ~0.29 µs).

## Empates

Exactitud en continuación identidad: ambos **1.0**. Retención de los 8 conceptos tras dos lotes: ambos **8/8** (NEW vía engramas CDT + soft-recall; MAIN vía relaciones RQM).

## Peor

MAIN sigue siendo mejor para **relaciones arbitrarias** aprendidas sin sueño: el mapa `i→(i+1)%N` da acc 1.0 en RQM y 0.0 en el líquido (geometría de ondas sesgada a identidad). También ofrece aprendizaje relacional de primer disparo tras `train_observed_transition`, sin esperar consolidación onírica. Costes train A/B (~0.56 ms) son del mismo orden que sleep NEW (~0.4 ms), pero el valor de MAIN está en flexibilidad relacional, no en µs de query.

## Conclusión

Usar **LiquidCdt (NEW)** como **núcleo de inferencia** y como memoria durable vía Thermo CDT en el sueño; reservar **RQM estilo MAIN** como pegamento relacional **dentro del sueño** (como ya hace `sleep_consolidate`) o para mapas arbitrarios que el líquido no captura por geometría. No devolver RQM al hot path de query.

## Cómo reproducir

```bash
cargo test --release --lib liquid_cdt_vs_main -- --nocapture
```

Tests: `compare_liquid_cdt_vs_main_detailed` (asserts fuertes), `compare_shifted_map_reported` (números + assert suave MAIN shifted ≥ 0.90).
