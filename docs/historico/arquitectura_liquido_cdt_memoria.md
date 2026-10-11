# Arquitectura de dos secciones: Líquido + Thermo CDT (memoria)

**Rama:** `exp/campo-hibrido-onda` · **Módulo fuente de verdad:** `src/liquid_cdt_memory.rs`  
**Fecha:** 2026-09-21

## Diagrama textual

```
                    ┌──────────────────────────────────────┐
  observación ──►   │  SECCIÓN 1 — LÍQUIDO (inferencia)    │
  + candidatos      │  LiquidInfer / WavePredictCore       │
                    │  interferencia pasado × futuro       │
                    │  frío y caliente: SIEMPRE aquí       │
                    └──────────────┬───────────────────────┘
                                   │ score alto
                                   ▼
                          wake_buffer (episodios)
                                   │
                         sleep_consolidate()
                    ┌──────────────┴───────────────────────┐
                    │  SUEÑO (no es query)                 │
                    │  1) encode_engram → Thermo CDT       │
                    │  2) RQM train_relation cue→label     │
                    │     (pegamento relacional, solo aquí)│
                    └──────────────┬───────────────────────┘
                                   ▼
                    ┌──────────────────────────────────────┐
                    │  SECCIÓN 2 — THERMO CDT (memoria)    │
                    │  CdtConsolidatedMemory               │
                    │  NativeThermoCdtSubstrate + engramas │
                    │  aprendizaje incremental (sin wipe)  │
                    └──────────────────────────────────────┘
```

## Roles

| Pieza | Rol |
|-------|-----|
| **Líquido** (`WavePredictCore`) | **Toda** la inferencia. Camino caliente de query. |
| **Thermo CDT** (`NativeThermoCdtSubstrate`) | Memoria dinámica consolidada **después** del sueño. |
| **RQM** (`NativeThermoRqmEprSubstrate`) | Pegamento relacional cue→label **solo en el sueño**. No es router de query. |

## Inferencia = líquido

`LiquidCdtSystem::infer` llama únicamente a `LiquidInfer::infer` → `WavePredictCore`.  
No avanza el tick del sustrato CDT ni llama a la API RQM (`used_memory=false`, `used_rqm=false`).

## Memoria = CDT tras el sueño

`sleep_consolidate`:

1. Toma episodios del `wake_buffer` con score ≥ umbral.
2. `memory.encode_engram(predicted)` — plantilla en nodos **disjuntos**; **no** reconstruye el sustrato; **no** borra engramas previos.
3. `rqm.train_observed_transition(cue, label)` — destilación relacional.
4. Vacía el buffer.

Soft-`recall` existe para tests/introspección; **no** es la inferencia primaria.

## RQM = pegamento en el sueño (no núcleo de query)

En diseños anteriores (`HybridWaveRqm`, docs de campo híbrido) el cue entrenado iba a RQM en query.  
**Eso queda supersedido como routing de inferencia:** el warm path ya no pasa por RQM en query.  
RQM permanece como escritura relacional durante la consolidación onírica.

## Por qué encaja con el POC

Del POC `docs/poc_liquido_vs_termo.md` (release):

| Brazo | µs/query | Exactitud |
|-------|----------|-----------|
| WavePredictCore (líquido) | **~0.24 µs** | 1.0 |
| NativeThermoCdt (small 2×48) | **~62 µs** | 1.0 |

El líquido gana ~259× en latencia de núcleo. Por eso la query es líquida y el CDT se reserva para memoria durable consolidada en sueño (coste en ms, no en el hot path).

## Tests

```bash
cargo test --lib liquid_cdt_memory -- --nocapture
```

- `liquid_only_for_inference` — tick CDT y contador RQM no cambian en infer.
- `sleep_stores_in_cdt_without_forgetting` — lotes 0..3 luego 4..7; 8 engramas; líquido acc=1.0.
- `rqm_only_during_sleep` — `rqm_api_calls` solo crece en `sleep_consolidate`.
- `bench_liquid_infer_vs_sleep_cost` — µs líquido vs ms sueño.

## Híbrido fusionado (siguiente paso)

Para mapas arbitrarios sin perder el núcleo líquido, ver **`docs/hibrido_liquido_cdt_rqm_fuse.md`** (`src/liquid_cdt_rqm_fuse.rs`): líquido primero + CDT en sueño + RQM solo como índice/fallback relacional.
