# Híbrido fusionado: Líquido + CDT + índice RQM

**Rama:** `exp/campo-hibrido-onda` · **Fuente de verdad:** `src/liquid_cdt_rqm_fuse.rs`  
**Fecha:** 2026-09-21 (America/Mexico_City)

## Qué se tomó de cada lado

| Origen | Pieza | Rol en el FUSE |
|--------|-------|----------------|
| **NEW LiquidCdt** | `WavePredictCore` / `LiquidInfer` | Núcleo primario de inferencia (~0.3 µs identidad). Siempre se intenta primero. |
| **NEW LiquidCdt** | `CdtConsolidatedMemory` | Memoria dinámica consolidada **tras el sueño** (engramas incrementales, sin olvido catastrófico). |
| **MAIN RQM** | `NativeThermoRqmEprSubstrate` | **Índice relacional** para mapas arbitrarios / desplazados (`teach_relation`). Escrito en sueño; leído en infer **solo** como fallback. |

## Arquitectura (3 piezas cooperativas)

```
obs + candidatos
        │
        ▼
┌───────────────────────┐
│ 1. LiquidInfer        │  ← siempre primero
│    WavePredictCore    │
└──────────┬────────────┘
           │ score alto y sin relación entrenada → respuesta Liquid
           │ cue ∈ relational_cues  o  score bajo →
           ▼
┌───────────────────────┐
│ 3. RqmRelationalIndex │  ← fallback (MAIN)
│    query cue→label    │
└───────────────────────┘

wake_buffer (observe / teach_relation)
        │
        ▼ sleep_consolidate()
┌───────────────────────┐
│ 2. CdtConsolidated    │  encode_engram (sin wipe)
│    Memory             │
│ + RQM train_observed  │  reafirma relaciones
└───────────────────────┘
```

## Reglas de enrutado

1. **Siempre** ejecutar el líquido primero.
2. Ir a **`RqmFallback`** si:
   - el cue está en `relational_cues` (enseñado con `teach_relation`), **o**
   - `liquid_score < liquid_min_score`,
   - **y** el índice RQM tiene el cue (o `force_rqm`).
3. En caso contrario → **`InferRoute::Liquid`**.
4. **Identidad sin `teach_relation`:** RQM **no** es el default (ruta Liquid, RQM≈0).
5. **Mapa arbitrario** tras `teach_relation` + sueño: ruta `RqmFallback` (fuerza MAIN, acc≥0.99).
6. Cue espacial: `cue_id = CUE_BASE + obs` (mismo patrón que benches; evita auto-relación).

## API

```rust
FusedLiquidCdt::infer(obs, candidates) -> FuseReport
FusedLiquidCdt::teach_relation(cue, label)
FusedLiquidCdt::observe(obs, candidates)
FusedLiquidCdt::sleep_consolidate() -> SleepReport
```

## Números (release, `fuse_beats_pure_arms`, 2026-09-21 CT)

Protocolo: N=8 · warmup=200 · bench=2000 · MAIN epochs=6.

| Brazo | ID mean µs | ID acc | ID Liquid% | SH mean µs | SH acc | SH RQM% | Engrams |
|-------|-----------:|-------:|-----------:|-----------:|-------:|--------:|--------:|
| **FUSE** | **0.2808** | **1.0000** | **100%** | 2.3342 | **1.0000** | **100%** | 8 |
| Liquid-only | 0.2737 | 1.0000 | 100% | 0.3370 | **0.0000** | 0% | 0 |
| MAIN RQM | 1.3975 | 1.0000 | 0% | 1.8431 | 1.0000 | 100% | 0 |

- Identidad FUSE ≈ **5.0×** más rápida que MAIN (0.28 vs 1.40 µs); ruta 100 % Liquid, RQM=0.
- Shifted FUSE = **1.0** (vía RQM) vs líquido puro **0.0**.
- Sueño CDT: 2 lotes → 8 engramas retenidos (`fuse_sleep_cdt_no_forgetting`).

Assert clave: `fuse.identity_mean_us < main.identity_mean_us * 1.2` y `fuse.shifted_acc ≫ liquid.shifted_acc`.

## Recomendación

Usar **FUSE** como arquitectura por defecto cuando se necesiten **ambos**:

- latencia de identidad de nivel líquido, **y**
- aprendizaje de relaciones arbitrarias (MAIN),

sin contaminar el hot path de identidad con RQM. El Thermo CDT sigue siendo la memoria durable post-sueño.

```bash
cargo test --release --lib liquid_cdt_rqm_fuse -- --nocapture
```
