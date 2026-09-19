# Bench: colapso→RQM (exp) vs RQM directo (estilo `main`)

**Rama:** `exp/fluido-3d-astra`  
**Módulo:** `src/spin_fluid_rqm_bench.rs`  
**Fecha:** 2026-09-19 (release, misma máquina box)

## Protocolo

| | |
|--|--|
| Tarea | mapa identidad 4 etiquetas |
| Motor RQM | `NativeThermoRqmEprSubstrate` (idéntico) |
| Térmico | 2 slices × ≥128 nodos, misma config |
| Train | 6 repeticiones × 4 pares |
| Infer | 50 loops × 4 = **200** queries/brazo |
| Warmup | 1 train + 1 query antes de cronometrar |

- **main_rqm_direct:** cue = nodo `MAIN_CUE_BASE+i` → label `i` (sin spin; RQM no acepta source==target).
- **exp_spin_collapse_rqm:** entrada → NLS enfocante 16³ → features → mismo RQM.

```bash
cargo test --release --lib spin_fluid_rqm_bench -- --nocapture
```

## Resultados

| brazo | train_ms | infer_ms | µs/query | accuracy | relations |
|-------|---------:|---------:|---------:|---------:|----------:|
| main_rqm_direct | 0.574 | 0.418 | 2.1 | **1.000** | 4 |
| exp_spin_collapse_rqm | 70.000 | 569.475 | 2847.4 | **1.000** | 12 |

**Slowdown:** train ×**122** · infer ×**1362**

## Lectura

- Exactitud empatada (1.0): el colapso no gana precisión en este mapa trivial.
- El coste lo come el NLS 16³ (∼28 pasos/query), no el RQM (∼2 µs).
- El brazo spin crea más relaciones (12 vs 4) porque cada colapso aporta 3 features.
- Útil si quieres **código dinámico** (interferencia / colapso); no si quieres latencia de `main`.

## No-claims

No es un bench del crate completo de `main` ni de Gemma/cuenca; solo RQM nativo directo vs pipeline spin→RQM en esta rama.
