# Optimización de inferencia (sin RQM) — 3 prioridades

**Rama:** `exp/fluido-3d-astra` · PR #13  
**Fecha:** 2026-09-19/20

## P1 — Solo simulador NS 3D (`fluid3d_native_infer`)

Entrada → blob de velocidad → features (E, |u|_max, COM, hist 4³) → centroides / MLP.

| Métrica | Resultado |
|---------|-----------|
| Centroid accuracy | **1.000** |
| MLP accuracy | **1.000** |
| IC-only µs/query | **~49** (vs ~3000–3700 spin collapse) |

```bash
cargo test --release --lib fluid3d_native_infer -- --nocapture
```

## P2 — Spin barato (`spin_fluid3d`)

- `SpinScratch` + `step_reuse` (sin alloc por paso)
- `SpinWindow` (integración local)
- `collapse_early` (pico estable / Δamp relativa)

| Test | Resultado |
|------|-----------|
| early exit | **4** pasos (vs max 80), exit=true |
| step_reuse ≡ step | max diff < 1e-12 |
| física previa | 4/4 OK + advección OK |

```bash
cargo test --release --lib spin_fluid3d -- --nocapture
```

## P3 — Surrogate del colapso (`spin_collapse_surrogate`)

Teacher NLS (early-exit) → MLP one-hot → features; infer = vecino L2 a targets teacher.

| Métrica | Resultado |
|---------|-----------|
| Accuracy | **1.000** |
| Surrogate µs/q | **~0.10** |
| Teacher early µs/q | **~444** |
| Speedup | **×~4500** |

```bash
cargo test --release --lib spin_collapse_surrogate -- --nocapture
```

## Ranking práctico

1. **Surrogate** o **NS IC features** → latencia tipo producción  
2. **Spin early-exit + ventana** → si aún quieres dinámica real  
3. Colapso fijo 28 pasos / RQM → solo laboratorio
