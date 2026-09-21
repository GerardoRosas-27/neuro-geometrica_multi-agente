# Bench detallado: colapso→RQM (exp) vs RQM directo (estilo `main`)

**Rama:** `exp/fluido-3d-astra`  
**Módulo:** `src/spin_fluid_rqm_bench.rs`  
**Fecha:** 2026-09-19 · `cargo test --release --lib spin_fluid_rqm_bench -- --nocapture`  
**Resultado:** **7/7** tests OK

## 1. Head-to-head (identidad 4 labels)

Protocolo: mismo `NativeThermoRqmEprSubstrate`, 2 slices × ≥128 nodos, train 6×4 pares, infer 50×4 = 200 queries.

| brazo | train_ms | infer_ms | µs/query | accuracy | relations |
|-------|---------:|---------:|---------:|---------:|----------:|
| main_rqm_direct | 0.599 | 0.335 | **1.7** | 1.000 | 4 |
| exp_spin_collapse_rqm | 90.308 | 680.089 | **3400** | 1.000 | 12 |

**Slowdown:** train ×**151** · infer ×**2032**

## 2. Desglose de fases (spin, n=40)

| fase | µs / sample |
|------|------------:|
| encode + collapse NLS | **3679.5** |
| extract features | 110.5 |
| RQM query | 18.8 |
| RQM train (micro) | 37.3 |
| total | 3846.4 |

**Fracción colapso = 0.957** → el NLS se come ~96 % del tiempo; RQM es ruido (~0.5 %).

## 3. Latencias por query

| brazo | n | mean | p50 | p90 | p99 | min | max |
|-------|--:|-----:|----:|----:|----:|----:|----:|
| main | 200 | 14.2 µs | 13.1 | 22.1 | 30.4 | 7.6 | 57.7 |
| spin | 80 | **3707 µs** | 3695 | 3882 | 4428 | 3309 | 4446 |

## 4. Curva de train (spin, identidad)

| epoch | accuracy | relations | epoch_ms |
|------:|---------:|----------:|---------:|
| 0 | 0.00 | 0 | — |
| 1 | **1.00** | 12 | 16.2 |
| 2–6 | 1.00 | 12 | ~15–17 |

Primer epoch perfecto: **1**.

## 5. Matrices de confusión (25 reps × 4 = 100 por brazo)

Ambos: accuracy **1.000**, recall `[1,1,1,1]`, diagonal pura (sin confusiones ni abstenciones).

## 6. Escalado con `collapse_steps`

| steps | µs/infer | accuracy | amp_mean |
|------:|---------:|---------:|---------:|
| 4 | 736 | 0.750 | 0.872 |
| 12 | 1903 | 0.750 | 0.895 |
| 28 (default) | 3625 | **1.000** | 1.022 |
| 48 | 5257 | 1.000 | 1.404 |

Con pocos pasos los features espaciales a veces colisionan (acc 0.75); a 28+ el colapso separa bien las clases.

## 7. Mapa desplazado \(i \mapsto i+1\) (solo spin)

| train_ms | infer_ms | accuracy | relations |
|---------:|---------:|---------:|----------:|
| 123.1 | 427.5 | **1.000** | 12 |

## Lectura

1. Exactitud empatada con `main` en el mapa trivial.
2. Coste ≈ proporcional a pasos NLS; RQM no es el cuello.
3. Spin aprende en **1 epoch**; el valor está en el código dinámico (colapso/interferencia), no en velocidad.
4. Bajar `collapse_steps` abaratará inferencia pero puede perder separación (ver §6).

```bash
cargo test --release --lib spin_fluid_rqm_bench -- --nocapture
```
