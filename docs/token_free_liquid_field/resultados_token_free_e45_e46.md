# Resultados Token-Free Liquid Field — ciclo 1 (E45, E46, paridad Rust)

Pre-registro: `preregistro_e45_e46.md` (commit 0a55cb8, anterior a toda ejecución). Semillas DEV 0,1,2 → DEV; confirmación 3,4,5 → TEST/OOD/COMPOSITION/LONG_HORIZON, **sin retocar nada** entre fases. Artefactos crudos en `artifacts/token_free/` (JSON + logs); dataset `tfl-gen-v1` con SHA-256 por split en el `manifest` de cada JSON. Checkpoints/exportaciones (`e46_*.json`, ~1 MB c/u) quedan en caché local, no versionados; sus SHA-256 están en los JSON de resultados y en `parity_all.txt`.

Entorno: JAX 0.11.2 / optax 0.2.8 / Python 3.13.5, CPU (8 núcleos). Rust release, `target-cpu=native`, 1 hilo para latencia.

**Limitación principal (flag):** el "encoder lingüístico" es un proxy aleatorio congelado, no Gemma. Los resultados no hablan aún de hidden states reales (queda para E55).

## E45 — Bottleneck: **PASS** (con la limitación anterior)

Exact-match relacional (probe lineal ∘ ridge por op, 1 paso), media de 3 semillas:

| Método | N | DEV (s0-2) | TEST (s3-5) | OOD (s3-5) | R² superficie (fuga léxica) |
|---|---|---|---|---|---|
| raw hidden | 512 | 0.942 | 0.942 | 0.933 | 0.98 |
| PCA | 64 / 256 | 0.080 / 0.940 | – | – | 0.96 / 0.98 |
| proyección aleatoria | 64 / 512 | 0.011 / 0.891 | – | – | 0.85 / 0.97 |
| encoder lineal entrenado | 64 | 0.972 | 0.970 | 0.972 | 0.06 |
| **encoder MLP entrenado** | 16 | 0.098 | 0.121 | 0.031 | 0.00 |
| | 32 | 0.625 | 0.617 | 0.316 | 0.01 |
| | **64** | **0.984** | **0.983** | **0.979** | **0.04–0.05** |
| | 128 | 0.980 | 0.979 | 0.976 | 0.35–0.40 |
| | 256 | 0.980 | 0.980 | 0.976 | 0.79 |
| | 512 | 0.952 | 0.965 | 0.950 | 0.79 |

- N\* = 64 (es el mejor de toda la tabla en DEV). En confirmación sigue siendo el mejor en TEST y OOD en las 3 semillas.
- Un Ψ de 64 dims entrenado supera al hidden raw de 512 (+4 pp) y descarta casi toda la información de superficie; PCA/aleatorio conservan la superficie y pierden la relación.
- Ruptura clara por debajo de 64 (32 → 0.62; en OOD 0.32).

## E46 — Liquid vs MLP vs SSM vs lineal (N=64, ~33.2k parámetros cada uno)

Exact-match por rollout puro en Ψ (decoder lineal solo al final):

DEV (semillas 0-2):

| Modelo | DEV h1 | DEV h4 | LONG h16 | LONG h64 | Eje 1 (DEV h4+LONG h16)/2 | Amplif. perturb. h16 | ‖J‖₂ medio |
|---|---|---|---|---|---|---|---|
| MLP recurrente | 0.964 | 0.904 | 0.528 | 0.023 | 0.716 | 0.52 | 1.93 |
| **Liquid (LTC-like)** | 0.906 | 0.681 | 0.121 | 0.000 | 0.401 | **1.65** | 2.07 |
| SSM selectivo | 0.970 | 0.933 | 0.746 | 0.261 | 0.840 | 0.50 | 1.98 |
| Lineal local | 0.978 | 0.953 | 0.722 | 0.000 | 0.838 | 0.56 | 1.81 |

Confirmación (semillas 3-5, TEST/OOD/COMP/LONG):

| Modelo | TEST h1 | TEST h4 | OOD h4 | COMP h4 | LONG h16 | LONG h64 | Eje 1 (OOD+COMP+LONG16)/3 | Amplif. h16 |
|---|---|---|---|---|---|---|---|---|
| MLP | 0.966 | 0.903 | 0.834 | 0.873 | 0.472 | 0.013 | 0.727 | 0.50 |
| **Liquid** | 0.902 | 0.692 | 0.601 | 0.519 | 0.100 | 0.001 | **0.407** | **1.75** |
| SSM | 0.969 | 0.936 | 0.915 | 0.898 | 0.766 | **0.317** | 0.859 | 0.52 |
| Lineal local | **0.977** | **0.955** | **0.944** | **0.953** | 0.689 | 0.000 | **0.862** | 0.58 |

(Valores por semilla en `e46_dev.json` / `e46_conf.json`; las tres semillas tienen el mismo orden en todas las métricas principales.)

Nota: en la fase DEV el eje 1 usa DEV h4 + LONG h16 porque las semillas DEV no se evaluaron en OOD/COMP (para no mirarlos antes de confirmación). Desviación menor documentada.

**Veredicto E46: Liquid `FAIL`.** Pierde contra todos los competidores en el eje 1 por 32–45 pp y es el único que amplifica perturbaciones (×1.75 vs ×0.5–0.58). En DEV y confirmación con la misma conclusión.
- Ganadores en esta configuración: **lineal local** (mejor en h≤4, OOD, composición, y 8× más rápido) y **SSM selectivo** (único con horizonte largo útil: 0.77 a h16, 0.32 a h64). En el eje 1 pre-registrado empatan (0.862 vs 0.859). Ninguno cumple "≥2 pp sobre cada rival" frente al otro.
- Interpretación honesta: la tarea es lineal en el espacio one-hot (permutaciones), así que un mapa lineal por op es casi el modelo correcto; esto favorece a D y limita lo que E46 puede decir sobre dinámicas no lineales. La variante líquida con Δt=1 y Euler explícito es inestable (‖J‖≈2.07); no se re-tuneó (pre-registro). Una variante con Δt menor/varios sub-pasos sería un **nuevo** experimento pre-registrado, no un rescate.
- Mismo patrón que v4 (E33): error de 1 paso del 2–10 % se compone en horizontes largos.

## Exportación y paridad Python→Rust: **PORT_PARITY_PASS** (24/24)

Corpus: 512 inicios de DEV × 16 ops (semilla 9), nunca TEST. Core Rust `src/token_free_field.rs` (schema `tfl-v1`), runner `src/bin/run_token_free_field.rs`.

| Modelo | max abs paso 16 (peor semilla) | match discreto paso 1 / 16 | µs/paso p50 / p95 / p99 (mediana 6 semillas) | decoder p50 µs |
|---|---|---|---|---|
| MLP | 1.4e-5 | 100 % / 100 % | 3.50 / 4.51 / 4.87 | 0.53 |
| Liquid | 1.7e-4 | 100 % / 100 % | 3.60 / 4.48 / 4.81 | 0.53 |
| SSM | 1.5e-5 | 100 % / 100 % | 3.64 / 4.61 / 5.03 | 0.53 |
| Lineal | 3.5e-5 | 100 % / 100 % | **0.45 / 0.54 / 0.56** | 0.62 |

Todas las tolerancias pre-registradas se cumplen (paso 1: max ≤1e-4, media ≤1e-5; paso 16: max ≤1e-3, p99 ≤1e-4). Liquid es el que más deriva (orden de suma + exp/softplus) pero dentro de tolerancia. Latencia del encoder no medida en Rust (proxy sintético en Python) → coste end-to-end pendiente.

## Auditoría (E54 estructural, no el kill-test completo)

La API del core solo acepta `FieldState(Vec<f32>)` y `FieldContext{op}`; `token_ids_seen_by_core=0`, `llm_calls_after_encoder=0` en los 24 runs. El decoder es lineal (3.9k parámetros), sin tabla/NN/lookup. No hay memoria episódica en este ciclo.

## Estado de la batería

| Exp | Estado |
|---|---|
| E45 | PASS (encoder proxy) |
| E46 | FAIL para Liquid; lineal/SSM mejores |
| Paridad | PORT_PARITY_PASS |
| E47–E56 | NOT_RUN |
| E57 | NOT_RUN (requiere v4 cerrada) |

## Recomendación

No mergear. La hipótesis "dinámica líquida" no se sostiene en su primera prueba; el siguiente paso defendible (según el orden obligatorio) es E47 sparse/E48 stopping sobre los ganadores (SSM/lineal), y una tarea con transformaciones no lineales + encoder real (Gemma hidden) antes de reabrir Liquid.
