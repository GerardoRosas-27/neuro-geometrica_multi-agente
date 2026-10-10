# Resultados Token-Free — ciclo 2: E46b (Gemma real, tarea no lineal), E47, E48 y H2H vs v4

Pre-registro: `preregistro_ciclo2.md` (commit anterior a toda ejecución de este ciclo). DEV → confirmación sin retune. Artefactos crudos en `artifacts/token_free/` (`e46b_*`, `e46b_parity.json`, `h2h_*`, `speedup.json`, `gemma_v2.sha256`).

## 1. Herramientas y speedup

Elegido: **JAX 0.11.2 + Optax 0.2.8** (los modelos son funciones puras con rollout; bucle de entrenamiento compilado con `lax.scan` y semillas en paralelo con `vmap`; una sola semántica que portar a Rust). Keras/TF no aportaban nada aquí.

| Configuración (E46 SSM N=64, 3000 pasos) | s/semilla | speedup |
|---|---|---|
| bucle Python + `jit` por paso (ciclo 1) | 14.3 | 1.0× |
| bucle compilado `lax.scan` | 11.5 | 1.24× |
| `lax.scan` + `vmap` 3 semillas | 8.5 | **1.67×** |

Medido con la CPU compartida (extracción de Gemma en paralelo); orden de magnitud modesto: el cuello real era la extracción Gemma (3000 textos, 2749 s) y el encoder de E45 (512-D), no el bucle. Con la caché de Ψ (3000×64) por semilla, E46b entrena cada dinámica en ~6 s/semilla (14 s `liquid_ss`).

## 2. E46b — hidden real de Gemma, tarea no lineal (`tfl-gen-v2`)

Gemma-2-2B-it Q3_K_L, media del hidden final (2304), 3000 textos (1000 estados × 3 paráfrasis en español con números en palabras). FieldEncoder 2304→256→64 (CE de decodificación converge a ~0 en TRAIN). N=64 fijo. ~33k parámetros por modelo (lineal 25k: tamaño fijo por op).

DEV (semillas 0-2; eje 1 = media(DEV h4, LONG h16)):

| Modelo | DEV h1 | DEV h4 | LONG h16 | amplif. h16 | Eje 1 |
|---|---|---|---|---|---|
| **mlp** | 0.850 | 0.563 | 0.117 | 3.01 | **0.340** |
| v4res | 0.854 | 0.561 | 0.105 | 1.98 | 0.333 |
| ssm | 0.833 | 0.515 | 0.079 | 1.49 | 0.297 |
| liquid (Δt=1) | 0.771 | 0.448 | 0.063 | 5.30 | 0.256 |
| **liquid_ss (K=4)** | 0.673 | 0.365 | 0.052 | 3.83 | **0.209** |
| linear | 0.630 | 0.258 | 0.024 | 0.31 | 0.141 |

Confirmación (semillas 3-5; eje 1 = media(OOD h4, COMP h4, LONG h16)):

| Modelo | TEST h1 | TEST h4 | OOD h4 | COMP h4 | LONG h16 | amplif. | Eje 1 | Rust µs/paso p50 | Paridad |
|---|---|---|---|---|---|---|---|---|---|
| **mlp** | 0.864 | 0.578 | 0.291 | 0.473 | 0.130 | 1.61 | **0.298** | 10.6 | FAIL estricto* |
| v4res | 0.855 | 0.560 | 0.309 | 0.462 | 0.101 | 1.54 | 0.291 | 10.4 | PASS |
| ssm | 0.844 | 0.541 | 0.279 | 0.452 | 0.106 | 1.47 | 0.279 | 11.0 | FAIL estricto* |
| liquid | 0.788 | 0.451 | 0.256 | 0.351 | 0.057 | 4.26 | 0.221 | 11.2 | PASS |
| **liquid_ss** | 0.693 | 0.390 | 0.188 | 0.307 | 0.048 | 7.57 | **0.181** | 44.9 | FAIL estricto* |
| linear | 0.653 | 0.268 | 0.194 | 0.164 | 0.023 | 0.33 | 0.127 | 1.4 | PASS |

\*Paridad (36 modelos, corpus DEV 256×16): paso 1 dentro de tolerancia en todos; decisiones discretas 100 % idénticas en paso 1 y 16 en todos. En paso 16 el error máximo absoluto excede 1e-3 en mlp (1.2e-3), ssm (2.6e-3) y liquid_ss (1.1e-2, y p99 3.1e-4 > 1e-4). Según la regla pre-registrada esto es `PORT_PARITY_FAIL` para esas variantes en horizonte 16 (deriva f32 amplificada por dinámicas expansivas: amplificación >1). Las métricas de calidad de la tabla son del prototipo JAX; para estas tres variantes no se atribuyen al core Rust más allá de h=1.

**Veredicto E46b: `liquid_ss` FAIL** (último salvo lineal en ambas fases; los sub-pasos empeoran respecto a Δt=1 y cuadruplican la latencia). No hay un ganador claro: mlp/v4res/ssm están a ≤2 pp entre sí. En la tarea no lineal el lineal cae al último lugar (al revés que en E46). Ninguna dinámica sostiene horizonte largo (LONG h16 ≤ 0.13).

## 3. E47 / E48

`PENDIENTE` (en ejecución; se actualizará esta sección con DEV y confirmación).

## 4. H2H token-free vs v4 (tarea sellada de v4, E35-style)

v4 `run_e35_core` @`1da52e7` sin cambios en un worktree desacoplado (`v4_bridge/h2h_v4.rs`); reproduce exactamente los números publicados de v4 (A 0.2314/0.2480, B 0.3386/0.3774). Token-free: Ψ = objeto completo (12), contexto v4 (10), ≈960 parámetros, mismo presupuesto (8000 pasos, batch 16, lr 3e-3), evaluación oficial en Rust desde parámetros exportados (proceso nuevo, almacén borrado). 16 semillas por fase. Accuracy = rel_err < 5 %.

| Fase | Sistema | acc h1 | rollout fam. iterables h1 / h2 / h4 | comp. 2 pasos | µs/consulta p50 | parámetros | entrenamiento | consultas a memoria en TEST |
|---|---|---|---|---|---|---|---|---|
| DEV | v4 Dφ A | 0.231 | 0.127 / 0.005 / 0 | 0.025 | 9.2† | 962 | 3.7 s (A+B) | 0 |
| DEV | **v4 Dφ B (CDT consolidado)** | **0.339** | 0.239 / 0.009 / 0 | **0.059** | 9.2† | 962 | (incl.) | 0 |
| DEV | TF mlp A / B | 0.000 / 0.000 | 0 / 0 / 0 | 0.000 | 0.40 | 957 | 0.4 s | 0 |
| DEV | TF liquid_ss A / B | 0.000 / 0.000 | 0 / 0 / 0 | 0.000 | 2.07 | 953 | 0.8 s | 0 |
| DEV | TF ssm A / B | 0.0002 / 0.0005 | 0 / 0 / 0 | 0.000 | 0.48 | 953 | 0.5 s | 0 |
| CONF | v4 Dφ A | 0.248 | 0.126 / 0.010 / 0.001 | 0.030 | 9.3† | 962 | 3.8 s | 0 |
| CONF | **v4 Dφ B** | **0.377** | 0.205 / 0.015 / 0 | **0.059** | 9.3† | 962 | (incl.) | 0 |
| CONF | TF mlp A / B | 0.000 / 0.000 | 0 / 0 / 0 | 0.000 | 0.40 | 957 | 0.4 s | 0 |
| CONF | TF liquid_ss A / B | 0.000 / 0.000 | 0 / 0 / 0 | 0.000 | 2.07 | 953 | 0.8 s | 0 |
| CONF | TF ssm A / B | 0.0002 / 0.0010 | 0.001 / 0 / 0 | 0.000 | 0.48 | 953 | 0.4 s | 0 |

TF_ssm_B − v4_B (mejor TF elegido en DEV): **−0.338** [IC95 −0.382, −0.293] DEV; **−0.377** [−0.421, −0.329] CONF. Criterio "token-free supera a v4" (≥ +0.05, IC>0): **FAIL**, claramente.

† Latencias no comparables directamente: v4 se mide en su propio harness (f64, asigna vectores por punto) y con la CPU ocupada por la extracción Gemma; token-free en el runner `tfl-v1` (f32). Los modelos token-free son ~20× más rápidos por consulta, pero con precisión nula esa ventaja no tiene valor.

Lectura honesta:
- Con el presupuesto de v4 (192 ejemplos, 8000×16 muestras) los modelos token-free sobre el objeto completo ni siquiera ajustan TRAIN (rel_err mediano ≈0.20 en TRAIN para mlp, frente a 0.05 requerido). v4 gana por su sesgo inductivo (Dφ por punto con pesos compartidos ⇒ 6× más muestras efectivas y equivariancia), y B añade la consolidación estructurada (que, como ya documentó v4, es destilación de un teacher de reglas).
- La "consolidación por lotes" del almacén episódico de token-free no añade nada (el almacén = TRAIN), tal como se pre-registró.
- Paridad H2H: paso 1 ≤1.3e-6 en las 192 ejecuciones; a h=64 las trayectorias divergen (normas hasta 1e26) y la paridad absoluta deja de tener sentido ⇒ `PORT_PARITY_FAIL` a h=64, irrelevante dado acc=0.
- 0 consultas a memoria en TEST en ambos sistemas (v4 auditado por contadores; token-free por construcción del tipo).

## 5. Estado

| Exp | Estado |
|---|---|
| E46b liquid_ss | FAIL |
| E46b paridad | PASS (v4res, liquid, linear); PORT_PARITY_FAIL a h16 (mlp, ssm, liquid_ss), discreto 100 % |
| E47, E48 | pendiente |
| H2H vs v4 | v4 gana (token-free FAIL) |

## 6. Recomendación

No mergear. La línea "liquid" falla en las dos tareas (sintética lineal y Gemma no lineal) y token-free pierde contra v4 en su propia tarea. Lo único positivo replicado es E45 (bottleneck de 64 dims) y la infraestructura (core Rust sin tokens, paridad, exportación).
