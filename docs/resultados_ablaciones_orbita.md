# Resultados de ablaciones y extrapolación — ÓRBITA

**Protocolo:** `docs/ablaciones_orbita.md` (`a6be441`). **Implementación registrada antes de correr:** `docs/preregistro_ablaciones_orbita_impl.md` (`5753636`). **Estado (2026-10-14 15:30 CDMX):** D completo (DEV + CONFIRM); A, B1 y C en entrenamiento (en curso; se añadirán aquí al terminar, sin modificar el protocolo).

## Experimento D — extrapolación fuera del horizonte entrenado (completo)

Sin entrenamiento nuevo: `orbita extrap` sobre los pesos publicados de ÓRBITA (32 seeds; multi-horizonte de entrenamiento sólo k ∈ {1,2,4,8}) y los checkpoints publicados de v4 E3. TEST iterable (rotate+reflect, 64 objetos por seed), free-run hasta h = 256, incluye horizontes no potencia de dos. MLP M3 no tiene checkpoint publicado → no evaluado. Auditoría: 0 queries en las 64 ejecuciones (`artifacts/orbita/ablations/D_extrap/queries_*.txt`). Consistencia: los valores en h = 1…64 coinciden con las tablas E33 publicadas (p. ej. h16 DEV 0.313 / 0.264).

Datos: `artifacts/orbita/ablations/D_extrap/extrap_{dev,confirm}.csv` (por seed, modelo y h); resumen generado por `python3 orbita/analyze_ablations.py D` → `artifacts/orbita/ablations/D_extrap/summary.md`. "acc teacher" = teacher consolidado iterado (el teacher es malo en reflect, por eso su acc es baja ya en h1). Colapso = estado no finito o norma fuera de [0.1, 10]×. Divergencia vs teacher = rel_err mediano entre trayectoria del modelo y del teacher.



## DEV (n=16)

| h | acc ÓRBITA | acc E3 | acc teacher | rel_err med ÓRBITA | rel_err med E3 | norma ÓRBITA | norma E3 | colapso ÓRBITA | colapso E3 | div. teacher ÓRBITA | div. teacher E3 | Δacc ÓRBITA−E3: media | mediana | sd | IC95 | seeds +/− |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 0.984 | 0.992 | 0.571 | 0.006 | 0.007 | 0.999 | 0.999 | 0.000 | 0.000 | 0.020 | 0.021 | -0.0078 | +0.0000 | 0.0332 | [-0.0254, +0.0059] | 2/3 |
| 2 | 0.984 | 0.985 | 0.536 | 0.009 | 0.011 | 0.998 | 0.998 | 0.000 | 0.000 | 0.029 | 0.029 | -0.0010 | +0.0000 | 0.0300 | [-0.0176, +0.0108] | 4/1 |
| 3 | 0.961 | 0.973 | 0.499 | 0.014 | 0.016 | 0.996 | 0.996 | 0.000 | 0.000 | 0.050 | 0.049 | -0.0117 | +0.0000 | 0.0440 | [-0.0361, +0.0049] | 4/6 |
| 4 | 0.942 | 0.925 | 0.438 | 0.018 | 0.021 | 0.995 | 0.995 | 0.000 | 0.000 | 0.058 | 0.058 | +0.0176 | +0.0234 | 0.0612 | [-0.0147, +0.0459] | 11/3 |
| 6 | 0.834 | 0.782 | 0.321 | 0.027 | 0.031 | 0.993 | 0.993 | 0.000 | 0.000 | 0.087 | 0.088 | +0.0518 | +0.0625 | 0.0929 | [+0.0068, +0.0976] | 11/3 |
| 8 | 0.706 | 0.632 | 0.243 | 0.036 | 0.041 | 0.991 | 0.991 | 0.000 | 0.000 | 0.114 | 0.116 | +0.0742 | +0.0781 | 0.1165 | [+0.0195, +0.1328] | 11/4 |
| 12 | 0.480 | 0.409 | 0.148 | 0.054 | 0.061 | 0.987 | 0.987 | 0.000 | 0.000 | 0.170 | 0.172 | +0.0713 | +0.0704 | 0.1038 | [+0.0225, +0.1230] | 12/3 |
| 16 | 0.313 | 0.264 | 0.098 | 0.071 | 0.081 | 0.982 | 0.983 | 0.000 | 0.000 | 0.220 | 0.227 | +0.0488 | +0.0078 | 0.1139 | [-0.0029, +0.1065] | 8/7 |
| 24 | 0.137 | 0.110 | 0.042 | 0.106 | 0.121 | 0.974 | 0.974 | 0.000 | 0.001 | 0.305 | 0.320 | +0.0264 | +0.0156 | 0.0755 | [-0.0098, +0.0635] | 9/5 |
| 32 | 0.079 | 0.050 | 0.023 | 0.142 | 0.159 | 0.966 | 0.967 | 0.001 | 0.003 | 0.379 | 0.398 | +0.0293 | +0.0235 | 0.0637 | [-0.0010, +0.0615] | 9/5 |
| 48 | 0.029 | 0.013 | 0.011 | 0.211 | 0.235 | 0.952 | 0.956 | 0.002 | 0.005 | 0.503 | 0.528 | +0.0166 | +0.0156 | 0.0347 | [-0.0000, +0.0342] | 10/4 |
| 64 | 0.017 | 0.005 | 0.007 | 0.277 | 0.307 | 0.939 | 0.947 | 0.002 | 0.007 | 0.601 | 0.611 | +0.0117 | +0.0156 | 0.0231 | [+0.0010, +0.0234] | 9/2 |
| 96 | 0.004 | 0.003 | 0.002 | 0.407 | 0.442 | 0.920 | 0.931 | 0.006 | 0.019 | 0.732 | 0.737 | +0.0010 | +0.0000 | 0.0103 | [-0.0039, +0.0058] | 4/3 |
| 128 | 0.000 | 0.001 | 0.001 | 0.523 | 0.575 | 0.902 | 0.920 | 0.008 | 0.043 | 0.823 | 0.844 | -0.0010 | +0.0000 | 0.0038 | [-0.0029, +0.0000] | 0/1 |
| 192 | 0.000 | 0.000 | 0.000 | 0.730 | 0.815 | 0.888 | 0.907 | 0.017 | 0.083 | 0.935 | 0.952 | +0.0000 | +0.0000 | 0.0000 | [+0.0000, +0.0000] | 0/0 |
| 256 | 0.000 | 0.000 | 0.000 | 0.932 | 1.026 | 0.880 | 0.904 | 0.027 | 0.112 | 1.056 | 1.057 | +0.0000 | +0.0000 | 0.0000 | [+0.0000, +0.0000] | 0/0 |

rel_err mediano h128, ÓRBITA − E3 (negativo = ÓRBITA mejor): media | mediana | sd | IC95 | seeds +/−: -0.0526 | -0.0479 | 0.1002 | [-0.1031, -0.0048] | 4/12

rel_err mediano h256, ÓRBITA − E3 (negativo = ÓRBITA mejor): media | mediana | sd | IC95 | seeds +/−: -0.0943 | -0.0922 | 0.1277 | [-0.1563, -0.0297] | 4/12

## CONFIRM (n=16)

| h | acc ÓRBITA | acc E3 | acc teacher | rel_err med ÓRBITA | rel_err med E3 | norma ÓRBITA | norma E3 | colapso ÓRBITA | colapso E3 | div. teacher ÓRBITA | div. teacher E3 | Δacc ÓRBITA−E3: media | mediana | sd | IC95 | seeds +/− |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 0.988 | 0.991 | 0.595 | 0.006 | 0.007 | 0.999 | 0.999 | 0.000 | 0.000 | 0.018 | 0.019 | -0.0029 | +0.0000 | 0.0363 | [-0.0225, +0.0117] | 4/3 |
| 2 | 0.982 | 0.978 | 0.560 | 0.009 | 0.012 | 0.998 | 0.998 | 0.000 | 0.000 | 0.025 | 0.026 | +0.0049 | +0.0000 | 0.0455 | [-0.0176, +0.0264] | 5/3 |
| 3 | 0.953 | 0.938 | 0.477 | 0.014 | 0.019 | 0.997 | 0.997 | 0.000 | 0.000 | 0.041 | 0.043 | +0.0147 | +0.0078 | 0.0794 | [-0.0224, +0.0557] | 8/5 |
| 4 | 0.912 | 0.869 | 0.432 | 0.018 | 0.024 | 0.995 | 0.996 | 0.000 | 0.000 | 0.049 | 0.050 | +0.0430 | +0.0313 | 0.0883 | [-0.0010, +0.0850] | 11/4 |
| 6 | 0.805 | 0.689 | 0.347 | 0.027 | 0.036 | 0.993 | 0.994 | 0.000 | 0.000 | 0.073 | 0.074 | +0.1152 | +0.1016 | 0.1319 | [+0.0508, +0.1807] | 13/3 |
| 8 | 0.687 | 0.551 | 0.275 | 0.036 | 0.048 | 0.991 | 0.992 | 0.000 | 0.000 | 0.097 | 0.099 | +0.1357 | +0.1250 | 0.1226 | [+0.0752, +0.1963] | 14/2 |
| 12 | 0.482 | 0.349 | 0.177 | 0.053 | 0.071 | 0.987 | 0.989 | 0.000 | 0.000 | 0.143 | 0.148 | +0.1338 | +0.1250 | 0.0898 | [+0.0908, +0.1777] | 15/1 |
| 16 | 0.323 | 0.217 | 0.114 | 0.070 | 0.094 | 0.983 | 0.986 | 0.000 | 0.000 | 0.188 | 0.196 | +0.1065 | +0.0859 | 0.0852 | [+0.0664, +0.1504] | 15/0 |
| 24 | 0.156 | 0.095 | 0.050 | 0.104 | 0.140 | 0.975 | 0.979 | 0.000 | 0.002 | 0.271 | 0.289 | +0.0615 | +0.0547 | 0.0612 | [+0.0322, +0.0918] | 14/1 |
| 32 | 0.073 | 0.044 | 0.020 | 0.136 | 0.187 | 0.969 | 0.971 | 0.000 | 0.002 | 0.348 | 0.373 | +0.0293 | +0.0313 | 0.0487 | [+0.0059, +0.0537] | 9/3 |
| 48 | 0.023 | 0.013 | 0.007 | 0.202 | 0.275 | 0.956 | 0.961 | 0.001 | 0.004 | 0.472 | 0.516 | +0.0107 | +0.0000 | 0.0239 | [+0.0000, +0.0225] | 7/4 |
| 64 | 0.008 | 0.008 | 0.004 | 0.266 | 0.360 | 0.945 | 0.952 | 0.004 | 0.015 | 0.569 | 0.632 | +0.0000 | +0.0000 | 0.0135 | [-0.0068, +0.0068] | 3/3 |
| 96 | 0.001 | 0.002 | 0.004 | 0.389 | 0.527 | 0.927 | 0.940 | 0.006 | 0.047 | 0.725 | 0.768 | -0.0010 | +0.0000 | 0.0067 | [-0.0039, +0.0019] | 1/2 |
| 128 | 0.000 | 0.001 | 0.002 | 0.504 | 0.676 | 0.914 | 0.931 | 0.009 | 0.081 | 0.834 | 0.858 | -0.0010 | +0.0000 | 0.0038 | [-0.0029, +0.0000] | 0/1 |
| 192 | 0.000 | 0.000 | 0.001 | 0.713 | 0.920 | 0.904 | 0.920 | 0.018 | 0.135 | 0.967 | 0.993 | +0.0000 | +0.0000 | 0.0000 | [+0.0000, +0.0000] | 0/0 |
| 256 | 0.000 | 0.000 | 0.001 | 0.888 | 1.072 | 0.899 | 0.918 | 0.025 | 0.170 | 1.029 | 1.135 | +0.0000 | +0.0000 | 0.0000 | [+0.0000, +0.0000] | 0/0 |

rel_err mediano h128, ÓRBITA − E3 (negativo = ÓRBITA mejor): media | mediana | sd | IC95 | seeds +/−: -0.1714 | -0.1356 | 0.1431 | [-0.2452, -0.1056] | 1/15

rel_err mediano h256, ÓRBITA − E3 (negativo = ÓRBITA mejor): media | mediana | sd | IC95 | seeds +/−: -0.1844 | -0.1501 | 0.1945 | [-0.2834, -0.0936] | 2/14

### Lectura de D (descriptiva; el protocolo no fija un criterio binario)
- **Accuracy (<5 %)**: ÓRBITA > E3 con IC95 > 0 en h = 6, 8, 12 en ambas fases y además en h = 16, 24, 32 en CONFIRM (y h = 64 en DEV). Ambos modelos llegan a acc ≈ 0 desde h ≈ 96; no hay régimen útil de accuracy en 128/256 para ninguno (extrapolación débil en el sentido de §10 del protocolo).
- **Tasa de degradación (error relativo)**: el error crece más lento en ÓRBITA. En los horizontes nunca vistos 128 y 256, rel_err mediano de ÓRBITA − E3 tiene IC95 < 0 en ambas fases (DEV h128 [−0.1031, −0.0048], h256 [−0.1563, −0.0297]; CONFIRM h128 [−0.2452, −0.1056], h256 [−0.2834, −0.0936]).
- **Colapso**: ÓRBITA colapsa menos a horizontes largos (h256: 0.027 / 0.025 frente a 0.112 / 0.170 de E3, DEV / CONFIRM); ambos encogen la norma (≈0.88–0.92 en h256).
- **Horizontes no potencia de dos** (3, 6, 12, 24, 48, 96, 192): siguen la misma curva que los potencia de dos (p. ej. h6 y h12 con ventaja de ÓRBITA en ambas fases), sin picos en 2,4,8 que sugieran memorización de los puntos de entrenamiento multi-horizonte.

## Experimentos A (corrector-only), B1 (K libre), C (sin corrector)
En curso. Se ejecutan con `orbita/run_ablation.sh A|B|C dev|confirm` (cadena DEV A→B→C, luego CONFIRM A→B→C), resultados en `artifacts/orbita/ablations/<variante>/{dev,confirm}/` y tabla con `python3 orbita/analyze_ablations.py ABC dev|confirm`. Parámetros: A 1702, B1 3302, C 2602 nominales (1674 efectivos).
