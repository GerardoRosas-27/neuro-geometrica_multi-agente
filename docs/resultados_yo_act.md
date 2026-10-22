# Resultados «el yo actúa»: re-observación adaptativa con ÓRBITA como verificador

Prerregistro `docs/preregistro_yo_act.md` (dd6724d), publicado antes de lanzar (la única ejecución previa fue una medición de tiempo sobre 0xA400
con salida descartada). Código: `orbita/src/yo.rs` (`yo-act`), `orbita/llm/yo_act.py`, `yo_act_chain.sh`. Ejecutado 2026-10-18 22:23–22:24:49 CST
(CPU 8 núcleos, ≈3,5 GB RAM disponible; ≈1,3 s por seed). Se reutilizaron rasgos/contextos de `/workspace/orbita-data/yo/` (sin regenerar Gemma).
0 queries v4 a TEST. Tablas completas: `artifacts/orbita/yo_act/analyze_{dev,confirm}.log`.

## Cifras (media [IC95], 16 seeds × 96 filas E2E_R)
| k=1 | DEV acc16 | DEV tasa | CONFIRM acc16 | CONFIRM tasa |
|---|---|---|---|---|
| BASE | 0,2630 | 0 | 0,2975 | 0 |
| ALWAYS_OBS | 0,4089 | 1 | 0,4583 | 1 |
| RANDOM_OBS | 0,3560 | 0,6452 | 0,3991 | 0,5964 |
| YO_OBS | 0,3594 | 0,6452 [0,4727, 0,7982] | 0,4082 | 0,5964 [0,4564, 0,7240] |
| SURPRISE_OBS | 0,3587 | 0,6660 | 0,4141 | 0,5859 |
| ORACLE | 0,4089 | 0,7370 | 0,4583 | 0,7025 |

k=2: ALWAYS 0,4603 / 0,5443; YO 0,4160 / 0,4870 (tasa 0,7324 / 0,7161); RANDOM 0,4075 / 0,4814; SURPRISE 0,4180 / 0,5020; ORACLE = ALWAYS.
ORACLE iguala a ALWAYS en ambas k: re-observar nunca estropea una fila que BASE acertaba; ALWAYS sólo cuesta observaciones.

## Hipótesis
| | DEV | CONFIRM | veredicto |
|---|---|---|---|
| **HA1** k=1 acc16 YO − RANDOM | +0,0034 [−0,0073, +0,0143] | +0,0091 [−0,0007, +0,0194] | **no aceptada** (IC cruza 0; equivalente dentro de δ) |
| HA1 k=2 | +0,0086 [−0,0041, +0,0218] | +0,0056 [−0,0075, +0,0178] | no aceptada (equivalente) |
| HA1 sec. acc8 YO − RANDOM, k=1 / k=2 | +0,0412 [+0,0234, +0,0607] / +0,0493 [+0,0287, +0,0729] | +0,0426 [+0,0243, +0,0614] / +0,0427 [+0,0239, +0,0632] | positivo en ambas (secundaria) |
| **HA2** k=1 acc16 YO − ALWAYS (IC inf > −0,05) | −0,0495 [−0,0938, −0,0176] | −0,0501 [−0,0729, −0,0299] | **rechazada** (aunque usa 60–65 % de las re-observaciones) |
| HA2 k=2 | −0,0443 [−0,0951, −0,0117] | −0,0573 [−0,0918, −0,0293] | rechazada |
| HA3 k=1 YO − SURPRISE | +0,0007 [−0,0013, +0,0026] | −0,0059 [−0,0202, +0,0072] | sin diferencia (equivalentes) |
| HA3 k=2 | −0,0020 [−0,0065, +0,0020] | −0,0150 [−0,0352, +0,0007] | sin diferencia |

Curva acc16 vs fracción re-observada (descriptiva, `analyze_*.log`): ordenar por p̂ o por sorpresa da curvas casi lineales entre BASE y ALWAYS,
cerca del azar a h16.

## Lectura
Re-observar y re-identificar con ÓRBITA como verificador sí sube acc16 (BASE → ALWAYS: +0,15 a +0,25). Pero el yo no elige mejor que el azar
qué filas re-observar a h16 (sí a h8, ≈+0,04), no alcanza a ALWAYS dentro de δ, y no se distingue de la sorpresa sola. Motivo probable: ~70 % de
las filas fallan en BASE (ORACLE re-observa 0,70–0,74), de modo que casi cualquier política con tasa alta acierta en las mismas filas, y a h16
muchos fallos no se arreglan re-observando (ALWAYS sigue < 0,55).
