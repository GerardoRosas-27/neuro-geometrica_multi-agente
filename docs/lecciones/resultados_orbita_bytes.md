# Resultados — ÓRBITA-Bytes (rama exp/orbita-bytes)

**Estado: cerrada sin CONFIRM por decisión del usuario 2026-10-23.**

- Solo DEV. No hay veredicto CONFIRM.
- Consultas a TEST: **0**.
- Prerregistro: `docs/prereg_orbita_bytes.md` (commit `d5c4f9c`).
- Fuente: `artifacts/orbita/bytes/summary_dev.md` (re-evaluación DEV tras reconstruir el binario `orbita`).
- Datos: `/workspace/orbita-data/bytes/`.

## Qué se comparó

Mismo núcleo ÓRBITA-v1 congelado (`w_<seed>.json`). Brazo **MAIN** = encoder de bytes (BLT + CALM + LCM). Brazo **GEMMA** = encoder Gemma + D2/D3-num. Ablaciones DEV: NOVAE, NOHASH, NONCE, FIXPATCH.

## Gate DEV — primarias (MAIN − GEMMA, δ = 0,05, 16 seeds, IC95 bootstrap pareado)

| hipótesis | media | IC95 | veredicto |
|---|---|---|---|
| **H1** familia (R) | −0.8984 | [−0.9421, −0.8522] | NO DEMOSTRADA |
| **H2** max\|p̂−p\| mediano (R) | +0.0000 | [+0.0000, +0.0000] | NO INFERIOR (sospechoso: numnorm idéntico) |
| **H3** acc h16 (R) | −0.2533 | [−0.3197, −0.1868] | NO DEMOSTRADA |
| **H6** AUROC sorpresa | +0.0000 | [+0.0000, +0.0000] | NO INFERIOR (sospechoso: AUROC idéntico 0.9868; `learn` parece no usar el encoder) |

## Lecturas absolutas (DEV)

| brazo | R fam | R acc16 | U fam | U acc16 |
|---|---|---|---|---|
| GEMMA | 0.9180 [0.8730, 0.9590] | 0.2630 [0.1953, 0.3301] | 0.4850 [0.4238, 0.5365] | 0.1374 [0.0951, 0.1810] |
| MAIN | 0.0195 [0.0046, 0.0417] | 0.0098 [0.0000, 0.0273] | 0.0241 [0.0059, 0.0469] | 0.0098 [0.0000, 0.0215] |

MAIN R fam queda por debajo del azar de 8 familias (0.125).

## Secundarias

| cantidad | media | IC95 |
|---|---|---|
| **H4** familia U MAIN − GEMMA | −0.4609 | [−0.5111, −0.4049] |
| **H5** razón latencia mediana GEMMA/BYTES | 4.47 | [4.23, 4.78] (n=32; umbral prerregistro ≥10×) |
| **H7** exactitud reconstrucción bytes (R, MAIN) | 0.4946 | [0.4776, 0.5095] (criterio 0.99; textos exactos 0.000) |
| bytes/parche en eval (R) | 1.649 | [1.6205, 1.6800] |

Memoria pico: GEMMA 1899 MB, BYTES 309 MB. Latencias medianas: GEMMA 6.970 s, BYTES 1.558 s.

## Ablaciones (familia R)

| variante | R fam | R acc16 |
|---|---|---|
| FIXPATCH | 0.1790 [0.1257, 0.2363] | 0.0449 [0.0286, 0.0638] |
| NOVAE | 0.1257 [0.0664, 0.1966] | 0.0404 [0.0150, 0.0710] |
| NOHASH | 0.0801 [0.0358, 0.1315] | 0.0215 [0.0059, 0.0410] |
| NONCE | 0.0664 [0.0312, 0.1068] | 0.0195 [0.0033, 0.0449] |
| MAIN | 0.0195 [0.0046, 0.0417] | 0.0098 [0.0000, 0.0273] |

Las ablaciones superan a MAIN; ninguna se acerca a GEMMA.

## Convergencia entrenamiento (MAIN)

- S0 (parche LM): 15/16 por paciencia.
- S1 (AE): 13/16 por paciencia; coseno de reconstrucción en val ~0.997.
- S2 (concepto): **0/16** por paciencia (todas tocan el tope de 3000 pasos). Acc familia en val de entrenamiento (hist S2): media ~0.887 (rango ~0.61–0.99).

## Incidentes de proceso

1. El worktree `/workspace/ngm-bytes` se recreó ~22:33 CDMX del 2026-10-22 y se perdió `target/`. El entrenamiento no usa el binario; las evals DEV (y el arranque de CONFIRM) fallaron por binario ausente.
2. `chain.sh` continuó a train CONFIRM aunque `analyze dev` falló. Quedaron modelos CONFIRM MAIN entrenados; **no hay eval ni análisis CONFIRM**.
3. Se reconstruyó el binario en `d5c4f9c` y se rehicieron solo los pasos DEV (eval, learn, latency, analyze). Luego se detuvo la cadena.

## Diagnóstico (no enmendado)

Hay indicios de bug de implementación en el camino de evaluación, no solo de un encoder que no aprendió: en entrenamiento S2 la familia llega a 0.63–0.99 y los parches promedian ~5.0 bytes; en eval los parches promedian ~1.65, la reconstrucción es ~0.49 y la familia queda bajo el azar. Posibles causas: umbral θg / patching distinto entre train y eval, o mapeo de etiquetas. H2/H6 con diferencia exactamente 0 sugieren que `learn` no usa el encoder del brazo. El prerregistro exige enmienda con commit propio ante fallo de implementación en DEV; el usuario eligió **cerrar** en lugar de enmendar (2026-10-23).

## Decisión

Cerrar `exp/orbita-bytes`. Se mantiene en `main` lo que ya funciona de ÓRBITA (encoder Gemma). No se fusiona el encoder de bytes. Rutina «Vigilar ÓRBITA-Bytes» eliminada.
