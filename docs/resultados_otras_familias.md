# Resultados: la pila final en otras familias (re-observación k = 1, 2)

Prerregistro `docs/preregistro_otras_familias.md` (9a5bf72), publicado antes de evaluar. Código: `orbita/src/fam.rs` (`fam-set`, `fam-eval`),
`orbita/llm/fam_eval.py`, `fam_chain.sh`. Ejecución 2026-10-18 CST: extracción Gemma 22:42:51–22:49:35 (480 textos), DEV hasta 22:51:31,
CONFIRM hasta 22:53:26 (CPU 8 núcleos, 7,4 GB RAM disponible al lanzar). 0 queries v4 a TEST y 0 parse_fail en las 32 seeds. El conjunto no usa
`ds.test` (RNG 0xFA4E) y es el mismo en todas las seeds. Tablas completas: `artifacts/orbita/otras_familias/analyze_{dev,confirm}.log`.

## acc16 por grupo (media de 16 seeds; DEV / CONFIRM)
| grupo | BASE | OBS1 | OBS2 | ORACLE_LS1 | TRUE |
|---|---|---|---|---|---|
| conocidas iterables (1, 2) | 0,3118 / 0,2988 | 0,4635 / 0,4408 | 0,5417 / 0,5260 | 1,0 / 1,0 | 0,3730 / 0,3613 |
| v4 no iterables (0, 3–7) | 0,2335 / 0,1997 | 0,2730 / 0,2587 | 0,2782 / 0,2652 | 0,8438 / 0,8438 | 0,2799 / 0,2637 |
| nuevas no afines (8 giro ondulado, 9 espiral) | 0,0007 / 0,0000 | 0,0020 / 0,0000 | 0,0020 / 0,0000 | 0,0312 / 0,0312 | — |

La re-identificación (OBS1) elige la familia verdadera en el 96–100 % de las filas de las familias v4. ORACLE_LS es determinista (igual en todas las
seeds) y resuelve todas las afines (1,0) pero casi ninguna no afín: familia 7 0,0625, 8 0,0625, 9 0. Por familia (DEV), OBS1 a h16: 0 → 0,2852,
3 → 0,4740, 4 → 0,2891, 5 → 0,4531, 6 → 0,1367, 7 → 0, 8 → 0,0039, 9 → 0.

## Hipótesis (IC95 > 0 en DEV y CONFIRM)
| | DEV | CONFIRM | veredicto |
|---|---|---|---|
| **HF1a** OBS1 − BASE, nuevas 8–9 | +0,0013 [0,0000, +0,0039] | 0,0000 [0, 0] | **rechazada** (sin mejora; ambas ≈ 0) |
| **HF1b** OBS1 − BASE, v4 no iterables | +0,0395 [+0,0243, +0,0566] | +0,0590 [+0,0423, +0,0779] | **aceptada** |
| **HF2a** OBS2 − BASE, nuevas 8–9 | +0,0013 [0,0000, +0,0039] | 0,0000 [0, 0] | **rechazada** |
| **HF2b** OBS2 − BASE, v4 no iterables | +0,0447 [+0,0289, +0,0625] | +0,0655 [+0,0484, +0,0842] | **aceptada** |

Brecha acc16 conocidas − nuevas: BASE +0,3112 / +0,2988; OBS1 +0,4616 / +0,4408; OBS2 +0,5397 / +0,5260. Brecha conocidas − v4 no iterables:
BASE +0,0783 / +0,0992; OBS1 +0,1905 / +0,1821; OBS2 +0,2635 / +0,2609. OBSk queda muy por debajo de ORACLE_LSk en las no iterables (≈ −0,57 a −0,59).

## Lectura
La re-observación ayuda en familias que ÓRBITA conoce (incluidas las que nunca iteró), pero **no generaliza a familias nuevas no afines**: la
re-identificación sólo puede elegir entre las 8 familias v4 y ÓRBITA no las representa (≈0 a h16). Incluso en las no iterables, ÓRBITA iterada
pierde contra un simple ajuste afín con una observación; el cuello de botella es la dinámica iterada de ÓRBITA, no la lectura del texto.
