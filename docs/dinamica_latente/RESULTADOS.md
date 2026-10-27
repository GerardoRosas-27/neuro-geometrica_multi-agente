# RESULTADOS — exp/orbita-dinamica-latente

Generado automáticamente desde analysis.json. Protocolo: docs/dinamica_latente/PREREG.md. 0 consultas a TEST.

## DEV
Seeds con resultado: 16/16 · varianza explicada PCA media: 0.7977

| Contraste G=log(MSE_B/MSE_A) | media | IC95 |
|---|---|---|
| H1 ORB_LIBRE vs AR2 (ho) | -0.0647 | [-0.1040, -0.0262] |
| H2 ORB_LIBRE vs PERS (ho) | +0.3968 | [+0.3601, +0.4314] |
| H3 ORB_LIBRE vs ORB_SESGO (ho) | -0.2876 | [-0.3147, -0.2635] |
| H4 ORB_LIBRE vs PERS (tf) | +0.2601 | [+0.1929, +0.3248] |
| S ORB_LIBRE vs AR2 (tf) | -0.0648 | [-0.1293, -0.0078] |
| S ORB_SESGO vs PERS (ho) | +0.6844 | [+0.6569, +0.7111] |
| S ORB_SESGO vs PERS (tf) | +0.1394 | [+0.0981, +0.1763] |
| S AR2 vs PERS (ho) | +0.4615 | [+0.4278, +0.5022] |
| S AR2 vs PERS (tf) | +0.3250 | [+0.2933, +0.3577] |
| S ORB_LIBRE vs AR2 (ho, geométricas) | -0.1764 | [-0.2566, -0.1035] |
| S ORB_LIBRE vs AR2 (ho, no geométricas) | -0.0058 | [-0.0530, +0.0387] |
| S ORB_LIBRE vs PERS (ho, no geométricas) | +0.6743 | [+0.6272, +0.7183] |
| S h3 ORB_LIBRE vs AR2 (ho) | -0.1489 | [-0.2101, -0.0980] |
| S h3 ORB_LIBRE vs PERS (ho) | +0.2443 | [+0.1769, +0.3023] |
| C1 ORB_LIBRE vs PERS (permutado) | +0.5572 | [+0.5216, +0.5830] |
| C2 ORB_LIBRE vs PERS (barajado) | +0.3220 | [+0.2944, +0.3467] |
| C2 ORB_LIBRE vs AR2 (barajado) | -0.0142 | [-0.0305, +0.0030] |
| C1 ORB_LIBRE vs AR2 (permutado) | -0.0216 | [-0.0426, -0.0024] |

Criterios: H1=no, H2=sí, H3_no_inferior=no, H4=sí, V1_sin_ganancia_espuria_C1=sí, V2_sin_ganancia_espuria_C2=sí
Gate DEV→CONFIRM: NO pasa → parada, CONFIRM no se ejecuta

## CONFIRM
falta (no ejecutado; ver criterio de parada)

## Veredicto
NO CONFIRMADO (parada en DEV)
