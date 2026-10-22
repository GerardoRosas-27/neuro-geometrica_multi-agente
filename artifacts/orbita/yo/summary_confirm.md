## CONFIRM (16 seeds)

| cantidad | media | IC95 | n |
|---|---|---|---|
| acc16 E2E_R (tasa de acierto real) | +0.2975 | [+0.2220, +0.3711] | 16 |
| brier YO | +0.1516 | [+0.1117, +0.1929] | 16 |
| brier MARGIN | +0.2144 | [+0.1877, +0.2396] | 16 |
| brier SURPRISE | +0.1362 | [+0.1014, +0.1724] | 16 |
| ece YO | +0.1716 | [+0.1402, +0.2080] | 16 |
| ece MARGIN | +0.1499 | [+0.1100, +0.1947] | 16 |
| ece SURPRISE | +0.1815 | [+0.1467, +0.2229] | 16 |
| aurc YO | +0.4997 | [+0.4123, +0.5923] | 16 |
| aurc MARGIN | +0.6850 | [+0.5858, +0.7772] | 16 |
| aurc SURPRISE | +0.4638 | [+0.3714, +0.5621] | 16 |
| aurc RANDOM | +0.7025 | [+0.6289, +0.7780] | 16 |
| **H1** Brier: min(MARGIN, SURPRISE) − YO | -0.0204 | [-0.0439, -0.0041] | 16 |
| H1 (sec.) ECE: min(MARGIN, SURPRISE) − YO | -0.0568 | [-0.0837, -0.0324] | 16 |
| **H2** AURC: min(MARGIN, SURPRISE, RANDOM) − YO | -0.0373 | [-0.0885, -0.0036] | 16 |
| AUROC(−p̂ YO → fallo), 96 filas | +0.8476 | [+0.7972, +0.8960] | 16 |
| familia recuperada YO | +1.0000 | [+1.0000, +1.0000] | 16 |
| |conf/100 − p̂| YO | +0.1943 | [+0.1543, +0.2370] | 16 |
| frase legible YO | +1.0000 | [+1.0000, +1.0000] | 16 |
| familia recuperada SHUF | +0.4473 | [+0.4297, +0.4622] | 16 |
| |conf/100 − p̂| SHUF | +0.3571 | [+0.3249, +0.3891] | 16 |
| frase legible SHUF | +0.9993 | [+0.9980, +1.0000] | 16 |
| familia recuperada EMPTY | +0.0000 | [+0.0000, +0.0000] | 16 |
| |conf/100 − p̂| EMPTY | +0.3235 | [+0.2639, +0.3854] | 16 |
| frase legible EMPTY | +0.0000 | [+0.0000, +0.0000] | 16 |
| familia recuperada NOYO | +0.0000 | [+0.0000, +0.0000] | 16 |
| |conf/100 − p̂| NOYO | +0.3262 | [+0.2697, +0.3855] | 16 |
| frase legible NOYO | +0.0000 | [+0.0000, +0.0000] | 16 |
| **H3a** familia: YO − max(SHUF, EMPTY, NOYO) | +0.5527 | [+0.5378, +0.5703] | 16 |
| **H3b** error conf: min(SHUF, EMPTY, NOYO) − YO | +0.0887 | [+0.0152, +0.1604] | 16 |
| AUROC(−conf Gemma → fallo) | +0.6590 | [+0.6004, +0.7156] | 16 |
| AUROC(−p̂ → fallo), mismas filas | +0.8476 | [+0.7972, +0.8960] | 16 |
| **H4** AUROC Gemma − AUROC p̂ | -0.1886 | [-0.2491, -0.1347] | 16 |
| generaciones por seed | +384.0000 | [+384.0000, +384.0000] | 16 |
| s por generación | +3.5319 | [+1.6073, +5.8022] | 16 |

Queries v4 (yo-feat, dev+test): 0
