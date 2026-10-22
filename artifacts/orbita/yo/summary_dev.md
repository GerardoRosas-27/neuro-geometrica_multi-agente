## DEV (16 seeds)

| cantidad | media | IC95 | n |
|---|---|---|---|
| acc16 E2E_R (tasa de acierto real) | +0.2630 | [+0.1953, +0.3301] | 16 |
| brier YO | +0.1536 | [+0.1174, +0.1923] | 16 |
| brier MARGIN | +0.2219 | [+0.1889, +0.2516] | 16 |
| brier SURPRISE | +0.1351 | [+0.1025, +0.1712] | 16 |
| ece YO | +0.2149 | [+0.1707, +0.2616] | 16 |
| ece MARGIN | +0.1962 | [+0.1560, +0.2390] | 16 |
| ece SURPRISE | +0.2042 | [+0.1672, +0.2433] | 16 |
| aurc YO | +0.5411 | [+0.4385, +0.6457] | 16 |
| aurc MARGIN | +0.7490 | [+0.6688, +0.8259] | 16 |
| aurc SURPRISE | +0.5285 | [+0.4249, +0.6351] | 16 |
| aurc RANDOM | +0.7370 | [+0.6699, +0.8047] | 16 |
| **H1** Brier: min(MARGIN, SURPRISE) − YO | -0.0230 | [-0.0391, -0.0101] | 16 |
| H1 (sec.) ECE: min(MARGIN, SURPRISE) − YO | -0.0565 | [-0.0885, -0.0260] | 16 |
| **H2** AURC: min(MARGIN, SURPRISE, RANDOM) − YO | -0.0130 | [-0.0289, -0.0025] | 16 |
| AUROC(−p̂ YO → fallo), 96 filas | +0.8214 | [+0.7418, +0.8943] | 15 |
| familia recuperada YO | +0.9993 | [+0.9980, +1.0000] | 16 |
| |conf/100 − p̂| YO | +0.1628 | [+0.1237, +0.2065] | 16 |
| frase legible YO | +0.9993 | [+0.9980, +1.0000] | 16 |
| familia recuperada SHUF | +0.4414 | [+0.4108, +0.4674] | 16 |
| |conf/100 − p̂| SHUF | +0.3188 | [+0.2928, +0.3465] | 16 |
| frase legible SHUF | +1.0000 | [+1.0000, +1.0000] | 16 |
| familia recuperada EMPTY | +0.0000 | [+0.0000, +0.0000] | 16 |
| |conf/100 − p̂| EMPTY | +0.3337 | [+0.2582, +0.4113] | 16 |
| frase legible EMPTY | +0.0000 | [+0.0000, +0.0000] | 16 |
| familia recuperada NOYO | +0.0000 | [+0.0000, +0.0000] | 16 |
| |conf/100 − p̂| NOYO | +0.3410 | [+0.2693, +0.4156] | 16 |
| frase legible NOYO | +0.0000 | [+0.0000, +0.0000] | 16 |
| **H3a** familia: YO − max(SHUF, EMPTY, NOYO) | +0.5579 | [+0.5319, +0.5879] | 16 |
| **H3b** error conf: min(SHUF, EMPTY, NOYO) − YO | +0.1014 | [+0.0201, +0.1741] | 16 |
| AUROC(−conf Gemma → fallo) | +0.7231 | [+0.6478, +0.7910] | 15 |
| AUROC(−p̂ → fallo), mismas filas | +0.8214 | [+0.7418, +0.8943] | 15 |
| **H4** AUROC Gemma − AUROC p̂ | -0.0982 | [-0.1702, -0.0242] | 15 |
| generaciones por seed | +384.0000 | [+384.0000, +384.0000] | 16 |
| s por generación | +6.4204 | [+3.7439, +9.3239] | 16 |

Queries v4 (yo-feat, dev+test): 0
