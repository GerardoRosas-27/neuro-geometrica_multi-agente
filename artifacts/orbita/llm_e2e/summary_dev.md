## DEV (16 seeds; 16 ejemplos TEST iterables × 6 textos por condición)

| cantidad | media | IC95 | n |
|---|---|---|---|
| acc h1 TRUE | +0.9844 | [+0.9648, +1.0000] | 16 |
| acc h8 TRUE | +0.6797 | [+0.6016, +0.7539] | 16 |
| acc h16 TRUE | +0.2969 | [+0.2305, +0.3633] | 16 |
| acc h32 TRUE | +0.0898 | [+0.0625, +0.1172] | 16 |
| rel_err mediano h1 TRUE | +0.0062 | [+0.0054, +0.0071] | 16 |
| acc h1 D2_DIG | +0.3320 | [+0.2852, +0.3757] | 16 |
| acc h8 D2_DIG | +0.3848 | [+0.3184, +0.4453] | 16 |
| acc h16 D2_DIG | +0.1784 | [+0.1322, +0.2259] | 16 |
| acc h32 D2_DIG | +0.0456 | [+0.0234, +0.0703] | 16 |
| rel_err mediano h1 D2_DIG | +0.1026 | [+0.0772, +0.1360] | 16 |
| familia correcta (argmax D2) D2_DIG | +0.9277 | [+0.8496, +0.9954] | 16 |
| acc h1 D3NUM_DIG | +0.9147 | [+0.8366, +0.9785] | 16 |
| acc h8 D3NUM_DIG | +0.6126 | [+0.5293, +0.6960] | 16 |
| acc h16 D3NUM_DIG | +0.2591 | [+0.1960, +0.3229] | 16 |
| acc h32 D3NUM_DIG | +0.0664 | [+0.0391, +0.1016] | 16 |
| rel_err mediano h1 D3NUM_DIG | +0.0076 | [+0.0066, +0.0086] | 16 |
| familia correcta (argmax D2) D3NUM_DIG | +0.9277 | [+0.8496, +0.9954] | 16 |
| acc h1 D2_R | +0.1100 | [+0.0957, +0.1230] | 16 |
| acc h8 D2_R | +0.3906 | [+0.3268, +0.4466] | 16 |
| acc h16 D2_R | +0.2259 | [+0.1693, +0.2871] | 16 |
| acc h32 D2_R | +0.0677 | [+0.0339, +0.1094] | 16 |
| rel_err mediano h1 D2_R | +0.2608 | [+0.2431, +0.2821] | 16 |
| familia correcta (argmax D2) D2_R | +0.9180 | [+0.8730, +0.9590] | 16 |
| acc h1 E2E_R | +0.9062 | [+0.8600, +0.9499] | 16 |
| acc h8 E2E_R | +0.6068 | [+0.5215, +0.6901] | 16 |
| acc h16 E2E_R | +0.2630 | [+0.1953, +0.3301] | 16 |
| acc h32 E2E_R | +0.0632 | [+0.0365, +0.0951] | 16 |
| rel_err mediano h1 E2E_R | +0.0076 | [+0.0065, +0.0088] | 16 |
| familia correcta (argmax D2) E2E_R | +0.9180 | [+0.8730, +0.9590] | 16 |
| acc16 E2E_R es | +0.2617 | [+0.1934, +0.3301] | 16 |
| acc16 E2E_R en | +0.2715 | [+0.2031, +0.3398] | 16 |
| acc16 E2E_R fr | +0.2559 | [+0.1895, +0.3242] | 16 |
| acc16 E2E_R, 1er parámetro en palabras | +0.2599 | [+0.1842, +0.3322] | 16 |
| acc16 E2E_R, 1er parámetro en fraccion | +0.2604 | [+0.1806, +0.3368] | 16 |
| acc16 E2E_R, 1er parámetro en grados | +0.2632 | [+0.2007, +0.3289] | 16 |
| acc16 E2E_R, 1er parámetro en vueltas | +0.2688 | [+0.1813, +0.3594] | 16 |
| acc16 E2E_R, 1er parámetro en radianes | +0.2625 | [+0.1906, +0.3375] | 16 |
| max|p̂−p| mediano E2E_R | +0.0036 | [+0.0034, +0.0037] | 16 |
| **H-R0** tasa parse_fail E2E_R | +0.0000 | [+0.0000, +0.0000] | 16 |
| **H-R1** acc16 E2E_R − D3NUM_DIG | +0.0039 | [-0.0111, +0.0241] | 16 |
| **H-R2** acc16 E2E_R − D2_R | +0.0371 | [-0.0052, +0.0814] | 16 |
| **H-R3** acc16 E2E_R − TRUE | -0.0339 | [-0.0658, -0.0007] | 16 |
| **H-C2a** familia teacher (E2E_R) − 1/8 | +0.7930 | [+0.7480, +0.8340] | 16 |
| **H-C2b** AUROC residuo16 → fallo − 0.5 | -0.0001 | [-0.1089, +0.1150] | 15 |
| residuo16 mediano (aciertos h16) | +0.6529 | [+0.3500, +0.9746] | 15 |
| residuo16 mediano (fallos h16) | +0.3998 | [+0.1579, +0.6993] | 16 |

Queries totales (llm-eval): 0; parse_fail: 0 de 1536 textos; por formato: {'fraccion|ok': 608, 'grados|ok': 608, 'palabras|ok': 624, 'radianes|ok': 624, 'vueltas|ok': 608}
