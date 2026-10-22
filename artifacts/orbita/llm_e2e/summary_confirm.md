## CONFIRM (16 seeds; 16 ejemplos TEST iterables × 6 textos por condición)

| cantidad | media | IC95 | n |
|---|---|---|---|
| acc h1 TRUE | +0.9883 | [+0.9648, +1.0000] | 16 |
| acc h8 TRUE | +0.6836 | [+0.6172, +0.7500] | 16 |
| acc h16 TRUE | +0.3164 | [+0.2344, +0.4023] | 16 |
| acc h32 TRUE | +0.0703 | [+0.0430, +0.0977] | 16 |
| rel_err mediano h1 TRUE | +0.0057 | [+0.0050, +0.0063] | 16 |
| acc h1 D2_DIG | +0.2689 | [+0.2292, +0.3079] | 16 |
| acc h8 D2_DIG | +0.3379 | [+0.2891, +0.3874] | 16 |
| acc h16 D2_DIG | +0.1543 | [+0.1061, +0.2051] | 16 |
| acc h32 D2_DIG | +0.0208 | [+0.0091, +0.0339] | 16 |
| rel_err mediano h1 D2_DIG | +0.1099 | [+0.0918, +0.1323] | 16 |
| familia correcta (argmax D2) D2_DIG | +0.9036 | [+0.8477, +0.9538] | 16 |
| acc h1 D3NUM_DIG | +0.8887 | [+0.8314, +0.9427] | 16 |
| acc h8 D3NUM_DIG | +0.6016 | [+0.5319, +0.6693] | 16 |
| acc h16 D3NUM_DIG | +0.2865 | [+0.2077, +0.3646] | 16 |
| acc h32 D3NUM_DIG | +0.0801 | [+0.0430, +0.1211] | 16 |
| rel_err mediano h1 D3NUM_DIG | +0.0075 | [+0.0066, +0.0085] | 16 |
| familia correcta (argmax D2) D3NUM_DIG | +0.9036 | [+0.8477, +0.9538] | 16 |
| acc h1 D2_R | +0.1172 | [+0.1016, +0.1328] | 16 |
| acc h8 D2_R | +0.4030 | [+0.3522, +0.4453] | 16 |
| acc h16 D2_R | +0.2279 | [+0.1751, +0.2793] | 16 |
| acc h32 D2_R | +0.0352 | [+0.0189, +0.0540] | 16 |
| rel_err mediano h1 D2_R | +0.2458 | [+0.2260, +0.2671] | 16 |
| familia correcta (argmax D2) D2_R | +0.9310 | [+0.8984, +0.9590] | 16 |
| acc h1 E2E_R | +0.9173 | [+0.8796, +0.9525] | 16 |
| acc h8 E2E_R | +0.6217 | [+0.5560, +0.6888] | 16 |
| acc h16 E2E_R | +0.2975 | [+0.2220, +0.3711] | 16 |
| acc h32 E2E_R | +0.0794 | [+0.0430, +0.1198] | 16 |
| rel_err mediano h1 E2E_R | +0.0073 | [+0.0064, +0.0082] | 16 |
| familia correcta (argmax D2) E2E_R | +0.9310 | [+0.8984, +0.9590] | 16 |
| acc16 E2E_R es | +0.3008 | [+0.2227, +0.3789] | 16 |
| acc16 E2E_R en | +0.2930 | [+0.2188, +0.3652] | 16 |
| acc16 E2E_R fr | +0.2988 | [+0.2246, +0.3730] | 16 |
| acc16 E2E_R, 1er parámetro en palabras | +0.3059 | [+0.2204, +0.3914] | 16 |
| acc16 E2E_R, 1er parámetro en fraccion | +0.2882 | [+0.1979, +0.3785] | 16 |
| acc16 E2E_R, 1er parámetro en grados | +0.2763 | [+0.2039, +0.3520] | 16 |
| acc16 E2E_R, 1er parámetro en vueltas | +0.3000 | [+0.2219, +0.3750] | 16 |
| acc16 E2E_R, 1er parámetro en radianes | +0.3156 | [+0.2281, +0.4031] | 16 |
| max|p̂−p| mediano E2E_R | +0.0034 | [+0.0032, +0.0036] | 16 |
| **H-R0** tasa parse_fail E2E_R | +0.0000 | [+0.0000, +0.0000] | 16 |
| **H-R1** acc16 E2E_R − D3NUM_DIG | +0.0111 | [-0.0033, +0.0293] | 16 |
| **H-R2** acc16 E2E_R − D2_R | +0.0697 | [+0.0000, +0.1387] | 16 |
| **H-R3** acc16 E2E_R − TRUE | -0.0189 | [-0.0599, +0.0234] | 16 |
| **H-C2a** familia teacher (E2E_R) − 1/8 | +0.8060 | [+0.7734, +0.8340] | 16 |
| **H-C2b** AUROC residuo16 → fallo − 0.5 | +0.0080 | [-0.0895, +0.1104] | 16 |
| residuo16 mediano (aciertos h16) | +0.4653 | [+0.1769, +0.8273] | 16 |
| residuo16 mediano (fallos h16) | +0.2456 | [+0.1401, +0.3622] | 16 |

Queries totales (llm-eval): 0; parse_fail: 0 de 1536 textos; por formato: {'fraccion|ok': 608, 'grados|ok': 608, 'palabras|ok': 624, 'radianes|ok': 624, 'vueltas|ok': 608}
