# ÓRBITA vs v4 — fase confirm (n=16 seeds pareadas)

| Brazo | params | acc 1 paso | rel_err₁ | gate E33 (región≥8, limpio, proceso nuevo) | región mediana | acc h1 | acc h2 | acc h4 | acc h8 | acc h16 | acc h32 | acc h64 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| v4_E3 | 2590 | 0.9864 | 0.0089 | 9/16 | 8.0 | 0.991 | 0.978 | 0.869 | 0.551 | 0.217 | 0.044 | 0.008 |
| v4_M3_mlp | 2590 | 0.9837 | 0.0093 | 10/16 | 8.0 | 0.994 | 0.981 | 0.878 | 0.558 | 0.210 | 0.055 | 0.011 |
| ORBITA | 2602 | 0.9819 | 0.0081 | 16/16 | 8.0 | 0.988 | 0.982 | 0.912 | 0.687 | 0.323 | 0.073 | 0.008 |

## Diferencias pareadas (IC95 bootstrap, 10 000 remuestreos, semilla 12345)

| Comparación | métrica | media | IC95 | seeds a favor |
|---|---|---|---|---|
| ORBITA − v4_E3 | acc h16 | +0.1065 | [+0.0664, +0.1504] | 15/16 |
| ORBITA − v4_E3 | región | +1.7500 | [+0.7500, +2.7500] | 7/16 |
| ORBITA − v4_E3 | acc h8 | +0.1357 | [+0.0752, +0.1963] | 14/16 |
| ORBITA − v4_E3 | acc 1 paso | -0.0044 | [-0.0108, +0.0017] | 6/16 |
| ORBITA − v4_M3_mlp | acc h16 | +0.1133 | [+0.0615, +0.1641] | 14/16 |
| ORBITA − v4_M3_mlp | región | +1.5000 | [+0.5000, +2.5000] | 6/16 |
| ORBITA − v4_M3_mlp | acc h8 | +0.1289 | [+0.0498, +0.2041] | 14/16 |
| ORBITA − v4_M3_mlp | acc 1 paso | -0.0017 | [-0.0098, +0.0068] | 5/16 |
