# ÓRBITA vs v4 — fase dev (n=16 seeds pareadas)

| Brazo | params | acc 1 paso | rel_err₁ | gate E33 (región≥8, limpio, proceso nuevo) | región mediana | acc h1 | acc h2 | acc h4 | acc h8 | acc h16 | acc h32 | acc h64 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| v4_E3 | 2590 | 0.9834 | 0.0090 | 13/16 | 8.0 | 0.992 | 0.985 | 0.925 | 0.632 | 0.264 | 0.050 | 0.005 |
| v4_M3_mlp | 2590 | 0.9802 | 0.0093 | 12/16 | 8.0 | 0.992 | 0.986 | 0.939 | 0.616 | 0.230 | 0.055 | 0.010 |
| ORBITA | 2602 | 0.9783 | 0.0082 | 16/16 | 8.0 | 0.984 | 0.984 | 0.942 | 0.706 | 0.313 | 0.079 | 0.017 |

## Diferencias pareadas (IC95 bootstrap, 10 000 remuestreos, semilla 12345)

| Comparación | métrica | media | IC95 | seeds a favor |
|---|---|---|---|---|
| ORBITA − v4_E3 | acc h16 | +0.0488 | [-0.0029, +0.1065] | 8/16 |
| ORBITA − v4_E3 | región | +1.2500 | [+0.2500, +2.5000] | 4/16 |
| ORBITA − v4_E3 | acc h8 | +0.0742 | [+0.0195, +0.1328] | 11/16 |
| ORBITA − v4_E3 | acc 1 paso | -0.0051 | [-0.0127, +0.0024] | 7/16 |
| ORBITA − v4_M3_mlp | acc h16 | +0.0830 | [+0.0224, +0.1416] | 12/16 |
| ORBITA − v4_M3_mlp | región | +1.5000 | [+0.5000, +2.7500] | 5/16 |
| ORBITA − v4_M3_mlp | acc h8 | +0.0898 | [+0.0225, +0.1562] | 12/16 |
| ORBITA − v4_M3_mlp | acc 1 paso | -0.0020 | [-0.0120, +0.0085] | 8/16 |
