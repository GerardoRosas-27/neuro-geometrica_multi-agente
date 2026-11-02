# RESULTADOS — BIO-CONSTELACIÓN bajo protocolo CONSTELACIÓN v4

Generado automáticamente por la cadena a partir de `artifacts/bio_constelacion/analysis.json` (PREREG: `PREREG.md`). Veredicto = CONFIRM.

## DEV

| brazo | new_med_log | known_med_log | new_acc16 | known_acc16 | E33 acc h1 |
|---|---|---|---|---|---|
| v4 | -0.836 | -9.992 | 0.073 | 0.875 | 1.000 |
| hyb | -1.119 | -1.283 | 0.000 | 0.074 | 0.000 |
| hyb_nobio | -1.183 | -1.320 | 0.062 | 0.141 | 0.000 |
| ÓRBITA-v1 | -0.823 | | | | |
| persistencia | 0.120 | | | | |

- **Primaria** hyb − v4 (new_med_log, mediana [IC95]): -0.262 [-0.422, -0.171] → PASA
- hyb − ÓRBITA-v1: -0.330 [-0.444, -0.128]
- hyb − persistencia: -1.199 [-1.394, -0.988]
- hyb − hyb_nobio (ablación bio): +0.066 [-0.053, +0.121]
- No-daño known_acc16: -0.812 [-0.820, -0.781]; E33 h16: -0.992 [-1.000, -0.984] → DAÑO
- v4 reproduce hashes de exp/constelacion-v4-mejoras: True

## CONFIRM

| brazo | new_med_log | known_med_log | new_acc16 | known_acc16 | E33 acc h1 |
|---|---|---|---|---|---|
| v4 | -0.920 | -9.906 | 0.062 | 0.875 | 1.000 |
| hyb | -1.029 | -1.307 | 0.000 | 0.062 | 0.000 |
| hyb_nobio | -1.009 | -1.304 | 0.052 | 0.137 | 0.000 |
| ÓRBITA-v1 | -0.768 | | | | |
| persistencia | 0.079 | | | | |

- **Primaria** hyb − v4 (new_med_log, mediana [IC95]): -0.112 [-0.185, +0.017] → NO PASA
- hyb − ÓRBITA-v1: -0.191 [-0.427, -0.048]
- hyb − persistencia: -1.128 [-1.291, -0.984]
- hyb − hyb_nobio (ablación bio): +0.015 [-0.039, +0.039]
- No-daño known_acc16: -0.816 [-0.824, -0.789]; E33 h16: -0.984 [-0.992, -0.969] → DAÑO
- v4 reproduce hashes de exp/constelacion-v4-mejoras: True
