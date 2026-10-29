# CONSTELACIÓN v4 — mejoras M1/M2/M3: RESULTADOS

Preregistro: `docs/constelacion_v4_mejoras/PREREG.md`. Análisis: `scripts/analyze_constelacion_v4_mejoras.py` (bootstrap pareado 10000, semilla 12345). 0 consultas TEST.

Primaria: Δ mediana log rel_err h16, familias nuevas, k=4 (brazo − v4; negativo = mejor). IC98.33% (Bonferroni×3). No-daño: IC95 Δacc16 conocidas y Δacc h16 E33 > −0.05.

## Veredictos

| idea | veredicto |
|---|---|
| m1 | NO CONFIRMADA |
| m2 | NO EVALUADA (M2 no aceptada en meta-val) |
| m3 | STOP en DEV |
| all | NO EVALUADA |

## DEV

Brazos (n semillas): {'m1': 16, 'm1_ctrl': 16, 'm3': 16, 'm3_orb': 16, 'm3_rf': 16, 'v4': 16}. Reproducción hash v4 vs artifacts/constelacion_v4: {'n': 16, 'match': 16}. v4 new med log medio -0.8748352722389341; ÓRBITA-v1 -0.7990228251506748.

| brazo | Δ primaria IC98.33 | p Holm | Δacc16 conocidas IC95 | Δacc E33 h16 IC95 | vs ÓRBITA IC95 | h1 − persistencia IC95 | NaN | pass | stop |
|---|---|---|---|---|---|---|---|---|---|
| m1 | +0.0001 [-0.0000, +0.0004] | 0.2448 | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | -0.0757 [-0.1415, -0.0011] | -1.6203 [-1.6921, -1.5497] | 0 | False | False |
| m3 | -0.0013 [-0.0038, +0.0004] | 0.2448 | -0.0845 [-0.1050, -0.0649] | +0.0000 [+0.0000, +0.0000] | -0.0771 [-0.1433, -0.0024] | -1.6210 [-1.6932, -1.5504] | 0 | False | True |
| m1_ctrl | +0.0001 [-0.0000, +0.0004] | — | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | -0.0757 [-0.1415, -0.0011] | -1.6203 [-1.6921, -1.5497] | 0 | False | False |
| m3_rf | -0.0013 [-0.0038, +0.0004] | — | -0.0845 [-0.1050, -0.0649] | +0.0000 [+0.0000, +0.0000] | -0.0771 [-0.1433, -0.0024] | -1.6210 [-1.6932, -1.5503] | 0 | False | True |
| m3_orb | -0.0263 [-0.0698, +0.0184] | — | -0.2988 [-0.3345, -0.2622] | -0.0645 [-0.0830, -0.0479] | -0.1021 [-0.1610, -0.0344] | -1.6060 [-1.6816, -1.5295] | 0 | False | True |
| m1_vs_m1_ctrl | Δ new med log IC95 -0.0000 [-0.0000, +0.0000] | | | | | | | | |

## CONFIRM

Brazos (n semillas): {'m1': 16, 'm1_ctrl': 16, 'v4': 16}. Reproducción hash v4 vs artifacts/constelacion_v4: {'n': 16, 'match': 16}. v4 new med log medio -0.8997498098912131; ÓRBITA-v1 -0.7927990978708592.

| brazo | Δ primaria IC98.33 | p Holm | Δacc16 conocidas IC95 | Δacc E33 h16 IC95 | vs ÓRBITA IC95 | h1 − persistencia IC95 | NaN | pass | stop |
|---|---|---|---|---|---|---|---|---|---|
| m1 | +0.0001 [-0.0000, +0.0003] | 0.0360 | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | -0.1068 [-0.1809, -0.0238] | -1.6777 [-1.7832, -1.5800] | 0 | False | False |
| m1_ctrl | +0.0001 [-0.0000, +0.0003] | — | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | -0.1068 [-0.1810, -0.0239] | -1.6777 [-1.7832, -1.5800] | 0 | False | False |
| m1_vs_m1_ctrl | Δ new med log IC95 +0.0000 [-0.0000, +0.0001] | | | | | | | | |

## Valores congelados

```json
{
 "m1": {
  "gamma": 0.25,
  "beta": 0.5,
  "lam": 0.0,
  "mu": -7.089143585270234,
  "sd": 2.884543558043023,
  "m_mean": 0.9320348982450442
 },
 "m2": {
  "accepted": false,
  "theta": [
   1.0,
   0.0,
   0.0,
   0.0,
   0.0,
   0.0
  ],
  "gain_val": 0.0
 },
 "m3": {
  "h": 32,
  "eta_v": 0.1,
  "eta_u": 0.05,
  "s3": 32
 },
 "tune_seeds": [
  "0xC420",
  "0xC421",
  "0xC422",
  "0xC423"
 ],
 "tune_crit_mean": {
  "v4": -10.120504130565774,
  "m1": -10.133889122767012,
  "m3": -5.759928752006331
 },
 "tune_pick": {
  "m1": "m1_g0.25_b0.5_l0",
  "m3": "m3_h32_ev0.1_eu0.05_s32"
 }
}
```

JSON completo: `artifacts/constelacion_v4_mejoras/analysis_dev_confirm.json`.
