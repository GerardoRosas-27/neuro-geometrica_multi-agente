# RESULTADOS — CONSTELACIÓN v4: campo rápido solo (FIELD_ONLY), sin ÓRBITA (exploratorio, DEV)

PREREG: `docs/cst_campo_solo/PREREG.md` (commit 62178e5). Análisis: `scripts/analyze_cst_campo_solo.py dev` → `artifacts/cst_campo_solo/analysis_dev.json`.

**Desviación declarada:** para que fuera rápido, DEV recortado a **0xA400–0xA407 (n=8)**; se cortaron 0xA408–0xA40F y no hubo CONFIRM. 6 jobs en paralelo (CPU compartida); ~940 s por semilla (cst ≈360 s, field_only ≈140 s, no_orb ≈360 s + eval).

## Paso 0 — PASS
`cst4-run --phase smoke --seed 0xC420 --arms cst` → hash **1985bdbad5eee41f** (idéntico al de v4). En DEV, el hash de cst en 0xA400 (54868c7b1bdcf956) es idéntico al de `artifacts/constelacion_v4/dev`. 0 consultas TEST, 0 NaN.

## Tabla (n=8; media con IC95 bootstrap pareado, 10000 réplicas, semilla 12345)
| brazo | med log h16 nuevas k=4 (↓) | acc16 conocidas | E33 región (16/16 pass) | acc E33 |
|---|---|---|---|---|
| v4 (cst) | −0.935 [−1.094, −0.779] | 0.877 [0.875, 0.881] | 16 (8/8) | 0.997 |
| FIELD_ONLY | −0.935 [−1.094, −0.779] | 0.877 [0.875, 0.881] | 0 (0/8) | 0.811 |
| NO_ORB (v4 sin cabeza ÓRBITA) | −0.935 (≡ v4) | 0.877 | 0 (0/8) | 0.811 |
| persistencia | +0.063 [−0.094, +0.242] | — | 0 (0/8) | 0.423 |

| contraste | Δ med log nuevas | Δ acc16 conocidas | Δ acc E33 |
|---|---|---|---|
| FIELD_ONLY − v4 | +0.00001 [−0.00005, +0.00008] | 0.000 | −0.186 [−0.191, −0.181] |
| FIELD_ONLY − persistencia | −0.998 [−1.180, −0.802] | — | +0.388 [+0.384, +0.393] |
| v4 − persistencia | −0.998 [−1.180, −0.802] | — | +0.574 |
| NO_ORB − v4 | 0 exacto | 0 | −0.186 |

## Lectura
- **El campo rápido W^F solo, sobre rasgos φ aleatorios congelados (sin W lento, W^P, Π, poda ni sueño), iguala a v4 en la métrica primaria y en acc16 conocidas** (Δ ≈ 1e-5, IC incluye 0) y supera claramente a persistencia (~1 log). Es ~2.6× más barato de entrenar (140 s vs 360 s).
- Implicación honesta (negativa para el resto de v4): en estas métricas de held-out, todo el aprendizaje lento aporta prácticamente nada; el rendimiento de v4 en held lo hace W^F por episodio.
- **E33 falla sin ÓRBITA** (región 0/8, acc 0.81 vs 0.997): el 16/16 de v4 depende enteramente de la cabeza x̂(1) delegada a ÓRBITA. NO_ORB ≡ FIELD_ONLY en E33, así que no es el entrenamiento lento sino la delegación lo que da E33.
- Veredicto: FIELD_ONLY **indistinguible** de v4 en la primaria (no es mejor), **peor en E33**. No hay CONFIRM (exploratorio, n=8).
