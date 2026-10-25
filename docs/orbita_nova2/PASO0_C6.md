# Paso 0 — Arreglo de C6 (biblioteca a largo plazo) · SOLO SMOKE

Fecha: 2026-10-24 (America/Mexico_City, UTC−6). Rama `exp/orbita-nova2`, base `4bbc98e`.
Aprobado por Bruster (brief «Tras el veredicto de Nova», punto 0). **No es una prueba de hipótesis**: no hay
IC, no hay CONFIRM, 0 consultas TEST. Sólo comprueba que C6 deja de ser inerte y que la biblioteca no explota.

## Cambio (variante nueva opt-in; exp1–3 y `nova-eval` no cambian)
Subcomandos nuevos `orbita nova2-p0-calib` y `orbita nova2-p0` (módulo `orbita/src/nova2p0.rs`).
- Firma = vec(ΔK), ΔK = UVᵀ ∈ R^{8×8} (identificable), en lugar de θ = (U,V).
- Consolidación promedia ΔK (ponderado por soporte) y re-factoriza a rango 2 por SVD (U = u√σ, V = v√σ).
- Fusión: cada clúster del buffer se fusiona con la entrada existente más cercana si d ≤ τ; si no, entra nueva.
- τ = P5 de la distancia (firma ΔK) al vecino más cercano de la otra familia, H DEV fam 1–2, semillas DEV 0xA400..0xA40F, k=4, calibrado una vez.
- λ_prior = 1e-2 cuando C6 recupera una entrada; sin recuperación se mantiene LAM_C4 = 1e-6 (como NOVA_BAYES de exp1).
- Brazos: ORBITA_C3, NOVA (C6 arreglado), NOVA−C6 (nunca recupera), NOVA_SHUF (firmas permutadas). C3/C5/C7 como NOVA de exp1.

## Criterio de paso (fijado antes de ejecutar)
Smoke en semillas DEV 0xA400–0xA403 (nunca CONFIRM), flujo n = 2400, consolidación cada 50, held-out 4 por familia.
En el último checkpoint, PASA si en **todas** las semillas smoke:
1. NOVA−C6 ≠ NOVA: alguna métrica agregada (new/known med_log o acc16, k ∈ {2,4,8}) difiere en > 1e-9, y
2. NOVA_SHUF ≠ NOVA con la misma regla, y
3. tamaño de biblioteca < 100 entradas.
Informativo (no decide): pureza de familia de las recuperaciones, tasa de recuperación, fracción de episodios con θ distinto
entre NOVA y NOVA−C6, y tamaño de una biblioteca «sombra» con la consolidación antigua alimentada con los mismos ítems.
Si falla, se reporta tal cual; no se reajusta nada para que pase.

## Resultado smoke (commit db053e9; artefactos `artifacts/orbita/nova2_paso0/`) — **FALLA** (criterio 3)
τ calibrado (P5, firma ΔK) = 0,02451 (P50 0,0837, P95 0,384; n=512). Control de regresión: P95 con firma θ = 0,9390 ≈ τ legado 0,9387.
Semillas DEV 0xA400–0xA403, n=2400, cada 50, 4 held/fam (44 held), último checkpoint:

| semilla | lib | fusiones | sombra legada | hits flujo (puros) | k2 known med_log NOVA / −C6 / SHUF | ep. con err distinto NOVA vs −C6 (k2/k4/k8) |
|---|---|---|---|---|---|---|
| A400 | 1310 | 218 | 431 | 772 (263) | −3,238 / −4,107 / −2,949 | 21/14/9 |
| A401 | 1535 | 209 | 559 | 517 (245) | −2,924 / −2,940 / −2,623 | 19/12/6 |
| A402 | 1415 | 296 | 545 | 701 (347) | −3,330 / −3,574 / −3,135 | 20/13/10 |
| A403 | 1434 | 313 | 447 | 645 (346) | −4,109 / −4,198 / −3,537 | 14/8/7 |

1–2. NOVA−C6 ≠ NOVA y SHUF ≠ NOVA en las 4 semillas: **cumplen** (C6 ya no es inerte), pero sólo en familias conocidas;
   en las nuevas 8/10/11 no hay ningún hit y las tres métricas son idénticas. En conocidas NOVA suele ser *peor* que NOVA−C6.
3. Biblioteca 1310–1535 ≫ 100 (peor que la sombra legada ~430–560): **falla**. Con τ = P5 casi todo clúster queda a > τ
   de cualquier entrada, así que la fusión apenas actúa (≈ 15 % de los clústeres) y nunca se olvida nada.
Pureza de hits en flujo 34–50 %. No se reajusta nada; siguiente paso a decidir por Bruster.
