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
