# ECO — Paso 0 (verificación previa)

Rama `exp/eco` desde `main` 590dd93. Artefacto: `artifacts/eco/paso0.json` (`orbita eco-paso0`, 32 pesos dev+confirm, semilla de datos 0xC400, 30 s).

| Puerta | Criterio | Resultado |
|---|---|---|
| 0a | Re-evaluar checkpoints E3 (`artifacts/v4/distill_e`) en TEST y coincidir con `metrics.csv` (E3_gated_trusted) a ≤5e-5 | **PASA**: dev 0.98340 (csv 0.98341, máx dif 4.4e-5), confirm 0.98633 (csv 0.98635, máx dif 5.0e-5) — reproduce 0.983/0.986 |
| 0b | Con Δ≡0, ECO ≡ ÓRBITA+LAT bit a bit | **PASA**: 516 096 valores comparados, 0 discrepancias (32 pesos); también paridad brazo `eco_zero` vs `orbita_lat` en cada run (assert) |
| 0c | Mapeo preregistrado a codificación D3-num (CDIM=15 por punto: [ŷ_p, q_p, s̃, ŷ_p−q_p, ĉ10]) | **PASA** (`d3_layout_ok`) |

Nota: 0a toca TEST v4 sólo para reproducir el número publicado (re-evaluación de checkpoints congelados, sin entrenamiento ni selección); los contadores de consultas TEST del experimento ECO quedan en 0.

Smoke (0xC400, pesos 0xA400, fuera de DEV/CONFIRM): 182 s/semilla, 0 consultas TEST, paridad 0/192. En el smoke ningún sueño pasó la puerta (0/8 aceptados) y ECO = ÓRBITA+LAT (auc_new −0.3659 en ambos).

Estimación: 16 semillas × ~3–3.5 min con 7 jobs ≈ 10–15 min por fase.
