# Diagnóstico M8 (cota Brenier) — ¿por qué BQR10 == BQ10 bit a bit?

Rama `exp/bio-quantum-v10-rust` @ 686c470. Semilla de ajuste 0xC4B0 (fase `bq_tune`), pesos `artifacts/orbita/dev/weights_0xA400.json`. 0 DEV/CONFIRM relanzados, 0 consultas TEST.

## Veredicto: (c) — sin bug de cableado. M8 se activa, cambia mucho unos pocos episodios, pero sólo en la cola divergente; mediana y acc16 son invariantes a eso.

## Traza del flag
- `Mech::bq(da, m8, m7)` fija `m.m8` (atlas.rs L20); BQR{3,10,30} = `bq(.., true, true)`, BQ* = `bq(.., false, true)` (L31–32). El control difiere sólo en m8.
- Uso: rollout h=K+1..16 en `run_ep`, `if md.m.m8 { brenier(&mut p, rb0·g^(h−K)) }` ANTES de guardar `preds` y de realimentar `step_ex` (L131–132). Se aplica antes de la métrica.
- La métrica (`eval_held_atlas`, L158–161) usa `p[HORIZON-1]` → mediana de log rel_err y acc16 (umbral REL_ERR_OK).
- M6 (L141–143) puede sustituir la predicción del brazo por la de v4 → descarta lo que hizo M8 en ese episodio.

## Contadores (instrumentación: `diag_m8_instrumentacion.patch`, log `diag_m8_0xC4B0.log`)
BQR10 en 0xC4B0, 240 episodios que pasan por `arm_preds` (held + E33/meta):
| contador | valor |
|---|---|
| llamadas a brenier (pasos de rollout) | 2880 |
| punto·paso evaluados | 17280 |
| activaciones (r > r_b) | 105 (0.61 %) |
| activaciones en h16 | 10 |
| episodios con ≥1 activación | 11 (4.6 %) |
| de esos, descartados por la compuerta M6 (sale v4) | 7 |
| episodios activados cuya predicción se usa | 4 |
| shrink máximo de radio | 0.988 |

Los 4 episodios usados (todos fam=7, conocida):
| err h16 sin M8 | err h16 con M8 | Δ log |
|---|---|---|
| 10.0 (tope de clamp_err) | 3.75 | −0.98 |
| 2.10 | 0.398 | −1.66 |
| 10.0 | 3.01 | −1.20 |
| 10.0 | 3.74 | −0.98 |

M8 mejora mucho los episodios divergentes, pero siguen muy por encima del umbral de acierto y muy lejos de la mediana (known med log ≈ −9). Por eso `known_acc16`, `known_med_log`, `new_med_log` y `frac_v4` son idénticos bit a bit (compruébese en `bq_tune/metrics_0xC4B0.json`: BQ10 y BQR10 iguales). Además, la compresión tanh es de 2.º orden cerca del umbral (sc·r ≈ r − O((r−r_b)³)), así que una activación marginal casi no cambia nada.

## Recomendación
- No hay nada que arreglar. Re-correr DEV/CONFIRM con la misma métrica primaria daría otra vez Δ = 0: **no vale la pena**.
- Si interesa el efecto en la cola: métrica secundaria (media de log err, p90 o fracción de episodios en el tope 10) en un prereg nuevo. Las preds no están guardadas, así que habría que re-correr v4+BQ10+BQR10: ~32 semillas × ~6.5 min (dominado por los ~370 s de entrenamiento) ≈ 3.5 h de CPU, ~40–60 min con 4–6 jobs.
