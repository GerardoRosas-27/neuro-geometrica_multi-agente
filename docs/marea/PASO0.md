# MAREA — Paso 0 y coste de preentrenamiento

Rama `exp/marea` desde `main` 590dd93. Módulo nuevo `orbita/src/marea/` (no usa pesos ÓRBITA ni v4; el generador de familias y la línea base ÓRBITA+LAT se copian de `exp/eco`, `orbita/src/eco/mod.rs`).

## Coste de preentrenamiento (medido en esta máquina, 1 hilo)
| Etapa | Presupuesto | Tiempo |
|---|---|---|
| JEPA (enc+pred+actor, EMA 0.996, SIGReg) + sonda (ridge + MLP, 100k pasos) | 200 000 pasos × batch 32, fam. 0–5, semilla 0xD400 | 438 s |
| Control «JEPA aleatorio» (sólo sonda) | 100 000 pasos | 152 s |
| Meta-aprendizaje regla 3 factores (ES 1+8, 40 gen., 32 episodios fam. 6–7, semilla 0xD401) | 320 evaluaciones | 2.3 s |
| **Total** | | **9 min 53 s** (≪ 24 h: viable) |

Estimación previa (corrida de 2000 pasos): 1.25 ms/paso → 200k pasos ≈ 4–5 min de JEPA. Evaluación: ~6.6 s/semilla (8 brazos + ÓRBITA+LAT).

Desarrollo previo al preregistro (sólo datos de preentrenamiento/validación 0xD400–0xD402, nunca DEV/CONFIRM): 1) la sonda conjunta divergía → sonda post-hoc; 2) sonda ridge lineal insuficiente → ridge + MLP; 3) envoltura equivariante a escala (x/rms(x_t)) porque la métrica es rel_err; 4) bug de sonda con capa oculta a cero corregido. En cada iteración el Paso 0a falló o pasó por poco; la versión final se congeló antes de tocar DEV.

## Puertas (artifacts/marea/paso0.json, semilla de datos 0xC400)
| Puerta | Criterio preregistrado | Resultado |
|---|---|---|
| 0a JEPA sin colapso | erank(cov z) ≥ 8/16, sd mín ≥ 0.05, y el predictor congelado bate a la copia x_{t+1}=x_t en validación fam. 0–5 (media ln rel_err) | **PASA**: erank 14.52, sd mín 0.549, pred −1.846 vs copia −1.562 (recon −2.507) |
| 0b TTL apagado ≡ base | η=0 ⇒ predicción idéntica bit a bit al predictor congelado | **PASA**: 0/36 864 discrepancias (también 0/36 864 con JEPA aleatorio); además assert en cada run |
| 0c Consultas TEST | 0 | **PASA**: 0 |

El control JEPA aleatorio no pasa 0a (erank 4.37, pred −1.410), como es de esperar en un control.

Smoke 0xC400 (pesos ÓRBITA 0xA400 sólo para la línea base): pendiente nuevas MAREA 0.0055 vs η=0 0.0175; media ln err nuevas −1.42 vs −1.30; ÓRBITA+LAT (t≥4) −2.86 vs MAREA −1.43.
