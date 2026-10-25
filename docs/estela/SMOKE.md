# ESTELA v0 — smoke (no analizado; sólo tiempo y estabilidad)

Commit `893d2b2` (PREREG ya congelado). `PHASE=smoke JOBS=2 bash orbita/scripts/estela.sh`: seeds DEV 0xA400 y 0xA401, H reducido a 4/familia (44 episodios), 3000 pasos por variante (valor congelado). 2026-10-24 20:49–20:54 (America/Mexico_City, UTC−6).

## Tiempo
- ~5 min por seed con 2 procesos en paralelo (8 núcleos): entrenamiento 42–49 s por variante (Full, −SIGReg, REC, MLP) y ~105 s r0; evaluación ≈ 1 s por variante con H=44 y brazos ÓRBITA/ÓRBITA+LAT ≈ 1,5 s.
- Estimación DEV/CONFIRM (H=176, 4× evaluación, E4 dispara poco): ~6 min por seed; con JOBS=7, 3 tandas ⇒ **≈ 20–30 min por fase**, ≈ 1 h DEV+CONFIRM.
- Paso 0 en línea: 0 discrepancias en ambas seeds; 0 consultas TEST.

## Señales de alarma (2 seeds, H reducido; no es un resultado)
- **No-daño en riesgo**: tras entrenar, acc16 en conocidas k=4 cae de ÓRBITA 0,19–0,28 a 0,03 (ESTELA) y 0 en `estela_base`; la pérdida de sonda sube durante el entrenamiento (0,017→0,034 y 0,026→0,071) mientras baja la JEPA. El objetivo latente con sonda de poco peso y codificador entrenable aleja la predicción física de ÓRBITA.
- **E4 no se dispara en nuevas** (trigger_new = 0): AUROC de `s_E` 0,31–0,42 (invertida) frente a 0,96–0,99 de ÓRBITA. Con τ_E calibrado en conocidas, ESTELA = ESTELA_base.
- **ESTELA-r0 diverge** (pérdida JEPA 0,18→4,3 y 0,35→6,8; ‖e2‖ 2,5–4,5; τ_E ≈ 120).
- ÓRBITA+LAT queda peor que ÓRBITA en nuevas en estas 2 seeds (log −0,09/−0,14 vs −0,89/−0,55).
Con el PREREG tal cual, S2 probablemente falla y S3 probablemente no se acepta. Cualquier cambio requiere una enmienda explícita aprobada por Bruster antes de DEV (y declarar que 0xA400/0xA401 se vieron en el smoke).
