# CONSTELACIÓN v3 — Resultados F2 (DEV 0xA400–0xA40F, CONFIRM 0xB400–0xB40F)

Preregistro: `PREREG.md` (congelado en 0e3d3fa, antes de DEV). Análisis: `python3 scripts/analyze_constelacion_v3.py` → `artifacts/constelacion_v3/analysis_dev_confirm.json`.
- Bootstrap pareado por semilla: 10 000 réplicas, semilla 12345, IC95 percentil, δ = 0,05.
- n = 16 semillas por fase, 0 NaN en todos los brazos, 0 consultas TEST (assert), 0 colapsos de capa en CST.
- ÓRBITA-v1 = ESTELA k4 en 16/16 semillas por fase. Sin reajuste entre DEV y CONFIRM.
- CONFIRM: las 16 semillas terminaron en la ejecución original (`.done` + `metrics_*.json`), así que no hubo que relanzar ninguna.

## Resumen del diagnóstico de v2 (ver DIAGNOSTICO.md)
- **Salida ≈ 0.** Dos topes actuaban juntos: el tope L1 por fila (B_in = 10) y el reescalado espectral. Limitaban la ganancia a ≈ 0,08. Además, la lectura h16 se hacía desde la capa superior con un F_L contractivo, así que la predicción decaía a 0.
- **Causa raíz.** v2 no tenía adaptación rápida por episodio ni residuo de persistencia. Una regla lenta y global no puede representar familias cuyo parámetro cambia en cada episodio. Lo confirma un oráculo afín in-context, que llega a −1,2/−1,3 en familias nuevas.
- **Colapso.** Bucle de realimentación: Π_0 cae a pi_min, el error ponderado se debilita y deja de haber aprendizaje. La inferencia larga, además, era inestable (κ·Π > 2/curvatura).

## Correcciones v3
- Cabeza temporal con residuo: x̂(t+1) = x(t) + (W^P + W^F)·φ.
- W^P lento con regla de tres factores.
- W^F rápido (campo líquido NLMS por episodio, que se reinicia en cada episodio).
- Π normalizada por la media geométrica de cada capa, con límites [0,1, 10].
- Paso de inferencia precondicionado.
- Rollout reinyectado.
- w_init = 1 y B_in = 40.
- Nuevo control informativo `no_fast`.

## Tabla (media [IC95])

| Criterio | DEV | CONFIRM |
|---|---|---|
| Primaria: med log rel_err h16 nuevas, CST − ÓRBITA-v1 | −0,089 [−0,152, −0,018] | −0,115 [−0,193, −0,030] |
| No-daño: acc16 conocidas, final − pre | +0,007 [0,000, +0,022] | +0,0005 [0,000, +0,0015] |
| E33 pase (CST / ÓRBITA) | 0/16 / 16/16 | 0/16 / 16/16 |
| E33 acc h16, CST − ÓRBITA | +0,688 [+0,631, +0,741] | +0,677 [+0,624, +0,732] |
| Atribución: CST − fixed_w | +0,00007 [−0,00003, +0,00022] | +0,00005 [−0,00002, +0,00013] |
| Atribución: CST − no_err | +0,0004 [−0,0007, +0,0014] | +0,0005 [−0,0005, +0,0016] |
| Atribución: CST − t1 | +0,00008 [−0,00004, +0,00024] | +0,00002 [−0,00004, +0,00009] |
| (inf.) CST − no_prec | +0,00005 [−0,00007, +0,00021] | +0,00002 [−0,00004, +0,00009] |
| (inf.) CST − no_prune | +0,00003 [−0,00009, +0,00020] | +0,00005 [−0,00001, +0,00012] |
| (inf.) CST − no_sleep | +0,00006 [−0,00003, +0,00021] | +0,00003 [−0,00003, +0,00009] |
| (inf.) CST − one_layer | −0,013 [−0,039, +0,0002] | +0,00007 [+0,0000005, +0,00015] |
| (solo reporte) CST − backprop | −0,010 [−0,033, +0,003] | −0,010 [−0,027, +0,0001] |
| (inf.) CST − no_fast | −3,19 [−3,32, −3,07] | −3,20 [−3,27, −3,11] |
| (inf.) v3 − v2, CST nuevas (pareado) | −0,896 [−1,028, −0,771] | −0,913 [−0,981, −0,837] |
| Colapso de capa CST | 0/16 | 0/16 |

Medias brutas (med log h16):
- Nuevas: CST −0,888 (DEV) / −0,908 (CONFIRM), frente a ÓRBITA −0,799 / −0,793.
- Conocidas: CST −9,33 / −9,40, frente a ÓRBITA −2,52 / −2,50.
- acc16 conocidas: CST 0,873 / 0,874, frente a ÓRBITA 0,354 / 0,338.

## Veredicto por criterio
- **Primaria: ACEPTADA.** El IC95 superior es < 0 en DEV (−0,018) y en CONFIRM (−0,030). El efecto es modesto (≈ −0,1 log) y coincide con la expectativa preregistrada (−0,07 a −0,10).
- **No-daño: OK.** El IC95 inferior es 0 > −δ en ambas fases. Esta vez no es vacío: acc16 de conocidas ≈ 0,87.
- **E33: FALLA.** 0/16 en ambas fases, como se anticipó: acc h1 ≈ 0,03 porque la primera predicción no tiene información de transición. En h16, en cambio, CST supera a ÓRBITA en +0,68.
- **Atribución a PC: NO.** Ningún control requerido (fixed_w, no_err, t1) tiene IC sup < 0. Las diferencias son ~1e-4 log. La ganancia viene del campo rápido W^F: sin él (`no_fast`) el resultado empeora ≈ 3,2 log. También estaba anticipado.

## Comparación con v2
- v2: primaria REFUTADA (+0,807 DEV / +0,798 CONFIRM), CST ≈ predicción nula, colapso en 4/16 y 5/16 semillas, no-daño vacío.
- v3 mejora a v2 en ≈ −0,9 log por semilla en familias nuevas (IC excluye 0 en ambas fases), elimina el colapso y pasa de peor a mejor que ÓRBITA-v1.
- E33 y atribución a PC siguen fallando en ambas versiones.

**Conclusión:** CONSTELACIÓN v3 cumple la primaria preregistrada y el no-daño. La mejora se debe a la adaptación rápida por episodio (W^F), no a los componentes de codificación predictiva: atribución NO y E33 FALLA.
