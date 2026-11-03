# CONSTELACIÓN v4 — Resultados F2 (DEV 0xA400–0xA40F, CONFIRM 0xB400–0xB40F)

**Fuentes**
- Preregistro: `PREREG.md`, congelado en `30e79d1` antes de DEV.
- Análisis: `python3 scripts/analyze_constelacion_v4.py`, que escribe `artifacts/constelacion_v4/analysis_dev_confirm.json` (commit `e930e18`).
- Cadena DEV→CONFIRM→análisis: `orbita/scripts/constelacion_v4_chain.sh`.

**Método y controles de calidad**
- Bootstrap pareado por semilla: 10 000 réplicas, semilla 12345, IC95 percentil, δ = 0,05.
- La estimación puntual es la **media entre semillas** de la diferencia pareada.
- n = 16 semillas por fase. Hubo 0 NaN en todos los brazos, 0 consultas TEST (assert) y 0 colapsos de capa en CST.
- ÓRBITA-v1 = ESTELA k4 en 16/16 semillas por fase.
- No hubo reajuste entre DEV y CONFIRM.

## Resumen del diagnóstico (DIAGNOSTICO.md)
- **E33 exige acc ≥ 0,5 empezando en h1.** En v3, x̂(1) se predecía con W^F = 0 y sin contexto del episodio, así que equivalía exactamente a la persistencia (acc h1 0,02–0,09). Por eso la región valía 0 aunque acc h16 = 1,0. No era un problema de entrenamiento: la transformación cambia en cada ejemplo y W^P es global.
- **Más entrenamiento lento no cambia la primaria.** Con N_A ×4 o N_B ×4 sale idéntica. Más iteraciones del campo rápido (t_f 32 → 128) llevan h2 a 1,0.
- **Cambios de v4:**
  - cabeza 1-paso informada por ÓRBITA, usada solo antes de observar la primera transición y solo en E33;
  - t_f = 128;
  - flujos de entrenamiento ×4.
- **Gates previos:**
  - Paso 0: todo PASS, incluida la 0h (`--v3` ≡ binario v3, bit a bit).
  - F0: PASS 4/4.
  - F1: gap PASS, sin colapso, E33 4/4 con región 16.

## Tabla (media [IC95])

| Criterio | DEV | CONFIRM |
|---|---|---|
| Primaria: med log rel_err h16 nuevas {8,10,11}, k=4, CST − ÓRBITA-v1 | −0,076 [−0,141, −0,001] | −0,107 [−0,181, −0,024] |
| No-daño: acc16 conocidas, final − pre | 0,000 [0,000, 0,000] | 0,000 [0,000, 0,000] |
| E33 pase CST (cabeza v4) | **16/16** | **16/16** |
| E33 pase CST con cabeza v3 (`e33_v3head`, informativo) | 0/16 | 0/16 |
| E33 pase ÓRBITA / persistencia | 16/16 / 0/16 | 16/16 / 0/16 |
| E33 acc h1, CST − ÓRBITA | 0,000 [0,000, 0,000] | 0,000 [0,000, 0,000] |
| E33 acc h16, CST − ÓRBITA | +0,688 [+0,631, +0,741] | +0,677 [+0,624, +0,732] |
| Atribución: CST − fixed_w | +0,000002 [−0,00003, +0,00003] | +0,00005 [+0,00002, +0,00009] |
| Atribución: CST − no_err | +0,0025 [+0,0005, +0,0048] | +0,0008 [−0,0002, +0,0018] |
| Atribución: CST − t1 | −0,00007 [−0,00033, +0,00009] | −0,000001 [−0,00004, +0,00005] |
| (inf.) CST − no_prec | −0,00002 [−0,00007, +0,00002] | +0,0000005 [−0,00003, +0,00004] |
| (inf.) CST − no_prune | −0,00003 [−0,00009, +0,00002] | −0,000004 [−0,00004, +0,00004] |
| (inf.) CST − no_sleep | −0,00003 [−0,00008, +0,00002] | −0,00001 [−0,00006, +0,00003] |
| (inf.) CST − one_layer | +0,00016 [−0,000002, +0,00041] | +0,00005 [−0,00002, +0,00014] |
| (solo reporte) CST − backprop | −0,00005 [−0,00013, +0,00002] | +0,00002 [−0,00005, +0,00008] |
| (inf.) CST − no_fast | −3,18 [−3,31, −3,05] | −3,20 [−3,27, −3,13] |
| (inf.) v4 − v3, CST nuevas (pareado por semilla) | +0,013 [+0,002, +0,026] | +0,009 [+0,002, +0,017] |
| Colapso de capa CST | 0/16 | 0/16 |

E33 acc media por horizonte (h1 / h2 / h4 / h8 / h16):
- **DEV:**
  - CST v4: 0,984 / 1,0 / 1,0 / 1,0 / 1,0 (mínimo por semilla en h1: 0,875);
  - CST con cabeza v3: 0,056 / 1,0 / 1,0 / 1,0 / 1,0;
  - ÓRBITA: 0,984 / 0,984 / 0,942 / 0,706 / 0,313;
  - persistencia: 0,056 / 0,527 / 0,511 / 0,509 / 0,512.
- **CONFIRM:**
  - CST v4: 0,988 / 1,0 / 1,0 / 1,0 / 1,0 (mínimo en h1: 0,875);
  - CST con cabeza v3: 0,052 / 1,0 / 1,0 / 1,0 / 1,0;
  - ÓRBITA: 0,988 / 0,982 / 0,912 / 0,687 / 0,323;
  - persistencia: 0,052 / 0,533 / 0,521 / 0,512 / 0,513.

Medias brutas (DEV / CONFIRM):

| brazo | med log h16 nuevas | acc16 nuevas | med log h16 conocidas | acc16 conocidas |
|---|---|---|---|---|
| CST v4 | −0,875 / −0,900 | 0,069 / 0,066 | −9,96 / −10,06 | 0,877 / 0,877 |
| ÓRBITA-v1 | −0,799 / −0,793 | 0,026 / 0,025 | −2,52 / −2,50 | 0,354 / 0,338 |
| no_fast | +2,30 / +2,30 | 0,0 / 0,0 | +2,06 / +2,10 | 0,0 / 0,0 |

## Veredicto por criterio
- **Primaria: ACEPTADA.**
  - El IC95 superior es < 0 en DEV (−0,001) y en CONFIRM (−0,024).
  - El efecto es modesto (≈ −0,08 / −0,11 log), en línea con lo esperado (≈ −0,1).
  - En DEV el margen es mínimo: el límite superior queda en −0,001.
- **No-daño: OK.** IC95 = [0, 0] en ambas fases. La acc16 de conocidas no cambia (≈ 0,877, no vacía).
- **E33 (objetivo ≥ 15/16 en ambas fases): PASS**, con 16/16 en DEV y 16/16 en CONFIRM (v3: 0/16).
  - La ganancia en h1 es **íntegramente de la cabeza ÓRBITA**: acc h1 CST − ÓRBITA = 0 exacto, y la misma red con la cabeza v3 sigue en 0/16, con h1 ≈ 0,05.
  - Lo que aporta CST es la estabilidad a horizonte largo: h16 es +0,68 sobre ÓRBITA, con CST en 1,0 frente a ≈ 0,32.
- **Atribución a PC: NO (rechazada).** Ningún control requerido (fixed_w, no_err, t1) tiene IC sup < 0 en ambas fases.
  - Hay dos IC enteramente > 0, aunque minúsculos: no_err en DEV (+0,0025) y fixed_w en CONFIRM (+0,00005). En esos casos CST queda marginalmente *peor* que el control.
  - La ganancia viene del campo rápido W^F: sin él (`no_fast`) el resultado empeora en ≈ 3,2 log.

Las cuatro expectativas declaradas en el PREREG se cumplieron: E33 ≈ 16/16, primaria ≈ −0,1, no-daño OK y atribución NO. La quinta («quizá un poco mejor que v3 por t_f») **no** se cumplió (ver abajo).

## Comparación con CONSTELACIÓN v3 (mismas semillas)

| | v3 DEV | v4 DEV | v3 CONFIRM | v4 CONFIRM |
|---|---|---|---|---|
| Primaria CST − ÓRBITA | −0,089 [−0,152, −0,018] | −0,076 [−0,141, −0,001] | −0,115 [−0,193, −0,030] | −0,107 [−0,181, −0,024] |
| E33 pase CST | 0/16 | 16/16 | 0/16 | 16/16 |
| acc16 conocidas CST | 0,873 | 0,877 | 0,874 | 0,877 |
| med log conocidas CST | −9,33 | −9,96 | −9,40 | −10,06 |
| med log nuevas CST | −0,888 | −0,875 | −0,908 | −0,900 |
| Atribución PC | NO | NO | NO | NO |

- Las cifras de v3 vienen de `docs/constelacion_v3/RESULTADOS.md` y de `v3_primary_same_seeds` en el JSON de v4.
- **Diferencia pareada v4 − v3 en nuevas:** +0,013 [+0,002, +0,026] en DEV y +0,009 [+0,002, +0,017] en CONFIRM. v4 es **ligeramente peor** que v3 en la métrica primaria, con IC que excluye 0 en ambas fases, aunque la diferencia es de ~0,01 log.
- En conocidas, v4 mejora en ≈ −0,6 nat la med log, con acc16 casi idéntica.
- **El cambio relevante es E33**, que pasa de 0/16 a 16/16, por la cabeza ÓRBITA.

v2, como referencia (`docs/constelacion_v2/RESULTADOS.md`, vía v3): primaria REFUTADA (+0,807 / +0,798) y colapso en 4/16 y 5/16 semillas.

## Comparación con ÓRBITA-v1
- **Referencia.** ÓRBITA-v1 congelada (`artifacts/orbita/{dev,confirm}`), idéntica a ESTELA k4 en 16/16 semillas por fase.
- **Familias nuevas (h16):** CST es mejor por −0,08 / −0,11 log (primaria). En valores absolutos:
  - med log: CST −0,875 / −0,900 frente a ÓRBITA −0,799 / −0,793;
  - acc16: 0,069 / 0,066 frente a 0,026 / 0,025. Ambos siguen bajos en términos absolutos.
- **Familias conocidas:**
  - med log h16: CST −9,96 / −10,06 frente a ÓRBITA −2,52 / −2,50;
  - acc16: 0,877 frente a 0,354 / 0,338.
- **E33:** ambos pasan 16/16. En h1 son idénticos (≈ 0,98–0,99), porque CST usa la predicción de ÓRBITA. Desde h4, ÓRBITA decae (h16 ≈ 0,31–0,32) y CST se mantiene en 1,0.
  - Esto coincide con `docs/resultados_orbita.md`, donde la región E33 de ÓRBITA es 8 en casi todas las semillas.
- No hay un número de ÓRBITA-v1 para la primaria fuera de esta comparación pareada, ni se necesita.

## Conclusión
**Veredicto global: v4 cumple todos los criterios preregistrados de aceptación** (primaria ACEPTADA, no-daño OK, E33 PASS 16/16 en ambas fases), con **atribución a PC: NO**.

Tres matices:
1. **E33 se arregla porque la primera predicción se delega a ÓRBITA**, no por la dinámica de codificación predictiva. Con la cabeza v3, la misma red sigue en 0/16.
2. **La ventaja sobre ÓRBITA en familias nuevas es pequeña** (≈ −0,1 log) y **algo menor que en v3** (+0,01 log, IC excluye 0). Más entrenamiento y t_f = 128 no mejoraron la primaria.
3. **Los componentes de codificación predictiva** (error, precisión, poda, sueño, plasticidad lenta) siguen sin aportar nada medible. Todo el efecto proviene del campo rápido por episodio (W^F) más el prior ÓRBITA en h1.
