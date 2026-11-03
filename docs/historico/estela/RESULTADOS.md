# ESTELA v0 + Enmienda 1 — resultados DEV y CONFIRM

Código y valores: commit `85d0b81` (implementación de `ENMIENDA_1.md`, enmienda en `ae6e341`), mismo binario en DEV y CONFIRM, sin cambios entre fases.
DEV 0xA400–0xA40F: 2026-10-24 22:40–22:54; CONFIRM 0xB400–0xB40F: 22:56–23:10 (America/Mexico_City, UTC−6). JOBS=7, sin caídas ni relanzamientos (16/16 `.done` en cada fase, 0 errores en log).
0 consultas TEST en las 32 seeds; Paso 0 en línea 0 discrepancias; núcleo ÓRBITA verificado sin cambios. Análisis: `python3 scripts/analyze_estela.py` (bootstrap pareado 10 000, semilla 12345, IC95 percentil, δ=0,05) → `artifacts/estela/ANALYSIS.json`.
Recordatorio (ENMIENDA_1 §0): 0xA400/0xA401 se vieron en el smoke; se reporta también DEV sin ellas (n=14).

## Criterios (media de diferencias pareadas por seed, IC95)

| Criterio | DEV (n=16) | DEV sin smoke (n=14) | CONFIRM (n=16) | Veredicto |
|---|---|---|---|---|
| **S3** ESTELA − ÓRBITA+LAT, log rel_err h16 nuevas, k=4 (<0 mejor) | −0,228 [−0,341, −0,113] | −0,253 [−0,373, −0,120] | −0,269 [−0,389, −0,149] | **aceptada** (ver advertencia) |
| **S2** ESTELA − ÓRBITA, acc16 conocidas, k=4 (inf > −0,05) | −0,267 [−0,307, −0,224] | −0,271 [−0,311, −0,227] | −0,246 [−0,274, −0,215] | **falla** |
| **S1** rango efectivo ẑ_1 ESTELA (≥ 6,4) | 3,54 | 3,53 | 3,65 | **no cumple** |
| S1 ESTELA − ESTELA−SIGReg (inf > 0) | +0,0006 [−0,0007, +0,0019] | +0,0006 [−0,0008, +0,0021] | −0,0003 [−0,0014, +0,0006] | no cumple |
| **S4** AUROC ESTELA − AUROC ÓRBITA (inf ≥ −0,05) | −0,510 [−0,554, −0,467] | −0,515 [−0,564, −0,465] | −0,544 [−0,572, −0,517] | **no cumple** |

**Éxito de ESTELA v0 (S3 ∧ S2): NO.** S2 (veto de no-daño) falla con claridad en ambas fases. El veredicto es idéntico con DEV n=16 y n=14.

### Advertencia sobre S3
S3 se acepta porque ÓRBITA+LAT es **peor** que ÓRBITA en familias nuevas (log mediana k=4: ÓRBITA −0,80/−0,79; ÓRBITA+LAT −0,36/−0,30 en DEV/CONFIRM). ESTELA (−0,59/−0,57) queda entre ambas: **peor que ÓRBITA congelada** (ESTELA − r0 = +0,213 [+0,137, +0,295] DEV; +0,223 [+0,132, +0,317] CONFIRM; r0 = ÓRBITA bit a bit). La «victoria» primaria no es una mejora sobre la base; es el rival preregistrado el que empeora.

## Valores absolutos (medias sobre seeds, k=4)

| Brazo | DEV nuevas log | DEV conocidas acc16 | CONFIRM nuevas log | CONFIRM conocidas acc16 |
|---|---|---|---|---|
| ÓRBITA | −0,799 | 0,354 | −0,793 | 0,338 |
| ÓRBITA+LAT | −0,357 | 0,430 | −0,298 | 0,426 |
| ESTELA | −0,586 | 0,087 | −0,567 | 0,092 |
| ESTELA_base (sin E4) | −0,586 | 0,014 | −0,567 | 0,022 |
| ESTELA-r0 (= ÓRBITA + E4) | −0,799 | 0,380 | −0,790 | 0,358 |
| ESTELA-REC | −0,789 | 0,139 | −0,780 | 0,149 |
| ESTELA−SIGReg | −0,603 | 0,086 | −0,570 | 0,093 |

AUROC detección (k=4): ESTELA 0,448 / 0,407 vs ÓRBITA 0,958 / 0,951 (DEV/CONFIRM). La sorpresa ŝ (error latente) sigue **por debajo del azar**: las familias nuevas producen menos error latente que las conocidas, aunque su error físico sea mayor. E4: disparo en nuevas 0 % (DEV y CONFIRM); en conocidas 7,9 % / 7,9 %, todos aceptados por el árbitro (por eso ESTELA mejora a ESTELA_base en acc16 conocidas, 0,014→0,087).

## Controles (ESTELA − control, nuevas log k=4; >0 = ESTELA peor)

| Control | DEV | CONFIRM |
|---|---|---|
| −SIGReg | +0,018 [+0,007, +0,027] | +0,003 [−0,003, +0,009] |
| REC | +0,204 [+0,137, +0,271] | +0,213 [+0,145, +0,287] |
| MLP (sin K) | −0,886 [−1,046, −0,728] | −0,801 [−0,962, −0,656] |
| r0 (= ÓRBITA) | +0,213 [+0,137, +0,295] | +0,223 [+0,132, +0,317] |

Lectura: K (Koopman de ÓRBITA) es imprescindible (MLP mucho peor). El residual aprendido + h_ψ + P **empeoran** a ÓRBITA congelada; el objetivo en espacio de estados (REC) es mejor que el latente; SIGReg no tiene efecto medible (con codificador congelado, como se anticipó en la enmienda).

## Exploratorios
- k=2: ESTELA − ÓRBITA+LAT −0,778 [−0,942, −0,642] DEV; −0,716 [−0,947, −0,546] CONFIRM. k=8: +0,451 [+0,324, +0,578] DEV; +0,599 [+0,463, +0,744] CONFIRM (con más pasos observados LAT funciona y ESTELA pierde).
- acc16 conocidas ESTELA − ÓRBITA: k=2 −0,265 / −0,251; k=8 −0,294 / −0,264 (DEV/CONFIRM), todos con IC lejos de −0,05.
- Comprobación de implementación 0xC400 (ENMIENDA_1 §3; H 4/familia, pesos ÓRBITA 0xA400, no forma parte del análisis): r0 = ÓRBITA bit a bit, sin NaN, 0 consultas TEST; ya anticipaba el daño en conocidas (acc16 ESTELA_base 0,0 vs ÓRBITA 0,344). No se cambió ningún valor tras verla.

## Conclusión
Negativo honesto: con la Enmienda 1, ESTELA v0 **no es un éxito**. Congelar el codificador y bajar lr corrigieron la divergencia y la deriva del objetivo (r0 ahora = ÓRBITA), pero el predictor residual entrenado con objetivo JEPA latente sigue degradando la física respecto de ÓRBITA congelada (acc16 conocidas −0,25 a −0,27; nuevas +0,21 en log), y la sorpresa latente no detecta novedad (AUROC < 0,5), así que LAT nunca se dispara en nuevas. S3 se acepta formalmente sólo porque ÓRBITA+LAT es peor que ÓRBITA en nuevas con k=4.
