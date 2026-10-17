# Resultados — encoder D2 (atención por token), H-E5a/b y repeticiones (H-E2', H-E3', H-E4', H-C1')

Prerregistro: `docs/preregistro_encoder_d2.md` (commit 965937e, δ = 0,05), comprometido antes de cualquier resultado. Base: D1 (`docs/resultados_encoder_d1.md`, 481f04a). Ejecución 2026-10-16 (CDMX) con `chain.sh` (`setsid nohup`, nice 0). Logs y archivos `.done` en `/workspace/orbita-data/llm_d2/` (`pool2.tok.done`, `dev/all.tok.done`, `confirm/all.tok.done`, `dev.done`, `confirm.done`). Artefactos en `artifacts/orbita/llm_d2/`: `summary_{dev,confirm}.md` (idénticos a `analyze_{dev,confirm}.log`), `llm_eval_<seed>.csv`, `queries_<seed>.txt` y `train_log.txt` por fase. Nada cambió entre DEV y CONFIRM. **Auditoría: 0 queries** en ambas fases.

**Coste real:**
- Extracción por token (Gemma 2 2B Q4_K_M, capa 13/26, 1 proceso): pool2 1536 textos / 63 264 tokens en 825 s (0,537 s/texto); TEST DEV 1536 / 67 584 tokens en 749 s (0,487 s/texto); TEST CONFIRM 1536 / 67 584 tokens en 776 s (0,505 s/texto). En la cadena, pool2 fue de 16:44 a 18:36 (incluye la espera inicial con poca RAM libre), DEV 18:36–18:48 y CONFIRM 18:48–19:01.
- Entrenamiento (D2 + D1_POOL2, 3000 pasos, primera seed sola y luego 8 en paralelo): 553,2 s por seed en DEV (rango 500,7–567,1) y 551,9 s en CONFIRM (rango 490,7–572,1). DEV terminó a las 19:28:59 y CONFIRM a las 19:55:55.

## Cifras clave (media [IC95], 16 seeds)

| cantidad | DEV | CONFIRM |
|---|---|---|
| acc h16 TRUE | 0,2969 [0,2305, 0,3633] | 0,3164 [0,2344, 0,4023] |
| acc h16 GEMMA_F0 | 0,0000 | 0,0000 |
| acc h16 RAND | 0,0286 [0,0208, 0,0371] | 0,0260 [0,0182, 0,0339] |
| acc h16 D1_NGRAM | 0,0280 [0,0137, 0,0436] | 0,0319 [0,0143, 0,0527] |
| acc h16 D1 | 0,1426 [0,0996, 0,1882] | 0,1478 [0,1165, 0,1803] |
| acc h16 D1_POOL2 | 0,1882 [0,1445, 0,2363] | 0,1882 [0,1582, 0,2194] |
| **acc h16 D2** | **0,1784 [0,1322, 0,2259]** | **0,1543 [0,1061, 0,2051]** |
| acc h1 / h8 / h32 D2 | 0,3320 / 0,3848 / 0,0456 | 0,2689 / 0,3379 / 0,0208 |
| acc h1 / h8 / h32 D1 | 0,0964 / 0,3366 / 0,0332 | 0,0807 / 0,3040 / 0,0234 |
| rel_err mediano h1: TRUE / D1 / D1_POOL2 / D2 | 0,0062 / 0,3281 / 0,2519 / 0,1026 | 0,0057 / 0,2905 / 0,2584 / 0,1099 |
| argmax de familia: D1 / D1_POOL2 / D2 | 0,8138 / 0,9271 / 0,9277 | 0,8652 / 0,9349 / 0,9036 |
| max\|p̂−p\| mediano: D1 / D1_POOL2 / D2 | 0,4583 / 0,4110 / 0,1392 | 0,4495 / 0,4227 / 0,1551 |

## Contrastes prerregistrados

| hipótesis | DEV | CONFIRM |
|---|---|---|
| H-E5a max\|p̂−p\| D2 − D1 | −0,3191 [−0,3444, −0,2938] | −0,2944 [−0,3118, −0,2763] |
| H-E5a acc16 D2 − D1 | +0,0358 [−0,0039, +0,0723] | +0,0065 [−0,0358, +0,0475] |
| H-E5b acc16 D2 − D1_POOL2 | −0,0098 [−0,0449, +0,0306] | −0,0339 [−0,0710, +0,0007] |
| (desc.) max\|p̂−p\| D2 − D1_POOL2 | −0,2718 [−0,2921, −0,2522] | −0,2676 [−0,2879, −0,2476] |
| H-E2' acc16 D2 − TRUE | −0,1185 [−0,1738, −0,0690] | −0,1621 [−0,2363, −0,0859] |
| H-E3'a es−en | +0,0117 [−0,0098, +0,0352] | −0,0098 [−0,0371, +0,0156] |
| H-E3'a es−fr | +0,0039 [−0,0234, +0,0273] | −0,0195 [−0,0410, +0,0020] |
| H-E3'a en−fr | −0,0078 [−0,0352, +0,0176] | −0,0098 [−0,0332, +0,0195] |
| H-E3'b acc16 D2 − RAND | +0,1497 [+0,1061, +0,1940] | +0,1283 [+0,0814, +0,1764] |
| H-E3'c D_same − D_diff | −0,4511 [−0,8496, −0,1917] | −0,4118 [−0,6494, −0,2111] |
| H-E4' acc16 D2 − D1_NGRAM | +0,1504 [+0,1107, +0,1901] | +0,1224 [+0,0677, +0,1778] |
| H-C1'a familia por teacher − 1/8 | +0,8027 [+0,7246, +0,8704] | +0,7786 [+0,7227, +0,8288] |
| H-C1'b AUROC − 0,5 (16 seeds evaluables) | −0,1799 [−0,2578, −0,0924] | −0,2523 [−0,3364, −0,1580] |

## Veredictos (aplicando las reglas prerregistradas)
- **H-E5a: INCONCLUSA.** La primera mitad se cumple con claridad en ambas fases: D2 reduce el error mediano en p de ~0,45 a ~0,14–0,16 (IC95 entero < 0). La segunda mitad no: acc16(D2) − acc16(D1) tiene IC95 que cruza 0 en DEV y en CONFIRM. Ninguna de las dos queda entera del lado contrario, así que no se refuta.
- **H-E5b: INCONCLUSA.** acc16(D2) − acc16(D1_POOL2) cruza 0 en ambas fases (en CONFIRM el extremo superior es +0,0007, casi entero < 0, pero no basta para refutar). Con las plantillas nuevas, la media de tokens rinde lo mismo que la atención en acc16.
- **H-E2': REFUTADA.** En CONFIRM el IC95 superior (−0,0859) < −δ: D2 queda ~0,12–0,16 por debajo de c verdadero en h16.
- **H-E3': ACEPTADA.** (a) los tres pares de idiomas tienen IC95 dentro de (−0,05, 0,05) en ambas fases; (b) D2 − RAND > 0 en ambas; (c) D_same − D_diff < 0 en ambas. Es la primera vez que la condición de invariancia al idioma se cumple entera (en D1 fallaban (a) y (c)).
- **H-E4': ACEPTADA.** D2 sobre Gemma supera a D1_NGRAM en ~0,12–0,15 de acc16 en ambas fases.
- **H-C1': REFUTADA.** (a) se cumple, pero (b) da AUROC − 0,5 con IC95 < 0 en CONFIRM (16 seeds evaluables): igual que en D1, `teacher_div16` está anticorrelacionado con el fallo de ÓRBITA.

## Diagnóstico (descriptivo, no prerregistrado)
- **La atención sí lee mejor los números, pero eso no se traduce en acc16.** p̂ mejora ~3× (max|p̂−p| 0,45 → 0,15) y el rel_err en h1 baja de ~0,29–0,33 a ~0,10–0,11, con acc h1 de ~0,08–0,10 a ~0,27–0,33. En h16 la ganancia desaparece: el error restante en p (~0,15) sigue un orden de magnitud por encima de la tolerancia medida en H-E0 (~0,005–0,01), y en horizontes largos ese error se amplifica.
- **Las plantillas nuevas explican la mejora en acc16 tanto como la atención.** D1_POOL2 llega a 0,188 en h16 en ambas fases, a la par o por encima de D2, aunque su p̂ es mucho peor (0,41–0,42). Así que acc16 depende sobre todo de la familia y de algo más que el max|p̂−p| no captura.
- **Sigue habiendo sobreajuste de p.** La pérdida de p en entrenamiento termina con mediana 0,00074 (DEV) y 0,00095 (CONFIRM) para D2, frente a ~0,009 para D1_POOL2 (`train_log.txt`, columna 3). En TEST el error es mucho mayor.
- **Nota de implementación:** el código comprometido en 965937e siembra el generador de entrenamiento con `seed ^ 0xD2E`, mientras que el texto del prerregistro dice `seed^0xD2`. Se ejecutó el código comprometido sin cambios. Solo afecta a la semilla del muestreo de lotes, no a datos, seeds ni evaluación.
