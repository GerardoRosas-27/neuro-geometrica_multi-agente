# Resultados — entrenamiento extenso (D2_EXT + yo recalibrado), ÓRBITA-v1 congelada (HX1–HX3)

Prerregistro: `docs/preregistro_yo_ext.md` (commit 786a237, δ = 0,05), publicado antes de lanzar. Código: `orbita/llm/yo_ext.py` y
`orbita/llm/yo_ext_chain.sh` (786a237). BASE = resultados de `docs/resultados_yo.md` (14d68e3). Ejecución 2026-10-18 (CDMX) con
`setsid nohup`, nice 0, en `/workspace/orbita-data/yo_ext/`, arrancada sola tras `yo/confirm.done`:
inicio 17:26:01; extracción de embeddings del pool ampliado hasta 19:05:03 (7168 textos, 290 359 tokens, 0,47 s/texto);
entrenamiento DEV hasta 19:44:48 (16 seeds, 8 en paralelo, ~1180–1195 s por seed), DEV completo 19:44:55; entrenamiento CONFIRM hasta
20:24:01 (~1155–1184 s por seed), CONFIRM completo 20:24:08. Cada D2_EXT se entrenó 9000 pasos (registro hasta el paso 8999).
Nada cambió entre DEV y CONFIRM.

**Auditoría:** 0 queries a TEST de v4 en ambas fases (64 contadores `q_*.txt`, todos 0). Las cifras BASE coinciden con
`resultados_yo.md` / `resultados_integracion_e2e.md` (acc16 E2E_R DEV 0,2630, CONFIRM 0,2975; Brier YO 0,1536 / 0,1516; AURC YO 0,5411 / 0,4997).

Artefactos en `artifacts/orbita/yo_ext/`: `summary_{dev,confirm}.md` (idénticos a `analyze_{dev,confirm}.log`), `chain.log`,
`train_{dev,confirm}.log`, `pool2.tok.json` y por fase `yo_<part>_<seed>.csv`, `train_<seed>.json` (curvas de pérdida D2_EXT) y
`q_<part>_<seed>.txt`. Los pesos `enc_D2_*.pt` y el `pool2.tok` (2,6 GB) quedan en `/workspace/orbita-data/yo_ext/`.

## Cifras clave (media [IC95], 16 seeds por fase)

| cantidad | DEV BASE | DEV EXT | CONFIRM BASE | CONFIRM EXT |
|---|---|---|---|---|
| acc16 E2E_R | 0,2630 [0,1953, 0,3301] | 0,2585 [0,1875, 0,3320] | 0,2975 [0,2220, 0,3711] | 0,2812 [0,2038, 0,3574] |
| familia correcta | 0,9180 [0,8730, 0,9590] | 0,9186 [0,8053, 0,9909] | 0,9310 [0,8984, 0,9590] | 0,8861 [0,8210, 0,9453] |
| Brier del yo | 0,1536 [0,1174, 0,1923] | 0,1594 [0,1192, 0,2021] | 0,1516 [0,1117, 0,1929] | 0,1382 [0,1062, 0,1692] |
| ECE del yo | 0,2149 [0,1707, 0,2616] | 0,2197 [0,1775, 0,2605] | 0,1716 [0,1402, 0,2080] | 0,1922 [0,1527, 0,2356] |
| AURC del yo | 0,5411 [0,4385, 0,6457] | 0,5612 [0,4488, 0,6744] | 0,4997 [0,4123, 0,5923] | 0,5018 [0,4048, 0,6025] |

## Contrastes prerregistrados (se acepta si IC95 > 0 en DEV y en CONFIRM; equivalencia: IC95 ⊂ (−δ, δ))

| hipótesis | DEV | CONFIRM | veredicto |
|---|---|---|---|
| **HX1** acc16 E2E_R EXT − BASE | −0,0046 [−0,0404, +0,0202] | −0,0163 [−0,0365, −0,0007] | **refutada** (equivalente dentro de δ en ambas fases; en CONFIRM EXT es ligeramente peor) |
| **HX2** familia correcta EXT − BASE | +0,0007 [−0,0885, +0,0632] | −0,0449 [−0,1009, +0,0052] | **no aceptada, inconclusa** (IC95 cruza 0 en ambas, sin equivalencia) |
| **HX3** Brier del yo BASE − EXT | −0,0057 [−0,0220, +0,0100] | +0,0133 [−0,0054, +0,0354] | **refutada** (equivalente dentro de δ en ambas fases) |
| HX3 sec. AURC BASE − EXT | −0,0201 [−0,0780, +0,0247] | −0,0021 [−0,0469, +0,0566] | **no aceptada, inconclusa** |

## Lectura

- Triplicar los pasos (3000 → 9000) y ampliar el pool (contextos nuevos, plantillas 10–11 con p en palabras/fracciones/grados/vueltas/
  radianes, y pt/it) **no mejora** el sistema en ninguna hipótesis. acc16 queda igual o algo peor (−0,016 en CONFIRM con IC95 justo por
  debajo de 0), la familia correcta no mejora y se vuelve más variable entre seeds, y la calibración del yo (Brier, AURC) es
  prácticamente la misma que en BASE.
- El cuello de botella no parece estar en la cantidad de entrenamiento ni en la variedad de textos del encoder D2: con ÓRBITA-v1
  congelada y el mismo calibrador, más datos y más pasos no se traducen en mejor acc16 ni en un yo mejor calibrado.

Resumen: HX1 y HX3 refutadas (equivalencia dentro de δ), HX2 y la secundaria AURC inconclusas. El entrenamiento extenso no aporta
sobre BASE; se mantiene D2 (3000 pasos, pool2) como referencia.
