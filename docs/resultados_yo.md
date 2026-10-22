# Resultados — el «yo» de ÓRBITA (1er orden) y Gemma como observador de 2º orden (H1–H4)

Prerregistro: `docs/preregistro_yo.md` (commit 8582be9, δ = 0,05), publicado antes de ejecutar nada. Código: `orbita/llm/yo.py` y
`orbita/llm/yo_chain.sh` (1f59893). Ejecución 2026-10-17/18 (CDMX) con `setsid nohup`, nice 0, en `/workspace/orbita-data/yo/`:
humo 23:44 (seed 0x5EED, descartada); extracción calibración DEV 23:44:19–00:08:59, rasgos DEV 00:09:06, generación DEV hasta 11:06:38;
extracción CONFIRM hasta 11:22:30, rasgos 11:22:36, generación CONFIRM hasta 17:24:20. Nada cambió entre DEV y CONFIRM.

**Coste**: la sonda estimó 3,86 h para el total completo (< 4 h), así que se usó el **conjunto completo**, no el subconjunto:
32 seeds × 96 filas × 4 condiciones (YO, SHUF, EMPTY, NOYO) = 12 288 generaciones (384 por seed en todas). Tiempo medio por generación:
DEV 6,42 s [3,74, 9,32], CONFIRM 3,53 s [1,61, 5,80] (incluye contención de máquina).

**Auditoría:** 0 queries a TEST de v4 en ambas fases (`q_*.txt`). Las filas de evaluación son las mismas que e2e (925015f): acc16 E2E_R
DEV 0,2630 [0,1953, 0,3301] y CONFIRM 0,2975 [0,2220, 0,3711], idénticas a `resultados_integracion_e2e.md`.

Artefactos en `artifacts/orbita/yo/`: `summary_{dev,confirm}.md` (idénticos a `analyze_{dev,confirm}.log`), `analyze_h12_*.log`,
`chain.log`, `smoke.log`, `cost_probe.json` y por fase `yo_<part>_<seed>.csv` (rasgos y p̂), `gen_<seed>.json` (salidas de Gemma),
`q_<part>_<seed>.txt` (contador de queries).

## Cifras clave (media [IC95], 16 seeds por fase)

| cantidad | DEV | CONFIRM |
|---|---|---|
| Brier YO / MARGIN / SURPRISE | 0,1536 / 0,2219 / 0,1351 | 0,1516 / 0,2144 / 0,1362 |
| ECE YO / MARGIN / SURPRISE | 0,2149 / 0,1962 / 0,2042 | 0,1716 / 0,1499 / 0,1815 |
| AURC YO / MARGIN / SURPRISE / RANDOM | 0,5411 / 0,7490 / 0,5285 / 0,7370 | 0,4997 / 0,6850 / 0,4638 / 0,7025 |
| AUROC(−p̂ YO → fallo), 96 filas | 0,8214 [0,7418, 0,8943] (n = 15) | 0,8476 [0,7972, 0,8960] (n = 16) |
| familia recuperada YO / SHUF / EMPTY / NOYO | 0,9993 / 0,4414 / 0 / 0 | 1,0000 / 0,4473 / 0 / 0 |
| \|conf/100 − p̂\| YO / SHUF / EMPTY / NOYO | 0,1628 / 0,3188 / 0,3337 / 0,3410 | 0,1943 / 0,3571 / 0,3235 / 0,3262 |
| frase legible YO / SHUF / EMPTY / NOYO | 0,9993 / 1,0000 / 0 / 0 | 1,0000 / 0,9993 / 0 / 0 |
| AUROC(−conf Gemma → fallo), YO | 0,7231 [0,6478, 0,7910] (n = 15) | 0,6590 [0,6004, 0,7156] (n = 16) |

(Una seed DEV sin ambas clases en ok16 se excluye de los AUROC, como estaba prerregistrado.)

## Contrastes prerregistrados (se acepta si IC95 > 0 en DEV y en CONFIRM)

| hipótesis | DEV | CONFIRM | veredicto |
|---|---|---|---|
| **H1** Brier: min(MARGIN, SURPRISE) − YO | −0,0230 [−0,0391, −0,0101] | −0,0204 [−0,0439, −0,0041] | **refutada** (equivalente dentro de δ en ambas fases) |
| H1 sec. ECE: min(MARGIN, SURPRISE) − YO | −0,0565 [−0,0885, −0,0260] | −0,0568 [−0,0837, −0,0324] | **refutada** |
| **H2** AURC: min(MARGIN, SURPRISE, RANDOM) − YO | −0,0130 [−0,0289, −0,0025] | −0,0373 [−0,0885, −0,0036] | **refutada** (equivalencia sólo en DEV) |
| **H3a** familia: YO − max(SHUF, EMPTY, NOYO) | +0,5579 [+0,5319, +0,5879] | +0,5527 [+0,5378, +0,5703] | **aceptada** |
| **H3b** error conf: min(SHUF, EMPTY, NOYO) − YO | +0,1014 [+0,0201, +0,1741] | +0,0887 [+0,0152, +0,1604] | **aceptada** |
| **H4** AUROC Gemma − AUROC p̂ | −0,0982 [−0,1702, −0,0242] (n = 15) | −0,1886 [−0,2491, −0,1347] (n = 16) | **refutada** |

## Lectura

- **1er orden (H1, H2).** El yo calibrado (6 rasgos) es mucho mejor que MARGIN de D2 (Brier ~0,15 frente a ~0,22; AURC ~0,50–0,54 frente
  a ~0,69–0,75) y que el orden aleatorio, y separa bien acierto de fallo (AUROC 0,82–0,85). Pero **no supera a SURPRISE sola**
  (la sorpresa frente a la primera observación): SURPRISE es ligeramente mejor en Brier y AURC con IC95 enteramente por debajo de 0 en
  ambas fases, así que H1 y H2 se rechazan. En Brier la desventaja cabe en (−δ, δ) en las dos fases: el yo es prácticamente equivalente a
  SURPRISE, sin aportar nada sobre ella. En ECE el yo es peor que el mejor baseline (MARGIN en CONFIRM).
- **2º orden (H3).** Gemma verbaliza fielmente al yo: recupera la familia en ~100 % de las filas YO frente a ~44 % con estado barajado y
  0 % sin estado, y su confianza sigue a p̂ (error ~0,16–0,19 frente a ~0,32–0,36 en los controles). Sin estado interno (EMPTY, NOYO)
  no produce frases legibles con el lector prerregistrado.
- **H4.** La confianza que dice Gemma predice el fallo **peor** que la p̂ del yo (AUROC 0,72/0,66 frente a 0,82/0,85). Según lo
  prerregistrado: **el 2º orden sólo verbaliza al yo; no añade predicción** (de hecho pierde información al cuantizar/reescribir p̂).

Resumen: H3a y H3b aceptadas; H1 (primaria y ECE), H2 y H4 refutadas. El rasgo útil del yo es esencialmente la sorpresa; el observador
Gemma es un portavoz fiel pero con pérdida.
