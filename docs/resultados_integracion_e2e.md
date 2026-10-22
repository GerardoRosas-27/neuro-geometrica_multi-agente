# Resultados — integración end-to-end D3-num (texto → Gemma/D2 + parser → ÓRBITA-v1 → CDT-RQM-epr) y robustez numérica

Prerregistro: `docs/preregistro_integracion_e2e.md` (commit 5c4d5ba, δ = 0,05), comprometido antes de extraer embeddings o
evaluar textos de robustez. Base: ganador A (D3-num) de `docs/resultados_d3_vs_robusta.md` (951fbf7). Ejecución 2026-10-17
(CDMX) con `/workspace/orbita-data/llm_e2e/chain.sh` (`setsid nohup`, nice 0): test de humo 07:51; extracción DEV
07:51:25–08:04:54; extracción CONFIRM 08:04:59–08:19:36; cadena terminada 08:19:41. Nada cambió entre DEV y CONFIRM.
Artefactos en `artifacts/orbita/llm_e2e/`: `summary_{dev,confirm}.md` (idénticos a `analyze_{dev,confirm}.log`),
`chain.log`, `smoke.log` y por fase `eval_<seed>.csv`, `queries_<seed>.txt`, `parse.json`.

**Auditoría:** 0 queries a TEST en ambas fases (contador de `llm-eval`). **parse_fail = 0 de 1536 textos** en DEV y en
CONFIRM (0 en cada seed); por formato (conteo de valores): palabras 624, fracción 608, grados 608, vueltas 608, radianes 624,
todos `ok`. El fallback a p̂ de D2 no se usó nunca.

## Cifras clave (media [IC95], 16 seeds; 16 ejemplos TEST × 6 textos por condición)

| cantidad | DEV | CONFIRM |
|---|---|---|
| TRUE acc h16 | 0,2969 [0,2305, 0,3633] | 0,3164 [0,2344, 0,4023] |
| D2_DIG acc h16 | 0,1784 [0,1322, 0,2259] | 0,1543 [0,1061, 0,2051] |
| D3NUM_DIG acc h16 | 0,2591 [0,1960, 0,3229] | 0,2865 [0,2077, 0,3646] |
| D2_R acc h16 (textos nuevos) | 0,2259 [0,1693, 0,2871] | 0,2279 [0,1751, 0,2793] |
| **E2E_R acc h16 (textos nuevos)** | **0,2630 [0,1953, 0,3301]** | **0,2975 [0,2220, 0,3711]** |
| E2E_R acc h1 / h8 / h32 | 0,9062 / 0,6068 / 0,0632 | 0,9173 / 0,6217 / 0,0794 |
| TRUE acc h1 / h8 / h32 | 0,9844 / 0,6797 / 0,0898 | 0,9883 / 0,6836 / 0,0703 |
| D2_R acc h1 / h8 / h32 | 0,1100 / 0,3906 / 0,0677 | 0,1172 / 0,4030 / 0,0352 |
| rel_err mediano h1: E2E_R / D3NUM_DIG / D2_R | 0,0076 / 0,0076 / 0,2608 | 0,0073 / 0,0075 / 0,2458 |
| max\|p̂−p\| mediano E2E_R | 0,0036 [0,0034, 0,0037] | 0,0034 [0,0032, 0,0036] |
| familia correcta (argmax D2): textos nuevos / dígitos | 0,9180 / 0,9277 | 0,9310 / 0,9036 |

## Contrastes prerregistrados (CONFIRM decide)

| hipótesis | DEV | CONFIRM | veredicto |
|---|---|---|---|
| **H-R0** tasa parse_fail E2E_R, sup < 0,05 | 0,0000 [0,0000, 0,0000] | 0,0000 [0,0000, 0,0000] | **Aceptada** |
| **H-R1** acc16 E2E_R − D3NUM_DIG, inf > −0,05 | +0,0039 [−0,0111, +0,0241] | +0,0111 [−0,0033, +0,0293] | **Aceptada** (el pipeline con textos nuevos no pierde frente a D3-num con dígitos) |
| **H-R2** acc16 E2E_R − D2_R, IC95 > 0 | +0,0371 [−0,0052, +0,0814] | +0,0697 [0,0000, +0,1387] | **Inconclusa** (el límite inferior en CONFIRM es exactamente 0,0 (recalculado a precisión completa con el mismo bootstrap), no > 0; tampoco se refuta, el superior es +0,1387) |
| **H-R3** acc16 E2E_R − TRUE, inf > −0,05 | −0,0339 [−0,0658, −0,0007] | −0,0189 [−0,0599, +0,0234] | **Inconclusa** (inferior −0,0599 < −0,05; superior +0,0234 no < −0,05) |
| **H-C2a** familia del teacher CDT (E2E_R) − 1/8, IC95 > 0 | +0,7930 [+0,7480, +0,8340] | +0,8060 [+0,7734, +0,8340] | **Aceptada** |
| **H-C2b** AUROC (residuo16 → fallo h16) − 0,5, IC95 > 0 | −0,0001 [−0,1089, +0,1150] (n = 15) | +0,0080 [−0,0895, +0,1104] | **Inconclusa** (IC95 cruza 0) |
| **H-P1** test de humo `test_e2e.sh` | — | — | **Aceptada** (`SMOKE OK`, `smoke.done`; un texto por la cadena completa con 0 queries) |

## Descriptivo (CONFIRM)

- acc16 E2E_R por idioma: es 0,3008 [0,2227, 0,3789]; en 0,2930 [0,2188, 0,3652]; fr 0,2988 [0,2246, 0,3730].
- acc16 E2E_R por formato del 1er parámetro: palabras 0,3059; fracción 0,2882; grados 0,2763; vueltas 0,3000; radianes 0,3156
  (todos los IC95 se solapan).
- Residuo16 mediano del teacher: aciertos h16 0,4653 [0,1769, 0,8273]; fallos h16 0,2456 [0,1401, 0,3622].

## Lectura

- El parser multilingüe leyó los dos valores en el 100 % de los 3072 textos (plantillas 6–7 nunca vistas, es/en/fr, cinco
  formatos), y el error mediano en p (≈0,003) es el mismo que D3-num con dígitos. Con eso, E2E_R iguala a D3NUM_DIG dentro de
  δ (H-R1) sin diferencias visibles por idioma ni por formato.
- Frente a D2 solo (D2_R), el pipeline mejora mucho a h1 (0,9173 vs 0,1172) y la media de acc16 es +0,07 mayor, pero el IC95
  inferior toca exactamente 0, así que H-R2 queda inconclusa según la regla prerregistrada.
- Respecto al contexto real (TRUE) la brecha a h16 es −0,019; el IC95 no basta para declarar empate dentro de δ (H-R3
  inconclusa), igual que H-A2 en la fase anterior. La parte restante debe venir de los ~7 % de errores de familia del argmax de D2.
- CDT-RQM-epr recupera la familia de la predicción de ÓRBITA muy por encima del azar (H-C2a), pero su residuo a h16 sigue sin
  predecir los fallos (H-C2b), como en D3.
