# Resultados — CDT-RQM-epr → inferencia de ÓRBITA: DIRECTO vs SUEÑO (+poda)

Prerregistro: `docs/preregistro_cdt_sueno.md` (ed07b76, δ = 0,05), publicado antes de cualquier rasgo o métrica. Ejecución
2026-10-17 09:15–09:34 (CDMX) con `chain.sh` (`setsid nohup`, nice 0); logs y `.done` en `/workspace/orbita-data/llm_cdt/`
(`pool3.tok.done`, `dev.done`, `confirm.done`). Artefactos en `artifacts/orbita/llm_cdt/` (`summary_{dev,confirm}.md`,
`eval_<seed>.csv`, `queries_<seed>.txt`, `feat_<señal>_<seed>.log`, `pred_<seed>.json`). **0 queries** a TEST
(llm-eval + cdt-feat) en ambas fases. Nada cambió entre DEV y CONFIRM.

**Costo:** extracción Gemma del pool3 (1536 textos) 13,5 min a ~0,5 s/texto (1 proceso); rasgos CDT de las 3 señales
~2 s por seed; entrenamiento de la cabeza (5 condiciones) ~1 s por seed; eval < 1 min por fase.

## CONFIRM (0xB400–0xB40F), conjunto R (plantillas 6–7, p en palabras/fracciones/unidades)
| cantidad | media | IC95 |
|---|---|---|
| acc16 TRUE / BASE | 0,3164 / 0,2975 | [0,2344, 0,4023] / [0,2220, 0,3711] |
| acc16 NOCDT = DIRECTO = SUENO = CORRUPT = SHUF | 0,3092 | [0,2350, 0,3828] |
| error de familia BASE / NOCDT / DIRECTO / SUENO | 0,0690 / 0,0169 / 0,0176 / 0,0169 | |
| tamaño señal full → sueño+poda (números; reglas) | 384 → 282 [273, 288]; 8 → 5,875 | |
| pesos de fusión medios w=[log r, sup, tr] DIRECTO / SUENO | [−0,0135, 0,0414, 0,0352] / [−0,0162, 0,0342, 0,0236] | |

| hipótesis | DEV | CONFIRM | veredicto |
|---|---|---|---|
| H-D1 DIRECTO − BASE (acc16) | +0,0130 [+0,0020, +0,0273] | +0,0117 [+0,0039, +0,0215] | **Aceptada** |
| H-S1 SUENO − BASE | +0,0130 [+0,0020, +0,0273] | +0,0117 [+0,0039, +0,0215] | **Aceptada** |
| H-D2 / H-S2 − NOCDT | 0 [0, 0] | 0 [0, 0] | **Inconclusas** (diferencia nula exacta: ni >0 ni <0) |
| H-D3 / H-S3 − CORRUPT | 0 [0, 0] | 0 [0, 0] | **Inconclusas** (nula) |
| H-D4 / H-S4 − SHUF | 0 [0, 0] | 0 [0, 0] | **Inconclusas** (nula) |
| H-SD SUENO − DIRECTO | 0 [0, 0] | 0 [0, 0] | **Equivalentes** (IC95 ⊂ (−δ, δ)) |
| H-F1 error familia DIRECTO − BASE | −0,0547 [−0,0944, −0,0202] | −0,0514 [−0,0840, −0,0247] | **Aceptada** |
| H-F2 error familia SUENO − BASE | −0,0547 [−0,0944, −0,0202] | −0,0521 [−0,0846, −0,0254] | **Aceptada** |
| H-SDf error familia SUENO − DIRECTO | 0 [0, 0] | −0,0007 [−0,0020, 0] | **Inconclusa** (dentro de ±δ) |

Descriptivo CONFIRM: acc16 DIRECTO/SUENO − TRUE = −0,0072 [−0,0430, +0,0326]; AUROC de r(familia elegida) → fallo h16:
DIRECTO 0,509, SUENO 0,507, CORRUPT 0,536, todos con IC95 que cruza 0,5. En DIG (plantillas 2–3) igual patrón: BASE 0,2865,
NOCDT/CORRUPT/SHUF 0,3099, DIRECTO/SUENO 0,3092; error de familia BASE 0,0964 → DIRECTO 0,0085. Tablas completas en
`artifacts/orbita/llm_cdt/summary_*.md`.

## Conclusión
- La mejora sobre BASE (+0,012 acc16, −5 pp de error de familia) viene del **reentrenamiento de la cabeza de familia de D2**
  en el pool3, no de lo consolidado: NOCDT (sin CDT), CORRUPT y SHUF logran exactamente lo mismo que DIRECTO y SUEÑO.
  Por la regla del prerregistro, **«lo consolidado aporta» no se cumple** (H-D2/H-S2 y H-D3/H-D4/H-S3/H-S4 no aceptadas).
- **Sueño vs directo: equivalentes.** La poda por máscara trusted reduce la señal de 384 a ~282 números (8 → ~5,9 reglas)
  sin cambiar la inferencia.
- Por qué (interpretación, no medida aparte): los rasgos CDT miden *acuerdo* entre ÓRBITA y el teacher para cada familia
  candidata sin ver la respuesta; ambos modelan bien varias familias, así que r_g no discrimina la familia correcta y la
  fusión aprende pesos ~0. Para que la CDT aporte a la inferencia haría falta una señal que dependa del objetivo
  (observaciones del estado siguiente) o usar el teacher como corrector del estado, lo que no se probó aquí.
