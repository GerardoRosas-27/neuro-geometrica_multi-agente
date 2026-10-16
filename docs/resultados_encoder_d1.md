# Resultados — encoder D1 agnóstico al lenguaje (H-E0…H-E5, H-C1)

Prerregistro: `docs/preregistro_encoder_d1.md` (commit c0265c7, δ = 0,05), comprometido antes de cualquier resultado. Diseño: `docs/encoder_agnostico_orbita.md` (54432f9). Ejecución 2026-10-16 (CDMX), con `setsid nohup` a nice 0. Logs y archivos `.done` en `/workspace/orbita-data/llm_d1/` (`noise_{dev,confirm}.done`, `dev.done`, `confirm.done`). Artefactos en `artifacts/orbita/llm_d1/`: `summary_{dev,confirm}.md`, `llm_eval_<seed>.csv`, `noise_eval_<seed>.csv`, `queries_*`/`noise_queries_*` y `train_log.txt` por fase. Nada cambió entre DEV y CONFIRM. **Auditoría: 0 queries** en ambas fases (D1 + H-E0).

**Coste real:** humo de 60 pasos en 9,1 s. Ejecución completa (3000 pasos × encoders GEMMA y NGRAM, 8 seeds en paralelo, 1 hilo cada una): 319,3 s por seed en DEV (rango 315,4–322,2) y 321,0 s en CONFIRM (rango 312,4–330,1). Sin extracción nueva de Gemma: se reutilizaron los embeddings de Fase 0.

## Cifras clave (media [IC95], 16 seeds)

| cantidad | DEV | CONFIRM |
|---|---|---|
| acc h16 TRUE | 0,2969 [0,2305, 0,3633] | 0,3164 [0,2344, 0,4023] |
| acc h16 GEMMA_F0 (Fase 0) | 0,0000 | 0,0000 |
| acc h16 RAND | 0,0286 [0,0208, 0,0371] | (ver summary) |
| **acc h1 / h8 / h16 D1** | 0,0964 / 0,3366 / 0,1426 [0,0996, 0,1882] | 0,0807 / — / 0,1478 [0,1165, 0,1803] |
| acc h16 D1_NGRAM | 0,0280 [0,0137, 0,0436] | 0,0319 [0,0143, 0,0527] |
| rel_err mediano h1: TRUE / D1 / GEMMA_F0 | 0,0062 / 0,3281 / 0,9273 | — / 0,2905 / — |
| argmax de familia D1 | 0,8138 [0,7174, 0,8926] | 0,8652 [0,7988, 0,9245] |
| max\|p̂−p\| mediano D1 | 0,4583 [0,4381, 0,4802] | 0,4495 [0,4309, 0,4676] |

### H-E0: tolerancia de ÓRBITA-v1 a ruido en p (acc16(σ) − acc16(TRUE))

| σ | DEV | CONFIRM |
|---|---|---|
| 0,005 | −0,0098 [−0,0423, +0,0234] | −0,0176 [−0,0573, +0,0247] |
| 0,01 | −0,0397 [−0,0892, +0,0039] | −0,0352 [−0,0775, +0,0091] |
| 0,02 | −0,0762 [−0,1296, −0,0312] | −0,0898 [−0,1380, −0,0404] |
| 0,05 | −0,1068 [−0,1628, −0,0592] | −0,1178 [−0,1771, −0,0586] |

### Contrastes prerregistrados

| hipótesis | DEV | CONFIRM |
|---|---|---|
| H-E1 acc16 D1 − GEMMA_F0 | +0,1426 [+0,0996, +0,1882] | +0,1478 [+0,1165, +0,1803] |
| H-E2 acc16 D1 − TRUE | −0,1543 [−0,2148, −0,0989] | −0,1686 [−0,2448, −0,0898] |
| H-E3a es−en / es−fr / en−fr | −0,0332 [−0,0820, +0,0117] / −0,0312 [−0,0625, −0,0020] / +0,0020 [−0,0449, +0,0547] | −0,0332 [−0,0820, +0,0176] / +0,0000 [−0,0352, +0,0313] / +0,0332 [−0,0156, +0,0840] |
| H-E3b acc16 D1 − RAND | +0,1139 [+0,0749, +0,1562] | +0,1217 [+0,0951, +0,1504] |
| H-E3c D_same − D_diff | +0,0469 [−0,2198, +0,2851] | +0,1361 [−0,0328, +0,3351] |
| H-E4 acc16 D1 − D1_NGRAM | +0,1146 [+0,0736, +0,1569] | +0,1159 [+0,0788, +0,1543] |
| H-C1a familia por teacher − 1/8 | +0,6888 [+0,5924, +0,7676] | +0,7402 [+0,6738, +0,7995] |
| H-C1b AUROC − 0,5 (16 seeds evaluables) | −0,2727 [−0,3232, −0,2201] | −0,2683 [−0,3087, −0,2233] |

## Veredictos (aplicando las reglas prerregistradas)
- **H-E0: INCONCLUSA.** σ = 0,005 cumple en DEV (IC95 inferior −0,042 > −0,05), pero no en CONFIRM (−0,057), así que no hay un σ* común a ambas fases. Tampoco se refuta, porque el IC95 superior es +0,025. Lectura: ÓRBITA-v1 tolera poco ruido en p. Con σ ≥ 0,02 pierde acc16 con IC95 enteramente < 0 en ambas fases, y la tolerancia útil queda en el orden de 0,005–0,01.
- **H-E1: ACEPTADA.** D1 supera a la P de Fase 0 en ambas fases (Fase 0 tenía acc 0).
- **H-E2: REFUTADA.** En CONFIRM el IC95 superior (−0,0898) < −δ: D1 queda unos 0,15–0,17 por debajo de c verdadero en h16.
- **H-E3: INCONCLUSA.**
  - (b) se cumple en ambas fases.
  - (a) falla: varios pares de idiomas tienen IC95 que sale de ±0,05, aunque ninguno queda entero fuera, por lo que no se refuta.
  - (c) no se cumple: D_same − D_diff tiene IC95 que cruza 0 en ambas fases.
- **H-E4: ACEPTADA.** Con el mismo cuello de botella y las mismas pérdidas, Gemma supera a los n-gramas en ~0,115 de acc16 en ambas fases.
- **H-E5: NOT_RUN** (D2 no se lanzó, según el prerregistro).
- **H-C1: REFUTADA.**
  - (a) se cumple: la familia inferida por el teacher consolidado acierta muy por encima del azar.
  - (b) da AUROC − 0,5 con IC95 < 0 en CONFIRM (16 seeds evaluables): `teacher_div16` está anticorrelacionado con el fallo de ÓRBITA, lo contrario de lo hipotetizado.

## Diagnóstico (descriptivo, no prerregistrado)
- **El cuello de botella funciona.** Poner la salida en la variedad de contextos (one-hot straight-through + tanh) lleva acc de 0 a ~0,14–0,15 en h16 y baja el rel_err en h1 de ~0,93 a ~0,29–0,33.
- **El cuello que queda es p.** El error mediano en p̂ (~0,45) está dos órdenes por encima de la tolerancia que mide H-E0 (~0,005–0,01). En entrenamiento, la pérdida de p sobre el pool termina en ~0,002–0,003 (`train_log.txt`), así que el encoder no generaliza p a las plantillas 2–3 nunca vistas: sobreajuste a la redacción, o a las combinaciones de parámetros que la trayectoria no distingue. Ejemplo de esto último: en rotate solo importa 0,6·p₁ + 0,15·p₂.
- **H-C1b invertido:** probablemente porque el teacher se evalúa con el p̂ erróneo. Su divergencia mide error de p̂, no fallo de ÓRBITA. Es una hipótesis, no está probada.
- **Siguiente paso coherente con el diseño:** D2, atención por token sobre capas intermedias para leer los números (H-E5), con más plantillas y parafraseos en el pool. Requiere un prerregistro nuevo.
