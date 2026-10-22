# Resultados — Fase 0 LLM → ÓRBITA → CDT-RQM-EPR

Prerregistro: `docs/preregistro_orbita_llm.md` (commit 9897860, δ = 0,05). Plan: `docs/plan_llm_orbita_cdt_fase0.md`. Cadena terminada 2026-10-15 23:56 (CDMX); DEV terminó 20:56. Nada se cambió entre DEV y CONFIRM.
Artefactos: `artifacts/orbita/llm/` (`fitP.json`, `summary_dev.md`, `summary_confirm.md`, `llm_eval_<seed>.csv` y `queries_<seed>` por fase). Estadístico por seed, IC95 bootstrap pareado sobre 16 seeds (10 000 remuestreos, semilla 12345). Auditoría: **0 queries** a memoria/TEST durante `llm-eval` en ambas fases.

## Ajuste de P (pool, 768 textos)
GEMMA: α = 1 (CV-MSE 0,00897). NGRAM: α = 1000 (CV-MSE 0,0653).

## Cifras clave (media [IC95], n = 16 seeds)

| cantidad | DEV | CONFIRM |
|---|---|---|
| acc h16 TRUE | 0,2969 [0,2305, 0,3633] | 0,3164 [0,2344, 0,4023] |
| acc h1 / h16 GEMMA | 0,0000 / 0,0000 | 0,0000 / 0,0000 |
| acc h16 NGRAM | 0,0000 | 0,0000 |
| acc h16 RAND | 0,0286 [0,0208, 0,0371] | 0,0260 [0,0182, 0,0339] |
| **H-L1** acc16 GEMMA − TRUE | −0,2969 [−0,3633, −0,2305] | −0,3164 [−0,4023, −0,2344] |
| **H-L2** D_same16 − D_diff16 | +26,71 [+8,28, +49,54] | +95,97 [−11,34, +286,67] |
| H-L2 D_same16 | 33,14 [12,25, 59,50] | 161,29 [28,06, 375,25] |
| **H-L3a** acc16 es−en, es−fr, en−fr | 0 [0, 0] (las tres) | 0 [0, 0] (las tres) |
| **H-L3b** acc16 GEMMA − NGRAM | 0 [0, 0] | 0 [0, 0] |
| **H-L3b** acc16 GEMMA − RAND | −0,0286 [−0,0371, −0,0208] | −0,0260 [−0,0339, −0,0182] |
| **H-C1a** familia teacher − 1/8 | +0,1237 [+0,0820, +0,1699] | +0,0345 [+0,0072, +0,0605] |
| familia argmax ĉ | 0,7415 [0,7337, 0,7500] | 0,7507 [0,7415, 0,7604] |
| **H-C1b** AUROC(fallo h16) − 0,5, GEMMA | no evaluable (n = 0) | no evaluable (n = 0) |
| tasa de fallo h16 GEMMA | 1,0000 | 1,0000 |

Tablas completas: `artifacts/orbita/llm/summary_{dev,confirm}.md`.

## Veredictos (reglas prerregistradas)
- **H-L1 — REFUTADA.** En CONFIRM el IC95 superior (−0,2344) < −δ = −0,05; igual en DEV. ÓRBITA con ĉ de Gemma no acierta ni en h1 (acc 0 en todos los horizontes).
- **H-L2 — INCONCLUSA.** No se acepta: D_same − D_diff > 0 en DEV (IC95 entero > 0) y D_same está muy lejos del umbral 0,05. No se refuta formalmente porque en CONFIRM el IC95 inferior (−11,34) cruza 0. Lectura: las trayectorias con ĉ explotan (D ≫ 1), la métrica no discrimina parafraseos.
- **H-L3 — INCONCLUSA.** (a) se cumple sólo de forma degenerada (acc 0 en los tres idiomas); (b) falla: GEMMA = NGRAM (IC95 [0, 0], no > 0) y GEMMA < RAND con IC95 entero < 0. No se refuta por la regla escrita (ningún par fuera de ±δ; GEMMA − NGRAM no tiene IC95 < 0, es exactamente 0).
- **H-C1 — INCONCLUSA.** (a) se cumple en DEV y CONFIRM (IC95 > 0). (b) no es evaluable: GEMMA falla en h16 en el 100 % de los ejemplos, ninguna seed tiene ambas clases (n = 0). Referencia no prerregistrada: con c verdadero, AUROC − 0,5 = +0,0355 [−0,0947, +0,1711] (DEV) y +0,0195 [−0,0922, +0,1340] (CONFIRM), IC95 cruza 0.

## Diagnóstico (descriptivo, no prerregistrado)
Mediana de rel_err por condición (todos los ejemplos, `llm_eval_*.csv`):

| cond | DEV h1 / h8 / h16 | CONFIRM h1 / h8 / h16 |
|---|---|---|
| TRUE | 0,0059 / 0,0370 / 0,0718 | 0,0055 / 0,0350 / 0,0678 |
| GEMMA | 0,9442 / 2,0757 / 2,7181 | 1,0013 / 2,5739 / 3,3677 |
| NGRAM | 0,6507 / 1,5604 / 2,0304 | 0,6723 / 1,7805 / 2,5338 |
| RAND | 1,1325 / 1,7654 / 2,0692 | 1,0486 / 1,7653 / 2,0839 |

El argmax de ĉ recupera la familia ~74–75 % de las veces, pero ÓRBITA-v1 recibe ĉ continuo (sin redondear ni proyectar a one-hot, como fija el prerregistro) y el error ya es ≈ 1 en h1. Hipótesis para una fase siguiente (no probada aquí): ÓRBITA es muy sensible a ĉ fuera de la variedad de contextos de entrenamiento; proyectar la parte de familia a one-hot o entrenar P con la pérdida predictiva del plan (no corrida en esta fase) son los candidatos naturales, con nuevo prerregistro.
