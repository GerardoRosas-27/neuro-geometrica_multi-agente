# Prerregistro — Fase 0 LLM → ÓRBITA → CDT-RQM-EPR

**Comprometido antes de ver cualquier resultado** (2026-10-15, CDMX). Plan: `docs/plan_llm_orbita_cdt_fase0.md`. Rama `exp/orbita-llm` (desde `exp/orbita`). Código v4 sin cambios; ÓRBITA-v1 congelada.

## Coste medido (antes de fijar tamaños)
`orbita gemma-embed` (candle, CPU, Gemma 2 2B it **Q4_K_M**, con `run_v6` ocupando ~6 de 8 núcleos):
- 26/26 capas: 60 textos en 1001.8 s de pared incluyendo carga (≈ 16.7 s/texto).
- 13/26 capas (`--exit-after 12`): 12 textos, 448 tokens, carga 2.71 s, 101.80 s de codificación → **8.48 s/texto, 4.40 tok/s**. RSS ≈ 1.9 GB por proceso.

## Desviaciones respecto del plan (decididas por coste, antes de resultados)
1. Representación: media sobre tokens (sin BOS) del estado oculto **tras la capa 13 de 26** (`exit_after=12`), no de la capa final.
2. P = **ridge supervisado a c** (estandarización + ridge dual; α por CV 5-fold agrupada por contexto en {1e-2,…,1e3}) sobre el pool. Las pérdidas JEPA/InfoNCE del plan **no se corren** en esta fase.
3. Tamaños: pool de 8 familias × 16 contextos × 3 idiomas × 2 plantillas = 768 textos; evaluación de 16 ejemplos TEST iterables por seed (los 8 primeros rotate y 8 primeros reflect) × 6 textos.
4. Sin GPU; 3 procesos Gemma en paralelo, `setsid nohup`, nice 0.

## Datos
- Textos: `orbita/llm/make_texts.py` (es/en/fr; decimales con coma en es/fr). Plantillas 0–1 sólo pool (semilla 0xC0DE, distribución TRAIN sin el hueco); plantillas 2–3 sólo evaluación (redacción nunca vista por P).
- Contextos de evaluación: los de los ejemplos TEST iterables (entradas de test, no objetivos), vía `orbita ctx-dump`.

## Condiciones (por ejemplo)
TRUE (c verdadero), GEMMA (ĉ = P·Gemma(texto), sin redondear ni proyectar a one-hot), NGRAM (hash de n-gramas de caracteres 1–3 → 2304 + mismo ridge), RAND (familia y p aleatorios, semilla 0xBAD), GEMMA_SWAP (texto de otro ejemplo de la misma familia; para H-L2).
Modelo: pesos ORBITA-v1 publicados `w_<seed>.json` de DEV/CONFIRM. Rollout libre `orbita llm-eval`, h ∈ {1,2,4,8,16,32}, acierto = criterio v4 (`m::correct`).

## Hipótesis y criterios (δ = 0,05)
Estadístico por seed; IC95 bootstrap pareado sobre las 16 seeds (10 000 remuestreos, semilla 12345). Se **acepta** si se cumple en DEV y CONFIRM; se **refuta** si en CONFIRM se cumple la condición opuesta indicada; si no, **inconclusa**.
- **H-L1** Δ = acc16(GEMMA) − acc16(TRUE). Acepta: IC95 inferior > −δ. Refuta: IC95 superior < −δ.
- **H-L2** D = mediana de ‖z16(tᵢ) − z16(tⱼ)‖/‖z16(TRUE)‖. Acepta: IC95 de D_same − D_diff < 0 **y** media D_same < 0,05. Refuta: IC95 de D_same − D_diff ≥ 0 entero (inferior ≥ 0).
- **H-L3** (a) para cada par de idiomas, IC95 de la diferencia de acc16 dentro de (−δ, δ); (b) acc16(GEMMA) − acc16(NGRAM) y − acc16(RAND) con IC95 > 0. Acepta: (a) y (b). Refuta: algún par con IC95 fuera de ±δ por completo, o GEMMA − NGRAM con IC95 < 0.
- **H-C1** sobre GEMMA: (a) familia inferida por el teacher consolidado (argmin_f rel_err(teacher(f, p̂, x), ÓRBITA₁(x, ĉ))) − 1/8 con IC95 > 0; (b) AUROC de `teacher_div16` = rel_err(ÓRBITA₁₆, teacher¹⁶(f̂, p̂)) para predecir fallo de ÓRBITA en h16, AUROC − 0,5 con IC95 > 0 (seeds sin ambas clases se excluyen y se reporta n). Acepta: (a) y (b). Refuta: (b) con IC95 < 0 o (a) con IC95 ≤ 0.
Auditoría: 0 queries durante `llm-eval` (reportado por seed).

## Orden
pool → fitP → DEV (textos, embeddings, ctx, eval, análisis) → CONFIRM idéntico, sin cambiar nada tras ver DEV.
