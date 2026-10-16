# Prerregistro — encoder D2 (atención por token), H-E5 y repeticiones con D2

**Comprometido antes de ver resultados** (2026-10-16, CDMX). Base: D1 (`docs/resultados_encoder_d1.md`, 481f04a). Código: `orbita/llm/d2.py`; extracción `orbita gemma-embed --per-token`. ÓRBITA-v1 congelada; v4 sin cambios. Logs/estado: `/workspace/orbita-data/llm_d2/`; artefactos: `artifacts/orbita/llm_d2/`.

## Coste medido (antes de escalar)
Extracción por token (Gemma 2 2B Q4_K_M, 13/26 capas, 1 proceso, CPU libre): 10 textos, 441 tokens, carga 3,71 s, **0,601 s/texto** (73,4 tok/s), ≈ 397 KB/texto en f32. RAM disponible medida antes: ~2,8 GB → **1 proceso Gemma** (~1,9 GB) para no provocar OOM.

## Datos
- **Pool2** (entrenamiento): los mismos 128 contextos del pool de Fase 0 (semilla 0xC0DE) con **plantillas 0, 1, 4, 5** (dos plantillas nuevas, 4–5, por idioma) × es/en/fr = 1536 textos. Las plantillas 0–3 no cambian (verificado: pool de Fase 0 y textos TEST regenerados son idénticos byte a byte).
- **TEST**: los mismos textos de evaluación de Fase 0 y D1, con **plantillas 2–3, que nunca se usan en entrenamiento** (split por plantilla). 1536 textos por fase.
- Representación: estados ocultos **por token** (sin BOS) tras la capa 13/26 (`exit_after=12`), la misma capa que en Fase 0 y D1, así D2 vs D1 aísla el pooling. Total ≈ 4608 textos.

## Modelos (hiperparámetros fijos, idénticos a D1 salvo el lector)
- **D2**: estandarización por dimensión (estadísticas de los tokens de pool2) → claves 2304→32, valores 2304→64, 3 queries entrenables (familia, p₁, p₂) con atención softmax sobre tokens → concatenación 192 → tanh 128 → familia one-hot straight-through + p̂ = tanh.
- **D1_POOL2** (control): arquitectura D1 (media de tokens → MLP 128) entrenada sobre pool2, para separar el efecto de la atención del efecto de las plantillas nuevas.
- Pérdidas y optimización como D1: CE + 10·‖p̂−p‖² + 1·L_T(k ∈ {1,2,4,8}, ÓRBITA de la seed, objetos TRAIN) + 0,1·InfoNCE(τ 0,1); Adam 1e-3, 3000 pasos, lote 32, semilla seed^0xD2.
- Baselines de evaluación sin cambios: TRUE, RAND, GEMMA_F0, **D1** y **D1_NGRAM** (los ĉ ya calculados en D1, `/workspace/orbita-data/llm_d1/<fase>/ctxin_*.csv`).

## Evaluación
Igual que D1: 16 ejemplos TEST iterables × 6 textos, `orbita llm-eval`, IC95 bootstrap pareado sobre 16 seeds (10 000, semilla 12345), DEV y después CONFIRM idéntico. 0 queries.

## Hipótesis (δ = 0,05). Se acepta si se cumple en DEV y en CONFIRM; se refuta si se cumple la condición de refutación en CONFIRM; si no, inconclusa.
- **H-E5a** (D2 mejora la lectura de p y el acierto): max|p̂−p| mediano de D2 − D1 con IC95 < 0 **y** acc16(D2) − acc16(D1) con IC95 > 0. Se refuta si alguno de los dos tiene el IC95 entero del lado contrario.
- **H-E5b** (el efecto se debe a la atención, no a las plantillas nuevas): acc16(D2) − acc16(D1_POOL2) con IC95 > 0. Se refuta con IC95 < 0.
- **H-E2'**: acc16(D2) − acc16(TRUE) con IC95 inferior > −δ. Se refuta con IC95 superior < −δ.
- **H-E3'**: mismas tres condiciones que H-E3, sobre D2: (a) pares de idiomas dentro de ±δ, (b) D2 − RAND > 0, (c) D_same − D_diff < 0. Mismas reglas de refutación.
- **H-E4'**: acc16(D2) − acc16(D1_NGRAM) con IC95 > 0. Se refuta con IC95 < 0.
- **H-C1'** (CDT-RQM-EPR, condición D2): (a) familia inferida por el teacher − 1/8 > 0 y (b) AUROC − 0,5 > 0, con al menos 8 seeds evaluables. Se refuta si (b) tiene IC95 < 0.
