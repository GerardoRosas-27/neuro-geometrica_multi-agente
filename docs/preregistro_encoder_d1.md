# Prerregistro — encoder D1 agnóstico al lenguaje (H-E0…H-E5, H-C1)

**Comprometido antes de ver resultados** (2026-10-16, CDMX). Diseño: `docs/encoder_agnostico_orbita.md` (54432f9). Base: Fase 0 (`docs/resultados_orbita_llm.md`, 5827f42). Código: `orbita/llm/d1.py`. ÓRBITA-v1 congelada (pesos publicados `w_<seed>.json`); v4 sin cambios. Logs y estado en `/workspace/orbita-data/llm_d1/`; artefactos en `artifacts/orbita/llm_d1/`.

## Coste medido antes de escalar
Prueba de humo (seed 0xA400 DEV, 60 pasos × 2 encoders GEMMA+NGRAM, 1 hilo, CPU sin carga): 9,1 s de pared, incluida la carga. Con 3000 pasos se esperan algunos minutos por seed; se ejecutan 8 seeds en paralelo.

## Encoder D1 (fijo)
Entrada: embeddings de Fase 0 (Gemma 2 2B Q4_K_M, media de tokens tras la capa 13/26), estandarizados con las estadísticas del pool. MLP 2304→128 (tanh). Cabezas: familia (8) con **one-hot straight-through** (forward = one-hot del argmax, gradiente de softmax) y p̂ = tanh(·) ∈ [−1,1]². Así ĉ es siempre un contexto válido.
Pérdida: CE(familia) + 10·‖p̂−p‖² + 1·L_T + 0,1·InfoNCE(τ = 0,1).
- L_T = media sobre k ∈ {1,2,4,8} de ‖Φᵏ(x,ĉ) − sg[Φᵏ(x,c)]‖²/‖Φᵏ(x,c)‖², recortado a 10, con ÓRBITA-v1 congelada de **esa seed** y objetos x tomados sólo de su TRAIN (8 por texto).
- InfoNCE: dos textos distintos del mismo contexto (idioma o plantilla), negativos dentro del lote.
Optimización: Adam, lr 1e-3, 3000 pasos, lote de 32 contextos, semilla seed^0xD1. Datos: pool de Fase 0 (768 textos, plantillas 0–1). **Nunca TEST.** Mismo encoder y pérdidas sobre n-gramas (D1_NGRAM) como control.

## Evaluación
Como Fase 0: 16 ejemplos TEST iterables por seed × 6 textos (plantillas 2–3 nunca vistas, es/en/fr), `orbita llm-eval`, h ∈ {1,2,4,8,16,32}. Condiciones: TRUE, GEMMA_F0 (P ridge de Fase 0), RAND, D1, D1_NGRAM, D1_SWAP (texto de otro ejemplo de la misma familia, para D_same/D_diff). IC95 bootstrap pareado sobre 16 seeds (10 000, semilla 12345). DEV primero; CONFIRM idéntico, sin cambios. Auditoría: 0 queries.

## Hipótesis (δ = 0,05). Se acepta si se cumple en DEV y CONFIRM; se refuta si se cumple la condición de refutación en CONFIRM; en otro caso, inconclusa.
- **H-E0** (sin entrenar): c verdadero con ruido gaussiano en p, σ ∈ {0,005; 0,01; 0,02; 0,05}, 6 draws por ejemplo (semilla 0xE0). σ* = mayor σ con IC95 inferior de acc16(σ) − acc16(TRUE) > −δ en ambas fases. Se acepta si existe σ* (al menos σ = 0,005). Se refuta si con σ = 0,005 el IC95 superior < −δ en CONFIRM. Además se reporta max|p̂−p| mediano de D1 frente a σ*.
- **H-E1**: acc16(D1) − acc16(GEMMA_F0), IC95 > 0. Se refuta con IC95 < 0.
- **H-E2**: acc16(D1) − acc16(TRUE), IC95 inferior > −δ. Se refuta con IC95 superior < −δ.
- **H-E3**: (a) los tres pares de idiomas con IC95 dentro de (−δ, δ); (b) acc16(D1) − acc16(RAND) con IC95 > 0; (c) D_same − D_diff con IC95 < 0. Se acepta si se cumplen (a), (b) y (c). Se refuta si algún par queda entero fuera de ±δ, o si (c) da IC95 > 0.
- **H-E4**: acc16(D1) − acc16(D1_NGRAM), IC95 > 0. Se refuta con IC95 < 0.
- **H-E5** (D2): **no se corre** en este lanzamiento; queda NOT_RUN. Se decidirá con H-E0 y el error de p̂.
- **H-C1** (CDT-RQM-EPR, consolidación de la seed, condición D1): (a) familia inferida por el teacher − 1/8, IC95 > 0; (b) AUROC(`teacher_div16` → fallo de ÓRBITA en h16) − 0,5, IC95 > 0, con al menos 8 seeds evaluables (que tengan ambas clases); con menos de 8, (b) es no evaluable. Se acepta si (a) y (b). Se refuta si (b) da IC95 < 0.
