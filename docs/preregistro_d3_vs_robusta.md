# Prerregistro — Opción A (D3-num) vs Opción B (ÓRBITA-robusta)

**Comprometido antes de ver resultados** (2026-10-16, CDMX). Base: D2 (`docs/resultados_encoder_d2.md`, f7d1385). Código: `orbita/llm/d3.py`; `orbita/train_orbita.py` gana las opciones `--init`, `--p_noise` y `--noise_ramp` (los valores por defecto no cambian el comportamiento previo). v4 sin cambios. Logs y `.done` en `/workspace/orbita-data/llm_d3/`; artefactos en `artifacts/orbita/llm_d3/`.

Verificación previa (22:56 CDMX): load 0,19; ningún `run_v6`, gemma-embed ni entrenamiento activo; RAM disponible ~3,9 GB.

## Opción A — D3-num (sin entrenamiento nuevo)
ĉ = [one-hot del argmax de familia del ĉ de D2 ya calculado; p̂ leído del texto por un **parser de números**]. El parser toma los dos primeros números decimales en orden (regex `-?\d+([.,]\d+)?`, acepta coma o punto decimal); las palabras-número ("uno", "two"…) se ignoran. p̂ se recorta a [−1,1]. Si encuentra menos de dos números, usa el p̂ de D2 y el caso se cuenta como `parse_fail`. Se evalúa con ÓRBITA-v1 congelada. No se entrena ninguna cabeza de regresión: el parser es la lectura explícita.

## Opción B — ÓRBITA-robusta
Fine-tune de cada `w_<seed>.json` de ÓRBITA-v1 con `train_orbita.py`:
- `--init w_<seed>.json --steps 64000 --lr 3e-4 --lr_final 3e-5 --lam_h 0.01 --mh_x 1`;
- ruido gaussiano en p del contexto de entrenamiento (TRAIN y replay; los objetivos no cambian), σ con rampa lineal de 0 a **0,15** en 32 000 pasos y después constante (`--p_noise 0.15 --noise_ramp 32000`).

Se conecta al ĉ de D2 tal cual. Se reporta su acc con c real en el subconjunto TEST iterable y en el protocolo completo `orbita eval` (§16: acc a 1 paso en TEST + gate E33).
Coste medido antes de escalar: 2000 pasos en 12 s (1 hilo, CPU libre). Se estiman ~6,4 min por seed, con 8 en paralelo.

## Evaluación
16 seeds DEV y luego 16 CONFIRM, idénticas; 16 ejemplos TEST iterables × 6 textos (plantillas 2–3), `orbita llm-eval`. IC95 bootstrap pareado por seed (10 000, semilla 12345). 0 queries.
- Con ÓRBITA-v1: TRUE, RAND, D2, D1_POOL2, D3NUM.
- Con ÓRBITA-B: TRUE, D2, D1_POOL2.

## Hipótesis (δ = 0,05). Se acepta si se cumple en DEV y en CONFIRM; se refuta si se cumple la condición de refutación en CONFIRM; si no, inconclusa.
- **H-A1**: acc16(v1+D3NUM) − acc16(v1+D2), IC95 > 0. Se refuta con IC95 < 0.
- **H-A2**: acc16(v1+D3NUM) − acc16(v1+TRUE), IC95 inferior > −δ. Se refuta con IC95 superior < −δ.
- **H-B1**: acc16(B+D2) − acc16(v1+D2), IC95 > 0. Se refuta con IC95 < 0.
- **H-B2** (B no pierde con c real): acc16(B+TRUE) − acc16(v1+TRUE), IC95 inferior > −δ. Se refuta con IC95 superior < −δ.
- **H-B3**: acc16(B+D2) − acc16(v1+TRUE), IC95 inferior > −δ. Se refuta con IC95 superior < −δ.
- **H-C1** (descriptivo, para A y B): familia inferida por el teacher − 1/8 y AUROC − 0,5.

## Regla de selección (fijada aquí)
1. **Elegibilidad:** B no es elegible si H-B2 queda refutada en CONFIRM (pierde más de δ con c real). A siempre es elegible.
2. **Primaria:** en CONFIRM, acc16(A) − acc16(B), donde A = v1+D3NUM y B = B+D2, IC95 pareado. Si el IC95 es > 0, gana A; si es < 0, gana B.
3. **Secundaria** (si el IC95 cruza 0): la misma regla con acc8 en CONFIRM.
4. **Desempate final:** **A**, porque conserva ÓRBITA-v1 sin reentrenar y es más barata.
