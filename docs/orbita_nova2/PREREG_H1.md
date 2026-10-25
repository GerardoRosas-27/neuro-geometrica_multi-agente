# ÓRBITA-Nova-2 · H1 NOVA_RET — selección de la biblioteca por error en k pasos, sin ajuste por episodio (preregistro)

Rama `exp/orbita-nova2` (desde `a6fc0b7`). Aprobado por Bruster (brief «Tras el veredicto de Nova», propuesta 1; MixER arXiv:2502.05335, KODE arXiv:2603.17416, Zebra arXiv:2410.03437). Fecha: 2026-10-24 (America/Mexico_City, UTC−6).
Este documento se congela con su commit **antes** de generar cualquier dato (smoke incluido). No modifica PREREG_EXP1/2/3 ni PASO0_C6. Módulo nuevo opt-in `orbita/src/nova2h1.rs`, subcomando `orbita nova2-h1`; exp1–3, Paso 0 y `nova-eval` no cambian.

Regla dura: entre DEV y CONFIRM no se cambia código, hiperparámetros, umbrales, familias, métricas ni criterios. No hay calibración nueva. 0 consultas TEST.

## 0. Base
Se conservan los arreglos de exp1 (`DIAGNOSTICO_HARNESS.md`): **C3 activo** en todos los brazos NOVA* (filtro en k pasos + free-run 16−k). LAT_EP no participa en H1. Núcleo ÓRBITA congelado (assert de pesos al final de cada seed).

## 1. Decisiones de diseño (fijadas aquí)
1. **Biblioteca = C6 legado** (la de exp2/exp3: `consolidate` de `nova_train`, firma θ, τ_C6 = 0,9386641928884383), construida por el flujo NOVA **idéntico al de exp3** (n=2400, consolidación cada 50, mismo `lora_prior`, misma recuperación con τ en el flujo). Motivo: el control NOVA queda idéntico al NOVA de exp2/exp3 (comparabilidad), y la biblioteca ΔK de `nova2p0` falló su smoke (1310–1535 entradas, sin hits en nuevas). NOVA_RET **no** escribe en la biblioteca: es sólo un brazo de evaluación sobre la biblioteca de NOVA en cada checkpoint.
2. **Candidatos**: como el Paso 0 mostró que la recuperación por firma con τ no devuelve nada cerca de las familias nuevas, NOVA_RET **no usa firma ni τ**. Se puntúan **todas** las entradas de la biblioteca por su error en los k pasos observados; top-4 = las 4 de menor puntuación; se elige la de menor puntuación de ese top-4. Se declara explícitamente que esto equivale a argmin sobre toda la biblioteca (el top-4 se registra sólo como diagnóstico y para H2 NOVA_MIX). Ninguna entrada se modifica: θ = θ_entrada, **cero ajuste por episodio** (no se llama a `fit_c4` ni a `fit_gn`).
3. **Puntuación** de una entrada θ_j: rollout libre de `lr_step(θ_j)` desde x0 durante k pasos con el contexto c de `ctx_for_eval`; s_j = media_t=1..k de ln(max(rel_err(ẑ_t, y_t),1e-12)), rel_err acotado a 10 y 10 si no finito. Empates: índice menor.
4. **Tras la selección**, idéntico a NOVA: C5 `arbitrate(θ*, kf, kv, EPS_SPEC)` + C7 `gate_nodamage`; si acepta, paso adaptado θ*, si no ÓRBITA; luego C3 en la evaluación h16. (La fuga de C5 —validar en pasos ya vistos— es la misma que en NOVA, así que el contraste es justo; se anota.)
5. **Biblioteca vacía** (n=0): NOVA_RET y NOVA_RET_SHUF = ÓRBITA+C3.

## 2. Brazos (evaluados en H por checkpoint y k ∈ {2,4,8})
- **ÓRBITA+C3** (control).
- **NOVA** (control; C4 + recuperación τ + C4 + C5 + C7 + C3, idéntico a exp3).
- **NOVA_RET** (§1).
- **NOVA_RET_SHUF** (control «biblioteca barajada»): la biblioteca se baraja y se toman como candidatos 4 entradas al azar (Xoshiro256**, semilla seed ⊕ 0x5EF1 ⊕ (n≪20) ⊕ (k≪16) ⊕ índice del episodio en H), se elige la de menor puntuación entre esas 4, y C5+C7+C3 igual. Conserva el mecanismo de selección por error en k pasos y destruye el ranking sobre la biblioteca completa.

## 3. Datos, semillas, métricas
Idénticos a exp3: `gen_stream` n=2400 (80 % conocidas 0–7 / 20 % nuevas {8,10,11}, k∼U{2,4,8}), H = `gen_heldout_n(seed,16)`, checkpoints {0,150,300,600,1200,2400}. Semillas DEV `0xA400–0xA40F` (pesos `artifacts/orbita/dev`), CONFIRM `0xB400–0xB40F` (pesos `artifacts/orbita/confirm`).
**Primaria**: por seed, mediana de log(rel_err h16) en episodios H de nuevas {8,10,11}, n=2400, k=2. Secundarias: lo mismo k=4/8, acc16 conocidas/nuevas, n=150 vs 2400. Diagnósticos: tasa de aceptación C5/C7 de NOVA_RET, nº de entradas distintas elegidas, pureza de familia del top-1 (familia dominante del soporte registrada en un espejo de la biblioteca que no altera `consolidate`), puntuación s* media.

## 4. Hipótesis y criterios
Contraste = media sobre 16 seeds de la diferencia pareada por seed; IC95 bootstrap pareado percentil, **10 000 réplicas, semilla 12345** (`scripts/analyze_nova2_h1.py`). δ = 0,05.
- **H1 (primaria)**: NOVA_RET − NOVA, nuevas, log, **k=2**, n=2400 < 0. **Aceptada** si IC95 sup < 0 en DEV **y** CONFIRM; **refutada** si en CONFIRM IC95 inf > 0; inconclusa en otro caso.
- **H1-ctrl (debe cumplirse para atribuir a la biblioteca)**: NOVA_RET − NOVA_RET_SHUF < 0 y NOVA_RET − ÓRBITA+C3 < 0 (nuevas, log, k=2): se reportan con IC95 en ambas fases; si H1 se acepta pero NOVA_RET − SHUF no tiene IC95 sup < 0, se informa como «ganancia no atribuible al ranking de la biblioteca».
- **Secundarias (no vetan)**: H1 con k=4 y k=8; N2: NOVA_RET(n=2400) − NOVA_RET(n=150) nuevas log k=2.
- **No-daño (veta)**: NOVA_RET − NOVA en conocidas 0–7, acc16, k=2/4/8: IC95 inf > −δ en ambas fases.
**Éxito de H1** ⇔ H1 aceptada ∧ no-daño aceptado. `RESULTADOS_H1.md` se publica en cualquier caso.

Nota honesta a priori: en exp1–3 NOVA ya ajusta θ por episodio con C4 (λ=1e-6), lo que en k=2 es casi interpolación; NOVA_RET sólo puede ganar si alguna entrada de la biblioteca (construida mayoritariamente con conocidas) generaliza a las nuevas mejor que ese ajuste. Puede salir refutada.
