# ÓRBITA-Nova-2 · Experimento 1 — C4 bayesiano (preregistro)

Rama `exp/orbita-nova2` (desde `exp/orbita-nova` @ `04dd2e1`). Aprobado por Bruster (Nova-2). Fecha: 2026-10-24 (America/Mexico_City, UTC−6).
**No modifica** `docs/orbita_nova/PREREG.md` (`f64b289`). Este documento se congela con su commit **antes** de generar cualquier dato DEV/CONFIRM de Nova-2. Un solo experimento; el exp2 (base compartida A,B + c) y exp3 (BOCD) no se abren aquí.

Regla dura: entre DEV y CONFIRM no se cambia código, hiperparámetros, umbrales, familias, métricas ni criterios. No hay calibración nueva: τ_C6 y τ_LAT se **reutilizan** congelados (abajo).

## 0. Base corregida (diagnóstico de los fallos del harness Nova)

Ver [`DIAGNOSTICO_HARNESS.md`](DIAGNOSTICO_HARNESS.md). Ambos fallos son **bugs**, y sus arreglos forman la base de Nova-2:

1. **C3 inactivo dentro de NOVA**: en `nova_train.rs` los brazos NOVA* nunca llamaban al filtro de innovación (`InnovFilter`); NOVA y NOVA−C3 ejecutaban el mismo código (por eso 16/16 idénticos). Arreglo: NOVA (y NOVA_BAYES) filtran el estado con C3 en los k pasos observados y hacen free-run con el operador elegido (adaptado si C5 acepta, ÓRBITA si no) los 16−k pasos restantes, exactamente como ÓRBITA+C3. NOVA−C3 conserva el rollout desde x0 (= NOVA de f64b289, sirve de control de regresión).
2. **LAT_EP ≡ ÓRBITA**: LAT_EP comparaba la sorpresa S_known (rel_err medio, escala ~1e-2) con τ_C6 = 0,93866 (distancia L2 de firmas C4, otra escala) → siempre "conocido" → siempre ÓRBITA. Arreglo: LAT_EP usa su propio τ_LAT(k) congelado del ciclo latente (`artifacts/orbita/latente_largo_plazo/tau.json`: k=2 0,0146557; k=4 0,0220203; k=8 0,0371425) y λ0 = 1e-4 (`selected.json`).

## 1. Hipótesis

**B1 (primaria)**: un C4 bayesiano —posterior Laplace/Gauss-Newton cerrada sobre ΔK = UVᵀ (r=2) con la entrada recuperada de C6 como media del prior, y C5 que acepta por verosimilitud predictiva en los pasos retenidos (no vistos por el ajuste)— reduce el error en familias nuevas frente a NOVA (base corregida) con **k=2**.
Motivación (no probada): MetaKoopman (arXiv:2607.26345), BONE (arXiv:2411.10153). En NOVA, C4 se ajusta con los k pasos completos y C5 valida sobre pasos que el ajuste ya vio (fuga); con k=2 esto deja a C5 casi sin poder de rechazo.

## 2. Brazo nuevo NOVA_BAYES (definición congelada)

Por episodio, con contexto re-identificado `cg` (igual que NOVA), (k_fit, k_val) = (1,1) | (3,1) | (6,2) para k=2|4|8:
- Firma y recuperación C6: idénticas a NOVA (firma = C4 desde `lora_prior`, recuperación L2 ≤ τ_C6). NOVA_BAYES mantiene **su propia** biblioteca, construida en el mismo flujo con la misma consolidación (N=50, clustering τ_C6, admisión si mejora retenida y no-daño).
- Prior: si C6 recupera entrada → μ = θ_lib, λ = **1e-3** (= λ_m congelado de LAT); si no → μ = `lora_prior`, λ = **1e-6** (= λ_C4).
- σ² = MSE de residuos normalizados de ÓRBITA en los k_fit pasos (empírico por episodio; piso 1e-10).
- Posterior: θ̂ = MAP GN/LM (`fit_gn`, 12 iter.) con k_fit pasos; precisión P = (J_fitᵀJ_fit + λI)/σ².
- C5 bayesiano: log-verosimilitud predictiva gaussiana en los k_val pasos retenidos (rollout desde x0), varianza σ² + diag(J_v P⁻¹ J_vᵀ), contra ÓRBITA con varianza σ². Acepta si ll_ada > ll_base y ‖I+ΔK‖ ≤ 1+ε (ε=0,5).
- Si acepta: θ final = MAP con los k pasos y el mismo prior; si no: ÓRBITA. Ambos con C3 como en NOVA.
- En consolidación: val_err = −ll_ada, base_err = −ll_base.

## 3. Datos, semillas y métrica

Mismo generador, flujo y held-out que Nova (`gen_stream`, `gen_heldout_n`; XOR documentados en `nova_train.rs`): flujo n=2400 (80 % conocidas 0–7 / 20 % nuevas {8,10,11}, k∼U{2,4,8}), H = 16 episodios por familia, checkpoints n ∈ {0,150,300,600,1200,2400}. Pesos ÓRBITA congelados `artifacts/orbita/{dev,confirm}/weights_0x*.json`.
Semillas: DEV `0xA400–0xA40F`, CONFIRM `0xB400–0xB40F` (16 + 16). τ_C6 = **0,9386641928884383** (calibrado en Nova DEV, `artifacts/orbita/nova_full/tau.json`; no se recalibra).

**Métrica primaria** (misma que Nova): por seed, mediana de `log(rel_err h16)` sobre episodios H de nuevas {8,10,11}; menor = mejor. Evaluada en n=2400. Secundarias: acc16 (rel_err<0,05) en conocidas y nuevas; known_med_log; curva n.

## 4. Brazos

| brazo | definición |
|---|---|
| ÓRBITA | núcleo congelado, free-run desde x0 |
| ÓRBITA+C3 | filtro de innovación en k pasos + free-run base |
| LAT_EP | adaptador latente por episodio con τ_LAT(k) (arreglado) |
| NOVA | C3+C4+C5+C6 (base corregida, C3 activo) |
| NOVA−C3 | NOVA sin C3 (= NOVA de f64b289; control de regresión) |
| **NOVA_BAYES** | §2 |
| NOVA_BAYES−C6 | NOVA_BAYES sin recuperación (prior siempre `lora_prior`, λ=1e-6): aísla el aporte de C6-como-prior |

## 5. Análisis y criterios

Contraste = media sobre las 16 seeds de la diferencia pareada por seed; IC95 bootstrap pareado percentil, 10 000 réplicas, semilla 12345 (`scripts/analyze_nova2_exp1.py`). δ = 0,05. Negativo = mejora de NOVA_BAYES.

- **B1 (primaria)**: NOVA_BAYES − NOVA, nuevas, log, **k=2**: aceptada si IC95 sup < 0 en DEV **y** CONFIRM; refutada si en CONFIRM IC95 inf > 0; inconclusa en otro caso.
- **B2 (secundarias)**: lo mismo con k=4 y k=8 (reportadas por separado; sin corrección múltiple; no vetan B1).
- **B3 (no-daño)**: NOVA_BAYES − NOVA en conocidas, acc16, k=2/4/8: inf. IC95 > −δ en ambas fases.
- **B4 (C6 como prior)**: NOVA_BAYES − NOVA_BAYES−C6, nuevas, log, k=2: IC95 < 0 (reportada; no veta).
- Controles reportados: NOVA − NOVA−C3 (¿C3 aporta ahora?), NOVA − LAT_EP, NOVA − ÓRBITA, NOVA_BAYES(2400) − NOVA_BAYES(0).

**Éxito del exp1** ⇔ B1 aceptada ∧ B3 aceptada. Se publica `RESULTADOS_EXP1.md` en cualquier caso.

## 6. Ejecución

Binario `orbita nova2-eval`; script `orbita/scripts/nova2_exp1.sh` (nohup, `nice 10`, reanudable por `.done`, seeds en paralelo). Salida `artifacts/orbita/nova2_exp1/{smoke,dev,confirm}/`. Orden: smoke corto (no analizado) → DEV completo → CONFIRM completo, mismo binario y commit. Sin `run_v6`, sin tocar `main`, sin force push.
