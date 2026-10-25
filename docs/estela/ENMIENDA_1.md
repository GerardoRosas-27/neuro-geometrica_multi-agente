# ESTELA v0 — Enmienda 1 (antes de DEV)

Rama `exp/estela`. Aprobada por Bruster («primero una enmienda escrita, luego lanzar»). Fecha: 2026-10-24 (America/Mexico_City, UTC−6).
Modifica `docs/estela/PREREG.md` (commit `893d2b2`) **sólo** en lo que se lista aquí; todo lo demás (familias, H, métricas primarias, brazos ÓRBITA/ÓRBITA+LAT, k_fit_val, bootstrap, δ, criterios S1–S4 salvo lo indicado) sigue igual.
Todos los valores de esta enmienda se fijan **en este documento y se publican con su commit antes de cualquier ejecución DEV**. Entre DEV y CONFIRM no se cambia nada (mismo binario y commit). 0 consultas TEST. Sin `run_v6`, sin tocar `main`, sin force push.

## 0. Semillas ya vistas (declaración explícita)

Las semillas DEV **0xA400 y 0xA401 ya se vieron en el smoke** (`docs/estela/SMOKE.md`, commit `0d8d9fa`, H reducido a 4/familia). Las señales del smoke en esas dos semillas motivaron esta enmienda. Tratamiento:
- El análisis DEV preregistrado se mantiene con las 16 semillas (n=16), **y además** se reporta un análisis de sensibilidad DEV **excluyendo 0xA400 y 0xA401** (n=14), marcado como tal en `ANALYSIS.json` (`dev_excl_smoke`) y en RESULTADOS.md. Si el veredicto DEV de algún criterio difiere entre n=16 y n=14, se señala explícitamente.
- CONFIRM (0xB400–0xB40F) no se ha visto nunca y es la fase decisiva (S3 exige DEV **y** CONFIRM; la refutación se decide en CONFIRM).

## 1. Cambios de diseño (valores fijados)

### 1.1 Codificador congelado (en lugar de subir el peso de la sonda)
Se elige **congelar el codificador g_θ** (w3, b3, w4, b4 = lifting de ÓRBITA) en todas las variantes; λ_probe se mantiene en **0,1**.
Justificación: en el smoke la pérdida JEPA bajaba mientras la sonda subía (0,017→0,034; 0,026→0,071) y acc16 conocidas caía de 0,19–0,28 a 0,03. Con objetivo = codificador online con stop-gradient, la JEPA se reduce moviendo el propio objetivo (deriva del codificador): en la traza de diagnóstico (§3) ‖Δg‖ crece hasta 4,8 (Full) y 14,2 (r0) en 3000 pasos. Subir λ_probe sólo compensaría la deriva con otra pérdida; congelar el codificador la elimina de raíz: el objetivo JEPA pasa a ser el latente de ÓRBITA del estado verdadero siguiente, fijo, y lo que se aprende (r_φ, h_ψ, P) es una corrección sobre ÓRBITA. Coste declarado: el «codificador iniciado desde ÓRBITA» deja de ser entrenable; SIGReg sólo actúa a través de los latentes de estados predichos re-codificados.

### 1.2 Tasa de aprendizaje
**lr = 3e-4** (antes 1e-3). Resto de Adam igual (β 0,9/0,999, ε 1e-8, recorte 1,0, 3000 pasos, BATCH 16, H_train 4).

### 1.3 Puntuación de sorpresa: escala, signo y umbral
En el smoke `s_E = E(E1(y)) − E(ẑ)` daba AUROC 0,31–0,42 (invertida) y τ_E escalas dispares (≈0,2 vs ≈120). El término aprendido h_ψ no tiene signo identificable respecto a la novedad. Nueva definición (fijada a priori, no ajustada a datos):
- `s_raw(k) = ln( media_{t<k, puntos} ½‖E1(y_t) − ẑ_t‖² + 1e-12 )`: el término cuadrático de la energía alrededor de la predicción ẑ_t a un paso desde lo observado (= E sin h_ψ). Es ≥ 0 antes del log y **mayor = más sorpresa por construcción** (peor predicción del estado observado).
- Estandarización por k con el conjunto de calibración propio (256 episodios familias 0–7, RNG `seed ⊕ 0xE57ECA11`, nunca H ni nuevas): `ŝ = (s_raw − μ_cal(k)) / σ_cal(k)` (σ poblacional; si σ<1e-12 se usa 1e-12).
- **Regla de disparo fija: E4 se dispara si ŝ > 1,645** (cuantil 0,95 de N(0,1)); sustituye a «cuantil 0,95 de s_E».
- AUROC de S4 se calcula con ŝ (monótona en s_raw, idéntica a la de s_raw).

### 1.4 S1 (anti-colapso) con codificador congelado
Con g congelado, el rango efectivo de E1(x0) es idéntico en todas las variantes y S1 sería trivial. S1 se mide sobre **ẑ_1 (latente predicho a un paso desde x0) en H conocidas** (mismos umbrales: media ≥ 0,8·M y ESTELA − ESTELA−SIGReg con IC95 inf > 0). Se reporta también el rango de E1(x0) como referencia.

## 2. Corrección de error: el control ESTELA-r0

**Síntoma (smoke):** r0 divergía (JEPA 0,18→4,3 y 0,35→6,8; ‖e2‖ 2,5–4,5; τ_E≈120), aunque con r_φ≡0 debería ser ÓRBITA congelada.
**Diagnóstico (seed 0xC400, §3):** la cinta es correcta (prueba de diferencias finitas pasa); el error era de especificación/implementación del control: v0 congelaba sólo r_φ pero seguía entrenando g_θ, h_ψ y P. Sin residual, la única forma de bajar la JEPA era mover el objetivo (‖Δg‖ 14,2) y usar el refinamiento por energía como residual sustituto (‖e2‖ 0→3,8), lo que realimenta el rollout y diverge (JEPA 0,25→5,0, sonda 0,003→0,79). Es decir, r0 no era el control declarado.
**Corrección:** ESTELA-r0 = **ÓRBITA congelada dentro de la ruta de ESTELA**: r_φ≡0, h_ψ≡0, P=[I 0], g = lifting de ÓRBITA, **sin entrenamiento**. Nueva prueba: el rollout de r0 es bit a bit `Orbita::step` (familias conocidas y nuevas). r0 conserva E4 (con su propia calibración ŝ), de modo que ESTELA − r0 mide lo que aportan r_φ, h_ψ y P aprendidos.

## 3. Comprobación de implementación fuera de DEV/CONFIRM

Sólo con la **seed 0xC400** (fuera de DEV 0xA400–0xA40F y CONFIRM 0xB400–0xB40F), usando los pesos ÓRBITA de 0xA400 (ya vistos en el smoke; no hay pesos ÓRBITA para 0xC400) y H reducido (4/familia), salida `artifacts/estela/sanity_c400/`. Uso: diagnóstico de r0 (traza con v0) y comprobar que la implementación de la enmienda corre sin NaN, r0 = ÓRBITA bit a bit y 0 consultas TEST. **Ningún valor de §1 se eligió ni se cambia con esta comprobación** (los valores están fijados en este commit, antes de ejecutarla). Sus números se reportan en RESULTADOS.md sólo como comprobación.

## 4. Resumen de valores congelados tras la enmienda

Codificador congelado (todas las variantes); lr=3e-4; λ_probe=0,1; λ_SIG=0,1; λ_REC=1,0; STEPS=3000; BATCH=16; H_train=4; clip=1,0; sorpresa ŝ (§1.3) con umbral 1,645; N_CAL=256; r0 = ÓRBITA congelada sin entrenar; S1 sobre ẑ_1; análisis DEV n=16 + sensibilidad n=14 sin 0xA400/0xA401.
