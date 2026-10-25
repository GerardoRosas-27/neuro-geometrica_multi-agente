# ESTELA v0 — arquitectura de estados sin tokens, complementaria a ÓRBITA (preregistro)

Rama `exp/estela` (desde `main` 590dd93). Aprobado por Bruster (inicio de ESTELA). Fecha: 2026-10-24 (America/Mexico_City, UTC−6).
Base de diseño: `arquitectura-sin-tokens-brief.md` (ESTELA = Espacio de ESTados Latentes Agnóstico).
Este documento se congela con su commit **antes** de generar cualquier dato DEV/CONFIRM. El smoke (§8) sólo mide tiempo y estabilidad numérica; no se analiza ni puede cambiar nada de este documento.

**Regla dura:** todos los hiperparámetros están fijados a priori en `orbita/src/estela/{mod,model}.rs` (copiados en §4). Entre DEV y CONFIRM no se cambia código, hiperparámetros, umbrales, familias, métricas ni criterios (mismo binario y commit). **0 consultas TEST** (`v4_provenance`, el binario aborta si ≠ 0). Núcleo ÓRBITA congelado (se verifica `raw` al final de cada seed). Sin `run_v6`, sin tocar `main`, sin force push.

## 0. Paso 0 — compuerta de paridad (ya superada en el commit de implementación)

Motivo: en Bytes la ruta de evaluación no era la del entrenamiento (~5,0 vs ~1,65 bytes/parche). En ESTELA existe **un único forward** (`estela::model::step`, genérico en el escalar `S`): la evaluación lo instancia con `f64` y la pérdida de entrenamiento con `Var` (cinta de autodiferenciación inversa propia, `estela/ad.rs`), que ejecuta las mismas operaciones f64 en el mismo orden.
Pruebas (`cargo test --release estela`, `orbita/src/estela/tests.rs`):
1. ESTELA en inicialización (r_φ=0, h_ψ=0, P=[I 0], lifting de ÓRBITA) reproduce `Orbita::step` **bit a bit** en 16 pasos, familias conocidas y nuevas (contexto `ctx_for_eval`), variantes con K.
2. El forward con cinta da **los mismos bits** que el forward f64 con parámetros perturbados (r_φ, h_ψ, P ≠ init).
3. Las predicciones q̂ que entran en la pérdida (`batch_loss`) son bit a bit las de `rollout` (evaluación) tras entrenar 20 pasos, en las 5 variantes.
4. Ida y vuelta de parámetros en disco con bits exactos. (Esta prueba detectó que `serde_json` sin `float_roundtrip` **no** conserva los bits de f64; los parámetros se guardan como bits u64 en hex.)
5. Mismo generador y formato de datos en entrenamiento, calibración y evaluación (6 puntos, horizonte 16, c∈R^10).
6. Gradiente de la cinta = diferencias finitas (codificador objetivo fijo, por ser stop-gradient).
`orbita estela-paso0` sobre los 32 pesos (DEV+CONFIRM): 0 discrepancias en 1 419 264 valores (`artifacts/estela/paso0.json`). Además `estela-run` repite (1)+(2) en cada seed antes de entrenar y aborta si falla.

## 1. Arquitectura (v0, sin memoria/biblioteca)

Por punto q∈R² del estado (nube de 6 puntos) con contexto c∈R^10 (ninguna entrada es un token; sin Gemma, sin vocabulario):
- **E1** codificador `z = [q; g_θ(q,c)] ∈ R^8`, g_θ MLP 12→16→6 tanh, **iniciado con el lifting de ÓRBITA** (w3,b3,w4,b4), entrenable.
- **E2** `u = K_ÓRBITA(c)·z + r_φ(z)`; K = núcleo Koopman de ÓRBITA (Cayley·rotación·escala + β), **congelado**; r_φ MLP 8→32→8 tanh con capa de salida **= 0** al inicio. `ẑ0 = B(u)`, B = corrector Brenier de ÓRBITA (8 dims), **congelado** — es la parte de ÓRBITA que lleva u al espacio de z (lectura de «encoder iniciado desde el Brenier»: el mapa estado→latente→estado de ESTELA parte exactamente de ÓRBITA).
- **E3** energía `E(z) = ½‖z − ẑ0‖² + h_ψ(z)`, h_ψ(z)=Σ_k e2_k tanh(e1_k·z+b_k) (16 unidades, e2=0 al inicio). Refinamiento `ẑ ← ẑ − α∇E(ẑ)`, α=0,5, 2 pasos. Sorpresa por episodio `s_E(k)` = media, sobre los k pasos observados y los puntos, de `E(E1(y_t)) − E(ẑ_t)` con predicción a un paso desde lo observado.
- **E5** sonda lineal `q̂' = P ẑ`, P∈R^{2×8} iniciada en [I 0].
- Rollout: el paso siguiente re-codifica q̂' (como ÓRBITA). Entrenamiento y evaluación usan este mismo rollout.
- **E4** (adaptación en test): si `s_E(k) > τ_E(k)` se ajusta un adaptador LAT sobre ESTELA (`u += Aᵀξ`, ξ=[z, x², xy, y², 1], 96 parámetros, Gauss-Newton/LM `latent::fit_gn`, prior 0, λ0=1e-4) con los primeros k_fit pasos y se valida en los k_val pasos **retenidos** (k_fit_val: k=2→(1,1), 4→(3,1), 8→(6,2)). Se acepta sólo si el error retenido mejora estrictamente frente a ESTELA sin adaptar; si se acepta, se reajusta con los k pasos. Si no, ESTELA sin adaptar.
- `τ_E(k)` = cuantil 0,95 de `s_E(k)` en un conjunto de calibración propio (256 episodios de familias 0–7, RNG `seed ⊕ 0xE57ECA11`), calculado automáticamente tras entrenar. Nunca usa H ni familias nuevas.

## 2. Entrenamiento (por seed y variante; sólo familias conocidas 0–7)

Datos: 2048 trayectorias (familias 0–7 equilibradas, contexto verdadero, horizonte 16, RNG `seed ⊕ 0xE57E7A11`, mismo `make_ep` que H). Minibatch: 16 ventanas (episodio, t0∈[0,12]) del mismo flujo RNG (`seed ⊕ 0xE57E0B70`) para todas las variantes.
Pérdida (multi-paso, H_train=4, desplegando el rollout de ESTELA desde el estado observado s_t0):
- JEPA: media de ‖ẑ_{t0+h} − sg(E1(s_{t0+h}))‖² (codificador objetivo = mismos pesos sin gradiente).
- `+ 0,1·SIGReg` (LeJEPA: Epps–Pulley contra N(0,1) de 16 proyecciones aleatorias por paso de las coordenadas aprendibles z[2..8] de todos los estados del minibatch; t∈[0,4], 17 nodos, peso e^{−t²/2}).
- `+ 0,1·sonda` (‖q̂_{t0+h} − s_{t0+h}‖²/‖s_{t0+h}‖²).
- **Sin reconstrucción ni vocabulario** (salvo el control REC).
Adam lr 1e-3 (β 0,9/0,999, ε 1e-8), recorte de norma 1,0, **3000 pasos**, init determinista (`seed ⊕ 0xE57E1A00`).

## 3. Brazos

- **ÓRBITA**: ÓRBITA congelada, contexto `ctx_for_eval` (conocidas: verdadero; nuevas: reid con ÓRBITA, k pasos), rollout 16 desde x0.
- **ÓRBITA+LAT**: = LAT_EP de Nova-2 (τ_LAT(k) = 0,0146557/0,0220203/0,0371425; λ0=1e-4; ajuste con `ctx8(p̂)`).
- **ESTELA**: E1–E5 (§1) con LAT disparado por sorpresa y árbitro en pasos retenidos.
- Controles (mismo entrenamiento, mismo E4, cada uno con su propio τ_E):
  - **ESTELA−SIGReg** (λ_SIG = 0).
  - **ESTELA-REC**: + 1,0·(sonda multi-paso + reconstrucción del estado actual ‖P·E1(s_t) − s_t‖²): objetivo en el espacio de estados.
  - **ESTELA-MLP** (sin Koopman): `u = z + r_φ(z)` (K sustituido por la identidad).
  - **ESTELA-r0**: r_φ ≡ 0 congelado (se entrenan g_θ, h_ψ, P).
- Reportados: `*_base` (cada variante sin E4).

## 4. Hiperparámetros congelados

RH=32, EH=16, NREF=2, α=0,5, STEPS=3000, BATCH=16, H_train=4, lr=1e-3, clip=1,0, λ_SIG=0,1, λ_probe=0,1, λ_REC=1,0 (sólo REC), N_SLICES=16, N_TRAIN=2048, N_CAL=256, cuantil τ_E=0,95, λ0 LAT=1e-4, k_fit_val como §1. Ningún valor se ajustó con datos; no hay búsqueda de hiperparámetros.

## 5. Datos de evaluación, semillas y métricas

Held-out H idéntico al de Nova (`gen_heldout_n`, RNG `seed ⊕ 0x4E0A501D`): 16 episodios por familia en conocidas 0–7 y **nuevas {8, 10, 11}** (176), horizonte 16; k ∈ {2, 4, 8} pasos observados. Familias nuevas sólo en evaluación. Pesos ÓRBITA por seed: `artifacts/orbita/{dev,confirm}/weights_0x….json`.
Semillas: **DEV 0xA400–0xA40F**, **CONFIRM 0xB400–0xB40F** (16 cada una).
- **Primaria**: por seed, mediana de `log(rel_err h16)` (rel_err acotado a 10; no finito = 10) en episodios H de nuevas {8,10,11}, **k=4**; menor = mejor.
- acc16 = fracción con rel_err h16 < 0,05.
- Rango efectivo de z = exp(entropía de los autovalores normalizados de la covarianza de z) sobre los puntos de x0 de H conocidas (M=8).
- AUROC de detección en H (positivo = nueva) con k=4: `s_E` de ESTELA frente a `s_known` de ÓRBITA.

## 6. Hipótesis y criterios

Contraste = media sobre 16 seeds de la diferencia pareada por seed; IC95 bootstrap pareado percentil, **10 000 réplicas, semilla 12345** (`scripts/analyze_estela.py`). δ = 0,05.
- **S3 (primaria)**: ESTELA − ÓRBITA+LAT, nuevas, log rel_err h16 mediana, k=4 < 0. **Aceptada** si IC95 sup < 0 en DEV **y** CONFIRM; **refutada** si en CONFIRM IC95 inf > 0; inconclusa en otro caso.
- **S2 (no-daño, veta)**: ESTELA − ÓRBITA, acc16 en conocidas 0–7, k=4: IC95 inf > −δ en DEV y CONFIRM.
- **S1 (anti-colapso, no veta)**: rango efectivo medio de ESTELA ≥ 0,8·M y ESTELA − ESTELA−SIGReg con IC95 inf > 0, en ambas fases.
- **S4 (sorpresa, no veta)**: AUROC(ESTELA) − AUROC(ÓRBITA) con IC95 inf ≥ −δ.
- **Controles (no vetan, atribución)**: ESTELA − {−SIGReg, REC, MLP, r0}, primaria k=4, con IC95. Lectura: si ESTELA no supera a r0, la ganancia no viene del residual; si no supera a MLP, no viene de K; si REC iguala o supera, no hace falta el objetivo latente.
- Exploratorios: k=2 y k=8; `*_base`; tasas de disparo/aceptación de E4.

**Éxito de ESTELA v0** ⇔ S3 aceptada ∧ S2 aceptada. `docs/estela/RESULTADOS.md` se publica en cualquier caso (negativos incluidos).

## 7. Decisiones de diseño y límites declarados

- Contexto de evaluación compartido por todos los brazos (reid de ÓRBITA para nuevas): la comparación aísla el predictor, no la re-identificación.
- El codificador objetivo es el online con stop-gradient (estilo LeJEPA, sin EMA); SIGReg es el único anti-colapso.
- z[0..2] = q por construcción (como ÓRBITA), por eso SIGReg actúa sobre z[2..8].
- Riesgos conocidos: r_φ puede quedarse ≈ 0; SIGReg con M=8 apenas probado; la energía puede perder contra la sorpresa de ÓRBITA (TFE); «sin tokens» ≠ comprensión de lenguaje (fuera de v0). Sin biblioteca/memoria (lección Cognitive Transformer).

## 8. Ejecución

Binario `orbita estela-run`; script `orbita/scripts/estela.sh` (setsid nohup, `nice 10`, JOBS=7, reanudable por `.done`). Salida `artifacts/estela/{smoke,dev,confirm}/` (`metrics_0x….json`, `params_<variante>_0x….json`). Orden: smoke (2 seeds DEV, H reducido a 4/familia, sólo tiempo; no analizado) → DEV → CONFIRM con el mismo binario y commit. Lanzamiento DEV:
`cd orbita && cargo build --release && cd .. && PHASE=dev JOBS=7 setsid nohup bash orbita/scripts/estela.sh > artifacts/estela/dev.nohup 2>&1 &`
