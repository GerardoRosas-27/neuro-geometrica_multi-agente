# PREREG — Familia METAFE (Arquitecturas Neuro-Geométricas Metacognitivas por Energía Libre)

Escrito y congelado en la rama `exp/metafe-free-energy` **antes** de ejecutar DEV (`0xA400–0xA40F`) y CONFIRM (`0xB400–0xB40F`). Sin re-ajuste entre DEV y CONFIRM. Cero consultas a TEST (`test_queries == 0`) y cero fuga de familias retenidas `{8, 10, 11}`.

## 1. Motivación y diagnóstico de las arquitecturas previas del repositorio
Tras auditar todas las arquitecturas creadas en `main` (ÓRBITA-v1, Campo v4, CONSTELACIÓN v2/v3/v4, ATLAS-FW `ALL`, Bio-Quantum v10 `BQ10`, Bio-Constelación, ÓRBITA-Transformer-FE, Yo 1er/2º orden y Soft-Bellman Simplicial v3), identificamos los cuatro cuellos de botella que explican el techo actual de `BQ10` (`new_med_log ≈ -1.51`, `new_acc16 ≈ 14.8%`):
1. **Desajuste dimensional y rigidez espectral en `BQ10`/`ATLAS-FW`:** 148 columnas (128 ocultas v4 + 17 biblioteca fija + 3) ajustadas sobre solo 24 filas (`k=4 × 6 pts`) mediante ridge Tikhonov con hiperparámetro fijo `da=10`, sin selección de evidencia bayesiana ni reversión epistémica fuera del soporte.
2. **Colapso variacional de TFE:** en `docs/lecciones/resultados_orbita_transformer_fe.md`, la Energía Libre con muestreo estocástico $\beta\cdot D_{\mathrm{KL}}$ colapsó la precisión (`h1 = 15%`) y no logró reducir la sorpresa de forma competitiva frente a la geometría exacta.
3. **Ausencia de inferencia sobre la geometría/simetría generativa:** las leyes físicas/geométricas de OOD abarcan variedades irreducibles entre sí (flujos afines $GL(2)\ltimes\mathbb{R}^2$, campos suaves cartesianos, acoplamientos simplécticos/hamiltonianos y variedades polares/equivariantes $SO(2)$).
4. **Falta de metacognición calibrada y autonomía de arranque en frío:** `v4`, `ATLAS-FW` y `BQ10` carecen de auto-evaluación de confianza calibrada ("Yo") y obtienen `0/16` en E33 sin depender de `ÓRBITA-v1` en `h=1`.

## 2. Diseño de la Familia METAFE (`orbita/src/constelacion4/metafe.rs`)
Arquitectura jerárquica de 3 niveles que **aprende por Energía Libre Variacional analítica exacta** ($\mathcal{F} = \text{Sorpresa} + D_{\mathrm{KL}}$) y posee **metacognición de 1er y 2º orden ("Yo-FE v2")**:

- **Nivel 0 — Banco Multi-Sustrato de 11 Hipótesis Generativas con VFE exacta:**
  - `LieAffine`: flujo afín exacto $GL(2)\ltimes\mathbb{R}^2$.
  - `FeArd`: Empirical Bayes / ARD variacional jerárquico por grado (reemplazo bayesiano de Tikhonov fijo de `BQ10`).
  - `GpCartM52` y `GpCartRbf`: Procesos Gaussianos semi-paramétricos cartesianos (media afín + kernel ARD aditivo/interacción Matérn-5/2 y RBF) con optimización analítica de Energía Libre perfilada $\mathcal{F}_m = \frac{N}{2}(\ln \hat{s}^2 + 1 + \ln 2\pi) + \frac{1}{2}\ln|\tilde{C}|$, varianza epistémica exacta $\sigma^2_{\mathrm{epi}}(q)$ y atajo afín exacto (residuo OLS $<10^{-8}$).
  - `GpCoMoving`: operator-splitting Lie-Trotter ($u = Aq+b$, $q' = u + g_{\mathrm{GP}}(u)$) con EM monótono de Energía Libre y selección automática de suavidad por VFE ($\nu\in\{5/2,\infty\}$).
  - `GpSympFwd`, `GpSympRev`, `GpCoSympFwd`, `GpCoSympRev`: acoplamientos simplécticos/hamiltonianos en espacio de Sobolev Matérn-5/2 (directos y co-móviles) para extrapolación autorregresiva estable.
  - `GpPolar`: variedad polar $SO(2)$-equivariante $(r,\theta)$ con kernel periódico exacto en $S^1$, selección automática de suavidad radial y corrección exacta de Jacobiano ($+2\sum\ln r'$) para comparabilidad en nats cartesianos.
  - `GpEquivCart`: kernel canónico producto de Schur $SO(2)$-equivariante $(q\cdot q')k_r(\|q\|,\|q'\|) + k_{\mathrm{cart}}$ (PSD garantizado, sin singularidad en $r=0$, selección automática de suavidad $\nu\in\{5/2,\infty\}$).
- **Nivel 1 — Inferencia Activa Multi-Paso y Sorpresa Precuencial:**
  - LOO-CV analítico exacto (Rasmussen & Williams Eq. 5.12 vía $[C^{-1}]_{ii}$).
  - Sorpresa precuencial fuera de tiempo en contexto (`OOT-1`: ajuste $0..K-1 \to K$; `OOT-2`: rollout cerrado $0..K-2 \to K-1,K$).
  - Refinamiento activo multi-paso del núcleo Lie-afín minimizando sorpresa de trayectoria cerrada.
- **Nivel 2 — Metacognición Entrenada de 2º Orden ("Yo-FE v2"):**
  - Entrenada estrictamente por semilla sobre episodios de familias conocidas `0..7` + familias sintéticas procedurales auditadas (`leaks()==false`, jamás `{8,10,11}`, `n_train=128`, `n_val=96`, 0 consultas TEST).
  - Meta-selección de pesos de Energía Libre `fe_w`, temperatura metacognitiva $\tau_{\mathrm{meta}}\in\{2,4,8,12\}$, ganancia epistémica $\gamma_{\mathrm{epi}}\in\{0,0.15,0.5\}$ e hiper-prior amortizado $p_\psi(M\mid c_e)$.
  - Auto-calibrador bietapa: Etapa A predice $\hat{e}_{\mathrm{meta}}\in[-7.5,2.3]$ sobre `cache_tr`; Etapa B calibra $\hat{p}_{\mathrm{ok}}=P(\text{err}_{h16}<0.05\mid \text{Yo-FE})$ mediante regresión logística Platt L2 (`lam=1.0`) **estrictamente fuera de muestra** en `cache_va` usando el margen $\ln(0.05)-\hat{e}_{\mathrm{meta}}$ y telemetría epistémica.
  - Prior autónomo de arranque en frío 1-paso (para E33 sin ÓRBITA): GP RBF por Energía Libre por familia $(p_0,p_1)\mapsto \mathrm{Affine2D}$ entrenado estrictamente sobre `ds.train` (0 consultas TEST).

## 3. Brazos congelados (7 brazos evaluados en DEV y CONFIRM)
- **Brazo primario:** `METAFE` (completo: 11 hipótesis + VFE + LOO + OOT + refinamiento multi-paso + BMA epistémico + Yo-FE v2).
- **Arquitecturas secundarias de la familia:** `FE-ARD`, `GP-Cart-FE`, `GP-Polar-FE`, `GP-Symp-FE`.
- **Ablaciones causales obligatorias:**
  - `METAFE-NoFE`: exactamente las mismas 11 hipótesis **sin** Energía Libre ni BMA/metacognición (solo replay MSE en contexto).
  - `METAFE-NoRoll`: Energía Libre VFE+LOO **sin** validación multi-paso OOT, sin refinamiento ni control epistémico.

## 4. Verificación Paso 0 y resumen de ajuste (`tune`, `0xC4B0–0xC4B3`)
- **Tests unitarios (`cargo test --release`):** 14/14 PASS (incluidos los 7 tests originales de `atlas`, `off_is_v4_bit_exact`, LOO analítico vs fuerza bruta $<10^{-9}$, monotonía ARD, PSD `GpEquivCart`, interpolación cold-start en `HOLE` con 0 consultas TEST).
- **Paso 0 (`metafe-paso0 --seed 0xC4B0`):** `paso0_ok = true` (0/8448 discrepancias bit a bit, 0 fuga `{8,10,11}`, 0 consultas TEST, error afín máx. `1.29e-12`, 100% reducción de Energía Libre).
- **Resumen `tune` (4 semillas `0xC4B0–0xC4B3`, pareado con `artifacts/bio_quantum_v10/bq_tune`):**
  - Baselines: `v4 = -0.8197 (5.73%)`, `CORE = -1.2077 (14.58%)`, `BQ10 = -1.4161 (14.58%)`.
  - `FE-ARD = -1.5193 (21.35%)`, `GP-Cart-FE = -2.8939 (48.96%)`, `GP-Symp-FE = -2.9067 (46.35%)`, `GP-Polar-FE = -3.2980 (54.17%)`.
  - Ablaciones: `METAFE-NoFE = -2.6976 (44.27%)`, `METAFE-NoRoll = -4.9494 (83.85%)`.
  - **Primario `METAFE`:** `new_med_log = -4.9922`, `new_acc16 = 85.94%`, `known_acc16 = 93.75%`, `syn_ood = -9.48 (65.1%)`, `e33 = 4/4`, `e33_autonomous = 4/4 (h1 = 100%)`, `AUROC = 0.9748`, `Brier = 0.0449`, `ECE = 0.0566`, $\Delta\mathcal{F} = +31.45$ nats (`100%`), `0 NaN`, `0` consultas TEST, `test_sha256` idéntico 4/4.

## 5. Hipótesis y protocolo estadístico congelado (DEV & CONFIRM)
- **Fases:** DEV (`0xA400–0xA40F`, 16 semillas) y CONFIRM (`0xB400–0xB40F`, 16 semillas), pareadas semilla a semilla con los artefactos congelados de `artifacts/bio_quantum_v10/{bq_dev,bq_confirm}/metrics_<S>.json` (verificando `test_sha256` idéntico 16/16 en ambas fases).
- **Bootstrap:** pareado por semilla, 10 000 réplicas, semilla `12345`, IC95 percentil, umbral de relevancia $\delta = 0.05$.
- **Hipótesis primaria (H1):** `METAFE − BQ10` en `new_med_log` (h16 familias nuevas `k=4`): éxito si IC95 sup $< 0$ y $|\Delta| \ge 0.05$ en DEV **y** CONFIRM.
- **Co-primarias declaradas (H2–H4):**
  - H2: `METAFE − ALL (ATLAS-FW)` y `METAFE − v4` en `new_med_log` (misma regla).
  - H3 (Causalidad Energía Libre): `METAFE − METAFE-NoFE` en `new_med_log` con IC95 sup $< 0$ y $|\Delta| \ge 0.05$ en DEV y CONFIRM.
  - H4 (Causalidad Inferencia Activa Multi-Paso): `METAFE − METAFE-NoRoll` en `new_med_log`, `new_acc16` y `syn_ood_med_log`.
- **No-daño y puertas de seguridad:**
  - `known_acc16` de `METAFE − BQ10` y `METAFE − v4` con IC95 inf $> -0.05$.
  - `E33` canónico `16/16` y `E33` autónomo (sin ÓRBITA) `16/16`.
  - `0 NaN`, `test_queries == 0` en las 32 semillas.
- **Criterios de metacognición y Energía Libre:**
  - Fracción de episodios con reducción positiva de Energía Libre ($\Delta\mathcal{F} > 0$) $\ge 99\%$.
  - Discriminación metacognitiva global `AUROC ≥ 0.90`, calibración `Brier ≤ 0.08`, y en familias nuevas OOD `AUROC ≥ 0.75` y correlación de auto-predicción de error `r ≥ 0.70`.
- **Veredicto final:** se decide exclusivamente en **CONFIRM**; cualquier resultado negativo o nulo se reportará íntegramente.
