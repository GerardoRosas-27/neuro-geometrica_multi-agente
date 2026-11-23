# PREREG — METAFE-3S (Control Metacognitivo de Tres Escalas sobre METAFE v2)

Escrito y congelado en la rama `exp/metafe-3s-neuromod` **antes** de ejecutar DEV (`0xA400–0xA40F`) y CONFIRM (`0xB400–0xB40F`). Prohibido modificar `orbita/src/constelacion4/metafe3s.rs`, `metafe.rs` o cualquier hiperparámetro tras este commit de congelación. Cero consultas a TEST (`test_queries == 0`), cero fuga de familias retenidas `{8, 10, 11}`, y paridad bit a bit con `METAFE v2` (`3871fae`).

---

## 1. Objetivo Científico y Arquitectura Congelada

Aplicar los **5 módulos matemáticos del Control Metacognitivo de Tres Escalas** sobre la arquitectura puntera previa del repositorio (**METAFE v2**, `new_med_log = -5.3338`, `88.67%` en `CONFIRM`), sin alterar el comportamiento de `METAFE v2` (véase [`docs/metafe3s/ARQUITECTURA.md`](ARQUITECTURA.md) y [`orbita/src/constelacion4/metafe3s.rs`](../../orbita/src/constelacion4/metafe3s.rs)):

1. **Módulo 1 — Núcleo Dual ($W_0$ congelado + $F_t \in \mathbb{R}^{2\times 6}$ acotado):** los 11 sustratos geométricos VFE de `METAFE v2` permanecen congelados; la adaptación efímera opera en la bola Frobenius $\Pi_{\|\cdot\|_F \le \rho}$ y se descarta al finalizar cada episodio (`fw.reset()`).
2. **Módulo 2 — Monitor de Conflicto ACC ($m_t \in \mathbb{R}^4$) + Refutador Cruzado Inverso ($\Phi_{\text{rev}}$):** telemetría interna de sorpresa/refutación, desacuerdo epistémico, estancamiento/bucles y saturación plástica; segunda rama $\Phi_{\text{rev}}$ ajustada exclusivamente en el contexto invertido $[x_K \dots x_0]$ (9 sustratos, fiabilidad $\pi_{\text{rev}}$ validada por ciclo + `rev_oot1`) para puntuación de trayectoria, corrección de variedad de ciclo (`cycle_manifold_correct`) y exceso de refutación ponderado en soporte epistémico.
3. **Módulo 3 — Controlador Neuromodulador Recurrente (`MetaNet` GRU $H=4$):** entrenado outer-loop por BPTT exacto ponderado (90 épocas, peso $6\times$ en choques, 0 fuga `{8, 10, 11}`), emitiendo $[\alpha_t, \lambda_t, \beta_t, g_t \in \Delta^3]$ en tiempo real.
4. **Módulo 4 — Plasticidad de Tres Factores con Trazas de Elegibilidad $E_t$ y Acotación Espectral $\Pi_\rho$:** $E_t = \gamma E_{t-1} + \phi(e_t)\otimes \psi_{\text{norm}}$, $F_{t+1} = \Pi_\rho(\lambda_t F_t + \alpha_t \delta_t E_t)$, con verificación precuencial OOT en régimen estacionario (cero daño en afines exactas) y adaptación online en régimen no estacionario con $W_0$ congelado (**RS-B**).
5. **Módulo 5 — Buffer Episódico + Consolidación Offline EWC ($\Omega_i$ diagonal de Fisher) + "Yo-FE 3S":** destilación offline regularizada hacia el hiper-prior lento $W_0^{\text{meta}}$ minimizando $\mathcal{L}_{\text{CE}} + \frac{\mu}{2}\sum_i \Omega_i(W_{0,i}-W_{0,i}^{\text{prev}})^2$, más metacognición desacoplada Etapa A-core (10 rasgos núcleo forward, ridge $\lambda=0.05$) + Etapa A-ref ($w_{\text{ref}}^{\text{meta}} \ge 0$ fuera de muestra sobre exceso de refutación en soporte) + Etapa B (Platt L2 $\lambda=1.5$ balanceado).

---

## 2. Brazos Congelados y Suites de Evaluación

Se evalúan **5 brazos pareados semilla a semilla** + **1 experimento dedicado de consolidación continua**:

| Brazo | $W_0$ VFE | $F_t + E_t$ | $\Phi_{\text{rev}}$ | `MetaNet` GRU | Propósito |
|---|---|---|---|---|---|
| `METAFE-v2` | Sí (intacto) | No | No | No | Puntera previa exacta (paridad bit a bit) |
| **`METAFE-3S`** (primario) | **Sí** | **Sí** | **Sí** | **Sí** | **Arquitectura completa (5 módulos)** |
| `METAFE-3S-NoFast` | Sí | **No** | Sí | Sí | Ablación causal Módulos 1+4 (pesos rápidos $F_t$) |
| `METAFE-3S-NoRefute` | Sí | Sí | **No** | Sí | Ablación causal Módulo 2 (refutador cruzado $\Phi_{\text{rev}}$) |
| `METAFE-3S-NoNM` | Sí | Sí | Sí | **No** | Ablación causal Módulo 3 (controlador neuromodulador) |

**Suites evaluadas en cada semilla (32 semillas totales: 16 DEV + 16 CONFIRM):**
1. **Canónico held-out (`gen_heldout`):** familias conocidas `0..7` y retenidas `{8, 10, 11}`, horizonte `h=16`, `k_obs=4` (pareado por `test_sha256` contra `artifacts/metafe/` y `artifacts/bio_quantum_v10/`).
2. **OOD Sintético procedural (`gen_ood_synthetic_eps`, 48 eps/semilla):** familias sintéticas no lineales auditadas fuera de `{8, 10, 11}`.
3. **Arranque en frío E33 (canónico con ÓRBITA y autónomo sin ÓRBITA):** 0 consultas TEST.
4. **Cambio de Régimen Online Abrupto (`RS-A`, 48 eps/semilla = 24 conocidas + 24 OOD):** ley física $A \to B$ conmuta a mitad de episodio observando solo los cuadros en flujo (`eval_regime_shift_suite`).
5. **Perturbación Online con $W_0$ Estrictamente Congelado (`RS-B`, 24 eps/semilla):** $W_0$ ajustado sobre el régimen inicial permanece congelado; la dinámica sufre una perturbación en caliente y solo la plasticidad rápida acotada $F_t$ gobernada por `MetaNet` puede adaptarse.
6. **Consolidación Continua Offline (`eval_continual_ewc`, 3 tareas secuenciales `T1 → T2 → T3`):** contrasta `EWC` ($\mu=25$) frente a `NoEWC` ($\mu=0$) en olvido de T1 (`t1_forgetting_ce`) y exactitud media T1–T3.

---

## 3. Verificación Paso 0 y Resumen Congelado en `tune` (`0xC4B0–0xC4B3`)

- **Tests unitarios (`cargo test --release`):** `19/19 PASS` (7 `atlas` + 7 `metafe` + 5 `metafe3s`: bola Frobenius y trazas de elegibilidad, refutador inverso en dinámica suave, respuesta BPTT de reset del ACC/`MetaNet`, cero fuga y cero daño en afines exactas, reducción de olvido EWC vs NoEWC).
- **Paso 0 (`metafe3s-paso0 --seed 0xC4B0`):** `paso0_ok = true`, `deterministic_mismatch = 0/8448`, `syn_rejected = 83` (0 fuga), `test_queries = 0`, `max_rel_err = 2.11e-12`, `max_fast_norm = 0.0`, `fe_reduced_all = true`, `ewc_delta_forgetting = -0.0533`.
- **Resumen `tune` (`0xC4B0–0xC4B3`, `test_sha256` 4/4 idéntico a BQ10 y METAFE v2, paridad `METAFE-v2` = `0.0`):**
  - Canónico `new_med_log`: `BQ10 = -1.4161 (14.58%)`, `METAFE-v2 = -4.9922 (85.94%)`, **`METAFE-3S = -5.2078 (85.94%)`** (`Δ vs v2 = -0.2156`, gana 3/4; `NoRefute = -4.9875`, causalidad refutador `Δ = -0.2203`, gana 3/4), `known_acc16 = 0.9375` (cero daño).
  - Metacognición `METAFE-3S`: `ov_AUROC = 0.9728`, `ov_Brier = 0.0456`, `new_AUROC = 0.8329` (supera `NoRefute 0.8234`), `new_r = 0.8172`, `new_ρ = 0.8379`, `syn_AUROC = 0.9801`, `syn_Brier = 0.0511`, $\Delta\mathcal{F} = +31.47$ nats (`100%`).
  - No estacionario `RS-A`: `METAFE-v2 = -0.2239` (`reset 0%`) vs **`METAFE-3S = -5.5108`** (`reset 87.5%`, `Δ = -5.2870`, gana 4/4); ablación `METAFE-3S-NoNM` colapsa a `-0.2397` (`Δ = -5.2711`, gana 4/4).
  - Perturbación $W_0$ congelado `RS-B`: `METAFE-v2 = +0.4447 (0%)` vs **`METAFE-3S = -2.0963 (18.8%)`** (`Δ = -2.5410`, gana 4/4); ablaciones `NoFast (+0.4921, 0%)` y `NoNM (+0.2365, 0%)` colapsan (4/4 cada una).
  - Consolidación continua EWC vs NoEWC: `Δ forgetting = -0.0400` (gana 4/4), `Δ avg_acc = +0.0590`.
  - Autonomía y tiempo: `E33 = 4/4`, `E33 Autónomo = 4/4 (h1 = 100%)`, `METAFE-3S = 19.3s/semilla` vs `METAFE-v2 = 24.6s/semilla`.

---

## 4. Hipótesis y Protocolo Estadístico Congelado (DEV & CONFIRM)

- **Fases:** DEV (`0xA400–0xA40F`, 16 semillas) y CONFIRM (`0xB400–0xB40F`, 16 semillas).
- **Bootstrap:** pareado por semilla, 10 000 réplicas, semilla `12345`, IC95 percentil, umbral de relevancia $\delta = 0.05$ (`scripts/analyze_metafe3s.py`).
- **H1 (Superación de baselines previos del repo):** `METAFE-3S` supera a `BQ10`, `ALL`, `CORE`, `v4` y `ORBITA` en `new_med_log` con IC95 sup $< 0$ y $|\Delta| \ge 0.05$ en DEV y CONFIRM, y `known_acc16` con IC95 inf $> -0.05$.
- **H2 (Cabeza a cabeza `METAFE-3S − METAFE-v2`):**
  - `H2a`: `new_med_log` en canónico (éxito estricto si IC95 sup $< 0$ y $|\Delta| \ge 0.05$ en DEV y CONFIRM; se reportará íntegramente si la ganancia es menor o nula).
  - `H2b`: no-daño en conocidas (`known_acc16` IC95 inf $> -0.05$ en DEV y CONFIRM).
  - `H2c / H2d`: calibración (`new_Brier`) y discriminación (`new_AUROC`) en familias nuevas OOD (reporte honesto de IC95 completo).
- **H3 (Cambio de Régimen Abrupto Online `RS-A` — Módulos 2+3):** `METAFE-3S − METAFE-v2` y ablación causal `METAFE-3S − METAFE-3S-NoNM` en `rs_a_all_med_log` con IC95 sup $< 0$ y $|\Delta| \ge 0.05$ en DEV y CONFIRM.
- **H4 (Perturbación Online con $W_0$ Congelado `RS-B` — Módulos 1+3+4):** `METAFE-3S − METAFE-v2`, `METAFE-3S − METAFE-3S-NoFast` y `METAFE-3S − METAFE-3S-NoNM` en `rs_b_frozen_w0_med_log` con IC95 sup $< 0$ y $|\Delta| \ge 0.05$ en DEV y CONFIRM.
- **H5 (Causalidad del Refutador Cruzado Inverso $\Phi_{\text{rev}}$):** `METAFE-3S − METAFE-3S-NoRefute` en `new_med_log` y `new_Brier` (reporte íntegro de IC95 en DEV y CONFIRM).
- **H6 (Consolidación Continua Offline EWC — Módulo 5):** `delta_forgetting_ewc_minus_noewc` con IC95 sup $< 0$ en DEV y CONFIRM.
- **Puertas de seguridad, autonomía y metacognición:**
  - `E33` canónico `16/16` y `E33` autónomo `16/16` en DEV y CONFIRM.
  - `0 NaN`, `test_queries == 0`, `test_sha256` idéntico `32/32` vs `BQ10` y `METAFE v2`, paridad `METAFE-v2` `max_diff == 0.0`.
  - Reducción de Energía Libre $\Delta\mathcal{F} > 0$ en $\ge 99\%$, `ov_AUROC ≥ 0.90`, `ov_Brier ≤ 0.08`, `new_AUROC ≥ 0.78`, `new_r ≥ 0.75`.
- **Regla de veredicto:** el veredicto final se determina exclusivamente en **CONFIRM** sin ajustes post-hoc; todo contraste nulo o negativo se documentará de forma explícita.
