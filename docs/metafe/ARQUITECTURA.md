# Familia METAFE — Auditoría de Arquitecturas del Repositorio y Diseño Matemático

Rama: `exp/metafe-free-energy` · Implementación: [`orbita/src/constelacion4/metafe.rs`](../../orbita/src/constelacion4/metafe.rs) · Preregistro: [`docs/metafe/PREREG.md`](PREREG.md).

---

## 1. Auditoría completa de las arquitecturas del repositorio

Antes de diseñar **METAFE**, se auditaron todas las arquitecturas implementadas y evaluadas en `main` y en el historial del repositorio (benchmark canónico ÓRBITA/CONSTELACIÓN sobre trayectorias 2D de 6 cuerpos a horizonte $h=16$ con $k=4$ cuadros de contexto, más líneas auxiliares):

| Familia / Arquitectura | Mecanismo central | Resultado CONFIRM (o primario propio) | Limitación / cuello de botella identificado |
|---|---|---|---|
| **ÓRBITA-v1** | Dinámica neuronal condicionada por contexto $c$ con pesos congelados (`artifacts/orbita/`). | `new_med_log = -0.7928`, `new_acc16 = 2.47%`, `known_acc16 = 33.84%`, E33 `16/16`. | Incapaz de adaptar dinámicas OOD sin entrenamiento; degradación fuerte a horizonte $h=16$ (`32.3%`). |
| **CONSTELACIÓN v2 / v3 / v4** (`orbita/src/constelacion4/mod.rs`) | Red de codificación predictiva + campo rápido en contexto $W^F$ (NLMS) + delegación a ÓRBITA en `h=1` para E33. | v4: `new_med_log = -0.8997`, `new_acc16 = 6.64%`, `known_acc16 = 87.70%`, E33 `16/16` (`0/16` sin ÓRBITA). | Toda la ganancia proviene de $W^F$ (`no_fast` pierde ~3.2 nats); `new_acc16` muy baja (6.6%); escalar capacidad hasta 62k params (`v4-ESCALADA`) no mejora nada. |
| **ATLAS-FW (`ALL`)** (`orbita/src/constelacion4/atlas.rs`) | Sobre v4 entrenada (~380–449 s): M1 (biblioteca de 17 funciones fija) + M2 (ridge + STLSQ en 148 cols × 24 filas) + M6 (compuerta contra v4). | `new_med_log = -1.3518`, `new_acc16 = 15.62%`, `known_acc16 = 90.28%`, E33 `16/16` (`0/16` sin ÓRBITA). | M3–M5 quedan inactivos (`α=0`, `ALL ≡ CORE` en CONFIRM); el ajuste 148×24 sobreajusta términos cúbicos que divergen al iterar 12 pasos. |
| **Bio-Quantum v10 (`BQ10`)** (`atlas.rs`, ganadora actual en `main`) | `CORE` (M1+M2+M6) + **M7 DA-STR** (Tikhonov ponderado por grado: lineal $\lambda$, cuadrático $10\lambda$, cúbico $100\lambda$; `da=10`). M8 inerte (`BQR10 ≡ BQ10`). | `new_med_log = -1.5081` ($\Delta$ vs v4 `-0.6083`, vs ALL `-0.1563`), `new_acc16 = 14.84%`, `known_acc16 = 88.72%`, E33 `16/16` (`0/16` sin ÓRBITA). | Compromiso sesgo-varianza rígido (`da=10` fijo): baja `known_acc16` (`-1.56 pp` vs ALL) y `new_acc16` (`15.6→14.8%`); la mejora 1-paso (`-0.75 nats`) se diluye a `-0.15 nats` en $h=16$; sin metacognición ni incertidumbre epistémica. |
| **ÓRBITA-Transformer-FE (`TFE`)** (`docs/lecciones/resultados_orbita_transformer_fe.md`) | Transformer de estados + latente estocástico $\mu\in\mathbb{R}^8$ entrenado con $\mathcal{L}_{\text{ÓRBITA}} + \beta\cdot D_{\mathrm{KL}}$ y 100 pasos Adam sobre $\mu$ en inferencia. | **Refutada** en H1/H2/H3b/H4a/H4d: `known_acc16 = 4.24%` (vs `21.96%` TF sin FE y `35.62%` ÓRBITA); AUROC FE `0.6616`. | El muestreo estocástico $\beta\cdot D_{\mathrm{KL}}$ destruye la conservación geométrica fina a $h=16$ y penaliza al modelo con un prior isotrópico que combate la dinámica real. |
| **Yo 1er/2º orden** (`docs/plan_yo_primer_segundo_orden.md`, `preregistro_yo*.md`) | Auto-modelo calibrado ("Yo") que sintetiza sorpresa, estabilidad y confianza $\hat{p}_{\text{ok}}$ con evaluación de AUROC, ECE y abstención selectiva. | Propuesto sobre pila LLM+ÓRBITA; sin arquitectura nativa acoplada a inferencia dinámica OOD. | La calibración metacognitiva no retroalimentaba la inferencia de hipótesis generativas ni la regularización por incertidumbre epistémica. |
| **Extensiones en otros arneses** (`gemma_campo`, `campo_estado_turno`, `soft_bellman_simplicial_v3_gpu`, `bq10-gpu-amd`) | Memoria de campo en LLMs (capas Gemma), geodésicas simpliciales 3D (Snell + Hodge-Fourier) y motor GPU x50. | Éxitos en sus respectivos arneses propios (ver [`docs/RESULTADOS_CONSOLIDADOS.md`](../RESULTADOS_CONSOLIDADOS.md)). | No reemplazan al benchmark canónico ÓRBITA/CONSTELACIÓN ni resuelven la generalización OOD a $h=16$ con $k=4$. |
| **Negativos / Inconclusos** (`MAREA`, `ECO`, `ESTELA`, `Nova-2`, `BIO-CONSTELACIÓN`, `v4 M1–M3`) | Sueños offline, BOCD, base compartida, neocórtex acoplado. | Negativos o causan daño severo en familias conocidas / E33. | Confirman que añadir complejidad heurística sin inferencia exacta de la geometría subyacente no rompe el techo de `-1.51 nats`. |

### Cuatro principios de diseño deducidos de la auditoría

1. **Energía Libre Variacional Analítica Exacta (sin colapso estocástico):** A diferencia de `TFE` (donde el muestreo de $\mu$ con $\beta\cdot D_{\mathrm{KL}}$ destruyó la precisión), la Energía Libre en espacios funcionales gaussianos y bayesianos jerárquicos posee **forma cerrada exacta** ($\mathcal{F} = -\ln p(Y\mid X, \theta)$), combinando de forma automática el ajuste a datos (sorpresa) y el determinante de complejidad de Occam ($\frac{1}{2}\ln|\tilde{C}|$) sin ruido estocástico.
2. **Inferencia de Simetría sobre Múltiples Sustratos Geométricos:** Las leyes dinámicas en $\mathbb{R}^2$ (y OOD) no viven todas en la misma base polinómica cartesiana: incluyen transformaciones afines $GL(2)\ltimes\mathbb{R}^2$, campos suaves locales, flujos co-móviles, acoplamientos hamiltonianos/simplécticos (que preservan volumen en espacio de fases) y dinámicas radiales/rotacionales $SO(2)$-equivariantes. Una arquitectura debe mantener un **banco multi-sustrato** y seleccionar/combinar las geometrías por evidencia bayesiana (Energía Libre).
3. **Inferencia Activa Multi-Paso y Control Epistémico:** Un modelo que minimiza error a 1 paso puede explotar al iterarse 12 pasos en lazo cerrado (como se observó en `ATLAS-FW` y al elegir RBF sobre acoplamientos simplécticos). La inferencia debe incorporar validación precuencial fuera de tiempo (`OOT-1`, `OOT-2`), refinamiento de trayectoria cerrada y regularización epistémica.
4. **Metacognición Entrenada de 1er y 2º Orden ("Yo-FE v2") + Autonomía en Frío:** El agente debe estimar tanto la magnitud esperada de su propio error como la probabilidad calibrada de acierto $\hat{p}_{\mathrm{ok}} = P(\text{err}_{h16} < 0.05)$ —calibrada estrictamente fuera de muestra— y resolver el arranque en frío (`E33` en $h=1$) de manera autónoma sin delegar a `ÓRBITA-v1`.

---

## 2. Arquitectura Jerárquica de 3 Niveles (`METAFE`)

Toda la familia METAFE está implementada en Rust puro (`f64`, 0 dependencias externas adicionales, determinista bit a bit) en [`orbita/src/constelacion4/metafe.rs`](../../orbita/src/constelacion4/metafe.rs) sin modificar ningún archivo de `v4`, `ATLAS-FW` ni `BQ10`.

```
Observación en contexto (frames 0..K, K=4, 24 transiciones q_t → q_{t+1})
│
├─► NIVEL 0: Banco Multi-Sustrato de 11 Hipótesis Generativas (Energía Libre Exacta)
│   ├─ LieAffine          : Flujo afín exacto GL(2) ⋉ R²
│   ├─ FeArd              : Empirical Bayes / ARD variacional jerárquico por grado
│   ├─ GpCartM52 / Rbf    : GP semi-paramétrico cartesiano (media afín + Matérn-5/2 / RBF)
│   ├─ GpCoMoving         : Operator-splitting Lie-Trotter + EM de VFE + auto-suavidad
│   ├─ GpSymp{Fwd,Rev}    : Acoplamiento simpléctico en espacio de Sobolev Matérn-5/2
│   ├─ GpCoSymp{Fwd,Rev}  : Operator-splitting simpléctico co-móvil (Matérn-5/2)
│   ├─ GpPolar            : Variedad polar SO(2) + kernel periódico S¹ + Jacobiano exacto
│   └─ GpEquivCart        : Kernel canónico producto de Schur SO(2)-equivariante (sin r=0)
│
├─► NIVEL 1: Inferencia Activa Multi-Paso y Sorpresa Precuencial
│   ├─ LOO-CV analítico exacto (Rasmussen & Williams Eq. 5.12 vía diag(C⁻¹))
│   ├─ Sorpresa precuencial fuera de tiempo: OOT-1 (0..K-1 → K) y OOT-2 (rollout 0..K-2 → K-1,K)
│   ├─ Refinamiento activo multi-paso del núcleo Lie-afín en trayectoria cerrada
│   └─ Diagnóstico de deriva radial y varianza epistémica σ²_epi(q) en rollout cerrado
│
└─► NIVEL 2: Metacognición "Yo-FE v2" y Prior Autónomo de Arranque en Frío
    ├─ Posterior BMA por Energía Libre Libre de Fuga: q(M | datos) ∝ p_ψ(M|c_e) exp(-τ_meta · E_M)
    ├─ Meta-entrenamiento por semilla (128 train + 96 val: conocidas 0..7 + sintéticas auditadas)
    ├─ Calibrador bietapa fuera de muestra:
    │   ├─ Etapa A (cache_tr): predicción de log-error esperado ê_meta ∈ [-7.5, 2.3]
    │   └─ Etapa B (cache_va): Platt L2 (λ=1.0) sobre margen ln(0.05) - ê_meta + telemetría epistémica
    └─ Prior Autónomo 1-paso (E33 sin ÓRBITA): GP RBF por VFE sobre ds.train (p0, p1) ↦ Affine2D
```

### 2.1 Nivel 0 — Formulación matemática de la Energía Libre Variacional

Dado un conjunto de $N$ pares de transición $(u_i, y_i)$ extraídos del contexto ($N = 24$ para $K=4$ y 6 cuerpos), cada canal GP semi-paramétrico descompone la salida como una media paramétrica afín $m(u) = H(u)\beta$ más un proceso gaussiano residual $f \sim \mathcal{GP}(0, s^2 \tilde{k}_\theta(u, u'))$ con ruido observacional $s^2 \sigma_n^2 I$.

Perfilando analíticamente la escala global de varianza $\hat{s}^2 = \frac{1}{N} r^\top \tilde{C}^{-1} r$ (donde $r = y - H\hat{\beta}_{\text{GLS}}$ y $\tilde{C} = \tilde{K}_\theta + \sigma_n^2 I$), la **Energía Libre Variacional negativa (evidencia log-marginal perfilada)** resulta exacta y cerrada ([`vfe_from_gram`](../../orbita/src/constelacion4/metafe.rs#L392), [`fit_gp_channel_fixed`](../../orbita/src/constelacion4/metafe.rs#L305)):

$$\mathcal{F}(\theta) = \underbrace{\frac{N}{2}\bigl(\ln \hat{s}^2 + 1 + \ln 2\pi\bigr)}_{\text{Sorpresa / Inexactitud}} + \underbrace{\frac{1}{2}\ln|\tilde{C}_\theta|}_{\text{Complejidad de Occam } (D_{\mathrm{KL}})}$$

Esta formulación resuelve el problema central de `TFE` y `BQ10`:
- Minimizar $\mathcal{F}(\theta)$ mediante búsqueda multi-escala en hiperparámetros (con bases de Gram precomputadas y atajo afín exacto [`exact_linear_rel_res`](../../orbita/src/constelacion4/metafe.rs#L427) cuando el residuo OLS relativo es $<10^{-8}$) **reduce la sorpresa y poda complejidad de forma determinista y monótona**, garantizando $\Delta\mathcal{F} = \mathcal{F}_{\text{null}} - \mathcal{F}_{\text{post}} > 0$.
- A diferencia de la regularización rígida `da=10` de `BQ10`, el brazo **`FE-ARD`** ([`fit_ard_channel`](../../orbita/src/constelacion4/metafe.rs#L854)) aplica Empirical Bayes / ARD variacional jerárquico por grado polinómico/trigonométrico, adaptando las precisiones $\alpha_g$ automáticamente a cada episodio.

#### Sustratos geométricos clave del banco de 11 hipótesis ([`HypKind`](../../orbita/src/constelacion4/metafe.rs#L989))

1. **Flujo afín y ARD (`LieAffine`, `FeArd`):** Capturan transformaciones exactas en $GL(2)\ltimes\mathbb{R}^2$ y campos polinómicos/espectrales suaves con selección automática de relevancia por grado.
2. **GP Cartesianos (`GpCartM52`, `GpCartRbf`) y Operator-Splitting Co-móvil (`GpCoMoving`):** Descomponen la transición en un paso afín $u = Aq+b$ seguido de una corrección no lineal $q' = u + g_{\mathrm{GP}}(u)$ optimizada por EM monótono de Energía Libre con selección automática de suavidad por VFE ($\nu \in \{5/2, \infty\}$, [`optimize_gp_channel_vfe_auto`](../../orbita/src/constelacion4/metafe.rs#L795)).
3. **Acoplamientos Simplécticos en Espacio de Sobolev (`GpSympFwd`, `GpSympRev`, `GpCoSympFwd`, `GpCoSympRev`):** Modelan dinámicas Hamiltonianas / preservadoras de área mediante pasos escalonados $(x' = x + f(x, y),\; y' = y + g(x', y))$ (y su simétrico inverso). Un hallazgo empírico clave en `tune` (`0xC4B0`) mostró que mantener el núcleo en el espacio de Sobolev **Matérn-5/2** (en lugar de RBF) evita el decaimiento super-exponencial fuera de la nube inicial durante el rollout autorregresivo de 12 pasos.
4. **Variedad Polar $SO(2)$ con Jacobiano Exacto (`GpPolar`):** Modela $(\Delta r, \Delta\theta)$ en coordenadas polares $(r,\theta)$ combinando un kernel radial con un kernel periódico exacto en $S^1$, $k_{S^1}(\theta, \theta') = \exp(-2\sin^2((\theta-\theta')/2)/\ell_\theta^2)$ ([`periodic_s1`](../../orbita/src/constelacion4/metafe.rs#L209)). Para que la Energía Libre polar sea **estrictamente conmensurable en nats cartesianos** con las hipótesis cartesianas, se suma la corrección exacta del Jacobiano del cambio de variables:
   $$\mathcal{F}_{\text{cart}}(\text{GpPolar}) = \mathcal{F}_r + \mathcal{F}_\theta + 2\sum_{i=1}^{N}\ln r'_i$$
5. **Kernel Canónico Producto de Schur $SO(2)$-Equivariante (`GpEquivCart`):** Para dinámicas rotacionalmente equivariantes cercanas al origen $r \to 0$ (donde las coordenadas polares presentan singularidad angular), define un núcleo definido positivo garantizado por el teorema del producto de Schur ([`kernel_eval`](../../orbita/src/constelacion4/metafe.rs#L215)):
   $$k_{\text{equiv}}(q, q') = (q \cdot q')\, k_r(\|q\|, \|q'\|) + k_{\text{cart}}(q, q')$$
   con selección automática de suavidad radial por VFE.

### 2.2 Nivel 1 — Inferencia Activa Multi-Paso y Sorpresa Precuencial

Para cerrar la brecha entre error a 1 paso y estabilidad a horizonte $h=16$ ([`evaluate_hypothesis`](../../orbita/src/constelacion4/metafe.rs#L1768)):
- **LOO-CV analítico exacto (Rasmussen & Williams, Eq. 5.12):** Usando $[C^{-1}]_{ii}$ obtenido en $O(N^2)$ desde la factorización de Cholesky ([`chol_inv_diag`](../../orbita/src/constelacion4/metafe.rs#L116)), el error leave-one-out y su log-verosimilitud predictiva se calculan sin re-ajustar el modelo:
  $$y_i - \hat{\mu}_{-i} = \frac{\alpha_i}{[C^{-1}]_{ii}}, \qquad \sigma^2_{-i} = \frac{1}{[C^{-1}]_{ii}}$$
- **Sorpresa Precuencial Fuera de Tiempo (`OOT-1` y `OOT-2`):** Re-ajusta la hipótesis en los primeros $K-1$ cuadros y mide la sorpresa real sobre el cuadro $K$ (`OOT-1`), y la re-ajusta en $0..K-2$ ejecutando un **rollout cerrado de 2 pasos** hacia $K-1, K$ (`OOT-2`). Esto detecta en tiempo de inferencia hipótesis que interpolan 1 paso pero divergen al iterarse.
- **Refinamiento Activo Multi-Paso (`refine_affine_multistep`):** Optimiza los parámetros del flujo Lie-afín minimizando directamente la sorpresa de trayectoria cerrada sobre el contexto ([`refine_affine_multistep`](../../orbita/src/constelacion4/metafe.rs#L1601)).

### 2.3 Nivel 2 — Metacognición "Yo-FE v2" y Prior Autónomo de Arranque en Frío

El controlador metacognitivo ([`MetaController`](../../orbita/src/constelacion4/metafe.rs#L1922), [`train_metafe_controller`](../../orbita/src/constelacion4/metafe.rs#L2515)) se entrena **independientemente por semilla** usando estrictamente 128 episodios de entrenamiento y 96 de validación (50% familias conocidas `0..7` + 50% familias sintéticas procedurales auditadas que garantizan 0 fuga hacia `{8, 10, 11}` y 0 consultas a TEST):

1. **Combinación Bayesiana por Energía Libre (BMA epistémico):** Para cada hipótesis $m$, forma la energía libre efectiva $E_m = w^\top [\mathcal{F}_m/N,\; \text{LOO}_m,\; \text{OOT1}_m,\; \text{OOT2}_m,\; \text{pen}_{\text{rad},m},\; \text{epi}_m]$ y calcula el posterior variacional amortizado:
   $$q(M=m \mid D) \propto p_\psi(M=m \mid c_e)\, \exp\bigl(-\tau_{\text{meta}}\, E_m\bigr)$$
   donde $c_e$ son invariantes geométricos del episodio ([`episode_invariants`](../../orbita/src/constelacion4/metafe.rs#L1880)). Durante el rollout cerrado hasta $h=16$, cada hipótesis aporta su predicción ponderada por $q(M)$ y atenuada epistémicamente según su varianza predictiva $\sigma^2_{\text{epi}}(q)$ ($\gamma_{\text{epi}}$).
2. **Auto-Calibrador Metacognitivo Bietapa (estrictamente fuera de muestra):**
   - **Etapa A (`cache_tr`):** Ridge L2 ($\lambda=0.05$) predice el log-error esperado a $h=16$, $\hat{e}_{\text{meta}} \in [-7.5, 2.3]$, a partir de la energía libre posterior, LOO, OOT-1/2, entropía posterior $H[q(M)]$, desacuerdo del ensamble y varianza epistémica media.
   - **Etapa B (`cache_va`, estrictamente fuera de muestra):** Regresión logística Platt L2 ($\lambda=1.0$) sobre 7 características metacognitivas lideradas por el margen $\ln(0.05) - \hat{e}_{\text{meta}}$, produciendo la confianza calibrada $\hat{p}_{\text{ok}} = P(\text{err}_{h16} < 0.05 \mid \text{Yo-FE})$.
3. **Prior Autónomo de Arranque en Frío (`E33` sin ÓRBITA):** Entrena por familia un GP RBF de Energía Libre $(p_0, p_1) \mapsto \mathrm{Affine2D}$ exclusivamente sobre `ds.train` (24 episodios/familia, 0 consultas TEST; [`train_cold_start_prior`](../../orbita/src/constelacion4/metafe.rs#L2445)). Esto permite predecir el primer paso $h=1$ interpolando suavemente a través del hueco `HOLE = (0.10, 0.45)` de `E33` sin depender de `ÓRBITA-v1`.

---

## 3. Familia de 7 Brazos Congelados

Para aislar causalmente cada componente en DEV y CONFIRM, [`metafe_arms`](../../orbita/src/constelacion4/metafe.rs#L1975) define 7 brazos:

| Brazo | Hipótesis activas | Energía Libre + BMA | OOT + Refinamiento | Rol experimental |
|---|---|---|---|---|
| `FE-ARD` | `LieAffine`, `FeArd` | Sí (ARD bayesiano) | Sí | Reemplazo bayesiano directo de `BQ10` (sin GP multi-sustrato) |
| `GP-Cart-FE` | `GpCartM52`, `GpCartRbf`, `GpCoMoving` | Sí (VFE) | Sí | Arquitectura GP cartesiana + co-móvil por Energía Libre |
| `GP-Symp-FE` | `GpSymp{Fwd,Rev}`, `GpCoSymp{Fwd,Rev}` | Sí (VFE) | Sí | Arquitectura simpléctica/hamiltoniana por Energía Libre |
| `GP-Polar-FE` | `GpPolar`, `GpEquivCart`, `GpCartM52` | Sí (VFE + Jacobiano) | Sí | Arquitectura $SO(2)$-equivariante por Energía Libre |
| `METAFE-NoFE` | Todas (11) | **No** (solo replay MSE) | **No** | **Ablación causal H3:** prueba si la Energía Libre + metacognición es causal |
| `METAFE-NoRoll` | Todas (11) | Sí (VFE + LOO) | **No** | **Ablación causal H4:** prueba el aporte de inferencia activa multi-paso OOT |
| **`METAFE`** | **Todas (11)** | **Sí (VFE + LOO + BMA)** | **Sí (OOT-1/2 + Ref.)** | **Arquitectura primaria insignia** |
