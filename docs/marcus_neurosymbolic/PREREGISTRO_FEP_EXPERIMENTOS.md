# Pre-registro Formal de Experimentos: FEP-MARCUS
## Arquitectura Basada 100% en Maximizar la Energía Libre de Evidencia (ELBO) y Reducir la Sorpresa

> **Científicos de Referencia:** Karl Friston (Free Energy Principle & Active Inference), Gary Marcus (*The Algebraic Mind*, *Rebooting AI*, *Taming Silicon Valley*), Judea Pearl (*Causality*).  
> **Rama Git:** `exp/marcus-free-energy-active-inference`  
> **Crate:** `marcus_neurosymbolic::free_energy`  
> **Protocolo:** DEV (16 semillas 0xA400–0xA40F) y CONFIRM ciega (16 semillas 0xB400–0xB40F) con Paired Bootstrap (10,000 remuestreos, IC95%, cota de no inferioridad $\delta = 0.05$).

---

## 1. Justificación y Fundamentación Teórica

El **Principio de Energía Libre (FEP)** postula que los agentes biológicos y cognitivos se auto-organizan minimizando una cota superior de la sorpresa sensorial:
$$\mathcal{S}(o) = -\ln p(o)$$
Esta cota es la **Energía Libre Variacional**:
$$\mathcal{F}(q, o) = \mathbb{E}_{q(s)} [\ln q(s) - \ln p(o, s)] = D_{\mathrm{KL}}(q(s) \parallel p(s)) - \mathbb{E}_{q(s)} [\ln p(o \mid s)] \ge \mathcal{S}(o)$$

En el aprendizaje automático y la termodinámica computacional, la función objetivo a maximizar es la **Energía Libre de Evidencia** o **ELBO**:
$$\text{ELBO} = - \mathcal{F}(q, o) = \mathbb{E}_{q(s)} [\ln p(o \mid s)] - D_{\mathrm{KL}}(q(s) \parallel p(s)) \le \ln p(o) = -\mathcal{S}(o)$$

### La Limitación Crítica del FEP Estándar sin Gary Marcus
Los modelos clásicos de Inferencia Activa asumen espacios latentes tabulares o continuos no estructurados ($\mathbb{R}^d$), lo cual ocasiona:
1. **Falta de ligamiento de variables:** Incapacidad de operar con variables formales $f(x) = x$ sobre entradas fuera de distribución (OOD).
2. **Crosstalk y diafonía sintáctica:** Ruptura de composicionalidad cuando múltiples entidades interactúan.
3. **Confusión causal:** Confusión entre condicionamiento observacional $P(Y \mid X)$ e intervención causal $P(Y \mid do(X))$.
4. **Alucinaciones por priors descalibrados (Dark Room Problem):** Colapso en estados degenerados.

### La Síntesis FEP-MARCUS
Al dotar al estado latente $s$ de estructura algebraica, física Spelke innata y grafos causales:
$$s = \langle \Gamma, \mathcal{T}, \mathcal{P}, \Pi \rangle$$
la minimización de $\mathcal{F}$ (maximización del ELBO) y la reducción de sorpresa $\mathcal{S}$ operan de forma explicable, verificable y con 100% de generalización zero-shot.

---

## 2. Hipótesis Formales Pre-registradas

| ID | Enunciado Formal | Métrica Primaria | Criterio de Aceptación (DEV & CONFIRM) |
|---|---|---|---|
| **H-FEP1** | **Convergencia Monótona de Energía Libre:** El optimizador variacional sobre el sustrato estructurado reduce la energía libre $\mathcal{F}$ maximizando el ELBO. | Tasa de reducción $\frac{\mathcal{F}_{\text{init}} - \mathcal{F}_{\text{opt}}}{\mathcal{F}_{\text{init}}}$ y monotonicidad. | $\text{Reducción} \ge 30.0\%$ con $\text{Monotonicidad} \ge 95.0\%$ en ambas fases. |
| **H-FEP2** | **Reducción de Sorpresa Multimodal:** La descomposición de la sorpresa en canales (perceptual, algebraico, físico y causal) permite erradicar anomalías. | Tasa de reducción de sorpresa $\mathcal{S}_{\text{total}}$ y aislamiento de anomalías físicas. | $\text{Reducción} \ge 85.0\%$ y aislamiento de anomalía física $\ge 95.0\%$. |
| **H-FEP3** | **Exploración Causal por Inferencia Activa:** Ante variables confusoras latentes, la minimización de Energía Libre Esperada $\mathcal{G}(\pi)$ selecciona intervenciones $do(X)$ en vez de acciones pasivas. | Exactitud en detección de causalidad verdadera y preferencia por acciones epistémicas. | Exactitud Causal $= 100\%$ y preferencia epistémica $\ge 95.0\%$. |
| **H-FEP4** | **Generalización OOD con Energía Libre Acotada:** Frente a tokens ajenos al entrenamiento, MARCUS-FEP mantiene la energía libre $\mathcal{F}_{\text{OOD}} \le 15.0$ con 100% de precisión. | Exactitud zero-shot en gramática algebraico $ABB$ vs $ABA$ y $\mathcal{F}_{\text{OOD}}$. | Exactitud OOD $= 100\%$, $\mathcal{F}_{\text{OOD}} \le 15.0$, superando a baseline con $\delta > 0.05$. |
| **H-FEP5** | **Estabilidad Sináptica con Ley de Dale:** La modulación dinámica de precisión $\Pi$ en la memoria BQ10 preserva la no-negatividad sináptica. | Recuento de violaciones de la Ley de Dale ($W^E < 0$ o $W^I < 0$). | Exactamente 0 violaciones en todas las semillas. |
| **H-FEP6** | **Supresión Absoluta de Ilusiones Físicas:** Cualquier hipótesis que implique teletransportación o interpenetración es vetada por disparar sorpresa infinita. | Recall de detección de violaciones físicas Spelke y tasa de falsos positivos (FPR). | $\text{Recall} = 1.0$, $\text{FPR} = 0.0$ en todas las semillas. |

---

## 3. Protocolo Estadístico

- **Unidad de Análisis:** Semilla pseudo-aleatoria (16 semillas en DEV, 16 en CONFIRM).
- **Remuestreo:** Paired Bootstrap con $N_B = 10,000$ iteraciones, generador `PCG64` con semilla canónica `12345`.
- **Nivel de Confianza:** 95% empírico basado en percentiles $[2.5\%, 97.5\%]$.
- **Cota de No Inferioridad / Superioridad:** $\delta = 0.05$.
- **Regla de Parada y Aceptación:** Una hipótesis sólo es declarada `CONFIRMADA` si el límite inferior del IC95% en CONFIRM supera estrictamente el umbral preregistrado sin retroceso en DEV.
