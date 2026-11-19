# METAFE-3S — Control Metacognitivo de Tres Escalas sobre Energía Libre Variacional

Rama: `exp/metafe-3s-neuromod` · Implementación: [`orbita/src/constelacion4/metafe3s.rs`](../../orbita/src/constelacion4/metafe3s.rs) · Base congelada intacta: [`orbita/src/constelacion4/metafe.rs`](../../orbita/src/constelacion4/metafe.rs) · Preregistro: [`docs/metafe3s/PREREG.md`](PREREG.md).

---

## 1. Motivación: De METAFE v2 al Control Metacognitivo de Tres Escalas

En la rama previa `exp/metafe-free-energy`, **METAFE v2** rompió el techo del benchmark canónico (`new_med_log = -5.3338`, `new_acc16 = 88.67%`, `known_acc16 = 93.70%`, E33 `16/16` canónico y autónomo) mediante Energía Libre Variacional analítica exacta sobre 11 sustratos geométricos + metacognición "Yo-FE v2". Sin embargo, una auditoría de sus límites reveló **cuatro carencias estructurales** frente al principio biológico de *Control Metacognitivo de Tres Escalas*:

1. **Sin detección online de conflicto ni reset contextual ($\lambda_t$):** En episodios no estacionarios donde la ley física cambia abruptamente a mitad de trayectoria (**RS-A**), `METAFE v2` integra todo el historial de forma rígida sin detectar el conflicto ni limpiar su memoria de trabajo.
2. **Sin plasticidad sináptica rápida acotada sobre núcleo congelado ($W_0 + F_t$):** Cuando el núcleo estructural $W_0$ permanece estrictamente congelado y el entorno sufre una perturbación en caliente (**RS-B**), `METAFE v2` no posee pesos rápidos efímeros $F_t$ ni trazas de elegibilidad de tres factores $E_t$ acotadas espectralmente ($\Pi_{\|\cdot\|_F \le \rho}$) para adaptarse sin reescribir $W_0$.
3. **Ausencia de refutación cruzada en rollout abierto (sin etiquetas externas):** Durante el despliegue autorregresivo $h = K+1 \dots 16$, las hipótesis hacia adelante $\Phi_{\text{fwd}}$ pueden derivar coherentemente entre sí sin una segunda rama independiente que verifique consistencia temporal inversa $\Phi_{\text{rev}}(x_t) \approx x_{t-1}$.
4. **Sin consolidación offline protegida frente al olvido catastrófico (EWC):** Al incorporar nuevas familias de tareas de forma secuencial en el hiper-prior amortizado del núcleo lento $W_0$, el ajuste ingenuo sobrescribe los parámetros importantes de tareas previas.

**METAFE-3S** integra los **5 módulos matemáticos del Control Metacognitivo de Tres Escalas** directamente sobre el motor de Energía Libre Variacional de `METAFE v2` (sin alterar un solo bit de `METAFE v2`, `BQ10`, `ATLAS-FW` ni `CONSTELACIÓN v4`).

---

## 2. Flujo de la Arquitectura (Tres Escalas Temporales)

```
Entrada x_t ──► [ Núcleo Base W_0 (11 sustratos VFE, congelado) + Pesos Rápidos F_t ∈ Π_ρ ] ──► Salida y_t
                         ▲                                                   │
                         │ Enrutamiento g_t, Ganancia β_t                    │ Estados h_t, varianza σ²_epi
                         │                                                   ▼
               [ Controlador Neuromodulador MetaNet GRU ] ◄── [ Monitor ACC m_t ∈ R⁴ + Refutador Inverso Φ_rev ]
                    (emite α_t, λ_t, β_t, g_t)                  (sorpresa/refutación, desacuerdo, bucles, ‖F_t‖_F/ρ)
                         │                                                   ▲
                         │ Plasticidad 3-factores (α_t, λ_t, δ_t)            │ Verificación OOT / refutación / señal r_t
                         ▼                                                   │
               [ Traza de Elegibilidad E_t ] ◄───────────────────────────────┘
                         │
                         ▼ (solo offline / consolidación entre tareas)
               [ Buffer Episódico ──► Consolidación EWC con Fisher Ω_i hacia W_0^meta ]
```

---

## 3. Especificación Matemática de los 5 Módulos

### Módulo 1 — Núcleo Dual: Pesos Lentos Congelados $W_0$ + Adaptador Rápido Acotado $F_t$
- **Núcleo lento congelado $W_0$:** Las 11 hipótesis geométricas ajustadas por Energía Libre Variacional analítica exacta (`LieAffine`, `FeArd`, `GpCartM52`, `GpCartRbf`, `GpCoMoving`, `GpSymp{Fwd,Rev}`, `GpCoSymp{Fwd,Rev}`, `GpPolar`, `GpEquivCart`) permanecen intactas durante la inferencia y el rollout.
- **Adaptador plástico efímero $F_t \in \mathbb{R}^{2\times 6}$** ([`FastWeights`](../../orbita/src/constelacion4/metafe3s.rs#L48)): opera sobre rasgos locales normalizados $\psi(q, u) = [1,\; u_x/s,\; u_y/s,\; q_x/s,\; q_y/s,\; (r-\bar{r})/s]^\top$ con lectura suavemente saturada:
  $$u_{\text{out}} = u_{\text{base}} + s \cdot 2\tanh\!\left(\frac{F_t\,\psi(q, u_{\text{base}})}{2}\right)$$
  manteniendo linealidad $>99\%$ en régimen fino y cota estricta $\|F_t\|_F \le \rho$ mediante proyección Frobenius $\Pi_{\|\cdot\|_F \le \rho}$. Al finalizar cada episodio, $F_t$ se descarta por completo (`fw.reset()`).

### Módulo 2 — Monitor de Conflicto y Sorpresa (ACC) + Refutación Cruzada Inversa $\Phi_{\text{rev}}$
- **Monitor ACC** ([`AccMonitor`](../../orbita/src/constelacion4/metafe3s.rs#L243)): extrae en cada paso $t$ el vector de telemetría $m_t \in \mathbb{R}^4$ sin acceso a etiquetas futuras:
  1. $m_{t,0}$ (**Sorpresa latente / refutación**): $\text{clamp}((\ln(\varepsilon_t) + 5)/4,\; -1.5,\; 2.0)$.
  2. $m_{t,1}$ (**Incertidumbre epistémica / desacuerdo**): desacuerdo ponderado entre ramas activas del ensamble.
  3. $m_{t,2}$ (**Índice de estancamiento / bucles**): máximo entre similitud coseno con la ventana reciente $\{h_{t-2}, \dots, h_{t-w}\}$ y reversión de dirección $-\cos(\Delta h_t, \Delta h_{t-1})$.
  4. $m_{t,3}$ (**Norma de saturación plástica**): $\|F_t\|_F / \rho \in [0, 1]$.
- **Refutador Cruzado en Tiempo Inverso $\Phi_{\text{rev}}$** ([`BackwardRefuter`](../../orbita/src/constelacion4/metafe3s.rs#L338)): para evitar retroalimentación ciega durante el rollout abierto sin entorno externo, ajusta una segunda rama independiente exclusivamente sobre el contexto **invertido temporalmente** $[x_K, x_{K-1}, \dots, x_0]$ seleccionando entre 9 sustratos geométricos por VFE + LOO + replay inverso (con atajo afín exacto cuando `LieAffine` cierra a $<10^{-7}$). Su fiabilidad intrínseca $\pi_{\text{rev}} \in [0,1]$ se certifica mediante error de ciclo en contexto y validación precuencial inversa `rev_oot1` ($[x_K..x_1] \to x_0$):
  $$\pi_{\text{rev}} = \exp\!\left(-\frac{e_{\text{cyc}}^2}{0.045^2}\right)\exp\!\left(-\frac{e_{\text{rev-oot1}}^2}{0.12^2}\right)$$
  En inferencia, $\Phi_{\text{rev}}$ (a) penaliza hipótesis cuyas trayectorias violan la consistencia de ciclo $\Phi_{\text{rev}}(\hat{x}_t) \approx x_{t-1}$ (peso $w_{\text{refute}} \cdot \pi_{\text{rev}}$), (b) ofrece fusión bayesiana por inversión Newton $2\times 2$ (`cycle_manifold_correct`) si es validada en meta-validación, y (c) nutre el monitor ACC y la metacognición "Yo-FE 3S".

### Módulo 3 — Controlador Neuromodulador Recurrente (`MetaNet` GRU)
- **Red recurrente compacta** ([`NeuromodNet`](../../orbita/src/constelacion4/metafe3s.rs#L494)): celda GRU de estado oculto $s_t \in \mathbb{R}^4$ con conexión directa desde $m_t$, emitiendo en tiempo real:
  $$[\alpha_t,\; \lambda_t,\; \beta_t,\; g_t] = \text{MetaNet}(m_t, s_{t-1})$$
  - $\alpha_t \in [0,1]$ (**Noradrenalina / tasa de escritura**): próxima a $0$ en régimen estable; se abre ante sorpresa sostenida o perturbación.
  - $\lambda_t \in [0.02, 0.995]$ (**Factor de olvido / Reset**): cae bruscamente ($<0.45$) cuando el ACC señala cambio de régimen abrupto, disparando limpieza de contexto y de $F_t$.
  - $\beta_t \in [1.5, 12.0]$ (**Acetilcolina / precisión inversa**): modula la nitidez del enrutamiento entre hipótesis activas.
  - $g_t \in \Delta^3$ (**Compuerta dlPFC**): distribuye peso entre modo heurístico base, refutación acoplada y adaptación plástica/reset.
- **Entrenamiento outer-loop:** BPTT analítico exacto ponderado (`train_bptt`, 90 épocas, peso $6\times$ en pasos de choque) sobre secuencias auditadas (cambios de régimen sintéticos + rollouts de `cache_tr` + secuencias de saturación/bucles; 0 fuga `{8,10,11}`).

### Módulo 4 — Plasticidad de Tres Factores con Trazas de Elegibilidad y Acotación Espectral
- **Regla de tres factores** ([`EligibilityTrace`](../../orbita/src/constelacion4/metafe3s.rs#L177)):
  $$E_t = \gamma E_{t-1} + \frac{1}{N_p}\sum_{p=1}^{N_p} \phi(e_{t,p}) \otimes \frac{\psi(q_p, u_p)}{\|\psi\|_2^2 + 0.5}$$
  $$F_{t+1} = \Pi_{\|\cdot\|_F \le \rho}\!\Big(\lambda_t F_t + \alpha_t \cdot \delta_t \cdot E_t\Big)$$
  donde $\phi(e) = 2\tanh(e/(2s))$ acota valores atípicos.
- **Compuerta de seguridad precuencial:** En régimen estacionario canónico (`adapt_fast_weights_in_context`), $F_t$ solo se activa si mejora estrictamente la validación fuera de tiempo en contexto (`err_oot_1 < 0.97 * base` y `err_oot_closed < 0.98 * base`), garantizando **cero daño en familias afines exactas** ($F_t \equiv 0$). En régimen no estacionario con $W_0$ congelado (**RS-B**), la plasticidad de 3 factores gobernada por $(\alpha_t, \lambda_t)$ de `MetaNet` absorbe la perturbación online sin modificar $W_0$.

### Módulo 5 — Buffer Episódico + Consolidación Offline EWC + "Yo-FE 3S"
- **Buffer Episódico** ([`EpisodicBuffer`](../../orbita/src/constelacion4/metafe3s.rs#L791)) y **consolidación EWC** ([`compute_fisher_diag`](../../orbita/src/constelacion4/metafe3s.rs#L885), [`consolidate_prior_ewc`](../../orbita/src/constelacion4/metafe3s.rs#L915)): tras completar cada fase de experiencias, destila las trayectorias discriminativas hacia el hiper-prior amortizado lento $W_0^{\text{meta}} \in \mathbb{R}^{11\times 10}$ minimizando:
  $$\mathcal{L}_{\text{consolidación}}(W_0) = \mathcal{L}_{\text{CE}}(W_0) + \frac{\mu}{2}\sum_{h,j} \Omega_{h,j}\bigl(W_{0,h,j} - W_{0,h,j}^{\text{prev}}\bigr)^2 + \frac{\lambda_2}{2}\|W_0\|_F^2$$
  donde $\Omega_{h,j} = \mathbb{E}\bigl[(\partial \mathcal{L}_{\text{CE}}/\partial W_{0,h,j})^2\bigr]$ es la diagonal exacta de la matriz de información de Fisher de las tareas anteriores.
- **Metacognición "Yo-FE 3S":** Desacopla la predicción de error en Etapa A-core (10 rasgos núcleo forward VFE+OOT+epistémicos con ridge $\lambda=0.05$ en `cache_tr`, preservando el condicionamiento de *Yo-FE v2*), Etapa A-ref (corrección no negativa L2-encogida $w_{\text{ref}}^{\text{meta}}\ge 0$ fuera de muestra en `cache_va` sobre el exceso de refutación estrictamente **en soporte epistémico** de $\Phi_{\text{rev}}$ ponderado por $w_{\text{rev},t}=\pi_{\text{rev}}/(1+\text{epi}_{\text{rev},t}/s^2_{\text{rev}})$), y Etapa B (regresión logística Platt L2 $\lambda=1.5$ en `cache_va` con 8 rasgos balanceando episodios no triviales $1.0$ vs afines triviales $0.35$). La selección de hiperparámetros valida los finalistas top-$K$ + mejor candidato sin refutación directamente bajo inferencia 3S exacta.

---

## 4. Mapeo Directo: Biología vs. Ingeniería en METAFE-3S

| Mecanismo Biológico | Equivalente en METAFE-3S | Falla que Previene |
|---|---|---|
| Corteza estructural (pesos lentos) | Núcleo base $W_0$ (11 hipótesis VFE congeladas) | Olvido catastrófico y deriva en caliente |
| Plasticidad sináptica a corto plazo | Matriz rápida acotada $F_t \in \Pi_\rho$ con decaimiento $\lambda_t$ | Explosión espectral y reescritura permanente |
| Cingulado anterior (ACC) | Monitor de conflicto $m_t \in \mathbb{R}^4$ + refutador inverso $\Phi_{\text{rev}}$ | Razonar en círculos o derivar en rollout sin darse cuenta |
| Locus Coeruleus (Noradrenalina) | Tasa de escritura $\alpha_t$ + señal de reset $\lambda_t$ de `MetaNet` | Persistencia ciega tras cambio abrupto de ley (RS-A) |
| Acetilcolina | Ganancia de precisión $\beta_t \in [1.5, 12.0]$ | Dogmatismo excesivo en OOD o dilución en régimen exacto |
| Córtex prefrontal dorsolateral | Compuerta de modos cognitivos $g_t \in \Delta^3$ | Reescribir pesos cuando basta con conmutar de ruta |
| Plasticidad de tres factores | Traza de elegibilidad $E_t$ modulada por $\alpha_t \cdot \delta_t$ | Auto-refuerzo ciego hebbiano en rollout |
| Consolidación hipocampal (Sueño) | Buffer Episódico + destilación offline EWC ($\Omega_i$) | Olvido catastrófico de tareas previas al consolidar $W_0^{\text{meta}}$ |

---

## 5. Brazos Congelados y Protocolo de Ablación Causal

Para aislar causalmente cada módulo sobre exactamente las mismas semillas y episodios ([`metafe3s_arms`](../../orbita/src/constelacion4/metafe3s.rs#L975)):

| Brazo | $W_0$ VFE | Pesos Rápidos $F_t$ + $E_t$ | Refutador $\Phi_{\text{rev}}$ | `MetaNet` GRU $(\alpha,\lambda,\beta,g)$ | Rol Experimental |
|---|---|---|---|---|---|
| `METAFE-v2` | Sí (intacto) | No | No | No | Línea base puntera previa (emparejada bit a bit) |
| **`METAFE-3S`** | **Sí** | **Sí ($\Pi_\rho$)** | **Sí ($\Phi_{\text{rev}}$)** | **Sí (BPTT)** | **Arquitectura insignia completa (5 módulos)** |
| `METAFE-3S-NoFast` | Sí | **No** | Sí | Sí | Ablación causal Módulos 1+4 (plasticidad rápida $F_t$) |
| `METAFE-3S-NoRefute` | Sí | Sí | **No** | Sí | Ablación causal Módulo 2 (refutación cruzada inversa) |
| `METAFE-3S-NoNM` | Sí | Sí | Sí | **No** (constante) | Ablación causal Módulo 3 (controlador neuromodulador) |

Adicionalmente, el módulo 5 se evalúa de forma dedicada en cada semilla mediante [`eval_continual_ewc`](../../orbita/src/constelacion4/metafe3s.rs#L2876) contrastando consolidación secuencial de 3 tareas (`T1 → T2 → T3`) con **`EWC` ($\mu=25$)** vs **`NoEWC` ($\mu=0$)**.
