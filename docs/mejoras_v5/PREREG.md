# PREREGISTRO — CONSTELACIÓN v5: mejoras estructurales M4 / M5 / M6 y `dinlat-v2` (M7)

Rama `exp/reporte-mejoras-v5` (base consolidada `exp/cst-vs-orbita` @ `c1b1f54` + `exp/constelacion-v4-mejoras` @ `0967539`). Especificación técnica detallada en [`REPORTE_MEJORAS.md`](REPORTE_MEJORAS.md) y diagnóstico empírico en [`DIAGNOSTICO.md`](DIAGNOSTICO.md). Ninguna corrida en semillas DEV (`0xA400–0xA40F`) ni CONFIRM (`0xB400–0xB40F`) se ejecuta antes de congelar los hiperparámetros de tuning en `0xC420–0xC423` / `0xD420–0xD423`.

---

## 1. Pregunta e hipótesis centrales

¿Pueden las mejoras estructurales sobre CONSTELACIÓN v4 —(M4) campo rápido bifásico afín+residual ortogonalizado en el prefijo completo $\mathcal{B}_{1:t}$, (M5) destilación causal del prior de ÓRBITA en $W^F(0)$ cuando hay contexto, y (M6) jerarquía de codificación predictiva de transición con pooling de constelación— superar a CONSTELACIÓN v4 en familias nuevas sin dañar las familias conocidas, resolver el colapso en `E33-ctx` / `1-paso` y lograr por primera vez atribución positiva de los componentes de codificación predictiva?

---

## 2. Brazos experimentales

| Brazo | Descripción |
|---|---|
| **`v4`** | CONSTELACIÓN v4 congelada (`Cfg::v4`, hash de referencia idéntico a `artifacts/constelacion_v4`). |
| **`m4`** | `v4` + **M4**: campo rápido bifásico en el prefijo $\mathcal{B}_{1:t}$ (`W^F_aff` por RLS/Cholesky $3\times 3$ con $\lambda_{\text{aff}}=10^{-6}$ $\oplus$ corrector ortogonalizado $g_{\text{res}}\cdot W^F_{\text{res}}$ sobre base espacial $\psi_\perp(x) \in \mathbb{R}^{24}$). |
| **`m4_aff`** (control de atribución M4) | Sólo Fase 1 de M4 (`g_res = 0` forzado): mide cuánto aporta el corrector residual ortogonalizado $W^F_{\text{res}}$ sobre el ajuste afín exacto en el prefijo. |
| **`m5`** | `v4` + **M5** (`orb_prior_wf`): inicialización de las estadísticas suficientes de $W^F(0)$ mediante $m_{\text{vir}}=2$ pasos imaginados de ÓRBITA cuando el contexto $c$ está disponible ($k=0$ en `E33-ctx` y `1-paso`). |
| **`m6`** | `v4` + **M6** (`PC-Trans`): entrada de transición $u_p(t) = [x_p(t-1), \alpha_\Delta \Delta x_p(t)] \in \mathbb{R}^4$, gradiente de $e^P$ acoplado a $\mu_1$ y banco lento de $K_B=8$ cabezas $W^P(\bar{\mu}_L)$. |
| **`v5_all`** | Combinación `M4 + M5 + M6` (exploratorio en DEV si $\ge 1$ brazo individual pasa la compuerta; pasa a CONFIRM si cumple su propia compuerta DEV). |
| **Controles de atribución PC (sobre `m6` / `v5_all`)** | `fixed_w` ($W_{1..L}$ y $W^P_b$ congelados), `no_err` (sin neuronas de error ascendentes), `t1` ($T_{\text{inf}}=1$), `no_fast` ($W^F=0$). |

---

## 3. Protocolo de Paso 0 y Tuning (antes de DEV)

### 3.1 Paso 0 (paridad bit a bit y sanidad)
- **0a (Paridad bit a bit con v4):** Con las banderas `m4 = false, m5 = false, m6 = false`, el binario debe producir en las semillas de humo `0xC420` y `0xC421` exactamente el mismo hash FNV-1a (`1985bdbad5eee41f`) que `cst4-run --arms cst` y `cst4m-run --arm v4`.
- **0b (Cero llamadas a ÓRBITA sin contexto):** En evaluación held-out ($k=4$, sin etiquetas de familia), `m4` y `m6` realizan `orb_calls == 0`.
- **0c (Causalidad estricta en el prefijo):** Perturbar los cuadros futuros $\tau > t$ no altera en un solo bit $G_{\text{aff}}(t)$, $B_{\text{aff}}(t)$, $g_{\text{res}}(t)$, $W^F_{\text{res}}(t)$ ni $\bar{\mu}_L(t)$.
- **0d (No-daño afín por construcción en M4):** En cualquier episodio sintético de las familias afines $0..6$ sin ruido, $\text{MSE}_{\text{aff}}(K) < 10^{-16}$ y $g_{\text{res}}(K) < 10^{-4}$.
- **0e (Cero consultas TEST):** `v4_provenance::snapshot().delta(&before).total() == 0`.

### 3.2 Tuning (exclusivamente en `0xC420–0xC423` y `0xD420–0xD423`)
- **M4:** rejilla sobre $\lambda_{\text{aff}} \in \{10^{-6}, 10^{-5}\}$, $\lambda_{\text{res}} \in \{10^{-3}, 10^{-2}, 10^{-1}\}$, $\tau_{\text{aff}} \in \{-14, -12, -10\}$ evaluada sólo en familias conocidas retenidas (`0..7`, criterio = media de `med_log` en $k=2$ y $k=8$) y en validación cruzada interna (`0..5` $\to$ `6..7`, donde `7 = radial_warp` es no lineal). Restricción dura de selección: ninguna configuración que empeore `known_med_log` respecto a `v4` (`−10,12`) puede seleccionarse.
- **M5:** selección de $\alpha_0 \in \{0{,}1, 0{,}25, 0{,}5, 1{,}0\}$ sobre `E33-ctx` en semillas `0xC420–0xC423`.
- **M6:** selección de $\alpha_\Delta \in \{1{,}0, 2{,}0\}$, $\kappa_P \in \{0{,}05, 0{,}1, 0{,}25\}$ y $K_B \in \{4, 8\}$ en `0xC430–0xC433`.

---

## 4. Diseño experimental DEV y CONFIRM

- **Semillas:** DEV `0xA400–0xA40F` ($n=16$), CONFIRM `0xB400–0xB40F` ($n=16$). Sin reajuste de ningún parámetro entre DEV y CONFIRM.
- **Estadística:** Bootstrap pareado por semilla, $10\,000$ réplicas, semilla `12345`, percentil. Corrección por multiplicidad de las 3 familias de mejoras (`M4`, `M5`, `M6`): **IC98,33% (Bonferroni $\times 3$)** para la métrica primaria; Holm-Bonferroni reportado como análisis de sensibilidad. $\delta = 0{,}05$.

### Criterios de decisión por brazo
1. **Primaria (para M4, M6 y `v5_all`):**
   $$\Delta\text{med\_log rel\_err } h_{16}\text{ en familias nuevas }\{8, 10, 11\},\; k=4,\; \text{brazo} - \text{v4} < 0.$$
   - **PASA** si el límite superior del IC98,33% es $< 0$ en DEV **y** en CONFIRM.
2. **Primaria específica de M5 (modo con contexto $k=0$):**
   $$\Delta\text{acc}_{16}\text{ en E33-ctx},\; \text{m5} - \text{v4} > +0{,}50 \quad \text{y} \quad \Delta\text{acc}_{1\text{-paso}} > +0{,}80$$
   con IC95 inferior por encima del umbral en DEV y CONFIRM, manteniendo $\Delta\text{med\_log nuevas } (k=4) = 0$ exacto cuando no se provee contexto.
3. **No-daño (obligatorio para todos los brazos):**
   - $\Delta\text{acc}_{16}\text{ conocidas (brazo} - \text{v4)}$: límite inferior del IC95 $> -\delta$ ($\delta = 0{,}05$).
   - $\Delta\text{acc } h_{16}\text{ en E33-k4 (brazo} - \text{v4)}$: límite inferior del IC95 $> -\delta$, y pase E33 $\ge 15/16$ semillas en ambas fases.
4. **Atribución de Codificación Predictiva (para M6 / `v5_all`):**
   - Se declara atribución positiva a PC si y sólo si `brazo − fixed_w < 0` y `brazo − no_err < 0` con IC95 superior $< 0$ en DEV y CONFIRM.
5. **Compuerta de parada DEV $\to$ CONFIRM:**
   - Un brazo pasa a CONFIRM si y sólo si cumple su criterio primario en DEV, cumple el no-daño en DEV y presenta $\le 1$ semilla con NaN/colapso.
   - Si en DEV el límite inferior del IC de la primaria es $> 0$ o el IC95 de no-daño cae íntegro por debajo de $-\delta$, el brazo se detiene y no corre CONFIRM.

---

## 5. Sub-experimento M7 (`dinlat-v2` sobre latentes Gemma capa 13)

- **Datos:** Mismos episodios y embeddings cacheados de `exp/orbita-dinamica-latente` (`16` semillas DEV `0xA400–0xA40F` + `16` CONFIRM `0xB400–0xB40F`), verificando identidad bit a bit de `PERS` y `AR2` con `res_<seed>.json`.
- **Hipótesis primaria (`H1_v2`):** $G(\text{ORB\_LIBRE\_v2 vs AR2, ho}) = \log(\text{MSE}_{\text{AR2}} / \text{MSE}_{\text{ORB\_LIBRE\_v2}})$ tiene límite inferior del IC95 $> \delta = 0{,}05$ en DEV y CONFIRM.
- **No-inferioridad garantizada (`H0_anchor`):** $G(\text{ORB\_LIBRE\_v2 vs AR2, ho})$ tiene límite inferior del IC95 $> -0{,}01$ gracias a la compuerta de validación cruzada interna $\gamma_{\text{res}} \in [0, 1]$ sobre el anclaje `VAR(2)`.
- **Transferencia (`H4_v2`):** $G(\text{ORB\_LIBRE\_v2 vs AR2, tf})$ con límite inferior del IC95 $> 0$ en familias nunca vistas (`cizalla`, `tienda`, `rodar_suma`).
