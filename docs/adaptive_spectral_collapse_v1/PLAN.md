# Plan Preregistrado — `Riemann Wave Inference → Collapse → Adaptive State Learning` (`exp/adaptive-spectral-collapse`)

**Escrito y comprometido en git antes de implementar cualquier código en `adaptive_spectral_collapse/`.**

- **Rama aislada:** `exp/adaptive-spectral-collapse` (desde `main@120e1e8`; **sin fusionar** con `main` ni tocar `orbita`, `aura-engine`, `exp/riemann-wave-inference`, `exp/riemann-wave-collapse` ni `exp/riemann-prime-inference`).
- **Mapa explícito de ramas (resolución de trazabilidad):**
  - `exp/riemann-wave-inference` → `riemann_wave/`, `docs/riemann_wave_v1/`
  - `exp/riemann-wave-collapse` → `riemann_collapse/`, `docs/riemann_collapse_v1/`
  - `exp/riemann-prime-inference` → `riemann_prime_inference/`, `docs/riemann_prime_inference_v1/`
  - `exp/adaptive-spectral-collapse` → `adaptive_spectral_collapse/`, `docs/adaptive_spectral_collapse_v1/` *(este experimento)*
- **Crate autónomo:** `adaptive_spectral_collapse/` (Rust puro con `[workspace]` propio y **0 dependencias externas**; orquestación, bootstrap pareado y graficación SVG en Python con biblioteca estándar).

---

## 1. Pregunta Científica (Evaluación Crítica a Resolver)

Los tres experimentos previos demostraron resonancia espectral, colapso Pasado × Futuro e inferencia bidireccional sobre un alfabeto pequeño ($K = 12$) con frecuencias de Riemann y codificación prima diseñadas a mano. La evaluación crítica señaló tres huecos fundamentales que este cuarto experimento somete a prueba falsable:

1. **Entrada desconocida (sin primos ni frecuencias prefijadas):** ¿puede el campo recibir conceptos arbitrarios y opacos ($\mathbf{x}_k \in \mathbb{R}^{16}$, sobrecompletos y no ortogonales, $K_{\text{base}} = 24 \to K_{\text{tot}} = 28$, con 276–378 pares) sin recibir jamás primos ni ceros de $\zeta(s)$?
2. **Aprendizaje autónomo del campo (descubrimiento vs. reconstrucción):** partiendo de frecuencias y estados **completamente aleatorios**, ¿puede el campo adaptar conjuntamente su geometría de estados $\mathbf{z}_k = W_{\text{enc}} \mathbf{x}_k \in \mathbb{R}^3$ y sus modos espectrales $(\omega, \mathbf{k}, a, \phi)$ únicamente mediante experiencia genérica (compatibilidad de colapso + repulsión anti-colisiones), sin función objetivo específica de primos ni de Riemann?
3. **Generalización real y comparación justa:** sobre **pares nunca vistos (60% de retención estricta)** y **4 conceptos completamente nuevos** añadidos de forma continua (`few-shot`), ¿qué ocurre al comparar el campo espectral adaptativo frente a **Riemann fijo, Fourier, Dyson GUE, Poisson, QMC (Halton) y una red neuronal convencional (MLP)** con **capacidad paramétrica y presupuesto computacional equivalentes**?

---

## 2. Entorno de Conceptos Desconocidos y Partición Estricta de Generalización

En cada semilla se genera un mundo sintético opaco independiente:
- **Conceptos:** $K_{\text{base}} = 24$ conceptos base (`0..24`, $\binom{24}{2} = 276$ pares distintos) más $K_{\text{new}} = 4$ conceptos nuevos (`24..28`) reservados exclusivamente para la Suite `A4` (aprendizaje continuo).
- **Entradas sensoriales opacas (sin primos ni etiquetas espectrales):** cada concepto $c_k$ posee un vector sensorial aleatorio $\mathbf{x}_k \sim \mathcal{N}(0, I_{16})$ normalizado ($\|\mathbf{x}_k\|_2 = 1$). Al ser 28 vectores en $\mathbb{R}^{16}$ (sobrecompletos), existen correlaciones cruzadas e interferencia real entre conceptos.
- **Observación compuesta:** al interactuar dos conceptos $\{a, b\}$, el mundo emite la mezcla continua ruidosa:
  $$\tilde{\mathbf{x}}_{a,b} = \frac{1}{\sqrt{2}}(\mathbf{x}_a + \mathbf{x}_b) + \boldsymbol{\epsilon}, \qquad \boldsymbol{\epsilon} \sim \mathcal{N}(0, \sigma_x^2 I_{16})$$
  con $\sigma_x = 0.015$ durante entrenamiento y $\sigma_x = 0.025$ en evaluación OOD estándar (más barrido hasta $\sigma_x = 0.100$ en Suite `A3b`).
- **Ley relacional no lineal oculta (nunca revelada):** cada concepto tiene rasgos causales latentes no observados $\mathbf{v}_k \sim \mathcal{N}(0, 0.7^2 I_4)$ y cada par produce:
  $$y_1(a, b) = \tanh\!\big(v_{a,0} + v_{b,0} + 0.85\, v_{a,1} v_{b,1}\big), \qquad y_2(a, b) = \sin\!\big(v_{a,2} + v_{b,2}\big) + 0.45\,\tanh\!\big((v_{a,3} - v_{b,3})^2 - 0.5\big)$$
  con clase discreta 4-way $c(a, b) = 2\,\mathbb{I}(y_1 \ge 0) + \mathbb{I}(y_2 \ge 0) \in \{0, 1, 2, 3\}$.
- **Partición estricta entrenamiento / nunca vistos:** de los 276 pares base, **110 pares (39.9%)** forman `TRAIN_PAIRS` (garantizando conectividad y $\ge 6$ apariciones por concepto) y **166 pares (60.1%)** forman `UNSEEN_PAIRS` (**jamás observados** en ningún paso de aprendizaje de representación, campo espectral ni cabeza causal).

---

## 3. Arquitectura Unificada y los 7 Brazos Comparados (Paridad de Capacidad y Cómputo)

### A. Campo Espectral Multi-Canal con Colapso Pasado × Futuro (6 brazos espectrales)
Cada brazo espectral proyecta $\mathbf{x} \in \mathbb{R}^{16} \mapsto \mathbf{z} = W_{\text{enc}} \mathbf{x} \in \mathbb{R}^3$ (`48` parámetros), de modo que la mezcla sensorial se mapea linealmente a la suma de fases $\tilde{\mathbf{u}}_{a,b} = \sqrt{2}\,W_{\text{enc}}\tilde{\mathbf{x}}_{a,b} = \mathbf{z}_a + \mathbf{z}_b + \tilde{\boldsymbol{\eta}} \in \mathbb{R}^3$. El campo consta de:
- **3 canales espectrales 1D** ($M_1 = 12$ modos por canal = 36 modos 1D): $\mathcal{W}_d(u) = \sum_{m=1}^{12} a_{d,m}\cos(\omega_{d,m} u + \phi_{d,m})$ (`108` parámetros).
- **1 campo de coherencia cruzada 3D** ($M_3 = 24$ vectores de onda $\mathbf{k}_m \in \mathbb{R}^3$): $\mathcal{W}_{\text{3D}}(\mathbf{u}) = \sum_{m=1}^{24} a_m \cos(\mathbf{k}_m \cdot \mathbf{u} + \phi_m)$ (`120` parámetros).
- **Cabeza causal composicional aprendida** (aprende rasgos $\hat{\mathbf{v}}_k = W_v \mathbf{x}_k + \mathbf{b}_v \in \mathbb{R}^4$ y red simétrica $[\hat{\mathbf{v}}_a + \hat{\mathbf{v}}_b,\; \hat{\mathbf{v}}_a \odot \hat{\mathbf{v}}_b,\; (\hat{\mathbf{v}}_a - \hat{\mathbf{v}}_b)^2] \to 16 \to 2$ únicamente sobre `TRAIN_PAIRS`): `310` parámetros.
- **Total brazo espectral:** **`586` parámetros** (idénticos en los 6 brazos espectrales).

**Aprendizaje autónomo del campo (sin primos ni $\zeta(s)$):**
1. **Autoorganización geométrica anti-colisiones ($W_{\text{enc}}$):** minimiza repulsión entre estados atómicos $\mathbf{z}_i, \mathbf{z}_j$, repulsión entre sumas distintas $(\mathbf{z}_a + \mathbf{z}_b)$ vs. $(\mathbf{z}_c + \mathbf{z}_d)$ únicamente en `TRAIN_PAIRS` (descubriendo una geometría tipo Sidon libre de colisiones), contención de cavidad $[-1.8, 1.8]^3$ y regularización de norma $\|W_{\text{enc}}\|_F^2$ para controlar la amplificación de ruido.
2. **Adaptación espectral por experiencia:** maximiza el contraste constructivo en estados observados $\mathbf{z}_k$, supresión destructiva en el vacío fuera de estados y compatibilidad de colapso Pasado × Futuro en `TRAIN_PAIRS`.
3. **Decodificación puramente ondulatoria por colapso (0 búsqueda por tabla de pares):** barrido continuo de 160 puntos en cada canal 1D ($s_d(u_d \mid \tilde{u}_d) = \max(0, \mathcal{W}_d(u_d))\max(0, \mathcal{W}_d(\tilde{u}_d - u_d))$), extracción de los top-5 picos locales por canal ($\le 125$ intersecciones continuas en $\mathbb{R}^3$), colapso conjunto con $\mathcal{W}_{\text{3D}}(\mathbf{u})\mathcal{W}_{\text{3D}}(\tilde{\mathbf{u}} - \mathbf{u})$ más refinamiento continuo local, y proyección al concepto atómico más cercano $(\hat{a}, \hat{b})$.

Los **6 brazos espectrales** comparten exactamente el mismo optimizador, número de pasos de entrenamiento, cabeza causal y codificador $W_{\text{enc}}$, diferenciándose únicamente en las frecuencias $(\omega_{d,m}, \mathbf{k}_m)$:
1. **`adaptive_spectral`** *(modelo propuesto)*: inicia con frecuencias aleatorias uniformes (idénticas a `poisson_fixed`) y **adapta libremente todas las frecuencias $(\omega_{d,m}, \mathbf{k}_m)$**, amplitudes y fases por gradiente desde la experiencia (**0 ceros de Riemann, 0 primos**).
2. **`riemann_fixed`**: frecuencias congeladas a los ceros exactos de Riemann $\gamma_n$ reescalados a la banda de cavidad; adapta $W_{\text{enc}}, a, \phi$.
3. **`fourier_fixed`**: armónicos enteros regulares de Fourier congelados; adapta $W_{\text{enc}}, a, \phi$.
4. **`dyson_gue_fixed`**: niveles con repulsión GUE de Dyson ($\beta = 2$) congelados; adapta $W_{\text{enc}}, a, \phi$.
5. **`poisson_fixed`**: frecuencias aleatorias uniformes (Poisson) congeladas; adapta $W_{\text{enc}}, a, \phi$.
6. **`qmc_halton_fixed`**: secuencia Quasi-Monte Carlo de baja discrepancia Halton (bases 2, 3, 5) congelada; adapta $W_{\text{enc}}, a, \phi$.

### B. Modelo Neuronal Convencional con Capacidad y Cómputo Equivalentes (7.º brazo)
7. **`mlp_conventional`**: red neuronal estándar entrenada end-to-end con Adam y regularización L2 directamente sobre las observaciones compuestas $\tilde{\mathbf{x}}_{a,b} \in \mathbb{R}^{16}$ (y los 24 conceptos atómicos) sobre el mismo `TRAIN_PAIRS`:
   - Arquitectura: `16 -> 20 (ReLU) -> 16 (ReLU)` + cabeza de abducción multi-etiqueta `16 -> 24` (BCE, top-2 logits) + cabeza predictiva `16 -> 2` (MSE).
   - **Parámetros totales:** `340 + 336 + 408 + 34` = **`1,118` parámetros** (capacidad **generosamente igual o superior** a los `586` parámetros de `adaptive_spectral`, garantizando cero desventaja artificial para el MLP) con presupuesto de cómputo y pasos de optimización equivalente (registrando parámetros y tiempos exactos en `run.json`).
   - En contrafactuales $do(c_a \to c_{\text{new}})$ y analogías $A:B::C:?$, opera directamente en el espacio sensorial desplazado $\tilde{\mathbf{x}}_{\text{cf}} = \tilde{\mathbf{x}}_{a,b} - \frac{1}{\sqrt{2}}\mathbf{x}_{\hat{a}} + \frac{1}{\sqrt{2}}\mathbf{x}_{c_{\text{new}}}$ y $\tilde{\mathbf{x}}_D = \tilde{\mathbf{x}}_C + (\tilde{\mathbf{x}}_B - \tilde{\mathbf{x}}_A)$.
   - En aprendizaje continuo (`A4`), expande su cabeza `24 -> 28` y ajusta con repetición de memoria (*experience replay*) de pares antiguos + los 16 pares nuevos bajo el mismo presupuesto de pasos que los brazos espectrales.

### §3.1 Adenda Transparente pre-Freeze (Diagnóstico `DEV`, antes de `FROZEN.json` y sin tocar ningún criterio de §4)
Durante el diagnóstico en `DEV` (`0xC100..=0xC10F`) se detectaron dos desbalances físicos que perjudicaban artificialmente a los 6 brazos espectrales frente al MLP (`1,118` parámetros):
1. Con `M_3D = 24` modos planos en 3D para $K = 24$ estados (`586` parámetros, la mitad que el MLP), la razón señal-a-ruido holográfica teórica era $\sqrt{M/K} = \sqrt{24/24} = 1.0$, produciendo techo de abducción incluso sin ruido. Al escalar el campo espectral a **$M_1 = 24$ modos/canal 1D** y **$M_3 = 96$ vectores de onda 3D** con decodificación continua de paquete de onda sobre ráster espacial uniforme $11^3$ + ascenso de gradiente continuo (0 tabla de pares), se alcanza verdadera paridad paramétrica 1:1 (**`962` parámetros espectrales** vs. **`1,118` parámetros MLP**).
2. La proyección lineal $W_v \in \mathbb{R}^{4 \times 16}$ tenía rango máximo 16 para 24 conceptos con rasgos 4D independientes; se sustituye por una tabla aprendida de rasgos causales por estado descubierto $\hat{\mathbf{v}}_k \in \mathbb{R}^4$ (`96` params) + red simétrica `12 -> 8 (tanh) -> 2` (`122` params, total `218` params en cabeza causal, `962` totales).
3. **Inmutabilidad:** Ninguna de las 8 hipótesis ni ninguno de los umbrales de §4 (`H_A1a`–`H_A4a`) fue modificado, y las 16 semillas frescas de `CONFIRM` (`0xD100..=0xD10F`) permanecen sin ejecutar hasta el congelamiento `FROZEN.json`.

---

## 4. Las 4 Suites Experimentales (`A1`–`A4`) y los 8 Criterios Preregistrados

- **`DEV`:** 16 semillas `0xC100..=0xC10F` · **Freeze pre-CONFIRM:** `docs/adaptive_spectral_collapse_v1/FROZEN.json` · **`CONFIRM`:** 16 semillas frescas `0xD100..=0xD10F` · **Replay bit-exacto:** `0xD100` · **Bootstrap pareado:** 10,000 remuestreos (`seed = 12345`), `IC 95%`.

| ID | Suite | Hipótesis / Criterio Preregistrado | Regla de Aceptación (`CONFIRM`, $N=16$) |
|---|---|---|---|
| **`H_A1a`** | A1 | Autoorganización sin primos ni Riemann: `adaptive_spectral` abduce pares nunca vistos desde $\mathbb{R}^{16}$ venciendo a `poisson_fixed` y `fourier_fixed` | `CI95_lo(acc_adapt) > 0.80`, `CI95_lo(acc_adapt - acc_poisson) > 0.05` y `CI95_lo(acc_adapt - acc_fourier) > 0.05` |
| **`H_A1b`** | A1 | Independencia espectral: `adaptive_spectral` iguala o supera a `riemann_fixed`, `dyson_gue_fixed` y `qmc_halton_fixed` sin recibir $\zeta(s)$ | `CI95_lo(acc_adapt - acc_riemann) > -0.02`, `CI95_lo(acc_adapt - acc_dyson) > -0.02` y `CI95_lo(acc_adapt - acc_qmc) > -0.02` |
| **`H_A1c`** | A1 | Ventaja de colapso conjunto sobre MLP convencional en abducción de pares nunca vistos (`166` pares OOD) | `CI95_lo(acc_adapt - acc_mlp) > 0.05` |
| **`H_A2a`** | A2 | Generalización predictiva no lineal "del otro lado" sobre los 166 pares nunca vistos frente a `mlp_conventional` | `CI95_hi(rmse_adapt - rmse_mlp) < -0.10` y `CI95_lo(cls_adapt - cls_mlp) > 0.05` |
| **`H_A2b`** | A2 | Generalización contrafactual $do(c_a \to c_{\text{new}})$ hacia pares nunca vistos frente a `mlp_conventional` y `fourier_fixed` | `CI95_lo(cf_adapt) > 0.75`, `CI95_lo(cf_adapt - cf_mlp) > 0.10` y `CI95_lo(cf_adapt - cf_fourier) > 0.05` |
| **`H_A3a`** | A3 | Transferencia analógica ($A:B::C:?$, 100 analogías hacia destinos en `UNSEEN_PAIRS`) frente a `mlp_conventional` y `poisson_fixed` | `CI95_lo(an_adapt) > 0.75`, `CI95_lo(an_adapt - an_mlp) > 0.10` y `CI95_lo(an_adapt - an_poisson) > 0.05` |
| **`H_A3b`** | A3 | **Autocrítica física honesta:** degradación real bajo ruido sensorial severo ($\sigma_x = 0.100$ vs. $0.000$) | `CI95_hi(acc_noise_0100 - acc_noise_0000) < -0.20` |
| **`H_A4a`** | A4 | **Aprendizaje continuo de 4 conceptos nuevos (`K: 24→28`, 4 pares/concepto):** generalización zero-shot a pares nunca vistos `(nuevo, viejo)` + `(nuevo, nuevo)` sin olvido catastrófico frente a `mlp_conventional` | `CI95_lo(novel_acc_adapt - novel_acc_mlp) > 0.15`, `CI95_hi(novel_rmse_adapt - novel_rmse_mlp) < -0.10` y `CI95_lo(old_ret_adapt) > 0.75` |

Además, se miden y reportan de forma descriptiva (sin umbral forzado): la **tasa libre de colisiones** sobre los 276 pares, la **razón de espaciado espectral emergente** $\langle r \rangle$ de las frecuencias aprendidas por `adaptive_spectral` (antes vs. después del aprendizaje), el **número exacto de parámetros** y el **tiempo/FLOPs de entrenamiento** por brazo.

---

## 5. Pruebas Unitarias y Aislamiento

- Exactitud de ceros de Riemann (`< 1e-9`), paridad de partición estrictamente disjunta (`TRAIN_PAIRS ∩ UNSEEN_PAIRS = ∅`), verificación de gradientes analíticos vs. diferencias finitas (`< 1e-5`) tanto en el campo espectral como en el MLP, ausencia de `NaN`/`Inf`, verificación explícita de que `adaptive_spectral` no invoca `compute_riemann_zeros` ni primos, y determinismo bit a bit.
- Aislamiento estricto en `adaptive_spectral_collapse/` y `docs/adaptive_spectral_collapse_v1/`. Checks de `main` y ÓRBITA (`Paso0`, `known_acc16`, `E33`, `aura-engine`, `orbita`): **NO EVALUADOS**.
