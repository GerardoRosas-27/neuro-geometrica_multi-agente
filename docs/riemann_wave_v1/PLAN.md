# Plan Preregistrado — Frecuencia de Ondas de Riemann como Medio de Inferencia (`exp/riemann-wave-inference`)

**Escrito antes de implementar cualquier código en `riemann_wave/`.**
- **Rama aislada:** `exp/riemann-wave-inference` (desde `main@120e1e8`, sin modificar ningún archivo existente de `main`).
- **Crate autónomo:** `riemann_wave/` (Rust puro con `[workspace]` propio y **0 dependencias externas**; orquestación y bootstrap estadístico en Python con biblioteca estándar únicamente).
- **Fuente archivada:** [`PROPUESTA_GEMINI.md`](PROPUESTA_GEMINI.md) (`https://share.gemini.google/TcEnIP9sYZmT`).
- **Alcance honesto:** Este experimento evalúa si los espectros con repulsión de niveles tipo Riemann/GUE (ceros reales de $\zeta(\tfrac{1}{2}+i\gamma)$ y gas de Dyson $\beta = 2$) actúan como un medio espectral de inferencia estable y bien condicionado. **No** afirma consciencia, cognición general ni superioridad mística de los ceros de Riemann sobre otros procesos puntuales determinantes (DPP) o bases ortogonales en dominios estrictamente periódicos. Un resultado mixto o negativo en controles fuertes (`fourier_uniform`, `golden_jitter`, `qmc_rff`) es explícitamente aceptable y se reportará sin maquillaje.

---

## 1. Arquitectura y Diseño de las 5 Suites (`R1`–`R5`)

### Suite R1 — Verificación Espectral & Física de Ondas de Riemann
1. **Cálculo exacto de ceros no triviales $\gamma_1, \dots, \gamma_{128}$ de $\zeta(\tfrac{1}{2} + i\gamma)$:**
   - Evaluación de $\zeta(\tfrac{1}{2} + it)$ mediante fórmula de Euler–Maclaurin con $N(t) = \max(32, \lceil 1.25 t \rceil)$ y 8 términos de Bernoulli ($B_2 \dots B_{16}$), multiplicada por $e^{i\theta(t)}$ (función $Z(t)$ de Hardy real usando $\ln\Gamma$ por Stirling desplazado) y localización de raíces por barrido + método de Brent a tolerancia $10^{-11}$.
   - Verificado en pruebas unitarias contra los primeros 10 ceros de Odlyzko con error absoluto $< 10^{-9}$.
2. **Interferencia de ondas de primos (fórmula explícita de Riemann–Mangoldt con ventana de Fejér):**
   - Evaluada en $x \in [1.5, 60.0]$ ($M = 128$ ceros):
     $$W(x) = -\frac{2}{\sqrt{x}} \sum_{n=1}^{M} \left(1 - \frac{n-1}{M}\right) \cos(\gamma_n \ln x)$$
   - Separación estadística de enteros potencias de primos ($p^k \in [2, 60]$) frente a compuestos no potencias de primos ($6, 10, 12, 14, 15, \dots$): **ROC-AUC** y **SNR** $= (\mu_{p^k} - \mu_{\text{comp}}) / \sigma_{\text{comp}}$.
   - **Control negativo pareado:** Reemplazar $\gamma_n$ por $M = 128$ frecuencias aleatorias uniformes (Poisson) en $[\gamma_1, \gamma_{128}]$ por semilla.
3. **Emergencia del patrón Riemann/GUE desde Gas de Dyson ($\beta = 2$):**
   - Para cada semilla, $M = 60$ frecuencias parten de distribución uniforme desordenada en $[-0.85 R, 0.85 R]$ ($R = 2\sqrt{M}$) y evolucionan $4000$ pasos con $\Delta t = 0.002$, $\varepsilon = 0.03$, drift $-\frac{1}{2}w_i + \sum_{j \neq i} \frac{w_i - w_j}{(w_i - w_j)^2 + \varepsilon^2}$ y ruido térmico $\sqrt{\Delta t}\,\xi_i$ ($D = 1/2 \Rightarrow \beta = 2$).
   - Muestreo de espaciamientos normalizados en el núcleo central (`[5..55]`) en la segunda mitad del equilibrio (`paso >= 2000`, cada 50 pasos).
   - Referencia adicional de **GUE matricial**: 20 matrices Hermitianas complejas aleatorias $D = 48$ diagonalizadas en Rust puro (Jacobi Hermitiano), unfolded mediante CDF exacta del semicírculo de Wigner.
   - Referencia real: los 128 ceros de Riemann unfolded mediante $N(T) = \frac{T}{2\pi}\ln\left(\frac{T}{2\pi e}\right)$.
   - Métricas: distancia de Kolmogorov–Smirnov $D_{\text{KS}}$ contra CDF teórica de Wigner GUE ($F_{\text{GUE}}(s) = \int_0^s \frac{32}{\pi^2} u^2 e^{-\frac{4}{\pi} u^2} du$) vs. CDF de Poisson ($1 - e^{-s}$), y tasa de repulsión en el origen $P(s < 0.2)$.

### Suite R2 — Inferencia por Interferencia de Ondas bajo Perturbación de Fase (Experimento Principal + Controles Críticos)
- **Geometría y tarea (fiel a la propuesta de Gemini):**
  - $M = 60$ modos ($P = 2M + 1 = 121$ pesos), dominio $x \in [1.5, 35.0]$, coordenada logarítmica $u(x) = 2\pi \frac{\ln(x/x_{\min})}{\ln(x_{\max}/x_{\min})} \in [0, 2\pi)$.
  - Por semilla: 5 conceptos objetivo distintos sorteados aleatoriamente en $u_c \in [0.55, 2\pi - 0.55]$ con separación mínima $\Delta u \ge 0.50$, ancho $\sigma_u = 0.055$.
  - Entrenamiento: $N_{\text{train}} = 260$ puntos uniformes en $[0, 2\pi)$, observaciones con ruido $\eta_i \sim \mathcal{N}(0, 0.04^2)$.
  - Evaluación: $N_{\text{eval}} = 600$ puntos uniformes en $[0, 2\pi)$ bajo perturbación interna de fase $\phi_k \sim \mathcal{N}(0, \sigma_\phi^2)$ idéntica para todos los modelos en cada semilla, evaluada en $\sigma_\phi = 0.015$ rad (nominal Gemini) y estrés $\sigma_\phi = 0.05$ rad.
- **6 Familias Espectrales (exactamente $M = 60$ frecuencias en $[1, M]$ cada una):**
  1. `dyson_gue`: consolidación Langevin $\beta = 2$ (semilla propia) + unfolding semicírculo.
  2. `riemann_zeros`: primeros $M = 60$ ceros reales $\gamma_n$ de $\zeta(\tfrac{1}{2}+i\gamma)$ unfolded a $[1, M]$ (más una traslación de fase/índice determinista fija para mantener espectro real exacto).
  3. `poisson`: $M = 60$ frecuencias aleatorias uniformes ordenadas en $[1, M]$ (control de Gemini sin repulsión).
  4. `goe_beta1`: gas de Dyson $\beta = 1$ ($D = 1$, ruido $\sqrt{2\Delta t}\,\xi_i$, repulsión lineal $P(s) \propto s$).
  5. `fourier_uniform` (**control crítico cristalino**): frecuencias enteras exactas $f_k = k$ ($k = 1, \dots, M$), límite $\beta \to \infty$.
  6. `golden_jitter` (**control crítico determinista inconmensurable**): $f_k = 1 + (M-1)\frac{k - 0.5}{M} + 0.15\left(\operatorname{frac}(k\varphi) - 0.5\right)$ con $\varphi = \frac{1+\sqrt{5}}{2}$.
- **2 Regímenes de ajuste mediante SVD exacta (QR Householder + Jacobi one-sided):**
  - **OLS ($\lambda = 0$)** (exactamente `lstsq` de Gemini, corte singular $\tau = 10^{-12}\sigma_{\max}$).
  - **Ridge ($\lambda = 10^{-2}$)** (prueba anti-artefacto para separar mal condicionamiento extremo de cobertura espectral real).
- **Métricas:** $\log_{10}\kappa_2(\Phi)$, norma de pesos $\|\mathbf{w}\|_2$, rango efectivo $\text{df}(\lambda)$, RMSE de inferencia ($\sigma_\phi = 0.015$ y $\sigma_\phi = 0.05$).

### Suite R3 — Anti-Aliasing Fuera de Periodo & Muestreo Irregular
1. **R3a — Supresión de Ecos Fantasma en $u \in [2\pi, 4\pi)$:**
   - Toma los modelos ya ajustados en R2 (Ridge $\lambda = 10^{-2}$, entrenados solo en $[0, 2\pi)$) y los evalúa en 600 puntos de la región virgen $u \in [2\pi + 0.25, 4\pi - 0.25]$ (con $\sigma_\phi = 0.015$), donde el objetivo verdadero es $y_{\text{true}}(u) = 0$.
   - Mide `ghost_peak` $= \max_u |\hat{y}(u)|$ y `ghost_rms` $= \sqrt{\frac{1}{N}\sum \hat{y}(u)^2}$.
   - Mide el compromiso combinado `joint_score = rmse_in_domain + ghost_rms` (equilibrio entre fidelidad en memoria y ausencia de alucinaciones periódicas).
2. **R3b — Muestreo Irregular / Asíncrono:**
   - $N_{\text{train}} = 180$ tiempos de muestreo aleatorios $u_i \sim \text{Uniform}(0, 2\pi)$ por semilla (mismos conceptos y ruido $\sigma_\eta = 0.04$, $\sigma_\phi = 0.015$).
   - Evalúa $\log_{10}\kappa_2(\Phi)$ y RMSE (OLS y Ridge $\lambda = 10^{-2}$) para las 6 familias.

### Suite R4 — Memoria Continua en Streaming & Ablación Causal de Repulsión
- **Streaming de $K = 8$ conceptos secuenciales:**
  - En cada semilla se generan 8 conceptos distintos en $[0.35, 2\pi - 0.35]$ separados por $\Delta u \ge 0.45$, presentados **uno a uno** en 8 episodios consecutivos (sin replay de episodios anteriores).
  - Cada episodio $e$ aporta 32 muestras locales alrededor de $u_{c_e}$ (ventana $\pm 0.22$) y 16 muestras de contexto aleatorio en $[0, 2\pi)$ con ruido $\sigma_\eta = 0.03$.
  - **Actualización online de pesos (RLS — Recursive Least Squares):** con covarianza inicial $P_0 = 10.0\,I$, sin factor de olvido ($\gamma = 1.0$) para preservar conceptos pasados, evaluado al final bajo perturbación de fase $\sigma_\phi = 0.02$ rad sobre el objetivo acumulado de los 8 conceptos.
  - **Brazos evaluados ($M = 60$):**
    - Espectros fijos: `dyson_gue`, `riemann_zeros`, `poisson`, `fourier_uniform`.
    - **Co-adaptación espectral online (ablación causal directa):**
      - `adaptive_dyson`: parte de frecuencias iniciales desordenadas idénticas a `poisson`, y tras cada episodio realiza 25 pasos de gradiente sobre frecuencias (tasa $\eta_\omega = 0.015$, clip $|\Delta \omega| \le 0.1$) + **repulsión de Dyson activa** ($\eta_{\text{rep}} = 0.008$, $\varepsilon = 0.05$, confinamiento a $[1, M]$) antes de re-proyectar los pesos en RLS.
      - `adaptive_no_repulsion`: **idéntico bit a bit** a `adaptive_dyson` salvo $\eta_{\text{rep}} = 0.0$ (sin repulsión).
  - **Métricas:** RMSE global final sobre los 8 conceptos (`stream_rmse`), RMSE sobre el primer concepto (`first_concept_rmse`), $\log_{10}\kappa_2(\Phi)$, norma $\|\mathbf{w}\|_2$ y espaciamiento mínimo final $\min_{i\neq j}|f_i - f_j|$.

### Suite R5 — Medio de Inferencia de Kernel (Wave-RFF / DPP-GUE) sobre Dinámicas No Lineales 1D y 2D
- **R5a — Aproximación de Kernel RBF 1D ($k(\Delta x) = \exp(-\frac{\Delta x^2}{2\ell^2})$, $\ell = 0.45$, $M = 48$ frecuencias):**
  - Transformación a densidad espectral Gaussiana $\mathcal{N}(0, \ell^{-2})$ vía cuantiles normales $\Phi_{\mathcal{N}}^{-1}(u_k)$ (aproximación Acklam de precisión $< 10^{-9}$):
    1. `poisson_rff`: $u_k \sim \text{Uniform}(0, 1)$ i.i.d. (Rahimi–Recht clásico).
    2. `dyson_rff`: $M = 48$ modos consolidados por Gas de Dyson $\beta = 2$, unfolded por CDF del semicírculo a $u_k \in (0, 1)$ (DPP-GUE real).
    3. `riemann_rff`: primeros $M = 48$ ceros reales $\gamma_n$ de Riemann unfolded a $u_k \in (0, 1)$ (con fase aleatoria compartida $b_k \sim \text{Uniform}(0, 2\pi)$ por semilla).
    4. `qmc_rff` (**control crítico más fuerte**): cuantiles deterministas equidistantes $u_k = \frac{k - 0.5}{M}$ (Cuasi-Monte Carlo 1D).
  - Métrica: error Frobenius relativo $\|\hat{K} - K_{\text{exact}}\|_F / \|K_{\text{exact}}\|_F$ en una malla de 80 puntos en $[-2.0, 2.0]$.
- **R5b — Inferencia Predictiva en Dinámica No Lineal 1D (Oscilador No Lineal Multi-Escala + Rollout 8 pasos):**
  - Sistema dinámico determinista suave con coeficientes por semilla:
    $$x_{t+1} = 0.72 x_t + a_1 \sin(2.4 x_t) + a_2 \cos(3.8 x_t) - 0.14 x_t^3$$
    con $a_1 \in [0.22, 0.34]$, $a_2 \in [-0.16, 0.16]$.
  - Entrenamiento: 120 pares $(x_t, x_{t+1} + \eta)$ con $x_t \sim \text{Uniform}(-1.6, 1.6)$, $\sigma_\eta = 0.015$, regresión Ridge ($\lambda = 10^{-3}$) sobre características de onda 1D ($M = 48$).
  - Evaluación held-out (flujo RNG independiente): 120 puntos 1-paso y 32 trayectorias en rollout autorregresivo de 8 pasos (métrica: `rmse_1step` y `rmse_rollout8`).
- **R5c — Extensión 2D Isótropa (Gas de Coulomb/Ginibre 2D $\beta = 2$ vs. RFF 2D i.i.d. vs. QMC Halton 2D):**
  - Dinámica no lineal 2D (vórtice no lineal acoplado tipo Duffing–rotación en $\mathbf{z} \in [-1.4, 1.4]^2$):
    $$z_{t+1} = R(\theta_0 + 0.35 \|\mathbf{z}_t\|^2)\,\left(0.84 \mathbf{z}_t + \mathbf{g}(\mathbf{z}_t)\right)$$
    donde $g_1(x,y) = c_1 \sin(2.0 y) - 0.08 x^3$, $g_2(x,y) = c_2 \sin(2.0 x) - 0.08 y^3$.
  - $M = 64$ frecuencias en $\mathbb{R}^2$, 220 muestras de entrenamiento con ruido $\sigma_\eta = 0.015$, Ridge $\lambda = 10^{-3}$, evaluación held-out en 160 estados (1-paso) y 24 trayectorias (rollout 8 pasos):
    1. `poisson_2d_rff`: $\boldsymbol{\omega}_k \sim \mathcal{N}(0, \ell^{-2} I_2)$ i.i.d.
    2. `ginibre_2d_rff`: **gas logarítmico 2D de Coulomb/Ginibre ($\beta = 2$)** (repulsión $2\text{D}$ $\sum_{j \neq i} \frac{\boldsymbol{\omega}_i - \boldsymbol{\omega}_j}{\|\boldsymbol{\omega}_i - \boldsymbol{\omega}_j\|^2 + \varepsilon^2}$ con confinamiento armónico, que reproduce exactamente el ensamble DPP-Ginibre 2D!) normalizado al perfil radial Gaussiano de escala $\ell^{-1}$.
    3. `qmc_2d_rff` (**control crítico**): secuencia Halton(2, 3) 2D transformada vía Box–Muller.

---

## 2. Splits de Semillas, Congelamiento y Protocolo Estadístico

- **`DEV` (calibración/verificación):** 16 semillas `0xA700..=0xA70F` (`42752..=42767`).
- **Congelamiento obligatorio pre-CONFIRM:** `docs/riemann_wave_v1/FROZEN.json` registra SHA-256 de todos los `.rs` de `riemann_wave/src/` y el commit padre antes de ejecutar `CONFIRM`.
- **`CONFIRM` (ejecución única congelada):** 16 semillas frescas `0xB700..=0xB70F` (`46848..=46863`). Prohibido reajustar hiperparámetros tras observar `CONFIRM`.
- **Inferencia estadística:** Bootstrap pareado sobre las 16 diferencias de semilla (`10,000` remuestreos deterministas, PRNG Python semilla `12345`, cuantiles percentiles linealmente interpolados al 95% `[CI95_lo, CI95_hi]`).
- **Replay determinista bit a bit:** Re-ejecución independiente de `0xB700` verificando igualdad exacta de todos los campos no temporales (`docs/riemann_wave_v1/replay.json`).
- **Reglas de parada y presupuesto:** Máximo 4 workers, sin NaN/Inf en ninguna métrica ni parámetro, tiempo total < 20 min, pruebas unitarias debug y release 100% verdes.

---

## 3. Los 12 Criterios Preregistrados (`H_R1a`–`H_R5c`)

Todos los criterios se juzgan estrictamente en **`CONFIRM` ($N = 16$ semillas)** por `CI95%` bootstrap pareado:

| ID | Suite | Hipótesis / Criterio Preregistrado | Regla de Aceptación (`CONFIRM`) |
|---|---|---|---|
| **`H_R1a`** | R1 | Interferencia de ceros reales de Riemann reconstruye primos/potencias vs. control Poisson | `prime_auc_riemann > 0.95` y `CI95_lo(snr_riemann - snr_poisson) > 2.0` |
| **`H_R1b`** | R1 | Gas de Dyson ($\beta=2$) reproduce espontáneamente estadística GUE/Riemann y repulsión en $s=0$ | `CI95_hi(ks_dyson_gue - ks_dyson_poisson) < -0.15` y `CI95_hi(p_small_dyson - p_small_poisson) < -0.10` |
| **`H_R2a`** | R2 | Replicación Gemini (OLS $\lambda=0$, $\sigma_\phi=0.015$): Dyson y Riemann evitan colapso vs. Poisson | `CI95_hi(log10_cond_dyson - log10_cond_poisson) < -1.0`, `CI95_hi(rmse_dyson_ols - rmse_poisson_ols) < 0.0` y `CI95_hi(rmse_riemann_ols - rmse_poisson_ols) < 0.0` |
| **`H_R2b`** | R2 | Prueba anti-artefacto (Ridge $\lambda=10^{-2}$, $\sigma_\phi=0.015$): Dyson sigue venciendo a Poisson tras regularizar | `CI95_hi(rmse_dyson_ridge - rmse_poisson_ridge) < 0.0` y `CI95_hi(rmse_riemann_ridge - rmse_poisson_ridge) < 0.0` |
| **`H_R2c`** | R2 | **Autocrítica honesta vs. `fourier_uniform` en dominio periódico** (exploratorio/verificación de límite cristalino) | Reportar `CI95(rmse_dyson_ridge - rmse_fourier_ridge)` y `CI95(log10_cond_dyson - log10_cond_fourier)`; criterio de honestidad: verificar explícitamente si `fourier_uniform` logra mejor condicionamiento (`CI95_lo > 0.0`) sin ocultarlo |
| **`H_R3a`** | R3 | Anti-aliasing fuera de periodo ($u \in [2\pi, 4\pi)$): Dyson y Riemann suprimen ecos fantasma vs. `fourier_uniform` | `CI95_hi(ghost_peak_dyson - ghost_peak_fourier) < -0.30` y `CI95_hi(ghost_peak_riemann - ghost_peak_fourier) < -0.30` |
| **`H_R3b`** | R3 | Muestreo irregular asíncrono ($N=180$): Dyson supera a Poisson en estabilidad y RMSE (Ridge) | `CI95_hi(log10_cond_dyson_irreg - log10_cond_poisson_irreg) < -0.5` y `CI95_hi(rmse_dyson_irreg_ridge - rmse_poisson_irreg_ridge) < 0.0` |
| **`H_R4a`** | R4 | Streaming online RLS (8 conceptos secuenciales, 0 replay): Dyson supera a Poisson | `CI95_hi(stream_rmse_dyson - stream_rmse_poisson) < 0.0` |
| **`H_R4b`** | R4 | Ablación causal de repulsión en co-adaptación espectral online (`adaptive_dyson` vs. `adaptive_no_repulsion`) | `CI95_hi(stream_rmse_adapt_dyson - stream_rmse_adapt_norep) < 0.0` y `CI95_lo(min_gap_adapt_dyson - min_gap_adapt_norep) > 0.0` |
| **`H_R5a`** | R5 | Inferencia de kernel 1D (DPP-GUE vs. RFF i.i.d.): Dyson-RFF y Riemann-RFF reducen error Frobenius vs. Poisson-RFF | `CI95_hi(kerr_dyson - kerr_poisson) < 0.0` y `CI95_hi(kerr_riemann - kerr_poisson) < 0.0` |
| **`H_R5b`** | R5 | Predicción de dinámica no lineal 1D (rollout 8 pasos): Dyson-RFF supera a Poisson-RFF | `CI95_hi(dyn1d_roll8_dyson - dyn1d_roll8_poisson) < 0.0` |
| **`H_R5c`** | R5 | Extensión 2D (Ginibre 2D $\beta=2$ vs. Poisson 2D) + reporte honesto contra QMC 1D/2D | `CI95_hi(dyn2d_roll8_ginibre - dyn2d_roll8_poisson) < 0.0` (y reporte explícito sin filtro frente a `qmc_rff` y `qmc_2d_rff`) |

---

## 4. Pruebas Unitarias Obligatorias y Controles de Integridad

- Exactitud de los primeros 10 ceros no triviales de $\zeta(\tfrac{1}{2}+i\gamma)$ vs. tabla de Odlyzko ($< 10^{-9}$), y $|\operatorname{Im}(e^{i\theta(t)}\zeta(\tfrac{1}{2}+it))| < 10^{-11}$.
- Diagonalización Hermitiana Jacobi para GUE: residual $\|H V - V\Lambda\|_F < 10^{-10}$, unitariedad $\|V^\dagger V - I\|_F < 10^{-10}$.
- SVD exacta (QR + Jacobi one-sided): reconstrucción $\|A - U\Sigma V^T\|_F < 10^{-11}$, condición exacta en matriz diagonal/Hilbert conocida, coincidencia OLS/Ridge con solución analítica.
- Cuantiles normales $\Phi_{\mathcal{N}}^{-1}(u)$: error de inversión $|\Phi(\Phi^{-1}(u)) - u| < 10^{-8}$.
- Gas de Dyson $\beta = 2$ y Ginibre 2D: finitud (0 NaN/Inf), determinismo bit a bit por semilla, preservación de orden y separación mínima.
- Ningún archivo de `main` ni de la aplicación existente (`src/`, `orbita/`, `aura-engine/`, `Cargo.toml`, `README.md`) se modifica; los checks de integración de la app principal se declaran **NO EVALUADOS** (experimento aislado).
