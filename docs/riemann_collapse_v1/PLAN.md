# Plan Preregistrado — Colapso de Onda Pasada y Onda Futura hacia la Secuencia de Primos de Riemann (`exp/riemann-wave-collapse`)

**Escrito antes de implementar cualquier código en `riemann_collapse/`.**
- **Rama aislada:** `exp/riemann-wave-collapse` (desde `main@120e1e8`, sin modificar ningún archivo de `main` ni de otras ramas).
- **Crate autónomo:** `riemann_collapse/` (Rust puro con `[workspace]` propio y **0 dependencias externas**; orquestación, bootstrap pareado y graficación SVG en Python con biblioteca estándar únicamente).
- **Objetivo del usuario:** Hacer colapsar dos ondas —una onda pasada (retardada) y una onda futura (avanzada)— de manera que de su interferencia surja la secuencia de números primos de Riemann, y **buscar automáticamente la configuración física exacta** que produce el surgimiento de este patrón.
- **Alcance honesto:** Este experimento evalúa la física matemática de colisión de ondas (superposición lineal dispersiva, colapso transaccional de semifrecuencias tipo Cramer/Aharonov, interferencia Born-rule con portadora holográfica, control bilineal ingenuo), la búsqueda inversa de la configuración física exacta de colapso, su paisaje de tolerancia/estabilidad y el **descubrimiento ciego de los propios ceros $\gamma_n$ de Riemann** únicamente desde la criba elemental de primos (sin evaluar $\zeta(s)$). No afirma retrocausalidad física literal en la naturaleza ni resuelve criptografía; cualquier límite de resolución (límite de Rayleigh, fracaso de operadores bilineales sin emparejamiento modal) se reporta sin maquillaje.

---

## 1. Diseño Físico-Matemático de las 4 Suites (`C1`–`C4`)

### Suite C1 — Comparación de 4 Operadores de Colapso Pasado–Futuro y Emergencia Estricta
Con los primeros $M = 128$ ceros exactos $\gamma_1, \dots, \gamma_{128}$ de $\zeta(\tfrac{1}{2} + i\gamma)$ (calculados en Rust puro vía Euler–Maclaurin + Hardy $Z(t)$ + Brent a $< 10^{-9}$) sobre $x \in [2, 100]$, evaluamos cinco configuraciones/operadores de colapso con dispersión $\omega_n = c\gamma_n + \beta\gamma_n^2$ ($c = 1.0$, $\beta = 0.008$) y ventana de Fejér $w_n = 1 - (n-1)/M$:

1. **`transactional_half_freq` (Colapso Transaccional Cramer/Aharonov de Semifrecuencias — Emergencia Estricta):**
   - Onda pasada (avanza en $\tau$ con semifrecuencias $+k_+ \gamma_n$, $k_+ = 0.5$):
     $$\psi_n^{\text{past}}(x, \tau) = \sqrt{w_n}\, x^{\sigma_+} e^{i\left(k_+ \gamma_n \ln x - \frac{1}{2}\omega_n \tau + \phi_+\right)}$$
   - Onda futura (retrocede en $\tau$ con semifrecuencias conjugadas $-k_- \gamma_n$, $k_- = 0.5$):
     $$\psi_n^{\text{future}}(x, \tau) = \sqrt{w_n}\, x^{\sigma_-} e^{i\left(-k_- \gamma_n \ln x + \frac{1}{2}\omega_n \tau + \phi_-\right)}$$
   - **Emergencia estricta incluso en $\tau = 0$:** Cada onda por separado porta solo semifrecuencias $\pm\gamma_n/2$ (picos solo en cuadrados $p^{2k} \in \{4, 9, 16, 25, 49, 64, 81\}$, **cero picos en los números primos** $2, 3, 5, 7, 11, 13\dots$). Solo el solapamiento modal pasado–futuro:
     $$\mathcal{I}_{\text{trans}}(x, \tau) = 2\operatorname{Re}\sum_{n=1}^{M} \psi_n^{\text{past}}(x, \tau)\,\psi_n^{\text{future}}(x, \tau)^* = 2\, x^{\sigma_+ + \sigma_-} \sum_{n=1}^{M} w_n \cos\!\left((k_+ + k_-)\gamma_n \ln x - \omega_n \tau + (\phi_+ - \phi_-)\right)$$
     suma ambas semifrecuencias ($k_+ + k_- = 1$) y hace emerger la secuencia exacta de primos en el instante de colapso $\tau = 0$ con $\Delta\phi = \phi_+ - \phi_- = \pi$.
2. **`linear_counterprop` (Superposición Lineal Dispersiva D'Alembert):**
   $$\mathcal{I}_{\text{lin}}(x, \tau) = \Psi_{\text{past}}(x, \tau) + \Psi_{\text{future}}(x, \tau) = -x^{\sigma_{\text{tot}}}\sum_{n=1}^{M} w_n \left[\cos(\gamma_n \ln x - \omega_n \tau + \phi_+) + \cos(\gamma_n \ln x + \omega_n \tau + \phi_-)\right]$$
   Fuera de $\tau = 0$ ambas ondas están completamente dispersadas por $\beta \gamma_n^2 \tau$; al colisionar en $\tau = 0$ se enfocan y colapsan en los primos.
3. **`holographic_born` (Interferencia Born-Rule con Portadora Futura Suave):**
   $$\mathcal{I}_{\text{born}}(x, \tau) = |\Psi_{\text{past}} + \Psi_{\text{future}}|^2 - |\Psi_{\text{past}}|^2 - |\Psi_{\text{future}}|^2 = 2\operatorname{Re}[\Psi_{\text{past}}(x, \tau)\,\Psi_{\text{future}}^*(x, \tau)]$$
   donde $\Psi_{\text{past}}$ lleva el espectro de ceros retardado y $\Psi_{\text{future}} = -x^{\sigma_-} e^{i\phi_-}$ actúa como onda portadora avanzada suave (polo $s = 1$).
4. **`naive_bilinear_ctrl` (Control Negativo Físico — Producto Bilineal Global Sin Emparejamiento Modal):**
   $$\mathcal{I}_{\text{naive}}(x, \tau) = 2\operatorname{Re}\!\left[\left(\sum_{n=1}^M \psi_n^{\text{past}}\right)\left(\sum_{m=1}^M \psi_m^{\text{future}}\right)^*\right]$$
   Los términos cruzados $n \neq m$ generan frecuencias mezcla $\frac{\gamma_n + \gamma_m}{2}$ que destruyen la estructura prima.
5. **`poisson_freq_ctrl` (Control Negativo Espectral por Semilla):**
   Idéntico operador `transactional_half_freq` en $\tau = 0$, pero con $M = 128$ frecuencias aleatorias uniformes (Poisson) en $[\gamma_1, \gamma_{128}]$.

**Métricas C1 (en $x \in [2, 100]$):** `prime_only_auc` (primos $p$ vs. todos los compuestos), `prime_power_auc` ($p^k$ vs. compuestos no-$p^k$), `prime_snr`, `past_alone_prime_auc` (onda pasada sola en $\tau = 0$), `future_alone_prime_auc` (onda futura sola en $\tau = 0$), `off_time_auc` (colapso en $\tau = 0.25$), y `emergence_gain = prime_only_auc - max(past_alone_prime_auc, future_alone_prime_auc)`.

---

### Suite C2 — Búsqueda Automática de la Configuración Física Exacta ($\boldsymbol{\theta}^* \in \mathbb{R}^7$)
Para el operador de emergencia estricta `transactional_half_freq` ($M = 128$), parametrizamos la configuración física mediante 7 parámetros continuos:
$$\boldsymbol{\theta} = \left(\tau,\; k_{\text{sum}},\; \Delta\phi,\; \sigma_{\text{tot}},\; \beta_{\text{disp}},\; p_{\text{win}},\; v_{\text{asym}}\right)$$
con rangos de búsqueda:
- $\tau \in [-1.0, +1.0]$ (instante de encuentro temporal; óptimo exacto teórico: $0.0$),
- $k_{\text{sum}} = k_+ + k_- \in [0.70, 1.30]$ (suma de escalas de semifrecuencia; óptimo exacto teórico: $1.0$),
- $\Delta\phi = \phi_+ - \phi_- \in [-\pi, +\pi]$ (fase relativa de colapso; óptimo exacto teórico: $\pi$),
- $\sigma_{\text{tot}} = \sigma_+ + \sigma_- \in [-0.75, +0.75]$ (exponente de escala $x^{\sigma_{\text{tot}}}$: $+0.5$ compensa exactamente el factor $p^{-1/2}$ de la fórmula explícita en espacio $\ln x$; $-0.5$ corresponde a la derivada $\psi'(x)$ en espacio $x$),
- $\beta_{\text{disp}} \in [-0.03, +0.03]$ (desajuste de dispersión cuadrática residual entre pasada y futura; óptimo exacto teórico: $0.0$),
- $p_{\text{win}} \in [0.0, 3.0]$ (exponente de ventana espectral $w_n = (1 - (n-1)/M)^{p_{\text{win}}}$),
- $v_{\text{asym}} \in [-0.40, +0.40]$ (asimetría de fase espacial lineal acoplada a $\tau$, desplazando el foco si $\tau \neq 0$).

**Función objetivo de búsqueda (estrictamente confinada a la ventana de entrenamiento $x \in [2, 60]$):**
$$J_{\text{train}}(\boldsymbol{\theta}) = \text{SNR}_{[2, 60]}(\boldsymbol{\theta}) + 20.0 \cdot \left(\text{AUC}_{[2, 60]}(\boldsymbol{\theta}) - 0.5\right)$$
donde el SNR y la AUC se evalúan únicamente sobre enteros $n \in [2, 60]$ (potencias de primos vs. compuestos no-$p^k$).

**Protocolo por semilla (16 DEV + 16 CONFIRM):**
1. **`init_random`**: una configuración aleatoria inicial en la caja de búsqueda (base sin calibrar).
2. **`random_search_ctrl`**: búsqueda aleatoria uniforme con exactamente **`3,000`** evaluaciones de $J_{\text{train}}$.
3. **`collapse_optimizer`** (Rust puro, 0 deps): **Differential Evolution** (`pop = 30`, `80` generaciones = `2,400` evaluaciones) seguido de **refinamiento continuo de coordenadas adaptativo (Pattern/Powell polish)** de `600` evaluaciones = exactamente **`3,000`** evaluaciones totales (presupuesto 100% equiparado a `random_search_ctrl`).
4. **Control de exponente fijo (`fixed_sigma_neg_half`)**: misma configuración óptima descubierta pero forzando $\sigma_{\text{tot}} = -0.5$ (fórmula clásica $-2/\sqrt{x}$ sin compensar $p^{-1/2}$) para verificar si el exponente descubierto automáticamente $\hat{\sigma}_{\text{tot}}^*$ mejora la visibilidad de primos grandes en $[61, 150]$.

**Evaluación de generalización pura (0 fuga de datos):**
Congelada $\hat{\boldsymbol{\theta}}^*$ descubierta solo en $[2, 60]$, evaluamos sobre los enteros nunca vistos **$x \in [61, 150]$** (que contienen 19 primos inéditos $61, 67, \dots, 149$ y 3 potencias de primos $64, 81, 121, 125, 128$): `heldout_prime_power_auc`, `heldout_prime_only_auc`, `heldout_snr`, y los errores absolutos de parámetro $|\hat{\tau}^*|$, $|\hat{k}_{\text{sum}}^* - 1|$, $|\text{wrap}(\hat{\Delta\phi}^* - \pi)|$, $|\hat{\beta}_{\text{disp}}^*|$.

---

### Suite C3 — Paisaje de Sensibilidad, Vida Media Temporal y Transición por Número de Modos $M$
Sobre la configuración exacta de colapso (`transactional_half_freq`), cuantificamos:
1. **Ventana temporal del colapso** $\tau \in [-0.30, +0.30]$ (con dispersión $c = 1.0, \beta = 0.008$): mide el ancho temporal **FWHM** donde `AUC > 0.85` en $[2, 100]$, confirmando que el patrón emerge bruscamente al cruzarse las ondas y se desvanece fuera de $\tau = 0$.
2. **Simetría de fase $\Delta\phi \in \{0, \pi/2, \pi, 3\pi/2\}$**: verifica inversión exacta (`AUC(π) > 0.98`, `AUC(0) < 0.02`, `AUC(±π/2) ≈ 0.50`).
3. **Agudeza de sintonía espectral $\delta k_{\text{sum}} \in \{0.0, 0.002, 0.005, 0.01, 0.02, 0.05\}$**: verifica el colapso del patrón bajo mínima desintonización ($|\delta k_{\text{sum}}| = 0.02 \Rightarrow \text{AUC} < 0.65$).
4. **Transición de fase por número de pares de ondas $M \in \{8, 16, 32, 64, 128, 256\}$** en $[2, 60]$ y $[61, 150]$ (comprobando el límite de resolución de Rayleigh).

---

### Suite C4 — Descubrimiento Ciego de los Ceros de Riemann desde el Colapso de Dos Ondas (0 Evaluaciones de $\zeta(s)$)
Probamos si un sistema de $K = 20$ pares de ondas pasado–futuro (frecuencias desconocidas $\omega_1, \dots, \omega_{20}$ inicializadas aleatoriamente en $[10, 82]$) puede **descubrir los propios ceros de Riemann $\gamma_1 = 14.1347 \dots \gamma_{20} = 77.1448$ sin evaluar jamás $\zeta(s)$**, interactuando únicamente con la criba elemental de primos en $x \in [2, 600]$ mediante la resonancia dual exacta de Guinand–Weil con ventana de Hann en espacio logarítmico ($u = \ln x$, $u_{\max} = \ln 600$):
$$R(\omega) = -\sum_{n=2}^{600} h(\ln n)\,\frac{\Lambda(n)}{\sqrt{n}}\,\cos(\omega \ln n) + \int_{1}^{600} h(\ln x)\,\frac{\cos(\omega \ln x)}{\sqrt{x}}\,dx, \qquad h(u) = \sin^2\!\left(\pi \frac{u}{u_{\max}}\right)$$
con gradiente analítico exacto $R'(\omega)$ y **ablación causal directa** (16 semillas `DEV` + 16 semillas `CONFIRM`):
1. **`blind_collapse_dyson`**: ascenso de gradiente sobre $R(\omega_k)$ + **repulsión espectral de Dyson** $\eta_{\text{rep}} \sum_{j \neq k} \frac{\omega_k - \omega_j}{(\omega_k - \omega_j)^2 + \varepsilon^2}$ (evita que múltiples modos colapsen sobre un mismo cero), seguido de un breve pulido puro sobre los picos encontrados.
2. **`blind_collapse_no_repulsion`**: **idéntica inicialización y gradiente**, pero con $\eta_{\text{rep}} = 0.0$ (ablación causal pura).
3. **`random_freq_ctrl`**: frecuencias iniciales sin optimizar.
4. **`exact_20_zeros`**: techo teórico con los primeros 20 ceros reales $\gamma_1, \dots, \gamma_{20}$.

**Evaluación triple (totalmente ciega respecto a $\zeta(s)$ durante la búsqueda):**
1. **`distinct_zeros_found`** (cuántos de los 20 primeros ceros exactos $\gamma_1 \dots \gamma_{20}$ quedan capturados dentro de $|\hat{\omega}_k - \gamma_j| < 0.35$) y **`mean_zero_err`** (error medio de emparejamiento 1-a-1 frente a $\gamma_1 \dots \gamma_{20}$).
2. **`auc_2_45`**: discriminación de potencias de primos en $x \in [2, 45]$ (rango resuelto por el límite de Rayleigh para $K = 20$ modos, $\gamma_{20} = 77.14$).
3. **`extrap_psi_corr`**: correlación de Pearson **fuera de muestra** entre la fluctuación real de Chebyshev $\Delta\psi(x) = \psi(x) - x$ y la predicción de colapso $\widehat{\Delta\psi}(x) = -2\sqrt{x}\sum_{k=1}^{20} \frac{\frac{1}{2}\cos(\hat{\omega}_k \ln x) + \hat{\omega}_k \sin(\hat{\omega}_k \ln x)}{\frac{1}{4} + \hat{\omega}_k^2}$ en la ventana nunca vista **$x \in [601, 1200]$**!

---

## 2. Splits, Congelamiento y Protocolo Estadístico

- **`DEV`:** 16 semillas `0xA800..=0xA80F`.
- **Freeze obligatorio pre-CONFIRM:** `docs/riemann_collapse_v1/FROZEN.json` (SHA-256 de todos los `.rs` y `Cargo.toml`) en commit previo a ejecutar `CONFIRM`.
- **`CONFIRM`:** 16 semillas frescas `0xB800..=0xB80F` con `--skip-build`.
- **Bootstrap pareado:** 10,000 remuestreos deterministas (`random.Random(12345)`), cuantiles percentiles al 95% `[CI95_lo, CI95_hi]`.
- **Replay bit-exacto:** Verificación en `0xB800` (`docs/riemann_collapse_v1/replay.json`) de todas las columnas no temporales.

---

## 3. Los 10 Criterios Preregistrados (`H_C1a`–`H_C4b`)

| ID | Suite | Hipótesis / Criterio Preregistrado | Regla de Aceptación (`CONFIRM`, $N=16$) |
|---|---|---|---|
| **`H_C1a`** | C1 | Emergencia estricta en `transactional_half_freq`: colapso revela primos que no existen en ninguna onda aislada | `prime_only_auc > 0.98`, `past_alone_prime_auc < 0.60`, `future_alone_prime_auc < 0.60`, `emergence_gain > 0.40` y `CI95_lo(snr_trans - snr_poisson) > 10.0` |
| **`H_C1b`** | C1 | Control negativo físico: producto bilineal global sin emparejamiento modal (`naive_bilinear_ctrl`) destruye el patrón | `naive_prime_power_auc < 0.65` y `CI95_hi(naive_auc - trans_auc) < -0.30` |
| **`H_C1c`** | C1 | Localización temporal: fuera del instante de colapso ($\tau = 0.25$) el patrón se disuelve | `off_time_auc < 0.65` en `transactional_half_freq` y `linear_counterprop` |
| **`H_C2a`** | C2 | Búsqueda inversa recupera la configuración física exacta desde inicio aleatorio (entrenando solo en $[2, 60]$) | Media `CONFIRM`: $|\hat{\tau}^*| < 0.005$, $|\hat{k}_{\text{sum}}^* - 1| < 0.002$, $|\text{wrap}(\hat{\Delta\phi}^* - \pi)| < 0.15$, $|\hat{\beta}_{\text{disp}}^*| < 0.002$ |
| **`H_C2b`** | C2 | Generalización a primos nunca vistos ($x \in [61, 150]$) de la configuración descubierta en $[2, 60]$ | `CI95_lo(heldout_auc_opt) > 0.95` y `CI95_lo(heldout_auc_opt - heldout_auc_rand) > 0.25` |
| **`H_C2c`** | C2 | Ventaja del exponente crítico descubierto $\hat{\sigma}_{\text{tot}}^*$ frente al escalado fijo $-0.5$ en $[61, 150]$ | `CI95_lo(heldout_snr_opt - heldout_snr_neg_half) > 0.0` |
| **`H_C3a`** | C3 | Reversibilidad de fase y agudeza espectral de la ventana de colapso | `AUC(Δφ=π) > 0.98`, `AUC(Δφ=0) < 0.02`, `AUC(|δk|=0.02) < 0.65` |
| **`H_C3b`** | C3 | Transición de resolución de Rayleigh por número de pares de ondas $M$ en $[61, 150]$ | `AUC(M=256) > 0.99` y `AUC(M=16) < 0.70` |
| **`H_C4a`** | C4 | Descubrimiento ciego de ceros reales de Riemann (0 llamadas a $\zeta(s)$) con ablación causal de repulsión | `CI95_lo(found_dyson) >= 15.0`, `CI95_lo(found_dyson - found_norep) > 5.0` y `CI95_hi(err_dyson - err_norep) < -0.5` |
| **`H_C4b`** | C4 | Reconstrucción de primos ($[2, 45]$) y predicción fuera de muestra de $\psi(x)-x$ en $[601, 1200]$ desde ceros descubiertos a ciegas | `CI95_lo(auc2_45_dyson - auc2_45_rand) > 0.25` y `CI95_lo(extrap_psi_corr_dyson) > 0.50` y `CI95_lo(extrap_psi_corr_dyson - extrap_psi_corr_rand) > 0.45` |

---

## 4. Pruebas Unitarias Obligatorias y Aislamiento

- Exactitud de los 10 primeros ceros de $\zeta(\tfrac{1}{2}+i\gamma)$ vs. Odlyzko ($< 10^{-9}$) y consistencia de $Z(t)$.
- Exactitud del gradiente analítico $R'(\omega)$ de Guinand–Weil frente a diferencias finitas centrales ($< 10^{-7}$).
- Verificación de que la función dual $R(\omega)$ (sin usar $\zeta(s)$) alcanza máximos locales en los primeros ceros reales $\gamma_1, \dots, \gamma_{10}$ con error $< 0.20$.
- Identidad matemática exacta entre el solapamiento complejo `transactional_half_freq` y su forma analítica cerrada, determinismo bit a bit por semilla y 0 NaN/Inf.
- Aislamiento estricto: ningún archivo fuera de `riemann_collapse/` y `docs/riemann_collapse_v1/` será modificado. Checks del crate raíz (`Paso0`, `known_acc16`, `E33`, `aura-engine`, `orbita`): **NO EVALUADOS**.
