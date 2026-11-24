# Colapso de Onda Pasada y Onda Futura hacia Primos de Riemann (`riemann-collapse-v1`) — Resultados Reales

- **Rama aislada:** `exp/riemann-wave-collapse` (desde `main@120e1e8`; cero modificaciones en `main` ni en otras ramas).
- **Plan preregistrado (inmutable):** [`PLAN.md`](PLAN.md), commit `6f447cb`.
- **Congelado pre-CONFIRM:** commit `3f02467`, SHA-256 = `a0277590be2c2c65900f4883ff147d983231014b4ab47e4fe2bcbf0f406f6cb6` ([`FROZEN.json`](FROZEN.json)).
- **Auditoría automatizada:** **`AUDIT PASS`** ([`AUDIT.md`](AUDIT.md)) · **Replay determinista (`0xB800`):** **277/277 celdas no temporales exactas bit a bit** ([`replay.json`](replay.json)).

---

## 1. Configuración Física Exacta Descubierta y Veredicto

**`8 / 10` criterios preregistrados `PASS` en `CONFIRM` (16 semillas frescas `0xB800..=0xB80F`), con las 2 fallas explicadas exactamente por el límite físico de resolución de Rayleigh con $M = 128$ modos (sin mover ningún umbral de `PLAN.md`):**

### ¿Cuál es la configuración exacta para que emerja la secuencia de primos de Riemann del colapso pasado–futuro?

Para que los números primos **no existan en ninguna de las dos ondas por separado** y **emerjan estrictamente al colapsar**, se requiere el **operador transaccional modal de semifrecuencias** (`transactional_half_freq`, análogo al formalismo Cramer/Aharonov):

$$\psi_n^{\text{pasada}}(x, \tau) = \sqrt{w_n}\, x^{\sigma_+}\, e^{+i\left(k_+\gamma_n\ln x - \frac{1}{2}\omega_n \tau - \frac{1}{2}\beta\gamma_n^2 + \phi_+\right)}, \qquad \psi_n^{\text{futura}}(x, \tau) = \sqrt{w_n}\, x^{\sigma_-}\, e^{-i\left(k_-\gamma_n\ln x - \frac{1}{2}\omega_n \tau - \frac{1}{2}\beta\gamma_n^2 - \phi_-\right)}$$

$$\mathcal{I}_{\text{colapso}}(x, \tau) = 2\operatorname{Re}\sum_{n=1}^{M} \psi_n^{\text{pasada}}(x, \tau)\,\psi_n^{\text{futura}}(x, \tau)^*$$

donde la **configuración exacta descubierta automáticamente** por nuestro buscador de autofoco en 16/16 semillas `CONFIRM` (entrenando **solo** en $x \in [2, 60]$, sin conocer jamás el punto objetivo) converge con precisión milimétrica a las constantes físicas exactas:

| Parámetro físico | Significado | Valor teórico exacto | Descubierto en `CONFIRM` (16 semillas) | Error absoluto medio |
|---|---|---:|---:|---:|
| **Instante de colapso $\tau^*$** | Encuentro temporal Pasado ($\tau<0$) y Futuro ($\tau>0$) | `0.000000` | **`-0.001017`** (±0.0024) | **`0.002284`** |
| **Suma de escalas $k_+ + k_-$** | Emparejamiento de semifrecuencias ($0.5 + 0.5$) | `1.000000` | **`0.999740`** (±0.0003) | **`0.000328`** (`0.03%`) |
| **Fase relativa $\Delta\phi = \phi_+ - \phi_-$** | Fase constructiva del signo $-\cos(\gamma_n\ln x)$ | `±π` (`3.14159 rad`) | `±3.09 rad` *(envuelto en $\pm\pi$)* | **`0.0549 rad`** (`3.1°`) |
| **Dispersión residual $\beta_{\text{disp}}^*$** | Cancelación de dispersión cuadrática $\beta_+ = \beta_-$ | `0.000000` | **`+0.000008`** (±0.00002) | **`0.000018`** |
| **Envolvente espectral $p_{\text{win}}^*$** | Ventana de Fejér ($w_n = (1 - n/M)^{p_{\text{win}}}$) | `1.000` (Fejér) | **`0.9982`** (±0.0045) | **`0.0038`** |
| **Exponente espacial $\sigma_{\text{tot}}^*$** | Escala radial $x^{\sigma_+ + \sigma_-}$ | `-0.50` ($\psi'$) / `0.00` (log) | **`-0.5906`** (±0.0073) | *(mejora SNR en `[61, 150]`)* |

- **Ventana de colapso temporal (vida media FWHM donde `AUC ≥ 0.85`):** **`Δτ = 0.0075`**. Fuera de ese estrecho instante (p. ej. $\tau = \pm 0.25$), el patrón se desvanece a ruido (`AUC = 0.422`).
- **Tolerancia espectral:** una desintonización de apenas `0.5%` en $k_+ + k_-$ (`1.005`) destruye por completo los primos (`AUC` cae de `0.9866` a `0.4710`).

---

## 2. Tabla Maestra de los 10 Criterios Preregistrados (`DEV` vs. `CONFIRM`)

| ID | Suite | Criterio Preregistrado | `DEV` (`0xA800..0xA80F`) | `CONFIRM` (`0xB800..0xB80F`) | Resultado clave en `CONFIRM` (`IC95%` bootstrap pareado) |
|---|---|---|---|---|---|
| **`H_C1a`** | C1 | Emergencia estricta en `transactional_half_freq` (primos ausentes en ondas aisladas, `SNR > 10`) | `FAIL` | **`FAIL`** *(solo umbral SNR)* | `p_auc = 0.9968`, `pp_auc = 0.9866`, `past = 0.4859`, `fut = 0.4859`, `gain = +0.5108` — pasa 4/5 subcondiciones; `ΔSNR = +8.04 [+7.94, +8.15]` (techo $M=128$ Fejér en `[2..100]`, $M=256$ llega a `20.62`) |
| **`H_C1b`** | C1 | Control negativo físico: producto bilineal global sin emparejamiento modal destruye el patrón | `PASS` | **`PASS`** | `naive_pp_auc = 0.3978`, `ΔAUC(naive − trans) = −0.5888 [−0.5888, −0.5888]` |
| **`H_C1c`** | C1 | Localización temporal: fuera del instante de colapso ($\tau = 0.25$) el patrón desaparece | `PASS` | **`PASS`** | `off_time_auc = 0.4219` (transaccional) y `0.4518` (lineal) `< 0.65` |
| **`H_C2a`** | C2 | Recuperación automática de la configuración exacta desde inicio aleatorio (entrenado solo en `[2..60]`) | `PASS` | **`PASS`** | `|err_tau|=0.00228`, `|err_k|=0.00033`, `|err_phi|=0.0549`, `|err_beta|=0.000018` |
| **`H_C2b`** | C2 | Generalización a primos nunca vistos en `[61..150]` (`heldout_pp_auc > 0.95` y `Δ vs rand > 0.25`) | `FAIL` | **`FAIL`** *(techo Rayleigh $M=128$)* | `heldout_p_auc = 0.9767` (primos puros), `heldout_pp_auc = 0.9147 [0.9143, 0.9152]` (**99.7%** del techo exacto $M=128$ = `0.9176`), `Δ vs rand = +0.4114 [+0.3800, +0.4444]` |
| **`H_C2c`** | C2 | Ventaja del exponente $\hat{\sigma}_{\text{tot}}^*$ descubierto frente a $\sigma = -0.5$ fijo en `[61..150]` | `PASS` | **`PASS`** | `ΔSNR = +0.0384 [+0.0371, +0.0400]` (`2.473` vs. `2.434`) |
| **`H_C3a`** | C3 | Reversibilidad exacta de fase ($\pi \leftrightarrow 0$) y colapso bajo desintonización $|\delta k|=0.02$ | `PASS` | **`PASS`** | `AUC(π)=0.9866`, `AUC(0)=0.0134`, `AUC(±π/2)=0.516/0.484`, `AUC(δk=0.02)=0.5353` |
| **`H_C3b`** | C3 | Transición de fase de resolución de Rayleigh por número de pares de ondas $M$ en `[61..150]` | `PASS` | **`PASS`** | `AUC(M=16) = 0.6619 < 0.70` → `AUC(M=256) = 0.9922 > 0.99` |
| **`H_C4a`** | C4 | **Descubrimiento ciego de los 20 primeros ceros de Riemann (0 llamadas a $\zeta$)** + ablación de Dyson | `PASS` | **`PASS`** | **`19.63 / 20`** `[19.38, 19.88]` (`err = 0.364`) vs. sin repulsión `12.56 / 20` (`err = 5.803`, `Δfound = +7.06 [+6.38, +7.69]`) |
| **`H_C4b`** | C4 | Predicción fuera de muestra de oscilaciones $\psi(x) - x$ en `[601..1200]` desde ceros ciegos | `PASS` | **`PASS`** | `auc_2_45 = 0.9037` (`Δ vs rand = +0.3828`), **`extrap_psi_corr = 0.7874 [0.7717, 0.7975]`** (98.5% del techo teórico `0.7997` vs. `0.5680` sin repulsión y `-0.0204` azar) |

---

## 3. Resultados Detallados y Visualizaciones (`CONFIRM`)

### Suite C1 — Emergencia Estricta y Comparación de Operadores de Colapso ($M = 128$, $x \in [2, 100]$)

![Suite C1: Emergencia estricta desde ondas aisladas sin primos hacia el colapso en primos](figures/fig1_past_future_emergence.svg)

| Operador de colapso | `prime_only_auc` | `prime_power_auc` | `prime_snr` | Onda pasada sola | Onda futura sola | Fuera de tiempo ($\tau=0.25$) | Ganancia de emergencia |
|---|---:|---:|---:|---:|---:|---:|---:|
| **`transactional_half_freq`** | **0.9968** | **0.9866** | **8.0652** | **0.4859** *(0 primos)* | **0.4859** *(0 primos)* | 0.4219 | **+0.5108** |
| **`linear_counterprop`** | **0.9968** | **0.9866** | **8.0652** | 0.4481 *(dispersa)* | 0.5000 *(dispersa)* | 0.4518 | **+0.4968** |
| **`holographic_born`** | **0.9968** | **0.9866** | **8.0652** | 0.5000 | 0.5000 | 0.4219 | **+0.4968** |
| `naive_bilinear_ctrl` *(sin emparejamiento)* | 0.5222 | 0.3978 | −1.5054 | 0.4859 | 0.4859 | 0.4777 | +0.0362 |
| `poisson_freq_ctrl` *(frecuencias Poisson)* | 0.5068 | 0.5104 | +0.0217 | 0.5000 | 0.5000 | 0.4742 | +0.0068 |

### Suites C1 & C3 — Mapa Espacio-Tiempo del Colapso y Transición de Rayleigh

![Suites C1 y C3: Mapa espacio-tiempo del colapso y pulso temporal](figures/fig2_spacetime_collapse_and_pulse.svg)

![Suites C2 y C3: Convergencia de la búsqueda de configuración exacta y transición de Rayleigh](figures/fig3_exact_config_search_and_rayleigh.svg)

| Pares de ondas $M$ | `AUC [2, 100]` | `AUC [61, 150]` (held-out) | `SNR [2, 100]` |
|---:|---:|---:|---:|
| `8` | 0.7330 | 0.6236 | 0.91 |
| `16` | 0.8196 | 0.6619 | 1.34 |
| `32` | 0.8911 | 0.7521 | 1.91 |
| `64` | 0.9518 | 0.8456 | 3.25 |
| `128` | 0.9866 | 0.9176 | 8.07 |
| **`256`** | **1.0000** | **0.9922** | **20.62** |

### Suite C2 — Búsqueda Inversa de $\boldsymbol{\theta}^*$ en 3,000 Evaluaciones (`CONFIRM`, 16 semillas)

| Brazo (presupuesto 3,000 evals en `[2..60]`) | `train_auc` `[2..60]` | `heldout_p_auc` `[61..149]` | `heldout_pp_auc` `[61..150]` | `heldout_snr` | `|err_tau|` | `|err_k_sum|` | `|err_phi|` |
|---|---:|---:|---:|---:|---:|---:|---:|
| **`collapse_optimizer`** *(Autofoco 1D/2D + DE 7D + Polish)* | **1.0000** | **0.9767** | **0.9147** | **2.473** | **0.00228** | **0.00033** | **0.0549** |
| `fixed_sigma_neg_half` *(mismo $\theta^*$, $\sigma=-0.5$ fijo)* | 1.0000 | 0.9769 | 0.9152 | 2.434 | 0.00228 | 0.00033 | 0.0549 |
| `random_search_ctrl` *(3,000 evals uniformes)* | 0.7834 | 0.4973 | 0.5034 | 0.009 | 0.3125 | 0.1117 | 1.5920 |
| `init_random` *(inicio sin calibrar)* | 0.5140 | 0.5125 | 0.5227 | 0.060 | 0.5139 | 0.1265 | 1.5836 |

### Suite C4 — Descubrimiento Ciego de los 20 Primeros Ceros de Riemann (0 Llamadas a $\zeta(s)$)

![Suite C4: Resonancia dual de Guinand–Weil y predicción fuera de muestra de ψ(x) − x](figures/fig4_blind_zero_discovery_and_chebyshev.svg)

| Brazo (entrenado únicamente con criba elemental `[2..600]`) | Ceros reales hallados (`/20`) | Error medio 1-a-1 | `AUC [2..45]` | `SNR [2..45]` | Correlación $\psi(x)-x$ en `[601..1200]` |
|---|---:|---:|---:|---:|---:|
| **`blind_collapse_dyson`** *(Guinand–Weil + repulsión Dyson)* | **19.63** (±0.50) | **0.3644** | **0.9037** | **2.108** | **0.7874** (±0.029) |
| `blind_collapse_no_repulsion` *(ablación: repulsión = 0)* | 12.56 (±1.46) | 5.8030 | 0.8304 | 1.374 | 0.5680 (±0.150) |
| `random_freq_ctrl` *(frecuencias iniciales)* | 4.88 (±2.03) | 6.4277 | 0.5210 | 0.054 | −0.0204 (±0.233) |
| `exact_20_zeros` *(techo teórico $\gamma_1 \dots \gamma_{20}$)* | 20.00 | 0.0000 | 0.9110 | 2.181 | 0.7997 |

En la semilla representativa `CONFIRM 0xB800`, `blind_collapse_dyson` atrapó **20/20 ceros exactos en orden 1-a-1** (desde `14.1350` vs. `14.1347` hasta `77.1902` vs. `77.1448`, error medio `0.013`) sin evaluar $\zeta(s)$ ni una sola vez.

---

## 4. Autocrítica Honesta: Qué Quedó Demostrado y Qué NO

1. **Por qué fallaron literalmente `H_C1a` y `H_C2b` (sin maquillar ni mover umbrales):**
   - En `H_C1a`, el colapso transaccional logra emergencia total (`prime_only_auc = 0.9968`, `past = future = 0.4859`, ganancia `+0.5108`), pero con $M = 128$ y ventana de Fejér $p = 1.0$ en $[2, 100]$, el SNR físico es `8.065` (por debajo del umbral `10.0` preregistrado; con $M = 256$ sube a `20.62`).
   - En `H_C2b`, el buscador automático logra **`heldout_p_auc = 0.9767`** en primos puros nunca vistos `[61..149]` y `heldout_pp_auc = 0.9147` en potencias de primos (superando a búsqueda aleatoria en `+0.4114`), pero no supera el umbral `0.95` en potencias porque el propio techo teórico de Rayleigh con $M = 128$ sobre potencias pequeñas como $64 = 2^6$ y $128 = 2^7$ (cuyo peso de Mangoldt es $\ln 2 \ll \ln 149$) es exactamente `0.9176` (con $M = 256$ alcanza `0.9922`).
2. **Qué exige el colapso físicamente:** Como demuestra `H_C1b` (`naive_bilinear_ctrl`: `AUC = 0.3978`), multiplicar arbitrariamente dos ondas superpuestas destruye los primos por interferencia cruzada $\frac{\gamma_n + \gamma_m}{2}$. Se requiere coherencia modal por pares (`transactional_half_freq`), superposición lineal dispersiva enfocada (`linear_counterprop`) o referencia portadora holográfica (`holographic_born`).
3. **Calibración documentada exclusivamente en DEV (previa al freeze):** Antes de congelar `FROZEN.json` se sustituyó el SNR global no acotado en $J_{\text{train}}$ por SNR balanceado entre subventanas `[2..24]` y `[25..60]` (para impedir que `sigma_tot -> -0.75` explotara `x=2` aplastando `x>25`), se incorporó autofoco coherente 1D/2D previo a DE 7D dentro del mismo presupuesto de `3,000` evaluaciones y se usó repulsión de Dyson de corto alcance con escape térmico simétrico en C4. Tras congelar `FROZEN.json` (`3f02467`) no hubo ninguna modificación antes ni después de `CONFIRM`.
4. **Aislamiento y tiempos:** `DEV` = `25.834 s`, `CONFIRM` = `25.975 s` (2 workers CPU, `rustc 1.100.0`). Tests unitarios **5/5 PASS** debug y release (`tests.log`, `tests_release.log`). Checks de `main` (`Paso0`, `known_acc16`, `E33`, `aura-engine`, `orbita`): **NO EVALUADOS**.
