# Plan Preregistrado — Inferencia Retrodictiva y Bidireccional mediante Colapso en Patrón de Ondas Primo de Riemann (`exp/riemann-prime-inference`)

**Escrito antes de implementar cualquier código en `riemann_prime_inference/`.**
- **Rama aislada:** `exp/riemann-prime-inference` (desde `main@120e1e8`, sin tocar `main`, `exp/riemann-wave-inference` ni `exp/riemann-wave-collapse`).
- **Crate autónomo:** `riemann_prime_inference/` (Rust puro con `[workspace]` propio y **0 dependencias externas**; orquestación, bootstrap pareado y graficación SVG en Python con biblioteca estándar).
- **Pregunta del usuario:** Usar el mecanismo de colapso de ondas Pasado × Futuro como **motor de inferencia**, comparando **colapso forzado sobre el patrón primo de Riemann** frente a **colapso aleatorio**, evaluando explícitamente:
  1. **Rastreo retrodictivo (abducción):** rastrear el resultado/patrón observado hacia atrás hasta su origen atómico exacto.
  2. **Inferencia "del otro lado":** qué capacidades inferenciales concretas emergen tras el colapso (predicción futura no lineal, intervención contrafactual $do(c_a \to c_{\text{new}})$, desambiguación Bayesiana Prior Pasado × Evidencia Futura, y transferencia analógica $A:B::C:?$).
- **Alcance honesto:** Este experimento demuestra y mide la inferencia composicional en un alfabeto simbólico-latente finito ($K = 12$ causas atómicas = 66 pares sin colisiones) sobre un canal continuo de fase. **No** factoriza claves RSA en tiempo polinomial ni escala a vocabularios de miles de símbolos en 1D sin sufrir el techo físico de separación $O(\ln K / K^2)$ — límite que se mide y reporta explícitamente en la Suite `I4`.

---

## 1. Fundamento Matemático y 4 Brazos Comparados ($M = 128$ modos)

Cada causa/concepto atómico $c_k$ ($k = 0, \dots, K-1$, $K = 12$) se asigna a un atractor primo $p_k \in \{2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37\}$, con coordenada logarítmica $u_k = \ln p_k$. Por el **Teorema Fundamental de la Aritmética**, la composición de dos causas $(c_a, c_b)$ en espacio de fase $u_{a,b} = u_a + u_b = \ln(p_a p_b)$ es **100% única (0 colisiones entre los $\binom{12}{2} = 66$ pares distintos)**, en contraste con la composición lineal aditiva $a + b$ (que colapsa los 66 pares en solo 21 sumas posibles, generando $>68\%$ de ambigüedad irreversible).

En todas las suites comparamos **4 medios de colapso** ($M = 128$ ondas cada uno):
1. **`riemann_prime_collapse`**: campo de colapso Pasado × Futuro con los 128 primeros ceros exactos $\gamma_1, \dots, \gamma_{128}$ de $\zeta(\tfrac{1}{2} + i\gamma)$ (ventana de Fejér en espacio logarítmico $\mathcal{W}(u) = -2\sum_{n=1}^{128}(1 - \frac{n-1}{128})\cos(\gamma_n u)$).
2. **`random_collapse`** (*control directo de colapso aleatorio*): frecuencias aleatorias uniformes (Poisson) en $[\gamma_1, \gamma_{128}]$ y fases aleatorias por modo.
3. **`dyson_gue_collapse`** (*ablación quirúrgica*): frecuencias con repulsión GUE de Dyson ($\beta = 2$) mapeadas a $[\gamma_1, \gamma_{128}]$ — preserva repulsión GUE exacta pero **sin** resonar sobre los primos $\ln p_k$.
4. **`fourier_linear_collapse`** (*control estructural aditivo*): 128 armónicos enteros de Fourier sobre rejilla aditiva lineal ($c_k \mapsto k+1 \in \{1, \dots, 12\}$, composición $S = (a+1) + (b+1)$ en periodo $L = 28$).

---

## 2. Diseño de las 4 Suites Experimentales (`I1`–`I4`)

### Suite I1 — Rastreo Retrodictivo de Origen (`Resultado N → Origen {c_a, c_b}`)
- Se evalúan los 66 pares distintos $\{c_a, c_b\}$ ($0 \le a < b < 12$) bajo observación ruidosa continua $\tilde{u}_N = \ln(p_a p_b) + \eta$, $\eta \sim \mathcal{N}(0, \sigma_\eta^2)$ ($\sigma_\eta \in \{0.0, 0.002, 0.006\}$; para el control lineal, ruido proporcional en fracción del paso mínimo del dominio $\sigma_{\text{lin}} = \sigma_\eta / \Delta u_{\min} \cdot \Delta s_{\min}$ o incluso **0 ruido** para no perjudicar a Fourier lineal: en `fourier_linear_collapse` usamos $\sigma_{\text{lin}} = 0.05$ en paso entero $1.0$, que es incluso más favorable que el ruido relativo de primos!).
- **Operador de rastreo puramente ondulatorio (sin tabla de división simbólica):**
  - Onda pasada sondea un origen candidato continuo $u \in [\ln 1.7,\; \tilde{u}_N / 2]$: $\mathcal{W}_{\text{pasada}}(u) = \mathcal{W}(u)$.
  - Onda futura retrodictiva viaja hacia atrás desde el resultado observado $\tilde{u}_N$: $\mathcal{W}_{\text{futura}}(u \mid \tilde{u}_N) = \mathcal{W}(\tilde{u}_N - u)$.
  - Campo de colapso retrodictivo:
    $$S_{\text{origen}}(u \mid \tilde{u}_N) = \max(0, \mathcal{W}_{\text{pasada}}(u)) \cdot \max(0, \mathcal{W}_{\text{futura}}(u \mid \tilde{u}_N))$$
  - En barrido continuo de 400 puntos sobre $u \in [\ln 1.7, \tilde{u}_N / 2]$, se extrae el pico máximo $\hat{u}_1 = \arg\max_u S_{\text{origen}}(u \mid \tilde{u}_N)$ y su conjugado $\hat{u}_2 = \tilde{u}_N - \hat{u}_1$, proyectando al concepto atómico más cercano $(\hat{c}_a, \hat{c}_b)$ (en `fourier_linear_collapse`, si varios pares enteros alcanzan igual pico máximo por ambigüedad de suma, se sortea uniformemente entre los pares empatados).
- **Métricas:** `origin_exact_acc` (ambas causas exactas), `origin_partial_f1`, `peak_margin_snr` (razón del pico verdadero frente al mayor pico espurio).

### Suite I2 — "Del Otro Lado": Inferencia Predictiva y Contrafactual $do(c_a \to c_{\text{new}})$
- Sobre cada evento de prueba (los 66 pares con ruido $\sigma_\eta = 0.002$), definimos una ley causal no lineal "del otro lado":
  $$y_1(a, b) = \sin(0.9 a + 1.3 b) + 0.45 \cos\!\left(\frac{(a+1)(b+1)}{4}\right), \qquad y_2(a, b) = \cos(1.1 a - 0.7 b) + 0.35 (-1)^{a+b}$$
  y clase discreta de 4 cuadrantes $k_{\text{fut}}(a, b) = 2\cdot\mathbb{I}(y_1 \ge 0) + \mathbb{I}(y_2 \ge 0) \in \{0,1,2,3\}$.
- **Flujo bidireccional completo:**
  1. **Abducción (hacia atrás):** desde $\tilde{u}_N$ ruidoso, rastrear el origen $(\hat{a}, \hat{b})$ mediante el colapso de ondas.
  2. **Predicción (del otro lado):** evaluar $\hat{y}_{\text{fut}} = y(\hat{a}, \hat{b})$ y $\hat{k}_{\text{fut}}$; medir `pred_rmse` y `class_acc`.
  3. **Intervención contrafactual $do(c_a \to c_{\text{new}})$:** dada una causa nueva $c_{\text{new}} \notin \{a, b\}$, el sistema desplaza la fase de onda observada usando su origen inferido $\tilde{u}_{\text{cf}} = \tilde{u}_N - \ln \hat{p}_a + \ln p_{\text{new}}$, vuelve a colapsar del otro lado para decodificar $(\hat{a}_{\text{cf}}, \hat{b}_{\text{cf}})$ y evalúa la consecuencia contrafactual $\hat{y}_{\text{cf}}$ (`cf_exact_acc` y `cf_rmse`).

### Suite I3 — Inferencia Transaccional Bayesiana (Prior Pasado × Evidencia Futura bajo Alto Ruido)
- Bajo ruido severo $\sigma_\eta = 0.015$ (superior a la brecha mínima entre productos cercanos como $17\times 19 = 323$ y $11\times 29 = 319$, $\Delta u = 0.0125$), generamos 200 eventos por semilla siguiendo una distribución **prior pasada** contextual no uniforme sobre causas atómicas (Zipf $P(c_k) \propto (k+1)^{-1.4}$ con permutación aleatoria por semilla).
- Comparamos 4 brazos:
  1. **`riemann_bayes_collapse`**: $S_{\text{post}}(u \mid \tilde{u}_N) = \pi_{\text{pasado}}(u, \tilde{u}_N - u) \cdot \max(0, \mathcal{W}_{\text{R}}(u)) \cdot \max(0, \mathcal{W}_{\text{R}}(\tilde{u}_N - u))$ (prior pasado suave sobre los atractores acoplado al colapso de Riemann).
  2. **`riemann_uniform_collapse`**: colapso de Riemann sin prior pasado ($\pi \equiv 1$).
  3. **`random_bayes_collapse`**: mismo prior pasado acoplado a colapso aleatorio.
  4. **`prior_only_ctrl`**: predicción puramente desde el prior pasado (sin ondas ni observación).

### Suite I4 — Inferencia Analógica ($A:B::C:?$) y Límites Físicos Honestos (Ruido y Escalabilidad)
- **I4a (Transferencia analógica composicional):** 80 tríos por semilla donde $A = p_a p_x$, $B = p_b p_x$ (comparten contexto $p_x$ y cambian $p_a \to p_b$), y dado $C = p_a p_y$ ruidoso ($\sigma_\eta = 0.002$), el sistema calcula el operador espectral de cambio $\Delta u = \tilde{u}_B - \tilde{u}_A$, aplica $\tilde{u}_D = \tilde{u}_C + \Delta u$ y colapsa sobre el patrón de ondas para recuperar el destino exacto $D = \{p_b, p_y\}$ (`analogy_exact_acc`).
- **I4b (Curva honesta de ruido y capacidad):**
  - Barrido de ruido $\sigma_\eta \in \{0.0, 0.001, 0.003, 0.008, 0.015, 0.030\}$ sobre `riemann_prime_collapse` ($K = 12$).
  - Barrido de capacidad de vocabulario $K \in \{8, 12, 18\}$ conceptos atómicos ($\binom{K}{2} \in \{28, 66, 153\}$ pares) bajo $\sigma_\eta = 0.003$ con $M = 128$ ceros, midiendo la degradación real al estrecharse $\Delta u_{\min} \sim O(\ln K / K^2)$.

---

## 3. Splits, Congelamiento y los 8 Criterios Preregistrados (`H_I1a`–`H_I4b`)

- **`DEV`:** 16 semillas `0xA900..=0xA90F` · **Freeze pre-CONFIRM:** `docs/riemann_prime_inference_v1/FROZEN.json` · **`CONFIRM`:** 16 semillas frescas `0xB900..=0xB90F` · **Replay bit-exacto:** `0xB900` · **Bootstrap pareado:** 10,000 remuestreos (`seed = 12345`), `IC95%`.

| ID | Suite | Hipótesis / Criterio Preregistrado | Regla de Aceptación (`CONFIRM`, $N=16$) |
|---|---|---|---|
| **`H_I1a`** | I1 | Rastreo de origen ($\sigma_\eta = 0.002$): colapso primo vence a colapso aleatorio y a Dyson sin primos | `CI95_lo(acc_riemann) > 0.90`, `CI95_lo(acc_riemann - acc_random) > 0.80` y `CI95_lo(acc_riemann - acc_dyson) > 0.80` |
| **`H_I1b`** | I1 | Ventaja estructural de factorización única frente a suma aditiva lineal (`fourier_linear_collapse`) | `CI95_lo(acc_riemann - acc_fourier) > 0.45` |
| **`H_I2a`** | I2 | Inferencia predictiva "del otro lado" ($\hat{y}_{\text{fut}}$, clase 4-way) | `CI95_lo(class_acc_riemann) > 0.90`, `CI95_hi(rmse_riemann - rmse_random) < -0.50`, `CI95_hi(rmse_riemann - rmse_fourier) < -0.25` |
| **`H_I2b`** | I2 | Inferencia contrafactual $do(c_a \to c_{\text{new}})$ mediante desplazamiento de fase y re-colapso | `CI95_lo(cf_acc_riemann) > 0.85` y `CI95_lo(cf_acc_riemann - cf_acc_fourier) > 0.40` |
| **`H_I3a`** | I3 | Desambiguación Bayesiana Pasado × Futuro bajo alto ruido ($\sigma_\eta = 0.015$) | `CI95_lo(bayes_riemann - unif_riemann) > 0.05`, `CI95_lo(bayes_riemann - bayes_random) > 0.30`, `CI95_lo(bayes_riemann - prior_only) > 0.30` |
| **`H_I4a`** | I4 | Transferencia analógica composicional ($A:B::C:?$) | `CI95_lo(analogy_riemann - analogy_random) > 0.70` y `CI95_lo(analogy_riemann - analogy_fourier) > 0.30` |
| **`H_I4b`** | I4 | **Autocrítica física honesta:** degradación real del canal escalar 1D bajo ruido severo ($\sigma_\eta = 0.030$) | `CI95_hi(acc_noise_030 - acc_noise_000) < -0.30` (confirma límite físico sin ocultarlo) |
| **`H_I4c`** | I4 | **Autocrítica de capacidad:** caída de margen mínimo al escalar de $K=8$ a $K=18$ bajo ruido $\sigma_\eta = 0.003$ | Reportar `acc(K=8)`, `acc(K=12)`, `acc(K=18)` y confirmar explícitamente que `CI95_hi(acc_K18 - acc_K8) < 0.0` |

---

## 4. Pruebas Unitarias y Aislamiento

- Exactitud de ceros de Riemann ($< 10^{-9}$), unicidad 0-colisiones de los 66 productos primos vs. colisiones de sumas lineales, recuperación exacta sin ruido en `riemann_prime_collapse` ($\ge 98\%$), determinismo bit a bit y 0 NaN/Inf.
- Aislamiento estricto en `riemann_prime_inference/` y `docs/riemann_prime_inference_v1/`. Checks de `main` (`Paso0`, `known_acc16`, `E33`, `aura-engine`, `orbita`): **NO EVALUADOS**.
