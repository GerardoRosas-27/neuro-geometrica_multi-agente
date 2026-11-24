# Propuesta Archivada — Frecuencia de Ondas de Riemann como Medio de Inferencia

- **Fuente compartida por el usuario:** `https://share.gemini.google/TcEnIP9sYZmT` (redirección canónica: `https://gemini.google.com/share/1fcb8ce2780f`)
- **Fecha de archivo:** 2026-11-24
- **Rama aislada:** `exp/riemann-wave-inference`

---

## 1. Intuición Central del Usuario

A partir de la conexión entre los ceros no triviales de la función zeta de Riemann ($s_n = \frac{1}{2} + i\gamma_n$), el caos cuántico (conjetura de Hilbert–Pólya y ensamble GUE) y la interferencia constructiva de ondas ("música de los primos"), el usuario formuló la hipótesis de diseño:

> *"Es como un sistema que está consolidando información y las ondas donde caen puede ser lo aprendido; es posible que sea la frecuencia de pensamiento de un sistema."*
> *"Sí, crea un sistema que simule como resultado el patrón de Riemann. Que las ondas de inferencia tengan el mismo patrón para que el sistema no termine en caos."*

---

## 2. Los Dos Mecanismos Propuestos en la Conversación Original

### Parte A — Interferencia de Ondas de Primos y Ley de Montgomery–Odlyzko (GUE vs. Riemann)

1. **Superposición de ondas de Riemann (fórmula explícita de Riemann–Mangoldt con ventana de Fejér):**
   $$W(x) = -\frac{2}{\sqrt{x}} \sum_{n=1}^{M} \left(1 - \frac{n-1}{M}\right) \cos(\gamma_n \ln x)$$
   donde las frecuencias son las partes imaginarias $\gamma_n$ de los ceros no triviales de $\zeta(\tfrac{1}{2} + i\gamma_n)$. La interferencia constructiva genera picos agudos en los números primos $p$ y potencias de primos $p^k$ sin guardar una lista explícita de primos.

2. **Repulsión de niveles (GUE vs. Riemann vs. Poisson):**
   - Unfolding de los ceros de Riemann mediante $N(T) \approx \frac{T}{2\pi}\ln\left(\frac{T}{2\pi e}\right)$.
   - Comparación con Hamiltonianos Hermitianos aleatorios complejos $H = \frac{A + A^\dagger}{2\sqrt{D}}$ (Gaussian Unitary Ensemble, GUE, con unfolding por la CDF del semicírculo de Wigner).
   - Comparación estadística frente a la ley teórica de Wigner GUE:
     $$P_{\text{GUE}}(s) = \frac{32}{\pi^2} s^2 e^{-\frac{4}{\pi} s^2}$$
     y frente al control sin repulsión de Poisson $P_{\text{Poisson}}(s) = e^{-s}$.

### Parte B — Memoria de Ondas con Autorregulación Espectral (Gas de Dyson $\beta = 2$)

1. **Consolidación espectral por dinámica de Langevin de Dyson ($\beta = 2$):**
   $$w_i^{(t+1)} = w_i^{(t)} + \left(-\frac{1}{2} w_i^{(t)} + \sum_{j \neq i} \frac{w_i^{(t)} - w_j^{(t)}}{(w_i^{(t)} - w_j^{(t)})^2 + \varepsilon^2}\right) \Delta t + \sqrt{\Delta t}\,\xi_i, \quad \xi_i \sim \mathcal{N}(0, 1)$$
   con normalización (*unfolding*) al intervalo $[1, M]$ mediante la CDF del semicírculo de Wigner ($R = 2\sqrt{M}$):
   $$f_i = 1 + (M - 1)\left[\frac{1}{2} + \frac{1}{\pi}\left(z_i\sqrt{1 - z_i^2} + \arcsin z_i\right)\right], \quad z_i = \operatorname{clip}\left(\frac{w_i}{R}, -0.9999, 0.9999\right)$$

2. **Inferencia por base de ondas en geometría logarítmica de Riemann:**
   $$u(x) = 2\pi \frac{\ln(x/x_{\min})}{\ln(x_{\max}/x_{\min})} \in [0, 2\pi)$$
   $$\Phi(u; \mathbf{f}, \boldsymbol{\phi}) = \begin{bmatrix} 1 & \cos(u f_1 + \phi_1) & \dots & \cos(u f_M + \phi_M) & \sin(u f_1 + \phi_1) & \dots & \sin(u f_M + \phi_M) \end{bmatrix}$$
   - Se aprenden picos discretos ("conceptos") a partir de observaciones con ruido sensorial ($\sigma_\eta = 0.04$) mediante mínimos cuadrados.
   - Se evalúa la inferencia bajo perturbación interna de fase ($\phi_k \sim \mathcal{N}(0, 0.015^2)$ rad), comparando el número de condición $\kappa(\Phi)$, la energía de pesos $\|\mathbf{w}\|_2$ y el error RMSE frente a un control sin repulsión (frecuencias aleatorias tipo Poisson).

---

## 3. Preguntas Críticas Científicas Añadidas para la Evaluación Rigurosa

El protocolo preregistrado en [`PLAN.md`](PLAN.md) extiende la propuesta original para evitar sesgos de confirmación y responder cinco preguntas abiertas clave:

1. **¿Es el colapso de Poisson solo un artefacto de mínimos cuadrados sin regularizar ($\lambda = 0$)?** Se evalúa tanto OLS puro como Ridge ($\lambda > 0$).
2. **¿Cómo se compara el patrón Riemann/Dyson contra una rejilla armónica entera (`fourier_uniform`, $\beta \to \infty$) y contra una rejilla determinista inconmensurable (`golden_jitter`)?**
3. **¿Dónde supera estructuralmente un espectro inconmensurable repulsivo (Riemann / Dyson) a Fourier entero?** Se prueban ecos fantasma fuera de periodo ($u \in [2\pi, 4\pi)$) y muestreo irregular asíncrono.
4. **¿La repulsión de Dyson evita el colapso/caos durante aprendizaje secuencial online cuando las frecuencias se co-adaptan?** Se añade una ablación causal directa (`adaptive_dyson` vs. `adaptive_no_repulsion`).
5. **¿Funciona el espectro Riemann/Dyson como medio real de inferencia estadística (Random Fourier Features / DPP-GUE) para aproximación de kernels y predicción de sistemas dinámicos no lineales 1D y 2D?** Se compara frente a RFF i.i.d. clásico (Rahimi–Recht) y contra el control más fuerte: Cuasi-Monte Carlo (`qmc_rff`).
