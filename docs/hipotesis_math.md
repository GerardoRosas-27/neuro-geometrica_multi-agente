# Hipótesis experimentales derivadas de `openai/math` para Dφ (neuro-geométrica multi-agente)

**Fecha:** 2026-10-13 · **Autor:** Grok Bot (para Bruster Rosas / Arquitecto-IA) · **Estado:** propuesta, sin ejecutar. **No hay resultados nuevos en este documento.** Todas las cifras del proyecto vienen de `docs/resultados_v4.md` (main) y de los preregistros citados.

## 0. Qué repositorio se usó y por qué es distinto a lo esperado

- `https://github.com/openai/math` **existe** (API: HTTP 200, no es fork, no está archivado, rama `main`). Clonado `--depth 1` en `/workspace/research/openai-math`, commit `1581d4cbdd69a0faf34b2089614fe2a814316a6a` (12-oct-2026, 14:55 −07:00 = 15:55 hora CDMX).
- **Es distinto a lo esperado:** no contiene código de entrenamiento, modelos, datasets de problemas ni reward models. Según `README.md` es una *colección de manuscritos matemáticos y artefactos de prueba producidos por un modelo interno de OpenAI*: 719 manuscritos en 372 familias (`CONTENTS.md`), 749 directorios en `preprints/` (PDF + fuentes), una biblioteca Lean (`lean/OAI/<área>`, `lean/formalization.yaml`, ~42 % de resultados principales formalizados según el README), retos para el verificador *Comparator* (`lean/ComparatorChallenges/*.lean|*.json`, enunciados con `sorry` que se verifican contra la prueba), resúmenes de razonamiento (`reasoning_traces/*.pdf`) y un registro de retiradas/correcciones (`history.md`; p. ej. el 7-oct-2026 se retiraron 3 manuscritos por un error de signo).
- Proceso declarado (`README.md`): ~4 000 problemas abiertos planteados al modelo, ~3 h de cómputo "ChatGPT Pro thinking" por resultado de media.
- Como el pedido original también menciona atención y reward models de proceso, que **no están en `openai/math`**, esas fórmulas se citan de literatura real y del repo relacionado `openai/prm800k` (existe, HTTP 200; no se clonó a fondo). Se indica explícitamente en cada caso.

## 1. Resumen del contenido real (archivos concretos)

| Elemento | Archivo | Qué aporta |
|---|---|---|
| Índice | `CONTENTS.md` | 372 familias con resumen y abstracts; enlaces a PDFs |
| Lean | `lean/README.md`, `lean/formalization.yaml` (schema v0.4) | Catálogo de pruebas formales, áreas `Algebra, Analysis, Combinatorics, Dynamics, ...` |
| Verificación | `lean/ComparatorChallenges/README.md` | `lake env comparator ComparatorChallenges/<X>.json` |
| Trazas | `reasoning_traces/` | 10 resúmenes de razonamiento (p. ej. `mezard-parisi-formula.tex`) |
| Erratas | `history.md` | Retiradas y revisiones versionadas |

Resultados con relación (indirecta) a modelos dinámicos/aprendizaje — los únicos que uso:

1. **374 – Estabilidad 1/3 de mapas de Brenier** (`preprints/Sharp-One-Third-Stability-of-Brenier-Maps-September-25-2026/article.pdf`; enunciado formal en `lean/ComparatorChallenges/Brenier.lean`, teoremas `one_third_stability` y `one_third_exponent_is_sharp`):
   $$\|T_\mu-T_\nu\|_{L^2(\rho)}\le C\,W_2(\mu,\nu)^{1/3},$$ exponente óptimo (incluso con objetivos de 3 átomos).
2. **140 – Cotas memoria–muestras en regresión gaussiana sin ruido** (`preprints/Memory-and-precision-in-noiseless-Gaussian-regression-September-27-2026/paper.pdf` y 5 compañeros; `lean/ComparatorChallenges/NoiselessRegression.lean`): un aprendiz de una pasada con $o(d^2)$ bits persistentes necesita $T=\Omega(d\log(1/\epsilon))$ muestras para error angular $\epsilon$.
3. **148 – Fórmula entropía/dimensión** (`preprints/The-entropy-rate-dimension-formula-for-self-similar-measures-on-the-line-September-24-2026/main.pdf`): $\dim_H\mu=\min\{1,h_{RW}/\chi\}$, con $\chi$ la contracción logarítmica media (exponente de Lyapunov).
4. **093 – Log-Sobolev libre de dimensión** (`preprints/A-dimension-free-logarithmic-Sobolev-inequality-for-subgaussian-log-concave-measures-September-23-2026/paper.pdf`): $\mathrm{Ent}_\mu(f^2)\le C a^2\int|Df|^2d\mu$.
5. **139 – Muestreo log-cóncavo** (`preprints/Subpolynomial-query-complexity-for-well-conditioned-log-concave-sampling-September-26-2026/article.pdf`): con $I\preceq\nabla^2V\preceq 2I$ bastan $C_\varepsilon d^\varepsilon$ consultas de gradiente.
6. **083 – Transformada de Hilbert a lo largo de campos Lipschitz** (`preprints/A-uniform-Hilbert-transform-estimate-for-Lipschitz-directions-September-25-2026/main.pdf`): cota $L^2$ uniforme a escalas $\lesssim 1/\mathrm{Lip}(v)$.
7. **145 – Mezcla múltiple de Rokhlin** (`preprints/Rokhlins-multiple-mixing-problem-for-one-transformation-September-23-2026/paper.pdf`).

**Advertencia honesta:** son teoremas de matemática pura, producidos por un modelo y en parte no formalizados (el README admite que pueden tener errores). Ninguno trata redes neuronales; su uso aquí es **analógico/motivacional**. Las hipótesis se justifican además con literatura de ML real citada.

## 2. Cómo construir el modelo apoyado en estas fórmulas

Contexto (de `docs/resultados_v4.md` §16): Dφ es un campo residual $\Psi_{t+1}=\Psi_t+\Delta_\theta(\Psi_t)$ con cabeza local-lineal con compuerta $\Delta=M(q,c)\,q+b(q,c)$ (2590 params), entrenado por destilación de un teacher consolidado (CDT) con *replay de confianza* (familias con residuo de ajuste ≥ τ=1 % excluidas). Resultado: error 1 paso 0.9 %, E33 13/16 DEV y 9/16 CONFIRM (en el umbral), h16 ≈ 0.26/0.22, MLP gated de control equivalente; perturbaciones con factor ≈0.99 (el error crece de forma lineal: **sesgo, no caos**).

### 2.1 Acumulación de error en rollout (Gronwall discreto, literatura estándar)
Si el error de un paso es $\varepsilon$ y la constante de Lipschitz local del paso es $L$:
$$e_{h}\le \varepsilon\sum_{k=0}^{h-1}L^{k}=\varepsilon\frac{L^{h}-1}{L-1}\;\xrightarrow{L\to1}\;h\,\varepsilon .$$
Con $L\approx0.99$ (medido) el régimen es lineal: para que h16 pase el umbral de 5 % se necesita sesgo medio por paso $\lesssim 0.05/16\approx0.3$ %, frente al 0.9 % actual (cálculo, no medición). El exponente de Lyapunov de 148, $\chi=\mathbb E[\log\|J\|]$, es la versión logarítmica de $L$.

### 2.2 Pérdidas
- Un paso (actual): $\mathcal L_1=\mathbb E\|\Psi_t+\Delta_\theta(\Psi_t)-\Psi_{t+1}\|^2/\|\Psi_{t+1}-\Psi_t\|^2$ (forma relativa, como `rel_err`).
- Multi-horizonte (v5, en curso, no tocar): $\mathcal L_{MH}=\sum_{k\in\{1,2,4,8\}}w_k\,\ell(\Phi_\theta^{k}(\Psi_t),\Psi_{t+k})$.
- Sobolev / Jacobiano (Czarnecki et al., 2017, *Sobolev Training for Neural Networks*): $\mathcal L_S=\mathcal L_1+\lambda\,\mathbb E\|J_\theta-J_T\|_F^2$.
- Destilación ponderada continua: $\mathcal L_w=\sum_i w_i\ell_i$, $w_i=\exp(-r_i/\tau)$, con $r_i$ el residuo de ajuste del teacher.

### 2.3 Reward model de proceso (literatura, no está en `openai/math`)
Lightman et al. 2023, *Let's Verify Step by Step* (datos en `openai/prm800k`): la puntuación de una solución es el producto de probabilidades de que cada paso sea correcto,
$$\mathrm{score}(s_{1:n})=\prod_{i=1}^{n}p_\phi(\text{correcto}\mid s_{1:i}),$$
entrenado con entropía cruzada por paso $-\sum_i[y_i\log p_\phi+(1-y_i)\log(1-p_\phi)]$. Trasladado a Dφ: un *verificador por paso* que puntúa cada transición del rollout.

### 2.4 Atención (literatura, Vaswani et al. 2017)
$\mathrm{Attn}(Q,K,V)=\mathrm{softmax}(QK^\top/\sqrt{d_k})V$. **No se recomienda** introducirla: el proyecto mostró que la mejora viene de la cabeza multiplicativa local, y añadir atención confundiría la comparación campo vs MLP a igual capacidad. Se incluye solo por completitud.

## 3. Hipótesis (preregistrables: tuning en `0xC500–0xC503`, DEV `0xA400–0xA40F`, CONFIRM `0xB400–0xB40F`, mismo binario/SHA, sin retune, auditoría queries=0/leakage=0/proceso nuevo, control MLP gated de 2590 params)

Todas parten de `main` (receta §16, E3). Ninguna modifica `exp/predictive-field-v5`; H2 y H6 deben esperar a `docs/comparacion_v4_v5.md`.

### H1 — Sesgo, no varianza, limita h16 (Gronwall lineal)
- **Motivación:** factor de perturbación ≈0.99 y crecimiento lineal (§16).
- **Formulación:** $e_h\approx h\,\|\bar b\|$ con $\bar b=\mathbb E[\Delta_\theta-\Delta^{\star}]$ sobre la órbita.
- **Cambio:** sin correctores por familia; se añade un término genérico $\mathcal L_{bias}=\|\mathbb E_{\text{batch}}[\Delta_\theta-\Delta_T]\|^2$ (λ fijado en tuning) además de $\mathcal L_1$.
- **Predicción falsable:** reducir el error medio con signo (no el RMS) baja `rel h8` y sube acc h16; si sube el RMS sin bajar el sesgo, falla.
- **Experimento mínimo:** E3 vs E3+$\mathcal L_{bias}$, mismo cómputo.
- **Métrica:** acc h16, región E33, sesgo con signo por paso medido sobre trayectoria verdadera.
- **Éxito:** diferencia pareada en acc h16 con IC95 bootstrap >0 en DEV y CONFIRM, sin empeorar acc 1 paso >0.02. **Fallo:** IC incluye 0 en cualquiera de las fases.

### H2 — Pérdida tipo PRM con prefijo confiable (process supervision)
- **Motivación:** PRM (Lightman 2023) supervisa cada paso; en v5 la pérdida multi-horizonte penaliza igual pasos tras un prefijo ya divergido.
- **Formulación:** $\mathcal L_{PRM}=\sum_{k=1}^{8}\Big[\prod_{j<k}\mathbb 1(\mathrm{rel}_j<\tau)\Big]\ell(\Phi^k_\theta\Psi_t,\Psi_{t+k})$ (gradiente solo mientras el prefijo es confiable).
- **Cambio:** máscara de prefijo sobre la receta v5; mismo nº de evaluaciones Dφ que `P2_gated_eqc`.
- **Predicción:** mayor acc h16 que P2 con cómputo igualado.
- **Experimento mínimo:** `P2_gated_eqc` vs `P2_prm_eqc` (tras cerrar v5).
- **Métrica / éxito:** igual que el criterio 2 de `docs/comparacion_v4_v5_preregistro.md` (IC95 >0 en región o h16, ambas fases). **Fallo:** empate o negativo.

### H3 — Escala muestras–precisión en la destilación (analogía con 140)
- **Motivación:** el teorema 140 dice que, con memoria persistente acotada, cada factor 2 de precisión cuesta ~$d$ muestras adicionales ($T=\Omega(d\log 1/\epsilon)$). Dφ es un aprendiz de memoria acotada (2590 params) alimentado por replay.
- **Formulación (hipótesis, no teorema aplicable):** $\log(1/\mathrm{rel\_err}_1)\approx\alpha+\beta\log N_{replay}$ o lineal en $N$; se ajustan ambas y se preregistra cuál se espera (lineal en $N/d$ según 140).
- **Cambio:** solo el tamaño del replay: $N\in\{0.5,1,2,4\}\times$ el actual, pasos Adam fijos.
- **Predicción:** rel_err₁ cae monótonamente; si se satura antes de 2×, el cuello es capacidad/sesgo de parametrización, no datos.
- **Métrica:** rel_err₁, acc h16, captura del teacher.
- **Éxito:** pendiente pareada <0 con IC95 excluyendo 0 en ambas fases. **Fallo:** pendiente ≈0 (saturación) → descarta "más replay".

### H4 — Destilación Sobolev (Jacobiano) motivada por log-Sobolev/Poincaré (093)
- **Motivación:** LSI implica Poincaré $\mathrm{Var}_\mu f\le C a^2\,\mathbb E|\nabla f|^2$: controlar el gradiente controla la variabilidad; E32 mostró σ_max expansivo 1.05–1.14 en isometrías.
- **Formulación:** $\mathcal L_S=\mathcal L_1+\lambda\,\mathbb E\|J_\theta-J_T\|_F^2$, $J_T$ por diferencias finitas del teacher (solo en familias con máscara de confianza activa).
- **Cambio:** término adicional; igual arquitectura.
- **Predicción:** $|\sigma_{max}(J_\theta)-\sigma_{max}(J^{\star})|$ baja y acc h16 sube; el MLP gated con la misma pérdida gana menos (ventaja específica del campo residual, porque $J=I+J_\Delta$).
- **Experimento:** 2×2 {campo, MLP}×{sin, con $\mathcal L_S$}.
- **Éxito:** interacción (campo−MLP) en h16 con IC95>0 en ambas fases. **Fallo:** ganan igual (no es del campo) o nadie gana.

### H5 — Máscara de confianza continua y sensibilidad 1/3 (Brenier, 374)
- **Motivación:** 374 muestra que el mapa óptimo puede moverse como $W_2^{1/3}$: pequeñas impurezas en la distribución objetivo producen cambios de mapa desproporcionados. Eso explica por qué el replay "casi bueno" (reflect con 9–13 % de error) dañaba tanto, y sugiere que la máscara dura τ=1 % puede ser frágil en el borde.
- **Formulación:** $w_i=\exp(-r_i/\tau)$ vs $w_i=\mathbb 1(r_i<\tau)$; además medir la curva $\|\Delta_{\theta_\mu}-\Delta_{\theta_\nu}\|_{L^2}$ vs $W_2(\mu,\nu)$ contaminando el replay con fracción $\eta\in\{0,1,2,4,8\}\%$ de targets ruidosos.
- **Predicción:** (a) peso continuo ≥ máscara dura en CONFIRM; (b) exponente empírico de sensibilidad < 1 (sub-lineal).
- **Éxito:** (a) IC95 ≥ 0 sin empeorar 1 paso, y menor varianza entre seeds; (b) pendiente log-log estimada con IC <1. **Fallo:** (a) negativo; (b) pendiente ≥1 → la analogía no aplica.

### H6 — Consistencia de semigrupo: la ventaja estructural del campo sobre el MLP
- **Motivación:** el campo debería ganar al MLP solo si usa una propiedad que el MLP directo no tenga. Un flujo cumple $\Phi_{2h}=\Phi_h\circ\Phi_h$ (literatura Neural ODE, Chen et al. 2018). El paso medio del campo $\Psi+\tfrac12\Delta$ es definible; para el MLP directo no.
- **Formulación:** $\mathcal L_{SG}=\mathbb E\|\Phi_\theta(\Psi)-\Phi_{\theta,1/2}(\Phi_{\theta,1/2}(\Psi))\|^2$ con $\Phi_{\theta,s}=\Psi+s\,\Delta_\theta(\Psi,s)$ (s como entrada).
- **Predicción:** campo+$\mathcal L_{SG}$ supera al MLP gated en h16 (hoy empatan: 0.26 vs 0.23 DEV).
- **Éxito:** campo−MLP h16 con IC95>0 en DEV y CONFIRM. **Fallo:** IC incluye 0 → la hipótesis "dinámica" sigue sin soporte.

### H7 — Contracción normal aprendida en vez de reproyección (Lyapunov, 148)
- **Motivación:** V1 (reproyección) reduce la sensibilidad a perturbaciones a la mitad (0.99→0.55); 148 formaliza que contracción media $\chi$ controla la dimensión del atractor.
- **Formulación:** $\mathcal L_N=\mathbb E\,[\log\|P_\perp J_\theta P_\perp\|-\log\rho]_+$ con $\rho<1$ y $P_\perp$ proyector normal a la variedad (la misma que usa V1).
- **Predicción:** sin reproyección en inferencia, factor de perturbación h8 ≤0.7 y acc h16 ≥ V1.
- **Éxito:** no inferior a V1 (diferencia pareada IC95 inferior > −0.02) en ambas fases y mejor que V0. **Fallo:** peor que V0 o requiere reproyección.

## 4. Orden sugerido
1) H1 y H3 (diagnósticas y baratas) → 2) H4, H5 → 3) H6, H7 → 4) H2 después de v5. Registrar negativos.

## Referencias
- `openai/math` @ `1581d4c`: `README.md`, `CONTENTS.md`, `history.md`, `lean/ComparatorChallenges/{Brenier,NoiselessRegression,LogConcaveQuery,LipschitzHilbert}.lean`, PDFs citados arriba.
- Lightman et al. (2023) *Let's Verify Step by Step*; `github.com/openai/prm800k`.
- Vaswani et al. (2017) *Attention Is All You Need*.
- Czarnecki et al. (2017) *Sobolev Training for Neural Networks*.
- Chen et al. (2018) *Neural Ordinary Differential Equations*.
- Proyecto: `docs/resultados_v4.md`, `docs/preregistro_v4.md` (main); `docs/plan_v4_dphi_teacher_distillation.md` (exp/field-autonomy-v4); `docs/comparacion_v4_v5_preregistro.md` (exp/predictive-field-v5).
