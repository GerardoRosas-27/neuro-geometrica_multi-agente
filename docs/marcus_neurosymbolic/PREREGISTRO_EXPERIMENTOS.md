# Protocolo de Pre-registro Experimental: MARCUS-I
## Validación Falsable de las Hipótesis Cognitivas de Gary Marcus

> **Proyecto:** Neuro-Geométrica Multi-Agente  
> **Línea de Investigación:** Arquitectura Neuro-Simbólica MARCUS-I  
> **Fecha de Pre-registro:** Noviembre 2026  
> **Estado:** PRE-REGISTRADO (Previo a la ejecución de confirmación)  
> **Objetivo:** Establecer criterios matemáticos y experimentales estrictos y falsables para contrastar la arquitectura neuro-simbólica basada en Gary Marcus contra un modelo puramente asociativo/estadístico (análogo al paradigma LLM/conexión clásica).

---

## 1. Hipótesis Científicas Pre-registradas

### Hipótesis H1: Generalización Zero-Shot Fuera de Distribución en Operaciones sobre Variables
* **Formulación Teórica (Marcus 2001):** Un sistema cognitivo que implementa operaciones algebraicas sobre variables puede generalizar reglas abstractas (e.g. la regla de identidad $f(x) = x$ o el patrón gramatical $ABB$ vs $ABA$) a símbolos completamente fuera del soporte de entrenamiento ($x \notin \mathcal{D}_{\text{train}}$) con precisión perfecta, mientras que un modelo puramente asociativo (n-grama / MLP / red autorregresiva) degrada su rendimiento a nivel de azar o cero al encontrar tokens OOD.
* **Métrica:**
  $$\text{Acc}_{\text{OOD}} = \frac{1}{|\mathcal{D}_{\text{OOD}}|} \sum_{i \in \mathcal{D}_{\text{OOD}}} \mathbb{I}(\hat{y}_i = y_i)$$
* **Criterio de Éxito:** $\text{Acc}_{\text{OOD}}(\text{MARCUS-I}) = 1.00$ (100%).
* **Criterio de Falsificación:** Si $\text{Acc}_{\text{OOD}}(\text{MARCUS-I}) < 0.95$, la hipótesis H1 queda falsificada.

---

### Hipótesis H2: Sistematicidad e Invarianza Composicional Jerárquica
* **Formulación Teórica (Marcus 2001, Fodor & Pylyshyn 1988):** Las estructuras semánticas y sintácticas están compuestas por relaciones jerárquicas no conmutativas. La arquitectura debe:
  1. Distinguir rigurosamente $P(a, b)$ de $P(b, a)$ (e.g., `Contiene(Caja, Gato)` vs. `Contiene(Gato, Caja)`).
  2. Evaluar recombinaciones composicionales no vistas sin diafonía vectorial (*crosstalk*).
* **Métrica:**
  $$\Delta_{\text{asimetría}} = | \text{Score}(P(a, b)) - \text{Score}(P(b, a)) |$$
  $$\text{Acc}_{\text{recomb}} = \text{Precisión de evaluación en pares de entidades recombinadas}$$
* **Criterio de Éxito:** $\text{Acc}_{\text{recomb}}(\text{MARCUS-I}) = 1.00$ y $\Delta_{\text{asimetría}} > 0.90$.
* **Criterio de Falsificación:** Si $\text{Acc}_{\text{recomb}} < 0.95$ o se detecta colapso simétrico, H2 queda falsificada.

---

### Hipótesis H3: Conservación Ontológica y Física Núcleo de Spelke
* **Formulación Teórica (Spelke 1990, Marcus 2019):** El conocimiento núcleo de física no puede depender de la co-ocurrencia estadística superficial. La arquitectura debe mantener un estado persistente del mundo que detecte de forma determinista:
  1. Violaciones de solidez/impenetrabilidad (dos cuerpos ocupando el mismo espacio).
  2. Violaciones de continuidad (teleportación espacio-temporal o desaparición mágica).
  3. Violaciones de permanencia del objeto (olvido de objetos tras oclusión).
* **Métrica:**
  $$\text{Recall}_{\text{violación}} = \frac{\text{Violaciones Detectadas}}{\text{Total de Eventos Físicamente Imposibles}}$$
  $$\text{FPR} = \frac{\text{Falsas Alarmas en Eventos Físicos Válidos}}{\text{Total de Eventos Válidos}}$$
* **Criterio de Éxito:** $\text{Recall}_{\text{violación}} = 1.00$ y $\text{FPR} = 0.00$.
* **Criterio de Falsificación:** Si $\text{Recall}_{\text{violación}} < 0.95$ o $\text{FPR} > 0.05$, H3 queda falsificada.

---

### Hipótesis H4: Inferencia Causal frente a Correlación Espuria (Pearl-Marcus)
* **Formulación Teórica (Marcus & Davis 2019, Pearl 2000):** Ante una estructura causal con confusión latente $Z \to X, Z \to Y$ donde $X$ e $Y$ están correlacionados estadísticamente pero $X$ no causa $Y$:
  1. El predictor puramente observacional predice erróneamente un cambio en $Y$ cuando se observa $X$.
  2. El motor causal MARCUS-I, al aplicar la intervención $do(X = x)$, mutila el grafo y predice correctamente que $P(Y \mid do(X)) = P(Y)$, demostrando ausencia de efecto causal.
* **Métrica:**
  $$\text{Error}_{\text{causal}} = | \hat{\mathbb{E}}[Y \mid do(X)] - \mathbb{E}_{\text{ground\_truth}}[Y \mid do(X)] |$$
* **Criterio de Éxito:** $\text{Error}_{\text{causal}}(\text{MARCUS-I}) < 10^{-4}$ mientras que el predictor asociativo comete un error estadístico significativo ($> 0.30$).
* **Criterio de Falsificación:** Si MARCUS-I confunde correlación con causalidad en intervenciones, H4 queda falsificada.

---

### Hipótesis H5: Vigilancia Epistémica y Supresión de Alucinaciones
* **Formulación Teórica (Marcus 2024 - *Taming Silicon Valley*):** Una arquitectura segura y confiable debe contar con un centinela epistémico que vete cualquier propuesta generada por el Sistema 1 que contradiga hechos verificados, viole leyes físicas o invente relaciones no fundamentadas, generando un árbol de justificación auditable.
* **Métrica:**
  $$\text{Tasa de Alucinación Admitida} = \frac{\text{Alucinaciones Aprobadas}}{\text{Total de Estímulos Adversarios}}$$
  $$\text{Completitud de Trazabilidad} = \frac{\text{Decisiones con Árbol de Procedencia}}{\text{Total de Decisiones Emitidas}}$$
* **Criterio de Éxito:** Tasa de Alucinación = $0.00\%$ y Completitud de Trazabilidad = $100.0\%$.
* **Criterio de Falsificación:** Si se aprueba cualquier alucinación física/lógica ($> 0\%$), H5 queda falsificada.

---

## 2. Conjuntos de Datos Sintéticos y Partición

Se estructuran dos bancos de pruebas totalmente independientes:
1. **DEV-SET (Desarrollo y calibración):** 500 instancias que cubren los casos base canónicos.
2. **CONFIRM-SET (Prueba ciega fuera de distribución):** 1,000 instancias generadas con semillas aleatorias fijas utilizando símbolos, trayectorias espaciales y relaciones causales completamente disjuntas del DEV-SET.

---

## 3. Protocolo de Ejecución

1. Compilar el crate `marcus_neurosymbolic` en modo release (`--release`).
2. Ejecutar la batería de tests unitarios e integrados con `cargo test`.
3. Ejecutar el arnés de benchmark formal mediante `cargo run --release --bin marcus_benchmark`.
4. Registrar los resultados en `docs/marcus_neurosymbolic/RESULTADOS_BENCHMARK.md`.
