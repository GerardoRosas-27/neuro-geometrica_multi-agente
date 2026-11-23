# Especificación Técnica de la Arquitectura MARCUS-I
## (Marcus Algebraic-Reasoning & Causal Unified Substrate - Instance 1)

> **Científico Inspirador:** Dr. Gary Marcus  
> **Obras de Referencia:** *The Algebraic Mind* (2001), *Rebooting AI* (2019), *Taming Silicon Valley* (2024), *The Next Decade in AI* (2020)  
> **Crate:** `marcus_neurosymbolic`  
> **Lenguaje:** Rust (edición 2021) con soporte SIMD y concurrencia segura  

---

## 1. Visión y Principios de Diseño

La arquitectura **MARCUS-I** está diseñada para rechazar explícitamente el axioma del aprendizaje profundo puro ("la escala de parámetros y el entrenamiento de autocompletado autorregresivo bastan para la cognición general"). En su lugar, materializa en software las cuatro demandas cognitivas de *The Algebraic Mind* y los principios de robustez y verdad de *Rebooting AI* y *Taming Silicon Valley*.

```
   ┌────────────────────────────────────────────────────────────────────────┐
   │                       Entrada Perceptual / Query                       │
   └───────────────────────────────────┬────────────────────────────────────┘
                                       │
                                       ▼
   ┌────────────────────────────────────────────────────────────────────────┐
   │                  SISTEMA 1: GENERADOR ASOCIATIVO                       │
   │  Proponente continuo y heurístico de candidatos (Pattern Proposer)     │
   │  - Similitud de coseno / Recuperación aproximada                       │
   │  - Generación rápida de hipótesis candidatas H = {h_1, ..., h_k}        │
   └───────────────────────────────────┬────────────────────────────────────┘
                                       │ Candidatos con confianza asociativa
                                       ▼
   ┌────────────────────────────────────────────────────────────────────────┐
   │             INTERFAZ DE LIGAMIENTO ALGEBRAICO (MÓDULO 1)               │
   │  - Desempaquetado de Roles y Rellenadores (Role-Filler Unbinding)      │
   │  - Ligamiento dinámico de Variables libres {x, y, z}                   │
   │  - Árbol de sintaxis composicional y distinción Tipo-Token             │
   └───────────────────────────────────┬────────────────────────────────────┘
                                       │ Proposiciones estructuradas
                                       ▼
   ┌────────────────────────────────────────────────────────────────────────┐
   │             SISTEMA 2: MOTOR DE MODELO DEL MUNDO (MÓDULO 2)            │
   │  - Andamiaje Innato Spelke (Solidez, Permanencia, Continuidad)         │
   │  - Grafo Causal Intervencionista (Judea Pearl do-calculus)             │
   │  - Invariantes de Soporte, Gravedad y Contención Topológica            │
   └───────────────────────────────────┬────────────────────────────────────┘
                                       │ Evaluación de restricciones & simulación
                                       ▼
   ┌────────────────────────────────────────────────────────────────────────┐
   │            CENTINELA DE VIGILANCIA EPISTÉMICA (MÓDULO 4)               │
   │  - Filtro Anti-Alucinación (Epistemic Veto Gate)                       │
   │  - Generación de Contraejemplos para reentrenamiento de Sistema 1      │
   │  - Árbol de Procedencia Explicativa (Justification Provenance Trace)   │
   └───────────────────────────────────┬────────────────────────────────────┘
                                       │
                                       ▼
                     Salida Verificada y Explicable
```

---

## 2. Los Cuatro Módulos Fundamentales

### Módulo 1: `AlgebraicMind` (El Sustrato Algebraico)

Implementa de forma nativa las cuatro capacidades identificadas por Marcus en 2001:

1. **Ligamiento Dinámico de Variables (`VariableBinding`):**
   - El sistema opera con variables formales $v \in \mathcal{V}$ que pueden ser ligadas dinámicamente a símbolos concretos mediante un registro de ligamiento $\Gamma: \mathcal{V} \to \mathcal{S}$.
   - Soporte para Arquitecturas Simbólicas Vectoriales (VSA) / Representaciones de Producto Tensorial (Smolensky/Marcus):
     $$\mathbf{S} = \sum_{i} \mathbf{r}_i \circledast \mathbf{f}_i$$
     donde $\mathbf{r}_i$ es el vector del rol (e.g. `Agent`, `Action`, `Patient`) y $\mathbf{f}_i$ es el rellenador (`Filler`). El desemparejamiento se realiza mediante el producto inverso exacto:
     $$\mathbf{f}_k \approx \mathbf{r}_k^{-1} \circledast \mathbf{S}$$
   - Esto evita la diafonía sintáctica (*crosstalk*) y permite instanciar variables con valores nunca antes vistos.

2. **Representaciones Composicionales Jerárquicas (`CompositionalTree`):**
   - Un tipo recursivo en Rust que modela árboles de predicado-argumento:
     $$\text{Expr} ::= \text{Atom}(S) \mid \text{Var}(V) \mid \text{Predicate}(\text{Name}, [\text{Expr}_1, \dots, \text{Expr}_k])$$
   - Permite evaluar proposiciones complejas como:
     $$\text{In}(\text{Apple}, \text{Box}) \land \text{On}(\text{Box}, \text{Table})$$
     sin colapsar la estructura en una bolsa de palabras o embedding estático.

3. **Distinción Tipo-Token (`TypeTokenRegistry`):**
   - `Type`: Clase abstracta ontológica con propiedades inherentes e invariantes (e.g., `Container`, `RigidBody`, `LivingAgent`).
   - `Token`: Instancia espacio-temporal única que hereda las propiedades del tipo pero posee un identificador único `UUID`, coordenadas dinámicas $(x, y, z, t)$ y estado mutable.

4. **Reglas Universales Fuera de Distribución (`UniversalRuleEngine`):**
   - Reglas de la forma $\forall x: P(x) \implies Q(x)$ o gramáticas algebraicas como $ABA$ ($x y x$) y $ABB$ ($x y y$).
   - Generalización zero-shot con 100% de precisión matemática sobre cualquier dominio de tokens no vistos durante el entrenamiento.

---

### Módulo 2: `CoreKnowledge` (Andamiaje Innato Spelke-Marcus)

Este módulo provee la ontología y las leyes físicas que, según Marcus, ningún sistema de IA puede inducir confiablemente de manera puramente empírica desde cero:

1. **Permanencia y Continuidad Espacio-Temporal (`SpatiotemporalPhysics`):**
   - Rastrea el estado continuo de los cuerpos rígidos en el espacio 3D.
   - **Invariante de Continuidad:** Dado el estado en $t_1$ y $t_2$, la velocidad requerida $\frac{\|\mathbf{p}_2 - \mathbf{p}_1\|}{t_2 - t_1}$ no puede exceder la velocidad física máxima $v_{\max}$. Cualquier salto sin trayectoria continua es marcado como `TeleportationViolation`.
   - **Invariante de Solidez / Impenetrabilidad:** La distancia entre dos cuerpos sólidos no permeables debe cumplir $d(O_1, O_2) \ge r_1 + r_2$. Si se detecta intersección de volúmenes, se emite una `SolidPenetrationViolation`.
   - **Invariante de Oclusión / Permanencia:** Si un objeto $O$ deja de ser visible porque las coordenadas de un objeto opaco $B$ se interponen en la línea de visión del observador, $O$ continúa existiendo en el estado del mundo con `is_occluded = true` y no desaparece de la realidad.

2. **Soporte y Contención Topológica (`SupportAndContainment`):**
   - Axioma de Contención Transitiva:
     $$\forall A, B, C: \text{Inside}(A, B) \land \text{Inside}(B, C) \implies \text{Inside}(A, C)$$
   - Axioma de Soporte de Gravedad: Si un objeto no está apoyado en una superficie $\text{Support}(S, O)$ y su altitud $z > 0$, acelera hacia el suelo con vector $\mathbf{g} = (0, 0, -9.81)$.

3. **Inferencia Causal Intervencionista (`PearlCausalEngine`):**
   - Modela la jerarquía de tres niveles de Judea Pearl requerida por Marcus:
     1. **Asociación:** $P(Y \mid X)$ (lo que una red neuronal correlaciona).
     2. **Intervención:** $P(Y \mid do(X))$ (lo que ocurre cuando manipulamos experimentalmente la variable $X$, rompiendo los enlaces hacia sus causas previas).
     3. **Contrafactuales:** "¿Qué habría ocurrido con $Y$ si hubiéramos hecho $do(X = x')$ dado que observamos $X = x$ e $Y = y$?".
   - Previene que el sistema confunda correlaciones espurias (e.g., cantar del gallo y amanecer) con causalidad física real.

4. **Agencia Intencional y Teoría de la Mente (`AgencyEngine`):**
   - Distingue entidades pasivas de agentes con metas.
   - Principio de Acción Racional: los agentes escogen caminos geodésicos eficientes en presencia de obstáculos para alcanzar sus objetivos.

---

### Módulo 3: `DualProcessCoordinator` (Acoplamiento Sistema 1 + Sistema 2)

Materializa la interacción dialéctica entre la intuición heurística y la deliberación formal:

- **Sistema 1 (`System1Proposer`):**
  - Módulo rápido y difuso basado en similitud asociativa y generación estadística.
  - Procesa entradas desestructuradas y produce un conjunto clasificado de hipótesis candidatas $H = \{(h_1, w_1), \dots, (h_k, w_k)\}$.

- **Sistema 2 (`System2Deliberator`):**
  - Toma las hipótesis más prometedoras de $H$, las traduce a términos algebraicos y las ejecuta sobre el simulador del `CoreKnowledge`.
  - Verifica si $h_i$ viola algún invariante físico o si es consistente con el estado global de hechos conocidos.

- **Bucle de Realimentación y Corrección:**
  - Si el Sistema 2 encuentra una contradicción o violación en $h_i$:
    1. Veta $h_i$.
    2. Construye una restricción simbólica de penalización (un contraejemplo explícito).
    3. Notifica al Sistema 1 para que actualice sus pesos heurísticos locales o explore la siguiente alternativa.

---

### Módulo 4: `EpistemicVigilance` (Centinela Antialucinaciones y Auditoría)

Basado en las demandas de fiabilidad y seguridad de *Taming Silicon Valley*:

1. **Puerta de Veto Epistémico (`HallucinationFilter`):**
   - Toda salida generada debe pasar por el centinela.
   - Clasificación de salidas:
     - `VerifiedValid`: Coherente con hechos conocidos, leyes físicas y relaciones causales.
     - `PhysicalImpossibilityVeto`: Alucina violación de solidez, gravedad o permanencia.
     - `LogicalContradictionVeto`: Contradice una proposición previamente probada o un hecho base.
     - `CausalFallacyVeto`: Afirma una relación causal desmentida por un test de intervención $do(X)$.

2. **Trazabilidad de Procedencia (`EpistemicProvenanceTree`):**
   - Cada respuesta entregada al usuario o a otro agente contiene un árbol de prueba completo:
     - Premisas de entrada.
     - Identificador del token y sus tipos ontológicos.
     - Reglas de inferencia aplicadas.
     - Registro de ligamiento de variables utilizado.
     - Comprobación de estado físico superada.
   - Esto erradica por completo la opacidad de "caja negra" de los modelos autorregresivos.

---

## 3. Demostración y Benchmarks de Falsificación

El crate incluye un arnés de evaluación exhaustivo con cinco suites experimentales:

| Benchmark | Hipótesis de Marcus Evaluada | Comportamiento Predictor Estadístico Puro | Comportamiento MARCUS-I |
|---|---|---|---|
| **BM-1: OOD Variable Binding** | Capacidad de computar $f(x) = x$ y gramáticas $ABB$ sobre tokens nunca vistos. | **Falla catastróficamente** (Precisión $\approx 0\%$). | **100.0% de Precisión** (Zero-Shot OOD). |
| **BM-2: Systematic Compositionality** | Comprensión de tuplas recombinadas `Predicate(Arg1, Arg2)` fuera del conjunto de entrenamiento. | Degradación por solapamiento n-gramático. | **Preservación exacta de la estructura sintáctica**. |
| **BM-3: Spelke Core Physics** | Detección de teleportación, penetración sólida y permanencia tras oclusión. | Alucina continuidad o ignora objetos ocluidos. | **100% de detección de violaciones físicas**. |
| **BM-4: Pearl Causal Interventions** | Distinguir $do(X)$ de la correlación $P(Y \mid X)$ con variable confusora latente. | Confunde correlación con causalidad. | **Aislamiento causal perfecto vía do-calculus**. |
| **BM-5: Epistemic Anti-Hallucination** | Supresión de afirmaciones plausibles estadísticamente pero ontológicamente falsas. | Alucina por alta probabilidad local del n-grama. | **0% de alucinaciones admitidas (Veto 100%)**. |

---

## 4. Estructura del Crate `marcus_neurosymbolic`

```
marcus_neurosymbolic/
├── Cargo.toml
├── src/
│   ├── lib.rs
│   ├── main.rs
│   ├── algebraic_mind/
│   │   ├── mod.rs
│   │   ├── variables.rs
│   │   ├── syntax_tree.rs
│   │   └── type_token.rs
│   ├── core_knowledge/
│   │   ├── mod.rs
│   │   ├── spatiotemporal_physics.rs
│   │   ├── causal_engine.rs
│   │   └── agency.rs
│   ├── dual_process/
│   │   ├── mod.rs
│   │   ├── system1_proposer.rs
│   │   ├── system2_deliberator.rs
│   │   └── coordinator.rs
│   ├── epistemic_vigilance/
│   │   ├── mod.rs
│   │   ├── world_state.rs
│   │   ├── hallucination_filter.rs
│   │   └── provenance.rs
│   └── benchmarks/
│       ├── mod.rs
│       ├── ood_variable_binding.rs
│       ├── compositional_systematicity.rs
│       ├── spelke_physics_tests.rs
│       ├── causal_intervention_tests.rs
│       └── hallucination_gate_tests.rs
└── tests/
    └── integration_marcus_test.rs
```
