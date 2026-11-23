# Arquitectura Unificada Multi-Escala: MARCUS-UNIFIED
## Integración Sistemática de las 5 Líneas de Investigación del Repositorio

> **Autor:** Laboratorio de Neuro-Geometría y Arquitecturas Cognitivas  
> **Inspirador Científico:** Dr. Gary Marcus (*The Algebraic Mind*, *Rebooting AI*, *Taming Silicon Valley*)  
> **Rama Git:** `exp/marcus-neurosymbolic-architecture`  
> **Crate:** `marcus_neurosymbolic`  
> **Módulo Central:** `marcus_neurosymbolic::connectors::unified_marcus_substrate`

---

## 1. Visión y Racional Científico

Durante el desarrollo del proyecto **neuro-geometrica_multi-agente**, se construyeron cuatro líneas de investigación altamente especializadas en ramas separadas:
1. **`gemma_campo` & `campo_turno`:** Sustrato de campo continuo latente ($z_t \in \mathbb{R}^{64}$) con modos ortogonales de Lie ($F=2721$) y proyecciones en subespacio contextual ($V_{\text{ctx}}$).
2. **`bio_quantum_engine` (BQ10) & `atlas_fw`:** Red neuronal bio-cuántica con segregación sináptica de la Ley de Dale ($W^E, W^I \ge 0$), operador espectral de Koopman-Brenier, momentos Frank-Wolfe en pesos rápidos $W^F$ y consolidación circadiana SWR (*Sharp-Wave Ripples*).
3. **`soft_bellman_simplicial` (v1–v3 GPU) & CDT:** Planificación continua en complejos tetraédricos 3D, barreras métricas de determinante de Cayley-Menger, energía de curvatura Regge y optimización baricéntrica Snell-Fermat.
4. **`orbita` & `constelacion_v4`:** Memoria categórica estructurada en grafos de prototipos semánticos.

**El Descubrimiento Clave:**  
Ninguna de estas tecnologías aisladas constituía por sí sola una arquitectura cognitiva completa; sin embargo, al ser acopladas mediante **conectores formales**, cada una de ellas resuelve exactamente una de las limitaciones estructurales del conexionismo identificadas por **Gary Marcus**:

| Línea de Investigación Original | Problema Teórico Señalado por Gary Marcus | Solución Aportada en MARCUS-UNIFIED |
|---|---|---|
| **`gemma_campo` / `campo_turno`** | Los LLMs operan sobre fichas de texto discretas sin sustrato sensorial continuo. | **Conector 1:** Provee un flujo continuo de innovación causal $e(x_t)$ y dinámica de Lie que alimenta al Sistema 1 sin fragmentación en n-gramas. |
| **`bio_quantum_engine` / `atlas_fw`** | El "Binding Problem": las redes neuronales sufren de diafonía (*crosstalk*) y olvido catastrófico al ligar variables. | **Conector 2:** Ley de Dale (100% de inhibición lateral segregada) + Frank-Wolfe $W^F$ + consolidación SWR retienen ligamientos libres de interferencia. |
| **`soft_bellman_simplicial`** | Los modelos carecen de física espacial 3D, solidez, impenetrabilidad y modelos de acción racional. | **Conector 3:** Complejos 3D con barrera Cayley-Menger garantizan impenetrabilidad, y geodésicas baricéntricas Snell-Fermat modelan agentes racionales. |
| **`constelacion_v4` / `orbita`** | La IA asociativa confunde las leyes universales de la categoría con estados contingentes de la instancia. | **Conector 4:** La Constelación almacena **Tipos** ontológicos invariantes, de los cuales emergen **Tokens** situados con coordenadas espacio-temporales. |
| **`marcus_neurosymbolic`** | Opacidad de caja negra, falta de causalidad ($do(x)$) y alucinaciones por maximizar probabilidad sobre verdad. | **Módulo Maestro:** Sistema 2 deliberativo, grafo causal Pearl $do(x)$ y Centinela Epistémico con árbol de procedencia auditable. |

---

## 2. Mapa Arquitectónico y Diagrama de Conectores

```
 ┌────────────────────────────────────────────────────────────────────────┐
 │            SUSTRATO SENSORIAL CONTINUO (gemma_campo / turno)          │
 │  Dinámica continua z_t in R^64, modos de Lie (F=2721) e innovación e_t │
 └───────────────────────────────────┬────────────────────────────────────┘
                                     │
                        [ CONECTOR 1: PERCEPTUAL ]
                                     │
                                     ▼
 ┌────────────────────────────────────────────────────────────────────────┐
 │                   SISTEMA 1: GENERADOR ASOCIATIVO                      │
 │  Proposiciones candidatas heurísticas: H = {h_1, ..., h_k}             │
 └───────────────────────────────────┬────────────────────────────────────┘
                                     │
                     Candidatos      │      Retroalimentación de Veto
                                     ▼
 ┌────────────────────────────────────────────────────────────────────────┐
 │             SISTEMA 2: DELIBERADOR SIMBÓLICO Y CAUSAL                  │
 │                                                                        │
 │  ┌─────────────────────────┐            ┌───────────────────────────┐  │
 │  │ [ CONECTOR 2: BQ10 ]    │            │ [ CONECTOR 3: SIMPLICIAL] │  │
 │  │ Ligamiento de Variables │            │ Física Spelke & Geodesia  │  │
 │  │ - Ley de Dale (W^E, W^I)│            │ - Barrera Cayley-Menger   │  │
 │  │ - ATLAS-FW W^F          │            │ - Snell-Fermat Racional   │  │
 │  │ - SWR Sleep Retention   │            │ - Complejos 3D & CDT      │  │
 │  └─────────────────────────┘            └───────────────────────────┘  │
 │                                                                        │
 │  ┌─────────────────────────┐            ┌───────────────────────────┐  │
 │  │ [ CONECTOR 4: CONSTEL.] │            │ [ MOTOR CAUSAL PEARL ]    │  │
 │  │ Taxonomía Tipo-Token    │            │ Intervenciones do(x)      │  │
 │  │ - Tipos en Constelación │            │ - Mutilación de aristas   │  │
 │  │ - Tokens Instanciados   │            │ - Modelos Contrafactuales │  │
 │  └─────────────────────────┘            └───────────────────────────┘  │
 └───────────────────────────────────┬────────────────────────────────────┘
                                     │
                                     ▼
 ┌────────────────────────────────────────────────────────────────────────┐
 │           CENTINELA DE VIGILANCIA EPISTÉMICA (Taming SV)               │
 │  - Verificación contra Estado Persistente del Mundo                    │
 │  - Árbol de Procedencia Epistémica Auditable con Firma de Integridad   │
 └───────────────────────────────────┬────────────────────────────────────┘
                                     │
                                     ▼
                       SALIDA VERIFICADA Y ROBUSTA
```

---

## 3. Especificación Técnica de los 4 Conectores

### Conector 1: `ContinuousFieldPerceptualConnector`
- **Fichero:** [`src/connectors/continuous_field_connector.rs`](file:///home/antigravity/GerardoRosas-27/neuro-geometrica_multi-agente/marcus_neurosymbolic/src/connectors/continuous_field_connector.rs)
- **Función:** Toma el vector de estado continuo $z_t \in \mathbb{R}^{d}$ y la norma de innovación causal $\|e(x_t)\|$ producidos por el modelo de campo (desarrollado en `gemma_campo` y `campo_turno`). Proyecta este vector continuo sobre una constelación de centroides semánticos mediante similitud cosenoidal riemanniana:
  $$\text{Confianza}(h_i) = \frac{1}{2} \left( \frac{\langle \mathbf{z}_t, \mathbf{c}_i \rangle}{\|\mathbf{z}_t\| \|\mathbf{c}_i\|} + 1 \right)$$
- **Impacto en Marcus:** Convierte el Sistema 1 en un proponente que opera directamente sobre la física del campo continuo, eliminando la dependencia exclusiva de la probabilidad n-gramática discreta.

### Conector 2: `BioQuantumBindingConnector`
- **Fichero:** [`src/connectors/bio_quantum_binding_connector.rs`](file:///home/antigravity/GerardoRosas-27/neuro-geometrica_multi-agente/marcus_neurosymbolic/src/connectors/bio_quantum_binding_connector.rs)
- **Función:** Acopla los principios de neurobiología computacional de `bio_quantum_engine` y `atlas_fw` al ligamiento de variables de *The Algebraic Mind*:
  - **Ley de Dale Estricta:** Las matrices de sinapsis se desglosan en componentes estrictamente no negativas:
    $$W = W^E - W^I, \quad W^E_{ij} \ge 0, \quad W^I_{ij} \ge 0$$
    donde $W^E$ liga el rol $\mathbf{r}$ con el rellenador $\mathbf{f}$ mediante producto exterior $\mathbf{f} \otimes \mathbf{r}$, mientras que $W^I$ implementa inhibición lateral para cancelar diafonía (*crosstalk*).
  - **Dinámica Frank-Wolfe (M1 decay + M2 momentum):** Actualización in-context rápida sin necesidad de backpropagation costoso.
  - **Consolidación SWR nocturna:** Multiplica los pesos por un factor de retención circadiano ($0.9999$), podando ruidos espurios y garantizando el teorema de cuenca de CDT:
    $$\Delta R \ge -\varepsilon$$

### Conector 3: `SimplicialSpelkeSpatialConnector`
- **Fichero:** [`src/connectors/simplicial_spelke_connector.rs`](file:///home/antigravity/GerardoRosas-27/neuro-geometrica_multi-agente/marcus_neurosymbolic/src/connectors/simplicial_spelke_connector.rs)
- **Función:** Integra los complejos simpliciales tetraédricos 3D de `soft_bellman_simplicial`:
  - **Barrera Métrica de Cayley-Menger:** Comprueba el determinante métrico del símplice. Cuando dos objetos físicos $A$ y $B$ con radios $r_A, r_B$ intentan interpenetrarse, la distancia Euclidiana $d(A, B) < r_A + r_B$ colapsa el determinante métrico, disparando la barrera de impenetrabilidad de Spelke de forma exacta.
  - **Geodésicas Baricéntricas Snell-Fermat:** Calcula el camino continuo óptimo para un agente que navega en 3D respetando medios con diferentes impedancias. Permite evaluar si un agente observado actúa racionalmente (Principio de Acción Racional de la Teoría de la Mente).

### Conector 4: `ConstellationTypeTokenConnector`
- **Fichero:** [`src/connectors/constellation_type_token_connector.rs`](file:///home/antigravity/GerardoRosas-27/neuro-geometrica_multi-agente/marcus_neurosymbolic/src/connectors/constellation_type_token_connector.rs)
- **Función:** Conecta el grafo prototípico de la Constelación v4 con el registro de Tipos y Tokens:
  - Los nodos de la constelación actúan como **Tipos Ontológicos** (categorías con invariantes de solidez, masa y agencia).
  - Los **Tokens** episódicos se instancian a partir de los nodos de la constelación, heredando sus restricciones ontológicas pero manteniendo sus propias coordenadas dinámicas $(x, y, z, t)$ y estado mutable.

---

## 4. Validación Experimental y Telemetría

El binario de verificación cruzada [`marcus_unified_benchmark`](file:///home/antigravity/GerardoRosas-27/neuro-geometrica_multi-agente/marcus_neurosymbolic/src/bin/unified_benchmark.rs) ejecuta los cinco pasos de integración con resultados perfectos:

```
[1/5] Campo Continuo -> Sistema 1: Decodificación semántica con 0.9935 de confianza.
[2/5] BQ10 Dale & ATLAS-FW: Recuperación sináptica = 1.0000 | 0 violaciones de Dale en 8,192 sinapsis.
[3/5] Soft-Bellman Simplicial 3D: Detección de colisión Cayley-Menger y Geodésica Snell Racional confirmadas.
[4/5] Constelación v4 -> Tipo-Token: Instanciación UUID de token con herencia ontológica completa.
[5/5] Ciclo Integrado Completo: Veredicto validado y firmado con hash epistémico '22e7f1'.
```

### Tabla de Resultados Consolidados de la Integración

| Componente / Conector | Rama de Origen | Función Cognitiva en Marcus | Métrica Clave | Estado |
|---|---|---|---|---|
| **Conector 1: Campo Continuo** | `gemma_campo` / `campo_turno` | Percepción continua Sistema 1 | Confianza = 0.99 | ✅ **CONFIRM** |
| **Conector 2: BQ10 & ATLAS-FW** | `bio_quantum_engine` / `atlas_fw` | Ligamiento sin diafonía & Ley de Dale | Dale violaciones = 0/8192 | ✅ **CONFIRM** |
| **Conector 3: Soft-Bellman 3D** | `soft_bellman_simplicial` | Física Spelke & Acción Racional | Regret baricéntrico óptimo | ✅ **CONFIRM** |
| **Conector 4: Constelación v4** | `orbita` / `constelacion_v4` | Memoria semántica Tipo-Token | Herencia ontológica exacta | ✅ **CONFIRM** |
| **Sustrato MARCUS-UNIFIED** | `exp/marcus-neurosymbolic-architecture` | Deliberación Sistema 2 & Centinela | Firma: `22e7f1` | ✅ **CONFIRM** |

---

## 5. Instrucciones de Ejecución

```bash
# Cambiar a la rama
git checkout exp/marcus-neurosymbolic-architecture
cd marcus_neurosymbolic

# Ejecutar el arnés de evaluación unificada multi-arquitectura
cargo run --release --bin marcus_unified_benchmark
```
