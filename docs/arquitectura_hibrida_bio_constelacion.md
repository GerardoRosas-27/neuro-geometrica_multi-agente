# Arquitectura Híbrida: Bio-Constelación (Neurobiológica + Geométrica)

Esta especificación describe la arquitectura híbrida **Bio-Constelación**, diseñada para integrar lo mejor de dos paradigmas complementarios desarrollados en este repositorio:
1. **Constelación v4** (`origin/exp/constelacion-v4`): Formulación geométrica de operadores de Koopman ortogonales escalados, predicción residual puntual equivariante y adaptación rápida de 128 iteraciones.
2. **Neocortex (`BioBrain`)** (`origin/exp/neocortex-bioinspirado`): Microcircuito cortical multicompartimental biofísicamente plausible, 100% compliant con la Ley de Dale ($W^E \ge 0, W^I \ge 0$), plasticidad sináptica conductual BTSP (Bittner et al., 2017), neuromodulación cuádruple (DA, ACh, NE, 5-HT) y memoria episódica hipocampal CLS con consolidación en sueño SWR.

La arquitectura se encuentra completamente aislada e implementada en el crate de Rust puro `bio_constelacion/` y su runner equivalente en Python `bio_constelacion/run_hybrid.py`, en la rama dedicada `exp/hibrido-constelacion-neocortex`.

---

## 1. Motivación y Comparativa: ¿Por qué un Híbrido?

| Dimensión de Diseño | Constelación v4 (`exp/constelacion-v4`) | Neocortex BioBrain (`exp/neocortex-bioinspirado`) | **Bio-Constelación (Híbrido)** |
| :--- | :--- | :--- | :--- |
| **Cumplimiento Ley de Dale** | ❌ **Violada (100%)** — Pesos sin signo algebraico, sin neuronas excitatorias/inhibitorias separadas. |  **100% Compliant** — $W_{\text{eff}} = W^E - W^I$ con $W^E \ge 0, W^I \ge 0$. |  **100% Compliant** — Todas las proyecciones lentas $W^P$ y campos rápidos $W^F$ segregan E/I. |
| **Autonomía del Prior (E33)** | ⚠️ **Dependiente** — Requiere cargar un archivo JSON externo congelado de ÓRBITA para no colapsar en $h=1$. | ❌ **Sin Prior Explícito** — No disponía de hiperred tálamo-cortical para tareas de un solo paso. |  **Autónomo** — Prior Tálamo-Cortical nativo de 2 capas Dale que mapea el contexto $c \in \mathbb{R}^{10}$ al operador $W^{\text{prior}}$. |
| **Coordenadas y Dinámica** |  **Residual puntual equivariante**: $\hat{q}_p(t+1) = q_p(t) + \Delta \hat{q}_p$. | ❌ **No-residual plana**: predecía el estado completo de 12D acumulando deriva en $h=16$. |  **Residual puntual equivariante** sobre base cortical L2/3 $\phi(q_p) \in \mathbb{R}^{10}$. |
| **Mecanismo de Adaptación** | ⚠️ **Gradiente algebraico** sin base biofísica ($t_f = 128$ pasos). Colapsa sin él (`no_fast` acc = 0%). |  **Bio-BTSP** (Bittner et al., 2017) con mesetas de calcio apicales y normalización PV+. |  **Dendritic BTSP Field** ($W^F$) coordinado con ciclos anidados Theta-Gamma ($N_{\text{cycles}} = 24$). |
| **Operador Generativo** |  **Bloques Koopman $2 \times 2$ contractivos**: $K_j = \rho_j R(\omega_j)$ con $\rho_j \le 1.0$. | ⚠️ Dinámica recurrente sin acotamiento espectral estricto. |  **Dendritic Koopman Column** en Capa L5: acopla bloques Koopman con levantamiento no lineal monótono $\psi(K \mu)$. |
| **Memoria a Largo Plazo** | ❌ Búfer plano CDT sin consolidación ni separación de patrones. |  **Hipocampo CLS** (Giro Dentado $k$-WTA + CA3 Hopfield continuo + Sueño SWR). |  **Hipocampo CLS + CDT Buffer**: almacena operadores $W^F$ en CA3 y consolida a $W^P$ en sueño NREM. |

---

## 2. Fundamentos Neurobiológicos Implementados

### 2.1. Neurona Piramidal de 3 Compartimentos y Disparo BAC
Inspirada en el modelo biofísico de Matthew Larkum (*Science*, 1999; *Trends in Neurosciences*, 2013):
- **Compartimento Basal ($\mathbf{v}_B$):** Recibe las señales sensoriales *bottom-up* del punto $q_p(t)$.
- **Compartimento Apical ($\mathbf{v}_A$):** Recibe señales contextuales y proyecciones *top-down* desde L5, el tálamo y el hipocampo.
- **Compartimento Somático ($\mathbf{u}_S$):** Integra el flujo basal y apical bajo el control de ganancia divisivo de interneuronas parvalbúmina (PV+).
- **Mecanismo BAC (Backpropagating Action Potential Activated $\text{Ca}^{2+}$ Spike):** Si una ráfaga somática coincide en una ventana de ~15 ms con una despolarización apical, se desencadena una espiga de meseta de calcio de larga duración en la zona de iniciación apical, abriendo la ventana de plasticidad asociativa.

### 2.2. Microcircuito Interneuronal Cortical (PV+, SST+, VIP+)
Implementado según la arquitectura de Sacramento et al. (2018) y Pfeffer et al. (2013):
- **Interneuronas PV+ (Parvalbumin):** Proporcionan inhibición perisomática rápida con normalización divisiva de conductancia:
  $$\Delta u_S = \frac{I_{\text{soma}}}{1 + \gamma_{\text{PV}} \cdot g_{\text{PV}}}$$
- **Interneuronas SST+ (Martinotti / Somatostatin):** Inhiben el penacho apical, prediciendo y cancelando la entrada *top-down* esperada para computar el error de predicción dendrítico.
- **Interneuronas VIP+ (Vasoactive Intestinal Peptide):** Inhiben específicamente a las neuronas SST+. Ante la detección de novedad sensorial (modulada por Acetilcolina), VIP+ desinhibe el penacho apical permitiendo la inducción de plasticidad rápida BTSP.

### 2.3. Ley de Dale Estricta
Cada conexión sináptica entre poblaciones neuronales es puramente excitatoria ($W^E \ge 0$) o puramente inhibitoria ($W^I \ge 0$). El peso efectivo neto se calcula como:
$$W_{\text{eff}} = W^E - W^I, \quad W^E \ge 0, \quad W^I \ge 0$$
Cualquier actualización plástica $\Delta W$ ajusta cooperativamente el par $(W^E, W^I)$ manteniendo ambos términos acotados inferiormente por $10^{-4}$, eliminando las violaciones biológicas presentes en Constelación v4.

### 2.4. Plasticidad Sináptica Conductual (Bio-BTSP)
Basada en Bittner et al. (*Science*, 2017) y Milstein et al. (*Nature Neuroscience*, 2021):
- A diferencia de STDP estándar (que requiere correlaciones a escala de milisegundos), BTSP induce cambios sinápticos masivos con una sola meseta de calcio apical de varios cientos de milisegundos.
- Se implementa a través del campo dendrítico rápido $W^F$:
  $$\Delta W^{F,E}_{ij} = \frac{\eta_{\text{fast}} \cdot \text{ACh} \cdot [e_i \cdot \phi_j]_+}{\sum_k \phi_k^2 + \epsilon}, \quad \Delta W^{F,I}_{ij} = \frac{\eta_{\text{fast}} \cdot \text{ACh} \cdot [-e_i \cdot \phi_j]_+}{\sum_k \phi_k^2 + \epsilon}$$

### 2.5. Memoria Episódica Hipocampal CLS y Consolidación en Sueño SWR
- **Giro Dentado (DG):** Proyección con inhibición feedforward lateral $k$-WTA (Winner-Take-All) que garantiza que exactamente $k=16$ de 128 unidades se activen, orthogonalizando los contextos para evitar interferencia catastrófica.
- **CA3:** Red autoasociativa continua que almacena los operadores de transición adaptados $W^F$. Cuando se presenta una tarea afín, CA3 recupera el operador con un softmax térmico y lo inyecta como arranque cálido en la corteza.
- **Consolidación SWR (Sharp-Wave Ripples):** Durante el sueño NREM, el hipocampo reproduce trayectorias salientes del búfer CDT a alta frecuencia con modulación colinérgica baja ($\text{ACh} \approx 0.05$). Esto destila los pesos rápidos volátiles ($W^F$) hacia las sinapsis lentas estructurales ($W^P$) según el principio de etiquetado sináptico de Frey & Morris (1997), aplicando además escalamiento homeostático (Turrigiano & Nelson, 2004) y poda de conexiones inactivas.

---

## 3. Fundamentos Geométricos y Dinámicos de Constelación v4

### 3.1. Representación Residual Puntual Equivariante
En lugar de predecir de forma monolítica un vector de 12 dimensiones, Bio-Constelación aplica un operador compartido $W$ de forma equivariante a cada punto $p \in \{0, \dots, 5\}$:
$$\hat{q}_p(t+1) = q_p(t) + \Delta \hat{q}_p(t)$$
donde $\Delta \hat{q}_p(t) = (W^P + W^F + W^{\text{prior}}) \phi(q_p(t))$.
Esta propiedad garantiza invariancia traslacional y estabilidad numérica a lo largo de rollouts autoregresivos de 16 pasos.

### 3.2. Base Receptiva Cortical L2/3 ($\mathbb{R}^{10}$)
Para cada punto $(x, y)$, la capa cortical L2/3 extrae una base no lineal enriquecida:
$$\phi(x, y) = \begin{bmatrix} x \\ y \\ 1.0 \\ \sqrt{x^2+y^2} \\ \tanh(2x) \\ \tanh(2y) \\ \sin(2x) \\ \sin(2y) \\ \mu_{\text{Koop}, x} \\ \mu_{\text{Koop}, y} \end{bmatrix}$$
Los términos trascendentes capturan dinámicas periódicas y límites no lineales, mientras que los estados latentes $\mu_{\text{Koop}}$ inyectan la memoria temporal de la columna L5.

### 3.3. Operadores de Koopman Contractivos en L5
Cada bloque de Koopman actúa en subespacios $2 \times 2$ mediante matrices de rotación-escala:
$$K_j = \rho_j \begin{bmatrix} \cos(\omega_j) & -\sin(\omega_j) \\ \sin(\omega_j) & \cos(\omega_j) \end{bmatrix}, \quad \text{con } \rho_j \in [0.90, 0.99]$$
Al restringir estrictamente $\rho_j \le 1.0$, el radio espectral del generador nunca supera la unidad, impidiendo explosiones numéricas independientemente del horizonte de predicción.

### 3.4. Prior Tálamo-Cortical Autónomo ($W^{\text{prior}}$)
En Constelación v4, el horizonte $h=1$ en el protocolo E33 dependía críticamente de cargar pesos congelados externos del branch `ÓRBITA` (`e33_v3head` colapsaba con $0/16$ pass sin dicho archivo).
En Bio-Constelación, una hiperred de 2 capas tálamo-corticales mapea el vector contextual inicial $c \in \mathbb{R}^{10}$ al espacio de operadores $W^{\text{prior}} \in \mathbb{R}^{2 \times 10}$, entrenada localmente mediante la regla de 3 factores guiada por Dopamina. Esto permite resolver $h=1$ y $h=16$ en E33 de manera **100% autónoma y reproducible sin dependencias externas**.

---

## 4. Estructura del Crate `bio_constelacion/`

```
bio_constelacion/
├── Cargo.toml                      # Crate standalone Rust 2021, 0 dependencias externas
├── .gitignore
├── run_hybrid.py                   # Runner CLI completo en Python
└── src/
    ├── rng.rs                      # Generador determinista Xoshiro256StarStar (SplitMix64)
    ├── dendritic_koopman.rs        # Neurona piramidal L5, bloques Koopman y microcircuito BAC
    ├── dale_stp_synapse.rs         # Sinapsis lenta Dale (STP, poda, consolidación) y Fast BTSP Field
    ├── hippocampus_cdt_sleep.rs    # DG (k-WTA), CA3 (Hopfield continuo), Neuromoduladores y Sueño SWR
    ├── hybrid_brain.rs             # BioConstelacion: orquestador del microcircuito cerebral híbrido
    ├── protocol_cst4.rs            # Protocolo exacto de Constelación v4 (K_obs=4, H=16, E33, Held-out)
    ├── lib.rs                      # Exportación de módulos y pruebas unitarias de integridad
    └── main.rs                     # CLI en Rust (subcomandos: paso0, smoke, run)
```

---

## 5. Instrucciones de Ejecución y Entrenamiento

### 5.1. Ejecución con Python (Inmediata en cualquier entorno)

El script `run_hybrid.py` replica fielmente la lógica matemática y geométrica de la implementación en Rust:

1. **Paso 0 (Verificación de Integridad y Ley de Dale):**
   ```powershell
   python bio_constelacion/run_hybrid.py --mode paso0
   ```
   *Salida esperada:*
   ```
   === PASO 0: VERIFICACIÓN DE INTEGRIDAD BIO-CONSTELACIÓN ===
   [1] Ley de Dale en todo el campo sináptico: 0 violaciones de 80 pesos (100% CUMPLE)
   [2] Integridad de base receptiva L2/3 y Koopman: OK
   >>> RESULTADO PASO 0: APROBADO <<<
   ```

2. **Smoke Test Rápido (32 episodios vs Constelación v4 congelada):**
   ```powershell
   python bio_constelacion/run_hybrid.py --mode smoke --seed 0xC400 --episodes 32
   ```

3. **Entrenamiento / Evaluación Formal por Fases:**
   ```powershell
   # Fase de Desarrollo (16 semillas 0xA400..0xA40F)
   python bio_constelacion/run_hybrid.py --mode dev --seed 0xA400 --episodes 64

   # Fase de Confirmación (16 semillas 0xB400..0xB40F)
   python bio_constelacion/run_hybrid.py --mode confirm --seed 0xB400 --episodes 64
   ```

### 5.2. Compilación y Ejecución con Rust Cargo

En entornos donde el linker del sistema operativo se encuentre configurado:

1. **Verificación de Tipos y Linter:**
   ```powershell
   cd bio_constelacion
   cargo check --all-targets
   cargo clippy --all-targets -- -D warnings
   ```

2. **Paso 0:**
   ```powershell
   cargo run -- paso0 --seed 0xC400
   ```

3. **Smoke Test:**
   ```powershell
   cargo run -- smoke --seed 0xC400 --episodes 32
   ```

4. **Estudio de Ablaciones:**
   ```powershell
   cargo run -- run --phase dev --seed 0xA400 --ablation no_fast_btsp
   cargo run -- run --phase dev --seed 0xA400 --ablation no_hippo
   cargo run -- run --phase dev --seed 0xA400 --ablation no_sleep
   cargo run -- run --phase dev --seed 0xA400 --ablation no_prior
   ```
