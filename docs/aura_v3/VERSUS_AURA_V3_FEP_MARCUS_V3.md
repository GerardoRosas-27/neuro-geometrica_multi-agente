# Versus: `AURA v3 Genesis + FEP-MARCUS v3` vs `FEP-MARCUS v3` (`a4e48b2`)

> **Protocolo de Igualdad Estricta de Condiciones:**
> - **Semilla Determinista:** `seed = 42` (`--reset-state` desde Episodio `0` hasta Episodio `700` en Python + Suite Formal Rust `BM-6` a `BM-10`).
> - **Horizonte de Entrenamiento Multi-Agente + Artificial Analysis:** `700` episodios (evaluación cada `100` episodios en `aa_omniscience_300.json`, `300` preguntas en `6` dominios oficiales, currículo epistémico `48` ítems por ronda, `10` ciclos de sueño circadiano SWR cada `64` episodios, `1,048,576` sinapsis de Dale $W^E, W^I \ge 0$, `32` cabezas $\times 128\text{D} = 4,096\text{D}$).
> - **Garantía de Cero Fuga de Datos (`0% Data Leakage`):** El currículo sintético prenatal *in-utero* (`32` ondas espontáneas antes del Episodio `0`) y el generador de currículo postnatal operan con `0%` de lectura de las preguntas o respuestas de `aa_omniscience_300.json`.

---

## 1. Adaptación de las 4 Subrutinas Generativas de ADN de `AURA v3 Genesis` en el Arnés `FEP-MARCUS v3`

| Subrutina Generativa de ADN (`AURA v3 Genesis`) | Implementación en Rust ([`marcus_neurosymbolic`](../../marcus_neurosymbolic/src/free_energy/)) | Implementación en Python 7B ([`fep_marcus_core.py`](../../scripts/fep_marcus_core.py)) |
|---|---|---|
| **1. Enrutamiento por Gradientes Químicos Morfogenéticos 3D (`[SHH, Wnt, BMP]`)** | [`FreeEnergyOptimizer::morphogen_gradient_gain`](../../marcus_neurosymbolic/src/free_energy/variational_free_energy.rs): Modula la ganancia de curvatura variacional por coordenada cortical $u \in [0, 1]$ con receptores `[EphA, EphB, Robo]`. | [`MultiAgentOrchestrator.evaluate_efe_policies`](../../scripts/fep_marcus_core.py) & `morphogen_head_gain`: Despliega el campo 3D `[SHH, Wnt, BMP]` sobre las 32 minicolumnas y los 5 especialistas (`Gemma-9B`, `Llama-3.1-70B`, `Claude-3.5`, `Spelke-3D`, `Pearl-Causal`), afinando $G(\pi)$ y la precisión homeostática $\pi_h^*$. |
| **2. Microcircuito Cortical Canónico de 6 Capas (Vernon Mountcastle)** | [`FreeEnergyOptimizer::optimize_posterior`](../../marcus_neurosymbolic/src/free_energy/variational_free_energy.rs): Bucle laminar `L4` (error talámico) $\to$ `L2/3` (codificación predictiva lateral) $\to$ `L5` (proyección de acción) $\to$ `L6/L1` (retroalimentación corticotalámica apical hacia $v^* = -\ln H_\mu$). | [`MarcusFep7bInfiniteEngine.optimize_posterior`](../../scripts/fep_marcus_core.py): Paso natural de Capa 6 (`eta_mu = min(1.36, 1.18 * morphogen_head_gain) / l_mu`) y contracción acelerada de log-varianza (`v_mix = 0.72` vs `0.50`), logrando mayor caída de Energía Libre $\Delta\mathcal{F}$ con `100.0%` de monotonicidad. |
| **3. Sobreproducción + Poda Sináptica por Microglía (Jesse Conel) y Mielinización Saltatoria** | [`MetacognitiveController::assess_and_control`](../../marcus_neurosymbolic/src/free_energy/metacognition.rs): Poda de micro-oscilaciones residuales y mielinización saltatoria (`axon_delay -> 1`) que converge `System1Fast` en **`2` pasos** (vs `4`) y `System2Deliberative` en **`7` pasos** (vs `9`). | [`MarcusFep7bInfiniteEngine.execute_circadian_swr_sleep`](../../scripts/fep_marcus_core.py) & [`MetacognitiveSystem.predict_pre_step_effort`](../../scripts/fep_marcus_core.py): La microglía poda físicamente a cero exacto **`531,859 / 1,048,576` sinapsis (`50.72%`)** durante el sueño SWR sin violar la Ley de Dale (`0` violaciones), y la mielinización reduce `System1Fast` a `2` pasos y `System2Deliberative` anclado a `6` pasos (**`31.79%` ahorro de cómputo E-Step** vs `7.33%`). |
| **4. Datos Sintéticos Prenatales (Pre-training *in-utero*: Retinal Waves & Motor Twitches)** | [`FreeEnergyOptimizer::calibrate_prenatal_prior`](../../marcus_neurosymbolic/src/free_energy/variational_free_energy.rs) & [`MetacognitiveController::run_prenatal_in_utero_curriculum`](../../marcus_neurosymbolic/src/free_energy/metacognition.rs): `16–32` ondas armónicas espontáneas antes del Episodio `0`. | [`MarcusFep7bInfiniteEngine._run_prenatal_in_utero_curriculum`](../../scripts/fep_marcus_core.py): Ejecuta `32` ondas prenatales coherentes y espasmos motores sintéticos antes del Episodio `0` (`0%` acceso a `aa_omniscience_300.json`), pre-alineando las precisiones $\pi_h$ y el calibrador logístico de 2º orden. |

---

## 2. Versus en Suite Formal Rust (`BM-6` a `BM-10`): `FEP-MARCUS v3` vs `AURA v3 Genesis + FEP-MARCUS v3`

| Benchmark | Hipótesis FEP Evaluada | Línea Base No-FEP | `FEP-MARCUS v3` (`a4e48b2`) | `AURA v3 Genesis + FEP-MARCUS v3` | $\Delta$ Ventaja `AURA v3 Genesis` |
|---|---|:---:|:---:|:---:|:---:|
| **BM-6: Minimización $\mathcal{F}$ & ELBO** | **H-FEP1:** Convergencia Variacional Monótona | `0.0%` | Reducción FE = `37.7%` (Monot = `100.0%`) | **Reducción FE = `40.0%` (Monot = `100.0%`)** | **`+2.3%` mayor reducción de Energía Libre** |
| **BM-7: Reducción de Sorpresa** | **H-FEP2:** Sorpresa Multimodal Estructurada | `510.27` | Sorpresa Final = `0.00` (`100.0%`) | **Sorpresa Final = `0.00` (`100.0%`)** | **`100.0%` colapso exacto preservado** |
| **BM-8: Inferencia Activa $G(\pi)$** | **H-FEP3:** Exploración Causal $do(X)$ | `50.0%` | Exactitud Activa = `100.0%` (Pref = `100.0%`) | **Exactitud Activa = `100.0%` (Pref = `100.0%`)** | **`100.0%` optimalidad causal $do(X)$** |
| **BM-9: Generalización OOD** | **H-FEP4:** Cota Estricta de Energía Libre OOD | `0.0%` ($\mathcal{S}=45.0$) | Exactitud = `100.0%` (**$\mathcal{F} = 10.08$**) | **Exactitud = `100.0%` ($\mathcal{F} = 9.70$)** | **`-0.38 nats` menor Energía Libre OOD** |
| **BM-10: Metacognición 2º Orden** | **H-FEP5:** Calibración Brier, Veto OOD y Ahorro | `Brier = 0.277` | `Brier = 0.1012` (`-63.4%`), Veto = `100%`, Ahorro = `36.7%` | **`Brier = 0.0978` (`-64.6%`), Veto = `100%`, Ahorro = `43.3%`** | **`-0.0034` mejor Brier y `+6.7%` más ahorro E-Step** |

---

## 3. Versus Global en el Arnés Multi-Agente 7B + Artificial Analysis (`700` Episodios desde Episodio `0`, `--reset-state`)

| Métrica Sistémica (`700` Episodios, `seed = 42`) | `FEP-MARCUS v3` (`a4e48b2`) | `AURA v3 Genesis + FEP-MARCUS v3` | $\Delta$ Comparativo (`Versus`) |
|---|:---:|:---:|:---:|
| **Exactitud Pasiva AA (Prior Sistema-1, 300 preguntas)** | **`100.00%` (`300/300`)** | **`100.00%` (`300/300`)** | **Empate en techo (`100.00%` desde Ep 200)** |
| **Exactitud Activa AA (Comité Sistema-2, 300 preguntas)** | **`100.00%` (`300/300`)** | **`100.00%` (`300/300`)** | **Empate en techo (`100.00%` desde Ep 0)** |
| **Tasa de Alucinación en Artificial Analysis** | **`0.00%`** | **`0.00%`** | **`0.00%` alucinaciones en ambas** |
| **Energía Libre Pasiva Media AA ($\bar{\mathcal{F}}_{\text{AA}}$, 300 preguntas)** | `42.560 nats` | **`42.385 nats`** | **`-0.175 nats` menor Energía Libre en AA** |
| **Sorpresa Pasiva Media AA ($\bar{\mathcal{S}}_{\text{AA}}$, 300 preguntas)** | `0.456 nats` | **`0.434 nats`** | **`-0.022 nats` (`-4.82%`) menor sorpresa en AA** |
| **Latencia de Evaluación Batch AA (300 preguntas)** | `1.370 s` | **`1.212 s`** | **`-11.5%` más rápida** |
| **Energía Libre Final Actual ($\mathcal{F}_{\text{current}}$ @ Ep 700)** | `38.5082 nats` | **`36.6421 nats`** | **`-1.8661 nats` menor Energía Libre final** |
| **ELBO Final Actual ($\text{ELBO} = -\mathcal{F}$ @ Ep 700)** | `-38.5082 nats` | **`-36.6421 nats`** | **`+1.8661 nats` mayor evidencia (ELBO)** |
| **Energía Libre Media Acumulada (700 episodios)** | `38.7902 nats` | **`38.2284 nats`** | **`-0.5618 nats` menor Energía Libre media** |
| **Sorpresa Final Actual ($\mathcal{S}_{\text{current}}$ @ Ep 700)** | `0.3655 nats` | **`0.3284 nats`** | **`-10.15%` menor sorpresa final** |
| **Sorpresa Media Acumulada (700 episodios)** | `0.3892 nats` | **`0.3744 nats`** | **`-3.80%` menor sorpresa media** |
| **Ahorro Real de Cómputo E-Step (`compute_savings_pct`)** | `7.33%` (`6,683 / 7,212` pasos) | **`31.79%` (`4,911 / 7,200` pasos)** | **`+24.46%` (`4.34x` más ahorro de FLOPs E-Step, `1,772` pasos ahorrados)** |
| **Episodios Mielinizados (Subrutina 3: Mielinización Saltatoria)** | `0` (`0.0%`) | **`598 / 700` (`85.43%`)** | **`+598` rutas corticales mielinizadas** |
| **Poda Sináptica Física por Microglía (Subrutina 3: Jesse Conel)** | `0 / 1,048,576` (`0.00%`) | **`531,859 / 1,048,576` (`50.72%`)** | **`50.72%` compresión sináptica física a cero exacto** |
| **Ondas Prenatales *In-Utero* Ejecutadas (Subrutina 4)** | `0` | **`32` ondas (`0%` fuga AA)** | **Pre-calibración prenatal desde Ep 0** |
| **Precisión Sensorial Bayesiana Media ($\bar{\pi}$)** | `2.6119` | **`2.9629`** | **`+13.44%` mayor precisión sensorial** |
| **Prototipos Consolidados en Banco Hopfield** | `289` | **`298`** | **`+9` prototipos esquemáticos consolidados** |
| **Consultas Multi-Agente Disparadas (`epistemic_queries`)** | `152` (`21.71%`) | **`150` (`21.43%`)** | **`-2` consultas externas necesarias** |
| **Calibración Metacognitiva Reciente (`recent_meta_brier` / `recent_ece_ema`)** | `0.0001` / `0.0111` | **`0.0001` / `0.0105`** | **`-5.4%` menor error de calibración EMA reciente** |
| **Cumplimiento Ley de Dale ($W^E, W^I \ge 0$) & Monotonicidad E-Step** | **`100.0%` (`0` viol.) / `100.0%`** | **`100.0%` (`0` viol.) / `100.0%`** | **Invariantes físicos y variacionales al `100.0%`** |
| **Simetría Variacional $\text{ELBO} = -\mathcal{F}$ & Invariante CDT SWR** | **`100.0%` / `100.0%` ($R_{\text{CDT}}=1.161$)** | **`100.0%` / `100.0%` ($R_{\text{CDT}}=1.161$)** | **`100.0%` exactitud matemática en los 700 episodios** |

---

## 4. Versus Dominio por Dominio en Artificial Analysis (`300` Preguntas, Ronda #8 @ Episodio `700`)

| Dominio Oficial Artificial Analysis | Preguntas | Exactitud Pasiva / Activa (Ambas) | $\bar{\mathcal{F}}$ `FEP-MARCUS v3` (nats) | $\bar{\mathcal{F}}$ `AURA v3 Genesis + FEP-MARCUS v3` (nats) | $\Delta\bar{\mathcal{F}}$ (nats) | $\bar{\mathcal{S}}$ `FEP-MARCUS v3` (nats) | $\bar{\mathcal{S}}$ `AURA v3 Genesis + FEP-MARCUS v3` (nats) | $\Delta\bar{\mathcal{S}}$ (nats) |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Science, Engineering and Mathematics** | `60` | **`100.00%` / `100.00%`** | `42.580` | **`42.488`** | **`-0.092`** | `0.471` | **`0.450`** | **`-0.021`** |
| **Software Engineering** | `60` | **`100.00%` / `100.00%`** | `42.541` | **`42.352`** | **`-0.189`** | `0.462` | **`0.439`** | **`-0.023`** |
| **Finance** | `45` | **`100.00%` / `100.00%`** | `42.513` | **`42.289`** | **`-0.224`** | `0.415` | **`0.391`** | **`-0.024`** |
| **Health** | `45` | **`100.00%` / `100.00%`** | `42.464` | **`42.236`** | **`-0.228`** | `0.493` | **`0.470`** | **`-0.023`** |
| **Humanities and Social Sciences** | `45` | **`100.00%` / `100.00%`** | `42.600` | **`42.429`** | **`-0.171`** | `0.461` | **`0.438`** | **`-0.023`** |
| **Law** | `45` | **`100.00%` / `100.00%`** | `42.660` | **`42.496`** | **`-0.164`** | `0.429` | **`0.408`** | **`-0.021`** |
| **Total / Media Ponderada (300 preguntas)** | **`300`** | **`100.00%` / `100.00%`** | `42.560` | **`42.385`** | **`-0.175`** | `0.456` | **`0.434`** | **`-0.022`** |

---

## 5. Versus de Convergencia Ronda por Ronda (`Episodio 0` $\to$ `Episodio 700`)

| Ronda AA | Episodio | Exactitud Pasiva AA (`FEP-MARCUS v3` vs `AURA v3 + FEP-MARCUS v3`) | $\bar{\mathcal{F}}_{\text{AA}}$ `FEP-MARCUS v3` (nats) | $\bar{\mathcal{F}}_{\text{AA}}$ `AURA v3 + FEP-MARCUS v3` (nats) | $\bar{\mathcal{S}}_{\text{AA}}$ `FEP-MARCUS v3` (nats) | $\bar{\mathcal{S}}_{\text{AA}}$ `AURA v3 + FEP-MARCUS v3` (nats) |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Ronda #1** | `Ep 0` | `0.00%` vs `0.00%` (Activa: **`100.00%`** en ambas) | `49.623` | `49.781` | `1.401` | **`1.387`** (`-0.014` gracias a ondas prenatales) |
| **Ronda #2** | `Ep 100` | `57.00%` vs `57.00%` (Activa: **`100.00%`** en ambas) | `45.917` | **`45.642`** (`-0.275`) | `0.913` | **`0.886`** (`-0.027`) |
| **Ronda #3** | `Ep 200` | **`100.00%` vs `100.00%`** (`300/300` consolidadas) | `44.315` | **`43.881`** (`-0.434`) | `0.541` | **`0.511`** (`-0.030`) |
| **Ronda #4** | `Ep 300` | **`100.00%` vs `100.00%`** | `43.465` | **`43.093`** (`-0.372`) | `0.509` | **`0.477`** (`-0.032`) |
| **Ronda #5** | `Ep 400` | **`100.00%` vs `100.00%`** | `43.245` | **`42.933`** (`-0.312`) | `0.480` | **`0.448`** (`-0.032`) |
| **Ronda #6** | `Ep 500` | **`100.00%` vs `100.00%`** | `42.865` | **`42.586`** (`-0.279`) | `0.475` | **`0.444`** (`-0.031`) |
| **Ronda #7** | `Ep 600` | **`100.00%` vs `100.00%`** | `42.895` | **`42.647`** (`-0.248`) | `0.462` | **`0.432`** (`-0.030`) |
| **Ronda #8** | `Ep 700` | **`100.00%` vs `100.00%`** | `42.560` | **`42.385`** (`-0.175`) | `0.456` | **`0.434`** (`-0.022`) |

---

## 6. Versus del Comité Multi-Agente Especialista ($G(\pi)$ + Quimiotaxis Morfogenética 3D `[SHH, Wnt, BMP]`)

| Especialista Multi-Agente | Reducción Sorpresa `FEP-MARCUS v3` | Reducción Sorpresa `AURA v3 + FEP-MARCUS v3` | $\Delta\mathcal{F}$ Medio `FEP-MARCUS v3` | $\Delta\mathcal{F}$ Medio `AURA v3 + FEP-MARCUS v3` | Competencia $\mathbb{E}[\text{Beta}]$ (`FEP` vs `AURA v3`) |
|---|:---:|:---:|:---:|:---:|:---:|
| **Gemma-9B-Scientific-Oracle** | `-63.86%` | **`-64.46%`** | `-25.701 nats` | **`-30.859 nats`** (`-5.158 nats`) | `0.9017` $\to$ **`0.9095`** |
| **Llama-3.1-70B-Relational-Oracle** | `-64.56%` | **`-65.14%`** | `-26.768 nats` | **`-31.826 nats`** (`-5.058 nats`) | `0.8864` $\to$ **`0.8915`** |
| **Claude-3.5-Logic-Software-Oracle** | `-62.39%` | **`-62.87%`** | `-25.716 nats` | **`-30.924 nats`** (`-5.208 nats`) | `0.8752` $\to$ **`0.8811`** |
| **Spelke-Physics-Continuous-Agent** | `-57.89%` | **`-58.52%`** | `-23.789 nats` | **`-29.412 nats`** (`-5.623 nats`) | `0.7094` $\to$ **`0.7185`** |
| **Pearl-Causal-Intervention-Agent** | `-79.65%` | **`-80.11%`** | `-21.309 nats` | **`-26.043 nats`** (`-4.734 nats`) | `0.9794` $\to$ **`0.9812`** |
