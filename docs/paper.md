# CDT-RQM-EPR

## Sustrato Causal-Relacional con Entrelazamiento Computacional, Criticalidad y Selección Energética

### Resumen

`CDT-RQM-EPR` es un sustrato experimental para memoria, predicción y consolidación causal. La arquitectura combina:

```text
CDT-Graphity  -> hardware causal foliado
RQM           -> software relacional dependiente del observador
EPR           -> enlaces coherentes de correlación computacional
Dvali         -> criticalidad, species bound y temperatura N-portrait
Ising/FEP     -> selección por energía y compresión geométrica
```

La hipótesis central es:

```text
La memoria útil puede vivir en una geometría causal dinámica, no en un vector denso.
Las predicciones deben ser relativas al observador, validadas por energía,
protegidas por causalidad y comprimidas por criticalidad.
```

El estado consolidado vigente es:

```text
data/cdt_rqm_evolutionary_kept.cdt_rqm
```

La evaluación principal es:

```text
cargo run --bin cdt_rqm_consolidated_evaluation
```

Resultado entrenado actual:

```text
suite=PASS
normal:             accuracy=100.0% leakage=10.5%
action_conditioned: accuracy=100.0% leakage=12.9%
typed_memory:       accuracy=100.0% leakage=8.1%
geometry:           edges=84 relations=1616 free_energy=4.482
causality:          violations=0
EPR:                active_links=75 coherence=1.000 entropy=0.000
```

---

## 1. Hipótesis Del Sustrato

El sistema se modela como un pequeño universo computacional:

```text
frontera activa + observador
  -> colapso relacional RQM
  -> sincronización EPR
  -> paso causal CDT
  -> compresión Graphity / Regge / FEP
  -> estado validado
```

Cada mejora se conserva solo si cumple:

```text
accuracy no baja
leakage no sube
free_energy baja o se conserva
compute_cost baja o se conserva
causality_violations = 0
sin bloat geométrico
```

---

## 2. Hardware Causal: CDT-Graphity

El hardware está implementado como `CdtGraphitySubstrate`.

Sus elementos principales son:

```text
nodes       -> grados discretos de libertad
edges       -> aristas espaciales o temporales
tetrahedra  -> soporte simplicial 3D discreto
temperature -> control de reconfiguración
Regge       -> penalización de curvatura/inestabilidad
```

La causalidad se impone por foliación:

```text
Spatial edge:  slice(a) == slice(b)
Temporal edge: slice(b) == slice(a) + 1
```

La métrica crítica es:

```text
causality_violations = 0
```

En el estado final:

```text
causality_violations=0
```

---

## 3. Software Relacional: RQM

El software está implementado como `RelationalFieldSubstrate`.

Una relación depende del observador:

```text
RelationKey {
  observer,
  a,
  b
}
```

El estado relacional contiene:

```text
RelationalState {
  amplitude,
  phase,
  coherence,
  uncertainty,
  last_observed_tick
}
```

La probabilidad relacional se aproxima como:

```text
P(a,b|O) = amplitude^2 * coherence * (1 - uncertainty)
```

El colapso relativo usa interferencia de fase:

```text
interference = amplitude * coherence * (1 - uncertainty) * cos(phase - observer_phase)
score        = max(interference, 0) + probability
```

Esto permite que una misma frontera causal tenga futuros distintos para observadores distintos.

---

## 4. Entrelazamiento Computacional: EPR

`EntanglementField` introduce enlaces de correlación remota:

```text
EntanglementLink {
  a,
  b,
  coherence,
  entropy,
  heat,
  active
}
```

Reglas:

```text
correlación repetida -> crear/reforzar EPR
sincronización útil  -> subir coherence, bajar entropy
contradicción        -> subir entropy/heat
heat alto            -> podar link
```

Evaluación final:

```text
active_links=75
mean_coherence=1.000
mean_entropy=0.000
```

---

## 5. Energía y Geometría

### 5.1 Acción Regge Discreta

Se usa una aproximación por déficit angular:

```text
I_R = sum_edges length(e) * |delta_e|
delta_e = 2*pi - incident_tetrahedra(e) * theta_target(e)
theta_target = 2*pi / target_incident_tetrahedra(e)
```

Resultado final:

```text
regge=517.500
deficit_regge=650.310
```

### 5.2 Free Energy Unificada

La energía libre experimental es:

```text
F = prediction_error
  + lambda_R * Regge_deficit
  + lambda_Lambda * cosmological_action
  + lambda_E * EPR_entropy
  + lambda_leak * leakage
  + lambda_C * causality_violations
  + lambda_K * criticality_distance
```

Resultado final:

```text
free_energy=4.482
```

---

## 6. Criticalidad Dvali

El sustrato conserva una condición tipo maximal packing:

```text
alpha = 1 / N
alpha * N ~= 1
T_N = 1 / sqrt(N)
```

Resultado final:

```text
N=84.0
alpha=0.011905
alphaN=1.000
T_N=0.1091
```

También se mide un species bound computacional:

```text
species_cutoff = 1 / sqrt(N_species)
```

Resultado:

```text
species=177.0
cutoff=0.0752
```

Y una carga de memoria:

```text
memory_burden = useful_memory / (occupation_number + useful_memory)
memory_burden=0.945
```

---

## 7. Mejoras Conservadas

El proceso evolutivo de validación conservó:

```text
typed_memory
contradiction_memory
adaptive_criticality
episodic_replay_sleep
ising_hamiltonian_anneal
```

Y descartó:

```text
energy_min_inference
action_world_model
latent_jepa_prediction
graph_planning_paths
global_state_control
compute_cost_gate
```

### 7.1 Memoria Tipada

Separó el espacio relacional por tipo:

```text
Semantic
Episodic
Causal
Skill
```

Resultado final:

```text
typed_memory accuracy=100.0%
typed_memory leakage=8.1%
```

### 7.2 Memoria De Contradicción

Agregó refuerzo negativo contra distractores:

```text
local -> distractor con success=0
EPR conflict(local, distractor)
```

Resultado final:

```text
contradiction_probe accuracy=100.0%
contradiction_probe leakage=10.5%
```

### 7.3 Criticalidad Adaptativa

Ajusta:

```text
temperature = 1 / sqrt(N)
candidate_budget *= species_cutoff
```

### 7.4 Replay Episódico

Reinyecta episodios y consolida:

```text
episodic trace -> replay -> stabilized relation
```

### 7.5 Hamiltoniano Tipo Ising

La mejora física más fuerte usa:

```text
H = Regge_deficit
  + criticality
  + leakage
  + prediction_error
  + causality
```

El resultado del proceso evolutivo fue:

```text
edges=84
deficit_regge=650.310
free_energy=4.482
compute_cost=0.384
```

---

## 8. Evidencia Consolidada

Comando:

```text
cargo run --bin cdt_rqm_consolidated_evaluation
```

Salida relevante:

```text
loaded=true state=data/cdt_rqm_evolutionary_kept.cdt_rqm

normal:
  accuracy=100.0%
  leakage=10.5%
  margin=49.805
  prediction_error=0.861

action_conditioned:
  accuracy=100.0%
  leakage=12.9%
  margin=48.909
  prediction_error=0.737

typed_memory:
  accuracy=100.0%
  leakage=8.1%
  margin=1.924
  prediction_error=0.758

geometry:
  edges=84
  relations=1616
  regge=517.500
  deficit_regge=650.310
  free_energy=4.482
  criticality_distance=0.526
  mera_gain=0.883
  compute_cost=0.384
  causality_violations=0

EPR:
  active_links=75
  mean_coherence=1.000
  mean_entropy=0.000

suite=PASS
```

---

## 9. Comandos Vigentes

Evaluar:

```powershell
cargo run --bin cdt_rqm_consolidated_evaluation
```

Entrenamiento continuo:

```powershell
$env:CDT_RQM_INFINITE_OUTPUT="data/cdt_rqm_evolutionary_kept.cdt_rqm"; cargo run --bin cdt_rqm_infinite_concept_trainer
```

Entrenamiento compacto con sueño/EPR:

```powershell
$env:CDT_RQM_EPR_SMALL_OUTPUT="data/cdt_rqm_evolutionary_kept.cdt_rqm"; cargo run --bin cdt_rqm_epr_small_sleep_trainer
```

Consolidación:

```powershell
$env:CDT_RQM_EPR_SLEEP_STATE="data/cdt_rqm_evolutionary_kept.cdt_rqm"; cargo run --bin cdt_rqm_epr_sleep_consolidate
```

---

## 10. Conclusión

El sistema vigente es `CDT-RQM-EPR`.

Su identidad técnica es:

```text
hardware causal CDT
+ software relacional RQM
+ EPR computacional
+ criticalidad Dvali
+ memoria tipada
+ Hamiltoniano de compresión
```

La evidencia actual muestra:

```text
accuracy=100.0%
typed_memory leakage=8.1%
edges=84
free_energy=4.482
causality_violations=0
EPR coherence=1.000
suite=PASS
```

La dirección futura es escalar este sustrato, no volver a arquitecturas legacy.
