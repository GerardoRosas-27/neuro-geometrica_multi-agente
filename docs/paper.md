# Motor Termodinámico Nativo CDT-RQM-EPR

## Sustrato Causal-Relacional Consolidado con Sueño Contrastivo

### Resumen

Este documento actualiza el estado de la investigación después de migrar el sustrato anterior `CDT-RQM-EPR` hacia una arquitectura de **motor termodinámico nativo**. La conclusión experimental actual es:

```text
Recomendación: continuar la investigación con el motor termodinámico nativo.
El sustrato anterior debe conservarse como baseline, fuente de checkpoints y validador.
```

La arquitectura vigente combina:

```text
checkpoint CDT-RQM-EPR entrenado
  -> adaptador de migración legacy->native
  -> NativeThermoCdtSubstrate
  -> NativeThermoRqmEprSubstrate
  -> sueño termodinámico contrastivo
  -> evaluación amplia de conocimiento consolidado
```

El estado entrenado usado como fuente sigue siendo:

```text
data/cdt_rqm_evolutionary_kept.cdt_rqm
```

El comando principal del motor consolidado es:

```powershell
cargo run --release --bin native_thermodynamic_engine
```

La evaluación amplia de conocimiento se ejecuta con:

```powershell
cargo run --release --bin consolidated_knowledge_evaluation
```

---

## 1. Hipótesis Actual

La hipótesis inicial era que la memoria útil podía vivir en una geometría causal dinámica, no en un vector denso. Los resultados recientes refinan esa hipótesis:

```text
La memoria útil se puede migrar desde una geometría CDT-RQM-EPR legacy
hacia un motor termodinámico nativo, siempre que exista una fase de sueño
que consolide no solo recuerdos positivos, sino también competencia contrastiva.
```

El criterio de conservación de una mejora es:

```text
accuracy no baja
leakage baja o se conserva
margin sube o se conserva
causality_violations = 0 en el baseline causal
runtime baja o se conserva
el conocimiento entrenado permanece recuperable
```

---

## 2. Sustrato Anterior: CDT-RQM-EPR

El sustrato anterior conserva tres capas principales:

```text
CDT-Graphity  -> hardware causal foliado
RQM           -> software relacional dependiente del observador
EPR           -> enlaces coherentes de correlación computacional
```

El flujo de inferencia legacy es:

```text
boundary + observer
  -> RelationalFieldSubstrate::observe_pattern
  -> RelationalGuidanceEngine::apply
  -> EntanglementField::synchronize_candidates
  -> CdtGraphitySubstrate::step
```

La pieza importante que el motor nativo no tenía inicialmente era:

```text
RelationalGuidanceEngine
```

Esta capa reordena candidatos usando soporte geométrico, flujo causal, costo Regge, potencial cuántico local y memoria capilar. En la práctica ayuda a reducir candidatos válidos pero fuera de contexto.

### Baseline Consolidado

Comando:

```powershell
cargo run --release --bin cdt_rqm_consolidated_evaluation
```

Resultado actual sobre `data/cdt_rqm_evolutionary_kept.cdt_rqm`:

```text
normal:             accuracy=100.0% leakage=9.9%  margin=105.170 prediction_error=0.936
action_conditioned: accuracy=100.0% leakage=12.3% margin=121.340 prediction_error=0.830
typed_memory:       accuracy=100.0% leakage=8.1%  margin=1.924   prediction_error=0.940
contradiction_probe:accuracy=100.0% leakage=9.9%  margin=105.170 prediction_error=0.936

geometry:
  edges=6509
  relations=4928
  regge=26103.500
  deficit_regge=40941.949
  free_energy=193.979
  criticality_distance=35.756
  compute_cost=10.940
  causality_violations=0

dvali:
  N=6509.0
  alpha=0.000154
  alphaN=1.000
  T_N=0.0124
  memory_burden=0.406

epr:
  active_links=250
  mean_coherence=1.000
  mean_entropy=0.000

suite=PASS
```

Interpretación:

```text
El sustrato anterior sigue siendo correcto y estable.
Su mayor valor actual es servir como baseline, checkpoint y mecanismo de validación causal.
```

---

## 3. Motor Termodinámico Nativo

El motor nativo consolida las capas anteriores en una arquitectura más directa:

```text
NativeThermoCdtSubstrate
  thermal_state
  amplitude
  phase
  temperature
  energy
  activation
  compiled sampling program

NativeThermoRqmEprSubstrate
  relations
  relation_lookup
  neighbor_index
  EntanglementField
  thermal scoring
```

El núcleo de consulta es:

```text
query(observer, phase, seeds)
  -> relational_candidate_scores
  -> EPR sync cuando hay ambigüedad
  -> pulse_compiled_pilot
  -> thermal_multiplier
  -> candidate ranking
```

La migración desde el sustrato anterior se implementa en:

```text
src/substrate_adapter.rs
```

La API consolidada está en:

```text
src/native_thermodynamic_engine.rs
```

El CLI productivo está en:

```text
src/bin/native_thermodynamic_engine.rs
```

---

## 4. Ventajas Físico-Computacionales Del Motor Nativo

El motor termodinámico nativo mejora al sustrato anterior porque convierte la inferencia en un problema local de relajación, muestreo y competencia energética. En vez de depender de pasos globales de geometría Graphity/Regge en cada consulta, usa un estado térmico vectorial por nodo y solo activa bloques impactados por la frontera y los candidatos.

### 4.1 Cálculo Local Compilado

El sustrato nativo compila un programa de muestreo:

```text
NativeSamplingProgram
  blocks
  schedule
  node_to_block
```

En inferencia, el pulso termodinámico no recorre todo el universo si no hace falta:

```text
block_ids = scheduled_impacted_blocks(seeds, candidates)
for block_id in block_ids:
  sample_block(block)
```

Esto cambia el costo efectivo:

```text
legacy: O(relaciones observadas + paso CDT/Graphity + guía geométrica)
nativo: O(relaciones vecinas + bloques impactados)
```

Evidencia:

```text
legacy_cdt_rqm_consolidated:
  us_per_case=1280.029 a 1649.442

native_thermodynamic_consolidated:
  us_per_case=339.240 a 343.507

ganancia_runtime=~3.7x a ~4.9x
```

### 4.2 Dinámica Termodinámica Tipo Langevin

El estado térmico evoluciona como una discretización de Langevin con difusión, confinamiento, fuerza piloto y ruido térmico:

```text
laplacian_i = sum_j w_ij * (x_j - x_i)
pilot_i     = amplitude_i * sin(phase_i) + activation_i
force_i     = diffusion * laplacian_i
            + pilot_gain * pilot_i
            - confinement * x_i
noise_i     = Normal(0, sqrt(2 * T_i * dt))
x_i(t+dt)   = clamp(x_i + force_i * dt + noise_i)
```

La fase también fluye localmente:

```text
phase_flow_i = sum_j w_ij * sin(phase_j - phase_i + edge_phase_ij)
phase_i'     = phase_i + phase_coupling * phase_flow_i * dt + x_i' * dt
```

Ventaja:

```text
La memoria no solo se recupera por score relacional.
También se estabiliza por atracción térmica, difusión local y fase.
```

### 4.3 Energía Efectiva Local

Cada nodo estima una energía efectiva:

```text
E_i = 0.5 * confinement * x_i^2
    - force_i * x_i
    + 0.5 * laplacian_i^2
```

Esto penaliza estados inestables y favorece atractores coherentes. La energía libre proxy usa una partición tipo Boltzmann:

```text
Z = sum_i exp(-E_i / T_i)
F = -T * ln(Z)
```

Evidencia del entrenamiento limpio:

```text
batch=338804
mean_energy=2.0188
free_energy=-1.5608
```

El valor negativo de `free_energy` indica que el sistema encontró atractores térmicos con partición favorable. No es directamente comparable en escala con el Regge legacy, pero sí es útil para estabilidad interna del motor nativo.

### 4.4 Muestreo Híbrido: Gaussian, Gibbs y Bernoulli

El motor no usa un único método de actualización. Cada bloque puede muestrear con:

```text
Gaussian:
  x' = x + force * dt + noise

Gibbs:
  proposal = tanh(force / T)
  x' = proposal + jitter

Bernoulli:
  p = sigmoid(force / T)
  x' -> {+1, -1}
```

Ventaja:

```text
Gaussian explora continuo.
Gibbs estabiliza según energía/temperatura.
Bernoulli fuerza decisiones discretas cuando conviene colapsar.
```

Esta mezcla explica por qué el motor aprende rápido y separa memorias con pocos nodos:

```text
nodes=640
relations=3878
epr_links=640
accuracy=100.0%
leakage=0.1%
```

### 4.5 Scoring Termodinámico

La inferencia nativa combina score relacional con un multiplicador térmico:

```text
score_final(candidate) = relational_score * thermal_multiplier
```

Donde:

```text
thermal_multiplier =
  1 + thermal_score_gain * (
        tanh(state)
      + 0.1 * amplitude
      + 0.05 * exp(-energy / temperature)
    )
```

Ventaja:

```text
Un candidato no gana solo por memoria relacional.
Debe ser compatible con el estado térmico, la amplitud y la energía local.
```

Esto explica el aumento de margen:

```text
legacy margin global=127.464
native margin global=420.909
native clean final margin=691.721
```

### 4.6 Sueño Contrastivo Como Optimización

El sueño nativo implementa una forma de optimización contrastiva:

```text
para cada memoria:
  reforzar cue -> target
  atenuar cue -> distractor explícito
  atenuar cue -> remotos de otras memorias
  relajar térmicamente
  aceptar solo si accuracy no baja y leakage/margin mejora
```

Formalmente, el objetivo implícito es:

```text
min L = leakage
      - margin_gain
      + penalty(accuracy_drop)
      + thermal_instability
```

El entrenamiento limpio muestra el efecto repetidamente:

```text
sleep=contrastive accepted=6
accuracy=97.2% -> 100.0%
leakage=7.7% -> 0.1%
margin=946.210 -> 691.721
```

Aunque el margen puede bajar durante sueño, el sistema acepta porque elimina fuga y recupera accuracy. La reducción de leakage es prioritaria cuando la separación ya es suficiente.

### 4.7 Evidencia Del Entrenamiento Desde Cero

El entrenamiento limpio nativo fue ejecutado desde cero, sin cargar el checkpoint legacy:

```powershell
cargo run --release --bin native_thermo_clean_trainer
```

Estado final guardado:

```text
checkpoint=data/native_thermo_clean.cdt_native
batch=338804
samples=5420864
semantic=1806954
causal=903477
skill=903477
episodic=1806956
sleep_runs=84701
growths=0
slices=4
nodes=640
relations=3878
epr_links=640
```

Métricas finales:

```text
accuracy=100.0%
leakage=0.1%
margin=691.721
mean_energy=2.0188
free_energy=-1.5608
relation_density=6.059
epr_density=1.000
```

Interpretación:

```text
El motor termodinámico nativo aprende desde cero.
No depende del checkpoint legacy para lograr baja fuga.
No necesitó crecer: la densidad quedó bajo los umbrales.
El sueño periódico mantiene el estado cerca del óptimo.
```

### 4.8 Resumen De Ventajas

```text
1. Menor costo de inferencia por bloques impactados.
2. Relajación física local en vez de paso geométrico global.
3. Score condicionado por energía y temperatura.
4. Muestreo híbrido que combina exploración y colapso.
5. Sueño contrastivo que separa memorias válidas entre sí.
6. Entrenamiento limpio desde cero con checkpoint nativo.
7. Mejor leakage, mejor margen y mayor velocidad que el baseline.
```

Por estas razones, el motor termodinámico nativo no es solo una versión más rápida. Es una arquitectura mejor optimizada para aprendizaje incremental, consolidación y recuperación robusta.

---

## 5. Adaptador Legacy -> Nativo

El adaptador carga el checkpoint anterior y migra:

```text
RelationalFieldSubstrate -> relaciones nativas dirigidas
EntanglementField        -> EPR nativo reutilizado
CdtGraphitySubstrate     -> estado térmico y aristas activas nativas
```

Resumen de migración:

```text
legacy_relations=4928
imported_relations=9856
nodes=640
imported_edges=6509
epr_links=250
```

Las relaciones se duplican en ambos sentidos:

```text
(a -> b, phase)
(b -> a, -phase)
```

Esto conserva consultas por cualquier lado de una relación legacy no dirigida.

---

## 6. Sueño Termodinámico Nativo

El primer sueño nativo solo hacía replay positivo y atenuaba el distractor explícito de cada lección. Eso mejoraba memoria directa, acción, memoria tipada, cues parciales y ruido, pero dejaba más alto el caso `cross_distractor`.

Diagnóstico:

```text
cross_distractor usa como distractor el target remoto de otra memoria aprendida.
Ese target no es ruido: es una memoria válida en otro contexto.
```

Por eso faltaba una función equivalente a competencia contextual:

```text
inhibición contrastiva entre recuerdos válidos
```

La versión actual del sueño nativo hace:

```text
1. replay protegido del target correcto
2. atenuación del distractor explícito
3. atenuación contrastiva de remotos de otras lecciones
4. relajación térmica
5. aceptación solo si preserva accuracy y mejora leakage o margin
```

Resultado del motor antes y después de sueño:

```text
native_before_sleep:
  accuracy=100.0%
  leakage=10.5%
  margin=0.009
  us_per_case=332.350

native_sleep:
  attempts=8
  accepted=8
  accuracy=100.0% -> 100.0%
  leakage=10.5% -> 0.1%
  margin=0.009 -> 443.611
  epr_links=250 -> 699

native_after_sleep:
  accuracy=100.0%
  leakage=0.1%
  margin=443.611
  us_per_case=339.240

decision=keep_native
```

---

## 7. Comparación Principal: Legacy vs Nativo Dormido

Comando:

```powershell
cargo run --release --bin native_thermodynamic_engine
```

Resultado:

```text
previous_cdt_rqm_epr:
  accuracy=100.0%
  leakage=10.1%
  margin=76.145
  dynamics=0.902
  us_per_case=1649.442
  relations=4928
  epr_links=250
  energy=26103.500

native_thermo_rqm_epr_after_sleep:
  accuracy=100.0%
  leakage=0.1%
  margin=443.611
  dynamics=4.060
  us_per_case=339.240
  relations=10519
  epr_links=699
  energy=5.604
```

Conclusión cuantitativa:

```text
accuracy:      igual, 100.0%
leakage:       nativo mejor, 10.1% -> 0.1%
margin:        nativo mejor, 76.145 -> 443.611
inferencia:    nativo mejor, ~4.9x más rápido
energía proxy: nativo mucho menor en escala nativa
```

---

## 8. Evaluación Amplia De Conocimiento

Comando:

```powershell
cargo run --release --bin consolidated_knowledge_evaluation
```

La suite evalúa 48 casos por repetición:

```text
direct_memory
action_conditioned
typed_memory
partial_cue
noisy_cue
cross_distractor
```

### Resultado Global

```text
legacy_cdt_rqm_consolidated:
  accuracy=100.0%
  leakage=9.1%
  margin=127.464
  expected_score=142.588
  distractor_score=15.124
  signal_ratio=9.428
  us_per_case=1280.029

native_thermodynamic_consolidated:
  accuracy=100.0%
  leakage=0.4%
  margin=420.909
  expected_score=422.515
  distractor_score=1.607
  signal_ratio=263.003
  us_per_case=343.507
```

Delta:

```text
accuracy_delta=+0.0pp
leakage_delta=-8.7pp
margin_delta=+293.445
signal_ratio_gain=27.896x
runtime_gain=~3.7x
```

### Resultado Por Categoría

Legacy:

```text
direct_memory:      accuracy=100.0% leakage=9.9%  margin=105.170 signal_ratio=8.732
action_conditioned: accuracy=100.0% leakage=12.3% margin=121.340 signal_ratio=6.969
typed_memory:       accuracy=100.0% leakage=8.1%  margin=1.924   signal_ratio=10.713
partial_cue:        accuracy=100.0% leakage=10.1% margin=52.900  signal_ratio=8.475
noisy_cue:          accuracy=100.0% leakage=9.6%  margin=103.725 signal_ratio=9.086
cross_distractor:   accuracy=100.0% leakage=8.8%  margin=106.447 signal_ratio=9.636
```

Nativo consolidado:

```text
direct_memory:      accuracy=100.0% leakage=0.1% margin=499.732 signal_ratio=1502.752
action_conditioned: accuracy=100.0% leakage=0.2% margin=565.833 signal_ratio=509.173
typed_memory:       accuracy=100.0% leakage=0.1% margin=265.267 signal_ratio=913.594
partial_cue:        accuracy=100.0% leakage=0.1% margin=262.204 signal_ratio=1531.399
noisy_cue:          accuracy=100.0% leakage=0.1% margin=498.266 signal_ratio=1512.380
cross_distractor:   accuracy=100.0% leakage=1.8% margin=492.313 signal_ratio=64.514
```

Interpretación:

```text
El motor nativo gana en todas las categorías medidas.
El caso más difícil sigue siendo cross_distractor, pero después de inhibición contrastiva
baja de 10.7% a 1.8% y queda mejor que el legacy.
```

---

## 9. Datos De Aprendizaje

Los datos indican tres fases de aprendizaje:

### 8.1 Aprendizaje Legacy

```text
relations=4928
epr_links=250
accuracy=100.0%
leakage_global=9.1% a 10.1%
```

El legacy aprende correctamente, pero mantiene fuga moderada porque muchos recuerdos válidos compiten dentro del mismo espacio causal.

### 8.2 Migración Nativa

```text
imported_relations=9856
imported_edges=6509
accuracy=100.0%
leakage_before_sleep=10.5%
margin_before_sleep=0.009
```

La migración conserva el conocimiento, pero inicialmente lo expresa con margen muy bajo porque traduce memoria sin consolidarla en la dinámica nativa.

### 8.3 Sueño Contrastivo Nativo

```text
relations_after_sleep=10519
epr_links_after_sleep=699
leakage_after_sleep=0.1% a 0.4%
margin_after_sleep=420.909 a 443.611
signal_ratio=263.003
```

El sueño no solo repite recuerdos; crea separación contextual. Esta fase es lo que convierte la migración en un motor superior.

---

## 10. Qué Faltaba Respecto Al Sustrato Anterior

La función faltante no era inferencia relacional ni EPR. Esas ya estaban cubiertas.

Lo faltante era:

```text
competencia contextual entre memorias válidas
```

En el legacy esa competencia aparecía parcialmente en:

```text
RelationalGuidanceEngine::apply
  + geometría CDT
  + causal_gate
  + local_regge_cost
  + capillary_memory
```

En el nativo se implementó como:

```text
sueño contrastivo
  target correcto: reforzar
  distractor explícito: atenuar
  remotos de otros recuerdos: atenuar suavemente
```

Resultado:

```text
cross_distractor leakage:
  antes=10.7%
  después=1.8%
```

---

## 11. Comandos Vigentes

Motor termodinámico consolidado:

```powershell
cargo run --release --bin native_thermodynamic_engine
```

Evaluación amplia de conocimiento:

```powershell
cargo run --release --bin consolidated_knowledge_evaluation
```

Baseline legacy:

```powershell
cargo run --release --bin cdt_rqm_consolidated_evaluation
```

Entrenamiento continuo legacy para generar checkpoints:

```powershell
$env:CDT_RQM_INFINITE_OUTPUT="data/cdt_rqm_evolutionary_kept.cdt_rqm"; cargo run --release --bin cdt_rqm_infinite_concept_trainer
```

Sueño legacy sobre checkpoint:

```powershell
$env:CDT_RQM_EPR_SLEEP_STATE="data/cdt_rqm_evolutionary_kept.cdt_rqm"; cargo run --release --bin cdt_rqm_epr_sleep_consolidate
```

Tests:

```powershell
cargo test --release
```

---

## 12. Recomendación De Investigación

Con los datos actuales, no se recomienda regresar al sustrato anterior como arquitectura principal.

Se recomienda:

```text
continuar con el motor termodinámico nativo consolidado
mantener CDT-RQM-EPR legacy como baseline, validador y fuente de checkpoints
investigar persistencia nativa del estado dormido
ampliar la suite de conocimiento con más categorías y distractores adversariales
formalizar la inhibición contrastiva como principio termodinámico de consolidación
```

Razón:

```text
El legacy es correcto.
El nativo consolidado es correcto, más rápido, menos filtrante y con mayor margen.
```

La decisión experimental vigente es:

```text
keep_native value=preserves_loaded_training_and_improves_runtime
```

---

## 13. Conclusión

El resultado central de esta etapa es que la arquitectura nativa deja de ser solo una optimización de rendimiento. Después de agregar sueño contrastivo, también supera al sustrato anterior en calidad de recuperación de conocimiento.

Resumen final:

```text
accuracy:         legacy=100.0% native=100.0%
leakage global:   legacy=9.1%   native=0.4%
margin global:    legacy=127.464 native=420.909
signal_ratio:     legacy=9.428  native=263.003
inferencia:       native ~3.7x a ~4.9x más rápido
cross_distractor: legacy=8.8% leakage, native=1.8% leakage
```

Por tanto:

```text
La investigación debe continuar sobre el motor termodinámico nativo consolidado.
```
