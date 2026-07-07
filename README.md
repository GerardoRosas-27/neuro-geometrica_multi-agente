# CDT-RQM-EPR

Sustrato experimental causal-relacional en Rust.

La rama actual conserva solo el sustrato **CDT-RQM-EPR**:

- `CDT-Graphity`: hardware causal foliado.
- `RQM`: software relacional dependiente del observador.
- `EPR`: enlaces coherentes de correlación computacional.
- `Dvali criticality`: control `alpha * N ~= 1`, species bound y temperatura `1 / sqrt(N)`.
- `Ising/FEP`: selección energética, acción Regge discreta y compresión geométrica.

El paper técnico está en:

```text
docs/paper.md
```

## Estado Consolidado

El estado entrenado vigente es:

```text
data/cdt_rqm_evolutionary_kept.cdt_rqm
```

## Evaluar

```powershell
cargo run --bin cdt_rqm_consolidated_evaluation
```

Resultado esperado actual:

```text
suite=PASS
normal accuracy=100.0%
typed_memory accuracy=100.0%
causality_violations=0
```

## Entrenar / Continuar

Entrenamiento continuo de conceptos, causalidad, habilidades y episodios:

```powershell
$env:CDT_RQM_INFINITE_OUTPUT="data/cdt_rqm_evolutionary_kept.cdt_rqm"; cargo run --bin cdt_rqm_infinite_concept_trainer
```

Entrenamiento compacto con sueño/consolidación EPR:

```powershell
$env:CDT_RQM_EPR_SMALL_OUTPUT="data/cdt_rqm_evolutionary_kept.cdt_rqm"; cargo run --bin cdt_rqm_epr_small_sleep_trainer
```

Consolidación tipo sueño sobre estado existente:

```powershell
$env:CDT_RQM_EPR_SLEEP_STATE="data/cdt_rqm_evolutionary_kept.cdt_rqm"; cargo run --bin cdt_rqm_epr_sleep_consolidate
```

## Validadores Conservados

```powershell
cargo run --bin cdt_rqm_consolidated_evaluation
cargo run --bin cdt_rqm_epr_information_validation
cargo run --bin cdt_rqm_rqm_quantum_validation
cargo run --bin cdt_rqm_lambda_regge_validation
cargo run --bin cdt_rqm_thermo_time_validation
cargo run --bin cdt_rqm_self_awareness_validation
cargo run --bin cdt_rqm_hawking_prune_validation
cargo run --bin cdt_graphity_substrate_experiment
```

## Arquitectura

```text
boundary activa + observador
  -> RQM collapse relativo
  -> EPR synchronization
  -> CDT causal step
  -> Graphity / Regge / FEP
  -> estado compacto validado por memoria
```

El criterio de conservación de mejoras es:

```text
preservar memoria
reducir energia/fuga/costo
no inflar geometria
mantener causality_violations = 0
```
