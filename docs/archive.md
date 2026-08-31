# Motores y módulos archivados respecto a la tesis de cuenca

Este inventario congela lo que **no alimenta** el resultado principal ni la
tesis v2 (memoria de *K* patrones con ΔR y curva de capacidad):

> Una consolidación de un patrón verificado deforma el paisaje fasorial y
> amplía de forma causal la cuenca de recuperación. El siguiente resultado
> es almacenar *K* patrones con ΔR ≥ −ε, no otro motor.

El código permanece en el árbol para reproducir trabajo previo. No es el
motor del preprint. Activar consumidores experimentales **no** forma parte
de cada push:

```powershell
cargo test --release --lib --features research
```

## Feature `research` (fuera del crate público por defecto)

| Módulo | Motivo |
|---|---|
| `emergent_cognition_training` | Gate con tests propios; el entrenador infinito reimplementa el suyo. No integrado. |
| `engine_comparison` | Comparador huérfano: sólo tests propios, sin binario asociado. |
| `native_plastic_symmetry_network` | Red plástica: ningún módulo ni binario la consume. |
| `cognitive_os` | SO cognitivo de laboratorio. Tests al 100 % no son evidencia de cuenca. |
| `cognitive_generalization_benchmark` | Protocolo de generalización estructural al 100 %. Fuera de smoke. |
| `advanced_cognitive_validation` | Validación adversarial al 100 %. Fuera de smoke. |
| `transformation_family_discovery` | Familias de transformación. Fuera de smoke. |

`cognitive_logistics` sigue compilando porque el visualizador de entrada lo
usa. Sus tests viven detrás de `research`.

## Compilan porque alguien los invoca; eso no los hace tesis

Estos módulos **no** están en el crate público *como resultado*. El hecho de
que el motor unificado o el entrenador gated los llamen no es justificación
para citarlos en el preprint ni para correr sus tests en cada push.

| Módulo / binario | Motivo |
|---|---|
| `variational_spin_liquid_vmc` | VMC Jastrow. Tests detrás de `research`. |
| `oxicuda_peps3d_backend` | Scaffold 3D-PEPS. Tests detrás de `research`. |
| `native_vmc_ratio_benchmark` | Benchmark del ratio Metropolis; no es el resultado principal. |
| `unified_spin_cognitive_engine` | Orquestador. El entrenador gated lo usa. Tests detrás de `research`. |
| `pyrochlore_graph_tensor_network` | Red tensorial de grafo. Tests detrás de `research`. |

## Binarios fuera del comando canónico

Hay decenas de binarios de chat, trainers, visualizadores y benchmarks. El
comando del resultado principal es uno:

```powershell
cargo run --release --bin native_consolidation_basin_experiment
```

El JSON canónico incluye `basin`, `baselines`, `bounded_forgetting` y
`capacity`. El resto se documenta en el README histórico y, a partir de la
higiene P5, pasa a `src/bin/archive/` o `examples/`.

## Qué no reactivar sin protocolo nuevo

- Entrenamiento infinito sin `symbolic_accuracy` y gate funcional en una
  corrida corta versionada.
- Citar 100 % de fixtures inyectados como evidencia de cognición.
- Tratar `paper.md` y el preprint fasorial como el mismo resultado.
- Correr 512/2048 nodos con *K*=1 y llamarlo escala de capacidad.
