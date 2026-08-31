# Motores y módulos archivados respecto a la tesis de cuenca

Este inventario congela lo que **no alimenta** el resultado principal:

> Una consolidación CDT de un patrón verificado deforma el paisaje fasorial
> y amplía de forma causal la cuenca de recuperación.

El código permanece en el árbol para reproducir trabajo previo. No es el
motor del preprint. Activar consumidores experimentales:

```powershell
cargo test --release --lib --features research
```

## Feature `research` (fuera del crate público por defecto)

| Módulo | Motivo |
|---|---|
| `emergent_cognition_training` | Gate con tests propios; el entrenador infinito reimplementa el suyo. No integrado. |
| `engine_comparison` | Comparador huérfano: sólo tests propios, sin binario asociado. |
| `native_plastic_symmetry_network` | Red plástica: ningún módulo ni binario la consume. |

## Congelados en documentación (siguen compilando: el motor unificado los invoca)

| Módulo / binario | Motivo |
|---|---|
| `variational_spin_liquid_vmc` | VMC Jastrow validado en tests; no entra en el protocolo de cuenca. |
| `oxicuda_peps3d_backend` | Scaffold 3D-PEPS; no cubre todos los enlaces físicos. |
| `native_vmc_ratio_benchmark` | Benchmark del ratio Metropolis; no es el resultado principal. |
| `unified_spin_cognitive_engine` | Orquestador CDT–spin–RQM–EPR. Infraestructura, no el paper de cuenca. |
| `pyrochlore_graph_tensor_network` | Red tensorial de grafo; mapa físico, no el experimento causal. |

## Binarios fuera del comando canónico

Hay decenas de binarios de chat, trainers, visualizadores y benchmarks. El
comando del resultado principal es uno:

```powershell
cargo run --release --bin native_consolidation_basin_experiment
```

El resto se documenta en el README histórico y, a partir de la higiene P5,
pasa a `src/bin/archive/` o `examples/`.

## Qué no reactivar sin protocolo nuevo

- Entrenamiento infinito sin `symbolic_accuracy` y gate funcional en una
  corrida corta versionada.
- Citar 100 % de fixtures inyectados como evidencia de cognición.
- Tratar `paper.md` y el preprint fasorial como el mismo resultado.
