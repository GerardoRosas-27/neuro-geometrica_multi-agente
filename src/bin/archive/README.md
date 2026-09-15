# Binarios archivados

No son puntos de entrada. Cargo no los descubre aquí (`src/bin/archive/*.rs`
no es `src/bin/*/main.rs`).

Puntos de entrada vigentes, un rol cada uno:

- chat: `native_gemma2_circadian_chat`
- cuenca: `native_consolidation_basin_experiment`
- trainer: `native_gemma2_spin_infinite_trainer` (gated)
- visualizador: `native_cognitive_sleep_visualizer`
- campo sin tokens (paralelo): `native_field_substrate_experiment`
- entrenador consola (dataset enorme): `native_field_console_trainer`

Para reactivar uno, muévelo de vuelta a `src/bin/`.
