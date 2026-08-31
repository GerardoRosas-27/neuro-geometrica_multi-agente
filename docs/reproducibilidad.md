# Reproducibilidad de cifras

Cada cifra del preprint vigente se regenera con un comando documentado.
Las cifras de sesión (ciclo 116350, 221,6 M de ejemplos, 115876 wake)
viven en `docs/paper.md` y en el apéndice de sesión del preprint; no son
la tabla principal.

## Resultado principal: deformación de cuenca

```powershell
cargo test --lib consolidation_basin_experiment -- --nocapture
cargo run --release --bin native_consolidation_basin_experiment
```

- semilla: `0xBA51_CD72_2026`
- nodos: 32
- presupuesto de iteraciones de evaluación: 300
- no hay checkpoint en Git: el fixture se regenera en el comando

## Holdout discriminante

```powershell
cargo test --release --lib scientific_holdout -- --nocapture
```

- semilla: `0xC0FF_EE42_2026` (ausente del fixture de desarrollo)
- nodos: 48
- corrupción: 0,20 / 0,45 / 0,55
- ruido de grafo: 15 % de aristas, ±0,40 rad

## Baselines externos

Incluidos en el comando canónico de cuenca. Test:

```powershell
cargo test --release --lib scientific_basin_baselines -- --nocapture
```

## Entrenador de desarrollo (gated)

No reanudar el ciclo infinito para «ver si pasa». Corrida corta versionada:

```powershell
$env:GEMMA_SPIN_MAX_CYCLES="9"
cargo run --release --bin native_gemma2_spin_infinite_trainer
```

El unbounded exige `GEMMA_SPIN_ALLOW_INFINITE=1` **y** un checkpoint con
`symbolic_accuracy > 0` y `functional_cognition_gate = true`. Ese checkpoint
no está en Git; si se publica, incluir hash SHA-256, comando y semilla, no
el número suelto.

Los artefactos viven en `data/gemma2_developmental_infinite_training/`
(ignorado por Git).

## Olvido acotado ΔR y escala

```powershell
cargo test --release --lib scientific_bounded_forgetting -- --nocapture
cargo test --release --lib scientific_basin_scale -- --nocapture
```

Escala completa (128 / 512 / 2048) vía `run_basin_scale_sweep` con
`BasinScaleConfig::default()`. 512 y 2048 no corren en smoke; el test CI
usa 128 nodos.
