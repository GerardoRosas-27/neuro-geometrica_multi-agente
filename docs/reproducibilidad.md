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

## Olvido acotado ΔR, capacidad y escala

```powershell
cargo test --release --lib scientific_bounded_forgetting -- --nocapture
cargo test --release --lib scientific_capacity_curve -- --nocapture
cargo test --release --lib scientific_basin_scale -- --nocapture
cargo run --release --bin native_consolidation_basin_experiment
```

El binario canónico imprime un JSON con `basin` (tabla 7.4, un slot),
`baselines`, `bounded_forgetting` (ΔR) y `capacity` (*K(N, ρ)* vs Hopfield
y Hebb). No se publican cifras inventadas: las del preprint 0.6 siguen
siendo el slot único; la tesis v2 se lee del JSON.

El test CI de 128 nodos aserta `decision` **y** `K_max ≥ 2`. 512 y 2048
quedan como nightly y **no** se corren con *K*=1. El feature `research`
no entra en cada push:

```powershell
cargo test --release --lib --features research
```

## Campo sin tokens (paralelo al preprint)

No regenera cifras de cuenca. El comando imprime la tabla contra Hopfield,
Hebb y vecinos, y guarda checkpoints por dataset en
`data/field_substrate_training/<id>/` (`latest.json`, `checkpoints/`,
`metrics.jsonl`).

```powershell
cargo test --release --lib field_substrate -- --nocapture
cargo run --release --bin native_field_substrate_experiment
```

Semillas `0..7`. Dataset por defecto: 32 patrones sintéticos, resume
activado. Ctrl+C persiste `latest.json` y sale. Un `dataset_id` distinto
no mezcla checkpoints.

```powershell
cargo run --release --bin native_field_substrate_experiment -- --dataset-size 4096 --dataset-id synthetic-4k
```

Encoder-sonda: Gemma congelada → `z`, decoder al banco. Semilla `0xE4C0`, 120 pasos:

```powershell
cargo test --release --lib field_encoder -- --nocapture
cargo run --release --bin native_field_substrate_experiment -- --no-train --no-resume
```
