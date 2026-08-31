# CDT-RQM-EPR · consolidación de cuenca

Laboratorio nativo en Rust (`cdt_rqm_epr`). **Una tesis, un motor, un paper.**

> Una consolidación CDT de un patrón verificado deforma el paisaje fasorial
> y amplía de forma causal la cuenca de recuperación.

Ese es el resultado principal. El resto del crate es infraestructura, demos
de ingeniería o trabajo en curso. No se afirma consciencia, cognición general,
ventaja en julios ni generalización conceptual.

## Resultado principal

```powershell
cargo test --lib consolidation_basin_experiment -- --nocapture
cargo run --release --bin native_consolidation_basin_experiment
```

Protocolo: snapshot pre, consolidación transaccional de un patrón ya
verificado, mismos cues y solver en post. El flip global Z₂ cuenta como
fallo. Es evidencia interna de deformación de cuenca, no de cognición
emergente.

- Paper vigente: [`docs/paper_inferencia_fasorial_consolidacion_cdt.md`](docs/paper_inferencia_fasorial_consolidacion_cdt.md)
- Bitácora histórica (no es el mismo resultado): [`docs/paper.md`](docs/paper.md)
- Motores congelados respecto a esta tesis: [`docs/archive.md`](docs/archive.md)
- Revisión de alcance: [`docs/revision_proyecto.md`](docs/revision_proyecto.md)

## Manuscritos

`paper_inferencia_fasorial_consolidacion_cdt.md` es el preprint. `paper.md`
conserva cifras de sesión, currículos y motores paralelos como apéndice de
compatibilidad; no citarlo como el resultado vigente.

## Integración continua

`.github/workflows/ci.yml` ejecuta `cargo fmt`, `clippy -D warnings` y
`cargo test --release --lib` (incluye el gate de cuenca).

`.cargo/config.toml` fija `target-cpu=native` en local. Los binarios y los
tiempos de pared no son comparables entre máquinas.

## Infraestructura (no es el resultado)

Chat Gemma 2 (demo de ingeniería, no claim del preprint):

```powershell
cargo run --release --bin native_gemma2_circadian_chat -- --chat dyamon
```

Entrenador de desarrollo: está **gated** (véase P4). No reanudar el ciclo
infinito para «ver si pasa». Los módulos huérfanos (VMC, PEPS, red plástica,
comparador, gate emergente no integrado) viven detrás del feature `research`
o en `docs/archive.md`.

```powershell
cargo test --release --lib --features research
```
