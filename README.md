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

`.github/workflows/ci.yml` separa **smoke** (`cargo fmt`, `clippy -D warnings`,
`cargo test --release --lib -- --skip scientific`) del **gate científico**
(`cargo test --release --lib scientific`): multisemilla de cuenca y holdout
con patrón no inyectado.

El 100 % post-sueño del patrón inyectado es el techo esperado de escribir un
atractor. La métrica discriminante publicada es la recuperación del patrón no
inyectado (media ± desviación).

`.cargo/config.toml` fija `target-cpu=native` en local. Los binarios y los
tiempos de pared no son comparables entre máquinas.

## Puntos de entrada

Un binario por rol. El resto está en `src/bin/archive/`.

| Rol | Comando |
|---|---|
| Resultado principal | `cargo run --release --bin native_consolidation_basin_experiment` |
| Chat (demo) | `cargo run --release --bin native_gemma2_circadian_chat -- --chat dyamon` |
| Trainer (gated) | `GEMMA_SPIN_MAX_CYCLES=9 cargo run --release --bin native_gemma2_spin_infinite_trainer` |
| Visualizador | `cargo run --release --bin native_cognitive_sleep_visualizer` |

## Infraestructura (no es el resultado)

Chat Gemma 2 (demo de ingeniería, no claim del preprint):

```powershell
cargo run --release --bin native_gemma2_circadian_chat -- --chat dyamon
```

Entrenador de desarrollo: **gated**. Sin `GEMMA_SPIN_MAX_CYCLES` o
`GEMMA_SPIN_TRAIN_HOURS` no arranca. El infinito exige un checkpoint que
ya pase `symbolic_accuracy` y el gate funcional. Detalle:
[`docs/reproducibilidad.md`](docs/reproducibilidad.md).

Los módulos huérfanos (VMC, PEPS, red plástica, comparador, gate emergente no
integrado) viven detrás del feature `research` o en `docs/archive.md`.

```powershell
cargo test --release --lib --features research
```
