# CDT-RQM-EPR · consolidación de cuenca

Laboratorio nativo en Rust (`cdt_rqm_epr`). **Una tesis, un motor, un paper.**

> Una consolidación CDT de un patrón verificado deforma el paisaje fasorial
> y amplía de forma causal la cuenca de recuperación.

Ese es el resultado **previo de un slot** (tabla 7.4 del preprint 0.6). El
resto del crate es infraestructura, demos de ingeniería o trabajo en curso.
No se afirma consciencia, cognición general, ventaja en julios ni
generalización conceptual.

**Tesis v2 (siguiente resultado, falsable):** el sistema almacena *K*
patrones verificados sobre el mismo sustrato fasorial. Consolidar el patrón
*k* deja la retención del conjunto retenido *A* en ΔR ≥ −ε. La curva
*K(N, ρ)* se publica junto a Hopfield y Hebb sobre las mismas cues. Si
ΔR < −ε o la curva no discrimina, el commit se rechaza y no hay resultado.
No se inventan cifras: las mide
`native_consolidation_basin_experiment` (`bounded_forgetting` + `capacity`).
Arquitectura: [`docs/arquitectura_siguiente_ciclo.md`](docs/arquitectura_siguiente_ciclo.md).

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
(`cargo test --release --lib scientific`): multisemilla de cuenca, holdout
desmezclado, ΔR como gate y curva de capacidad. Los protocolos cognitivos al
100 % no corren en smoke; viven detrás de `--features research` y **no** se
lanzan en cada push.

El 100 % post-sueño del patrón inyectado es el techo esperado de escribir un
atractor. La métrica discriminante del slot único es el holdout; la de la
tesis v2 es ΔR y *K_max(N, ρ)* frente a Hopfield/Hebb.

`.cargo/config.toml` fija `target-cpu=native` en local. Los binarios y los
tiempos de pared no son comparables entre máquinas.

## Experimentos 8–10 (líquido)

Protocolo reproducible para invariancia de representación, composición de relaciones y predicción dinámica: [`docs/experimentos_8_9_10_liquido.md`](docs/experimentos_8_9_10_liquido.md).

## Etapa 2 — autonomía del campo (E18–E30)

Protocolo de aprendizaje de reglas desde experiencia consolidada (FIELD_ONLY, anti-leakage, CDT como experiencia no como lookup): [`docs/etapa_2_autonomia_campo_experimentos.md`](docs/etapa_2_autonomia_campo_experimentos.md) — **§29+ Clean-Room v2 prevalece**.

- **Clean-Room v2 (vigente):** módulo `src/field_autonomy_stage2_v2.rs`, seeds `0xA300–0xA307`, resultados [`docs/resultados_etapa_2_cleanroom_v2.md`](docs/resultados_etapa_2_cleanroom_v2.md). Smoke: `cargo run --example smoke_stage2_v2`. Suite DEV: `cargo run --release --example run_stage2_v2_dev`.
- **Histórico / pre-cleanroom:** `src/field_autonomy_stage2.rs` + [`docs/resultados_etapa_2_autonomia.md`](docs/resultados_etapa_2_autonomia.md) (seeds `0xE1800..`) — no mezclar con v2.

## Ciclo siguiente (rama experimental)

Plan de mejoras de producto + experimentos E31+ / endurecimiento E13–E22–E30 (docs-first, **no** merge a main):
[`docs/propuesta_mejoras_experimentos_ciclo_siguiente.md`](docs/propuesta_mejoras_experimentos_ciclo_siguiente.md).
Plantilla de resultados (vacía hasta corrida real): [`docs/resultados_ciclo_siguiente.md`](docs/resultados_ciclo_siguiente.md).
Rama: `exp/mejoras-siguiente-ciclo` (desde `origin/main` @ `e05cef1`).

## Experimentos 11–17 (campo entrenable)

Protocolo para entrenar el sustrato de campo con el LLM congelado como periferia, incluyendo invariancia lingüística real, relaciones, aprendizaje incremental, dinámica predictiva, estados no observados y controles anti-leakage: [`docs/experimentos_11_17_campo_entrenable.md`](docs/experimentos_11_17_campo_entrenable.md). Resultados: [`docs/resultados_experimentos_11_17.md`](docs/resultados_experimentos_11_17.md). Hallazgos E13/E15 (WIP): [`docs/hallazgos_e13_e15_endurecimiento.md`](docs/hallazgos_e13_e15_endurecimiento.md).

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

Los módulos de laboratorio (OS cognitivo, generalización 100 %, VMC, PEPS,
unificado de test, red plástica) viven detrás del feature `research` o en
[`docs/archive.md`](docs/archive.md). Eso **no** se corre en cada push:

```powershell
cargo test --release --lib --features research
```
