# Experimento: campo sin tokens

**Fecha:** 13 de septiembre de 2026  
**Rama:** `exp/campo-sin-tokens`  
**Módulo:** `src/field_substrate.rs`  
**Qué es:** un sustrato mínimo `Ψ = (T, z, m, C)` — complejo simplicial, fasores, máscara de consolidación, cavidades Casimir. Cero `vocab_size`. Cero next-token.

## Qué no reclama

No consciencia. No ventaja frente a un transformer. No cifra del preprint de cuenca. No tok/s ni Gemma. T es un octaedro fijo (no crece ni se poda la geometría en v1; la poda apaga amplitud).

## Ciclo medido

1. Escribe un patrón de campo (hemisferios en fase 0 y π/2) sobre 12 aristas.
2. Hebb local en `L1`, handshake si la correlación de fase ≥ 0,80, Casimir entre caras rígidas.
3. Enmascara 4 aristas (`z = 0`), relaja 80 pasos (Langevin + atención geométrica).
4. Exige reconstruir la región oculta. Baseline: no relajar (error = 1).

## Comando

```bash
cargo test --release --lib field_substrate -- --nocapture
```

En el crate suelto de laboratorio:

```bash
cargo test --release --lib experiment_table -- --nocapture
```

Semillas `0..7`, Xoshiro256**. Determinista.

## Resultados (13 sep 2026, Linux x86_64, rustc 1.98.1, release)

| seed | F antes | F después | handshake | rígidas | Casimir | vivas | recon | baseline |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 20,150 | 4,549 | 0,950 | 10 | 6 | 12 | 0,293 | 1,000 |
| 1 | 20,150 | 4,553 | 0,949 | 10 | 6 | 12 | 0,299 | 1,000 |
| 2 | 20,150 | 4,617 | 0,949 | 10 | 6 | 12 | 0,299 | 1,000 |
| 3 | 20,150 | 4,559 | 0,950 | 10 | 6 | 12 | 0,293 | 1,000 |
| 4 | 20,150 | 4,541 | 0,949 | 10 | 6 | 12 | 0,297 | 1,000 |
| 5 | 20,150 | 4,604 | 0,949 | 10 | 6 | 12 | 0,302 | 1,000 |
| 6 | 20,150 | 4,563 | 0,950 | 10 | 6 | 12 | 0,291 | 1,000 |
| 7 | 20,150 | 4,625 | 0,949 | 10 | 6 | 12 | 0,299 | 1,000 |

Media: **recon = 0,296** vs **baseline = 1,000**. F cae ~20,15 → ~4,58. Handshake ~0,95. 10/12 aristas rígidas, 6 cavidades Casimir.

## Tests

- el estado no guarda vocabulario / IDs de token
- atención geométrica = 0 entre aristas que no comparten vértice
- handshake pone `m=1` solo si las fases traban
- relax no sube `F` (temperatura 0) sobre una perturbación del atractor
- predicción de campo gana al baseline de no hacer nada
- la tabla es determinista

## Lectura honesta

El campo reconstruye mejor que dejar la región en cero. Eso es memoria de atractor + geometría, no un LLM. El complejo no crece. Casimir aquí es un término que penaliza amplitud en aristas-puente entre caras ya rígidas, no un cálculo de modos de vacío.
