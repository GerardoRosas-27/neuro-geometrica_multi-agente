# Protocolo — E13/E15 re-run ciclo siguiente

**Seeds:** `0xE1100–0xE1107` (8). Smoke previo `0xE1100` no sustituye la suite.  
**Rama:** `exp/mejoras-siguiente-ciclo`  
**Confirmation:** no aplica (familia E11xx histórica).

## E13 — holdout is-a

1. Partir de `docs/hallazgos_e13_e15_endurecimiento.md`.
2. Ablation matrix: baseline / −residual / −animal_ball / −tax_ancestors.
3. Eval `(cue,label)` con ancestros; **prohibido** soft-target en cues holdout.
4. Métricas: `D_dynamic` seen/unseen; holdout bits; leakage; RQM/table holdout (=0).
5. Runner: `cargo run --example smoke_e13_e15` (ampliar a loop 8 seeds en follow-up si hace falta).

**PASS / PARTIAL / FAIL:** ver propuesta §4.1.

## E15 — composición factorial

1. Re-run suite 8 seeds; opcional holdouts sujeto/verbo OOD.
2. Métricas: structure, rank_acc, margin_E, d_man, stab.
3. **PASS:** structure 3/3 y rank≥0.70 en ≥6/8.

## Thin wrapper

`examples/smoke_e13_e15.rs` ya existe; usarlo como smoke 1-seed. Suite 8 seeds = mismo harness en loop (CLI), no botón Pruebas UI.
