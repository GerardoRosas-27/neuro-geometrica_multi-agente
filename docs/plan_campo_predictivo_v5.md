# Plan — campo predictivo v5

**Rama:** `exp/predictive-field-v5`
**Base:** tip de `exp/field-autonomy-v4` (incluye §16 DIST_E y §18 Ψ64-Gemma). No se mergea a `main`.

## Por qué esta rama

E3 es el mejor número de v4 y el primer PASS de E33. Lo consigue imitando un teacher de un paso, y un MLP del mismo tamaño lo iguala. El horizonte 16 sigue en ~0.2. Esta rama entrena la cantidad que E33 mide: el campo iterado contra el futuro del episodio, con la CDT como buffer de episodios y sin teacher de reglas.

El detalle operativo está pre-registrado en `docs/preregistro_v5.md`.

## Qué se reutiliza

Generador, auditor, seeds, E33, checkpoint de proceso nuevo, `Dphi::gated_h`. Lo líquido, Ψ64 y Gemma no entran. El método nuevo es `Dphi::train_horizon`: backprop a través de la iteración, pérdida relativa en k = 1, 2, 4, 8.

## Fases

1. **P2 (esta implementación).** Tres brazos: campo gated, MLP gated, episodios con el futuro permutado. DEV y luego CONFIRM, sin retocar.
2. **P3, sólo si P2 supera a E3 en las dos fases.** Predictor con encoder de promedio móvil y stop-gradient, si aparece colapso. No está en este código.
3. **P4, sólo si el un paso ya es < 1 % y h8 sigue bajo.** Dos escalas de tiempo. No está en este código.

## Cómo correr

```text
cargo run --release --bin run_predictive_field_v5 -- run --phase smoke
cargo run --release --bin run_predictive_field_v5 -- run --phase dev
cargo run --release --bin run_predictive_field_v5 -- run --phase confirm
```

Artefactos en `artifacts/v5/<phase>/`. Smoke no es evidencia.

## Estado: DEV cerrado (FAIL). CONFIRM incompleto

DEV terminó el 14 de octubre de 2026 (SHA de corrida `a0dbb169`, tip de rama `19bf9ac`). Artefactos en `artifacts/v5/dev/`.

Gate pre-registrado: `R_P2_gated` ≥ 9/16. Resultado: **4/16 PASS** (FAIL). `R_P2_mlp` también 4/16. `R_P2_shuffled` 0/16. Línea base `R_E3_v4` en las mismas seeds: **13/16**. P2 no supera a E3; **P3 no aplica**.

CONFIRM quedó a medias (seeds/checkpoints parciales en `artifacts/v5/confirm/`, sin `summary.md`). El protocolo pedía CONFIRM con el mismo binario; no se retocaron umbrales ni presupuesto.

Smoke sigue sin ser evidencia.
