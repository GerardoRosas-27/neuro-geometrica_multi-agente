# Pre-registro — comparación cara a cara v4 (§16 E3, main) vs v5 (P2)

**Rama:** `exp/predictive-field-v5` · **Fecha:** 2026-10-10/11 · escrito **antes** de ejecutar DEV y CONFIRM.
No se mergea a `main`. Este documento extiende `docs/preregistro_v5.md`; sus gates y umbrales no cambian.

## Por qué hace falta

`docs/preregistro_v5.md` comparaba P2 contra los artefactos guardados de E3 (`artifacts/v4/distill_e/`),
generados en otro proceso, con otro binario y sin tiempos de entrenamiento ni latencia. Para condiciones
iguales, E3 se **reentrena en el mismo proceso, misma seed y mismo split sellado** que v5, con la receta de
`main` (`v4_distill::train_onestep_trusted_b`, idéntica en ambas ramas salvo un refactor sin efecto).

## Brazos (todos por seed, mismo `generate_and_seal(seed)`, mismo `Auditor`, mismo harness E33)

| Brazo | Receta | Red | Pasos Adam (batch 32, 3e-3→3e-5 coseno) | Evaluaciones Dφ por ejemplo |
|---|---|---|---|---|
| `E3_v4` | v4 §16: CDT → consolidación → máscara de confianza (τ=0.01) → Dφ un paso, 50 % replay | `gated_h` H=136, 2590 params, init `seed^0xD0` | 256 000 | 1 |
| `P2_gated` | v5: rollout contra futuro del episodio, pérdida en k=1,2,4,8 | igual | 256 000 (mismos pasos) | 8 |
| `P2_gated_eqc` | v5, **cómputo igualado** a E3 (pasos/8) | igual | 32 000 | 8 |
| `P2_mlp`, `P2_shuffled` | controles de `preregistro_v5.md` | igual (mlp: no residual) | 256 000 | 8 |

Params iguales (2590) en todos. Presupuesto: `P2_gated` iguala pasos de optimizador; `P2_gated_eqc` iguala
forward+backward de Dφ (≈ cómputo). Ambos se reportan; ninguno se elige a posteriori.

## Seeds y fases

DEV `0xA400–0xA40F`, CONFIRM `0xB400–0xB40F` (16 + 16). DEV y luego CONFIRM con el **mismo binario/SHA**, sin
retocar nada entre fases. Smoke `0x5A00` (200 pasos) solo comprueba que corre.

## Métricas (idénticas para todos los brazos)

- Exactitud de un paso en TEST (error relativo < 5 %) y error relativo medio de un paso.
- E33: exactitud en h = 1, 2, 4, 8, 16, 32, 64 (rotate+reflect, free-run) y región contigua (≥ 0.5).
- OOD/composición: desglose `onestep_breakdown` por transformación (incl. `compose_rot_scale`, `radial_warp`,
  `reflect`) y `in_dist` vs `interp_hole`.
- Auditoría de memoria cero: queries TEST = 0, fugas = 0 (frames futuros en bola TEST para v5; flags CDT/replay para E3).
- Proceso nuevo: checkpoint de **cada** brazo evaluado en un proceso hijo, |Δacc| < 1e-9 y 0 queries.
- Latencia Rust: mediana µs por `Dφ.step` (5 pasadas sobre TEST). Tiempo de entrenamiento: s de pared por brazo.

## Criterios de decisión (cerrados)

1. Gate por brazo (como v4/v5): región ≥ 8, limpio, proceso nuevo OK. Cuenta X/16 por fase.
2. **v5 gana** a v4 solo si `P2_gated` − `E3_v4` (pareado por seed) en región **o** en exactitud h16 tiene IC95
   bootstrap (10 000 remuestreos, semilla fija) > 0 en DEV **y** en CONFIRM, **sin** empeorar la exactitud de un
   paso en más de 0.02 (media). Igual análisis para `P2_gated_eqc` (cómputo igualado), reportado aparte.
3. **v4 gana** si la diferencia tiene IC95 < 0 en ambas fases. En otro caso: **empate / no concluyente**.
4. Coste: se reporta el cociente de tiempo de entrenamiento; una ventaja de v5 que solo exista a 8× cómputo se
   declara como tal.
5. Sin barridos ni cambios tras ver DEV. Si algo falla, se documenta como negativo.

Análisis: `scripts/v5_compare.py` sobre `artifacts/v5/{dev,confirm}/metrics.csv`. Resultado en
`docs/comparacion_v4_v5.md`.

## Enmienda de infraestructura (2026-10-12) — sin cambios metodológicos

Cambio **solo de infraestructura**; ninguna constante, semilla, muestreador, schedule, arquitectura, métrica,
criterio ni script de análisis cambia. Motivo: el box se reinicia cada pocas horas y las seeds (~2.4 h) no
terminaban.

- Checkpoint intra-seed determinista (`src/train_ckpt.rs`): cada `V5_CKPT_SECS` (300 s por defecto) y al final de
  cada brazo se guarda el estado completo del bucle (paso siguiente, RNG xoshiro, Adam m/v/t, parámetros, pérdida,
  max-grad, contadores de consultas del bucle y los `hits` del auditor de E3) en bits f64 exactos, en
  `artifacts/v5/<fase>/train_ckpt/`. Al reanudar el resultado es bit-idéntico al de una ejecución sin interrumpir
  (verificado en smoke con 8 kills aleatorios: `metrics.csv`, manifest y summary idénticos salvo `train_s`/`lat_us`,
  que son tiempos de pared). Los `hits` de E3 se acumulan como conjunto (solo alimentan un `HashSet`/flag).
- `train_s` de un brazo reanudado mide solo el tramo final (tiempo de pared); el criterio 4 (coste) debe leerse
  con seeds no interrumpidas o como aproximado. `lat_us` no se ve afectado.
- La caché por seed acepta resultados de los commits `0ecd4e9` y `3cdf733` (`RESULT_COMPAT_SHAS`), cuyo código de
  resultados es idéntico (solo difieren en un `#[allow(clippy)]` y el script de comparación).
- `V5_JOBS` opcional fija el nº de seeds en paralelo (cada seed es independiente y monohilo; no afecta resultados).
  Medido: 8 trabajos dan el mayor throughput en 8 núcleos, se mantiene el valor por defecto.
- `scripts/v5_resume.sh` relanza DEV→CONFIRM si no está corriendo.
