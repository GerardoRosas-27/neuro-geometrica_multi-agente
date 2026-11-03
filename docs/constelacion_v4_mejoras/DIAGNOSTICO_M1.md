# Diagnóstico M1 — ¿por qué M1 ≈ m1_ctrl (Δ +0.0001)?

Rama `exp/constelacion-v4-mejoras` @ ce63ac3. Semilla de ajuste 0xC420 (`cst4m-run --phase smoke`, presupuesto completo), frozen.json. 0 DEV/CONFIRM relanzados, 0 TEST.

## Veredicto: no hay bug de cableado y M1 NO es idéntico a su control. Es (b)+(c): la mitad "acelerar con sorpresa" no puede actuar por la cota m_max = 1.0, y lo que queda (frenar hasta ×0.75) casi no mueve el error h16 en familias nuevas.

## Traza
- `arm_cfg("m1")` → `set_m1(γ=0.25, β=0.5, λ=0, μ, σ)` (mejoras.rs L24/L29); `m1_ctrl` igual pero γ=0 y `m1_const = 0.932` (L30).
- `needs_orb()` (mod.rs L117) se activa con γ>0 → ÓRBITA calcula `surp` por cuadro (L439–441) → `fast_general` (mod.rs L887–888): `m = clamp(1 + γ·tanh(z), m_min, m_max)`, paso `η_F·m` con momento β.
- `m_bounds` (mejoras.rs L14): `m_max = min(2, 2(1−β)/η_F)`. Con β=0.5 y η_F=1 da **m_max = 1.0**, tal como está en el PREREG (L7).
- En artefactos DEV 0xA400: hashes distintos (m1 871a…, ctrl e7c4…), `orb_calls` 187552 contra 0, `train_m_mean` 0.948 contra 0.932. El flag llega y la rama se ejecuta.

## Contadores (`diag_m1_instrumentacion.patch`, logs `diag_m1_0xC420_*.log`)
| | m1 conocidas | m1 nuevas | ctrl |
|---|---|---|---|
| pasos con m | 512 | 192 | 512 / 192 |
| media m | 0.942 | 0.990 | 0.932 (constante) |
| sd m | 0.078 | 0.034 | 0 |
| rango m | [0.755, 1.000] | [0.779, 1.000] | — |
| pasos saturados en m_max=1.0 | 265 (52 %) | **162 (84 %)** | — |

Efecto en la predicción h16, por episodio (m1 contra ctrl, misma semilla):
- conocidas (128 eps): 0/128 iguales bit a bit, media |Δlog err| 0.065, máx 0.48; med log −10.109 contra −10.113; acc 112 = 112.
- **nuevas (48 eps, métrica primaria)**: 0/48 iguales, media |Δlog err| **0.00043**, máx 0.0089; med log −0.55219 contra −0.55224; acc 4 = 4.

En DEV/CONFIRM pasa lo mismo: Δ por semilla m1−ctrl en new_med_log ∈ [−2.5e−4, +3.1e−4] y 0/16 iguales. "Idéntico" es sólo el redondeo a 4 decimales. M1−v4 también da +0.0001: en familias nuevas, con k=4, el error h16 (~O(1)) no depende de la dinámica rápida de W^F. Es el techo del modelo, no de M1.

## Recomendación
- No hay nada que arreglar. Re-correr M1 tal cual: **no vale la pena**.
- Si se quiere probar la idea de verdad hace falta un prereg nuevo (sería otra mejora, no un fix): β=0 (m_max=2, el aumento por sorpresa existe) o γ mayor, y una métrica donde W^F pese (k=8, conocidas). Costo: tune en 0xC4xx 4 semillas × 2–3 configs ≈ 10 min cada una; DEV+CONFIRM 32 semillas × 2 brazos × ~10 min ≈ 11 h de CPU, ~2 h con 6 jobs.
