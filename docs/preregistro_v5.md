# Pre-registro v5 — campo predictivo (antes de DEV)

**Rama:** `exp/predictive-field-v5` (sale del tip de `exp/field-autonomy-v4`, no se mergea a `main`)
**Fecha:** 2026-10-10
Escrito antes de seeds DEV (`0xA400–0xA40F`) y CONFIRM (`0xB400–0xB40F`).

## 1. Pregunta

v4 §16 pasó E33 por primera vez (E3: 13/16 DEV, 9/16 CONFIRM) destilando un teacher de un paso. Un MLP del mismo tamaño también pasó. El error de un paso es 0.9 % y en h16 la accuracy queda en 0.26 / 0.22.

v5 no cambia la red. Cambia el objetivo. El campo se itera y la pérdida compara ese rollout con el futuro real de un episodio TRAIN en los horizontes 1, 2, 4 y 8. La CDT guarda los episodios y se tira antes del TEST. No hay teacher afín ni máscara de confianza.

## 2. Qué queda congelado de v4

- Generador, manifests, seeds, umbral de acierto (error relativo < 5 %), protocolo E33 (rotate+reflect, free-run, horizontes 1…64, región = racha contigua con accuracy ≥ 0.5).
- Red: `Dphi::gated_h`, H=136, init `seed ^ 0xD0` (la misma que E3).
- Presupuesto de DEV/CONFIRM: 256 000 pasos, batch 32, Adam 3e-3 → 3e-5 en coseno. Es el presupuesto de E3, no el lock de 8 000.
- Línea de base E3, no se reentrena: `artifacts/v4/distill_e/{dev,confirm}/`.

Smoke `0x5A00` usa 200 pasos y no es evidencia. Sirve para ver que el binario corre y que el gradiente de un paso de horizonte coincide con una diferencia finita (test de unidad).

## 3. Datos

Cada ejemplo TRAIN se convierte en un episodio de 9 estados aplicando la regla del generador sólo sobre ese ejemplo. `states[0]` es la entrada, `states[1]` es la etiqueta de un paso que v4 ya usaba, `states[2..8]` son el futuro. El manifest hashea esos frames. Un frame k≥2 que cae en la bola de un blanco TEST cuenta como fuga y el gate de esa seed no puede pasar.

La ablación `P2_shuffled` permuta, dentro de cada familia, los frames k≥1 entre episodios. El pasado no se mueve.

## 4. Brazos

| Condición | Red | Datos | Sal del muestreo |
|---|---|---|---|
| `P2_gated` | gated residual, 2590 params | episodios reales en CDT | `seed ^ 0x71` |
| `P2_mlp` | gated no residual, mismos params | los mismos episodios | `seed ^ 0x73` |
| `P2_shuffled` | igual que `P2_gated` | futuros permutados | `seed ^ 0x72` |

Durante el entrenamiento cada lectura de episodio incrementa el contador CDT. Antes del TEST el buffer se suelta. El TEST no tiene teacher forcing.

## 5. Gates

Por seed, `R_P2_gated`:

- región E33 ≥ 8,
- frames futuros filtrados = 0,
- queries de TEST = 0,
- proceso nuevo bit-exacto en accuracy de un paso (`|Δ| < 1e-9`, queries = 0).

El experimento pasa si eso ocurre en ≥ 9/16 seeds en DEV y en CONFIRM.

Mejora respecto de E3, además del gate: diferencia pareada (mismas seeds) de la región, o de la accuracy en h16, con IC95 bootstrap por encima de 0 en las dos fases. E3 ya está en 13/16 y 9/16. Pasar el umbral de 9 sin superar a E3 no es una mejora.

Meta secundaria, no sustituye al gate: media de accuracy en h16 ≥ 0.50 en las dos fases. Error de un paso del brazo gated ≤ 0.014.

`P2_mlp` se reporta siempre. Si iguala a `P2_gated`, la mejora no es de la forma residual. `P2_shuffled` tiene que quedar por debajo del brazo real en accuracy de un paso; si no, el contenido del episodio no transporta la señal.

## 6. Parada

No hay barrido de hiperparámetros. Si el gradiente del smoke de 200 pasos no es finito, se para y se documenta: no se recorta el gradiente ni se cambia el horizonte después de ver DEV. CONFIRM usa el mismo binario que DEV.
