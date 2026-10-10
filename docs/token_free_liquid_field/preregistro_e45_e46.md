# Pre-registro E45 / E46 / paridad Rust (antes de ejecutar)

Fecha: 2026-10-10. Escrito y commiteado **antes** de correr cualquier semilla.

## Alcance de este ciclo

Siguiendo el "Orden obligatorio" del protocolo: (1) E45, (2) E46, (3) exportación y paridad Python→Rust con latencia oficial en Rust. E47+ quedan `NOT_RUN` en este ciclo salvo que se indique otra cosa en resultados.

## Decisiones ante ambigüedades (documentadas y marcadas)

1. **Encoder lingüístico.** Los docs piden "hidden raw" de un encoder congelado (Gemma). En este ciclo se usa un **proxy congelado sintético**: una red aleatoria fija (60→256→512, GELU/tanh) sobre el estado latente estructurado + 16 variables de "superficie" (paráfrasis/léxico) irrelevantes para la dinámica + ruido gaussiano σ=0.05. Motivo: permite splits limpios, ground truth exacto y cero riesgo de fuga lexical; Gemma real queda para E55 (encoder swap). **Esto limita la validez externa**: E45 aquí mide bottleneck sobre un hidden sintético, no sobre Gemma.
2. **Framework.** JAX 0.11.2 directo (+optax 0.2.8), que es el backend de la opción recomendada "Keras 3 + JAX". No se usa la capa Keras porque los modelos recurrentes con rollout son funciones puras más simples en JAX; Keras 3.15.1 está instalado pero no aporta nada aquí.
3. **Formato de exportación.** JSON versionado (`schema=tfl-v1`) con arrays f32 row-major, más SHA-256; Rust lo lee con serde_json (dependencia ya existente). safetensors no se añade.

## Tarea sintética (generador `tfl-gen-v1`)

- Estado latente: 6 slots, valores en Z₁₀.
- Contexto estructurado permitido `c` = id de operación (one-hot de 8): op0..op5 = `slot_k += 1 mod 10`; op6 = rotar slots a la izquierda; op7 = intercambiar pares (0↔1, 2↔3, 4↔5).
- Región OOD: estados con `z0 == z2` (≈10 %). TRAIN/DEV/TEST/COMPOSITION solo contienen inicios fuera de esa región; en TRAIN además todos los estados de la trayectoria quedan fuera.
- COMPOSITION: secuencias de 4 ops que contienen al menos un bigrama prohibido {(6,7),(7,6),(0,6),(6,0),(3,7)}; esos bigramas nunca aparecen en TRAIN.
- LONG_HORIZON: secuencias aleatorias de 64 ops (evaluadas en h=1,2,4,8,16,32,64).
- TRAIN: secuencias de longitud 4 (rollout supervisado h≤4). DEV y TEST: secuencias de longitud 4 de inicios no-OOD con semillas de muestreo distintas.
- Decoder: lineal Ψ→6×10 logits (no puede ser tabla de respuestas: 60·N+60 parámetros). Métrica principal: exact-match (los 6 slots correctos).

## Semillas

- DEV: semillas 0,1,2 evaluadas en DEV. Se seleccionan aquí todas las decisiones (dimensión, hiperparámetros).
- Confirmación: semillas 3,4,5 evaluadas en TEST/OOD/COMP/LONG **sin retocar nada**.

## E45 — umbrales

Métodos: raw (512), proyección aleatoria, PCA, encoder lineal entrenado, encoder no lineal entrenado (MLP). N ∈ {16,32,64,128,256,512}.
Calidad relacional = exact-match de `probe(ridge_op(Ψ_t))` vs z_{t+1}, con probe lineal y ridge por op ajustados en TRAIN.
- **N\*** = menor N del encoder no lineal entrenado cuyo exact-match relacional en DEV (media 3 semillas) está a ≤2 pp del mejor (cualquier método/N, raw incluido).
- **PASS** si N\* ≤ 128 y en confirmación (TEST y OOD) el encoder no lineal a N\* queda a ≤3 pp del mejor en todas las semillas o en la media con todas las semillas a ≤5 pp. `PARTIAL` si solo DEV o solo TEST. `FAIL` en otro caso.
- Fuga lexical: probe de las variables de superficie desde Ψ (R² reportado, no gate).

## E46 — umbrales

Modelos con presupuesto de parámetros ≈ igual (±15 %), mismo dataset, 3000 pasos Adam lr 2e-3, batch 256, pérdida = MSE(Ψ̂_t, Eφ(h_t)) + CE(decoder(Ψ̂_t)) sobre h=1..4:
A. MLP recurrente `Ψ' = MLP([Ψ,c])`; B. Liquid (LTC-like) `Ψ' = Ψ + Δt·(−Ψ/τ(Ψ,c) + tanh(W[Ψ,c]+b))`, τ = 0.5+softplus(·); C. SSM selectivo diagonal `Ψ' = a(c)⊙Ψ + B·gelu(W[Ψ,c])` con a=sigmoid; D. lineal local `Ψ' = A_c Ψ + b_c`.
Ejes:
1. Generalización/rollout = media de exact-match en OOD(h=4), COMP(h=4) y LONG(h=16).
2. Eficiencia/estabilidad = µs/paso en Rust y amplificación de perturbación a h=16.
- **Liquid PASS** si supera a cada competidor en el eje 1 por ≥2 pp (media de semillas, signo igual en ≥2/3 semillas) **y** no es peor en el eje 2 (µs/paso ≤1.1× del competidor o amplificación ≤ la del competidor), en DEV y repetido en confirmación. Si gana otro modelo, se reporta como tal.

## Paridad Python↔Rust (tolerancias fijadas ahora)

Corpus de paridad: 512 secuencias de DEV (nunca TEST), 16 pasos.
- paso 1: error abs máx ≤ 1e-4, media ≤ 1e-6·N… se fija como **media ≤ 1e-5**.
- paso 16: error abs máx ≤ 1e-3, p99 ≤ 1e-4.
- decisiones discretas (argmax por slot) idénticas en 100 % en paso 1 y ≥ 99.5 % en paso 16.
Si falla → `PORT_PARITY_FAIL` y el resultado no se atribuye al core Rust.
