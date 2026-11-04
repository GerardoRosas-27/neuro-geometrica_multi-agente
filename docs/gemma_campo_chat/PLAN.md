# gemma_campo_chat: preregistro

Rama: `exp/gemma-campo-chat` (sale de `origin/main`). Este documento se fija **antes** de ver cualquier número de Gemma real.
Los números de `smoke/` salen de un Gemma2 **diminuto y aleatorio** con tokenizador de bytes. Sirven para probar la tubería y **no son resultados**.

## Pipeline
1. **Gemma 2 2B congelado** (`google/gemma-2-2b`, d=2304, 26 capas). Se guarda el estado oculto por token de la capa media `L=13` (`hidden_states[13]`, salida de la capa 13; configurable con `--layer`).
2. **Codificador E congelado**: PCA blanqueada 2304→`d_field`=64 (configurable). Se ajusta **solo en train** (covarianza en f64).
3. **Sustrato de campo** que predice z_{t+1} (y k pasos por rollout autoregresivo):
   * **Lenta**: Δz = Wᵀφ, con φ = [1 | z_t..z_{t−3} (C=4) | términos cuadráticos z_t⊗z_t (i≤j)], F = 1+256+2080 = 2337.
     - Init: ridge M7 **por grado**, con λ₁ (lineal) y λ₂ (cuadrático) en la rejilla {1e-6..1e-1} elegidos en **val**. Se resuelve en forma cerrada en f64 (Cholesky) con hasta 50k muestras de train.
     - Refinado: SGD con momentum en **GPU (WGSL)** sobre todo train, con el mismo objetivo y los mismos λ. Hay checkpoint por época. Se queda el mejor según val.
   * **Rápida**: una matriz A (d×d) por conversación que aprende online (NLMS) el residuo de la lenta: `A ← γA + η (r − Az) zᵀ/(|z|²+ε)`. Es solo de inferencia y causal (A se actualiza después de ver z_{t+1}). Durante el rollout k-paso A queda congelada. η∈{0,…,0.3} y γ∈{1,.99,.95} se eligen en val. η=0 apaga la parte rápida, así que se puede rechazar.
   * **Lo que NO se usa** del motor bio_quantum_engine: Brenier/φ_j(q) de puntos 2D, Dale exc/inh, SWR "sleep" (devolvía un valor fijo) y `condition_fast_weights` (inyectaba la respuesta). Se reutiliza el patrón wgpu (adaptador AMD, buffers, readback) y la idea de parte rápida + parte lenta.
4. **Decodificador D**: lineal z→h (2304), ridge en train. Después vienen las capas L+1..26 de Gemma y el LM head, sin cambios (se sustituye la salida de la capa L con un forward hook).

## Entrenamiento GPU sin carreras (requisito 1)
`gemma_campo/src/train.wgsl`:
* K1 `k_forward`: un hilo por (b,o) escribe solo `err[b,o]`.
* K2 `k_update`: un hilo por (f,o) suma el gradiente sobre el batch en un registro local y escribe solo `W[f,o]` y `V[f,o]`.
* K3 `k_reduce`: reducción en árbol por workgroup a `partial[wg]`. La CPU suma los parciales en f64.

No hay `+=` sobre memoria compartida entre hilos, no hay atómicas y no hay valores devueltos fijos. El orden de suma es fijo, así que el resultado es determinista (se comprueba).
Tests: `cargo test` (`tests/gpu.rs`: pérdida y pesos GPU == gemelo CPU `cpu::sgd_step_ref`) y `gemma_campo check-gpu`.

## Splits
* Por **conversación**, con una permutación de semilla 12345 sobre el dataset. Fracciones 80/10/10.
* Por defecto: `HuggingFaceTB/everyday-conversations-llama3.1-2k` (`train_sft`), 600 conversaciones (480/60/60), máximo 384 tokens cada una.
* `extract.py` comprueba que los conjuntos de conversaciones son disjuntos y guarda los ids en `extract_meta.json`.
* Todo hiperparámetro (λ, η, γ, época, d_field, L) se fija en **val**. **Test se evalúa una sola vez.**

## Controles y métricas (solo TEST)
* Controles:
  - **gemma**: Gemma sola.
  - **repeat**: repetir el último estado, D(E(h_{t−1})).
  - **oracle**: el cuello de botella sin predicción, D(E(h_t)).
  - **lenta**: el campo sin la parte rápida (solo en z-MSE).
* z-MSE a 1 paso y a k=1..8 pasos (rollouts que arrancan cada 4 tokens), en el espacio z blanqueado. Primero se promedia dentro de cada conversación.
* Perplejidad: NLL/token sobre las posiciones t≥1 con la salida de la capa L sustituida en **todas** las posiciones t≥1 (teacher forcing). Se informa ΔNLL campo−gemma, campo−repeat, campo−oracle, oracle−gemma y repeat−gemma.
* Estadística: bootstrap **pareado por conversación**, 10000 remuestreos, semilla 12345. Se da la media, el IC95 % y P(Δ≤0).
* Textos de muestra: argmax por posición para cada condición (3 conversaciones de test) y `chat_demo.py` en modo campo frente a modo gemma.

## Hipótesis (direccionales; "se cumple" = IC95 % entero < 0)
* **H1 (primaria)**: z-MSE 1-paso campo < repeat.
* **H2**: z-MSE k-paso campo < repeat para k=2, 4, 8.
* **H3**: z-MSE 1-paso campo < lenta (la parte rápida aporta).
* **H4 (primaria)**: ΔNLL campo−repeat < 0.
* **H5 (descriptiva, sin umbral de éxito)**: ΔNLL campo−gemma y campo−oracle. Se espera que sean > 0: el campo predice h_t **sin ver el token t**, así que pierde la identidad del token actual. Se informa tal cual.

## Criterios de parada
* Si en val la lenta (ridge) no baja de repeat en z-MSE 1-paso, no se pasa a test. Se informa como negativo.
* Si oracle−gemma > 1 nat/token en val, el cuello d_field es demasiado estrecho. Se sube d_field (128, 256) **solo con val** y se anota aquí antes de test.
* Si la pérdida no es finita o el check GPU==CPU falla, se para. No hay resultados con un entrenador no verificado.
* Test no se repite. Cualquier cambio posterior es exploratorio y se etiqueta como tal.
