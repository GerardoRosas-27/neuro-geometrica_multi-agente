# Pre-registro ciclo 2 — E46b (Gemma real, tarea no lineal), E47, E48 y H2H vs v4

Fecha: 2026-10-10. Commiteado **antes** de entrenar/evaluar cualquier modelo de este ciclo (la extracción de hidden states de Gemma ya está en curso; es un paso determinista sin decisiones).

## Herramientas

JAX 0.11.2 + Optax 0.2.8 (ya fijadas en el ciclo 1). Motivo: los modelos son funciones de estado puras con rollout; JAX permite compilar el bucle de entrenamiento completo (`lax.scan`) y entrenar varias semillas a la vez (`vmap`) sin reescribir los modelos. Keras/TF no aportan nada adicional aquí y añadirían una segunda semántica que portar a Rust. Speedup medido en `artifacts/token_free/speedup.json` (no es un gate).

## E46b — tarea no lineal con hidden real de Gemma (`tfl-gen-v2`)

- Estado (a,b,c) ∈ Z₁₀³. Texto en español con 3 plantillas de paráfrasis y números en palabras (`texts_v2.py`); plantilla elegida al azar por aparición.
- Encoder lingüístico: Gemma-2-2B-it Q3_K_L, media del hidden final (2304), congelado, cacheado (`gemma_v2.f32`, SHA-256 en el manifest). Procedimiento idéntico al de v4 E43 (infra genérica).
- Operaciones (contexto = id): op0 a←a·b; op1 b←b²+1; op2 c←a+c²; op3 swap(a,c); op4 a←a+1; op5 b←a·c+b (todo mod 10). No lineales en one-hot salvo op3/op4.
- OOD: a == c. COMPOSITION: bigramas prohibidos en TRAIN {(0,5),(5,0),(1,2),(3,0)}. TRAIN/DEV/TEST/OOD/COMP: secuencias de 4 ops; LONG_HORIZON: 32 ops (h=1,2,4,8,16,32). TRAIN excluye estados OOD en toda la trayectoria.
- FieldEncoder: MLP 2304→256→64 (N=64 fijado por E45, no se re-selecciona), entrenado con CE de decodificación en estados TRAIN, 2000 pasos; luego congelado. Decoder lineal 64→30.
- Modelos (~33k parámetros ±15 %, 3000 pasos Adam 2e-3, batch 256, pérdida MSE+CE en h=1..4): `mlp`, `liquid` (Δt=1, referencia ciclo 1), **`liquid_ss` (K=4 sub-pasos Euler, dt=0.25)**, `ssm`, `linear` (A_op), `v4res` (Ψ + MLP([Ψ,c]), análogo arquitectónico del Dφ residual de v4 reimplementado en JAX; no es código v4).
- Semillas: DEV 0,1,2 (DEV + LONG); confirmación 3,4,5 (TEST, OOD, COMP, LONG). Sin retune.
- Ejes y gate: idénticos a E46. Eje 1 conf = media(OOD h4, COMP h4, LONG h16); en DEV = media(DEV h4, LONG h16). `liquid_ss` PASS si supera a cada competidor ≥2 pp en eje 1 (≥2/3 semillas) y no es peor en eje 2 (µs/paso Rust ≤1.1× o amplificación ≤). Paridad Rust con las tolerancias del ciclo 1 para todos.

## E47 — sparse top-k (magnitud)

- Sobre modelos E46b congelados (sin reentrenar): `liquid_ss` y el mejor modelo de E46b por eje 1 en DEV.
- k% ∈ {1,2,5,10,20,50,100} → k = {1,2,4,7,13,32,64}. Máscara m = top-k(|Ψ|) (empates: índice menor). Entrada x=[Ψ⊙m, c]; solo se actualizan componentes en m (resto se copia); en `liquid_ss` la máscara se fija al inicio de cada op.
- Rust implementa la misma semántica saltando filas/columnas no activas; la latencia incluye la selección top-k.
- k\* = menor k con caída de eje 1 ≤1 pp vs denso en DEV. PASS si en confirmación (a k\*) caída ≤1 pp **y** µs/paso Rust < denso. Umbral adaptativo, routing aprendido y bloques locales: `NOT_RUN`.

## E48 — parada adaptativa (sobre `liquid_ss`)

- Fijo: K ∈ {1,2,4,8,16,32,64} sub-pasos (dt=1/K, tiempo total 1 por op).
- Adaptativo: K=1,2,4,… hasta ‖Ψ_K−Ψ_{K/2}‖/(‖Ψ_K‖+1e-6) < ε, tope 64; coste acumulado (se cuentan todos los sub-pasos calculados).
- ε ∈ {0.1, 0.03, 0.01, 0.003}: se elige en DEV el mayor ε cuyo eje 1 queda a ≤1 pp de K=4 fijo. Congelado para confirmación.
- PASS si en confirmación: µs/op medio Rust < K=4 fijo y caída ≤1 pp en OOD h4 y COMP h4.

## H2H — token-free vs v4 sobre la tarea sellada de v4 (E35-style)

- Datos: `generate_and_seal` de v4 @`1da52e7` con sus semillas (DEV 0xA400–0xA40F, CONF 0xB400–0xB40F), exportados por `training/token_free/v4_bridge/h2h_v4.rs` (ejecutado en un worktree desacoplado; v4 no se modifica). 192 TRAIN / 256 TEST por semilla.
- v4: `run_e35_core` sin cambios (A = Dφ control, B = Dφ tras CDT→consolidación→adapt, CDT borrado). Se mide h1 (rel_err<5 %), rollout en familias iterables (h ∈ 1…64), composición de 2 pasos (pares deterministas exportados), latencia por predicción, tiempo de pared E35.
- Token-free: encoder identidad (como v4), Ψ = objeto completo (12), contexto = vector v4 (10 dims, permitido/estructurado). Modelos `mlp` (H=27), `liquid_ss` (H=19, K=4), `ssm` (H=19) con ≈962 parámetros (=Dφ v4 ±15 %). Presupuesto = lock v4: 8000 pasos, batch 16, lr 3e-3, Adam, pérdida MSE.
  - A: entrenado en TRAIN.
  - B: "consolidación por lotes" del almacén episódico (mismo contenido que el CDT de v4 = episodios TRAIN): mismo presupuesto, 50 % de muestras del almacén. El almacén se borra, se exportan parámetros y TEST corre en un proceso Rust nuevo (0 accesos a memoria por construcción).
  - Se espera B≈A (el almacén no añade información); se reporta tal cual.
- Selección: mejor modelo token-free (B) por h1 en DEV; confirmación sin cambios.
- Criterio principal: token-free supera a v4 si `acc_h1(TF_B) − acc_h1(v4_B) ≥ +0.05` con IC95 bootstrap pareado (16 semillas) > 0, en DEV y CONF. Igualmente se reportan rollout, comp2, µs/consulta Rust, tiempo de entrenamiento y parámetros. Paridad Rust obligatoria (mismas tolerancias).
- Limitación declarada: v4 Dφ actúa por punto con pesos compartidos (sesgo inductivo de equivariancia); los modelos token-free actúan sobre el objeto completo según su diseño. No se añade una variante por punto (sería mezclar arquitecturas).
