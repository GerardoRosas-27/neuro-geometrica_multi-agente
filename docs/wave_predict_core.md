# Core de predicción por interferencia de ondas

**Rama:** `exp/fluido-3d-astra` · `src/wave_predict_core.rs`  
**Sin RQM.** Camino óptimo para “pasado × futuro → colapso/interferencia = predicción”.

## Idea

1. Inyectar **ondas del pasado** (observaciones) como paquetes gaussianos.
2. Cada hipótesis del **futuro** es otra onda (mismo canal de fase/centro si “continúa” el pasado).
3. Donde pasado y futuro **interfieren constructivamente** (`cos(Δφ)·overlap > 0`), la predicción se valida.
4. El resultado es el futuro de **máxima puntuación**.

## Camino óptimo

| Capa | Coste | Uso |
|------|------:|-----|
| **Analítico** (default) | ~**0.23 µs**/query | producción / predicción |
| Rejilla 2D 24² lineal (opcional) | ms bajos | refinar / visualizar foco |

No integra NLS 16³ ni RQM. El “líquido” aquí es el **campo de ondas** (paquetes + interferencia).

## API

```rust
let mut core = WavePredictCore::new();
let r = core.predict_from_observation(3, &[0, 1, 2, 3, 4, 5, 6, 7]);
// r.best_content == 3, r.best_score > 0
```

## Resultados (`cargo test --release --lib wave_predict_core`)

| Test | Resultado |
|------|-----------|
| match > mismatch | 0.81 vs −0.001 |
| 8/8 predicciones correctas | OK |
| multi-pasado coherente | OK |
| grid acuerda con analítico | winner=1 |
| latencia analítica | **acc 1.0 · 0.23 µs/q** |

## Vs caminos previos

| Método | µs/q (orden) | Rol |
|--------|-------------:|-----|
| RQM directo | ~2 | relaciones, no ondas |
| Spin NLS colapso | ~3000+ | dinámica cara |
| Surrogate colapso | ~0.1 | imita colapso |
| **Wave predict (este)** | **~0.23** | pasado×futuro por interferencia |
