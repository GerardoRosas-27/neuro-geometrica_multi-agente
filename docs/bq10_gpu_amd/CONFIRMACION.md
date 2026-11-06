# Confirmación exp/bq10-gpu-amd (solo evaluación, sin entrenar)

Commit evaluado: `2982f34`. Binario: `bio_quantum_engine eval dev|confirm` (CPU, nice 10). Logs en `logs/`.

## Pesos
- `bio_quantum_v10_x50_trained.zip` / `x20` descomprimidos: `weights_x50.bin` sha256 `3cdd6d5f…564a`, `weights_x20.bin` `d9b68e8f…c6e8`, idénticos a los de `bio_quantum_engine/`.
- **Ningún código del repo ni del zip carga estos pesos** (solo `File::create` al exportar tras `gpu-train`). `eval` construye un motor nuevo desde la semilla (`BioQuantumConstellationEngine::new(config, rng)`, 72 sinapsis CPU) → los pesos x50 (4096 sinapsis, 1024D) no participan y son incompatibles con el motor de evaluación. No es posible "evaluar con esos pesos" sin escribir código nuevo de carga + motor GPU de inferencia.

## Resultados reproducidos (iguales a lo que imprime el harness; determinista)
| Fase | known acc16 | known med log | new acc16 | new med log | E33 | Dale |
|---|---|---|---|---|---|---|
| DEV 0xA400–0xA40F | 100.00% | −12.747 | 99.80% | −8.193 | 16/16 | 0 |
| CONFIRM 0xB400–0xB40F | 100.00% | −12.818 | 100.00% | −7.885 | 16/16 | 0 |

## Validez / por qué NO es comparable con BQ10 (DEV −1.525 / CONFIRM −1.508)
- Harness propio del crate `bio_quantum_engine` (`protocol_cst4.rs` local, 6 puntos), no el CST4 canónico de `orbita` donde se midieron v4/ATLAS/BQ10. Métrica "med log" = media de medianas por semilla, no mediana global.
- Sin bootstrap pareado (10000, seed 12345, δ=0.05), sin baseline pareado, sin Paso 0 (hash v4 1985bdbad5eee41f en 0xC420 no soportado).
- log err ≈ −12.7 (≈3e-6) en conocidas indica ajuste casi exacto de dinámica sintética simple, no evidencia de superioridad.
- Los números de DEV/CONFIRM no dependen del entrenamiento GPU (1.63M episodios); el doc de resultados GPU solo reporta MSE de entrenamiento (0.000552→0.000478).

Conclusión: los números del harness se reproducen, pero **no confirman** que los pesos entrenados en GPU AMD mejoren algo; la evaluación no los usa y no sigue el protocolo canónico.
