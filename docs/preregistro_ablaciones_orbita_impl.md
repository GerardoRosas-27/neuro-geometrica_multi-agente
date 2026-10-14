# Implementación de las ablaciones de `docs/ablaciones_orbita.md` (registrada ANTES de correrlas)

**Fecha:** 2026-10-14 · **Rama:** `exp/orbita` · Protocolo: `docs/ablaciones_orbita.md` (commit `a6be441`, autor Gerardo Rosas). Este archivo sólo fija los detalles de implementación; no cambia el protocolo.

Receta común (idéntica a ÓRBITA-v1, `docs/preregistro_orbita.md`): mismos datos exportados por seed (`orbita export`, TRAIN + pool de replay del teacher, pesos de confianza), 256 000 pasos Adam, batch 32, lr coseno 3e-3→3e-5, pérdida relativa, λ_B = 1, λ_C = 0.01, misma inicialización (`torch.manual_seed(seed^0x0B)`), mismas seeds DEV `0xA400–0xA40F` y CONFIRM `0xB400–0xB40F`, misma evaluación (`orbita eval`: 1 paso TEST, E33 h1…h64, auditoría, proceso nuevo, paridad Rust↔PyTorch). Sin tuning nuevo.

| Variante | Implementación (`train_orbita.py --variant`, `orbita/src/model.rs`) | Parámetros totales | Multi-horizonte |
|---|---|---|---|
| ÓRBITA completa | ya entrenada (pesos de DEV/CONFIRM publicados) | 2602 | xy, k∈{1,2,4,8}, λ_H=0.01 |
| **A** corrector-only | `z=[q;g(q,c)]` → corrector de Brenier (r=2, η=0.5) → `q'=u[0:2]`. Sin Q, ρ, ω, A ni K^h. La cabeza sólo produce β (no usado), b_k, a_k. | 1702 (−900; no se añade capacidad) | **desactivado** (no existe K^k; no se sustituye por otro mecanismo) |
| **B1** K libre por paso | `z' = (I + K(c)) z + β(c)`, K(c) ∈ ℝ^{8×8} sin restricción (64 salidas de la cabeza en lugar de ω,s,A), luego corrector igual. Sin ortogonalidad, sin bloques, sin exp, sin semigrupo analítico. | 3302 (+700, se reporta; no se iguala) | xy, (I+K)^k iterado, λ_H=0.01 |
| **C** sin corrector | núcleo ÓRBITA idéntico, r = 0 (corrector no aplicado) | 2602 nominal (las salidas b_k,a_k y W_c, 928 parámetros, quedan sin efecto → 1674 efectivos) | igual que ÓRBITA |
| **D** extrapolación | sin entrenamiento: `orbita extrap` sobre los pesos ya entrenados de ÓRBITA (32 seeds) y los checkpoints v4 E3 publicados; free-run TEST iterable (rotate+reflect) en h = 1,2,3,4,6,8,12,16,24,32,48,64,96,128,192,256; acc, rel_err, norma, colapso (no finito o norma fuera de [0.1,10]), divergencia vs teacher iterado, acc del teacher. M3 no tiene checkpoint publicado → no se evalúa en D. | — | — |

B2 (K por horizonte) no se corre (opcional en el protocolo). Orden de ejecución: D; luego A, B1, C en DEV; luego en CONFIRM sin cambios. Estadística: `orbita/analyze_ablations.py` (media, mediana, desviación, IC95 bootstrap pareado por seed, 10 000 remuestreos, semilla 12345, seeds a favor). Resultados en `docs/resultados_ablaciones_orbita.md` y `artifacts/orbita/ablations/<variante>/{dev,confirm}/`.
