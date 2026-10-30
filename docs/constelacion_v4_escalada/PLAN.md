# CONSTELACIÓN v4-ESCALADA — Plan

**Pedido:** tomar la mejor CONSTELACIÓN (v4), agregar más capas, hacerla más grande y darle más entrenamiento, y ver si mejora. Si mejora, el usuario decide si pasa a `main`. Esta rama no toca `main`.

## Base
- Rama `exp/constelacion-v4` @ `fd803207b4151892852ba7503d80da14123be6f3` (RESULTADOS F2). Código congelado en `30e79d1`, sin cambios en el código hasta `fd80320`.
- v4 vs ÓRBITA-v1, primaria log h16 nuevas:
  - DEV −0.076 [−0.141, −0.001];
  - CONFIRM −0.107 [−0.181, −0.024];
  - E33 16/16 en ambas fases.
- Las mejoras M1/M2/M3 (`exp/constelacion-v4-mejoras`) fallaron y **no se usan**.

## Variantes (ablación separable; cada interruptor apagado ⇒ exactamente v4)
| Brazo | Capas (neuronas ÓRBITA por capa, m=8, h=8) | Flujo A / Flujo B (nuevas+conocidas) | Params (n_w + cabeza) |
|---|---|---|---|
| v4 (control, re-ejecutado) | [16, 8, 4] (L=3, 28 neuronas) | 3072 / 192+192 | 10 758 |
| **P**, más capas | [16, 8, 8, 4, 4] (L=5, 40 neuronas) | 3072 / 192+192 | 15 878 |
| **A**, más ancho | [32, 16, 8] (L=3, 56 neuronas) | 3072 / 192+192 | 41 990 |
| **T**, más entrenamiento | [16, 8, 4] | 12288 / 768+768 (×4) | 10 758 |
| **PAT**, todo | [32, 16, 16, 8, 8] (L=5, 80 neuronas) | 12288 / 768+768 | 62 470 |

- Todo lo demás es `Cfg::v4()` idéntico: t_f=128, cabeza ÓRBITA 1-paso, Π, τ/poda, sueño, η, etc.
- Los flujos T usan el mismo generador y la misma semilla; solo se agregan más episodios.

## Código (esta rama)
- `orbita/src/constelacion4/escalada.rs`: comandos `cst4esc-run` y `cst4esc-paso0`. En `mod.rs` solo se añade `pub mod escalada;`, así que v4 no cambia.
- `orbita/scripts/constelacion_v4_escalada.sh`: corre un proceso por (semilla, brazo), con JOBS=7, nice 10 y reanudable con `.done`.
- `orbita/scripts/constelacion_v4_escalada_chain.sh`: cadena DEV → gate → CONFIRM → análisis. **No se ha lanzado.**
- `scripts/analyze_constelacion_v4_escalada.py`: bootstrap, gate y veredicto.

## Smoke y gates previos (hechos, semillas de ajuste 0xC4E0/0xC4E1)
- `cst4esc-paso0 --seed 0xC4E0`: **PASS**.
  - r1: cfg(Esc::V4) ≡ Cfg::v4.
  - r2: bit a bit, hash de pesos `20b97c03e1fb8198`.
  - r3: cada variante cambia solo lo declarado y no da NaN en el flujo corto.
- Equivalencia con el binario v4: `cst4-run --arms cst` frente a `cst4esc-run --arms v4` (0xC4E1, na=96, nb=16).
  - Hash de pesos idéntico, `b258b7fb6ac87441`.
  - held, pre_known, E33 y colapso idénticos.
- Tiempo real de una semilla completa (0xC4E0, presupuesto pleno, 5 brazos en paralelo en 8 núcleos, nice 10):
  - v4: 355 s;
  - P: 543 s;
  - A: 1192 s;
  - T: 1444 s;
  - PAT: en curso al momento del commit (> 1943 s). La extrapolación multiplicativa (3,36 × 1,53 × 4,07 × 355 s) da ≈ 7 400 s ≈ 2 h. **Es una estimación, no una medición.**

## Estimación de coste
- Por semilla (CPU): ≈ 356 + 544 + 1193 + 1445 + ~7400 ≈ 10 900 s.
- **DEV**, 16 semillas × 5 brazos, JOBS=7: ≈ 175 000 s de CPU / 7 ≈ 7 h. Cota inferior por PAT: 16 trabajos de ~2 h en 7 ranuras = 3 tandas, ≈ 6–7 h. **Estimación: ~7–8 h.**
- **CONFIRM:** solo v4 más las variantes que pasen el gate. Con todas: ~7–8 h; sin PAT: ~2 h.
- Memoria: holgada (procesos pequeños).
