# Resultados — ÓRBITA vs v4 §16 (DEV + CONFIRM)

**Rama:** `exp/orbita` · **Pre-registro:** `docs/preregistro_orbita.md` (commit `f5d7781`, escrito antes de DEV/CONFIRM) · **Código:** `orbita/` sin cambios desde `f5d7781` en ambas fases (`git diff f5d7781 HEAD -- orbita/` vacío) · **Fecha:** 2026-10-14.
Comparadores v4 (`E3_gated_trusted` / `R_E3_V0_pure`) y control MLP (`M3_mlp_gated_trusted` / `R_M3_V0`) tomados de los artefactos publicados `artifacts/v4/distill_e/{dev,confirm}/metrics.csv`, como fija el pre-registro. No se compara con v5.

Tablas generadas por `python3 orbita/analyze_orbita.py dev|confirm` (copias en `artifacts/orbita/{dev,confirm}/paired_stats.md`).

## DEV (seeds `0xA400–0xA40F`, n = 16)

| Brazo | params | acc 1 paso | rel_err₁ | gate E33 (región≥8, limpio, proceso nuevo) | región mediana | acc h1 | acc h2 | acc h4 | acc h8 | acc h16 | acc h32 | acc h64 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| v4_E3 | 2590 | 0.9834 | 0.0090 | 13/16 | 8.0 | 0.992 | 0.985 | 0.925 | 0.632 | 0.264 | 0.050 | 0.005 |
| v4_M3_mlp | 2590 | 0.9802 | 0.0093 | 12/16 | 8.0 | 0.992 | 0.986 | 0.939 | 0.616 | 0.230 | 0.055 | 0.010 |
| ORBITA | 2602 | 0.9783 | 0.0082 | 16/16 | 8.0 | 0.984 | 0.984 | 0.942 | 0.706 | 0.313 | 0.079 | 0.017 |

| Comparación | métrica | media | IC95 | seeds a favor |
|---|---|---|---|---|
| ORBITA − v4_E3 | acc h16 | +0.0488 | [-0.0029, +0.1065] | 8/16 |
| ORBITA − v4_E3 | región | +1.2500 | [+0.2500, +2.5000] | 4/16 |
| ORBITA − v4_E3 | acc h8 | +0.0742 | [+0.0195, +0.1328] | 11/16 |
| ORBITA − v4_E3 | acc 1 paso | -0.0051 | [-0.0127, +0.0024] | 7/16 |
| ORBITA − v4_M3_mlp | acc h16 | +0.0830 | [+0.0224, +0.1416] | 12/16 |
| ORBITA − v4_M3_mlp | región | +1.5000 | [+0.5000, +2.7500] | 5/16 |
| ORBITA − v4_M3_mlp | acc h8 | +0.0898 | [+0.0225, +0.1562] | 12/16 |
| ORBITA − v4_M3_mlp | acc 1 paso | -0.0020 | [-0.0120, +0.0085] | 8/16 |

## CONFIRM (seeds `0xB400–0xB40F`, n = 16)

| Brazo | params | acc 1 paso | rel_err₁ | gate E33 (región≥8, limpio, proceso nuevo) | región mediana | acc h1 | acc h2 | acc h4 | acc h8 | acc h16 | acc h32 | acc h64 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| v4_E3 | 2590 | 0.9864 | 0.0089 | 9/16 | 8.0 | 0.991 | 0.978 | 0.869 | 0.551 | 0.217 | 0.044 | 0.008 |
| v4_M3_mlp | 2590 | 0.9837 | 0.0093 | 10/16 | 8.0 | 0.994 | 0.981 | 0.878 | 0.558 | 0.210 | 0.055 | 0.011 |
| ORBITA | 2602 | 0.9819 | 0.0081 | 16/16 | 8.0 | 0.988 | 0.982 | 0.912 | 0.687 | 0.323 | 0.073 | 0.008 |

| Comparación | métrica | media | IC95 | seeds a favor |
|---|---|---|---|---|
| ORBITA − v4_E3 | acc h16 | +0.1065 | [+0.0664, +0.1504] | 15/16 |
| ORBITA − v4_E3 | región | +1.7500 | [+0.7500, +2.7500] | 7/16 |
| ORBITA − v4_E3 | acc h8 | +0.1357 | [+0.0752, +0.1963] | 14/16 |
| ORBITA − v4_E3 | acc 1 paso | -0.0044 | [-0.0108, +0.0017] | 6/16 |
| ORBITA − v4_M3_mlp | acc h16 | +0.1133 | [+0.0615, +0.1641] | 14/16 |
| ORBITA − v4_M3_mlp | región | +1.5000 | [+0.5000, +2.5000] | 6/16 |
| ORBITA − v4_M3_mlp | acc h8 | +0.1289 | [+0.0498, +0.2041] | 14/16 |
| ORBITA − v4_M3_mlp | acc 1 paso | -0.0017 | [-0.0098, +0.0068] | 5/16 |

IC95 bootstrap pareado, 10 000 remuestreos, semilla 12345. "Seeds a favor" = seeds con diferencia estrictamente > 0 (las regiones empatan en 8 en la mayoría de seeds).

## Auditoría (ÓRBITA, ambas fases)
- 16/16 seeds CLEAN en DEV y en CONFIRM; leakage = 0 y 0 queries (CDT, RQM, tabla, NN, atractor, memoria directa) en todas las filas.
- Proceso nuevo (`orbita fresh`): `fresh_ok=true` en 16/16 seeds por fase, 0 queries.
- Paridad Rust↔PyTorch: max ‖Δ‖₂ = 4.9e-15 (DEV) y 4.1e-15 (CONFIRM), < 1e-9.
- Región E33 de ÓRBITA: 8 en todas las seeds salvo `0xA40D` en DEV (16).

## Veredicto según el pre-registro (§4)
1. **Gate ÓRBITA:** PASS en ambas fases (16/16 en DEV, 16/16 en CONFIRM; umbral ≥ 9/16).
2. **¿ÓRBITA gana a v4?** **No.**
   - (a) gate en ambas fases: cumple.
   - (b) IC95 de ORBITA − E3 en acc h16 > 0 en DEV **y** CONFIRM: **falla en DEV** ([-0.0029, +0.1065], cruza 0); en CONFIRM sí cumple ([+0.0664, +0.1504]).
   - (c) acc 1 paso no peor que E3 en más de 0.02: cumple (−0.0051 DEV, −0.0044 CONFIRM).
3. **¿v4 gana?** No: el IC95 de acc h16 no es < 0 en ninguna fase.
4. **Resultado: empate / no concluyente** frente a v4 E3, por el criterio pre-registrado. No se propone el nombre v6 (criterio 5).
5. **Secundario, contra el control MLP M3:** la diferencia en acc h16 tiene IC95 > 0 en DEV ([+0.0224, +0.1416]) y en CONFIRM ([+0.0615, +0.1641]), con gate PASS en ambas fases y acc 1 paso dentro de 0.02. Bajo el análisis análogo, ÓRBITA supera al MLP.

### Lectura (no altera el veredicto)
La señal va en la misma dirección en las dos fases: más horizonte útil (h8 con IC95 > 0 contra E3 en ambas fases, h16 media positiva en ambas), gate más robusto (16/16 frente a 13/16 y 9/16 de E3) y un coste pequeño en acc de un paso (~0.5 pp, IC95 cruzando 0). Pero el criterio principal exigía IC95 > 0 en h16 en las dos fases y DEV no lo alcanzó, así que el resultado se reporta como no concluyente. Pendiente (§4.7): ablación λ_H = 0 en DEV/CONFIRM para separar el efecto del núcleo ortogonal del de la pérdida multi-horizonte; cualquier nuevo ciclo necesita pre-registro y seeds nuevas.
