# CONSTELACIÓN v4-ESCALADA — Resultados

**Veredicto (regla del PREREG): escalar no mejora v4.** Ninguna variante pasa el gate DEV → CONFIRM, así que no se corrió CONFIRM. No hay candidata a main. 0 consultas TEST.

Fuente: `artifacts/constelacion_v4_escalada/analysis_dev.json` y `gate.txt` (commit dfcdc46). DEV: semillas 0xA450–0xA45F, 16/16 en los cinco brazos, 0 NaN, 0 colapso.

## DEV por brazo (medias por semilla)

| Brazo | Params | new_med_log | new_acc16 | known_med_log | known_acc16 | E33 pass | Sorpresa AUROC | Tiempo medio (s) |
|---|---|---|---|---|---|---|---|---|
| v4 | 10758 | −0.88371 | 0.0755 | −9.701 | 0.8765 | 16/16 | 0.96052 | 389 |
| P | 15878 | −0.88367 | 0.0768 | −9.553 | 0.8765 | 16/16 | 0.96054 | 544 |
| A | 41990 | −0.87324 | 0.0716 | −8.783 | 0.8340 | 15/16 | 0.96054 | 1358 |
| T | 10758 | −0.88368 | 0.0755 | −9.948 | 0.8765 | 16/16 | 0.96046 | 1654 |
| PAT | 62470 | −0.88366 | 0.0755 | −9.730 | 0.8765 | 16/16 | 0.96047 | 7748 |
| ÓRBITA-v1 | — | −0.79279 | 0.0182 | −2.529 | 0.3545 | 14/16 | — | — |

## V − v4 en la primaria (new_med_log; media [IC])
IC 98.75% para P, A y T; IC95 para PAT.

- P: +0.0000462 [−0.0000649, +0.0001689]
- A: +0.0104773 [+0.0000033, +0.0417214]
- T: +0.0000339 [−0.0000492, +0.0001188]
- PAT: +0.0000563 [−0.0000226, +0.0001385]

## Otras comparaciones vs v4 (media [IC])
- known_acc16: P, T y PAT 0 [0, 0]; A −0.0425 [−0.1274, 0] (falla no-daño, IC inf < −δ).
- No-daño final − pre en known_acc16: v4, P, T y PAT 0 [0, 0]; A 0 [−0.1274, +0.1274].
- E33 acc h1: 0 en todos. E33 acc h16: A −0.0596 [−0.1787, 0]; resto 0.
- Sorpresa AUROC: diferencias del orden de 1e-5 a 6e-5 en todos los brazos (no inferior).

## vs ÓRBITA-v1 (V − ÓRBITA, new_med_log, IC95)
- v4: −0.0909 [−0.1577, −0.0260]
- P: −0.0909 [−0.1576, −0.0261]
- A: −0.0804 [−0.1527, −0.0083]
- T: −0.0909 [−0.1577, −0.0260]
- PAT: −0.0909 [−0.1576, −0.0260]

Todos siguen por debajo de ÓRBITA-v1 en DEV (H3 se cumpliría en DEV), pero eso ya lo hacía v4.

## Gate
Requisito: media(V − v4) en new_med_log < 0, H2 OK, E33 ≥ 15/16, 16/16 sin NaN/colapso.
- P, T y PAT: fallan solo por la media de la primaria (> 0, prácticamente idéntica a v4).
- A: falla la primaria (peor) y H2 (no-daño en conocidas).

Resultado `gate: v4` → sin CONFIRM. Coincide con la expectativa declarada de un nulo: más capas, más ancho o más entrenamiento no mueven la primaria de v4; el ancho (A) la empeora un poco. main sin tocar.
