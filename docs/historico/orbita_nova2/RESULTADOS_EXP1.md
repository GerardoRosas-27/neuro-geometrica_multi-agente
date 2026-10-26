# Nova-2 · Exp1 (C4 bayesiano) — Resultados

**Estado: DEV y CONFIRM completos (16/16 + 16/16) — exp1 FRACASA: B1 refutada, B3 falla en ambas fases** (CONFIRM lanzado 06:38 y terminado 09:44 del 24-oct-2026, America/Mexico_City, UTC−6). Preregistro [`PREREG_EXP1.md`](PREREG_EXP1.md) (`2a09daf`), aplicado sin reajustes. Datos `artifacts/orbita/nova2_exp1/{dev,confirm}/metrics_0x*.json`; análisis `scripts/analyze_nova2_exp1.py` → `ANALYSIS.json`. Smoke (2 seeds, n=60) OK y no analizado.

Contraste = media de diferencias pareadas por seed, n=2400, bootstrap pareado 10 000, semilla 12345, IC95. δ=0,05. Negativo = mejora de NOVA_BAYES (en log); en acc16 positivo = mejor.

## DEV (0xA400–0xA40F)

| contraste | k=2 | k=4 | k=8 |
|---|---|---|---|
| **B1** NOVA_BAYES − NOVA, nuevas log | +0,016 [−0,041; +0,067] | −0,016 [−0,080; +0,046] | −0,036 [−0,097; +0,028] |
| **B3** NOVA_BAYES − NOVA, conocidas acc16 | −0,060 [−0,081; −0,040] | −0,077 [−0,095; −0,062] | −0,043 [−0,060; −0,028] |
| B4 NOVA_BAYES − NOVA_BAYES−C6, nuevas log | +0,002 [0,000; +0,007] | 0,000 [0; 0] | 0,000 [0; 0] |
| NOVA − NOVA−C3, nuevas log | −0,008 [−0,025; +0,009] | −0,023 [−0,060; +0,013] | −0,101 [−0,162; −0,042] |
| NOVA − LAT_EP, nuevas log | −0,859 [−1,050; −0,682] | −0,460 [−0,591; −0,323] | −0,075 [−0,183; +0,020] |
| NOVA − ÓRBITA, nuevas log | +0,101 [+0,036; +0,170] | −0,018 [−0,090; +0,064] | −0,269 [−0,341; −0,200] |

Lectura DEV: B1 inconclusa en k=2 (y k=4/8); B3 **falla** en los tres k (inf. IC < −δ en k=2,4; en k=8 −0,060 < −0,05) → con B3 fallida en DEV el exp1 **no puede tener éxito** (§5 exige DEV y CONFIRM). C6-como-prior no aporta (B4 ≈ 0). Regresión OK: NOVA−C3 en k=4 reproduce exactamente NOVA de f64b289 (−0,795 / 0,686 / 0,053). Con los arreglos, NOVA ≠ NOVA−C3 y LAT_EP ≠ ÓRBITA en 16/16; C3 aporta en k=8.

## CONFIRM (0xB400–0xB40F)

| contraste | k=2 | k=4 | k=8 |
|---|---|---|---|
| **B1** NOVA_BAYES − NOVA, nuevas log | +0,096 [+0,038; +0,165] | +0,007 [−0,048; +0,068] | −0,079 [−0,142; −0,028] |
| **B3** NOVA_BAYES − NOVA, conocidas acc16 | −0,065 [−0,086; −0,045] | −0,066 [−0,083; −0,048] | −0,049 [−0,063; −0,034] |
| B4 NOVA_BAYES − NOVA_BAYES−C6, nuevas log | −0,006 [−0,017; +0,004] | 0,000 [0; 0] | +0,001 [−0,005; +0,009] |
| NOVA − NOVA−C3, nuevas log | −0,007 [−0,027; +0,014] | −0,033 [−0,067; +0,002] | −0,107 [−0,157; −0,058] |
| NOVA − LAT_EP, nuevas log | −0,824 [−1,064; −0,631] | −0,523 [−0,636; −0,409] | +0,013 [−0,098; +0,120] |
| NOVA − ÓRBITA, nuevas log | +0,109 [+0,042; +0,185] | −0,028 [−0,084; +0,023] | −0,338 [−0,397; −0,276] |

Regresión: con los arreglos de C3/LAT_EP, NOVA ≠ NOVA−C3 y LAT_EP ≠ ÓRBITA en 16/16 seeds también en CONFIRM; NOVA−C3 en k=4 (−0,788 / 0,669 / 0,046) coincide con DEV/f64b289 dentro del ruido entre seeds.

## Veredicto (§5 del preregistro)

- **B1 (primaria, k=2): REFUTADA.** En CONFIRM el IC95 inf es +0,038 > 0: el C4 bayesiano *empeora* las familias nuevas con k=2 (en DEV fue inconcluso).
- **B2 (secundarias):** k=4 inconclusa en ambas fases. k=8 mejora en CONFIRM (−0,079 [−0,142; −0,028]) pero no en DEV (sup +0,028) → no se acepta; es sólo una pista para k grande.
- **B3 (no-daño): FALLA** en DEV y CONFIRM, en los tres k: NOVA_BAYES pierde 4–8 puntos de acc16 en familias conocidas (inf. IC < −δ).
- **B4 (C6 como prior):** no aporta (≈ 0 en ambas fases).
- **Éxito del exp1 = B1 ∧ B3 → NO.** Los arreglos de C3 y LAT_EP sí se sostienen: C3 aporta en k=8 (−0,10 a −0,11 en ambas fases) y LAT_EP ya no es idéntico a ÓRBITA.
