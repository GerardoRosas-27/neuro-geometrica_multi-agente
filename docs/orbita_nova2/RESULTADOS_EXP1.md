# Nova-2 · Exp1 (C4 bayesiano) — Resultados

**Estado: DEV completo (16/16); CONFIRM en curso** (lanzado 06:38 America/Mexico_City, UTC−6). Preregistro [`PREREG_EXP1.md`](PREREG_EXP1.md) (`2a09daf`), aplicado sin reajustes. Datos `artifacts/orbita/nova2_exp1/{dev,confirm}/metrics_0x*.json`; análisis `scripts/analyze_nova2_exp1.py` → `ANALYSIS.json`. Smoke (2 seeds, n=60) OK y no analizado.

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

Pendiente. Se corre igual por preregistro, sin cambios.
