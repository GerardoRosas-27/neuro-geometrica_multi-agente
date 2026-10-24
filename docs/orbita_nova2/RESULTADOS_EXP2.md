# Nova-2 · Exp2 (base compartida A,B + código c∈ℝ²) — Resultados

**Estado: DEV y CONFIRM completos (16/16 + 16/16) — exp2 FRACASA: E1 aceptada, pero E3 (no-daño) falla con fuerza en ambas fases.** DEV 10:59–11:18 y CONFIRM 11:18–11:36 del 24-oct-2026 (America/Mexico_City, UTC−6), mismo binario y commit `6a50f68`. Preregistro [`PREREG_EXP2.md`](PREREG_EXP2.md) (`ba21413`), aplicado sin reajustes. Datos `artifacts/orbita/nova2_exp2/{dev,confirm}/metrics_0x*.json`; análisis `scripts/analyze_nova2_exp2.py` → `ANALYSIS.json`. Smoke OK y no analizado.

Contraste = media de diferencias pareadas por seed, n=2400, bootstrap pareado 10 000, semilla 12345, IC95. δ=0,05. En log, negativo = mejora de NOVA_BASE; en acc16, positivo = mejor.

## DEV (0xA400–0xA40F)

| contraste | k=2 | k=4 | k=8 |
|---|---|---|---|
| **E1/E2** NOVA_BASE − NOVA, nuevas log | −0,096 [−0,167; −0,027] | −0,052 [−0,131; +0,023] | −0,034 [−0,087; +0,019] |
| **E3** NOVA_BASE − NOVA, conocidas acc16 | −0,214 [−0,241; −0,187] | −0,251 [−0,277; −0,229] | −0,250 [−0,272; −0,231] |
| E4 NOVA_BASE(2400) − NOVA_BASE(0), nuevas log | +0,043 [−0,011; +0,097] | +0,011 [−0,035; +0,060] | −0,074 [−0,112; −0,039] |
| E5 NOVA_BASE − NOVA_BASE−sparse, nuevas log | +0,002 [−0,022; +0,024] | +0,008 [−0,005; +0,022] | −0,007 [−0,018; +0,005] |
| E5 NOVA_BASE − NOVA_BASE−sparse, conocidas acc16 | +0,002 [−0,002; +0,005] | −0,002 [−0,007; +0,004] | −0,001 [−0,004; +0,003] |
| NOVA − NOVA−C3, nuevas log | −0,008 [−0,025; +0,009] | −0,023 [−0,060; +0,013] | −0,101 [−0,162; −0,042] |
| NOVA − LAT_EP, nuevas log | −0,859 [−1,050; −0,682] | −0,460 [−0,591; −0,323] | −0,075 [−0,183; +0,020] |
| NOVA − ÓRBITA, nuevas log | +0,101 [+0,036; +0,170] | −0,018 [−0,090; +0,064] | −0,269 [−0,341; −0,200] |

Valores absolutos (media de seeds, n=2400; nuevas log / conocidas acc16 / nuevas acc16):

| brazo | k=2 | k=4 | k=8 |
|---|---|---|---|
| ÓRBITA | −0,718 / 0,354 / 0,025 | −0,799 / 0,354 / 0,026 | −0,927 / 0,354 / 0,029 |
| NOVA | −0,617 / 0,631 / 0,036 | −0,817 / 0,677 / 0,046 | −1,195 / 0,741 / 0,074 |
| NOVA_BASE | −0,713 / 0,417 / 0,031 | −0,869 / 0,426 / 0,036 | −1,229 / 0,491 / 0,053 |
| NOVA_BASE−sparse | −0,715 / 0,415 / 0,034 | −0,877 / 0,427 / 0,036 | −1,223 / 0,491 / 0,056 |

Pool a n=2400 (media): 2104 ítems (1873 conocidas); base sparse construida con 195 ítems, base completa con 2104. Tasa de aceptación de NOVA en el flujo 0,877.

## CONFIRM (0xB400–0xB40F)

| contraste | k=2 | k=4 | k=8 |
|---|---|---|---|
| **E1/E2** NOVA_BASE − NOVA, nuevas log | −0,080 [−0,157; −0,004] | −0,033 [−0,088; +0,020] | +0,003 [−0,073; +0,072] |
| **E3** NOVA_BASE − NOVA, conocidas acc16 | −0,241 [−0,268; −0,215] | −0,256 [−0,284; −0,230] | −0,272 [−0,301; −0,243] |
| E4 NOVA_BASE(2400) − NOVA_BASE(0), nuevas log | +0,051 [+0,011; +0,091] | +0,021 [−0,004; +0,049] | −0,036 [−0,060; −0,014] |
| E5 NOVA_BASE − NOVA_BASE−sparse, nuevas log | +0,015 [−0,008; +0,038] | +0,011 [−0,017; +0,042] | −0,001 [−0,011; +0,009] |
| E5 NOVA_BASE − NOVA_BASE−sparse, conocidas acc16 | −0,003 [−0,007; +0,001] | −0,002 [−0,007; +0,002] | 0,000 [−0,004; +0,004] |
| NOVA − NOVA−C3, nuevas log | −0,007 [−0,027; +0,014] | −0,033 [−0,067; +0,002] | −0,107 [−0,157; −0,058] |
| NOVA − LAT_EP, nuevas log | −0,824 [−1,064; −0,631] | −0,523 [−0,636; −0,409] | +0,013 [−0,098; +0,120] |
| NOVA − ÓRBITA, nuevas log | +0,109 [+0,042; +0,185] | −0,028 [−0,084; +0,023] | −0,338 [−0,397; −0,276] |

Valores absolutos (media de seeds, n=2400; nuevas log / conocidas acc16 / nuevas acc16):

| brazo | k=2 | k=4 | k=8 |
|---|---|---|---|
| ÓRBITA | −0,750 / 0,338 / 0,023 | −0,793 / 0,338 / 0,025 | −0,891 / 0,338 / 0,030 |
| NOVA | −0,640 / 0,628 / 0,033 | −0,821 / 0,658 / 0,042 | −1,230 / 0,744 / 0,069 |
| NOVA_BASE | −0,720 / 0,388 / 0,027 | −0,854 / 0,402 / 0,027 | −1,227 / 0,472 / 0,053 |
| NOVA_BASE−sparse | −0,735 / 0,391 / 0,026 | −0,865 / 0,405 / 0,023 | −1,226 / 0,472 / 0,055 |

Pool a n=2400 (media): 2104 ítems (1862 conocidas); base sparse construida con 209 ítems, base completa con 2104. Tasa de aceptación de NOVA en el flujo 0,877.

Regresión: NOVA, NOVA−C3, LAT_EP y ÓRBITA reproducen exactamente los números del exp1 (mismos seeds); NOVA ≠ NOVA−C3 y LAT_EP ≠ ÓRBITA en 16/16 seeds en ambas fases. NOVA_BASE ≠ NOVA y NOVA_BASE ≠ NOVA_BASE−sparse en 16/16.

## Veredicto (§5 del preregistro)

- **E1 (primaria, k=2): ACEPTADA.** IC95 sup < 0 en DEV (−0,096 [−0,167; −0,027]) y CONFIRM (−0,080 [−0,157; −0,004]): la base compartida reduce el error en familias nuevas con k=2. Es el primer resultado positivo de Nova-2, aunque en CONFIRM el margen es pequeño.
- **E2 (k=4, k=8): inconclusas** en ambas fases (los IC cruzan 0).
- **E3 (no-daño): FALLA** en DEV y CONFIRM, en los tres k, y con fuerza: NOVA_BASE pierde 21–27 puntos de acc16 en familias conocidas frente a NOVA (inf. IC ≈ −0,19 a −0,30 ≪ −δ). En valor absoluto, NOVA_BASE en conocidas (0,39–0,49) queda cerca de ÓRBITA sin adaptación (0,34–0,35), no de NOVA (0,63–0,74).
- **E4 (aprende con experiencia, k=2): no se cumple.** DEV +0,043 [−0,011; +0,097] y CONFIRM +0,051 [+0,011; +0,091]: con k=2 la base consolidada no mejora (en CONFIRM empeora) respecto a n=0. Con k=8 sí mejora en ambas fases (−0,074 y −0,036).
- **E5 (sparse):** sin efecto; las dos versiones de la base dan prácticamente lo mismo (|Δ| ≤ 0,015 en log y ≤ 0,003 en acc16).
- **Éxito del exp2 = E1 ∧ E3 → NO.**

Lectura (no preregistrada): un ΔK de rango 2 con A,B fijos y sólo 2 grados de libertad por episodio regulariza lo suficiente para ayudar con k=2 en familias nuevas, pero no puede representar la variedad de ajustes que NOVA hace en las familias conocidas, de ahí el gran daño en conocidas. Que la versión sparse y la completa coincidan sugiere que la base está dominada por la misma dirección principal con o sin selección.
