# Nova-2 · Exp3 (BOCD en lugar del umbral fijo τ) — Resultados

**Estado: DEV y CONFIRM completos (16/16 + 16/16) — exp3 FRACASA: F1 inconclusa (diferencia exactamente 0) y F3 (no-daño) falla por NOVA_BOCD.** DEV 12:41–15:10 y CONFIRM 15:10–16:42 del 24-oct-2026 (America/Mexico_City, UTC−6), mismo binario y commit `b89f900`. Preregistro [`PREREG_EXP3.md`](PREREG_EXP3.md) (`a237e12`), aplicado sin reajustes. Calibración congelada (`artifacts/orbita/nova2_exp3/calib.json`): μ_ref = −5,301, s²_ref = 0,489 (DEV fam 1–2, 4096 obs). Datos `artifacts/orbita/nova2_exp3/{dev,confirm}/metrics_0x*.json`; análisis `scripts/analyze_nova2_exp3.py` → `ANALYSIS.json`. Smoke OK y no analizado.

Contraste = media de diferencias pareadas por seed, n=2400, bootstrap pareado 10 000, semilla 12345, IC95. δ=0,05. En log, negativo = mejora; en acc16 y bacc, positivo = mejor.

## Veredicto

- **F1 (primaria) — inconclusa.** LAT_BOCD − LAT_EP, nuevas log, k=2: +0,000 [+0,000; +0,000] en DEV y +0,000 [+0,000; +0,000] en CONFIRM. LAT_BOCD y LAT_EP producen exactamente la misma mediana en nuevas (interpretación probable: en los episodios que fijan la mediana ambos detectores deciden «nuevo» y el ajuste LAT es idéntico). Ni se acepta ni se refuta.
- **F3 (no-daño, veta) — falla.** LAT_BOCD sí pasa (incluso mejora conocidas acc16: +0,03 a +0,22), pero NOVA_BOCD − NOVA en conocidas acc16 cae −0,17/−0,14/−0,05 (DEV) y −0,18/−0,14/−0,06 (CONFIRM) en k=2/4/8, con IC inferior por debajo de −δ en las 6 comparaciones.
- **F2** — sin efecto: NOVA_BOCD − NOVA nuevas log = 0 en todo k; LAT_BOCD − LAT_EP en k=4 = 0 y en k=8 ≈ +0,001/+0,009 (no mejora).
- **F4 (detección)** — BOCD detecta peor que τ_LAT en todo k y ambas fases (bacc −0,03/−0,09/−0,18 DEV; −0,03/−0,09/−0,19 CONFIRM), por falsos positivos altos (FPR 0,27–0,67 vs 0,19–0,24).
- **F5 (control nulo)** — no se cumple en k=2/4: LAT_BOCD es *peor* que sus decisiones barajadas (+0,56/+0,57 en k=2), confirmando la nota a priori: el ajuste LAT daña las nuevas en k pequeño y lo que ayuda es adaptar menos, no detectar mejor. En k=8 sí: −0,089 [−0,153; −0,037] DEV, −0,057 [−0,123; +0,016] CONFIRM.

**Éxito del exp3 = F1 aceptada ∧ F3 aceptada → NO.** El BOCD tal como se preregistró (h=1/16, umbral 0,5, priors de fam 1–2) no mejora la decisión conocido/nuevo respecto al umbral fijo τ: detecta peor y, en NOVA, sus falsos positivos disparan C4/C5 en conocidas y dañan su precisión.

## DEV (0xA400–0xA40F)

| contraste | k=2 | k=4 | k=8 |
|---|---|---|---|
| **F1/F2** LAT_BOCD − LAT_EP, nuevas log | +0,000 [+0,000; +0,000] | +0,000 [+0,000; +0,000] | +0,001 [+0,000; +0,004] |
| F2 NOVA_BOCD − NOVA, nuevas log | +0,000 [+0,000; +0,000] | +0,000 [+0,000; +0,000] | +0,000 [+0,000; +0,000] |
| **F3** LAT_BOCD − LAT_EP, conocidas acc16 | +0,033 [+0,020; +0,047] | +0,076 [+0,063; +0,090] | +0,200 [+0,179; +0,221] |
| **F3** NOVA_BOCD − NOVA, conocidas acc16 | −0,172 [−0,201; −0,146] | −0,142 [−0,167; −0,117] | −0,055 [−0,068; −0,042] |
| F4 bacc(BOCD) − bacc(τ_LAT) | −0,033 [−0,041; −0,024] | −0,089 [−0,098; −0,080] | −0,181 [−0,193; −0,168] |
| F5 LAT_BOCD − LAT_BOCD_SHUF, nuevas log | +0,564 [+0,426; +0,722] | +0,197 [+0,128; +0,271] | −0,089 [−0,153; −0,037] |
| NOVA − LAT_EP, nuevas log | −0,859 [−1,050; −0,682] | −0,460 [−0,591; −0,322] | −0,075 [−0,183; +0,020] |
| NOVA − ÓRBITA, nuevas log | +0,101 [+0,036; +0,170] | −0,018 [−0,090; +0,064] | −0,269 [−0,341; −0,200] |
| NOVA − NOVA−C3, nuevas log | −0,008 [−0,025; +0,009] | −0,022 [−0,059; +0,013] | −0,101 [−0,162; −0,042] |

Detección en H (media de seeds, n=2400):

| k | AUROC p_cambio | AUROC s_known | bacc BOCD | bacc τ_LAT | FPR BOCD | FPR τ_LAT |
|---|---|---|---|---|---|---|
| 2 | 0,960 | 0,964 | 0,858 | 0,891 | 0,268 | 0,192 |
| 4 | 0,942 | 0,958 | 0,793 | 0,882 | 0,405 | 0,209 |
| 8 | 0,889 | 0,938 | 0,676 | 0,857 | 0,644 | 0,244 |

Valores absolutos (media de seeds, n=2400; nuevas log / conocidas acc16 / nuevas acc16):

| brazo | k=2 | k=4 | k=8 |
|---|---|---|---|
| ÓRBITA | −0,718 / 0,354 / 0,025 | −0,799 / 0,354 / 0,026 | −0,927 / 0,354 / 0,029 |
| ÓRBITA+C3 | −0,756 / 0,367 / 0,030 | −0,880 / 0,387 / 0,033 | −1,156 / 0,443 / 0,042 |
| LAT_EP | +0,242 / 0,411 / 0,036 | −0,357 / 0,430 / 0,040 | −1,120 / 0,471 / 0,103 |
| NOVA | −0,617 / 0,631 / 0,036 | −0,817 / 0,677 / 0,046 | −1,195 / 0,741 / 0,074 |
| NOVA−C3 | −0,609 / 0,628 / 0,039 | −0,795 / 0,686 / 0,053 | −1,094 / 0,771 / 0,089 |
| LAT_BOCD | +0,242 / 0,443 / 0,039 | −0,357 / 0,506 / 0,047 | −1,119 / 0,670 / 0,108 |
| NOVA_BOCD | −0,617 / 0,458 / 0,033 | −0,817 / 0,535 / 0,043 | −1,195 / 0,686 / 0,073 |
| LAT_BOCD_SHUF | −0,321 / 0,454 / 0,033 | −0,554 / 0,518 / 0,033 | −1,030 / 0,666 / 0,089 |

Flujo (media): biblioteca 497.5; episodios marcados «nuevo» por BOCD 1286.4 de 2400 (821.3 sobre conocidas de 1925.4, 465.1 sobre nuevas de 474.6); aceptación NOVA 2104.2; κ efectivo de NG_ref 4390.9.

## CONFIRM (0xB400–0xB40F)

| contraste | k=2 | k=4 | k=8 |
|---|---|---|---|
| **F1/F2** LAT_BOCD − LAT_EP, nuevas log | +0,000 [+0,000; +0,000] | +0,000 [+0,000; +0,000] | +0,009 [+0,000; +0,027] |
| F2 NOVA_BOCD − NOVA, nuevas log | +0,000 [+0,000; +0,000] | +0,000 [+0,000; +0,000] | +0,000 [+0,000; +0,000] |
| **F3** LAT_BOCD − LAT_EP, conocidas acc16 | +0,028 [+0,023; +0,034] | +0,074 [+0,062; +0,085] | +0,220 [+0,204; +0,234] |
| **F3** NOVA_BOCD − NOVA, conocidas acc16 | −0,176 [−0,201; −0,152] | −0,141 [−0,160; −0,122] | −0,064 [−0,074; −0,054] |
| F4 bacc(BOCD) − bacc(τ_LAT) | −0,033 [−0,042; −0,025] | −0,095 [−0,107; −0,083] | −0,193 [−0,207; −0,179] |
| F5 LAT_BOCD − LAT_BOCD_SHUF, nuevas log | +0,573 [+0,392; +0,821] | +0,294 [+0,179; +0,410] | −0,057 [−0,123; +0,016] |
| NOVA − LAT_EP, nuevas log | −0,824 [−1,064; −0,631] | −0,523 [−0,636; −0,409] | +0,013 [−0,098; +0,120] |
| NOVA − ÓRBITA, nuevas log | +0,109 [+0,042; +0,185] | −0,028 [−0,084; +0,023] | −0,338 [−0,397; −0,276] |
| NOVA − NOVA−C3, nuevas log | −0,007 [−0,027; +0,014] | −0,033 [−0,067; +0,002] | −0,106 [−0,157; −0,058] |

Detección en H (media de seeds, n=2400):

| k | AUROC p_cambio | AUROC s_known | bacc BOCD | bacc τ_LAT | FPR BOCD | FPR τ_LAT |
|---|---|---|---|---|---|---|
| 2 | 0,951 | 0,957 | 0,850 | 0,883 | 0,275 | 0,202 |
| 4 | 0,937 | 0,951 | 0,783 | 0,878 | 0,423 | 0,213 |
| 8 | 0,882 | 0,929 | 0,663 | 0,856 | 0,668 | 0,243 |

Valores absolutos (media de seeds, n=2400; nuevas log / conocidas acc16 / nuevas acc16):

| brazo | k=2 | k=4 | k=8 |
|---|---|---|---|
| ÓRBITA | −0,750 / 0,338 / 0,023 | −0,793 / 0,338 / 0,025 | −0,891 / 0,338 / 0,030 |
| ÓRBITA+C3 | −0,771 / 0,352 / 0,023 | −0,875 / 0,370 / 0,030 | −1,190 / 0,424 / 0,051 |
| LAT_EP | +0,184 / 0,411 / 0,034 | −0,298 / 0,426 / 0,055 | −1,243 / 0,461 / 0,094 |
| NOVA | −0,640 / 0,628 / 0,033 | −0,821 / 0,658 / 0,042 | −1,230 / 0,744 / 0,069 |
| NOVA−C3 | −0,634 / 0,630 / 0,036 | −0,788 / 0,669 / 0,046 | −1,123 / 0,771 / 0,085 |
| LAT_BOCD | +0,184 / 0,439 / 0,035 | −0,298 / 0,500 / 0,060 | −1,234 / 0,681 / 0,107 |
| NOVA_BOCD | −0,640 / 0,452 / 0,026 | −0,821 / 0,517 / 0,038 | −1,230 / 0,679 / 0,069 |
| LAT_BOCD_SHUF | −0,389 / 0,446 / 0,036 | −0,592 / 0,528 / 0,046 | −1,177 / 0,676 / 0,086 |

Flujo (media): biblioteca 531.9; episodios marcados «nuevo» por BOCD 1308.5 de 2400 (835.2 sobre conocidas de 1917.1, 473.3 sobre nuevas de 482.9); aceptación NOVA 2103.9; κ efectivo de NG_ref 4306.0.

## Nota

Las diferencias de 0,000 exactas en F1/F2 salen tal cual de los metrics por seed. Lectura probable (no verificada episodio a episodio): en nuevas, BOCD y τ toman las mismas decisiones en los episodios que determinan la mediana, así que los brazos BOCD reproducen a sus pares; la diferencia entre brazos se concentra en los episodios conocidos (falsos positivos de BOCD).
