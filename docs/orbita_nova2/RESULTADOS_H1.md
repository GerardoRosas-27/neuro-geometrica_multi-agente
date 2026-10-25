# ÓRBITA-Nova-2 · H1 NOVA_RET — resultados

Preregistro `PREREG_H1.md` (commit `362e603`), código `f73191a`. DEV 16/16 (fin 2026-10-24 19:24) y CONFIRM 16/16 (fin 19:38), America/Mexico_City (UTC−6). 0 consultas TEST. No se cambió nada entre fases.
Análisis: `python3 scripts/analyze_nova2_h1.py` → `artifacts/orbita/nova2_h1/ANALYSIS.json`. Media de la diferencia pareada por seed (n=16), IC95 bootstrap 10 000 / semilla 12345, n=2400. δ = 0,05.
Biblioteca C6 legada al final: 497,5 (DEV) / 531,9 (CONFIRM) entradas, igual que exp3 (control de identidad del flujo).

## Veredicto: **H1 aceptada, no-daño FALLA → H1 no tiene éxito**. La ganancia no es atribuible al ranking de la biblioteca en k=2.

| contraste (nuevas, med log h16 salvo indicación) | DEV | CONFIRM |
|---|---|---|
| **H1 primaria** NOVA_RET − NOVA, k=2 | −0,135 [−0,213, −0,058] | −0,125 [−0,212, −0,041] |
| NOVA_RET − NOVA, k=4 | −0,107 [−0,190, −0,024] | −0,086 [−0,161, −0,009] |
| NOVA_RET − NOVA, k=8 | −0,101 [−0,167, −0,037] | −0,083 [−0,145, −0,017] |
| ctrl NOVA_RET − NOVA_RET_SHUF, k=2 | −0,023 [−0,108, +0,056] | −0,038 [−0,115, +0,045] |
| ctrl NOVA_RET − SHUF, k=4 | −0,025 [−0,088, +0,030] | −0,047 [−0,114, +0,017] |
| ctrl NOVA_RET − SHUF, k=8 | −0,146 [−0,197, −0,094] | −0,128 [−0,205, −0,046] |
| ctrl NOVA_RET − ÓRBITA+C3, k=2 | +0,003 [−0,074, +0,076] | +0,006 [−0,066, +0,078] |
| ctrl NOVA_RET − ÓRBITA+C3, k=4 | −0,044 [−0,108, +0,015] | −0,032 [−0,114, +0,045] |
| ctrl NOVA_RET − ÓRBITA+C3, k=8 | −0,140 [−0,191, −0,091] | −0,123 [−0,188, −0,051] |
| ref NOVA − ÓRBITA+C3, k=2 | +0,139 [+0,077, +0,204] | +0,131 [+0,063, +0,211] |
| N2 NOVA_RET(2400) − NOVA_RET(150), k=2 | −0,086 [−0,151, −0,025] | −0,064 [−0,105, −0,020] |
| **no-daño** acc16 conocidas NOVA_RET − NOVA, k=2 | −0,160 [−0,188, −0,131] | −0,170 [−0,200, −0,144] |
| no-daño k=4 | −0,191 [−0,217, −0,167] | −0,171 [−0,201, −0,145] |
| no-daño k=8 | −0,202 [−0,223, −0,182] | −0,206 [−0,239, −0,178] |

Absolutos (media de seeds; nuevas med log / conocidas acc16), k=2: ÓRBITA+C3 −0,756/0,367 · NOVA −0,617/0,631 · NOVA_RET −0,752/0,471 · SHUF −0,729/0,372 (DEV); CONFIRM −0,771/0,352 · −0,640/0,628 · −0,765/0,458 · −0,727/0,356.

Diagnósticos NOVA_RET (DEV / CONFIRM):
- Coincidencia de familia del top-1 (familia dominante del soporte de la entrada = familia del episodio): nuevas 0,302/0,311 (k2), 0,310/0,319 (k4), 0,306/0,299 (k8); conocidas 0,368/0,365, 0,377/0,376, 0,411/0,410.
- Tasa de paso C5+C7: nuevas 0,966/0,944 (k2), 0,810/0,827 (k4), 0,715/0,734 (k8); conocidas 0,696/0,737, 0,684/0,706, 0,726/0,728.
- Entradas distintas elegidas ≈ 88/92 de ~500; puntuación k pasos mediana en nuevas −2,85 (RET) vs −2,31 (SHUF) en k=2.

## Lectura
1. H1 se acepta formalmente (IC95 sup < 0 en ambas fases, también k=4/8), pero en k=2 NOVA_RET ≈ ÓRBITA+C3 ≈ SHUF: la «ganancia» es sobre todo que NOVA es **peor** que ÓRBITA+C3 en nuevas k=2 (+0,13, sobreajuste de C4 a 1 paso) y NOVA_RET no lo empeora. No atribuible a la biblioteca (`atribuible_biblioteca = false`).
2. En k=8 sí hay señal de la biblioteca: NOVA_RET mejora a SHUF (−0,13/−0,15) y a ÓRBITA+C3 (−0,12/−0,14) con IC95 < 0 en ambas fases (secundario, no decide).
3. No-daño falla con claridad: en conocidas NOVA_RET pierde 16–21 pp de acc16 frente a NOVA (que ajusta por episodio), aunque mejora a ÓRBITA+C3 (+0,10).
4. Pureza baja (~0,30 en nuevas): la entrada elegida rara vez proviene de la propia familia; la biblioteca funciona como banco de operadores genéricos, no como memoria por familia.
