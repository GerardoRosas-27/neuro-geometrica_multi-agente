# ÓRBITA-Nova-2 · H2 NOVA_MIX — resultados

Preregistro `PREREG_H2.md` (`5341df4`), código `23804f3`. DEV 16/16 (fin 2026-10-24 20:02) y CONFIRM 16/16 (fin 20:12), America/Mexico_City (UTC−6). 0 consultas TEST; nada cambió entre fases. Análisis `python3 scripts/analyze_nova2_h2.py` → `artifacts/orbita/nova2_h2/ANALYSIS.json`. Media de diferencias pareadas por seed (n=16, n=2400), IC95 bootstrap 10 000 / 12345, δ=0,05. Biblioteca C6 legada 497,5 / 531,9 entradas (= H1); NOVA_RET reproduce H1 exactamente.

## Veredicto: **H2 aceptada; no-daño FALLA → H2 no tiene éxito.** Atribuible a la biblioteca en k=2 (vence a SHUF en ambas fases); **no** vence a ÓRBITA+C3 en k=2.

| contraste (nuevas, med log h16) | DEV | CONFIRM |
|---|---|---|
| **H2** NOVA_MIX − NOVA, k=2 | −0,184 [−0,248, −0,129] | −0,191 [−0,288, −0,095] |
| NOVA_MIX − NOVA, k=4 | −0,161 [−0,250, −0,074] | −0,143 [−0,209, −0,074] |
| NOVA_MIX − NOVA, k=8 | −0,139 [−0,196, −0,084] | −0,149 [−0,218, −0,085] |
| ctrl MIX − MIX_SHUF, k=2 | −0,087 [−0,153, −0,030] | −0,118 [−0,220, −0,013] |
| ctrl MIX − SHUF, k=4 | −0,088 [−0,164, −0,014] | −0,049 [−0,100, +0,005] |
| ctrl MIX − SHUF, k=8 | −0,075 [−0,133, −0,016] | −0,090 [−0,146, −0,037] |
| ctrl MIX − ÓRBITA+C3, k=2 | −0,045 [−0,118, +0,024] | −0,060 [−0,130, +0,005] |
| ctrl MIX − ÓRBITA+C3, k=4 | −0,098 [−0,161, −0,039] | −0,089 [−0,147, −0,027] |
| ctrl MIX − ÓRBITA+C3, k=8 | −0,178 [−0,241, −0,122] | −0,188 [−0,241, −0,135] |
| MIX − RET, k=2 | −0,049 [−0,088, −0,010] | −0,066 [−0,123, −0,018] |
| MIX − RET, k=4 | −0,054 [−0,088, −0,020] | −0,057 [−0,096, −0,009] |
| MIX − RET, k=8 | −0,038 [−0,085, +0,011] | −0,066 [−0,130, −0,004] |
| **no-daño** acc16 conocidas MIX − NOVA, k=2 | −0,158 [−0,185, −0,130] | −0,163 [−0,191, −0,137] |
| no-daño k=4 | −0,179 [−0,203, −0,156] | −0,162 [−0,191, −0,136] |
| no-daño k=8 | −0,171 [−0,190, −0,154] | −0,181 [−0,216, −0,152] |

Absolutos k=2 (nuevas med log / conocidas acc16), DEV: ÓRBITA+C3 −0,756/0,367 · NOVA −0,617/0,631 · RET −0,752/0,471 · MIX −0,801/0,473 · MIX_SHUF −0,714/0,380. CONFIRM: −0,771/0,352 · −0,640/0,628 · −0,765/0,458 · −0,831/0,465 · −0,712/0,366.

## Diagnósticos (DEV / CONFIRM)
- **Pesos NOVA_MIX** (k2 → k4 → k8): w medio ≈ (0,98, 0,03, 0,02, 0,02) → (0,96, 0,05, 0,04, 0,04) → (0,93, 0,10, 0,08, 0,07); cuota max|w|/|w|₁ 0,93 → 0,87 → 0,78; ‖w−w⁰‖ medio 0,068 → 0,15 → 0,31; fracción con ‖w−w⁰‖>0,1: 0,22 → 0,34 → 0,58 (idéntico en ambas fases ±0,01). Casi one-hot en k=2; mezcla real crece con k.
- **Pesos MIX_SHUF**: w₁ medio 0,76 → 0,61 → 0,45, |w|₁ 0,96 → 0,85 → 0,78 (encoge las entradas aleatorias hacia 0).
- **Coincidencia de familia top-1** (MIX = RET): nuevas 0,30/0,31 (k2), 0,31/0,32 (k4), 0,31/0,30 (k8); conocidas 0,37/0,37, 0,38/0,38, 0,41/0,41. SHUF: nuevas 0,08/0,08, 0,11/0,07, 0,09/0,07.
- **Paso C5+C7** MIX: nuevas 0,95/0,93, 0,81/0,84, 0,73/0,74; conocidas 0,70/0,74, 0,73/0,73, 0,83/0,81. MIX_SHUF: nuevas 0,73/0,73, 0,73/0,74, 0,71/0,70; conocidas 0,18/0,20, 0,28/0,29, 0,57/0,57.

## Lectura
1. H2 aceptada con margen y, a diferencia de H1, la mezcla vence a SHUF en k=2 en ambas fases (y en k=8): el ranking de la biblioteca + pesos aporta algo atribuible a la biblioteca. Los pesos aportan sobre RET (−0,05/−0,07 en k=2).
2. Pero en k=2 NOVA_MIX no supera a ÓRBITA+C3 (IC95 cruza 0); buena parte de "−NOVA" sigue siendo el sobreajuste de NOVA a 1 paso. En k=4/8 sí mejora a ÓRBITA+C3 con IC95 < 0.
3. No-daño falla con claridad (−16 a −18 pp de acc16 en conocidas frente a NOVA), como en H1: 4 pesos sobre operadores de biblioteca no igualan el ajuste completo de C4 en familias conocidas.
