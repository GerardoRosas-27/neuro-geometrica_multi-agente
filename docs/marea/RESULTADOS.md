# MAREA — Resultados (DEV + CONFIRM, sin reajuste)

`python3 scripts/analyze_marea.py` → `artifacts/marea/analysis.json`. 16+16 semillas; bootstrap pareado 10000, semilla 12345, δ=0.05. Asserts en las 32: 0 consultas TEST, paridad η=0 ≡ congelado (0 discrepancias), hash del JEPA intacto. ~7 s/semilla; cada fase < 1 min con 7 jobs.

| Criterio | DEV media [IC95] | CONFIRM media [IC95] | Veredicto |
|---|---|---|---|
| **Primaria**: pendiente ln err nuevas, MAREA − η=0 | −0.0115 [−0.0126, −0.0105] | −0.0120 [−0.0131, −0.0110] | **ACEPTADA** |
| Secundaria: media ln err nuevas, MAREA − η=0 | −0.113 [−0.119, −0.107] | −0.110 [−0.116, −0.104] | ACEPTADA |
| No-daño: media ln err conocidas, MAREA − η=0 (< +δ) | −0.237 [−0.250, −0.225] | −0.240 [−0.257, −0.224] | OK (mejora) |
| Externa: ln err nuevas t≥4, MAREA − ÓRBITA+LAT | +1.513 [+1.469, +1.554] | +1.439 [+1.396, +1.481] | **MAREA PEOR** |
| Externa: ln err conocidas t≥4, MAREA − ÓRBITA+LAT | +2.955 [+2.862, +3.048] | +2.936 [+2.849, +3.044] | MAREA PEOR |

Controles, pendiente nuevas (MAREA − control; <0 = MAREA mejor):
| Control | DEV | CONFIRM |
|---|---|---|
| m aleatorio | −0.0121 [−0.0129, −0.0111] | −0.0126 [−0.0135, −0.0116] |
| sin rollback | −0.0009 [−0.0012, −0.0006] | −0.0004 [−0.0007, −0.0001] |
| sin Hopfield | −0.0001 [−0.0002, 0.0000] | −0.0001 [−0.0002, −0.0000] |
| claves barajadas | −0.0001 [−0.0002, −0.0001] | −0.0001 [−0.0002, −0.0000] |
| JEPA aleatorio | −0.0005 [−0.0029, +0.0017] | −0.0031 [−0.0061, +0.0003] |
| K=1 | +0.0021 [+0.0015, +0.0026] | +0.0022 [+0.0015, +0.0029] |

Media ln err nuevas (MAREA − control): m aleatorio −0.172; JEPA aleatorio −0.57/−0.60; K=1 −0.008/−0.010; sin Hopfield +0.002 (ambas fases); sin rollback −0.002/+0.001; claves barajadas ≈0.

Medias absolutas (DEV/CONFIRM): pendiente MAREA 0.0104/0.0111 vs η=0 0.0219/0.0231; ln err nuevas MAREA −1.40 vs η=0 −1.29 vs ÓRBITA+LAT −2.91/−2.84 (ÓRBITA sola −2.83/−2.81). Por semilla: ~155 rollbacks, ~160 recuperaciones Hopfield, ~147 episodios «+».

## Lectura honesta
- La memoria test-time con tercer factor **sí** aplana la curva de error dentro del episodio (pendiente reducida ~a la mitad) y baja el error medio, sin daño en conocidas; el tercer factor importa (m aleatorio no aprende).
- Pero la pendiente sigue siendo **positiva** (el error crece dentro del episodio, sólo más lento), y MAREA es mucho peor que ÓRBITA+LAT en valor absoluto: ~e^1.5 ≈ 4.5× más error relativo en nuevas. El sistema completo no es competitivo.
- Hopfield ± y claves barajadas: efecto ≈0 (≤1e-4 en pendiente); sin Hopfield es incluso ligeramente mejor en ln err (+0.002). El componente Hopfield no aporta.
- Rollback: aporte pequeño (−0.0004…−0.0009 en pendiente).
- K=1 tiene **mejor** pendiente que R=3 (aunque peor media); el actor recursivo no ayuda a la pendiente.
- JEPA aleatorio: misma pendiente (IC incluye 0) pero error mucho mayor; el preentrenamiento aporta nivel, no adaptación.
