# ÓRBITA-Nova — Resultados DEV + CONFIRM

**Preregistro**: [`PREREG.md`](PREREG.md) (`f64b289`), aplicado sin reajustes. **Datos**: `artifacts/orbita/nova_full/{dev,confirm}/metrics_0x*.json` (16/16 seeds por fase; DEV `0xA400–0xA40F`, CONFIRM `0xB400–0xB40F`). **Análisis**: `scripts/analyze_nova.py` → `artifacts/orbita/nova_full/ANALYSIS.json`.
Fecha: 2026-10-24 (America/Mexico_City, UTC−6).

Métrica primaria: mediana de `log(rel_err h16)` por seed sobre nuevas {8,10,11}, k=4 (menor = mejor), en n=2400 salvo indicación. Contraste = media sobre seeds de la diferencia pareada; bootstrap pareado por seed, 10 000 réplicas, semilla 12345, IC95 percentil. δ = 0,05. τ calibrado en DEV = **0,93866** (idéntico en CONFIRM). Biblioteca final media: 497,5 (DEV) / 531,9 (CONFIRM).

## Veredicto: **FRACASO de la rama** (según §6)

Éxito exige N1 ∧ N2 ∧ N3 ∧ N4. Sólo **N3** se acepta. **N2 no se acepta** (no hay aprendizaje a largo plazo medible en nuevas), lo que por sí solo es criterio de fracaso. N1 queda inconclusa (Nova ≈ LAT_EP), N4 falla por la mitad NOVA_SHUF.

## Tabla resumen (k=4)

| hip. | contraste | DEV media [IC95] | CONFIRM media [IC95] | resultado |
|---|---|---|---|---|
| N1 | NOVA − LAT_EP, nuevas, log | +0,0045 [−0,057; +0,071] | +0,0050 [−0,047; +0,055] | **inconclusa** |
| N2 | NOVA(2400) − NOVA(0), nuevas, log | +0,0141 [0,000; +0,042] | −0,0052 [−0,016; 0,000] | **no aceptada** |
| N3a | NOVA − ÓRBITA, conocidas, acc16 | +0,332 [+0,306; +0,360] | +0,331 [+0,304; +0,358] | ok (inf > −δ) |
| N3b | NOVA(2400) − NOVA(150), conocidas, acc16 | +0,014 [+0,005; +0,023] | +0,007 [−0,006; +0,020] | ok (inf > −δ) |
| **N3** | | | | **aceptada** |
| N4a | NOVA − NOVA_SHUF, nuevas, log | +0,0141 [0,000; +0,042] | −0,0011 [−0,003; 0,000] | no (IC toca 0) |
| N4b | NOVA − MLP, nuevas, log | −0,337 [−0,415; −0,265] | −0,356 [−0,504; −0,214] | sí |
| **N4** | | | | **no aceptada** |
| N5a | NOVA − NOVA−C5, nuevas, log | −0,277 [−0,373; −0,185] | −0,338 [−0,477; −0,207] | sí |
| N5b | NOVA − NOVA−C6, nuevas, log | +0,0141 [0,000; +0,042] | −0,0052 [−0,016; 0,000] | no |
| **N5** | | | | **no aceptada** (sólo C5 aporta) |
| N6 | ruido σ∈{0,01; 0,05} | — | — | **no evaluable**: los artefactos no contienen corridas con ruido |

Ninguna hipótesis queda formalmente *refutada* (ningún IC de CONFIRM entero del lado contrario), pero N2 no aceptada basta para fracaso.

## Comparaciones clave

- **Nova vs ÓRBITA (nuevas, log)**: +0,0045 [−0,057; +0,071] DEV; +0,0050 [−0,047; +0,055] CONFIRM → sin diferencia. En **conocidas** Nova duplica acc16 (0,686 vs 0,354 DEV; 0,669 vs 0,338 CONFIRM) y supera a ÓRBITA+C3 en +0,30 [+0,27; +0,33] (ambas fases).
- **Nova vs LAT_EP (nuevas)**: idéntico a lo anterior: LAT_EP produce métricas **idénticas a ÓRBITA en 16/16 seeds** en ambas fases (y LAT = LAT_EP en 15/16), es decir, el adaptador latente no movió nada en este harness. El control N1 es por tanto débil: Nova no mejora ni siquiera al baseline congelado en familias nuevas.
- Secundarias k: N1 con k=2 empeora (+0,109 DEV / +0,116 CONFIRM, IC > 0); con k=8 mejora (−0,168 [−0,241; −0,098] DEV / −0,232 [−0,295; −0,169] CONFIRM). Hay señal sólo con más contexto por episodio.

Medias absolutas (n=2400, k=4; new log / known acc16 / new acc16):

| brazo | DEV | CONFIRM |
|---|---|---|
| ÓRBITA | −0,799 / 0,354 / 0,026 | −0,793 / 0,338 / 0,025 |
| ÓRBITA+C3 | −0,880 / 0,387 / 0,033 | −0,875 / 0,370 / 0,030 |
| LAT_EP | −0,799 / 0,354 / 0,026 | −0,793 / 0,338 / 0,025 |
| LAT | −0,800 / 0,354 / 0,026 | −0,793 / 0,339 / 0,025 |
| NOVA | −0,795 / 0,686 / 0,053 | −0,788 / 0,669 / 0,046 |
| NOVA−C3 | −0,795 / 0,686 / 0,053 | −0,788 / 0,669 / 0,046 |
| NOVA−C5 | −0,518 / 0,687 / 0,048 | −0,449 / 0,671 / 0,044 |
| NOVA−C6 | −0,809 / 0,649 / 0,056 | −0,783 / 0,645 / 0,057 |
| NOVA_SHUF | −0,809 / 0,599 / 0,049 | −0,787 / 0,594 / 0,049 |
| MLP | −0,457 / 0,700 / 0,046 | −0,432 / 0,688 / 0,051 |
| ORACLE_EP | −0,358 / 0,880 / 0,059 | −0,365 / 0,881 / 0,070 |

Curva precuencial NOVA (nuevas, log) n=0→2400: DEV −0,809 → −0,795 (plana desde n=150); CONFIRM −0,783 → −0,788.

## Lectura honesta

1. **No hay aprendizaje a largo plazo en familias nuevas**: la biblioteca crece (~500 entradas) pero la métrica primaria no se mueve; NOVA−C6 y NOVA_SHUF dan lo mismo que NOVA → la recuperación/consolidación C6 no aporta en nuevas.
2. La ganancia real de Nova está en **conocidas** (acc16 ×2 vs ÓRBITA, sin olvido): N3 sólida.
3. C5 (árbitro de validación retenida) sí importa: sin él, nuevas empeoran ~0,3 nats; MLP sin estructura UVᵀ empeora igual. Es decir, el beneficio vs MLP/−C5 es *evitar daño* por sobreajuste, no mejorar sobre el baseline.
4. Anomalías de implementación a revisar: NOVA−C3 idéntico a NOVA en 16/16 (C3 parece inactivo dentro de NOVA); LAT_EP ≡ ÓRBITA en 16/16. Valor curioso: ORACLE_EP tiene peor log-mediana en nuevas que NOVA (−0,36 vs −0,79) pese a mejor acc16, lo que sugiere colas pesadas o un techo mal especificado en h16.
5. N6 no se ejecutó en el harness; queda pendiente.

Según §7, no se abre fase 5 (bytes). Sin merge a `main`.
