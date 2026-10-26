# CST v4 vs ÓRBITA-v1 — revisión rápida

> **Revisión rápida, no preregistrada, exploratoria.** No se reentrenó nada ni se ejecutó ningún modelo nuevo: se reanalizan los artefactos por semilla ya generados por CONSTELACIÓN v4 (`artifacts/constelacion_v4/{dev,confirm}/metrics_0x*.json`, commit de código `30e79d1`), donde ÓRBITA-v1 (pesos `artifacts/orbita/{dev,confirm}/weights_<seed>.json`) se evaluó sobre **los mismos ejemplos** que CST. Script: `python3 scripts/cst_vs_orbita_quick.py` → `artifacts/cst_vs_orbita_quick.json`. Bootstrap pareado por semilla, 10 000 réplicas, semilla 12345, IC95 percentil. n = 16 semillas por fase (DEV 0xA400–0xA40F, CONFIRM 0xB400–0xB40F). Tiempo: < 1 s.

## ¿Es el mismo protocolo de ÓRBITA?
Para E33, sí en lo que existe: la acc por horizonte de ÓRBITA en estos artefactos reproduce **exactamente** la de `docs/resultados_orbita.md` (DEV 0,984 / 0,984 / 0,942 / 0,706 / 0,313; CONFIRM 0,988 / 0,982 / 0,912 / 0,687 / 0,323; región 8 en todas las semillas salvo 0xA40D en DEV = 16). Mismo `test_sha256`, n = 64 ejemplos E33 por semilla.
**No cubierto** (no existe en los artefactos y requeriría ejecutar código nuevo): acc de 1 paso, horizontes h32/h64, rel_err₁ y la comparación con v4_E3/MLP del protocolo ÓRBITA. La región E33 de CST está topada en 16 porque sólo se evaluó hasta h16.

## (a) Protocolo ÓRBITA (E33, acc por horizonte)
| Métrica | DEV CST / ÓRBITA | DEV CST − ÓRBITA [IC95] | CONFIRM CST / ÓRBITA | CONFIRM CST − ÓRBITA [IC95] |
|---|---|---|---|---|
| E33 pase | 16/16 / 16/16 | — | 16/16 / 16/16 | — |
| Región E33 | 16 en 16/16 / 8 (15/16), 16 (1/16) | — | 16 en 16/16 / 8 en 16/16 | — |
| acc h1 | 0,984 / 0,984 | 0,000 [0,000, 0,000] | 0,988 / 0,988 | 0,000 [0,000, 0,000] |
| acc h2 | 1,000 / 0,984 | +0,016 [+0,002, +0,038] | 1,000 / 0,982 | +0,018 [+0,005, +0,034] |
| acc h4 | 1,000 / 0,942 | +0,058 [+0,025, +0,097] | 1,000 / 0,912 | +0,088 [+0,062, +0,118] |
| acc h8 | 1,000 / 0,706 | +0,294 [+0,234, +0,354] | 1,000 / 0,687 | +0,314 [+0,264, +0,366] |
| acc h16 | 1,000 / 0,313 | +0,688 [+0,630, +0,741] | 1,000 / 0,323 | +0,677 [+0,623, +0,732] |

## (b) Protocolo CST (held-out, k=4, h16)
| Métrica | DEV CST / ÓRBITA | DEV CST − ÓRBITA [IC95] | CONFIRM CST / ÓRBITA | CONFIRM CST − ÓRBITA [IC95] |
|---|---|---|---|---|
| med log rel_err nuevas | −0,875 / −0,799 | −0,076 [−0,141, −0,002] | −0,900 / −0,793 | −0,107 [−0,183, −0,024] |
| acc16 nuevas | 0,069 / 0,026 | +0,043 [+0,025, +0,061] | 0,066 / 0,025 | +0,042 [+0,022, +0,061] |
| med log rel_err conocidas | −9,96 / −2,52 | −7,45 [−7,76, −7,14] | −10,06 / −2,50 | −7,57 [−7,84, −7,33] |
| acc16 conocidas | 0,877 / 0,354 | +0,523 [+0,484, +0,563] | 0,877 / 0,338 | +0,539 [+0,509, +0,568] |

(Los IC de la primaria difieren en la 3.ª decimal del análisis oficial por la implementación del bootstrap; el oficial es `analysis_dev_confirm.json`.)

## Lectura honesta — por qué esto NO es una comparación justa
1. **Adaptación en test.** CST ajusta su campo rápido W^F con las transiciones observadas de cada episodio (k=4 en held-out; en E33 tras el primer paso). ÓRBITA-v1 se evalúa **congelada**. La ventaja de CST en h2–h16 y en conocidas es en gran parte adaptación por episodio (sin W^F, `no_fast`, CST empeora ≈3,2 log). El análogo adaptativo de ÓRBITA es ÓRBITA+LAT, que no está en estos artefactos con estas métricas.
2. **h1 es ÓRBITA.** En E33 la primera predicción de CST la da la cabeza de ÓRBITA, por eso acc h1 es idéntica.
3. **Entrenamiento distinto.** ÓRBITA: 256 000 pasos Adam, batch 32 (≈8,2 M muestras), TRAIN + 50 % replay del teacher (pool de 100 000), pérdida multi-horizonte. CST v4: aprendizaje local sobre flujo A (3072 episodios) + flujo B (192 nuevos + 192 conocidos), y además parte de los pesos de ÓRBITA para h1. No hay un presupuesto común (muestras, pasos ni cómputo).
4. Conclusión exploratoria: con los modelos ya entrenados y las mismas métricas, CST v4 supera a ÓRBITA-v1 congelada en todas las métricas comparables salvo h1 (empate), pero la comparación mezcla arquitectura con adaptación en test y entrenamiento distinto. La comparación justa está preregistrada en `PREREG.md`.
