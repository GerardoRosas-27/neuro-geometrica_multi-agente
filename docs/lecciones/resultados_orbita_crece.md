# Resultados: ÓRBITA que crece (expertos añadidos por sorpresa persistente)

Prerregistro `docs/prereg_orbita_crece.md` (53e7f96), empujado antes de generar los datos. Código `orbita/src/grow.rs` (`grow-eval`),
`orbita/llm/crece.py`, `orbita/llm/crece_chain.sh`. Ejecución 2026-10-19 (hora CST, UTC−6), una cadena `setsid nohup` reanudable con `.done`:
Gemma DEV H 09:18:33–09:24:48, DEV S 09:24:48–09:49:02, CONFIRM H 09:49:54–09:55:37, CONFIRM S 09:55:37–10:20:54 (un solo proceso Gemma);
τ 10:22:32; malla DEV 12:18:59 (relanzada a las 10:37, ver Desviaciones); CONFIRM 14:37:08; análisis 14:37:10 (DEV) y 14:37:12 (CONFIRM).
16 seeds por fase, bootstrap pareado por seed (10000 réplicas, seed 12345), IC95, δ = 0,05. Tablas completas en
`artifacts/orbita/orbita_crece/summary_{dev,confirm}.md` (idénticas a `analyze_{dev,confirm}.log`); logs de cadena, τ y selección en la misma carpeta.

## Veredicto
- **«GROW aprende lo nuevo a largo plazo» (G1 ∧ G2 ∧ G4 ∧ G6, k_eval = 2, DEV y CONFIRM): NO.** En DEV se cumplen las cuatro; en CONFIRM sólo G6.
- **«El crecimiento aporta sobre un modelo fijo de igual tamaño» (H_MONO en DEV y CONFIRM): NO** (CONFIRM inconclusa).
- H_LAT: refutada en CONFIRM (la memoria latente LAT supera a GROW en las nuevas).
- Sin olvido en conocidas (G6 aceptada en ambas fases). Núcleo intacto (sha256 idéntico), 0 queries.

## Calibración y selección (sólo DEV)
- τ₂ = 0,016297990499999995, τ₄ = 0,024292169, τ₈ = 0,041172585 (P95 de S_known, familias 1–2, H DEV, n = 512 por k; `tau.json`, `calib.log`).
- Rejilla ce (`selected.json`; acc16 GROW held-out DEV n = 2400, media de familias y k): ce = 1 → 0,3426649305555556; **ce = 2 → 0,3557942708333333**;
  ce = 4 → 0,3424479166666667. Elegido **ce = 2**; CONFIRM se ejecutó una sola vez con ce = 2.

## Hipótesis (k_eval = 2, familias nuevas 8–11; media de 16 seeds [IC95])
Regla del prerregistro: cada criterio debe cumplirse en DEV **y** en CONFIRM. Como DEV las cumple todas, decide CONFIRM. El prerregistro sólo
fija el criterio de aceptación; para etiquetar el resto uso: **refutada** si el IC95 de CONFIRM queda entero del lado contrario (IC95 < 0, o
inferior ≤ −δ en G6) e **inconclusa** si el IC95 contiene o toca el umbral.

| id | cantidad | criterio | DEV | CONFIRM | veredicto |
|---|---|---|---|---|---|
| **G1** | pendiente acc16 GROW nuevas vs log₂ n (n = 150…2400) | IC95 > 0 | +0,0055 [+0,0034, +0,0078] | −0,0014 [−0,0026, −0,0001] | **refutada** |
| **G2** | acc16 GROW nuevas n = 2400 − n = 0 | IC95 > 0 | +0,0254 [+0,0156, +0,0371] | −0,0059 [−0,0117, +0,0000] | **inconclusa** (no se cumple; el IC toca 0) |
| **G4** | GROW n = 2400 − ORACLE_EP | ambos IC95 > 0 | +0,0254 [+0,0156, +0,0371] | −0,0244 [−0,0303, −0,0186] | **refutada** |
| | GROW n = 2400 − ORACLE_MS | | +0,0254 [+0,0156, +0,0371] | −0,0088 [−0,0146, −0,0029] | |
| **G6** | conocidas 0–7: GROW n = 2400 − OBS | ambos IC95 inf. > −δ | +0,0508 [+0,0410, +0,0615] | +0,0464 [+0,0356, +0,0576] | **aceptada** |
| | conocidas 0–7: GROW n = 2400 − n = 150 | | +0,0098 [+0,0039, +0,0161] | +0,0078 [−0,0010, +0,0161] | |
| **H_LAT** | GROW − LAT nuevas n = 2400 | IC95 > 0 | +0,0254 [+0,0156, +0,0371] | −0,0215 [−0,0293, −0,0146] | **refutada** |
| **H_MONO** | GROW − MONO nuevas n = 2400 | IC95 > 0 | +0,0254 [+0,0156, +0,0371] | −0,0049 [−0,0098, +0,0010] | **inconclusa** (no se cumple) |

Nota sobre DEV: en el H de DEV, con k = 2, todas las referencias valen 0 en las nuevas 8–11 (GROW n = 0, MONO, LAT, LAT_EP, ORACLE_EP,
ORACLE_MS: 0,0000 [0, 0]; OBS 0,0176). Por eso todas las diferencias de DEV coinciden con GROW n = 2400 = 0,0254, y esa ganancia viene entera
de las familias 10–11 (0,0508 [0,0312, 0,0742]; 8–9: 0,0000). En CONFIRM no se replica.

### Secundarios k_eval = 4 y 8 (DEV / CONFIRM)
| | k = 4 | k = 8 |
|---|---|---|
| G1 | +0,0063 [+0,0036, +0,0099] / +0,0000 [−0,0015, +0,0017] | +0,0032 [−0,0001, +0,0066] / +0,0009 [−0,0004, +0,0023] |
| G2 | +0,0293 [+0,0176, +0,0439] / +0,0010 [−0,0059, +0,0088] | +0,0156 [+0,0010, +0,0303] / +0,0049 [−0,0020, +0,0127] |
| G4 (EP) | +0,0312 [+0,0186, +0,0459] / −0,0273 [−0,0352, −0,0195] | +0,0381 [+0,0254, +0,0508] / +0,0049 [−0,0049, +0,0156] |
| G4 (MS) | +0,0312 [+0,0186, +0,0459] / −0,0117 [−0,0195, −0,0039] | +0,0381 [+0,0254, +0,0508] / +0,0049 [−0,0049, +0,0156] |
| G6 (− OBS) | +0,0654 [+0,0532, +0,0786] / +0,0542 [+0,0439, +0,0645] | +0,0894 [+0,0801, +0,0991] / +0,0713 [+0,0581, +0,0850] |
| G6 (− n = 150) | +0,0083 [+0,0015, +0,0156] / +0,0078 [−0,0020, +0,0171] | −0,0015 [−0,0107, +0,0073] / +0,0049 [−0,0054, +0,0151] |
| H_LAT | +0,0283 [+0,0146, +0,0439] / −0,0059 [−0,0127, +0,0010] | +0,0244 [+0,0088, +0,0400] / +0,0029 [−0,0039, +0,0107] |
| H_MONO | +0,0293 [+0,0176, +0,0439] / +0,0020 [−0,0049, +0,0098] | +0,0156 [+0,0010, +0,0303] / +0,0049 [−0,0020, +0,0127] |

Tampoco con k = 4 u 8 se cumple «aprende lo nuevo» en CONFIRM (G1 y G2 contienen 0).

## acc16 en held-out, familias nuevas 8–11, k = 2, n = 2400 (DEV / CONFIRM)
| GROW | MONO | LAT | LAT_EP | OBS | ORACLE_EP | ORACLE_MS | ORACLE_FAMNET (techo) |
|---|---|---|---|---|---|---|---|
| 0,0254 / 0,0225 | 0 / 0,0273 | 0 / 0,0439 | 0 / 0,0283 | 0,0176 / 0,0010 | 0 / 0,0469 | 0 / 0,0312 | 0,2959 / 0,2949 |

- CONFIRM, GROW por punto de control (nuevas 8–11, k = 2): 0,0283 en n = 0…600 (= LAT_EP, el fallback), 0,0264 en n = 1200, 0,0225 en
  n = 2400. Por subgrupo: 8–9 0,0430 → 0,0391 (LAT 0,0566); 10–11 0,0137 → 0,0059 (LAT 0,0312).
- GROW − ORACLE_FAMNET nuevas n = 2400: −0,2705 [−0,2979, −0,2441] / −0,2725 [−0,2979, −0,2461]. El techo con etiqueta privilegiada demuestra
  que un MLP 4→32→32→2 por familia sí puede aprender estas dinámicas (10–11 CONFIRM: 0,4727) si el criterio de agrupación fuera perfecto.

## Crecimiento, rutas y eventos (DEV / CONFIRM; media por seed [IC95])
| | DEV | CONFIRM |
|---|---|---|
| nº expertos tras ep. 150 / 300 | 0 / 0 | 0 / 0 |
| nº expertos tras ep. 600 | 0,1250 [0, 0,3125] | 0,1875 [0, 0,3750] |
| nº expertos tras ep. 1200 | 1,8125 [1,4375, 2,1875] | 1,5000 [1,1875, 1,8125] |
| nº expertos tras ep. 2400 | 4,2500 [3,6875, 4,8125] | 3,5000 [3,0000, 4,0000] |
| eventos grow | 5,0625 [4,6250, 5,6250] | 3,8750 [3,3125, 4,5000] |
| eventos grow_fail | 42,9375 [42,3750, 43,3750] | 44,1250 [43,5000, 44,6875] |
| eventos merge | 0,6875 [0,4375, 0,8750] | 0,2500 [0,0625, 0,5000] |
| eventos evict | 0 | 0 |
| fracción de episodios reclamados al crecer que son de familias nuevas | 0,5382 [0,4855, 0,5896] | 0,4933 [0,4303, 0,5579] |
| tasa de ruta a experto en H, nuevas, n = 2400, k = 2 | 0,1631 [0,1270, 0,2061] | 0,0762 [0,0596, 0,0947] |
| tasa de ruta a experto en H, conocidas, n = 2400, k = 2 | 0,1050 [0,0918, 0,1201] | 0,0752 [0,0649, 0,0864] |
| pureza del experto (fam. mayoritaria = real), nuevas, n = 2400, k = 2 | 0,9571 [0,9229, 0,9861] | 0,9091 [0,8344, 0,9688] |

El presupuesto E_MAX = 8 nunca se llenó (0 desalojos). La gran mayoría de los intentos de crecimiento fallan (42,9375 / 44,1250 por seed frente a 5,0625 / 3,8750 éxitos),
y entre lo reclamado por los expertos creados la fracción de familias nuevas es 0,5382 / 0,4933: el resto son episodios de familias conocidas.

## Olvido en conocidas 0–7 (k = 2; DEV / CONFIRM)
| GROW n = 0 | GROW n = 150 | GROW n = 2400 | MONO n = 2400 | LAT n = 2400 | LAT_EP | OBS |
|---|---|---|---|---|---|---|
| 0,4287 / 0,3501 | 0,4287 / 0,3501 | 0,4385 / 0,3579 | 0,4248 / 0,3438 | 0,6099 / 0,5117 | 0,4287 / 0,3501 | 0,3877 / 0,3115 |

- Sin olvido: GROW n = 2400 ≥ n = 150 en ambas fases (G6). Conocidas 1–2: GROW 0,5840 / 0,5762 igual en n = 0 y n = 2400.
- Pero en conocidas LAT sigue muy por encima: GROW − LAT 0–7 n = 2400 −0,1714 [−0,1992, −0,1455] / −0,1538 [−0,1714, −0,1377].

## Flujo (precuencial, k mixto; bloques de 300)
- Nuevas, GROW − LAT_EP con k = 2: 0 en los tres primeros bloques en ambas fases. Último bloque (2100–2399): DEV +0,0065 [−0,0024, +0,0163],
  CONFIRM +0,0073 [0, +0,0158]. Único bloque con IC > 0: DEV 1800–2099, +0,0118 [+0,0046, +0,0199].
- Nuevas, acc16 en el bloque 2100–2399 (DEV / CONFIRM): GROW 0,0665 / 0,0688, MONO 0,0574 / 0,0650, LAT 0,0653 / 0,0670,
  LAT_EP 0,0587 / 0,0663, OBS 0,0329 / 0,0399. Sin tendencia de subida en el flujo.

## Desviaciones (`DEVIATIONS.txt`)
- 10:37: se relanzó la malla DEV. El binario usado entre 10:22 y 10:36 era anterior al cambio CAP_POOL 200 → 600 (la fuente de 53e7f96 dice 600).
  Esas salidas se movieron a `dev/run_stale_cap200` y no se usan en ningún número de este documento. Todas las cifras salen del binario con
  CAP_POOL = 600, tal como fija el prerregistro.
- Ninguna otra desviación. Hiperparámetros los del prerregistro; ce fijado sólo con DEV.

## Integridad
- sha256 de los pesos de ÓRBITA idéntico antes y después en todas las ejecuciones (`sha256 pesos idéntico en todas: True`, DEV y CONFIRM).
- queries a TEST: `queries=0` en ambas fases.
- 16/16 seeds por fase (`job ok` en `confirm.log` y `grid.log`); análisis con n = 16 en todas las filas de hipótesis.

## Lectura
1. **Crecer por sorpresa persistente no aprende lo nuevo de forma que se replique.** El positivo de DEV (+0,0254, todo en las familias 10–11)
   no se replica en CONFIRM. Allí GROW empieza igual que LAT_EP (fallback) y **baja** al añadir expertos (0,0283 → 0,0225). Esto sugiere que la
   ruta a expertos sustituye algunos casos que LAT_EP resolvía.
2. **El cuello de botella es el criterio, no la capacidad.** El techo ORACLE_FAMNET (mismo MLP, agrupación por familia real) llega a 0,2959 / 0,2949
   en nuevas, y a 0,4727 en 10–11 en CONFIRM. GROW, en cambio, acumula 42,9375 / 44,1250 intentos fallidos por seed frente a 5,0625 / 3,8750
   crecimientos. Además, sólo 0,5382 / 0,4933 de lo que reclama es de familias nuevas, y en H enruta a experto 0,1631 / 0,0762 de los
   episodios nuevos. La pureza alta (0,9571 / 0,9091) indica que, cuando enruta, acierta la
   familia: lo que falta es reclamación fuera de muestra y cobertura.
3. **Sin olvido** (G6) y núcleo intacto, pero en conocidas la memoria LAT de 9166f1f sigue siendo mejor (GROW − LAT −0,1714 / −0,1538).
4. Siguiente paso posible (no probado): agrupar el pool por similitud de dinámica observada en lugar de por g* (el g* de D2 es una familia
   conocida y mezcla nuevas con conocidas), y crecer con un criterio de aceptación por mejora de predicción en datos retenidos en vez de
   S ≤ ce·τ. Otra opción es combinar GROW con LAT (memoria para conocidas y expertos sólo para lo que LAT no explica).

## Limitaciones
- 4 familias nuevas; H con 16 textos por familia e idioma. En DEV, con k = 2, todas las referencias valen 0 en las nuevas, lo que hace que las
  diferencias de DEV sean triviales (sólo GROW ≠ 0).
- acc16 binaria (umbral 0,05) y muy estricta para dinámicas no afines; no se reportan métricas continuas porque no estaban prerregistradas.
- Mezcla fija del flujo (4 de 12 familias nuevas) y k uniforme en {2, 4, 8}.
