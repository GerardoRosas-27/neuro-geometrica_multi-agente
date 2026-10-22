# Resultados: ÓRBITA-TF — Transformer sobre estados continuos + energía libre vs ÓRBITA-v1

Prerregistro `docs/prereg_orbita_transformer_fe.md` (`9333cb4`), empujado antes de entrenar/evaluar. Código `orbita/tf/` (`tfe.py`, `eval_tf.py`,
`gen_eps.py`, `analyze_tf.py`, `chain.sh`), sin cambios de código ni hiperparámetros entre DEV y CONFIRM. Una sola cadena `setsid nohup chain.sh`
(J = 7 trabajos en paralelo, reanudable con `.done`), hora CST (UTC−6): episodios DEV 2026-10-20 13:11; DEV 13:54 → análisis DEV 2026-10-21 00:00:56;
episodios CONFIRM 00:01; CONFIRM → análisis CONFIRM 10:59:17; `chain done` 10:59:17. 64/64 trabajos por fase, sin fallos en `chain.log`.
16 seeds por fase (DEV `0xA400–0xA40F`, CONFIRM `0xB400–0xB40F`), 768 episodios por seed (64 × familias 0–11; 8–11 = nuevas no afines),
bootstrap pareado por seed (10 000 remuestreos, semilla 12345), IC95, δ = 0.05. 0 queries a TEST.
Tablas completas (idénticas a la salida de `analyze_tf.py`; re-ejecutado el 2026-10-21 y la salida coincide byte a byte) en
`artifacts/orbita/transformer_fe/analyze_{dev,confirm}.md`; valores por seed en `per_seed_{dev,confirm}.json`; log de cadena en `chain.log`.

## Setup (resumen)
- **TFE** (principal, 2618 params): 1 bloque Transformer pre-LN (d = 12, 2 cabezas, FFN 44) cuyos elementos de atención son los 6 puntos 2-D
  + 1 latente μ ∈ ℝ⁸; prior amortizado q(μ|c); pérdida ÓRBITA + β·KL (β = 1e-3). En test, con c desconocido: OBS (malla de candidatos) → FE
  (100 pasos Adam sobre μ minimizando error de predicción ponderado + complejidad) → free-run hasta t = 16.
- **TF** (2418 params): mismo Transformer, μ = m(c) determinista, sin KL; inferencia PE (sólo error de predicción).
- **MLP** (2642 params): MLP por punto, residual; inferencia OBS.
- **ÓRBITA-v1** (2602 params): pesos publicados congelados; `fam-eval` (c verdadero) y `lat-eval exp1` (OBS, LAT_EP, LAT, ORACLE_LS) con τ y λ fijados.
- Mismo presupuesto que ÓRBITA: mismos datos por seed, 256 000 pasos Adam, batch 32, lr coseno 3e-3→3e-5, float64.

## Veredicto global
- **El Transformer de estados con energía libre (TFE) NO supera a ÓRBITA-v1 en nada de lo prerregistrado.** H1 (conocidas), H2 (nuevas, k = 2)
  y H3b (calibración vs ÓRBITA) **refutadas** en CONFIRM.
- La energía libre como **objetivo de entrenamiento** perjudica: TFE queda por debajo de su control sin FE (TF) en conocidas (H4d refutada) y en
  nuevas (H4a refutada); el término de complejidad en inferencia también resta a k = 2 (H4c refutada).
- Lo único aceptado: F es un detector de error mejor que el azar (H3a) y la inferencia iterativa sobre μ mejora sobre la búsqueda en malla
  del propio TFE (H4b). Ninguno de los dos basta para competir con ÓRBITA.

## Hipótesis (familias nuevas = 8–11; media de 16 seeds [IC95])
Regla del prerregistro: cada criterio debe cumplirse en DEV **y** CONFIRM → **aceptada**; **refutada** si el IC95 de CONFIRM queda entero del lado
contrario; **inconclusa** el resto. Para H1 el prerregistro fija explícitamente: refutada si IC95 sup < −δ.

| id | cantidad (A − B) | criterio | DEV | CONFIRM | veredicto |
|---|---|---|---|---|---|
| **H1** | TFE − ÓRBITA, conocidas, c verdadero, acc16 | superior IC95 > 0 / no inferior inf. > −δ | −0.3174 [−0.3381, −0.2949] | −0.3138 [−0.3389, −0.2894] | **refutada** (sup < −δ) |
| **H2a** | TFE-FE − ÓRBITA-LAT_EP, nuevas k = 2, acc16 | IC95 > 0 | −0.0254 [−0.0310, −0.0200] | −0.0215 [−0.0276, −0.0161] | **refutada** |
| **H2b** | TFE-FE − ÓRBITA-OBS, nuevas k = 2 | IC95 > 0 | −0.0015 [−0.0063, +0.0042] | −0.0066 [−0.0117, −0.0017] | **refutada** |
| **H2** | H2a ∧ H2b | | | | **refutada** |
| H2c (sec.) | TFE-FE − ÓRBITA-LAT, nuevas k = 2 | IC95 > 0 | −0.0334 [−0.0386, −0.0283] | −0.0234 [−0.0298, −0.0173] | refutada |
| **H3a** | AUROC(F_TFE → fallo TFE-FE h16) − 0.5, k = 2 | inf. > δ | +0.2189 [+0.1840, +0.2529] (n = 16) | +0.1616 [+0.1280, +0.1957] (n = 16) | **aceptada** |
| **H3b** | AUROC TFE − AUROC ÓRBITA (s_known → fallo OBS), k = 2 | IC95 > 0 | −0.1184 [−0.1557, −0.0799] | −0.1134 [−0.1549, −0.0693] | **refutada** |
| **H4a** | TFE-FE − TF-PE, nuevas k = 2 | IC95 > 0 | −0.0483 [−0.0544, −0.0420] | −0.0420 [−0.0474, −0.0366] | **refutada** |
| **H4b** | TFE-FE − TFE-OBS, nuevas k = 2 | IC95 > 0 | +0.0066 [+0.0032, +0.0110] | +0.0078 [+0.0039, +0.0122] | **aceptada** |
| H4c (sec.) | TFE-FE − TFE-PE, nuevas k = 2 | IC95 > 0 | −0.0137 [−0.0195, −0.0078] | −0.0168 [−0.0212, −0.0122] | refutada |
| **H4d** | TFE − TF, conocidas c verdadero, acc16 | IC95 > 0 | −0.1664 [−0.1804, −0.1515] | −0.1772 [−0.1960, −0.1603] | **refutada** |
| control | TFE-FE − MLP-OBS, nuevas k = 2 | IC95 > 0 | +0.0059 [+0.0007, +0.0110] | +0.0020 [−0.0024, +0.0066] | inconclusa |
| control | TFE-FE − TF-OBS, nuevas k = 2 | IC95 > 0 | +0.0022 [−0.0037, +0.0083] | −0.0034 [−0.0083, +0.0027] | inconclusa |
| control | TFE − MLP, conocidas c verdadero, acc16 | IC95 > 0 | −0.0955 [−0.1028, −0.0884] | −0.1222 [−0.1344, −0.1094] | refutada |
| H5 | coste (descriptivo) | TFE más caro que ÓRBITA | ver abajo | ver abajo | no se sostiene de forma consistente |

### Secundarios k = 4 y 8 (DEV / CONFIRM)
| | k = 4 | k = 8 |
|---|---|---|
| H2a TFE-FE − ÓRBITA-LAT_EP | −0.0352 [−0.0408, −0.0291] / −0.0144 [−0.0220, −0.0071] | −0.0269 [−0.0325, −0.0210] / −0.0173 [−0.0288, −0.0063] |
| H2b TFE-FE − ÓRBITA-OBS | +0.0046 [−0.0007, +0.0100] / +0.0166 [+0.0095, +0.0242] | +0.0496 [+0.0430, +0.0564] / +0.0376 [+0.0281, +0.0466] |
| H2c TFE-FE − ÓRBITA-LAT | −0.0396 [−0.0447, −0.0344] / −0.0149 [−0.0222, −0.0073] | −0.0349 [−0.0398, −0.0300] / −0.0269 [−0.0374, −0.0159] |
| H3b AUROC TFE − ÓRBITA | −0.0306 [−0.0511, −0.0095] / −0.0262 [−0.0533, +0.0001] | −0.0354 [−0.0434, −0.0278] / −0.0077 [−0.0200, +0.0032] |
| H4a TFE-FE − TF-PE | −0.0454 [−0.0508, −0.0398] / −0.0293 [−0.0369, −0.0217] | −0.0415 [−0.0513, −0.0310] / −0.0361 [−0.0486, −0.0247] |
| H4b TFE-FE − TFE-OBS | +0.0237 [+0.0195, +0.0281] / +0.0352 [+0.0293, +0.0410] | +0.0610 [+0.0547, +0.0671] / +0.0601 [+0.0525, +0.0671] |
| H4c TFE-FE − TFE-PE | −0.0032 [−0.0076, +0.0012] / +0.0012 [−0.0039, +0.0066] | +0.0010 [−0.0022, +0.0042] / +0.0002 [−0.0049, +0.0054] |
| TFE-FE − MLP-OBS | +0.0164 [+0.0112, +0.0220] / +0.0281 [+0.0215, +0.0354] | +0.0596 [+0.0522, +0.0669] / +0.0520 [+0.0432, +0.0615] |
| TFE-FE − TF-OBS | +0.0098 [+0.0044, +0.0151] / +0.0229 [+0.0151, +0.0315] | +0.0508 [+0.0449, +0.0566] / +0.0452 [+0.0352, +0.0552] |
| conocidas TFE-FE − ÓRBITA-LAT_EP | −0.4740 [−0.4908, −0.4589] / −0.4752 [−0.4990, −0.4520] | −0.5602 [−0.5863, −0.5339] / −0.5319 [−0.5518, −0.5129] |

Con k = 4 y 8, TFE-FE supera a ÓRBITA-OBS (sólo malla) en CONFIRM, pero sigue por debajo de la adaptación latente de ÓRBITA (LAT_EP, LAT).

## Números clave: acc16 por brazo (media [IC95], DEV / CONFIRM)
**Conocidas 0–7, c verdadero (free-run 16 pasos):**
| brazo | acc16 | acc8 |
|---|---|---|
| TFE | 0.0319 [0.0258, 0.0386] / 0.0424 [0.0337, 0.0515] | 0.1271 / 0.1326 |
| TF | 0.1982 [0.1803, 0.2157] / 0.2196 [0.2036, 0.2369] | 0.5464 / 0.5236 |
| MLP | 0.1273 [0.1195, 0.1346] / 0.1646 [0.1564, 0.1716] | 0.4271 / 0.4448 |
| ÓRBITA-v1 | 0.3492 [0.3263, 0.3699] / 0.3562 [0.3331, 0.3799] | 0.6689 / 0.6652 |

TF − ÓRBITA (sec.): −0.1510 [−0.1774, −0.1262] / −0.1366 [−0.1597, −0.1144].

**Nuevas 8–11, c desconocido, k = 2:**
| TFE-FE | TFE-PE | TFE-OBS | TF-PE | TF-OBS | MLP-OBS | ÓRBITA-OBS | ÓRBITA-LAT_EP | ÓRBITA-LAT | ÓRBITA-ORACLE_LS |
|---|---|---|---|---|---|---|---|---|---|
| 0.0122 / 0.0125 | 0.0259 / 0.0293 | 0.0056 / 0.0046 | 0.0605 / 0.0544 | 0.0100 / 0.0159 | 0.0063 / 0.0105 | 0.0137 / 0.0190 | 0.0376 / 0.0339 | 0.0457 / 0.0359 | 0.0820 / 0.0742 |

**Conocidas 0–7, c desconocido, k = 2:** TFE-FE 0.0325 / 0.0428; TF-PE 0.3710 / 0.3721; MLP-OBS 0.0502 / 0.0737; ÓRBITA-OBS 0.1587 / 0.1808;
ÓRBITA-LAT_EP 0.4216 / 0.4481; ÓRBITA-LAT 0.4259 / 0.4523; ORACLE_LS 0.8809 / 0.8770.

**AUROC detector de error, k = 2 (DEV / CONFIRM):** F_TFE 0.7189 / 0.6616; F_PE de TF 0.8371 / 0.8536; ÓRBITA s_known → OBS 0.8373 / 0.7750;
ÓRBITA s_known → LAT_EP 0.6962 / 0.6932. Con k = 8: F_TFE 0.9108 / 0.9148 vs ÓRBITA→OBS 0.9462 / 0.9226.

## Coste (H5, descriptivo; media por seed, segundos de pared, DEV / CONFIRM)
| | DEV | CONFIRM |
|---|---|---|
| entrenamiento TFE | 6498.50 (2475–11037) | 4554.44 (2034–21677) |
| entrenamiento TF | 8401.38 (2063–10904) | 9603.12 (2993–21618) |
| entrenamiento MLP | 699.44 | 1156.75 |
| entrenamiento ÓRBITA (log publicado original) | 5088.31 | 18776.25 |
| eval TFE k = 2: OBS / FE / PE (s/768 ep.) | 270.58 / 132.36 / 2.58 | 11.43 / 3.05 / 2.38 |
| eval TF k = 2: OBS / PE | 11.68 / 3.16 | 11.88 / 3.13 |
| eval MLP k = 2: OBS | 2.74 | 2.92 |
| eval ÓRBITA `lat-eval exp1` k = 2 (incluye OBS, LAT, LAT_EP, mallas oráculo) | 44.72 | 46.09 |

La predicción «TFE más caro que ÓRBITA» no se sostiene de forma consistente: en entrenamiento TFE > ÓRBITA en DEV pero < en CONFIRM, y los
tiempos de ÓRBITA proceden de sus logs originales (otras condiciones de carga). Los tiempos de pared de este experimento se midieron con 7 trabajos
en paralelo y otros procesos en la máquina; hay valores atípicos (máximos de ~2000–2500 s en evaluaciones que normalmente tardan 3–50 s, p. ej.
TFE OBS/FE k = 2 en DEV, TFE k = 8 y TF OBS k = 4/8 en CONFIRM), por lo que las medias de coste no son fiables; las medianas típicas sí indican que
la inferencia FE/PE del Transformer (≈3 s/768 ep. a k = 2) es más barata que la búsqueda OBS en malla (≈11 s) y que `lat-eval` de ÓRBITA (≈45 s,
que además calcula LAT, LAT_EP y oráculos).

## Observaciones clave
1. **El entrenamiento variacional (β·KL con μ muestreado) destruye la capacidad de predicción a 16 pasos**: TFE conocidas acc16 0.0424 (CONFIRM)
   vs TF 0.2196 y MLP 0.1646 con la misma arquitectura/presupuesto; incluso con c verdadero queda muy por debajo de ÓRBITA (0.3562).
2. **La atención sobre estados, sin FE, es el mejor brazo nuevo pero sigue por debajo de ÓRBITA** en conocidas (TF − ÓRBITA −0.1366 [−0.1597, −0.1144]).
3. **Exploratorio (no prerregistrado, sin IC pareado):** en nuevas k = 2, TF-PE (0.0605 / 0.0544) tiene medias por encima de ÓRBITA-LAT_EP
   (0.0376 / 0.0339) y ÓRBITA-LAT (0.0457 / 0.0359) en ambas fases; con k = 4 y 8 también (TF-PE 0.0688 y 0.1084 en CONFIRM vs LAT 0.0544 y
   0.0991). La adaptación por gradiente sobre un latente en un Transformer determinista podría ser útil para dinámicas nuevas; habría que
   prerregistrarlo como hipótesis propia (TF-PE − ÓRBITA-LAT_EP) antes de afirmarlo.
4. **Inferencia iterativa vs malla:** H4b aceptada y creciente con k (CONFIRM +0.0078 k = 2, +0.0352 k = 4, +0.0601 k = 8); el término de
   complejidad no aporta (H4c −0.0168 a k = 2, ≈0 a k = 4/8).
5. **Calibración:** F detecta fallos por encima del azar (H3a), pero peor que la sorpresa de ÓRBITA (H3b) y peor que el error de predicción del
   TF sin FE (0.8536 vs 0.6616 en CONFIRM, k = 2).
6. Todos los brazos quedan lejos del techo ORACLE_LS en nuevas (0.0742 CONFIRM, k = 2): las familias 8–11 siguen fuera de distribución.

## Desviaciones respecto al prerregistro
- Ninguna en código, hiperparámetros, conjuntos, seeds ni análisis. La cadena terminó sin fallos ni reanudaciones (los logs de entrenamiento no
  muestran reanudación; `chain.log` sin líneas `incomplete`).
- H4d y los controles no tienen criterio numérico explícito en el prerregistro; se aplicó el mismo criterio que H4a/H4b (IC95 > 0).
- H5 es descriptiva; las medias de coste están contaminadas por carga concurrente (ver arriba).

## Archivos
- Salidas crudas: `/workspace/orbita-data/tf/{dev,confirm}/{TFE,TF,MLP}/` (`w_<seed>.pt`, `train_<seed>.log`, `eval_<seed>.csv`, `eval_<seed>.csv.time.json`),
  `/workspace/orbita-data/tf/{dev,confirm}/ORBITA/` (`fam_<seed>.csv`, `lat_<seed>_k{2,4,8}.csv`, `.time`), episodios en `.../eps/`, `chain.log`.
- En el repo: `artifacts/orbita/transformer_fe/analyze_{dev,confirm}.md`, `per_seed_{dev,confirm}.json`, `chain.log`.
