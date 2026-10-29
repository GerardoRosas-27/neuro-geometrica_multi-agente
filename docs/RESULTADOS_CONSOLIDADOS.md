# Resultados consolidados — selección de la mejor arquitectura y ciclo de mejoras v5

**Ganadora: CONSTELACIÓN v4** (rama `exp/constelacion-v4`, commit `fd80320`, confirmada con entrenamiento igualado en `exp/cst-vs-orbita` @ `c1b1f54`).  
**Reporte de mejoras del siguiente ciclo (CONSTELACIÓN v5):** [`docs/mejoras_v5/REPORTE_MEJORAS.md`](mejoras_v5/REPORTE_MEJORAS.md) · [`docs/mejoras_v5/DIAGNOSTICO.md`](mejoras_v5/DIAGNOSTICO.md) · [`docs/mejoras_v5/PREREG.md`](mejoras_v5/PREREG.md).

Todas las cifras están copiadas de los `RESULTADOS.md` de cada rama (copias literales en `docs/historico/<exp>/`) y, para v4 y `cst-vs-orbita`, contrastadas con `artifacts/constelacion_v4/analysis_dev_confirm.json` y `artifacts/cst_vs_orbita/analysis.json`. Método común: 16 semillas DEV (0xA400–0xA40F) + 16 CONFIRM (0xB400–0xB40F), bootstrap pareado por semilla 10 000 réplicas, semilla 12345, IC95 percentil, δ = 0,05, sin reajuste entre fases, 0 consultas TEST.

## Tabla de experimentos

| Experimento | Rama @ commit | Primaria DEV | Primaria CONFIRM | No-daño | E33 (CST) | Veredicto |
|---|---|---|---|---|---|---|
| ÓRBITA-v1 (línea base) | `exp/orbita` @ `1bc624f` (docs en `release/orbita-main` @ `b7fe616`) | vs v4 E3 acc h16 +0,0488 [−0,0029, +0,1065] | +0,1065 [+0,0664, +0,1504] | — (es la referencia) | 16/16 y 16/16 | **Empate / no concluyente** vs v4 (falla en DEV); supera al MLP M3; es la base de todos los experimentos posteriores |
| ESTELA v0 + Enm. 1 | `exp/estela` @ `db42e6a` | S3: −0,228 [−0,341, −0,113] | S3: −0,269 [−0,389, −0,149] | **FALLA**: S2 acc16 conocidas −0,267 / −0,246 [−0,274, −0,215] | — (AUROC sorpresa 0,448 / 0,407 vs ÓRBITA 0,958 / 0,951) | **Negativo.** S3 sólo pasa porque ÓRBITA+LAT empeora; ESTELA es peor que ÓRBITA congelada (+0,223 log CONFIRM) |
| ECO | `exp/eco` @ `b731ff3` | 0,0000 [0, 0] | 0,0000 [0, 0] | OK (efecto de LAT, no de ECO) | 16/16 (vía LAT) | **No concluyente.** 0/256 sueños aceptados; Δ nunca actuó |
| MAREA | `exp/marea` @ `ee99a91` | pendiente −0,0115 [−0,0126, −0,0105] | −0,0120 [−0,0131, −0,0110] | OK (−0,237 / −0,240 ln err conocidas) | — | **Primaria aceptada, pero no competitiva**: vs ÓRBITA+LAT +1,513 / +1,439 ln err nuevas (≈4,5× más error) |
| CONSTELACIÓN v2 | `exp/constelacion-v2` @ `fdefd62` | +0,807 [+0,702, +0,908] | +0,798 [+0,688, +0,916] | OK formal pero vacío (acc16 = 0) | 0/16 | **Refutada**; colapso 4/16 y 5/16 |
| CONSTELACIÓN v3 | `exp/constelacion-v3` @ `e5718e8` | −0,089 [−0,152, −0,018] | −0,115 [−0,193, −0,030] | OK (+0,007 / +0,0005) | **0/16** | Primaria + no-daño OK; **E33 falla** |
| **CONSTELACIÓN v4** | `exp/constelacion-v4` @ `fd80320` | −0,076 [−0,141, −0,001] | **−0,107 [−0,181, −0,024]** | OK (0,000 [0, 0]; acc16 ≈ 0,877) | **16/16 y 16/16** | **Cumple los 3 criterios preregistrados** |
| **CST v4 vs ÓRBITA-eq+LAT** (entrenamiento igualado 122 880 muestras) | `exp/cst-vs-orbita` @ `c1b1f54` | H1: **−0,6578 [−0,7750, −0,5416]** | H1: **−0,6902 [−0,8083, −0,5725]** | H3: +0,5024 / +0,4771 [+0,4551, +0,4995] | E33-k4: 1,000 vs 0,664 (H2 +0,3359) | **CST gana** (H1∧H3 en DEV y CONFIRM) |
| CONSTELACIÓN v4 — Mejoras M1/M2/M3 | `exp/constelacion-v4-mejoras` @ `0967539` | Paso 0 PASS; M2 rechazada en meta-val (0,00 ≤ 0,02); M3 degrada conocidas en tuning (−5,76 vs −10,12) | PREREG congelado (pendiente F2) | — | — | **Diagnóstico clave para v5:** el MLP rápido M3 sin separar base afín daña conocidas |
| ÓRBITA Dinámica Latente (Gemma capa 13 → PCA 16) | `exp/orbita-dinamica-latente` @ `18ba4a4` | H1 ORB_LIBRE vs AR2: **−0,0647 [−0,1040, −0,0262]** | No ejecutado (parada en gate DEV) | H3 vs ORB_SESGO: −0,2876 (no pasa) | — | **No confirmado (parada en DEV)**: AR2 supera a ORB_LIBRE |
| ÓRBITA vs TimesFM 2.5 (sobre latentes `dinlat`) | `exp/orbita-vs-timesfm` @ `e56ccd6` | H1 ORB_SESGO vs TIMESFM: **+0,3296 [+0,2979, +0,3615]**; H1b ORB_LIBRE vs TIMESFM: +0,0419 [+0,0039, +0,0796] | Pendiente (relanzamiento sin NaN) | H2 TIMESFM vs PERS: +0,3548 | — | **Pasa gate DEV→CONFIRM** (con ORB_SESGO; ORB_LIBRE pierde en geométricas −0,4207) |

Primaria de CONSTELACIÓN: med log rel_err h16 en familias nuevas, CST − ÓRBITA-v1 (<0 = mejor). En ESTELA/ECO/MAREA/`cst-vs-orbita`/`dinlat`/`vstimesfm` la primaria tiene definición propia (ver su fila y su documento).

### Contexto: Nova-2 (`exp/orbita-nova2` @ `b25c97e`) — los tres fracasan
- **Exp1 (C4 bayesiano):** B1 primaria REFUTADA en CONFIRM (IC inf +0,038 > 0); B3 no-daño falla.
- **Exp2 (base compartida):** E1 primaria aceptada (DEV −0,096 [−0,167; −0,027], CONFIRM −0,080 [−0,157; −0,004]) pero **E3 no-daño falla** con fuerza (−21 a −27 puntos acc16 en conocidas).
- **Exp3 (BOCD):** F1 inconclusa (+0,000 exacto ambas fases); F3 no-daño falla por NOVA_BOCD (−0,17/−0,14/−0,05 DEV, −0,18/−0,14/−0,06 CONFIRM).

## ÓRBITA-v1: la línea base
Fuente: `docs/resultados_orbita.md` (copia en `docs/historico/orbita/`, junto con `resultados_orbita_completo.md`, `preregistro_orbita.md` y `resultados_latente_largo_plazo.md`). Preregistro `f5d7781`; seeds y bootstrap iguales al resto.

**Experimento propio (ÓRBITA vs v4 §16):**
| | DEV | CONFIRM |
|---|---|---|
| Gate E33 ÓRBITA (v4_E3) | 16/16 (13/16) | 16/16 (9/16) |
| acc 1 paso ÓRBITA | 0,9783 | 0,9819 |
| acc h1 / h2 / h4 / h8 / h16 | 0,984 / 0,984 / 0,942 / 0,706 / 0,313 | 0,988 / 0,982 / 0,912 / 0,687 / 0,323 |
| ÓRBITA − v4_E3, acc h16 | +0,0488 [−0,0029, +0,1065] | +0,1065 [+0,0664, +0,1504] |
| ÓRBITA − MLP M3, acc h16 | +0,0830 [+0,0224, +0,1416] | +0,1133 [+0,0615, +0,1641] |

Veredicto preregistrado: **empate / no concluyente** frente a v4 (el IC de h16 cruza 0 en DEV); supera al control MLP en ambas fases.

**ÓRBITA como referencia dentro de los demás experimentos** (k=4, DEV / CONFIRM):
| Métrica | ÓRBITA | ÓRBITA+LAT | Fuente |
|---|---|---|---|
| med log rel_err h16 nuevas | −0,799 / −0,793 | −0,357 / −0,298 | ESTELA, CONSTELACIÓN v4 |
| acc16 conocidas | 0,354 / 0,338 | 0,430 / 0,426 | ESTELA, CONSTELACIÓN v4 |
| acc16 nuevas | 0,026 / 0,025 | — | CONSTELACIÓN v4 |
| med log h16 conocidas | −2,52 / −2,50 | — | CONSTELACIÓN v4 |
| AUROC detección de novedad | 0,958 / 0,951 | — | ESTELA |
| ln err nuevas (t≥4, escala MAREA) | −2,83 / −2,81 | −2,91 / −2,84 | MAREA |

## Comparación homologada con presupuesto igualado (`exp/cst-vs-orbita` @ `c1b1f54`)
Con exactamente las mismas **122 880 transiciones de entrenamiento** por semilla (flujos A+B sin replay del teacher) y evaluación homologada en `reid(k=4)` sin etiquetas verdaderas (`artifacts/cst_vs_orbita/RESULTADOS.md`):
- **DEV (n=16):** H1 `cst − orbita_eq_lat` = **−0,6578 [−0,7750, −0,5416]** (PASA); H2 `E33-k4 acc16` = **+0,3555 [+0,3223, +0,3838]** (PASA); H3 `known_acc16` = **+0,5024 [+0,4810, +0,5244]** (PASA).
- **CONFIRM (n=16):** H1 `cst − orbita_eq_lat` = **−0,6902 [−0,8083, −0,5725]** (PASA); H2 `E33-k4 acc16` = **+0,3359 [+0,3066, +0,3643]** (PASA); H3 `known_acc16` = **+0,4771 [+0,4551, +0,4995]** (PASA).
- **Debilidad expuesta de CST en $k=0$:** cuando se evalúa sin prefijo observado pero con contexto verdadero (`1-paso acc` y `E33-ctx`), CST obtiene `0,0417` y `0,0029` (`0,0137` con `cst_prior`), frente a `0,9819` y `0,3232` de `orbita_v1`. Esta debilidad se resuelve con la **Mejora M5 (`orb_prior_wf`)** en [`docs/mejoras_v5/REPORTE_MEJORAS.md`](mejoras_v5/REPORTE_MEJORAS.md).

## Por qué gana CONSTELACIÓN v4 y qué resuelve el reporte de mejoras v5
1. Es el **único** experimento que cumple los tres criterios preregistrados en DEV y CONFIRM tanto frente a ÓRBITA-v1 congelada como frente a `ÓRBITA-eq+LAT` con presupuesto igualado.
2. Sin embargo, el análisis cruzado de `constelacion-v4`, `cst-vs-orbita`, `constelacion-v4-mejoras` y `dinlat` revela cinco cuellos de botella estructurales documentados y resueltos en [`docs/mejoras_v5/REPORTE_MEJORAS.md`](mejoras_v5/REPORTE_MEJORAS.md):
   - **M4:** Campo rápido bifásico en prefijo completo (`W^F_aff` exacto por RLS/Cholesky $3\times 3$ $\oplus$ corrector residual ortogonalizado $g_{\text{res}}\cdot W^F_{\text{res}}$), que cierra la brecha de `0,35–0,45 log` frente a `oracle_icl` e impide por construcción la degradación en conocidas de M3 (`−10,12 → −5,76`).
   - **M5:** Destilación causal del prior de ÓRBITA en $W^F(0)$ (`orb_prior_wf`) para elevar `1-paso acc` (`0,0417 → ≈0,982`) y `E33-ctx` (`0,0137 → ≥0,95`).
   - **M6:** Codificación predictiva de transición $u_p(t) = [x_p(t-1), \alpha_\Delta \Delta x_p(t)]$ + pooling de constelación $\bar{\mu}_L$ + banco lento $W^P(\bar{\mu}_L)$ para obtener atribución real de PC (`CST − fixed_w < 0`).
   - **M7:** Anclaje `VAR(2)` + Koopman-Brenier residual + RLS episódico (`dinlat-v2`) para superar a `AR2` y `TimesFM 2.5` sobre latentes Gemma capa 13.
   - **M8:** Integración en `agentic_web`, rutas portables con fallback local y gate `cst4-paso0` en CI.

## Documentos originales e históricos
- Reporte de mejoras v5: [`docs/mejoras_v5/REPORTE_MEJORAS.md`](mejoras_v5/REPORTE_MEJORAS.md), [`docs/mejoras_v5/DIAGNOSTICO.md`](mejoras_v5/DIAGNOSTICO.md), [`docs/mejoras_v5/PREREG.md`](mejoras_v5/PREREG.md).
- Copias literales de resultados por rama: `docs/historico/{estela,eco,marea,constelacion_v2,constelacion_v3,constelacion_v4,constelacion_v4_mejoras,cst_vs_orbita,orbita,orbita_nova2,dinamica_latente,orbita_vs_timesfm}/`.
