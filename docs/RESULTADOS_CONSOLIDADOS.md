# Resultados consolidados — selección de la mejor arquitectura

**Ganadora: CONSTELACIÓN v4** (rama `exp/constelacion-v4`, commit `fd80320`). Esta rama (`exp/mejor-arquitectura`) parte de ese commit sin cambios de código; sólo añade documentación.

Todas las cifras están copiadas de los `RESULTADOS.md` de cada rama (copias literales en `docs/historico/<exp>/`) y, para v4, contrastadas con `artifacts/constelacion_v4/analysis_dev_confirm.json`. Método común: 16 semillas DEV (0xA400–0xA40F) + 16 CONFIRM (0xB400–0xB40F), bootstrap pareado por semilla 10 000 réplicas, semilla 12345, IC95 percentil, δ = 0,05, sin reajuste entre fases, 0 consultas TEST.

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

Primaria de CONSTELACIÓN: med log rel_err h16 en familias nuevas, CST − ÓRBITA-v1 (<0 = mejor). En ESTELA/ECO/MAREA la primaria tiene definición propia (ver su fila y su documento).

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

LAT (latente de Koopman, `resultados_latente_largo_plazo.md`): Exp. 1 H1a–H3 **rechazadas** (fallan en DEV), H4 no-inferioridad vs oráculos y H5 sin olvido **aceptadas**; Exp. 2 **no aprende a largo plazo** (L1–L3 fallan). CONSTELACIÓN v4 mejora a ÓRBITA en nuevas en −0,076 / −0,107 log y en conocidas (acc16 0,877 vs 0,354 / 0,338), y usa la cabeza de ÓRBITA para h1 en E33.

No encontré en las ramas un único documento que dé la primaria de CONSTELACIÓN (med log h16 nuevas) para ÓRBITA+LAT; esa comparación no se hizo.

## Estado respecto a `main`
CONSTELACIÓN v4 **no está en `main`**. Esta rama se propone para revisión; no se ha fusionado.

## Por qué gana CONSTELACIÓN v4
1. Es el **único** experimento que cumple los tres criterios preregistrados en DEV y CONFIRM: primaria aceptada (IC sup < 0: −0,001 y −0,024), no-daño no vacío (acc16 conocidas 0,877 sin cambio) y E33 16/16.
2. v3 tiene una primaria ligeramente mejor, pero **falla E33 (0/16)**, que es un criterio preregistrado. MAREA y ECO no compiten en valor absoluto o no hacen nada; ESTELA y v2 son negativos.

### Matiz frente a v3
Pareado por semilla, v4 − v3 en nuevas: **+0,013 [+0,002, +0,026] DEV; +0,009 [+0,002, +0,017] CONFIRM** — v4 es ligeramente *peor* que v3 en la primaria (~0,01 log, IC excluye 0). Si E33 no importara, v3 sería marginalmente preferible. En conocidas v4 mejora la med log (−9,96/−10,06 vs −9,33/−9,40).

## Limitaciones (honestas)
- **Atribución a codificación predictiva: NO.** Ningún control requerido (fixed_w, no_err, t1) da IC sup < 0. Toda la ganancia viene del **campo rápido por episodio W^F**: `no_fast` empeora −3,18 / −3,20 log.
- **E33 se resuelve delegando h1 a ÓRBITA**: acc h1 CST − ÓRBITA = 0 exacto; con la cabeza v3 la misma red sigue en 0/16. Lo propio de CST es la estabilidad a largo horizonte (h16 1,0 vs ÓRBITA ≈ 0,31–0,32; +0,688 / +0,677).
- **Ventaja pequeña en familias nuevas** (≈ −0,08 / −0,11 log), y en DEV el margen es mínimo (IC sup −0,001).
- **acc16 absoluta en familias nuevas muy baja**: CST 0,069 / 0,066 (ÓRBITA 0,026 / 0,025).

## Cómo reproducir / ejecutar
Desde la raíz del repo, con pesos ÓRBITA en `artifacts/orbita/{dev,confirm}/weights_<seed>.json`:
```bash
cargo build --release --manifest-path orbita/Cargo.toml   # binario orbita/target/release/orbita
JOBS=7 orbita/scripts/constelacion_v4.sh dev       # 16 semillas 0xA400–0xA40F (reanudable vía .done)
JOBS=7 orbita/scripts/constelacion_v4.sh confirm   # 16 semillas 0xB400–0xB40F
python3 scripts/analyze_constelacion_v4.py > artifacts/constelacion_v4/analysis_dev_confirm.json
```
Por semilla: `orbita/target/release/orbita cst4-run --phase dev --weights artifacts/orbita/dev/weights_0xA400.json --seed 0xA400 --out artifacts/constelacion_v4`. `orbita/scripts/constelacion_v4_chain.sh` encadena DEV→CONFIRM→análisis (tiene ruta absoluta del worktree original y hace commit/push; revísalo antes de usarlo). Preregistro: `docs/constelacion_v4/PREREG.md`; diagnóstico: `docs/constelacion_v4/DIAGNOSTICO.md`.

## Documentos originales
`docs/historico/{estela,eco,marea,constelacion_v2,constelacion_v3,constelacion_v4,orbita_nova2}/` — copias literales de los RESULTADOS de cada rama.
