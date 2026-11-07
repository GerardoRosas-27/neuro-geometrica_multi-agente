# Resultados Consolidados — Arquitectura Ganadora BQ10 y Líneas de Investigación

## Estructura de las Líneas de Investigación

El proyecto cuenta con dos líneas de investigación principales basadas en la arquitectura ganadora **BQ10 (M1+M2+M6+M7 da=10)** ejecutada nativamente sobre GPU AMD Radeon RX 7800 XT (WGPU/Vulkan):

### 1. Línea Principal: Sustrato de Campo Continuo Token a Token (`gemma_campo`)
* **Objetivo:** Operar como memoria latente continua paso a paso durante el flujo generativo del LLM, acoplando innovación causal $e(x_t)$ y proyección en subespacio contextual $V_{\text{ctx}}$ ($k_{\text{skip}}=16$).
* **Ramas:** `exp/gemma-campo-chat`, `exp/gemma-campo-infinito`.
* **Resultados en Test ($n=60$ conversaciones, bootstrap 10,000):**
  * $z$-MSE a 1 paso: **0.195968** (vs repeat 1.238695, mejora de **84.2%**).
  * $\Delta\text{NLL}(\text{campo}-\text{oracle}) = \mathbf{+0.018593}$ (IC95 $[+0.010154, +0.027282] \le 0.05 \implies$ **ÉXITO H2**).
  * $\Delta\text{NLL}(\text{campo}-\text{repeat}) = \mathbf{-1.315852}$ (IC95 $[-1.389505, -1.242132] < 0 \implies$ **ÉXITO H3/H4**).
  * Perplejidad (PPL): **5.661** (campo) vs **5.554** (LLM completo intacto) vs **21.104** (repeat).
  * Olvido catastrófico en Shard 0 tras 9 shards continuos (3 horas): **+0.064%** ($\le 5.0\%$).
  * Reproducibilidad bit a bit: $\max|W_a - W_b| = 0.000\times 10^0$.
  * Todas las hipótesis prerregistradas cumplidas: **H1, H1k, H2, H3, OLV, EST, REP**.
  * Reporte detallado: [`docs/gemma_campo_chat/run_3h_bq10/RESULTADOS.md`](gemma_campo_chat/run_3h_bq10/RESULTADOS.md).

### 2. Segunda Línea de Investigación: Campo de Estado Continuo por Turno (`campo_estado_turno`)
* **Objetivo:** Predecir y condensar la trayectoria de la conversación a nivel de turno semántico completo ($z_{\text{turn}} \in \mathbb{R}^{64}$ con $F=2721$ modos ortogonales de Lie) y condicionar la generación del LLM congelado mediante un adaptador soft-prompt ($z \to P \cdot H$).
* **Rama:** `exp/campo-estado-turno`.
* **Resultados en Test ($n=200$ conversaciones, 759 turnos de asistente, bootstrap 10,000):**
  * **T1 (Predicción de Dinámica de Turno):** $z\text{-MSE}(\text{campo}) = \mathbf{0.368282}$ vs repeat $1.091795$ ($\Delta = -0.723514$, $P \le 0 = 1.0 \implies$ **ÉXITO**).
  * **T2 (vs Control MLP 2 capas):** $\Delta = \mathbf{-0.018591}$ (IC95 $[-0.024680, -0.012612] \le 0.019344 \implies$ **ÉXITO**). El campo BQ10 supera al MLP.
  * **T3 (Eficacia Decodificación Soft-Prompt):** $\text{NLL}(\text{campo}) - \text{NLL}(\text{none}) = \mathbf{-1.317461}$ ($P \le 0 = 1.0 \implies$ **ÉXITO**). La perplejidad cae de **6.999** a **1.874** (reducción del 73.2% en incertidumbre).
  * **T4 (Especificidad Contextual):** $\text{NLL}(\text{campo}) - \text{NLL}(\text{shuf}) = \mathbf{-0.411890}$ ($P \le 0 = 1.0 \implies$ **ÉXITO**). Brecha vs oráculo perfecto $z_{\text{true}}$ es de solo $+0.044$ nats.
  * **No-Daño:** Pesos del LLM 100% intactos (SHA-256 verificado: `78685cdf...`), identidad $\max|\text{embeds} - \text{ids}| = 0.0 < 0.05$.
  * Reporte detallado: [`docs/campo_estado_turno/run_3h_bq10/RESULTADOS.md`](campo_estado_turno/run_3h_bq10/RESULTADOS.md).

---

# Resultados consolidados — selección de la mejor arquitectura

**Ganadora actual: BQ10 (bio-quantum v10) = núcleo ATLAS-FW (M1+M2+M6) + M7 DA-STR** (regularización ridge por grado, `da=10`; rama `exp/bio-quantum-v10-rust` @ `f2b6813`, integrada vía `integra/bio-quantum-v10-main`). CONFIRM: BQ10−v4 new_med_log **−0.6083 [−0.7415, −0.4652]**, BQ10−ATLAS-FW(ALL) **−0.1563 [−0.2692, −0.0434]**, known_acc16 +0.0103 vs v4, E33 16/16, 0 NaN, 0 TEST. Ver [`bio_quantum_v10/RESULTADOS.md`](bio_quantum_v10/RESULTADOS.md) y [`bio_quantum_v10/ANALISIS_POR_QUE_GANA.md`](bio_quantum_v10/ANALISIS_POR_QUE_GANA.md).

**Ganadora previa: ATLAS-FW** (PR #47, merge `1c845d6`): CONFIRM (seeds propias 0xB480–) V−v4 −0.4082 [−0.4985, −0.3190]. Ver [`atlas_fw/README.md`](atlas_fw/README.md). **Base: CONSTELACIÓN v4** (`fd80320`).

## Tabla de métricas de las mejores versiones (arnés común BQ10)

Fuente: `artifacts/bio_quantum_v10/{bq_dev,bq_confirm}/metrics_0x*.json` (las 4 arquitecturas se evalúan en la **misma** corrida por semilla, sobre la misma v4 entrenada → contrastes pareados). Generada con `python3 scripts/consolidate_bq10.py --md` (JSON: `artifacts/bio_quantum_v10/consolidado.json`). Bootstrap pareado por semilla 10 000, seed 12345, IC95 percentil. Tiempos = `wall_s` registrado en los artefactos (box, 7 jobs en paralelo, nice 10). ÓRBITA-v1 es la referencia con pesos congelados de `artifacts/orbita/` (su coste de entrenamiento no se mide en este arnés). Nota: las cifras ATLAS-FW aquí (seeds 0xA400/0xB400) difieren de las de su PREREG (seeds 0xA480/0xB480, −0.408); ambas son reales.

#### DEV 0xA400–0xA40F — medias [IC95]

| arquitectura | new_med_log h16 k=4 | known_acc16 | new_acc16 | E33 | tiempo/semilla del brazo (s) |
|---|---|---|---|---|---|
| ÓRBITA-v1 (ref., pesos congelados) | -0.7990 [-0.9008, -0.6937] | 0.3540 [0.3145, 0.3936] | 0.0260 [0.0143, 0.0404] | 16/16 | no medido (se entrena aparte) |
| CONSTELACIÓN v4 | -0.8748 [-1.0116, -0.7460] | 0.8770 [0.8750, 0.8794] | 0.0690 [0.0508, 0.0885] | 16/16 | 4.5 (+380 s entrenar v4, compartido) |
| ATLAS-FW (ALL) | -1.3856 [-1.5014, -1.2745] | 0.8970 [0.8916, 0.9028] | 0.1549 [0.1328, 0.1771] | 16/16 | 41.6 (sobre v4 entrenada) |
| **BQ10 (bio-quantum v10)** | -1.5245 [-1.6494, -1.4050] | 0.8877 [0.8833, 0.8931] | 0.1445 [0.1276, 0.1628] | 16/16 | 13.0 (sobre v4 entrenada) |

Pareados (por semilla) — Δ [IC95]:

| contraste | Δ new_med_log | Δ known_acc16 | Δ new_acc16 |
|---|---|---|---|
| BQ10-v4 | -0.6496 [-0.7479, -0.5486] | +0.0107 [+0.0073, +0.0142] | +0.0755 [+0.0599, +0.0911] |
| BQ10-ALL | -0.1388 [-0.2202, -0.0550] | -0.0093 [-0.0142, -0.0049] | -0.0104 [-0.0286, +0.0091] |
| ALL-v4 | -0.5108 [-0.6051, -0.3963] | +0.0200 [+0.0151, +0.0249] | +0.0859 [+0.0612, +0.1107] |
| v4-ÓRBITA | -0.0758 [-0.1415, -0.0013] | +0.5229 [+0.4834, +0.5620] | +0.0430 [+0.0247, +0.0612] |
| BQ10-ÓRBITA | -0.7254 [-0.8288, -0.6243] | +0.5337 [+0.4941, +0.5728] | +0.1185 [+0.1016, +0.1367] |

Wall total por semilla (v4 train + v4, ALL, CORE, BQ10, BQR10 + ÓRBITA): 465 s.

#### CONFIRM 0xB400–0xB40F — medias [IC95]

| arquitectura | new_med_log h16 k=4 | known_acc16 | new_acc16 | E33 | tiempo/semilla del brazo (s) |
|---|---|---|---|---|---|
| ÓRBITA-v1 (ref., pesos congelados) | -0.7928 [-0.9119, -0.6826] | 0.3384 [0.3081, 0.3677] | 0.0247 [0.0143, 0.0365] | 16/16 | no medido (se entrena aparte) |
| CONSTELACIÓN v4 | -0.8997 [-0.9660, -0.8265] | 0.8770 [0.8750, 0.8799] | 0.0664 [0.0482, 0.0872] | 16/16 | 4.5 (+449 s entrenar v4, compartido) |
| ATLAS-FW (ALL) | -1.3518 [-1.4830, -1.2185] | 0.9028 [0.8975, 0.9082] | 0.1562 [0.1263, 0.1875] | 16/16 | 41.0 (sobre v4 entrenada) |
| **BQ10 (bio-quantum v10)** | -1.5081 [-1.6382, -1.3865] | 0.8872 [0.8838, 0.8906] | 0.1484 [0.1250, 0.1758] | 16/16 | 12.3 (sobre v4 entrenada) |

Pareados (por semilla) — Δ [IC95]:

| contraste | Δ new_med_log | Δ known_acc16 | Δ new_acc16 |
|---|---|---|---|
| BQ10-v4 | -0.6083 [-0.7415, -0.4652] | +0.0103 [+0.0068, +0.0137] | +0.0820 [+0.0664, +0.0977] |
| BQ10-ALL | -0.1563 [-0.2692, -0.0434] | -0.0156 [-0.0190, -0.0122] | -0.0078 [-0.0286, +0.0130] |
| ALL-v4 | -0.4520 [-0.5613, -0.3247] | +0.0259 [+0.0205, +0.0312] | +0.0898 [+0.0651, +0.1133] |
| v4-ÓRBITA | -0.1070 [-0.1811, -0.0238] | +0.5386 [+0.5093, +0.5684] | +0.0417 [+0.0221, +0.0612] |
| BQ10-ÓRBITA | -0.7153 [-0.8588, -0.5613] | +0.5488 [+0.5186, +0.5791] | +0.1237 [+0.0990, +0.1497] |

Wall total por semilla (v4 train + v4, ALL, CORE, BQ10, BQR10 + ÓRBITA): 533 s.


Lectura honesta: BQ10 gana en la primaria (new_med_log) a v4 y a ATLAS-FW con IC excluyendo 0 en DEV y CONFIRM, y es ~3× más barato por semilla que ALL (sin M3–M5). Coste: known_acc16 −0.009/−0.016 frente a ALL (sigue +0.010 sobre v4, dentro del no-daño δ=0.05), y new_acc16 sin diferencia significativa frente a ALL. M8 (cota Brenier, brazo BQR10) es **inerte** (idéntico a BQ10; ver `bio_quantum_v10/DIAGNOSTICO_M8.md`) y queda desactivado.

### Cómo correr BQ10
```bash
cargo build --release --manifest-path orbita/Cargo.toml
# una semilla (DEV): BQ10 + v4 (v4 siempre corre como contraste pareado)
orbita/target/release/orbita atlas-run --phase bq_dev --seed 0xA400 --weights artifacts/orbita/dev/weights_0xA400.json --out artifacts/bio_quantum_v10 --arms v4,BQ10
# fases completas (reanudable vía .done) y análisis
JOBS=7 orbita/scripts/bq10_phase.sh bq_dev v4,ALL,CORE,BQ10,BQR10
JOBS=7 orbita/scripts/bq10_phase.sh bq_confirm v4,ALL,CORE,BQ10,BQR10
python3 scripts/analyze_bq10.py --md; python3 scripts/consolidate_bq10.py --md
```
Sin `--arms`, `atlas-run` corre todos los brazos (v4, M1…M6, ALL, ablaciones, BQ*); el default no se cambió para no alterar la reproducibilidad de ATLAS-FW. Valores congelados: `docs/bio_quantum_v10/frozen.json` (`primary_arm: BQ10`, `da=10`, `LIB_DEG`, `RB_K`).

## BQ10-GPU (ARNÉS PROPIO — no entra en la tabla de arriba)

Línea `exp/bq10-gpu-amd` @ `6808294`, crate `bio_quantum_engine/`. Informe: [`bq10/README.md`](bq10/README.md).

Esta línea **no desplaza** al BQ10 canónico ni a ATLAS-FW: usa otro generador, otras métricas (media de medianas por semilla) y no tiene contraste pareado contra v4. El `eval` del crate **no carga** los pesos entrenados en GPU (`weights_x50.bin`). La comparación «+14.5× vs 6.9 % en v4» era un texto fijo en el binario y quedó marcada como no verificada.

Entrenamiento GPU x50 (JSON `bio_quantum_engine/training_results_gpu_rx7800xt.json`): **1 638 400** episodios, **86.804 s**, adaptador `D3D12 (AMD Radeon RX 7800 XT)`, 1 024 puntos / 1 024D / 4 096 sinapsis, MSE reportada 0.000552 → 0.000478, 0 violaciones de Dale. Detalle de x10/x20 (JSON histórico): ver `docs/bq10/README.md`.

Evaluación del arnés propio (`docs/bq10_gpu_amd/logs/eval_{dev,confirm}.log`):

| Fase | known acc16 | known med log | OOD acc16 | OOD med log | E33 |
|---|---:|---:|---:|---:|---|
| DEV `0xA400–0xA40F` | 100.00 % | −12.7469 | 99.80 % | −8.1930 | 16/16 |
| CONFIRM `0xB400–0xB40F` | 100.00 % | −12.8178 | 100.00 % | −7.8845 | 16/16 |

Cuando se evalúa *con* los pesos GPU (`eval_trained`), x50 queda en 0.00 % acc16 y med log ≈ +0.59 / +0.45 (DEV), peor que persistencia. Ver [`bq10_gpu_amd/EVAL_PESOS_ENTRENADOS.md`](bq10_gpu_amd/EVAL_PESOS_ENTRENADOS.md).

## Campo sobre LLM congelado (no entra en la tabla CST4)

Línea principal: `gemma_campo` @ `a420311` (`docs/gemma_campo_chat/run_20261104/`). Camino alterno: estado por turno @ `a2c9c9f` (`docs/campo_estado_turno/run_20261105/`). Informe y reproducción: [`campo/README.md`](campo/README.md).

**No desplaza** a ATLAS-FW ni al BQ10 canónico: otro arnés (activaciones de `qwen/qwen3-vl-4b`, capa 13), otras métricas (z-MSE y NLL de lenguaje), un solo run. Las dos ramas no son comparables entre sí (token vs turno; test n=60 vs n=200).

Línea principal (test n=60, bootstrap 10 000 / semilla 12345, `eval_test.json`): z-MSE k=1 con `e(x_t)` campo 0.197222 vs repeat 1.238695 (Δ −1.041472 [−1.063950, −1.016694]); NLL campo 1.7339503705501556 vs LLM intacto 1.7145656049251556 vs oracle 1.7150465667247772 (campo−oracle +0.01890380382537842 [0.010692580093940099, 0.027486240863800047] ≤ 0.05 → H2 éxito). H1, H1k, H3, OLV, EST, REP también éxito.

Camino alterno (test n=200): T1–T4 y no-daño pasan. z-MSE campo 0.3685527244210243 vs repeat 1.0917954957485199 vs mlp 0.38687286220490935. NLL campo 0.6475953880107791 vs none 1.9457939405404034 vs shuf 0.9346708592687366 vs mlp 0.6469799832911522 (campo≈mlp en decodificación; T3 vs none infla el estilo del adaptador).

## Historial de experimentos


| Experimento | Rama @ commit | Primaria DEV | Primaria CONFIRM | No-daño | E33 (CST) | Veredicto |
|---|---|---|---|---|---|---|
| ÓRBITA-v1 (línea base) | `exp/orbita` @ `1bc624f` (docs en `release/orbita-main` @ `b7fe616`) | vs v4 E3 acc h16 +0,0488 [−0,0029, +0,1065] | +0,1065 [+0,0664, +0,1504] | — (es la referencia) | 16/16 y 16/16 | **Empate / no concluyente** vs v4 (falla en DEV); supera al MLP M3; es la base de todos los experimentos posteriores |
| ESTELA v0 + Enm. 1 | `exp/estela` @ `db42e6a` | S3: −0,228 [−0,341, −0,113] | S3: −0,269 [−0,389, −0,149] | **FALLA**: S2 acc16 conocidas −0,267 / −0,246 [−0,274, −0,215] | — (AUROC sorpresa 0,448 / 0,407 vs ÓRBITA 0,958 / 0,951) | **Negativo.** S3 sólo pasa porque ÓRBITA+LAT empeora; ESTELA es peor que ÓRBITA congelada (+0,223 log CONFIRM) |
| ECO | `exp/eco` @ `b731ff3` | 0,0000 [0, 0] | 0,0000 [0, 0] | OK (efecto de LAT, no de ECO) | 16/16 (vía LAT) | **No concluyente.** 0/256 sueños aceptados; Δ nunca actuó |
| MAREA | `exp/marea` @ `ee99a91` | pendiente −0,0115 [−0,0126, −0,0105] | −0,0120 [−0,0131, −0,0110] | OK (−0,237 / −0,240 ln err conocidas) | — | **Primaria aceptada, pero no competitiva**: vs ÓRBITA+LAT +1,513 / +1,439 ln err nuevas (≈4,5× más error) |
| CONSTELACIÓN v2 | `exp/constelacion-v2` @ `fdefd62` | +0,807 [+0,702, +0,908] | +0,798 [+0,688, +0,916] | OK formal pero vacío (acc16 = 0) | 0/16 | **Refutada**; colapso 4/16 y 5/16 |
| CONSTELACIÓN v3 | `exp/constelacion-v3` @ `e5718e8` | −0,089 [−0,152, −0,018] | −0,115 [−0,193, −0,030] | OK (+0,007 / +0,0005) | **0/16** | Primaria + no-daño OK; **E33 falla** |
| **CONSTELACIÓN v4** | `exp/constelacion-v4` @ `fd80320` | −0,076 [−0,141, −0,001] | **−0,107 [−0,181, −0,024]** | OK (0,000 [0, 0]; acc16 ≈ 0,877) | **16/16 y 16/16** | **Cumple los 3 criterios preregistrados** |
| CST vs ÓRBITA-eq (capacidad igualada) | `exp/cst-vs-orbita` @ `c1b1f54` | H1 −0,658 [−0,775, −0,542] | H1 −0,690 [−0,808, −0,573] | H3 +0,502 / +0,477 acc16 conocidas | H2 +0,356 / +0,336 | **CST gana** a ÓRBITA de igual tamaño (+LAT) en DEV y CONFIRM. Ver `docs/cst_vs_orbita/RESULTADOS.md` |
| v4 + mejoras M1/M2/M3 | `exp/constelacion-v4-mejoras` @ `ce63ac3` | M1 +0,0001; M3 −0,0013 | M1 +0,0001 [−0,0000, +0,0003] | M1 OK; **M3 falla** (−0,085 acc16 conocidas) | M1 OK | **Negativo**: M1 inerte (= control), M2 rechazada en meta-val, M3 STOP en DEV por no-daño. v4 reproduce 16/16 hashes. Sólo docs/artefactos en `main` |
| BIO-CONSTELACIÓN (híbrido v4 + neocórtex) | `exp/hibrido-constelacion-neocortex` @ `e9d7002` | hyb − v4 −0,262 [−0,422, −0,171] | −0,112 [−0,185, +0,017] (**no pasa**) | **DAÑO** (−0,81 acc16 conocidas) | **DAÑO** (E33 h16 −0,98) | **Negativo**: primaria no se confirma y destruye conocidas/E33. Sólo docs/artefactos en `main` |

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
**BQ10 está en `main`** (`orbita/src/constelacion4/atlas.rs`: brazos `BQ3/BQ10/BQ30`, `BQR*`, `CORE`, `BQ-M7R`; M8 opcional e inerte). ATLAS-FW y v4 siguen intactos (Paso 0 hash v4 `1985bdbad5eee41f` en 0xC420). Texto histórico previo:

CONSTELACIÓN v4 **está en `main`** (PR `integra/constelacion-v4-main`). El código de M1–M3 (`mejoras.rs`) y del híbrido (`bio.rs`, crate `bio_constelacion/`) **no** se integró: ambos son negativos y modifican `constelacion4/mod.rs`; quedan en sus ramas. Tras v4 se probaron mejoras (M1–M3) y un híbrido bio; ninguno supera a v4 sin daño, así que v4 sigue siendo la versión estable.

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
