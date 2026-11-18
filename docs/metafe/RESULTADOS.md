# Resultados Preregistrados — Familia METAFE (DEV y CONFIRM)

Rama: `exp/metafe-free-energy` · Preregistro congelado previo a DEV/CONFIRM: [`docs/metafe/PREREG.md`](PREREG.md) (`62e06ea`) · Configuración congelada: [`docs/metafe/frozen.json`](frozen.json) · Arquitectura: [`docs/metafe/ARQUITECTURA.md`](ARQUITECTURA.md) · Artefacto de análisis: [`artifacts/metafe/analysis.json`](../../artifacts/metafe/analysis.json).

---

## 1. Integridad del protocolo y veredicto global

- **Cero fuga de datos y cero consultas a TEST:** `test_queries == 0` en las **32/32 semillas** (16 DEV + 16 CONFIRM). Las familias retenidas `{8, 10, 11}` jamás entran en el entrenamiento metacognitivo ni en el prior de arranque en frío.
- **Verificación criptográfica exacta del conjunto de prueba:** `test_sha256` coincide bit a bit **16/16 en DEV** y **16/16 en CONFIRM** (32/32) entre los artefactos de `artifacts/metafe/{dev,confirm}` y los de `artifacts/bio_quantum_v10/{bq_dev,bq_confirm}`.
- **Cero re-ajustes post-congelación:** el código fuente en [`orbita/src/constelacion4/metafe.rs`](../../orbita/src/constelacion4/metafe.rs) se congeló en el commit `62e06ea` **antes** de ejecutar DEV y CONFIRM. `0 NaN` en todos los brazos y semillas; 32/32 retornos `rc = 0`.
- **Protocolo estadístico:** bootstrap pareado por semilla (10 000 réplicas, `seed = 12345`, IC95 percentil, umbral de relevancia $\delta = 0.05$), idéntico a [`scripts/analyze_bq10.py`](../../scripts/analyze_bq10.py).

> **Veredicto final (CONFIRM `0xB400–0xB40F`):** **TODAS las hipótesis preregistradas (H1, H2, H3, H4), puertas de no-daño, E33 canónico y autónomo (sin ÓRBITA), y puertas de metacognición y Energía Libre PASAN en DEV y CONFIRM.**
>
> La arquitectura insignia **`METAFE`** alcanza **`new_med_log = -5.3338 [-5.5700, -5.1044]`** y **`new_acc16 = 88.67%`** en CONFIRM (`-5.1667` / `88.02%` en DEV), superando a la ganadora previa **`BQ10`** (`-1.5081` / `14.84%`) en **`-3.8258 nats [-3.9899, -3.6467]`** (**45.9× menor error mediano a $h=16$**, **5.98× más precisión**, ganando en **32/32 semillas**), mejorando además las familias conocidas (`known_acc16 = 93.70%`, `+4.98 pp` sobre `BQ10`), resolviendo **E33 de forma autónoma `16/16`** sin `ÓRBITA-v1` y corriendo **49.2× más rápido** de extremo a extremo (`9.4 s` vs `462 s`).

---

## 2. Comparativa principal pareada (DEV y CONFIRM)

Todas las arquitecturas se evalúan sobre el **mismo** conjunto de prueba por semilla (`k = 4`, horizonte `h = 16`, 16 semillas por fase). Tiempos de referencia (`v4`, `CORE`, `ALL`, `BQ10`) indican coste del brazo y coste total extremo a extremo (`e2e`, incluyendo entrenamiento de `v4`); los tiempos de la familia `METAFE` son **totales extremo a extremo** (meta-entrenamiento + cold-start GP + evaluación completa).

### 2.1 Fase DEV (`0xA400–0xA40F`, $n = 16$)

| Arquitectura | `new_med_log` [IC95] | `new_acc16` | `known_acc16` | $\Delta$ vs `BQ10` (`new_med_log`) | $\Delta$ vs `ALL` (`new_med_log`) | E33 | E33 Autónomo (`h1`) | Tiempo/sem (s) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `ÓRBITA-v1` (ref.) | -0.7990 [-0.9008, -0.6937] | 0.0260 | 0.3540 | — | — | 16/16 | 16/16 | — |
| `CONSTELACIÓN v4` | -0.8748 [-1.0116, -0.7460] | 0.0690 | 0.8770 | — | — | 16/16 | 0/16 (`0.056`) | 4.5 (`384` e2e) |
| `CORE` | -1.3733 [-1.4962, -1.2617] | 0.1562 | 0.8970 | — | — | 16/16 | 0/16 (`0.056`) | 12.4 (`392` e2e) |
| `ATLAS-FW (ALL)` | -1.3856 [-1.5014, -1.2745] | 0.1549 | 0.8970 | — | — | 16/16 | 0/16 (`0.056`) | 41.6 (`422` e2e) |
| `BQ10` (ganadora `main`) | -1.5245 [-1.6494, -1.4050] | 0.1445 | 0.8877 | — | — | 16/16 | 0/16 (`0.056`) | 13.0 (`393` e2e) |
| `FE-ARD` | -1.6578 [-1.7900, -1.5363] | 0.2214 | 0.9238 | -0.1333 [-0.2566, -0.0085] | -0.2721 [-0.3904, -0.1635] | 16/16 | **16/16** (`1.000`) | **0.4** |
| `GP-Symp-FE` | -2.7909 [-2.9607, -2.6273] | 0.4766 | 0.9067 | -1.2665 [-1.4058, -1.1361] | -1.4053 [-1.5472, -1.2829] | 16/16 | **16/16** (`1.000`) | 3.8 |
| `GP-Cart-FE` | -3.0031 [-3.1722, -2.8500] | 0.4909 | 0.9204 | -1.4786 [-1.5877, -1.3711] | -1.6174 [-1.7307, -1.5030] | 16/16 | **16/16** (`1.000`) | 2.9 |
| `GP-Polar-FE` | -3.3737 [-3.5478, -3.2052] | 0.5664 | 0.9385 | -1.8493 [-1.9789, -1.7160] | -1.9881 [-2.1200, -1.8519] | 16/16 | **16/16** (`1.000`) | 3.4 |
| `METAFE-NoFE` *(abl.)* | -2.6667 [-2.8979, -2.4495] | 0.4492 | 0.9102 | -1.1423 [-1.3500, -0.9430] | -1.2811 [-1.4880, -1.0827] | 16/16 | **16/16** (`1.000`) | 5.8 |
| `METAFE-NoRoll` *(abl.)* | -5.0272 [-5.2916, -4.7761] | 0.8581 | 0.9316 | -3.5027 [-3.7183, -3.2946] | -3.6415 [-3.8310, -3.4504] | 16/16 | **16/16** (`1.000`) | 5.8 |
| **`METAFE` (Primaria)** | **-5.1667 [-5.4302, -4.9175]** | **0.8802** | **0.9395** | **-3.6422 [-3.8489, -3.4309]** | **-3.7810 [-3.9746, -3.5796]** | **16/16** | **16/16** (`1.000`) | **9.5** |

### 2.2 Fase CONFIRM (`0xB400–0xB40F`, $n = 16$ — fase decisoria)

| Arquitectura | `new_med_log` [IC95] | `new_acc16` | `known_acc16` | $\Delta$ vs `BQ10` (`new_med_log`) | $\Delta$ vs `ALL` (`new_med_log`) | E33 | E33 Autónomo (`h1`) | Tiempo/sem (s) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `ÓRBITA-v1` (ref.) | -0.7928 [-0.9119, -0.6826] | 0.0247 | 0.3384 | — | — | 16/16 | 16/16 | — |
| `CONSTELACIÓN v4` | -0.8997 [-0.9660, -0.8265] | 0.0664 | 0.8770 | — | — | 16/16 | 0/16 (`0.052`) | 4.5 (`454` e2e) |
| `CORE` | -1.3518 [-1.4830, -1.2185] | 0.1562 | 0.9028 | — | — | 16/16 | 0/16 (`0.052`) | 12.7 (`462` e2e) |
| `ATLAS-FW (ALL)` | -1.3518 [-1.4830, -1.2185] | 0.1562 | 0.9028 | — | — | 16/16 | 0/16 (`0.052`) | 41.0 (`490` e2e) |
| `BQ10` (ganadora `main`) | -1.5081 [-1.6382, -1.3865] | 0.1484 | 0.8872 | — | — | 16/16 | 0/16 (`0.052`) | 12.3 (`462` e2e) |
| `FE-ARD` | -1.6568 [-1.8051, -1.5018] | 0.2292 | 0.9185 | -0.1487 [-0.2825, -0.0158] | -0.3050 [-0.4113, -0.1949] | 16/16 | **16/16** (`1.000`) | **0.4** |
| `GP-Symp-FE` | -2.7395 [-2.8651, -2.6237] | 0.4583 | 0.9097 | -1.2315 [-1.3702, -1.1058] | -1.3877 [-1.5076, -1.2593] | 16/16 | **16/16** (`1.000`) | 3.7 |
| `GP-Cart-FE` | -2.8416 [-2.9622, -2.7186] | 0.4648 | 0.9219 | -1.3335 [-1.4815, -1.1940] | -1.4898 [-1.5828, -1.3907] | 16/16 | **16/16** (`1.000`) | 2.8 |
| `GP-Polar-FE` | -3.5732 [-3.8356, -3.3189] | 0.5807 | 0.9370 | -2.0652 [-2.2781, -1.8537] | -2.2214 [-2.4707, -1.9767] | 16/16 | **16/16** (`1.000`) | 3.3 |
| `METAFE-NoFE` *(abl.)* | -2.5971 [-2.7824, -2.4115] | 0.4375 | 0.9136 | -1.0890 [-1.3042, -0.8752] | -1.2453 [-1.4570, -1.0541] | 16/16 | **16/16** (`1.000`) | 5.7 |
| `METAFE-NoRoll` *(abl.)* | -5.2209 [-5.4605, -4.9826] | 0.8737 | 0.9341 | -3.7128 [-3.8938, -3.5204] | -3.8691 [-4.0672, -3.6697] | 16/16 | **16/16** (`1.000`) | 5.7 |
| **`METAFE` (Primaria)** | **-5.3338 [-5.5700, -5.1044]** | **0.8867** | **0.9370** | **-3.8258 [-3.9899, -3.6467]** | **-3.9821 [-4.1784, -3.7855]** | **16/16** | **16/16** (`1.000`) | **9.4** |

### 2.3 Contrastes pareados completos de `METAFE` contra todas las referencias

| Contraste | Fase | $\Delta$ `new_med_log` [IC95] | Factor de reducción de error ($e^{|\Delta|}$) | $\Delta$ `new_acc16` [IC95] | $\Delta$ `known_acc16` [IC95] | Semillas ganadas |
|---|---|---:|---:|---:|---:|---:|
| **`METAFE − BQ10`** | DEV | **-3.6422 [-3.8489, -3.4309]** | **38.2×** | +0.7357 [+0.7148, +0.7591] | +0.0518 [+0.0459, +0.0576] | **16/16** |
| **`METAFE − BQ10`** | CONFIRM | **-3.8258 [-3.9899, -3.6467]** | **45.9×** | +0.7383 [+0.6992, +0.7747] | +0.0498 [+0.0459, +0.0537] | **16/16** |
| **`METAFE − ALL`** | DEV | **-3.7810 [-3.9746, -3.5796]** | **43.9×** | +0.7253 [+0.6966, +0.7552] | +0.0425 [+0.0361, +0.0488] | **16/16** |
| **`METAFE − ALL`** | CONFIRM | **-3.9821 [-4.1784, -3.7855]** | **53.6×** | +0.7305 [+0.6888, +0.7721] | +0.0342 [+0.0288, +0.0400] | **16/16** |
| **`METAFE − v4`** | DEV | **-4.2918 [-4.4829, -4.0979]** | **73.1×** | +0.8112 [+0.7877, +0.8346] | +0.0625 [+0.0571, +0.0679] | **16/16** |
| **`METAFE − v4`** | CONFIRM | **-4.4341 [-4.6492, -4.2310]** | **84.3×** | +0.8203 [+0.7852, +0.8529] | +0.0601 [+0.0566, +0.0635] | **16/16** |
| **`METAFE − ÓRBITA-v1`** | DEV | **-4.3676 [-4.5758, -4.1620]** | **78.9×** | +0.8542 [+0.8294, +0.8802] | +0.5854 [+0.5439, +0.6270] | **16/16** |
| **`METAFE − ÓRBITA-v1`** | CONFIRM | **-4.5410 [-4.7399, -4.3627]** | **93.8×** | +0.8620 [+0.8320, +0.8893] | +0.5986 [+0.5698, +0.6270] | **16/16** |

> Incluso **`FE-ARD`** solo (reemplazo bayesiano de `BQ10` que tarda **`0.4 s`/semilla**, **~1 150× más rápido** extremo a extremo que `BQ10`) supera estadísticamente a `BQ10` en ambas fases (`DEV: -0.1333 [-0.2566, -0.0085]`, `CONFIRM: -0.1487 [-0.2825, -0.0158]`, subiendo `new_acc16` de `14.8%` a `22.9%` y `known_acc16` de `88.7%` a `91.9%`), demostrando que incluso sin GPs multi-sustrato, la **Energía Libre ARD bayesiana** supera a la regularización Tikhonov rígida `da=10` de `BQ10`.

---

## 3. Prueba causal mediante ablaciones (H3 y H4)

Para descartar que las ganancias se deban únicamente al banco de 11 sustratos geométricos, comparamos **`METAFE`** contra las ablaciones preregistradas exactamente sobre las mismas 11 hipótesis:

| Contraste causal | Fase | $\Delta$ `new_med_log` [IC95] | $\Delta$ `new_acc16` [IC95] | $\Delta$ `known_acc16` [IC95] | $\Delta$ `syn_ood_med_log` [IC95] | $\Delta$ `syn_ood_acc16` [IC95] |
|---|---|---:|---:|---:|---:|---:|
| **H3: `METAFE − METAFE-NoFE`** *(causalidad Energía Libre + BMA)* | DEV | **-2.4999 [-2.7017, -2.2990]** | **+0.4310 [+0.4023, +0.4622]** | +0.0293 [+0.0244, +0.0342] | **-3.3016 [-4.3858, -2.2190]** | +0.1484 [+0.1224, +0.1745] |
| **H3: `METAFE − METAFE-NoFE`** *(causalidad Energía Libre + BMA)* | CONFIRM | **-2.7368 [-2.9660, -2.5452]** | **+0.4492 [+0.4128, +0.4857]** | +0.0234 [+0.0176, +0.0293] | **-4.5122 [-5.5905, -3.4939]** | +0.1432 [+0.1198, +0.1654] |
| **H4: `METAFE − METAFE-NoRoll`** *(causalidad OOT + Refinamiento)* | DEV | **-0.1395 [-0.2665, -0.0229]** | **+0.0221 [+0.0104, +0.0339]** | +0.0078 [+0.0044, +0.0112] | **-0.4450 [-0.7518, -0.1870]** | +0.0391 [+0.0221, +0.0560] |
| **H4: `METAFE − METAFE-NoRoll`** *(causalidad OOT + Refinamiento)* | CONFIRM | **-0.1130 [-0.1927, -0.0430]** | **+0.0130 [+0.0039, +0.0234]** | +0.0029 [+0.0000, +0.0063] | **-0.5367 [-0.8587, -0.2651]** | +0.0299 [+0.0182, +0.0430] |

- **H3 confirmada con contundencia (32/32 semillas):** Quitar la Energía Libre y el BMA metacognitivo (`METAFE-NoFE`, manteniendo las mismas 11 hipótesis por replay MSE) desploma el rendimiento de `-5.3338 (88.67%)` a `-2.5971 (43.75%)`. La Energía Libre aporta causalmente **`-2.7368 nats` (15.4× menor error)** y **`+44.92 pp`** en familias nuevas, además de **`-4.5122 nats` (`+14.32 pp`)** en OOD procedural no visto (`syn_ood`).
- **H4 confirmada en DEV y CONFIRM:** Añadir validación precuencial fuera de tiempo (`OOT-1`/`OOT-2`) y refinamiento multi-paso aporta mejoras estadísticamente significativas en todas las métricas (`-0.1130 nats` y `+1.30 pp` en nuevas; `-0.5367 nats` y `+2.99 pp` en `syn_ood`), además de mejorar la calibración (`Brier` global `-0.0097` en CONFIRM).

---

## 4. Desglose por familia y descubrimiento emergente de simetrías

### 4.1 Error mediano `log(rel_err)` a $h=16$ por familia (`METAFE`)

| Familia | Tipo | DEV [IC95] | CONFIRM [IC95] |
|---|---|---:|---:|
| `fam_0` – `fam_5` | Conocidas afines/lineales | `-27.6310` (piso `1e-12`) | `-27.6310` (piso `1e-12`) |
| `fam_6` | Conocida suave | -27.6179 [-27.6310, -27.5918] | `-27.6310` (piso `1e-12`) |
| `fam_7` | Conocida no lineal ($SO(2)$ radial) | -3.6903 [-4.5874, -2.8564] | -3.3475 [-3.8878, -2.8043] |
| **`fam_8`** | **OOD nueva (retenida)** | **-5.1221 [-5.5277, -4.7371]** | **-5.2882 [-5.6998, -4.9059]** |
| **`fam_10`** | **OOD nueva (retenida)** | **-7.9883 [-8.5212, -7.4543]** | **-8.3896 [-9.0020, -7.7714]** |
| **`fam_11`** | **OOD nueva (retenida)** | **-4.3968 [-4.5869, -4.2130]** | **-4.1938 [-4.3689, -4.0145]** |
| **`syn_ood`** | **OOD procedural no visto (48 ep./sem)** | **-9.6513 [-13.5358, -6.4107]** (`66.28%`) | **-11.0521 [-13.5842, -8.8931]** (`71.35%`) |

### 4.2 Descubrimiento emergente de simetría vía Energía Libre (`hyp_selection` en `METAFE`)

Sin conocer las ecuaciones ni las etiquetas de las 3 familias OOD retenidas `{8, 10, 11}` (cada una representa exactamente 1/3 = `33.33%` de los episodios nuevos), la minimización de Energía Libre + sorpresa precuencial particiona espontáneamente las familias OOD en sus variedades geométricas intrínsecas:

- **En nuevas OOD `{8, 10, 11}` (CONFIRM):** `GpPolar` **`33.33%`** (exactamente 1/3), `GpSympFwd` **`32.81%`** (~1/3), y `GpCoMoving` + `GpCartRbf` = `17.45% + 15.49%` = **`32.94%`** (~1/3). Patrón idéntico en DEV (`33.07%` / `32.55%` / `33.73%`).
- **En conocidas `0..7` (CONFIRM):** `GpCoSympRev` `70.46%` (atajo afín exacto), `GpPolar` `17.92%` y `GpEquivCart` `10.16%` (resolviendo `fam_7` sin singularidad en $r=0$).
- **En `syn_ood` procedural (CONFIRM):** reclutamiento diverso de las 10 hipótesis activas (`GpPolar 36.85%`, `GpCoMoving 12.37%`, `GpSympRev 12.37%`, `GpCoSympRev 11.07%`, `GpEquivCart 9.11%`, `GpCartRbf 7.94%`, etc.).

---

## 5. Metacognición ("Yo-FE v2") y reducción de Energía Libre

| Subconjunto (`METAFE`) | Fase | AUROC [IC95] | Brier [IC95] | ECE [IC95] | AURC [IC95] | Pearson $r$ [IC95] | Spearman $\rho$ [IC95] | $\Delta\mathcal{F}$ (nats) | % $\Delta\mathcal{F}>0$ |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **Global (`overall`)** | DEV | **0.9728** [0.9663, 0.9791] | **0.0484** [0.0404, 0.0565] | 0.0627 [0.0517, 0.0731] | **0.0055** [0.0046, 0.0065] | **0.9127** [0.9051, 0.9202] | 0.8298 [0.8198, 0.8377] | **+31.53** | **100.00%** |
| **Global (`overall`)** | CONFIRM | **0.9713** [0.9647, 0.9779] | **0.0473** [0.0393, 0.0562] | 0.0595 [0.0490, 0.0707] | **0.0055** [0.0049, 0.0060] | **0.9183** [0.9112, 0.9255] | 0.8283 [0.8215, 0.8351] | **+31.55** | **100.00%** |
| Conocidas (`known`) | DEV | 0.9999 | 0.0046 | 0.0171 | 0.0022 | 0.9583 | 0.4876 | +35.58 | 100.00% |
| Conocidas (`known`) | CONFIRM | 0.9994 | 0.0072 | 0.0196 | 0.0023 | 0.9580 | 0.5030 | +35.57 | 100.00% |
| **Nuevas OOD (`new`)** | DEV | **0.8394** [0.7991, 0.8805] | 0.1650 [0.1343, 0.1962] | 0.2202 [0.1764, 0.2625] | **0.0318** [0.0229, 0.0409] | **0.8166** [0.8004, 0.8310] | **0.8270** [0.8011, 0.8526] | **+20.73** | **100.00%** |
| **Nuevas OOD (`new`)** | CONFIRM | **0.8322** [0.7865, 0.8752] | 0.1541 [0.1253, 0.1854] | 0.2090 [0.1693, 0.2506] | **0.0286** [0.0224, 0.0350] | **0.8523** [0.8362, 0.8692] | **0.8605** [0.8398, 0.8807] | **+20.83** | **100.00%** |
| **Procedural (`syn_ood`)** | DEV | **0.9851** [0.9753, 0.9930] | **0.0474** [0.0384, 0.0571] | 0.0712 [0.0615, 0.0810] | 0.0813 [0.0572, 0.1096] | **0.9557** [0.9473, 0.9637] | **0.8945** [0.8774, 0.9101] | **+20.58** | **100.00%** |
| **Procedural (`syn_ood`)** | CONFIRM | **0.9853** [0.9782, 0.9917] | **0.0446** [0.0339, 0.0557] | 0.0694 [0.0601, 0.0792] | 0.0552 [0.0448, 0.0647] | **0.9586** [0.9526, 0.9635] | **0.8840** [0.8665, 0.8991] | **+21.94** | **100.00%** |

### Lectura honesta y matices metacognitivos
1. **Reducción universal de Energía Libre:** En el **100.00%** de los episodios evaluados en DEV y CONFIRM (conocidas, nuevas OOD y `syn_ood`), la inferencia reduce estrictamente la Energía Libre respecto al modelo nulo ($\Delta\mathcal{F} = +31.55\text{ nats}$ global; $+20.83\text{ nats}$ en OOD retenidas), evitando por completo el colapso variacional que refutó a `TFE` (`known_acc16 = 4.24%`, `AUROC = 0.6616`).
2. **Riesgo selectivo y auto-predicción de error:** El auto-modelo predice con alta fidelidad la magnitud del error futuro a $h=16$ incluso en familias OOD jamás vistas (`Pearson r = 0.8523`, `Spearman ρ = 0.8605`) y alcanza un riesgo selectivo `AURC = 0.0286` en OOD (8.8× menor que `METAFE-NoFE`, cuya `AURC` en OOD es `0.2521`).
3. **Matiz transparente sobre calibración bajo cambio de distribución:** En `new` OOD, el `Brier` (`0.1541`) y el `ECE` (`0.2090`) son mayores que en `overall` (`0.0473` / `0.0595`) y `syn_ood` (`0.0446` / `0.0694`). Esto refleja un sesgo de umbral esperado bajo desplazamiento de distribución porque el calibrador Platt de Etapa B se entrenó **estrictamente sin acceso** a `{8, 10, 11}`, mientras que el ordenamiento de incertidumbre (`AUROC = 0.8322`, `ρ = 0.8605`) permanece muy alto. Asimismo, el `AUROC` nominal en `new` de `METAFE-NoFE` resulta ligeramente superior al de `METAFE` (`Δ = -0.0899`) porque `METAFE` acierta el `88.7%` de episodios OOD (dejando solo casos limítrofes en el ~11% restante), mientras que `METAFE-NoFE` falla el `56.2%` con errores grandes más fáciles de separar; el riesgo selectivo `AURC` refleja correctamente la ventaja real (`-0.2235` a favor de `METAFE`).

---

## 6. Autonomía en E33 (sin ÓRBITA-v1) y eficiencia computacional

- **E33 Autónomo (`16/16` DEV y `16/16` CONFIRM):** Sin la cabeza de `ÓRBITA-v1`, `v4`, `CORE`, `ALL` y `BQ10` fallan completamente E33 (`0/16`, `h1 = 5.18%` en CONFIRM), exactamente igual que el control sin prior `e33_raw` de `METAFE` (`5.18%`, `0/16`). Al activar el prior autónomo GP RBF por Energía Libre entrenado exclusivamente sobre `ds.train` (0 consultas TEST), **todos los brazos METAFE alcanzan `16/16` en DEV y `16/16` en CONFIRM con `acc(h=1) = 1.000` y `acc(h=16) = 1.000`**.
- **Velocidad:** En CONFIRM, `METAFE` requiere **`9.39 s`/semilla** (`4.28 s` meta-entrenamiento + `5.11 s` evaluación), lo que supone:
  - **49.2× más rápido** extremo a extremo que `BQ10` (`462 s`: `449 s` v4 + `12.3 s` cabeza) y **1.32× más rápido** que la evaluación sola de `BQ10`.
  - **52.2× más rápido** extremo a extremo que `ATLAS-FW (ALL)` (`490 s`) y **4.36× más rápido** que la evaluación sola de `ALL` (`41.0 s`).
  - `FE-ARD` logra superar a `BQ10` en **`0.4 s`/semilla** (**~1 150× más rápido** extremo a extremo).

---

## 7. Reproducción exacta

```bash
cargo build --release --manifest-path orbita/Cargo.toml
cargo test --release --manifest-path orbita/Cargo.toml            # 14/14 PASS
./orbita/target/release/orbita metafe-paso0 --seed 0xC4B0        # paso0_ok: true
JOBS=2 orbita/scripts/metafe_phase.sh tune                       # 0xC4B0–0xC4B3
JOBS=2 orbita/scripts/metafe_phase.sh dev                        # 0xA400–0xA40F
JOBS=2 orbita/scripts/metafe_phase.sh confirm                    # 0xB400–0xB40F
python3 scripts/analyze_metafe.py > artifacts/metafe/analysis.json
python3 scripts/analyze_metafe.py --md
```
