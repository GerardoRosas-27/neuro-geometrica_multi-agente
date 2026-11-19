# METAFE-3S — Resultados Preregistrados (DEV y CONFIRM, 32 Semillas Pareadas)

Rama: `exp/metafe-3s-neuromod` · Commit de congelación preregistrada: `6ed6518` · Implementación: [`orbita/src/constelacion4/metafe3s.rs`](../../orbita/src/constelacion4/metafe3s.rs) · Preregistro: [`docs/metafe3s/PREREG.md`](PREREG.md) · Arquitectura: [`docs/metafe3s/ARQUITECTURA.md`](ARQUITECTURA.md) · JSON completo: [`artifacts/metafe3s/analysis.json`](../../artifacts/metafe3s/analysis.json).

---

## 1. Resumen Ejecutivo y Auditoría de Integridad

- **Integridad y paridad bit a bit:** `test_sha256` idéntico **`32/32`** (`16/16` DEV y `16/16` CONFIRM) frente a `artifacts/bio_quantum_v10/` y `artifacts/metafe/`; paridad de `METAFE-v2` **`max_diff = 0.0`** (cero alteración a la base puntera previa); **`0 NaN`**, **`test_queries = 0`** en las 32 semillas; `19/19` tests unitarios PASS.
- **Superación masiva frente a los baselines previos del repo (`H1` — PASA DEV y CONFIRM):** en `CONFIRM`, **`METAFE-3S`** logra `new_med_log = -5.3665 [-5.6213, -5.1186]` (`new_acc16 = 87.76%`, `known_acc16 = 93.75%`), superando a `BQ10` (`-1.5081`, `14.84%`) por **`Δ = -3.8585 [-4.0411, -3.6687]`** (**`47.4×` menor error mediano**), a `ATLAS-FW (ALL)` por **`-4.0147`**, a `v4` por **`-4.4668`** y a `ÓRBITA` por **`-4.5737`**, con **`E33 = 16/16` canónico y `16/16` autónomo sin ÓRBITA (`h1 = 100%`)**, y siendo **`1.30×` más rápida que `METAFE-v2`** (`18.9 s` vs `24.5 s`/semilla).
- **Ruptura decisiva y causal en los 3 nuevos regímenes biológicos de tres escalas (`H3`, `H4`, `H6` — PASA DEV y CONFIRM, `32/32` semillas):**
  1. **Cambio abrupto de ley online (`RS-A`, Módulos 2+3):** en `CONFIRM`, `METAFE-3S` alcanza **`-6.6761`** (`known_acc = 83.1%`, `ood_acc = 59.1%`, `reset = 89.2%`) frente a `METAFE-v2 = -0.2798` (`ood_acc = 0.0%`, `reset = 0%`) → **`Δ = -6.3963 [-8.0440, -5.1315]`** (**`600×` menor error mediano**, gana **`32/32`** semillas DEV+CONFIRM). Desactivar el neuromodulador (`METAFE-3S-NoNM`) colapsa el desempeño de vuelta a `-0.3001` (`Δ = -6.3761`, **`32/32`** semillas).
  2. **Perturbación online con $W_0$ estrictamente congelado (`RS-B`, Módulos 1+3+4):** en `CONFIRM`, `METAFE-3S` logra **`-2.1474`** (`23.7%` acc16) frente a `METAFE-v2 = +0.4702` (`0.0%`) → **`Δ = -2.6176 [-2.7364, -2.5167]`** (**`13.7×` menor error mediano**, gana **`32/32`** semillas). Tanto `NoFast` (`+0.5052`, `0%`, `Δ = -2.6526`) como `NoNM` (`+0.3035`, `0%`, `Δ = -2.4509`) colapsan por completo en **`32/32`** semillas, demostrando necesidad causal conjunta de los pesos rápidos $F_t$, trazas de 3 factores $E_t$ y `MetaNet`.
  3. **Consolidación continua offline con EWC (`T1 → T2 → T3`, Módulo 5):** `EWC` ($\mu=25$) reduce el olvido catastrófico de T1 frente a `NoEWC` ($\mu=0$) en **`32/32` semillas (`100%`)**: `Δ = -0.0418 [-0.0551, -0.0319]` (`-42.1%` olvido) en DEV y **`Δ = -0.0423 [-0.0516, -0.0337]`** (`-31.9%` olvido) en CONFIRM, elevando además la exactitud media T1–T3 (`+4.43 pp` DEV, `+0.87 pp` CONFIRM).
- **Lectura honesta cabeza a cabeza frente a `METAFE-v2` en régimen estacionario (`H2` y `H5`):**
  - En el benchmark canónico estacionario (`new_med_log`), `METAFE-3S` establece una nueva mejor mediana en `CONFIRM` (`-5.3665` vs `-5.3338`, `Δ = -0.0327 [-0.1222, +0.0552]`, ganando `10/16` semillas en DEV y `10/16` en CONFIRM = `20/32`), pero **los IC95 cruzan 0** (`DEV Δ = +0.0302 [-0.0360, +0.1050]`), por lo que **`H2a` no alcanza superación estricta $\delta = 0.05$ en régimen estacionario**; además, en `CONFIRM` `new_acc16` presenta un leve compromiso de `-0.91 pp` (`87.76%` vs `88.67%`, empate exacto `88.02%` en DEV). En conocidas, **cero daño (`H2b` PASA)**: `93.95%` DEV y `93.75%` vs `93.70%` CONFIRM.
  - En **metacognición OOD ("Yo-FE 3S")**, **`METAFE-3S` supera significativamente a `METAFE-v2` en `CONFIRM new_AUROC` (`0.8560` vs `0.8322`, `Δ = +0.0238 [+0.0037, +0.0472]`)** y reduce `new_Brier` puntualmente en ambas fases (`0.1559` vs `0.1650` DEV; `0.1485` vs `0.1541` CONFIRM), aunque no cruza significancia simultánea en DEV+CONFIRM (`H2c`/`H2d` no pasan ambos). Todas las puertas de metacognición y reducción de Energía Libre (`100%`, $\Delta\mathcal{F} = +31.59$ nats) **PASAN** en DEV y CONFIRM.
  - La ablación del refutador inverso (`H5`: `METAFE-3S − METAFE-3S-NoRefute`) es **significativa en DEV** tanto en `new_med_log` (`Δ = -0.0323 [-0.0737, -0.0026]`) como en `new_Brier` (`Δ = -0.0115 [-0.0212, -0.0033]`) y `new_ECE` (`Δ = -0.0180 [-0.0320, -0.0049]`), y mantiene dirección favorable en CONFIRM (`-0.0110`, `10/16` semillas; `new_Brier -0.0021`), pero con IC95 cruzando 0 en CONFIRM.

---

## 2. Benchmark Canónico, OOD Sintético y Autonomía E33

Bootstrap pareado por semilla (`10 000` réplicas, seed `12345`, IC95 percentil, $\delta = 0.05$).

### DEV (`0xA400–0xA40F`, $n=16$)

| Arquitectura | `new_med_log` [IC95] | `new_acc16` | `known_acc16` | `syn_ood_med` [IC95] | $\Delta$ vs `BQ10` [IC95] | E33 | E33 Autónomo | `wall_s` |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `ORBITA` | `-0.7990 [-0.9008, -0.6937]` | `0.0260` | `0.3540` | — | — | `16/16` | — | — |
| `v4` | `-0.8748 [-1.0116, -0.7460]` | `0.0690` | `0.8770` | — | — | `16/16` | — | `4.5` |
| `CORE` | `-1.3733 [-1.4962, -1.2617]` | `0.1562` | `0.8970` | — | — | `16/16` | — | `13.4` |
| `ALL (ATLAS-FW)` | `-1.3856 [-1.5014, -1.2745]` | `0.1549` | `0.8970` | — | — | `16/16` | — | `41.6` |
| `BQ10` | `-1.5245 [-1.6494, -1.4050]` | `0.1445` | `0.8877` | — | — | `16/16` | — | `13.0` |
| `METAFE-v2` | `-5.1667 [-5.4302, -4.9175]` | `0.8802` | `0.9395` | `-9.6513 [-13.5249, -6.3428]` | `-3.6422 [-3.8489, -3.4309]` | `16/16` | `16/16 (1.000)` | `24.7` |
| **`METAFE-3S`** | **`-5.1364 [-5.3822, -4.9086]`** | **`0.8802`** | **`0.9395`** | **`-9.7555 [-13.6213, -6.5453]`** | **`-3.6120 [-3.8002, -3.4188]`** | **`16/16`** | **`16/16 (1.000)`** | **`19.8`** |
| `METAFE-3S-NoFast` | `-5.1351 [-5.3802, -4.9091]` | `0.8802` | `0.9395` | `-9.7537 [-13.6225, -6.5324]` | `-3.6106 [-3.7981, -3.4181]` | `16/16` | `16/16 (1.000)` | `19.5` |
| `METAFE-3S-NoRefute` | `-5.1041 [-5.3454, -4.8782]` | `0.8763` | `0.9390` | `-9.7334 [-13.6164, -6.5046]` | `-3.5797 [-3.7650, -3.3869]` | `16/16` | `16/16 (1.000)` | `13.3` |
| `METAFE-3S-NoNM` | `-5.1494 [-5.3976, -4.9201]` | `0.8789` | `0.9395` | `-9.7678 [-13.6364, -6.5486]` | `-3.6250 [-3.8137, -3.4296]` | `16/16` | `16/16 (1.000)` | `37.9` |

### CONFIRM (`0xB400–0xB40F`, $n=16$)

| Arquitectura | `new_med_log` [IC95] | `new_acc16` | `known_acc16` | `syn_ood_med` [IC95] | $\Delta$ vs `BQ10` [IC95] | E33 | E33 Autónomo | `wall_s` |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `ORBITA` | `-0.7928 [-0.9119, -0.6826]` | `0.0247` | `0.3384` | — | — | `16/16` | — | — |
| `v4` | `-0.8997 [-0.9660, -0.8265]` | `0.0664` | `0.8770` | — | — | `16/16` | — | `4.5` |
| `CORE` | `-1.3518 [-1.4830, -1.2185]` | `0.1562` | `0.9028` | — | — | `16/16` | — | `12.7` |
| `ALL (ATLAS-FW)` | `-1.3518 [-1.4830, -1.2185]` | `0.1562` | `0.9028` | — | — | `16/16` | — | `41.0` |
| `BQ10` | `-1.5081 [-1.6382, -1.3865]` | `0.1484` | `0.8872` | — | — | `16/16` | — | `12.3` |
| `METAFE-v2` | `-5.3338 [-5.5700, -5.1044]` | `0.8867` | `0.9370` | `-11.0521 [-13.5842, -8.8931]` | `-3.8258 [-3.9899, -3.6467]` | `16/16` | `16/16 (1.000)` | `24.5` |
| **`METAFE-3S`** | **`-5.3665 [-5.6213, -5.1186]`** | **`0.8776`** | **`0.9375`** | **`-11.0145 [-13.5602, -8.8651]`** | **`-3.8585 [-4.0411, -3.6687]`** | **`16/16`** | **`16/16 (1.000)`** | **`18.9`** |
| `METAFE-3S-NoFast` | `-5.3616 [-5.6168, -5.1153]` | `0.8776` | `0.9380` | `-11.0174 [-13.5625, -8.8671]` | `-3.8535 [-4.0324, -3.6667]` | `16/16` | `16/16 (1.000)` | `18.6` |
| `METAFE-3S-NoRefute` | `-5.3556 [-5.6154, -5.1077]` | `0.8776` | `0.9375` | `-11.0413 [-13.5790, -8.8998]` | `-3.8475 [-4.0288, -3.6574]` | `16/16` | `16/16 (1.000)` | `12.6` |
| `METAFE-3S-NoNM` | `-5.3509 [-5.6049, -5.1065]` | `0.8776` | `0.9380` | `-11.0231 [-13.5673, -8.8782]` | `-3.8428 [-4.0148, -3.6607]` | `16/16` | `16/16 (1.000)` | `37.6` |

---

## 3. Regímenes No Estacionarios Online (`RS-A` y `RS-B`) y Consolidación Continua (`EWC`)

### Cambio de Régimen Abrupto (`RS-A`) y Perturbación con $W_0$ Congelado (`RS-B`)

| Fase / Brazo | `RS-A all_med` [IC95] | `RS-A known_acc` | `RS-A ood_acc` | `RS-A reset` | `RS-B frozen W_0 med` [IC95] | `RS-B acc16` |
|---|---:|---:|---:|---:|---:|---:|
| **DEV** `METAFE-v2` | `-0.3244 [-0.3776, -0.2775]` | `0.094` | `0.003` | `0.000` | `+0.4938 [+0.4444, +0.5458]` | `0.0000` |
| **DEV `METAFE-3S`** | **`-5.7955 [-6.6050, -4.9928]`** | **`0.818`** | **`0.531`** | **`0.857`** | **`-2.0007 [-2.1201, -1.8812]`** | **`0.2161`** |
| **DEV** `METAFE-3S-NoFast` | `-5.7958 [-6.6056, -4.9932]` | `0.820` | `0.531` | `0.857` | `+0.4548 [+0.4297, +0.4820]` | `0.0000` |
| **DEV** `METAFE-3S-NoRefute` | `-5.8020 [-6.6228, -5.0151]` | `0.823` | `0.529` | `0.857` | `-1.9978 [-2.1174, -1.8780]` | `0.2161` |
| **DEV** `METAFE-3S-NoNM` | `-0.3141 [-0.3640, -0.2680]` | `0.094` | `0.003` | `0.000` | `+0.2094 [+0.1607, +0.2637]` | `0.0000` |
| **CONFIRM** `METAFE-v2` | `-0.2798 [-0.3390, -0.2239]` | `0.115` | `0.000` | `0.000` | `+0.4702 [+0.4271, +0.5163]` | `0.0000` |
| **CONFIRM `METAFE-3S`** | **`-6.6761 [-8.2923, -5.4439]`** | **`0.831`** | **`0.591`** | **`0.892`** | **`-2.1474 [-2.2580, -2.0435]`** | **`0.2370`** |
| **CONFIRM** `METAFE-3S-NoFast` | `-6.6816 [-8.3096, -5.4439]` | `0.831` | `0.591` | `0.892` | `+0.5052 [+0.4752, +0.5358]` | `0.0000` |
| **CONFIRM** `METAFE-3S-NoRefute` | `-6.6505 [-8.2716, -5.4314]` | `0.831` | `0.583` | `0.892` | `-2.1482 [-2.2582, -2.0453]` | `0.2396` |
| **CONFIRM** `METAFE-3S-NoNM` | `-0.3001 [-0.3585, -0.2467]` | `0.115` | `0.000` | `0.000` | `+0.3035 [+0.2554, +0.3454]` | `0.0000` |

### Consolidación Continua Offline (`EWC` $\mu=25$ vs `NoEWC` $\mu=0$, Módulo 5)

| Fase | Olvido T1 `EWC` (CE) | Olvido T1 `NoEWC` (CE) | $\Delta$ Olvido (`EWC − NoEWC`) [IC95] | Semillas ganadas | $\Delta$ Exactitud media T1–T3 [IC95] |
|---|---:|---:|---:|---:|---:|
| **DEV** | `0.0576 [0.0380, 0.0787]` | `0.0994 [0.0725, 0.1308]` | **`-0.0418 [-0.0551, -0.0319]`** (`-42.1%`) | **`16/16`** | `+0.0443 [-0.0035, +0.0938]` |
| **CONFIRM** | `0.0905 [0.0666, 0.1173]` | `0.1328 [0.1017, 0.1679]` | **`-0.0423 [-0.0516, -0.0337]`** (`-31.9%`) | **`16/16`** | `+0.0087 [-0.0052, +0.0260]` |

---

## 4. Contrastes Pareados y Ablaciones Causales

### DEV (`0xA400–0xA40F`)

| Contraste | $\Delta$ `new_med_log` | $\Delta$ `new_acc16` | $\Delta$ `new_AUROC` | $\Delta$ `new_Brier` | $\Delta$ `RS-A all` | $\Delta$ `RS-B med` | Victorias (`new`/`RS-A`/`RS-B`) |
|---|---|---|---|---|---|---|---|
| `METAFE-3S − METAFE-v2` | `+0.0302 [-0.0360, +0.1050]` | `+0.0000 [-0.0065, +0.0065]` | `-0.0052 [-0.0199, +0.0082]` | `-0.0091 [-0.0226, +0.0046]` | **`-5.4712 [-6.2795, -4.6676]`** | **`-2.4945 [-2.6282, -2.3537]`** | `10 / 16 / 16` |
| `METAFE-3S − NoFast` | `-0.0014 [-0.0081, +0.0050]` | `+0.0000 [-0.0052, +0.0052]` | `+0.0011 [-0.0077, +0.0112]` | `-0.0001 [-0.0059, +0.0072]` | `+0.0002 [-0.0005, +0.0012]` | **`-2.4555 [-2.5719, -2.3360]`** | `2 / 1 / 16` |
| `METAFE-3S − NoRefute` | **`-0.0323 [-0.0737, -0.0026]`** | `+0.0039 [-0.0039, +0.0130]` | `+0.0003 [-0.0138, +0.0133]` | **`-0.0115 [-0.0212, -0.0033]`** | `+0.0064 [-0.0884, +0.0876]` | `-0.0029 [-0.0120, +0.0028]` | `10 / 2 / 4` |
| `METAFE-3S − NoNM` | `+0.0130 [-0.0011, +0.0329]` | `+0.0013 [-0.0026, +0.0052]` | `+0.0001 [-0.0078, +0.0093]` | `+0.0078 [+0.0021, +0.0147]` | **`-5.4814 [-6.2976, -4.6653]`** | **`-2.2101 [-2.3145, -2.1047]`** | `5 / 16 / 16` |

### CONFIRM (`0xB400–0xB40F`)

| Contraste | $\Delta$ `new_med_log` | $\Delta$ `new_acc16` | $\Delta$ `new_AUROC` | $\Delta$ `new_Brier` | $\Delta$ `RS-A all` | $\Delta$ `RS-B med` | Victorias (`new`/`RS-A`/`RS-B`) |
|---|---|---|---|---|---|---|---|
| `METAFE-3S − METAFE-v2` | `-0.0327 [-0.1222, +0.0552]` | `-0.0091 [-0.0169, -0.0026]` | **`+0.0238 [+0.0037, +0.0472]`** | `-0.0056 [-0.0139, +0.0019]` | **`-6.3963 [-8.0440, -5.1315]`** | **`-2.6176 [-2.7364, -2.5167]`** | `10 / 16 / 16` |
| `METAFE-3S − NoFast` | `-0.0050 [-0.0149, +0.0000]` | `+0.0000 [-0.0039, +0.0039]` | `+0.0000 [-0.0044, +0.0047]` | `-0.0030 [-0.0107, +0.0018]` | `+0.0055 [-0.0003, +0.0165]` | **`-2.6526 [-2.7576, -2.5566]`** | `2 / 2 / 16` |
| `METAFE-3S − NoRefute` | `-0.0110 [-0.0494, +0.0283]` | `+0.0000 [-0.0065, +0.0065]` | `+0.0035 [-0.0212, +0.0284]` | `-0.0021 [-0.0148, +0.0110]` | `-0.0256 [-0.0973, +0.0164]` | `+0.0009 [-0.0018, +0.0037]` | `10 / 4 / 3` |
| `METAFE-3S − NoNM` | `-0.0156 [-0.0388, +0.0004]` | `+0.0000 [+0.0000, +0.0000]` | `-0.0025 [-0.0071, +0.0014]` | `+0.0018 [-0.0033, +0.0081]` | **`-6.3761 [-8.0219, -5.1130]`** | **`-2.4509 [-2.5518, -2.3639]`** | `4 / 16 / 16` |

---

## 5. Telemetría Metacognitiva ("Yo-FE 3S") y Reducción de Energía Libre

| Fase / Brazo | `ov_AUROC` | `ov_Brier` | `ov_ECE` | `ov_r` | `new_AUROC` | `new_Brier` | `new_ECE` | `new_AURC` | `new_r` | `new_ρ` | `syn_AUROC` | $\Delta\mathcal{F}$ (nats / frac) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **DEV** `METAFE-v2` | `0.9728` | `0.0484` | `0.0627` | `0.9127` | `0.8394` | `0.1650` | `0.2202` | `0.0318` | `0.8166` | `0.8270` | `0.9851` | `+31.53` (`100%`) |
| **DEV `METAFE-3S`** | `0.9727` | **`0.0465`** | `0.0647` | **`0.9167`** | `0.8343` | **`0.1559`** | **`0.2179`** | `0.0327` | `0.8130` | `0.8208` | `0.9851` | **`+31.55`** (`100%`) |
| **CONFIRM** `METAFE-v2` | `0.9713` | `0.0473` | `0.0595` | `0.9183` | `0.8322` | `0.1541` | `0.2090` | `0.0286` | `0.8523` | `0.8605` | `0.9853` | `+31.55` (`100%`) |
| **CONFIRM `METAFE-3S`** | **`0.9736`** | **`0.0461`** | `0.0662` | **`0.9215`** | **`0.8560`** | **`0.1485`** | `0.2190` | `0.0298` | `0.8514` | `0.8573` | **`0.9864`** | **`+31.59`** (`100%`) |

---

## 6. Veredictos Preregistrados

- **`H1` (`METAFE-3S` vs `BQ10`, `ALL`, `CORE`, `v4`, `ORBITA`):** **PASA (DEV y CONFIRM)** en los 5 baselines.
- **`H2a` (`METAFE-3S − METAFE-v2` canónico `new_med_log`):** **SIN SUPERACIÓN ESTRICTA $\delta = 0.05$** (`DEV +0.0302 [-0.0360, +0.1050]`, `CONFIRM -0.0327 [-0.1222, +0.0552]`; paridad estadística en estacionario, mejor mediana absoluta en CONFIRM `-5.3665`).
- **`H2b` (No-daño en conocidas `known_acc16`):** **PASA (DEV y CONFIRM, IC inf > -0.05)**.
- **`H2c / H2d` (`new_Brier` / `new_AUROC` vs `METAFE-v2`):** **NO PASA AMBOS** (`new_Brier` mejora puntualmente en ambos sin excluir 0; `new_AUROC` **PASA en CONFIRM `+0.0238 [+0.0037, +0.0472]`**, nulo en DEV).
- **`H3` (`RS-A` vs `METAFE-v2` y ablación causal `NoNM`):** **PASA (DEV y CONFIRM, `32/32` semillas)**.
- **`H4` (`RS-B` vs `METAFE-v2` y ablaciones causales `NoFast` y `NoNM`):** **PASA (DEV y CONFIRM, `32/32` semillas)**.
- **`H5` (Ablación causal `NoRefute`):** **PASA EN DEV (`new_med_log -0.0323`, `new_Brier -0.0115`)**, dirección consistente pero IC95 cruzando 0 en CONFIRM (`-0.0110`, `new_Brier -0.0021`).
- **`H6` (`EWC` reduce olvido catastrófico vs `NoEWC`):** **PASA (DEV y CONFIRM, `32/32` semillas)**.
- **Puertas de autonomía, seguridad, paridad y metacognición (`E33`, `E33 Auto`, `0 NaN`, `0 TEST`, `32/32 SHA-256`, `v2 bit-exact`, `FE=100%`, `AUROC/Brier/r`):** **TODAS PASAN (DEV y CONFIRM)**.
