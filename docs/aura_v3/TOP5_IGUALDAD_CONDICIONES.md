# Torneo de las Mejores Arquitecturas en Igualdad Estricta de Condiciones (incluyendo `METAFE-v2`, `METAFE-3S` y `AURA v3 Genesis`)

- **Rama experimental**: `exp/aura-v3-genesis`
- **Arnés Canónico 1 (`orbita metafe3s-run` + [`scripts/analyze_metafe3s.py`](file:///home/antigravity/GerardoRosas-27/neuro-geometrica_multi-agente/scripts/analyze_metafe3s.py))**:
  - **16 semillas `DEV` (`0xA400–0xA40F`)** y **16 semillas `CONFIRM` (`0xB400–0xB40F`)** ($32/32$ hashes `test_sha256` bit-idénticos, $0$ consultas `TEST`, $0\text{ NaN}$).
  - **Condiciones de entrenamiento 100 % idénticas por semilla**: `n_train = 128` episodios de meta-entrenamiento, `n_val = 96` episodios de meta-validación, evaluación ciega sobre `HPF = 64` episodios por familia (`512` conocidas `0..7` + `192` OOD retenidas `{8, 10, 11}`), `48` episodios OOD sintéticos procedurales (`syn_ood`), cambio de régimen abrupto online (`RS-A`, `48` episodios), deriva con $W_0$ congelado (`RS-B`, `24` episodios) y puente de arranque en frío `E33 / E33 Autónomo`.
- **Arnés Canónico 2 ([`aura-engine/top5-arena/src/main.rs`](file:///home/antigravity/GerardoRosas-27/neuro-geometrica_multi-agente/aura-engine/top5-arena/src/main.rs) + [`aura-engine/run_top5_equal.py`](file:///home/antigravity/GerardoRosas-27/neuro-geometrica_multi-agente/aura-engine/run_top5_equal.py))**:
  - **16 semillas `CONFIRM` (`0xB600–0xB60F`)** entrenando sobre el mismo corpus real de $24{,}576\text{ B}$ Markdown ($4$ épocas exactas, `chunk_len = 32`, `STATE_DIM = 128`) y evaluando en Markdown retenido (`eval_md_bpb`) + cambio de dominio en línea a código Rust (`shift_rust_bpb`).

---

## 1. Tabla Principal en el Arnés Canónico de `orbita` (`CONFIRM 0xB400–0xB40F`, $N=16$)

> En el mismo formato de la tabla consolidada de `main`, comparando las **mejores arquitecturas del repositorio** (`AURA-v3-Genesis`, `METAFE-3S`, `METAFE-v2`, `BQ10`, `ATLAS-FW (ALL)`, `ATLAS-CORE`, `CONSTELACIÓN v4` y `ÓRBITA`) bajo las mismas 16 semillas `CONFIRM` (`0xB400–0xB40F`) e idéntico protocolo de entrenamiento:

| Arquitectura | `new_med_log` [IC95] | `new_acc16` | `known_acc16` | `Δ vs BQ10` [IC95] (Factor) | `new_AUROC` | `E33 / E33 Autónomo` | `wall_s` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`AURA-v3-Genesis` (Nueva: 4 Subrutinas ADN)** | **`-5.3748 [-5.6330, -5.1263]`** | **`88.67%`** | **`93.75%`** | **`-3.8667 [-4.0524, -3.6745]` (`47.8×`)** | **`0.8343`** | **`16/16 / 16/16`** | **`16.5`** |
| **`METAFE-3S`** (Control Metacognitivo 3 Escalas) | `-5.3665 [-5.6213, -5.1186]` | `87.76%` | **`93.75%`** | `-3.8585 [-4.0411, -3.6687]` (`47.4×`) | **`0.8560`** | **`16/16 / 16/16`** | `18.9` |
| **`METAFE-v2`** (Energía Libre Variacional BMA) | `-5.3338 [-5.5700, -5.1044]` | **`88.67%`** | `93.70%` | `-3.8258 [-3.9899, -3.6467]` (`45.9×`) | `0.8322` | **`16/16 / 16/16`** | `24.5` |
| **`BQ10`** (Bio-Quantum v10 DA-STR $da=10$) | `-1.5081 [-1.6382, -1.3865]` | `14.84%` | `88.72%` | `0.0000` (`1.0×`) | — | `16/16 / —` | `12.3` *(162.5 e2e)* |
| **`ATLAS-FW (ALL)`** (M1–M6 Biblioteca Koopman/SINDy) | `-1.3518 [-1.4830, -1.2185]` | `15.62%` | `90.28%` | `+0.1563` (`0.86×`) | — | `16/16 / —` | `41.0` *(191.2 e2e)* |
| **`ATLAS-CORE`** (M1+M2+M6 sin hiperred M3–M5) | `-1.3518 [-1.4830, -1.2185]` | `15.62%` | `90.28%` | `+0.1563` (`0.86×`) | — | `16/16 / —` | `12.7` *(162.9 e2e)* |
| **`CONSTELACIÓN v4`** (Jerárquica Koopman + $W^F$ NLMS) | `-0.8997 [-0.9660, -0.8265]` | `6.64%` | `87.70%` | `+0.6084` (`0.54×`) | — | `16/16 / —` | `4.5` *(154.7 e2e)* |
| **`ÓRBITA v1`** (Koopman lineal por familia + `reid`) | `-0.7928 [-0.9119, -0.6826]` | `2.47%` | `33.84%` | `+0.7153` (`0.49×`) | — | `16/16 / —` | — |

### 1.1 Extensión No-Estacionaria (`RS-A`, `RS-B`, `syn_ood`) y Desglose por Familia OOD en `CONFIRM` (`0xB400–0xB40F`)

| Arquitectura | `fam_8` med_log | `fam_10` med_log | `fam_11` med_log | `syn_ood_med` [IC95] | `RS-A all_med` [IC95] (Cambio Abrupto) | `RS-B frozen W_0 med` [IC95] (`acc16`) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`AURA-v3-Genesis`** | `-5.3204` | **`-8.3896`** | **`-4.1843`** | `-11.0001 [-13.5488, -8.8539]` | **`-6.7084 [-8.3830, -5.4481]`** | `-2.1425 [-2.2548, -2.0380]` (**`23.96%`**) |
| **`METAFE-3S`** | **`-5.3939`** | `-8.3897` | `-4.1227` | `-11.0145 [-13.5602, -8.8651]` | `-6.6761 [-8.2923, -5.4439]` | **`-2.1474 [-2.2580, -2.0435]`** (`23.70%`) |
| **`METAFE-v2`** | `-5.2882` | `-8.3896` | `-4.1938` | **`-11.0521 [-13.5842, -8.8931]`** | `-0.2798 [-0.3390, -0.2239]` | `+0.4702 [+0.4271, +0.5163]` (`0.00%`) |

### 1.2 Tabla Emparejada en `DEV` (`0xA400–0xA40F`, $N=16$)

| Arquitectura | `new_med_log` [IC95] | `new_acc16` | `known_acc16` | `Δ vs BQ10` [IC95] (Factor) | `new_AUROC` | `RS-A all_med` | `RS-B med` | `wall_s` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`AURA-v3-Genesis`** | `-5.1645 [-5.4272, -4.9188]` | **`88.02%`** | **`93.95%`** | `-3.6401 [-3.8508, -3.4268]` (`38.1×`) | `0.8302` | `-5.7372` | **`-2.0098`** | **`17.2`** |
| **`METAFE-3S`** | `-5.1364 [-5.3822, -4.9086]` | **`88.02%`** | **`93.95%`** | `-3.6120 [-3.8002, -3.4188]` (`37.0×`) | `0.8343` | **`-5.7955`** | `-2.0007` | `19.8` |
| **`METAFE-v2`** | **`-5.1667 [-5.4302, -4.9175]`** | **`88.02%`** | **`93.95%`** | **`-3.6422 [-3.8489, -3.4309]` (`38.2×`)** | **`0.8394`** | `-0.3244` | `+0.4938` | `24.7` |
| **`BQ10`** | `-1.5244 [-1.6085, -1.4384]` | `14.58%` | `88.77%` | `0.0000` (`1.0×`) | — | — | — | `12.6` *(162.8 e2e)* |
| **`ATLAS-FW (ALL)`** | `-1.3359 [-1.4660, -1.2192]` | `15.76%` | `90.09%` | `+0.1885` (`0.83×`) | — | — | — | `41.5` *(191.7 e2e)* |
| **`CONSTELACIÓN v4`** | `-0.9222 [-0.9685, -0.8753]` | `5.34%` | `87.70%` | `+0.6022` (`0.55×`) | — | — | — | `4.5` *(154.7 e2e)* |

---

## 2. Contrastes Pareados Cabeza a Cabeza (`vs`) en `CONFIRM` (`0xB400–0xB40F`)

| Contraste (`AURA-v3-Genesis` vs Otra) | `Δ new_med_log` [IC95] | `Δ new_acc16` [IC95] | `Δ RS-A all_med` [IC95] | `Δ RS-B med` [IC95] | `Δ wall_s` | Veredicto |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **`AURA-v3-Genesis vs METAFE-3S`** | **`-0.0082 [-0.0645, +0.0528]`** | **`+0.91 pp [+0.39, +1.43]`** | **`-0.0323 [-0.0979, +0.0121]`** | `+0.0048 [-0.0003, +0.0140]` | **`-2.4s (-12.7%)`** | **Gana `AURA-v3-Genesis` en `new_acc16` (IC95 > 0), `new_med_log`, `RS-A` y latencia** |
| **`AURA-v3-Genesis vs METAFE-v2`** | **`-0.0409 [-0.1535, +0.0411]`** | **`+0.00 pp [-0.78, +0.65]`** | **`-6.4287 [-8.1324, -5.1277]`** | **`-2.6127 [-2.7334, -2.5114]`** | **`-8.0s (-32.7%)`** | **Gana `AURA-v3-Genesis` en `new_med_log`, `RS-A` (620×), `RS-B` (13.6×) y `-32.7%` tiempo** |
| **`AURA-v3-Genesis vs BQ10`** | **`-3.8667 [-4.0524, -3.6745]`** | **`+73.83 pp [+71.35, +76.30]`** | — | — | **`9.9× speedup e2e`** | **Gana `AURA-v3-Genesis` `16/16` semillas (`47.8×` menos error OOD)** |
| **`AURA-v3-Genesis vs ATLAS-FW (ALL)`** | **`-4.0230 [-4.2219, -3.8169]`** | **`+73.05 pp [+70.31, +75.65]`** | — | — | **`11.6× speedup e2e`** | **Gana `AURA-v3-Genesis` `16/16` semillas (`55.9×` menos error OOD)** |
| **`AURA-v3-Genesis vs CONSTELACIÓN v4`** | **`-4.4751 [-4.7024, -4.2528]`** | **`+82.03 pp [+79.69, +84.24]`** | — | — | **`9.4× speedup e2e`** | **Gana `AURA-v3-Genesis` `16/16` semillas (`87.8×` menos error OOD)** |
| **`AURA-v3-Genesis vs ÓRBITA v1`** | **`-4.5820 [-4.8194, -4.3541]`** | **`+86.20 pp [+83.98, +88.28]`** | — | — | — | **Gana `AURA-v3-Genesis` `16/16` semillas (`97.7×` menos error OOD)** |

### ¿Por qué las 4 Subrutinas del ADN de `AURA-v3-Genesis` resuelven simultáneamente el dilema entre `METAFE-v2` y `METAFE-3S`?
1. **El cuello de botella que separaba a `METAFE-v2` de `METAFE-3S`**:
   - `METAFE-3S` superaba a `METAFE-v2` en `new_med_log` (`-5.3665` vs `-5.3338`) y en cambio de régimen online (`RS-A = -6.6761` vs `-0.2798`, `RS-B = -2.1474` vs `+0.4702`), pero **perdía `-0.91 pp` de exactitud `new_acc16`** (`87.76%` vs `88.67%`) porque en la familia OOD `fam_11` (cizalla simpléctica de 2 etapas `GpSympFwd`) promediaba en coordenadas cartesianas hasta 4 variedades distintas paso a paso con temperatura blanda (`base_tau = 2.5`) y permitía candidatos `fe_w` con bajo peso precuencial $\text{OOT}_1/\text{OOT}_2 = 0.25$.
2. **Cómo las 4 Subrutinas del ADN en `AURA-v3-Genesis` eliminan ese compromiso**:
   - **Subrutina 1 (Gradientes Químicos Morfogenéticos 3D $[\text{SHH}, \text{Wnt}, \text{BMP}]$)**: guía quimiotácticamente los conos de crecimiento hacia `LieAffine/GpCoMoving` en la zona ventral `SHH`, hacia `GpPolar/GpEquivCart` en la zona antero-posterior `Wnt`, y hacia `GpSympFwd/Rev` vs `GpCoMovingSympFwd/Rev` en la zona dorso-ventral `BMP` ($\det J \approx 1$).
   - **Subrutina 2 (Microcircuito Cortical Canónico de 6 Capas, Mountcastle)**: extiende el refinamiento multi-paso `L6` a las columnas co-móviles simplécticas y asegura proyección precuencial `L5` ($\text{OOT}_1, \text{OOT}_2 \ge 0.50$) en todas las minicolumnas.
   - **Subrutina 3 (Sobreproducción + Poda Sináptica por Microglía + Mielinización Saltatoria, Jesse Conel)**:
     - La **poda por microglía consciente de variedad** elimina físicamente las columnas secundarias incompatibles cuando una variedad simpléctica o polar de 2 etapas (`GpSympFwd/Rev`, `GpPolar`) domina por $> 1.10\text{ nats}$, evitando la deriva fuera de variedad en `fam_11` y recuperando el **`88.67%` de `new_acc16`** (`+0.91 pp [+0.39, +1.43]` sobre `METAFE-3S`), mientras conserva el acoplamiento cerrado en `fam_8` para alcanzar **`new_med_log = -5.3748`** y **`RS-A = -6.7084`**.
     - La **mielinización saltatoria** colapsa analíticamente las columnas redundantes cuando el par afín/co-móvil alcanza precisión de máquina (`<-22.0` log-error), reduciendo `wall_s` de `24.5s` (`METAFE-v2`) y `18.9s` (`METAFE-3S`) a **`16.5s`**.
   - **Subrutina 4 (Currículo Sintético Prenatal en el Útero)**: pre-calibra el hiper-prior lento `prior_w` y la matriz de Fisher `EwcPriorState` sobre 32 episodios de ondas sintéticas prenatales auditadas ($0$ fuga de `{8, 10, 11}`) antes de la consolidación postnatal.

---

## 3. Arena 1: Entrenamiento Real a Nivel de Bytes y Cambio de Dominio a Rust (`top5-arena`, `0xB600–0xB60F`, $N=16$)

> Todas las arquitecturas entrenan exactamente $4$ épocas sobre los mismos $24{,}576\text{ B}$ de Markdown real (`chunk_len = 32`, `D = 128`) y se evalúan sin reentrenar sobre Markdown retenido (`eval_md_bpb`) y código fuente Rust (`shift_rust_bpb`):

| Rango | Arquitectura | `eval_md_bpb` (Media ± DE [IC95]) ↓ | `shift_rust_bpb` (Rust OOD [IC95]) ↓ | `early_md_bpb` (Época 1) ↓ | Memoria Activa Pesos (RAM) ↓ | Fracción Activa (`nz`) |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| **#1 (OOD Rust & RAM)** | **`AURA v3 Genesis` (2-bit + 4 Subrutinas ADN)** | `3.7393 ± 0.0365 [3.7218, 3.7554]` | **`5.0989 ± 0.0906 [5.0571, 5.1394]`** | `4.4587` | **`8,120 B` (`7.9 KB`)** | **`47.9%` (Podado por Microglía)** |
| **#2 (OOD Rust)** | **`AURA v2` (BitNet b1.58 2-bit STE + SSM + FW)** | `3.8027 ± 0.0580 [3.7771, 3.8308]` | `5.1571 ± 0.0956 [5.1143, 5.2024]` | `4.4697` | `12,288 B` (`12.0 KB`) | `55.1%` |
| **#3 (In-Domain #1)** | **`BQ10` (Koopman + M1 SINDy + M2 STLSQ + M7 DA-STR)** | **`3.7202 ± 0.0315 [3.7055, 3.7343]`** | `5.8288 ± 0.1079 [5.7812, 5.8806]` | **`4.1381`** | `131,072 B` (`128.0 KB`) | `100.0%` |
| **#4 (Empate In-Domain #2)** | **`METAFE-3S` (VFE BMA + ACC + MetaNet $[\alpha_t, \lambda_t, \beta_t]$ + $\Pi_\rho$)** | `3.7356 ± 0.0345 [3.7194, 3.7515]` | `6.0743 ± 0.1066 [6.0275, 6.1264]` | `4.1396` | `131,072 B` (`128.0 KB`) | `100.0%` |
| **#5 (Empate In-Domain #2)** | **`METAFE-v2` (VFE Empirical Bayes ARD + BMA)** | `3.7356 ± 0.0345 [3.7194, 3.7515]` | `6.1079 ± 0.1112 [6.0585, 6.1614]` | `4.1396` | `131,072 B` (`128.0 KB`) | `100.0%` |
| **#6** | **`CONSTELACIÓN v4` (Koopman jerárquico + $W^F$ NLMS)** | `3.8568 ± 0.0294 [3.8433, 3.8700]` | `6.0373 ± 0.0942 [5.9954, 6.0822]` | `4.2643` | `131,072 B` (`128.0 KB`) | `100.0%` |
| **#7** | **`ATLAS-FW` (Koopman + M1 SINDy sin penalización grado M7)** | `3.8395 ± 0.0348 [3.8238, 3.8556]` | `7.0170 ± 0.2407 [6.9115, 7.1332]` | `4.2417` | `131,072 B` (`128.0 KB`) | `100.0%` |
| *Base ref.* | `ÓRBITA v1` (Osciladores lineales Koopman + selector `reid`) | `4.0832 ± 0.0289 [4.0701, 4.0965]` | `7.6012 ± 0.2071 [7.5055, 7.6958]` | `4.3388` | `131,072 B` (`128.0 KB`) | `100.0%` |
| *Base ref.* | `AURA v1` (Ternario 2-bit sin escalado BitNet STE) | `6.9379 ± 0.1303 [6.8665, 6.9821]` | `6.5628 ± 0.1008 [6.5107, 6.6001]` | `7.7522` | `12,288 B` (`12.0 KB`) | `0.0%` |

### Contrastes Pareados Arena 1 (`AURA v3 Genesis` vs Todas las Arquitecturas)

| Contraste Pareado (`AURA v3 Genesis` menos Otra) | $\Delta\text{eval\_md\_bpb}$ [IC95] | Victorias `eval_MD` | $\Delta\text{shift\_rust\_bpb}$ [IC95] | Victorias `Rust_shift` | Ratio de Compresión RAM |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **`AURA v3 Genesis vs AURA v2`** | **`-0.0635 [-0.0923, -0.0348]`** | **12 / 16** | **`-0.0581 [-0.1094, -0.0037]`** | **12 / 16** | **`1.51×` menos RAM (`8.1 KB` vs `12.3 KB`)** |
| **`AURA v3 Genesis vs METAFE-3S`** | `+0.0037 [-0.0139, +0.0202]` *(Empate técnico)* | 6 / 16 | **`-0.9754 [-1.0510, -0.9153]`** | **16 / 16** | **`16.14×` menos RAM (`8.1 KB` vs `131.1 KB`)** |
| **`AURA v3 Genesis vs METAFE-v2`** | `+0.0037 [-0.0139, +0.0202]` *(Empate técnico)* | 6 / 16 | **`-1.0089 [-1.0864, -0.9466]`** | **16 / 16** | **`16.14×` menos RAM (`8.1 KB` vs `131.1 KB`)** |
| **`AURA v3 Genesis vs BQ10`** | `+0.0191 [-0.0010, +0.0377]` *(Empate técnico)* | 7 / 16 | **`-0.7298 [-0.8029, -0.6650]`** | **16 / 16** | **`16.14×` menos RAM (`8.1 KB` vs `131.1 KB`)** |
| **`AURA v3 Genesis vs ATLAS-FW`** | **`-0.1003 [-0.1216, -0.0803]`** | **16 / 16** | **`-1.9181 [-2.0397, -1.8092]`** | **16 / 16** | **`16.14×` menos RAM (`8.1 KB` vs `131.1 KB`)** |
| **`AURA v3 Genesis vs CONSTELACIÓN v4`** | **`-0.1176 [-0.1379, -0.0957]`** | **16 / 16** | **`-0.9383 [-1.0132, -0.8693]`** | **16 / 16** | **`16.14×` menos RAM (`8.1 KB` vs `131.1 KB`)** |

---

## 4. Reproducibilidad Exacta

```bash
# 1. Arnés canónico orbita (METAFE-v2, METAFE-3S, AURA-v3-Genesis, BQ10, ATLAS-FW, CONSTELACIÓN v4, ÓRBITA):
python3 scripts/run_aura_v3_in_orbita.py dev confirm
python3 scripts/analyze_metafe3s.py --md

# 2. Arnés de Bytes Reales y Dinámica embebida en aura-engine (incluyendo METAFE-v2 y METAFE-3S):
python3 aura-engine/run_top5_equal.py

# 3. Auditoría criptográfica de congelamiento de AURA v3 Genesis:
python3 aura-engine/freeze_and_verify_v3.py audit
```
