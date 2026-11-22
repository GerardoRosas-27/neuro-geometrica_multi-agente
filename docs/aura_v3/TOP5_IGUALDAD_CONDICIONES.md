# Torneo Head-to-Head en Igualdad Estricta de Condiciones — Las 5 Mejores Arquitecturas del Repositorio (Incluyendo AURA v3 Genesis)

- **Rama:** `exp/aura-v3-genesis` (código base congelado de `aura-engine/crates/` intacto: `SHA-256 = cc90b95faa8a1885e308b81270a94babcb57b624a7323599edb8b38f0f366171`, `AUDIT-v3 PASS`)
- **Arnés de torneo en igualdad de condiciones:** [`aura-engine/top5-arena/src/main.rs`](../../aura-engine/top5-arena/src/main.rs) + [`aura-engine/run_top5_equal.py`](../../aura-engine/run_top5_equal.py)
- **Semillas evaluadas:** 16 semillas `CONFIRM` (`0xB600–0xB60F`), bootstrap pareado de 4,000 remuestreos (IC 95%).
- **Datos crudos y resumen JSON:** [`top5_equal_arena1_bytes.csv`](top5_equal_arena1_bytes.csv), [`top5_equal_arena2_dynamics.csv`](top5_equal_arena2_dynamics.csv), [`top5_equal_summary.json`](top5_equal_summary.json).

---

## 1. Las 5 Mejores Arquitecturas Comparadas (+ 2 Baselines Históricos)

Para evitar cualquier sesgo de arnés o de pesos pre-entrenados externos, las **5 arquitecturas ganadoras de las distintas etapas del repositorio** (más `ÓRBITA-v1` y `AURA v1` como referencias históricas) se entrenaron y evaluaron **desde cero bajo exactamente las mismas condiciones** en dos dominios complementarios:

1. **`CONSTELACIÓN v4` (`constelacion_v4` — Ganadora etapa v4):** Bloques de rotación-escala de Koopman $(\rho_j, \omega_j)$ con levantamiento no lineal sigmoidal $\sigma(R K \mu + b)$, cabeza predictiva lenta $W^P$ y campo rápido por episodio $W^F$ adaptado con **NLMS multi-paso (`V4_T_F = 128`, $\eta_f = 1.0$)** y precisión normalizada $\Pi$.
2. **`ATLAS-FW` (`atlas_fw` — Ganadora etapa ATLAS):** Núcleo `v4` + biblioteca **M1** de generadores no lineales Koopman/SINDy (polinomios de grado 2 y 3, Fourier $\sin/\cos$, $\tanh$, radial $r\cdot x$) + regresión dispersa iterada **M2 (STLSQ, `thr = 0.05`, `N_IT = 3`)** + prior meta-contextual **M3–M5** + compuerta precuencial **M6 (CLS)** en los cuadros $1..K$.
3. **`BQ10` (`bq10` — Bio-Quantum v10, Ganadora previa en `main`):** Núcleo `ATLAS-FW` (`M1 + M2 STLSQ + M6 CLS`, sin overhead M3–M5) + **M7 DA-STR (`da = 10`)**, que penaliza cada generador de grado $g \in \{1, 2, 3\}$ con $\lambda_j = \lambda \cdot 10^{g-1}$ vía reescalado de columnas $10^{-(g-1)/2}$.
4. **`AURA v2` (`aura_v2` — Ganadora neuromórfica v2):** Pesos ternarios 2-bit (`{-1, 0, +1}`) con **BitNet b1.58 STE** (escala por fila $\alpha_r$ + AdamW) + **Escaneo Paralelo Asociativo de Blelloch** $O(\log L)$ para el SSM selectivo + `EntropyPatcher` BLT con **VICReg** + adaptación online con Fast-Weights NLMS + **Hipocampo HNSW** $O(\log N)$ con compuerta epistémica reducible.
5. **`AURA v3 Genesis` (`aura_v3_genesis` — Nueva Arquitectura):** `AURA v2` mejorada con las **4 subrutinas generativas de descompresión del ADN**:
   - **(1) Enrutamiento por gradientes químicos morfogenéticos 3D (`Shh`, `Wnt`, `BMP`, `Eph/ephrin`)**: coordenadas topográficas y canales polares-armónicos sin direcciones explícitas (`4,209.8×` de compresión genética).
   - **(2) Microcircuito cortical canónico de Mountcastle de 6 capas (`L1–L6`)**: plantilla única de 64 B replicada en minicolumnas con inhibición lateral k-WTA y compuerta apical `L6` sobre la innovación sensorial $\Delta x_K$.
   - **(3) Sobreproducción + Poda física por Microglía + Mielinización (Curva de Jesse Conel)**: explosión sináptica temprana seguida de poda física de sinapsis redundantes (`47.9%` densidad activa en bytes; `10.1 / 52` sinapsis activas en dinámica 2D) y mielinización saltatoria.
   - **(4) Currículo sintético prenatal en el útero (0 fuga de datos)**: ondas coherentes sintéticas previas al primer dato real de entrenamiento.
- *Referencias base incluidas en la misma corrida:* **`ÓRBITA-v1` (`orbita_v1`)** y **`AURA v1` (`aura_v1`)**.

---

## 2. Arena 1: Entrenamiento End-to-End en Secuencias de Bytes Reales + Cambio de Dominio Online (Markdown $\to$ Rust)

### Condiciones idénticas de entrenamiento
- **Corpus de entrenamiento:** Exactamente los mismos `24,576 B` de Markdown real del repositorio (`README.md` + `docs/RESULTADOS_CONSOLIDADOS.md`, bloques disjuntos de 1 KiB).
- **Presupuesto:** Exactamente **4 épocas**, `chunk_len = 32`, dimensión oculta/estado `D = 128`, mismo orden de chunks por semilla (`0` pesos externos pre-entrenados).
- **Evaluación held-out:** Mismos `6,144 B` de Markdown no visto (`early_md_bpb` tras época 1; `eval_md_bpb` tras época 4).
- **Evaluación cambio de dominio online:** Mismos `6,144 B` de código fuente Rust no visto (`orbita/src/model.rs` + `orbita/src/fam.rs`, `shift_rust_bpb`), evaluados prequencialmente byte a byte con pesos lentos congelados y el mecanismo rápido propio de cada arquitectura.

### Tabla de Resultados — Arena 1 (`CONFIRM`, 16 semillas `0xB600–0xB60F`)

| Ranking Global | Arquitectura | `eval_md_bpb` (Held-Out Markdown) | **`shift_rust_bpb` (Online Rust Shift)** | **Media Global `(MD+Rust)/2`** | `early_md_bpb` (Época 1) | RAM Pesos Activos | Densidad No-Cero | `erank` |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| **#1** | **`AURA v3 Genesis` (Nueva · Top-5)** | **`3.7393`** `[3.7218, 3.7554]` | **`5.0989`** `[5.0571, 5.1394]` | **`4.4191 BPB`** | `4.4587` | **`8,120 B`** | **`47.9%`** | `31.77` |
| **#2** | **`AURA v2` (Top-5)** | `3.8027` `[3.7771, 3.8308]` | `5.1571` `[5.1143, 5.2024]` | **`4.4799 BPB`** | `4.4697` | `12,288 B` | `55.1%` | `31.74` |
| **#3** | **`BQ10 (Bio-Quantum v10)` (Top-5)** | **`3.7202`** `[3.7055, 3.7343]` | `5.8304` `[5.7828, 5.8823]` | **`4.7753 BPB`** | `4.1381` | `131,072 B` | `100.0%` | `12.82` |
| **#4** | **`CONSTELACIÓN v4` (Top-5)** | `3.8568` `[3.8433, 3.8700]` | `6.0373` `[5.9954, 6.0822]` | **`4.9471 BPB`** | `4.2643` | `131,072 B` | `100.0%` | `10.19` |
| **#5** | **`ATLAS-FW` (Top-5)** | `3.8395` `[3.8238, 3.8556]` | `7.0204` `[6.9145, 7.1370]` | **`5.4300 BPB`** | `4.2417` | `131,072 B` | `100.0%` | `18.30` |
| *(Ref)* | `ÓRBITA-v1` (Baseline lineal) | `4.0832` `[4.0701, 4.0965]` | `7.6012` `[7.5055, 7.6958]` | `5.8422 BPB` | `4.3388` | `131,072 B` | `100.0%` | `6.87` |
| *(Ref)* | `AURA v1` (Baseline ternario sin escala) | `6.9379` `[6.8665, 6.9821]` | `6.5628` `[6.5107, 6.6001]` | `6.7504 BPB` | `7.7522` | `12,288 B` | `54.9%` | `0.13` |

### Contrastes Pareados Head-to-Head (`vs`) en Arena 1 (`AURA v3 Genesis` menos Otra; `< 0` gana AURA v3 Genesis)

| Contraste Pareado (`N = 16` semillas) | $\Delta$ `eval_md_bpb` `[IC 95%]` | Semillas ganadas (MD) | $\Delta$ `shift_rust_bpb` `[IC 95%]` | Semillas ganadas (Rust) | Compresión de Memoria de Pesos |
|---|---:|---:|---:|---:|---:|
| **`AURA v3 Genesis` vs `AURA v2`** | **`-0.0635 [-0.0923, -0.0348]`** | **12 / 16** | **`-0.0581 [-0.1094, -0.0037]`** | **12 / 16** | **`1.51×` menos pesos activos** (`47.9%` vs `55.1%`) |
| **`AURA v3 Genesis` vs `BQ10`** | `+0.0191 [-0.0010, +0.0377]` *(empate estadístico)* | 7 / 16 | **`-0.7315 [-0.8051, -0.6664]`** | **16 / 16** | **`16.14×` menos RAM** (`8.1 KB` 2-bit vs `131.1 KB` `f32`) |
| **`AURA v3 Genesis` vs `CONSTELACIÓN v4`** | **`-0.1176 [-0.1379, -0.0957]`** | **16 / 16** | **`-0.9383 [-1.0132, -0.8693]`** | **16 / 16** | **`16.14×` menos RAM** |
| **`AURA v3 Genesis` vs `ATLAS-FW`** | **`-0.1003 [-0.1216, -0.0803]`** | **16 / 16** | **`-1.9214 [-2.0437, -1.8123]`** | **16 / 16** | **`16.14×` menos RAM** |
| **`AURA v3 Genesis` vs `ÓRBITA-v1`** | **`-0.3440 [-0.3631, -0.3255]`** | **16 / 16** | **`-2.5022 [-2.6054, -2.4010]`** | **16 / 16** | **`16.14×` menos RAM** |
| **`AURA v3 Genesis` vs `AURA v1`** | **`-3.1987 [-3.2509, -3.1280]`** | **16 / 16** | **`-1.4639 [-1.5274, -1.3977]`** | **16 / 16** | **`1.51×` menos pesos activos** |

> **Lectura honesta de Arena 1:** En Markdown estático held-out, `BQ10` con pesos `f32` densos (`131 KB`) y `AURA v3 Genesis` con pesos ternarios 2-bit podados por microglía (`8.1 KB`, **16.1× menos memoria**) empatan estadísticamente (`3.7202` vs `3.7393`, el IC95 `[-0.0010, +0.0377]` incluye 0), y ambos superan significativamente a `AURA v2` (`3.8027`), `ATLAS-FW` (`3.8395`) y `CONSTELACIÓN v4` (`3.8568`). Sin embargo, cuando ocurre el **cambio brusco de dominio a código Rust no visto (`shift_rust_bpb`)**, **`AURA v3 Genesis` gana de forma aplastante (`5.0989 BPB`)**, superando a `BQ10` por **`-0.7315 BPB` en 16/16 semillas**, a `CONSTELACIÓN v4` por **`-0.9383 BPB`**, a `ATLAS-FW` por **`-1.9214 BPB`** y a `AURA v2` por **`-0.0581 BPB`**, gracias a la combinación de memoria episódica hipocampal HNSW + Fast-Weights NLMS + poda microglial.

---

## 3. Arena 2: Dinámica Geométrica Multi-Horizonte 2D (`CST4`: 8 Familias Conocidas + 3 Familias OOD Nuevas `{8, 10, 11}`, $K=4 \to h=16$)

### Condiciones idénticas de entrenamiento
- **Entrenamiento desde cero por semilla:** Exactamente los mismos `320` episodios de las 8 familias geométricas conocidas (`0..8`: traslación, rotación, reflexión, escala, cizalla, afín, composición rot-escala y deformación radial; $K_{\text{points}} = 8$ puntos 2D, horizonte $h = 16$).
- **Evaluación held-out:** Exactamente los mismos `128` episodios de familias conocidas (`0..8`) y `96` episodios OOD de las 3 familias nuevas nunca vistas en entrenamiento (`{8, 10, 11}`: rotación con acoplamiento sinusoidal, modulación armónica radial de 3 lóbulos y cizalla hiperbólica acoplada).
- **Protocolo $K = 4 \to h = 16$:** Cada arquitectura observa únicamente las primeras $K = 4$ transiciones (`t = 0..=4`) en contexto y predice autorregresivamente (reinyectando sus propias predicciones) hasta $h = 16$.

### Tabla de Resultados — Arena 2 (`CONFIRM`, 16 semillas `0xB600–0xB60F`)

| Ranking Global | Arquitectura | **`new_med_log` ($h=16$ OOD Nuevas)** | `known_med_log` ($h=16$ Conocidas) | **`known_acc16` ($<5\%$ err)** | **`new_acc16` ($<5\%$ err OOD)** | Sinapsis Activas / Ep. | Tiempo / Semilla |
|---:|---|---:|---:|---:|---:|---:|---:|
| **#1** | **`AURA v3 Genesis` (Nueva · Top-5)** | **`-2.2169`** `[-2.2893, -2.1417]` | **`-12.6589`** | **`88.87%`** `[87.55, 90.23]` | **`27.21%`** `[25.00, 29.43]` | **`10.1`** *(podadas de 52)* | `54.8 ms` |
| **#2** | **`BQ10 (Bio-Quantum v10)` (Top-5)** | `-1.7255` `[-1.8002, -1.6459]` | `-8.1336` | `86.08%` `[84.72, 87.30]` | `18.49%` `[16.34, 20.57]` | `10.3` *(de 40)* | `36.1 ms` |
| **#3** | **`AURA v2` (Top-5)** | `-1.3145` `[-1.3778, -1.2620]` | `-4.7973` | `82.57%` `[81.10, 84.03]` | `11.98%` `[10.09, 13.80]` | `14.0` | `38.6 ms` |
| **#4** | **`CONSTELACIÓN v4` (Top-5)** | `-1.3011` `[-1.3474, -1.2558]` | `-2.4362` | `32.52%` `[30.27, 34.81]` | `2.28%` `[1.43, 3.19]` | `18.0` | `15.5 ms` |
| **#5** | **`ATLAS-FW` (Top-5)** | `-1.2923` `[-1.3437, -1.2385]` | `-2.4704` | `29.79%` `[27.54, 31.98]` | `3.65%` `[2.60, 4.75]` | `20.2` | `35.6 ms` |
| *(Ref)* | `ÓRBITA-v1` (Baseline lineal) | `-0.7049` `[-0.7636, -0.6474]` | `-1.4360` | `20.02%` `[18.51, 21.48]` | `2.08%` `[1.30, 2.93]` | `6.0` | `17.1 ms` |

### Contrastes Pareados Head-to-Head (`vs`) en Arena 2 (`AURA v3 Genesis` menos Otra)

| Contraste Pareado (`N = 16` semillas) | $\Delta$ `new_med_log` ($<0$ mejor) `[IC 95%]` | Semillas ganadas | $\Delta$ `known_med_log` ($<0$ mejor) | $\Delta$ `known_acc16` (`pp`) `[IC 95%]` | $\Delta$ `new_acc16` (`pp`) `[IC 95%]` |
|---|---:|---:|---:|---:|---:|
| **`AURA v3 Genesis` vs `BQ10`** | **`-0.4914 [-0.5533, -0.4324]`** | **16 / 16** | **`-4.5253`** | **`+2.78 pp [+1.86, +3.71]`** | **`+8.72 pp [+7.68, +9.70]`** |
| **`AURA v3 Genesis` vs `AURA v2`** | **`-0.9024 [-0.9632, -0.8444]`** | **16 / 16** | **`-7.8616`** | **`+6.30 pp [+5.13, +7.47]`** | **`+15.23 pp [+13.28, +17.38]`** |
| **`AURA v3 Genesis` vs `CONSTELACIÓN v4`** | **`-0.9158 [-0.9695, -0.8640]`** | **16 / 16** | **`-10.2227`** | **`+56.35 pp [+54.20, +58.50]`** | **`+24.93 pp [+22.66, +27.41]`** |
| **`AURA v3 Genesis` vs `ATLAS-FW`** | **`-0.9246 [-0.9822, -0.8655]`** | **16 / 16** | **`-10.1885`** | **`+59.08 pp [+56.98, +61.18]`** | **`+23.57 pp [+21.35, +25.98]`** |
| **`AURA v3 Genesis` vs `ÓRBITA-v1`** | **`-1.5120 [-1.5633, -1.4592]`** | **16 / 16** | **`-11.2229`** | **`+68.85 pp [+67.24, +70.46]`** | **`+25.13 pp [+22.98, +27.54]`** |

---

## 4. ¿Por Qué Gana AURA v3 Genesis Bajo Igualdad de Condiciones?

1. **En Dinámica Geométrica Multi-Horizonte 2D (Arena 2):**
   - `BQ10` supera claramente a `ATLAS-FW` y `CONSTELACIÓN v4` gracias a **M7 DA-STR (`da = 10`)**, que frena la explosión de varianza de los polinomios de grado 2 y 3 al extrapolar de $K=4$ a $h=16$.
   - **`AURA v3 Genesis` supera a `BQ10` en 16/16 semillas (`Δnew_med_log = -0.4914 [-0.5533, -0.4324]`, `+8.72 pp` en `new_acc16`, `+2.78 pp` en `known_acc16` y `-4.53` en `known_med_log`)** porque añade las **4 subrutinas biológicas del ADN**:
     - Los **gradientes morfogenéticos polares-armónicos (Subrutina 1)** proveen coordenadas radiales/angulares naturales para deformaciones no cartesianas.
     - La **compuerta apical `L6` del microcircuito canónico (Subrutina 2)** evalúa la energía libre predictiva de la **innovación $\Delta x_K = x(K) - x(K-1)$ en la transición $t=K$ retenida** (dentro de la ventana $0..=K$), evitando el sobreajuste in-sample del que sufre la compuerta M6 cuando evalúa sobre los mismos cuadros usados para ajustar STLSQ.
     - La **poda física por Microglía (Subrutina 3)** elimina todas las ramas no lineales (`46` de `52` sinapsis podadas, dejando solo `6` sinapsis mielinizadas) cuando la dinámica es afín (`fe_k3 < 1e-3`), y poda las ramas polares externas salvo que reduzcan el error de innovación en $t=K$ en $\ge 10\%$, logrando menor número medio de sinapsis activas (`10.1` vs `10.3` de `BQ10` y `20.2` de `ATLAS-FW`).
2. **En Secuencias de Bytes Reales y Cambio de Dominio a Código Rust (Arena 1):**
   - Mientras los modelos basados solo en regresión continua por ventana (`CONSTELACIÓN v4`, `ATLAS-FW`, `BQ10`) carecen de memoria episódica de largo alcance y sufren al saltar de prosa Markdown a código Rust (`5.83–7.02 BPB`), **`AURA v2` (`5.1571 BPB`) y `AURA v3 Genesis` (`5.0989 BPB`)** combinan **BitNet b1.58 2-bit STE + Escaneo Paralelo de Blelloch + Fast-Weights NLMS + Hipocampo HNSW**, logrando la mejor generalización global con **16.1× menos memoria de pesos (`8,120 B` vs `131,072 B`)**.
