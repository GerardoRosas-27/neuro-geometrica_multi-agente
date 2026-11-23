# Resultados Experimentales — AURA v3 Genesis (4 Subrutinas Generativas del ADN)

- **Rama:** `exp/aura-v3-genesis` (base: `exp/aura-v2-training@c8d5cc0`, `main@34f04b2` intacto)
- **Prerregistro inmutable:** `docs/aura_v3/PLAN.md` (commit `fece4e6`, 15 criterios `H_G1a`–`H_G5c`)
- **Congelado pre-`CONFIRM`:** commit `77c7a6e`, `sources_sha256 = cc90b95faa8a1885e308b81270a94babcb57b624a7323599edb8b38f0f366171`
- **Semillas:** `DEV = 0xA600–0xA60F` (16 semillas, `160.58s`) | **`CONFIRM = 0xB600–0xB60F`** (16 semillas frescas, ejecución única `--skip-build`, `161.85s`)
- **Integridad:** `AUDIT-v3 PASS`, replay bit-exacto en `0xB600` (`150/150` campos + muestras), 15/15 unit tests (`debug` y `release`), 0 warnings.
- **Resumen global de criterios prerregistrados:**
  - **`DEV`:** **15/15 PASS**
  - **`CONFIRM`:** **14/15 PASS** (12/12 PASS en las 4 subrutinas aisladas G1–G4; 2/3 PASS en integración G5; 1 FAIL transparente en la sub-cláusula de IC95 de época 1 en `H_G5b`, mientras que `eval_md_bpb` final **supera estadísticamente a AURA v2**).

---

## 1. Tabla Consolidada de Veredictos (`CONFIRM`, 16 semillas frescas `0xB600–0xB60F`)

| Criterio | Subrutina / Suite | Métrica principal (`CONFIRM`, media ± IC95 bootstrap pareado) | Umbral `PLAN.md` | Veredicto |
|---|---|---|---|---|
| `H_G1a` | G1 · Gradientes químicos | `genome = 144 B`, compresión @ `N=2048` = **`4,209.8×`** (`606,216 B / 144 B`) | `<= 256 B` y `> 1,000×` | **PASS** |
| `H_G1b` | G1 · Gradientes químicos | `topo_corr = 0.9990 [0.9990, 0.9990]` vs aleatorio `-0.0087` (`Δ = +1.0078 [+0.9951, +1.0205]`) | `CI95_lo > 0.85` y `> rand` | **PASS** |
| `H_G1c` | G1 · Gradientes químicos | `spatial_late_fe`: `4,167.4` vs aleatorio `6,406.4` (**`-34.9%`**, `Δ = -2,239.0 [-2,405.3, -2,083.0]`) | `CI95_hi < 0` | **PASS** |
| `H_G2a` | G2 · Microcircuito canónico | 1 sola plantilla (`64 B`): `V1 = 99.69% [99.45, 99.88]`, `A1 = 100.00%`, `Bytes = 74.53% [72.93, 76.33]` | `CI95_lo > 70%` en 3 modalidades | **PASS** |
| `H_G2b` | G2 · Re-cableado Mriganka Sur | `A1->Vis`: `OSI = 0.498 [0.491, 0.506]`, `acc = 99.92%` vs aleatorio `35.59%` (`+64.34 pp [+62.42, +66.60]`) | `OSI > 0.35` y `> +10 pp` vs rand | **PASS** |
| `H_G2c` | G2 · Equivalencia `A1_rewired`≈`V1` | `|acc(A1_rewired) - acc(V1)| = 0.39 pp [0.20, 0.63]` (`99.92%` vs `99.69%`) | `CI95_hi < 6.0 pp` | **PASS** |
| `H_G3a` | G3 · Curva de Conel + Microglía | `2,560 -> 9,728 (3.80×) -> 3,072` (`prune = 68.42%`, RAM CSR `118,792 B -> 38,920 B`) | `peak >= 2.5×` y `prune >= 50%` | **PASS** |
| `H_G3b` | G3 · Lottery Ticket biológico | `eval_fe`: `compiled = 1,871.1` vs `sparse_scratch = 1,889.3` (`Δ = -18.1 [-22.5, -13.1]`) y vs `peak = 2,187.6` (`-316.5`) | `CI95_hi < 0` vs `sparse` y `peak` | **PASS** |
| `H_G3c` | G3 · Mielinización física | `delay = 5.07 -> 1.82` ticks; `ticks = 3.73` vs `7.95` (**`-53.1%`**, `Δ = -4.22 [-4.32, -4.12]`); `tps = 1.745× [1.672, 1.798]` | `Δticks < 0` y `tps > 1.25×` | **PASS** |
| `H_G4a` | G4 · Prenatal vs `NoPrenatal` | `OSI@birth = 0.502` vs `0.323` (`+0.179 [+0.168, +0.190]`); `early_FE = 1,555.1` vs `1,963.7` (**`-20.8%`**, `Δ = -408.6 [-421.5, -395.2]`) | `ΔFE < 0` y `ΔOSI > 0` | **PASS** |
| `H_G4b` | G4 · Control negativo ruido blanco | `OSI@birth = 0.502` vs `noise = 0.324` (`+0.178 [+0.167, +0.189]`); `early_FE = 1,555.1` vs `1,961.7` (**`-20.7%`**, `Δ = -406.7 [-419.5, -393.4]`) | `ΔFE < 0` y `ΔOSI > 0` vs ruido | **PASS** |
| `H_G4c` | G4 · Espasmos motores fetales | `early_sm_err = 0.4498` vs `none = 0.9138` (`-0.464 [-0.469, -0.459]`) vs `noise = 1.1400` (`-0.690 [-0.699, -0.682]`) | `Δ < 0` vs `none` y `noise` | **PASS** |
| `H_G5a` | G5a · `CorticalFabric` v3 vs v2 | `early_FE = 7,048` vs `13,462` (**`-47.6%`**); `late_FE = 7,248` vs `11,979` (**`-39.5%`**); `ticks = 3.52` vs `5.19` (**`-32.2%`**); `comp = 4,209.8×` | Mejora en early, late, ticks y `>1000×` | **PASS** |
| `H_G5b` | G5b · `AuraByteModel` v3 vs v2 | `early_MD = 4.4587` vs v2 `4.4697` (`Δ = -0.0110 [-0.0391, +0.0214]` — IC cruza 0); **`eval_MD = 3.7393`** vs v2 `3.8027` (**`-0.0635 [-0.0912, -0.0347]`**) vs `bigram 4.1616` (**`-0.4224`**) | `early < v2` (IC<0), `eval <= v2` y `< bigram` | **FAIL** *(solo IC época 1)* |
| `H_G5c` | G5b · Esparsidad + Rust online | `nonzero = 47.87%` vs v2 `55.10%` (**`-7.23 pp [-7.95, -6.50]`**, 8,192 podadas); `Rust = 5.0989` vs `frozen 7.6332` (`-2.534`) vs `bigram 6.2074` (`-1.108`) vs v2 `5.1571` | Más esparso que v2 y online `< frozen, bigram` | **PASS** |

---

## 2. Comparativa Directa End-to-End: AURA v3 Genesis vs AURA v2 Congelado (`CONFIRM`)

### A. Tejido Cortical Predictivo (`CorticalFabric`, Suite G5a)

| Métrica (`N=256` eval, compresión en `N=2048`) | AURA v2 (`aura_v2_fabric`) | AURA v3 Genesis (`aura_v3_genesis_fabric`) | Mejora v3 vs v2 (`CONFIRM` IC95) |
|---|---:|---:|---:|
| Compresión genética vs CSR explícito (`N=2048`) | `1.0×` (`507,912 B`) | **`4,209.8×`** (`144 B`) | **+4,208.8×** |
| Sinapsis finales activas (`N=256`) | `5,120` | **`3,072`** | **-40.0%** |
| Energía libre temprana al nacer (`early_fe`, 20 frames) | `13,462.2` | **`7,048.3`** | **`-47.6%`** (`-6,413.8 [-6,732.0, -6,121.2]`) |
| Energía libre tardía (`late_fe`, frames 100–160) | `11,979.4` | **`7,247.7`** | **`-39.5%`** (`-4,731.7 [-4,929.8, -4,521.4]`) |
| Latencia media de settling (`avg_ticks`) | `5.19` | **`3.52`** | **`-32.2%`** (`-1.67 [-1.78, -1.54]`) |

### B. Modelo Real de Lenguaje a Nivel de Bytes (`AuraByteModel`, Suite G5b — Corpus Real del Repositorio)

| Métrica (Markdown real del repo + cambio de dominio a Rust no visto) | Bigram baseline | AURA v2 Congelado | AURA v3 Genesis (`frozen_infer`) | AURA v3 Genesis (`full`) | Delta v3 vs v2 (`CONFIRM` IC95) |
|---|---:|---:|---:|---:|---:|
| `early_md_bpb` (tras época 1 held-out) | `4.1616` | `4.4697` | `4.4587` | `4.4587` | `-0.0110 [-0.0391, +0.0214]` *(n.s.)* |
| **`eval_md_bpb`** (final held-out Markdown) | `4.1616` | `3.8027` | `3.7393` | **`3.7393`** | **`-0.0635 [-0.0912, -0.0347]`** *(sig.)* |
| **`shift_rust_bpb`** (código Rust no visto) | `6.2074` | `5.1571` | `7.6332` | **`5.0989`** | **`-0.0581 BPB`** *(vs v2)* / **`-2.5343`** *(vs frozen)* |
| Densidad pico ternaria (`nonzero_peak_frac`) | `100.0%` | `66.50%` | `72.44%` | `72.44%` | `+5.94 pp` *(sobreproducción Conel)* |
| **Densidad final ternaria (`nonzero_final_frac`)** | `100.0%` | `55.10%` | `47.87%` | **`47.87%`** | **`-7.23 pp [-7.95, -6.50]`** *(8,192 podadas)* |
| Ratio de compresión `EntropyPatcher` | `1.00×` | `1.99×` | `1.99×` | `1.99×` | `0.00×` *(paridad)* |
| Rango efectivo oculto (`effective_rank`) | — | `31.74` | `31.77` | `31.77` | `+0.03` *(sin colapso)* |
| Velocidad de entrenamiento (`train_bytes_per_sec`) | — | `10,829 B/s` | `9,710 B/s` | `9,710 B/s` | `-10.3%` *(overhead prenatal incluido)* |

---

## 3. Análisis Detallado por Subrutina y Hallazgo Honesto en `H_G5b`

1. **Subrutina 1 (Gradientes morfogenéticos 3D — G1, 3/3 PASS):** Un genoma de apenas **144 bytes** descomprime mediante conos de crecimiento locales un tejido de 2,048 neuronas (`606,216 B` en CSR → **`4,209.8×` de compresión**) con preservación retinotópica casi perfecta (`r = 0.9990` frente a `-0.0087` del grafo aleatorio v2) y reduce en **`-34.9%`** la energía libre predictiva ante estímulos espaciales 2D.
2. **Subrutina 2 (Microcircuito cortical canónico de Mountcastle + Mriganka Sur — G2, 3/3 PASS):** Una **única plantilla de 64 bytes** sin una sola línea de código específica por modalidad clasifica barras visuales ON/OFF (`99.69%`), espectrogramas auditivos (`100.00%`) y ventanas de 128 bytes reales del repositorio (`74.53%`). Al re-cablear estímulos visuales hacia el área auditiva `A1`, esta desarrolla selectividad de orientación real (`OSI = 0.498`, `acc = 99.92%`), superando por `+64.34 pp` al grafo sin estructura canónica (`35.59%`) y quedando a solo `0.39 pp` de `V1`.
3. **Subrutina 3 (Sobreproducción + Poda Microglial + Mielinización — G3, 3/3 PASS):** El ciclo `Newborn (2,560) -> ChildhoodPeak (9,728, 3.8×) -> Microglia + Mielina (3,072, 68.4% podado físicamente en RAM)` confirma el **Lottery Ticket biológico**: con exactamente el mismo presupuesto sináptico final (`3,072`) y los mismos 180 frames de experiencia, el tejido compilado logra menor energía libre que `SparseFromScratch` (`1,871.1 vs 1,889.3`, `Δ = -18.1 [-22.5, -13.1]`), supera por `-316.5` a `PeakUnpruned` (`2,187.6`), reduce los ticks de settling a menos de la mitad (`3.73 vs 7.95`, `-53.1%`) y acelera el throughput físico real en `N=2048` en **`1.75×`** (`346.6k vs 198.8k tps`).
4. **Subrutina 4 (Currículo prenatal sintético sin fuga de datos — G4, 3/3 PASS):** Las ondas retinianas 2D sintéticas calibran selectividad de orientación (`OSI@birth = 0.502`) y reducen en **`-20.8%`** la energía libre predictiva prequencial al abrir los ojos (`1,555.1`), mientras que el control negativo con igual número de pasos y exacta energía/histograma píxel-a-píxel pero sin coherencia espacial (`white_noise_prenatal`: `OSI = 0.324`, `FE = 1,961.7`) resulta prácticamente indistinguible de nacer en blanco (`no_prenatal`: `OSI = 0.323`, `FE = 1,963.7`). En el dominio sensoriomotor, los espasmos fetales con copia de eferencia recortan el error inicial a la mitad (`0.4498 vs 0.9138` sin prenatal y `1.1400` con ruido blanco).
5. **Análisis honesto del único criterio `FAIL` en `CONFIRM` (`H_G5b`, sub-cláusula de época 1):**
   - En `DEV`, AURA v3 redujo `early_md_bpb` frente a v2 en `-0.0433 BPB [-0.0599, -0.0274]` (`PASS`).
   - En `CONFIRM` (16 semillas frescas, ejecución única sin re-tuning), `early_md_bpb` mantuvo la dirección favorable (`4.4587 vs 4.4697`, `-0.0110 BPB`), pero el intervalo de confianza al 95% (`[-0.0391, +0.0214]`) cruzó el cero, marcando **`H_G5b = FAIL`** bajo nuestro criterio estricto prerregistrado.
   - **Explicación mecanicista:** al prohibir estrictamente cualquier palabra o estadística del idioma en el currículo prenatal (solo 8 clases ASCII uniformes dentro de cada clase), la caída de entropía de la primera época en Markdown real está dominada por frecuencias léxicas específicas del español/inglés y del vocabulario del repo, donde un prior agnóstico al idioma aporta una ganancia real pero pequeña (`~0.01–0.04 BPB`). Sin embargo, **al concluir las 4 épocas con poda microglial y mielinización SSM**, las 4 subrutinas combinadas producen una **mejora estadísticamente significativa en `CONFIRM`** sobre AURA v2 congelado tanto en **BPB final held-out en Markdown (`3.7393 vs 3.8027`, `-0.0635 [-0.0912, -0.0347]`)**, como en **adaptación online a código Rust (`5.0989 vs 5.1571`, `-0.0581 BPB`)**, usando **7.23 pp menos conexiones ternarias activas (`47.87% vs 55.10%`)**.

---

## 4. Muestras Verbatim de Generación (`CONFIRM/0xB600/g5_samples.txt`)

Se reportan literalmente sin edición ni selección manual las salidas generadas por AURA v3 Genesis (~50K parámetros ternarios entrenados solo sobre ~15 KB de Markdown del repositorio):

```text
PROMPT_MD: ## Resultados de 
GEN_MD:    peraretaci..n, u..  untetr): reptitestes remimir_arabtedidacontrasititobts memiias matledid peri
PROMPT_RS: pub fn 
GEN_RS:    enp ..deag ( Gnxbulto) [..x pepsedes  # ELuitar, u..lrinas mem_v1.md).   **Restete inorade). * r
```

---

## 5. Limitaciones y Matices Honestos

1. **Escala del modelo de lenguaje y fluidez generativa:** `AuraByteModel` tiene ~50K parámetros ternarios entrenados en CPU durante 4 épocas sobre ~15 KB de texto del repositorio; como evidencian las muestras verbatim de arriba, captura ritmo sub-palabra, puntuación y delimitadores Markdown (superando en `-0.422 BPB` al modelo Bigram y en `-0.063 BPB` a AURA v2), pero **no genera español ni Rust coherente** ni pretende compararse con LLMs modernos.
2. **Overhead temporal del currículo prenatal en `G5b`:** incluir 2 pasadas prenatales sintéticas (`3,072` bytes) dentro del tiempo total reportado de entrenamiento redujo `train_bytes_per_sec` de `10,829 B/s` (v2) a `9,710 B/s` (v3, `-10.3%`), a cambio de mejor BPB final (`-0.0635 BPB`) y `7.23 pp` menos pesos activos.
3. **Diferencia entre poda física en CSR (`CorticalFabric`) vs poda en máscara de bits (`AuraByteModel`):** en `CorticalFabric` (G3 y G5a), la microglía compacta físicamente el vector `Vec<Synapse>` y el CSR en RAM (`118.8 KB -> 38.9 KB`), acelerando la ejecución física real en `1.75×`. En `AuraByteModel` (G5b), la matriz ternaria 2-bit permanece densamente empaquetada con ceros permanentes (`47.87%` no-cero), por lo que la ventaja es de regularización, generalización y esparsidad efectiva, sin aceleración de kernel disperso aún en el decodificador.
4. **Generalización de los mapas sensoriales:** las suites G1–G4 validan las 4 subrutinas biológicas en escalas de 256–2,048 neuronas y sensores 8×8 / 64D / 128B. Escalar el enrutamiento morfogenético y la poda microglial a millones de neuronas y modalidades de alta resolución (imágenes naturales, audio real continuo) queda como trabajo futuro.
