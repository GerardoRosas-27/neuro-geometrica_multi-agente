# Trazabilidad Científica — AURA v3 Genesis (`exp/aura-v3-genesis`)

## 1. Cadena de Commits y Aislamiento de Ramas

- **`main` (intacto):** `34f04b2` — línea base del repositorio multi-agente neuro-geométrico.
- **`exp/aura-v2-training` (intacto):** `c8d5cc0` — AURA v2 + CorticalFabric v2 (6 cuellos de botella B1–B6 resueltos, 17/17 PASS en `CONFIRM`).
- **Rama experimental actual:** `exp/aura-v3-genesis` creada desde `c8d5cc0` sin modificar `main`, `exp/aura-v2-training` ni `docs/aura_v2/`.
- **Prerregistro previo al código (`PLAN.md` inmutable):** `fece4e6` (`docs/aura_v3/PROPUESTA_GENESIS.md` y `docs/aura_v3/PLAN.md`, 15 criterios `H_G1a`–`H_G5c`).
- **Congelado pre-`CONFIRM` (`FROZEN.json`):** commit actual en `exp/aura-v3-genesis` con `sources_sha256 = cc90b95faa8a1885e308b81270a94babcb57b624a7323599edb8b38f0f366171` (22 archivos `.rs` en `aura-engine/crates/`), 15/15 unit tests PASS en modo `debug` y `release` con 0 warnings de compilación, y 15/15 PASS en fase `DEV` (`0xA600–0xA60F`).

## 2. Arquitectura Añadida (Zero-External-Deps Rust)

1. Nuevo crate `aura-genesis` (`aura-engine/crates/aura-genesis/src/lib.rs`, 9.º crate del workspace):
   - **Subrutina 1 — `MorphogenGenome` (144 B) + `MorphogenRouter`:** gradientes morfogenéticos 3D (`Shh`, `Wnt`, `BMP`, `EphA/ephrin-A`, `EphB/ephrin-B`) y navegación autónoma local del cono de crecimiento (`decompress_fabric`, `measure_retinotopic_correlation`).
   - **Subrutina 2 — `CanonicalColumnTemplate` (64 B) + `CanonicalCortex`:** plantilla cortical canónica universal de 6 capas (`L1`, `L2/3`, `L4`, `L5`, `L6`) con inhibición lateral k-WTA ON/OFF, plasticidad Oja no supervisada, acoplamiento horizontal morfogenético, `measure_osi` sobre curvas de sintonía sin clipping y `predictive_free_energy` top-down `L2/3 -> L4`.
   - **Subrutina 3 — `DevelopmentalPruner::microglia_prune_and_myelinate`:** curva de Jesse Conel (`Newborn -> ChildhoodPeak -> CompiledMyelinated`), poda física de `Vec<Synapse>` con re-compactación de CSR en RAM (equilibrada por lámina FF/FB y alineamiento apical positivo), mielinización saltatoria (`axon_delay -> 1` tick) y escalado homeostático tipo Turrigiano (`gain_shift = 3`).
   - **Subrutina 4 — `PrenatalCurriculum`:** ondas retinianas viajeras 2D coherentes, control negativo de ruido espacialmente barajado con histograma/energía idénticos por frame (`generate_shuffled_noise_frames`), espasmos motores fetales con copia de eferencia espinal (`encode_efference_copy`, `simulate_fetal_body_echo`, `run_fetal_motor_twitches`) y ondas estructurales sintéticas ASCII de 8 categorías universales (`generate_prenatal_byte_waves`, 0 bytes del repositorio).
2. Extensiones retrocompatibles (0 cambios de conducta en v2):
   - `aura-math`: máscara permanente de poda microglial `prune_mask`, `nonzero_fraction()` y `microglia_prune_ternary()` en `ScaledTernaryLayer`.
   - `aura-cortex`: `myelin_delay_enabled: bool` (por defecto `false` para preservar bit a bit a v2), `axon_delay`, `syn_usage`, `from_custom_graph`, `csr_memory_bytes`, `clamp_sensory_pattern` y `settle_pattern_amortized` respetando la latencia física de conducción axonal.
   - `aura-perception`: `EntropyPatcher::reset_stream_keep_vicreg()` para conservar calibración VICReg prenatal al cruzar el nacimiento sin arrastrar contexto de bytes sintéticos.
   - `aura-decoder`: `AuraByteModel::new_genesis`, `run_prenatal_structural_epoch` (8 categorías ASCII universales sin vocabulario del repo), `train_corpus_v2_tracked` y `train_corpus_genesis`.
   - `aura-runtime`: suites `bench_g1.rs`–`bench_g5.rs` y subcomando CLI `aura-v2 genesis`.

## 3. Registro Transparente de Calibraciones en Fase `DEV` (`0xA600–0xA60F`)

Todas las calibraciones se ejecutaron **exclusivamente** en `DEV` (`0xA600–0xA60F`), sin tocar jamás ninguna semilla de `CONFIRM` (`0xB600–0xB60F`):

1. **Métrica `OSI` sin sesgo de clipping (`aura-genesis`):** la versión inicial medía respuestas post-k-WTA recortadas (inflando el `OSI` en blanco a `0.761`). Se corrigió para medir directamente las curvas de sintonía sin clipping `|w_i · x_norm|` con plasticidad congelada (`OSI` en blanco `~0.33`, tras ondas retinianas `~0.50`).
2. **Estabilidad apical y balance FF/FB en poda microglial (`aura-genesis`):** `exc_init_prob = 0.44` y `gain_fb_shift = 1` saturaban la capa oculta y provocaban sobreimpulso apical en píxeles oscuros. Se ajustó a `exc = 0.20, inh = 0.16, gain_ff = 2, gain_fb = 2`, poda microglial separada por lámina FF/FB, puntuación por alineamiento positivo `ApicalFeedback` y escalado homeostático `gain_shift = 3` en axones mielinizados.
3. **Separación real en G2 (`aura-genesis`, `bench_g2.rs`):** barras de polaridad única eran linealmente separables al 100% por características aleatorias y ventanas de 64B confundían Markdown/Rust. Se adoptaron barras con polaridad ON/OFF (`±1`) + clutter `0.46`, readout L5 eficiente en muestras tras autoorganización no supervisada y ventanas de 128B (`V1 = 99.7%`, `A1 = 100.0%`, `Bytes = 77.0%`, `Sur = 99.8%` vs `Random = 35.9%`).
4. **Consolidación adolescente post-poda en G3 (`bench_g3.rs`):** en la primera corrida `DEV` (`11/15`), `compiled_myelinated` ya mejoraba en media a `sparse_from_scratch` (`1,901.9 vs 1,908.3`), pero recibía 0 frames de consolidación tras podar en el frame 180. Se dividió exactamente el mismo presupuesto de 180 frames en `150` infancia + `30` consolidación adolescente para todos los brazos (`fe_vs_sparse_scratch = -17.78 [-24.51, -11.27]`).
5. **Control negativo sin confusor DC en G4 (`aura-genesis`, `bench_g4.rs`):** el ruido blanco uniforme `[0,1]` tenía brillo medio `0.50` frente a `0.22` de las ondas retinianas (confusor DC 2.3×) y los estímulos de prueba tenían fondo gris `0.325`. Se reemplazó el control negativo por permutación espacial exacta píxel-a-píxel (`generate_shuffled_noise_frames`, misma energía/histograma por frame, cero coherencia) y se evaluó la energía libre predictiva prequencial `L2/3 -> L4` directamente en `CanonicalCortex::predictive_free_energy` (entrada centrada de norma unitaria, 0 confusor DC: `1,556.8` vs `noise = 1,959.2` vs `none = 1,960.4`).
6. **Latencia física axonal en `settle_pattern_amortized` (`aura-cortex`):** cuando `myelin_delay_enabled = true`, el voltaje apical residual del frame anterior permitía salida temprana en el tick 3 en tejidos no mielinizados (`axon_delay = 3`) antes de completar un ciclo físico de conducción. Se exigió `eff_min_k = min_k.max(mean_sensory_delay + 1)` únicamente cuando `myelin_delay_enabled = true` (0 impacto en v2).
7. **Currículo prenatal estructural sin vocabulario del repo en G5b (`aura-decoder`, `aura-perception`, `bench_g5.rs`):** se descartó preentrenamiento en 256 bytes con pseudopalabras (sesgo de ortografía inventada) y se reemplazó por 2 pasadas sobre `3,072` bytes sintéticos universales prediciendo las 8 categorías ASCII estructurales (0 bytes del repo), preservando estadísticas VICReg al nacer (`reset_stream_keep_vicreg`), aplicando únicamente el log-prior escalar uniforme-dentro-de-clase `log((P_prenatal(c) + eps) / N_c)` en `l2.bias` (idéntico para las 26 letras minúsculas, 26 mayúsculas y 10 dígitos, preservando ruptura de simetría de rango completo en `l2.shadow_w`) y emparejando el orden de chunks postnatales con AURA v2.

## 4. Estado Pre-`CONFIRM`

- **SHA-256 congelado (`docs/aura_v3/FROZEN.json`):** `cc90b95faa8a1885e308b81270a94babcb57b624a7323599edb8b38f0f366171`
- **Unit tests (`tests.log` + `tests_release.log`):** 15/15 PASS (`debug` y `release`), 0 warnings.
- **`DEV` (`0xA600–0xA60F`, `summary_dev.json`):** **15/15 PASS**.
- **`CONFIRM` (`0xB600–0xB60F`):** Pendiente de ejecución única con binario congelado `--skip-build` inmediatamente después de este commit.
