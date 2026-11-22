# Plan Preregistrado — AURA v3 Genesis (`exp/aura-v3-genesis`)

**Fecha de preregistro:** Previo a escribir cualquier código de `aura-genesis` o modificaciones v3.
**Rama base congelada:** `exp/aura-v2-training` (`c8d5cc0`, `main@34f04b2` intacto).
**Objetivo:** Integrar de forma algorítmica real (no cosmética) las **4 Subrutinas Generativas del ADN** en `aura-engine` (nuevo crate `aura-genesis` + integración en `aura-cortex`, `aura-decoder` y `aura-runtime`) y demostrar con rigor estadístico (16 semillas `DEV` + ejecución única congelada en 16 semillas frescas `CONFIRM`, bootstrap pareado 95%, controles negativos explícitos y replay bit-a-bit) qué mejora cada subrutina y qué mejora la integración completa **AURA v3 Genesis vs AURA v2**.

---

## 1. Diseño Arquitectónico (9º Crate `aura-genesis`, 0 dependencias externas)

1. **Subrutina 1 — `MorphogenRouter` (Enrutamiento 3D por Gradientes Químicos, sin direcciones explícitas):**
   - `MorphogenGenome` compacto (**≤ 256 bytes** serializados) codifica gradientes morfogenéticos 3D `(x, y, z) ∈ [0, 1]^3` (Shh, Wnt, BMP, EphA/ephrin-A, EphB/ephrin-B) + reglas de navegación local del cono de crecimiento (*chemoattraction*, *chemorepulsion*, paso, radio de captura y presupuesto de ramificación).
   - Cada axón navega localmente por gradiente químico sin almacenar IDs de destino; el retardo axonal `delay` deriva físicamente de la distancia 3D recorrida.
   - Descomprime un `CorticalFabric` completo de `N = 2,048` neuronas (`~40k–55k` sinapsis) logrando compresión genética extrema (**> 1,000×** frente a CSR explícito) y **preservación topográfica retinotópica** (`r_topography > 0.85` frente a `~0.0` del cableado aleatorio v2).

2. **Subrutina 2 — `CanonicalColumnTemplate` y `CanonicalCortex` (Microcircuito Cortical Canónico de 6 Capas + Re-cableado tipo Mriganka Sur):**
   - Una única plantilla laminar vertical (`L1, L2/3, L4, L5, L6`, 80% excitadoras / 20% interneuronas inhibitorias locales, ~96 bytes de definición) se instancia masivamente (`M` minicolumnas) sobre el tejido 3D y se acopla lateralmente vía `MorphogenRouter`.
   - **Prueba de re-cableado cross-modal (Mriganka Sur, MIT):** Exactamente el mismo circuito canónico (0 código específico por modalidad, misma plantilla genética) procesa 3 modalidades distintas conectando solo los cables sensoriales a `L4`:
     - **Visión 2D:** barras orientadas (`0°, 45°, 90°, 135°` en retina 8×8) → mide índice de selectividad de orientación (**OSI**) y precisión de clasificación.
     - **Audición:** barridos tonotópicos tiempo-frecuencia (ascendente, descendente, armónico, pulso) → mide selectividad y precisión.
     - **Re-cableado óptico → corteza auditiva (`A1_rewired_visual`):** la misma región cortical `A1` recibe entrada visual y desarrolla selectividad de orientación (OSI) y precisión comparables a `V1`, superando ampliamente a un grafo aleatorio sin microcircuito canónico laminar.

3. **Subrutina 3 — `DevelopmentalPruner` (Sobreproducción + Poda Microglial + Mielinización Física — Curva de Jesse Conel):**
   - Ciclo ontogenético en 3 estadios verificables:
     - `Newborn` (escasa inicial, ~10 sinapsis/neurona) →
     - `ChildhoodPeak` (sobreproducción hiperdensa ~3.5–4×, ~36–42 sinapsis/neurona) →
     - `AdolescenceCompiled` (experiencia + **poda física microglial** eliminando `≥ 55%` de las sinapsis menos útiles del `Vec<Synapse>` y compactando `row_ptr` + **mielinización** de rutas activas: reducción física del retardo axonal `delay -> 1` tick, reordenamiento contiguo en CSR y refuerzo de ganancia).
   - **Control crítico (Lottery Ticket Biológico):** Comparación frente a:
     1. `OverproducedUnpruned` (pico infantil sin podar — alto costo computacional/RAM),
     2. `SparseFromScratch` (inicializado directamente con el **mismo número exacto de sinapsis finales** que el tejido podado, sin fase de sobreproducción).
   - También se aplica sobreproducción → poda microglial (esparsidad ternaria estructural real) + mielinización de canales SSM en `AuraByteModel` v3.

4. **Subrutina 4 — `PrenatalCurriculum` (Datos Sintéticos Prenatales en el Útero — Cero Fuga de Datos):**
   - Ejecutado **exclusivamente antes de ver el primer dato real**, usando 0 bytes del corpus o tareas de evaluación:
     - `RetinalWaveGenerator`: ondas viajeras 2D coherentes (*starburst amacrine waves*) con frentes suaves, ángulos y velocidades variables.
     - `MotorTwitchGenerator`: espasmos motores espontáneos con copia de eferencia y retroalimentación propioceptiva sintética.
     - `PrenatalByteWaveGenerator`: secuencias rítmicas sintéticas, simetrías y delimitadores jerárquicos puros (sin vocabulario del repositorio) para calibrar canales multi-escala SSM + `EntropyPatcher`.
   - **Control negativo obligatorio:** `WhiteNoisePrenatal` (mismo número exacto de pasos prenatales pero con ruido blanco sin coherencia espacio-temporal) y `NoPrenatal` (nacimiento en blanco), demostrando que la ganancia proviene de la **coherencia espacio-temporal** y no de "más pasos de cómputo".

5. **Integración End-to-End `G5` — AURA v3 Genesis vs AURA v2 Congelado:**
   - `G5a` (`CorticalFabric` v3 vs v2): combina las 4 subrutinas (morfogénesis + columnas canónicas + sobreproducción/poda/mielinización + ondas prenatales) y evalúa energía libre temprana (arranque al nacer), energía libre tardía, ticks de settling y throughput `tps` frente a `CorticalFabric` v2.
   - `G5b` (`AuraByteModel` v3 vs v2 en corpus real de `main`): compara AURA v3 Genesis (inicialización morfogenética/canónica + currículo prenatal rítmico sintético + sobreproducción→poda microglial en pesos 2-bit + mielinización SSM) contra **AURA v2 congelado** en:
     - **Eficiencia muestral temprana** (BPB en Markdown held-out tras 1 época / arranque temprano),
     - **BPB final en Markdown held-out** (`README.md` + `docs/RESULTADOS_CONSOLIDADOS.md`),
     - **Shift online a Rust no visto** (`orbita/src/model.rs` + `orbita/src/fam.rs`),
     - **Esparsidad estructural / parámetros activos** tras poda microglial.

---

## 2. Splits de Semillas y Protocolo Estadístico

- **`DEV` (calibración):** 16 semillas `0xA600..=0xA60F`.
- **Freeze:** `docs/aura_v3/FROZEN.json` (SHA-256 de todos los `.rs` del workspace) fijado en commit **antes** de ejecutar `CONFIRM`.
- **`CONFIRM` (ejecución única congelada):** 16 semillas frescas `0xB600..=0xB60F` con `--skip-build` verificando `FROZEN.json`.
- **Inferencia estadística:** Intervalos de confianza bootstrap pareados al 95% (4,000 remuestreos deterministas) sobre las 16 semillas.
- **Replay determinista:** Verificación bit-a-bit en `0xB600` (`docs/aura_v3/replay.json`) excluyendo únicamente métricas de tiempo de pared (`us`, `ms`, `tps`, `bytes_per_sec`).

---

## 3. Los 15 Criterios Preregistrados (`H_G1a`–`H_G5c`)

Todos los criterios se evalúan sobre el `CI95%` bootstrap pareado en `CONFIRM` (N = 16 semillas):

### Suite G1 — Subrutina 1: Enrutamiento por Gradientes Químicos
- **`H_G1a` (Compresión genética > 1,000×):** `genome_bytes <= 256` y `compression_ratio_vs_csr > 1000.0` (descomprimiendo `N = 2,048` neuronas sin almacenar aristas explícitas).
- **`H_G1b` (Preservación retinotópica real):** Correlación topográfica `topo_corr` del grafo morfogenético `CI95_lo > 0.85` y diferencia pareada frente a cableado aleatorio v2 `CI95_lo > 0.80`.
- **`H_G1c` (Ventaja funcional en estímulos espaciales):** Energía libre predictiva del `CorticalFabric` morfogenético menor que la del grafo aleatorio v2 con igual presupuesto sináptico (`fe_morpho - fe_random`: `CI95_hi < 0.0`).

### Suite G2 — Subrutina 2: Microcircuito Cortical Canónico (Re-cableado Mriganka Sur)
- **`H_G2a` (Universalidad cross-modal con 1 sola plantilla):** Con la misma plantilla canónica de 6 capas (0 código específico por modalidad), `acc_visual` y `acc_audio` y `acc_bytes` superan ampliamente el azar (`25%`), con `CI95_lo > 70.0%` en las tres modalidades.
- **`H_G2b` (Re-cableado Mriganka Sur `A1 -> Visual`):** Al redirigir cables visuales hacia el área auditiva `A1`, `A1_rewired_visual` desarrolla selectividad de orientación real (`osi_rewired`: `CI95_lo > 0.35`) y supera al grafo aleatorio sin estructura canónica en precisión visual (`acc_rewired - acc_random`: `CI95_lo > 10.0 pp`).
- **`H_G2c` (Equivalencia funcional `A1_rewired` vs `V1`):** La diferencia absoluta de precisión visual entre `A1_rewired_visual` y `V1_canonical` es pequeña (`|acc_rewired - acc_v1|`: `CI95_hi < 6.0 pp`), confirmando que la especialización proviene del cableado de entrada y no de código distinto.

### Suite G3 — Subrutina 3: Sobreproducción + Poda Microglial + Mielinización (Curva de Conel)
- **`H_G3a` (Curva de Conel y poda física real):** `synapses_peak >= 2.5 * synapses_newborn` y `prune_frac >= 0.50` (`synapses_compiled <= 0.50 * synapses_peak`), con compactación física real del CSR en RAM.
- **`H_G3b` (Lottery Ticket biológico — supera a `SparseFromScratch` a igualdad de sinapsis):** El tejido compilado (`Overproduce -> Prune + Myelin`) logra menor energía libre predictiva que `SparseFromScratch` con igual presupuesto sináptico final (`fe_compiled - fe_sparse_scratch`: `CI95_hi < 0.0`) y supera o iguala a `PeakUnpruned` con menos de la mitad de sinapsis.
- **`H_G3c` (Aceleración física por poda + mielinización):** `CompiledMyelinated` logra menor latencia de settling en ticks (`ticks_compiled - ticks_peak`: `CI95_hi < 0.0`) y mayor throughput `tps` que `PeakUnpruned` (`tps_ratio`: `CI95_lo > 1.25×`).

### Suite G4 — Subrutina 4: Datos Sintéticos Prenatales (Ondas Retinianas y Espasmos Motores)
- **`H_G4a` (Calibración prenatal al nacer vs `NoPrenatal`):** Tras ondas retinianas + espasmos motores (0 datos reales), la energía libre temprana (primeras 20 ventanas reales) es menor que `NoPrenatal` (`early_fe_waves - early_fe_none`: `CI95_hi < 0.0`) y la selectividad topográfica al paso 0 real (`osi_at_birth`) es mayor (`osi_waves - osi_none`: `CI95_lo > 0.0`).
- **`H_G4b` (Control negativo — coherencia vs ruido blanco):** `PrenatalWaves` supera a `WhiteNoisePrenatal` (mismos pasos sin coherencia espacio-temporal) tanto en `early_fe` (`CI95_hi < 0.0`) como en `osi_at_birth` (`CI95_lo > 0.0`).
- **`H_G4c` (Calibración sensoriomotora por espasmos fetales):** El error temprano de predicción sensoriomotora (copia de eferencia acción→sensación) es significativamente menor con espasmos motores prenatales que sin ellos y que con ruido blanco (`sm_err_twitch - sm_err_none`: `CI95_hi < 0.0` y `sm_err_twitch - sm_err_noise`: `CI95_hi < 0.0`).

### Suite G5 — Integración End-to-End: AURA v3 Genesis vs AURA v2 Congelado
- **`H_G5a` (`CorticalFabric` v3 Genesis vs v2):** AURA v3 (4 subrutinas juntas) logra menor energía libre temprana al nacer (`CI95_hi < 0.0`), menos ticks de settling (`CI95_hi < 0.0`) y menor o igual energía libre tardía que `CorticalFabric` v2 (`fe_late_v3 - fe_late_v2`: `CI95_hi <= 0.0`), manteniendo compresión genética `> 1,000×`.
- **`H_G5b` (`AuraByteModel` v3 Genesis — arranque temprano y generalización real en Markdown):** Con las 4 subrutinas (incluyendo currículo prenatal sintético sin 1 solo byte del repo + poda microglial + mielinización SSM), AURA v3 supera a AURA v2 congelado en BPB de arranque temprano (`early_md_bpb_v3 - early_md_bpb_v2`: `CI95_hi < 0.0`) y logra BPB final held-out en Markdown inferior o igual a v2 y estrictamente mejor que `bigram` (`eval_md_bpb_v3 - bigram`: `CI95_hi < 0.0`).
- **`H_G5c` (`AuraByteModel` v3 Genesis — esparsidad microglial + adaptación online Rust):** AURA v3 alcanza mayor esparsidad estructural (menos conexiones 2-bit no-cero activas que v2 tras poda microglial: `nonzero_v3 - nonzero_v2`: `CI95_hi < 0.0`) manteniendo adaptación online superior en código Rust no visto (`shift_rust_v3 - frozen_v3`: `CI95_hi < 0.0` y `shift_rust_v3 < bigram_rust`).
