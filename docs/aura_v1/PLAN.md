# Prerregistro: AURA + CorticalFabric v1 (Arquitectura Cognitiva Neuromórfica y de Triple Escala Temporal en Rust)

**Fecha de prerregistro:** 2026-11-20 (antes de implementación y DEV).
**Rama:** `exp/aura-neuromorphic-rust` (creada desde `main` @ `34f04b2`; no modifica `main`, no altera `orbita/`, `gemma_campo/`, `soft_bellman*` ni `agentic_web`).
**Fuente completa archivada:** [`PROPUESTA.md`](PROPUESTA.md), capturada de la página pública `https://gemini.google.com/share/bb4ada4c377a` (enlace corto `https://share.gemini.google/PlhXdw38cTyQ`, título *"Arquitectura Cognitiva Neuromórfica en Rust"*).

---

## 1. Alcance e Implementación Fiel de la Propuesta

El documento fuente propone dos sistemas complementarios en Rust puro sin tokens fijos (BPE) ni multiplicaciones de matrices densas en punto flotante:

1. **Arquitectura AURA (Adaptive Unified Recurrent-Associative Core)** — Workspace Cargo modular de 7 crates (`aura-engine/`), 100% Rust estándar (cero dependencias externas, cero BLAS/MatMul oculto):
   - `aura-math`: `LatentVec` cuantizado `i8` (`LATENT_DIM = 256`, alineado a 64 bytes), distancia L1 de Energía Libre de Laplace, `TernaryMatrix` en planos de bits `u64` (`pos_planes`, `neg_planes`, `shadow_acc`) con proyección `project_zero_mul` (cero instrucciones de multiplicación) y consolidación con histéresis (`consolidate_sleep_weight`), `FastWeightMemory` con regla Delta normalizada entera en punto fijo Q12 (sin floats, sin asignaciones en hot loop, contractiva demostrable), tabla `COS_LUT` Q8 para galga $U(1)$, generador determinista `Xorshift64`, y diagnóstico de **rango efectivo** (`effective_rank`) vía eigendescomposición simétrica de Jacobi para vigilar colapso latente (§6).
   - `aura-perception`: Codificador perceptual sin tokens (*Zero-Token Entropy Patcher*) con predictor causal ligero sobre flujo crudo de bytes (tabla de contextos con hash + `LOG2_LUT` entera), detección dinámica de frontera de parche ($H_{\text{local}} \ge \tau_{\text{corte}}$), acumulación recurrente entera libre de multiplicaciones, control de línea base de ventana fija (`FixedPatcher`), y codificador delta-modulado de eventos de bytes para `CorticalFabric`.
   - `aura-cortex`:
     - **Núcleo Cortical Dual (AURA):** Pesos lentos ternarios (`TernaryMatrix`), estado continuo SSM selectivo con compuerta de tiempo discreta por bit-shifts ($h_t \leftarrow h_{t-1} + (\Phi_{\text{slow}}(z_t) - h_{t-1}) \gg \tau_t$), y memoria asociativa rápida (`FastWeightMemory`, regla Delta online modulada por sorpresa/NE).
     - **Núcleo Neuromórfico Event-Driven (`CorticalFabric`, Parte 1 de la propuesta):** Estructura SoA contigua, grafo disperso CSR, búfer circular de retardos axónicos (`MAX_DELAY = 16`, respetando retardos reales), cero `f32`/`f64` en `step()`, atención por transporte paralelo de fase $U(1)$ (`COS_LUT[(theta_i + A_ij - theta_j) mod 256] >> gain_shift`), Energía Libre de Laplace ($|\varepsilon_j| \gg p_j$), competencia lateral en álgebra tropical max-plus por minicolumna, aprendizaje local de 3 factores con histéresis ternaria y alineación de galga $A_{ij}$, y bucle metacognitivo global (Noradrenalina, Acetilcolina, Dopamina).
   - `aura-world`: Modelo de mundo predictivo en espacio latente estilo JEPA con ensamble de $K=4$ cabezas independientes (`ENSEMBLE_HEADS = 4`), codificador objetivo EMA ($\bar{\theta} \leftarrow 0.996\,\bar{\theta} + 0.004\,\theta$), regularización anti-colapso de Varianza-Covarianza (VICReg entera/híbrida en entrenamiento) y separación explícita entre incertidumbre **epistémica** (desacuerdo L1 por pares entre las $K=4$ cabezas) e incertidumbre **aleatoria** (ruido irreducible con residuo de media cero).
   - `aura-hippocampus`: Memoria episódica indexada mediante grafo navegable disperso L1 tipo HNSW (inserción con grado acotado $M$, búsqueda beam $ef$, serialización binaria bit-exacta) y motor de **Ciclo de Sueño (`SleepConsolidator`)** con replay balanceado (prioridad de sorpresa + preservación estratificada de tareas anteriores) hacia `TernaryMatrix::consolidate_sleep_weight`.
   - `aura-metacog`: Controlador metacognitivo que arbitra entre `System1Reflex` ($\sigma_{\text{epi}} < \tau_{\text{duda}}$), `System2Deliberate` (consulta hipocampal → inyección one-shot en `FastWeightMemory` → rollouts mentales de profundidad $H=2$ minimizando Energía Libre Esperada = distancia al objetivo + riesgo epistémico) y `AbstainAndAsk` (cuando la incertidumbre epistémica mínima tras deliberar supera $\tau_{\text{abstain}}$), acoplado al bucle neuromodulador NE/ACh/DA.
   - `aura-runtime`: Orquestador `AuraAgent`, suite científica de 4 experimentos + microbenchmark de ingeniería Zero-Mul, CLI determinista por semilla y exportación CSV/JSON.

---

## 2. Protocolo Experimental (4 Benchmarks + Ingeniería Zero-Mul)

Se siguen estrictamente las tres etapas propuestas en §5 del documento fuente más la validación de `CorticalFabric` (Parte 1) y los riesgos de §6:

### Semillas, Splits y Regla de Congelación
- **DEV:** 16 semillas independientes `0xA400..=0xA40F`. Permitido calibrar hiperparámetros, umbrales metacognitivos y tasas de aprendizaje exclusivamente sobre DEV.
- **Congelación:** Antes de ejecutar CONFIRM, se congela el código Rust y todos los hiperparámetros en `docs/aura_v1/FROZEN.json` (con SHA-256 de todos los fuentes `.rs`) y se realiza commit.
- **CONFIRM:** 16 semillas frescas independientes `0xB400..=0xB40F`, ejecutadas **una sola vez** con el binario congelado. Ningún ajuste posterior.
- **Unidad estadística:** Las 16 semillas independientes (promedio primero dentro de cada semilla; nunca tratar miles de pasos correlacionados como réplicas independientes). Bootstrap pareado de 10,000 remuestreos, semilla `12345`, $\alpha = 0.05$, percentiles interpolados.

---

### Experimento E1 (Etapa 1A): Flujo Crudo de Bytes Sin Tokens, Cambio de Regla, Estática Aleatoria y Rango Efectivo
- **Flujo:** 2,400 bytes crudos por semilla en 4 fases continuas con reglas generadas por semilla:
  1. **Fase 1 (Régimen A, 0..600):** Protocolo estructurado determinista con mapeo clave→valor oculto $R_A$.
  2. **Fase 2 (Cambio abrupto a Régimen B, 600..1200):** Nueva regla oculta $R_B$ (cambia la ley sin aviso).
  3. **Fase 3 (Estática de TV / Ruido Aleatorio, 1200..1500):** Bytes uniformemente aleatorios (incertidumbre aleatoria irreducible) para probar el problema del televisor con estática (§3.3).
  4. **Fase 4 (Retorno a Régimen A + Régimen C nuevo, 1500..2400):** 600 bytes de retorno a $R_A$ (prueba memoria episódica hipocampal) + 300 bytes de un Régimen C nunca visto (prueba incertidumbre epistémica OOD).
- **Brazos pareados (mismo flujo exacto por semilla):**
  - `aura_full`, `no_fast_weights`, `no_hippocampus`, `fixed_patcher`, `no_vicreg`, `persistence`.
- **Criterios Prerregistrados E1:**
  - **H1a (Primario — Recuperación de Sorpresa Post-Cambio):** En Fase 2 (primeros 40 parches tras el cambio $R_A \to R_B$), el error L1 latente medio de `aura_full` es estrictamente menor que el de `no_fast_weights` en CONFIRM con IC95% superior de `(aura_full - no_fast_weights) < 0`.
  - **H1b (Secundario — Recall Hipocampal):** En Fase 4 (retorno a $R_A$), `aura_full` obtiene menor error de recuperación que `no_hippocampus` (IC95% superior < 0).
  - **H1c (Secundario — Anti-Colapso §6):** El rango efectivo (`effective_rank`) de `LatentVec` en `aura_full` (con VICReg + EMA) supera a `no_vicreg` y permanece $> 8.0$ (sin colapso dimensional).
  - **H1d (Secundario — Compresión sin Tokens):** El *Entropy Patcher* emite menos parches por byte en segmentos repetitivos predecibles que en fronteras/transiciones de cambio (`compression_ratio_pred_vs_trans > 1.2`).
  - **H1e (Exploratorio — Estática vs Epistémica):** La incertidumbre epistémica $\sigma_{\text{epi}}$ en Régimen C no visto supera a la medida en Fase 1 familiar, y la estática de TV (Fase 3) exhibe alta sorpresa pero baja consistencia direccional de error frente a Fase 2.

---

### Experimento E2 (Etapa 1B): Mundo Activo 2D con Cambio de Ley Física + Metacognición Sistema 1 / Sistema 2 / Abstain
- **Entorno:** Mundo discreto $6 \times 6$ codificado en tramas de bytes crudos, 4 acciones direccionales `{N, S, E, W}` + acción metacognitiva `AbstainAndAsk`:
  - **Fase 1 (Ley $L_1$, 24 episodios):** Navegación hacia meta con obstáculos específicos de la semilla; entrena modelo de mundo $K=4$ y política refleja Sistema 1.
  - **Fase 2 (Cambio abrupto de física $L_1 \to L_2$, 24 episodios):** Permutación oculta de ejes de acción + desplazamiento de obstáculo. Evalúa adaptación online de Pesos Rápidos + planificación Sistema 2 por Energía Libre Esperada (rollout $H=2$).
  - **Fase 3 (Sondas de Incertidumbre OOD / Trampas Ambiguas, 16 episodios):** Mezcla de estados familiares (resueltos rápido en Sistema 1), estados de ley cambiada (resueltos en Sistema 2) y estados trampa OOD genuinamente ambiguos donde una acción ciega errónea penaliza `-10`, mientras `AbstainAndAsk` cuesta `-1` y recibe la guía segura.
- **Brazos pareados:** `aura_full`, `system1_only` (sin Sistema 2 ni abstención), `no_fast_weights` (Sistema 2 pero pesos rápidos congelados), `random_control`.
- **Criterios Prerregistrados E2:**
  - **H2a (Primario — Ventaja Metacognitiva + Adaptación):** En Fases 2+3 combinadas, el retorno total por episodio de `aura_full` supera tanto a `system1_only` como a `no_fast_weights` en CONFIRM con IC95% inferior de las diferencias $> 0$.
  - **H2b (Secundario — Calibración de Abstención):** En Fase 3, `aura_full` abstiene con tasa significativamente mayor en estados trampa OOD ambiguos que en estados familiares (evitando alucinación ciega sin paralizar el Sistema 1).

---

### Experimento E3 (Etapa 2): Validación del Ciclo de Sueño (Anti-Olvido en Planos de Bits Ternarios)
- **Protocolo exacto de §5 Etapa 2:**
  - **Tarea A** (64 ejemplos train / 128 test disjunto `Test_A`) y **Tarea B** (64 ejemplos train / 128 test disjunto `Test_B`) con subespacio de entrada parcialmente solapado (interferencia real si se sobrescribe).
  - **Paso 1:** Consolidar Tarea A en `W_slow` (`TernaryMatrix`, 2 bits/peso) y almacenar episodios salientes en `EpisodicHippocampus`. Medir `acc_A_pre` sobre `Test_A` con `FastWeightMemory = 0`.
  - **Paso 2:** Exponer **solo** Tarea B (cero acceso al dataset de A) y ejecutar consolidación/sueño según brazo:
    - `aura_sleep_full`: Replay balanceado del Hipocampo + consolidación en `TernaryMatrix` con histéresis (`HI=96, LO=24`).
    - `ablation_no_replay`: Solo Tarea B con histéresis, sin replay hipocampal.
    - `ablation_no_hysteresis`: Con replay hipocampal, sin histéresis (`HI=1, LO=0`).
    - `ablation_naive`: Sin replay y sin histéresis.
  - **Paso 3:** Poner a cero `FastWeightMemory` y evaluar `W_slow` puro (`project_zero_mul`) en `Test_A` y `Test_B` no vistos.
- **Criterios Prerregistrados E3:**
  - **H3a (Primario — Anti-Olvido en Pesos Ternarios):** La precisión combinada $\frac{1}{2}(\text{acc}_{A,\text{post}} + \text{acc}_{B,\text{post}})$ de `aura_sleep_full` supera a `ablation_no_replay` en CONFIRM (IC95% inferior > 0), y el olvido `forgetting_A = acc_A_pre - acc_A_post` es estrictamente menor que en `ablation_no_replay` (IC95% superior < 0).
  - **H3b (Secundario — Contribución de la Histéresis):** Contraste pareado `aura_sleep_full` vs `ablation_no_hysteresis` en `forgetting_A` y precisión combinada (se reporta honestamente sea positivo, nulo o negativo).

---

### Experimento E4 (Parte 1 de la Propuesta): Núcleo Event-Driven `CorticalFabric` ($U(1)$ + Laplace FE + 3 Factores + NE/ACh/DA)
- **Red:** 256 neuronas en 3 láminas corticales jerárquicas (Gamma sensorial `freq=17`, Beta asociativa `freq=9`, Theta apical `freq=4`), grafo CSR disperso, retardos axónicos reales `1..8`, inhibición lateral tropical por minicolumnas, 0 floats en `step()`.
- **Estímulo:** Flujo de 1,200 ticks de bytes crudos delta-modulados con motivo estructurado periódico (ticks 0..600) y cambio abrupto de motivo en tick 600 (ticks 600..1200).
- **Brazos pareados:** `cortical_full`, `cortical_frozen` (sin plasticidad 3-factores), `cortical_no_phase_gate` (sin compuerta de coherencia $U(1)$), `cortical_no_neuromod` (NE/ACh/DA fijos).
- **Criterios Prerregistrados E4:**
  - **H4a (Primario — Minimización Local de Energía Libre):** La Energía Libre de Laplace media en el último tercio de cada régimen de `cortical_full` es estrictamente menor que la de `cortical_frozen` en CONFIRM (IC95% superior < 0), manteniendo actividad cortical dispersa (entre 1% y 25% de neuronas activas por tick, sin colapso silencioso ni avalancha).
  - **H4b (Secundario — Respuesta Metacognitiva NE):** En los 30 ticks posteriores al cambio de motivo (tick 600), la Noradrenalina media supera a los 30 ticks previos en `cortical_full` (IC95% inferior > 0).
  - **H4c (Exploratorio — Compuerta $U(1)$ y Neuromodulación):** Contrastes pareados `cortical_full` vs `cortical_no_phase_gate` y `cortical_no_neuromod`.

---

### Verificación de Ingeniería (E5): Cero Multiplicaciones, Densidad de Memoria y Reproducibilidad
- Exactitud algebraica bit-a-bit de `TernaryMatrix::project_zero_mul` frente a producto denso `i32` (`max_abs_diff == 0`).
- Contracción demostrable de `FastWeightMemory` normalizada entera (error decrece monótonamente en 1–3 actualizaciones sin divergir).
- Reducción de memoria 16× (`2 bits/peso` vs `32 bits/peso f32`) y benchmark de latencia/throughput CPU (`256×256` y `1024×1024`) en `--release`.
- Reproducibilidad bit-a-bit: re-ejecución independiente de la semilla `0xB400` produce exactamente las mismas métricas (descartando columnas de tiempo de pared) verificada en `replay.json`.

---

## 3. Límites Honestos Declarados de Antemano
1. **No es un LLM de lenguaje natural fluido ni sustituye a BQ10/ATLAS-FW/Transformers:** Todos los entornos E1–E4 son bancos sintéticos cerrados y controlados de tamaño compacto, exactamente como recomienda la Etapa 1 y 2 de §5 de la propuesta. No se evaluaron `known_acc16`, `E33`, Paso 0 `1985bdbad5eee41f`, ni datasets de texto humano masivo.
2. **Backend GPU (Etapa 3):** Este entorno cuenta con 2 vCPUs Intel Xeon Platinum 8481C sin GPU física; por tanto la aceleración GPU WGSL se documenta como extensión arquitectónica sobre los traits de `aura-world` / `aura-math` (etapa futura), mientras que el núcleo CPU zero-mul y el motor event-driven sí se miden y verifican nativamente en este hardware.
3. **Honestidad ante negativos:** Si alguna ablación (por ejemplo `no_hysteresis`, `fixed_patcher`, `cortical_no_phase_gate` o `no_vicreg`) no muestra mejora o resulta mejor que el brazo completo, se reportará explícitamente sin ocultar datos ni modificar CONFIRM.
4. **Higiene de recursos:** Compilación y ejecución con `nice -n 10`, máximo 2 procesos paralelos (en línea con 2 vCPUs disponibles), sin dejar tareas en segundo plano.
