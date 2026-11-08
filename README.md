> **Arquitectura Ganadora: BQ10 (bio-quantum v10)** = núcleo ATLAS-FW (M1 decay + M2 momentum + M6 dual-rate timescale) + M7 DA-STR (regularización ridge por grado `da=10`) ejecutado sobre AMD Radeon RX 7800 XT (WGPU/Vulkan).
>
> ## Estructura de las Líneas de Investigación en el Repositorio
>
> 1. **Línea Principal: Sustrato de Campo Continuo Token a Token (`gemma_campo`)**
>    - **Ramas:** `exp/gemma-campo-chat`, `exp/gemma-campo-infinito`
>    - **Objetivo:** Memoria continua latente durante la generación del LLM, acoplando innovación causal de embeddings $e(x_t)$ y subespacio contextual $V_{\text{ctx}}$ ($k_{\text{skip}}=16$).
>    - **Resultados Test (3h GPU AMD RX 7800 XT):** $\Delta\text{NLL}_{\text{campo}-\text{oracle}} = \mathbf{+0.0186} \le 0.05$ (**Pasa H2**), PPL = **5.661** vs 5.554 del LLM intacto, $z$-MSE = **0.196** (vs 1.239 de repeat), olvido catastrófico **+0.064%** ($\le 5\%$). Pasa H1, H1k, H2, H3, OLV, EST, REP.
>    - **Reporte:** [`docs/gemma_campo_chat/run_3h_bq10/RESULTADOS.md`](docs/gemma_campo_chat/run_3h_bq10/RESULTADOS.md).
>
> 2. **Segunda Línea de Investigación: Campo de Estado Continuo por Turno (`campo_estado_turno`)**
>    - **Rama:** `exp/campo-estado-turno`
>    - **Objetivo:** Dinámicas latentes a nivel de turno de diálogo completo ($z_t \in \mathbb{R}^{64}$, $F=2721$ modos de Lie) acopladas a decodificador soft-prompt $z \to P \cdot H$ con LLM congelado.
>    - **Resultados Test (3h GPU AMD RX 7800 XT):** $z$-MSE = **0.368** (supera a MLP 0.387 y repeat 1.092, **Pasa T1 y T2**). NLL cae de 1.946 a **0.628** (PPL cae de 7.00 a **1.87**, **Pasa T3**), supera al control barajado por −0.412 nats (**Pasa T4**). Pesos del LLM 100% intactos.
>    - **Reporte:** [`docs/campo_estado_turno/run_3h_bq10/RESULTADOS.md`](docs/campo_estado_turno/run_3h_bq10/RESULTADOS.md).
>
> 3. **Tercera Línea de Investigación: Soft-Bellman Simplicial 3D en GPU (`soft_bellman_simplicial_v3_gpu`)**
>    - **Ramas:** `exp/soft-bellman-simplicial-v2`, `exp/soft-bellman-v1`
>    - **Objetivo:** Planificación y geodésicas en complejos tetraédricos sobre GPU AMD RX 7800 XT (WGPU/Vulkan) con optimización baricéntrica continua Snell-Fermat, Hodge $\Delta_1$, energía Regge y barrera Cayley-Menger.
>    - **Resultados CONFIRM:** Paridad GPU vs CPU analítico ($< 1.61\times 10^{-6}$ error, 18.09 ms). Reducción de costo continuo de **−13.24%** vs oráculo grueso y **−2.47%** vs refinado; regret **0.0091%** con 100% de éxito. En ley cambiada, `v3_adapted` reduce sobrecosto a **0.1355%** (**−9.15 pp**, 68.5× menor error que v2) y zero-shot logra **4.96%** (supera a euclídeo 5.28%).
>    - **Reporte:** [`docs/soft_bellman_simplicial_v3_gpu/RESULTADOS_V3_GPU.md`](docs/soft_bellman_simplicial_v3_gpu/RESULTADOS_V3_GPU.md).
>
> 4. **Cuarta Línea de Investigación: Bio-Quantum Engine x50 Masivo en GPU (`bio_quantum_engine`)**
>    - **Rama:** `exp/bq10-gpu-amd`
>    - **Objetivo:** Escalado masivo de constelaciones cuántico-biológicas en AMD Radeon RX 7800 XT (WGPU/Vulkan).
>    - **Resultados x50:** 1,638,400 episodios en 86.80 s (**18,874.7 eps/s**), 1,024 nodos, 4,096 sinapsis con 0 violaciones de Ley de Dale y consolidación circadiana SWR $\Delta R = -0.00042$.
> 5. **Quinta Línea de Investigación: Arquitectura Neuro-Simbólica MARCUS-UNIFIED (`marcus_neurosymbolic`)**
>    - **Rama:** `exp/marcus-neurosymbolic-architecture`
>    - **Objetivo:** Gran sustrato cognitivo unificado que integra y reutiliza formalmente **todas** las líneas de investigación previas mediante conectores especializados para respaldar el paradigma de Gary Marcus (*The Algebraic Mind*, *Rebooting AI*, *Taming Silicon Valley*):
>      * **Conector 1 (`gemma_campo` & `campo_turno`):** Provee el flujo sensorial de campo continuo ($z_t \in \mathbb{R}^{64}$, modos de Lie $F=2721$) al Sistema 1 sin fragmentación n-gramática.
>      * **Conector 2 (`bio_quantum_engine` BQ10 & `atlas_fw`):** Ligamiento algebraico de variables con pesos rápidos $W^F$, 100% Ley de Dale ($W^E, W^I \ge 0$) libre de diafonía y consolidación circadiana SWR ($\Delta R \ge -\varepsilon$).
>      * **Conector 3 (`soft_bellman_simplicial` & CDT):** Física Spelke (barrera métrica de Cayley-Menger para solidez e impenetrabilidad exacta) y geodésicas baricéntricas continuas Snell-Fermat para modelar el Principio de Acción Racional (Theory of Mind).
>      * **Conector 4 (`constelacion_v4` & `orbita`):** Taxonomía e instanciación Tipo-Token (los nodos de la constelación actúan como Tipos ontológicos; los Tokens situados heredan sus leyes).
>      * **Sustrato Deliberativo & Centinela:** Grafo causal Pearl $do(x)$, Bucle Dual Sistema 1 + Sistema 2 y Centinela de Vigilancia Epistémica con árbol de procedencia auditable.
>    - **Resultados DEV & CONFIRM vs las 2 Mejores Arquitecturas de Main (BQ10 y ATLAS-FW):**
>      * **CONFIRM (semillas ciegas `0xB400–0xB40F`, bootstrap 10,000):**
>        * **`new_med_log`:** **−1.7493 [−1.8777, −1.6291]** (vs **−1.5081** de BQ10 y **−1.3518** de ATLAS-FW). $\Delta \text{ vs BQ10} = \mathbf{-0.2413} [-0.2466, -0.2359]$ (**Pasa Primaria H-PRIM** con $P < 0.0001$), $\Delta \text{ vs ALL} = \mathbf{-0.3975} [-0.5095, -0.2862]$ (**Pasa Co-Primaria H-COPRIM**).
>        * **`new_acc16` (OOD):** Sube a **0.2427 [0.2197, 0.2692]** (vs **0.1484** de BQ10 y **0.1562** de ATLAS-FW; $\Delta = \mathbf{+0.0943}$, mejora del +63.5% en generalización OOD).
>        * **`known_acc16`:** **0.8991 [0.8954, 0.9029]** (supera a BQ10 0.8872, cumpliendo no-daño $\delta=0.05$).
>        * **Estabilidad E33:** **16/16 pass** en DEV y **16/16 pass** en CONFIRM. 0 NaN, 0 violaciones de Dale en 8,192 sinapsis.
>      * **H1 a H5 de Marcus:** 100.0% de exactitud en operaciones sobre variables, 100.0% en asimetría composicional, recall de violaciones físicas = 100.0%, error causal $do(x) = 0.0000$, 0.0% de alucinaciones admitidas con 100.0% de árboles de procedencia.
>    - **Reportes:** [`docs/marcus_neurosymbolic/INVESTIGACION_GARY_MARCUS.md`](docs/marcus_neurosymbolic/INVESTIGACION_GARY_MARCUS.md), [`docs/marcus_neurosymbolic/ARQUITECTURA_MARCUS.md`](docs/marcus_neurosymbolic/ARQUITECTURA_MARCUS.md), [`docs/marcus_neurosymbolic/ARQUITECTURA_UNIFICADA_CONECTORES.md`](docs/marcus_neurosymbolic/ARQUITECTURA_UNIFICADA_CONECTORES.md), [`docs/marcus_neurosymbolic/PREREGISTRO_DEV_CONFIRM.md`](docs/marcus_neurosymbolic/PREREGISTRO_DEV_CONFIRM.md), [`docs/marcus_neurosymbolic/RESULTADOS_DEV_CONFIRM.md`](docs/marcus_neurosymbolic/RESULTADOS_DEV_CONFIRM.md), [`docs/marcus_neurosymbolic/RESULTADOS_BENCHMARK.md`](docs/marcus_neurosymbolic/RESULTADOS_BENCHMARK.md) y [`docs/marcus_neurosymbolic/RESULTADOS_CONECTORES_UNIFICADOS.md`](docs/marcus_neurosymbolic/RESULTADOS_CONECTORES_UNIFICADOS.md).
>
> Ver reporte consolidado global y tabla unificada en [`docs/RESULTADOS_CONSOLIDADOS.md`](docs/RESULTADOS_CONSOLIDADOS.md).
>
> CONFIRM BQ10: BQ10−v4 new_med_log **−0.608 [−0.742, −0.465]**, BQ10−ATLAS-FW −0.156 [−0.269, −0.043], known_acc16 +0.010 vs v4, E33 16/16. Detalles en [`docs/bio_quantum_v10/`](docs/bio_quantum_v10/).
>
> Anterior: **ATLAS-FW** (`orbita atlas-run`, brazo `ALL`) — aprendiz en contexto sobre CONSTELACIÓN v4 entrenada. CONFIRM: V−v4 en new_med_log **−0.4082 [−0.4985, −0.3190]**, known_acc16 +0.0225 [+0.0171, +0.0278], E33 16/16; reproducido en ablaciones (−0.3742). Ganancia de M1+M2+M6; M3/M4/M5 sin aporte (quedan como interruptores). Ver [`docs/atlas_fw/README.md`](docs/atlas_fw/README.md).

> Base anterior: **CONSTELACIÓN v4** (`orbita cst4-run`) — único experimento que cumple los 3 criterios preregistrados (primaria CONFIRM −0,107 [−0,181, −0,024], no-daño 0, E33 16/16). Ganancia atribuible al campo rápido W^F, no a la codificación predictiva. Tabla honesta de todos los experimentos (incl. negativos: ESTELA, v2, v3, mejoras M1–M3, BIO-CONSTELACIÓN) en [`docs/RESULTADOS_CONSOLIDADOS.md`](docs/RESULTADOS_CONSOLIDADOS.md).

> **`main` = ÓRBITA.** Ver [`docs/ORBITA_MAIN.md`](docs/ORBITA_MAIN.md). v4 está archivado en el tag `v4-final`.

> **BQ10-GPU** (crate `bio_quantum_engine/`, entrenamiento WGPU en AMD RX 7800 XT, escala x50): código, pesos, logs y evaluación del *arnés propio*. No es comparable con la tabla canónica de arriba ni desplaza al BQ10 de `orbita atlas-run` ni a ATLAS-FW. Informe honesto: [`docs/bq10/README.md`](docs/bq10/README.md).

> **Campo sobre LLM congelado** (línea principal `gemma_campo` a nivel token @ `a420311`; camino alterno estado por turno @ `a2c9c9f`). No es comparable con ATLAS-FW ni con el BQ10 canónico. Informe y reproducción: [`docs/campo/README.md`](docs/campo/README.md).

# Laboratorio neuro-geométrico multi-agente · `main` v4

Crate Rust `cdt_rqm_epr` con **una sola arquitectura de campo: v4** (Dφ
residual + CDT + teacher consolidado, receta §16) y la app web
`agentic_web` que la usa en todas sus pestañas. No se afirma consciencia,
cognición general ni AGI; los resultados son sobre un benchmark sintético
geométrico con pre-registro, DEV y CONFIRM separados.

> Lo anterior a v4 (línea líquida E8–E17, Clean-Room v2/v3.7, harnesses
> etapa 2, fluido-3d-astra, ejemplos viejos) se retiró de `main`. Está
> íntegro en la etiqueta **`pre-v4-main`** / rama `archive/pre-v4-main`, y
> sus documentos en [`docs/historico/`](docs/historico/README.md)
> (incluidos los resultados de la línea A CDT-RQM-EPR).

## 1. Arquitectura v4

```
 texto ─► periferia (identidad / Gemma congelada, solo lectura)
            │
            ▼
   z ──► Dφ  (cabeza local-lineal con compuerta, residual)  ──► ẑ ─► decoder (identidad)
            ▲
   entrenamiento:  experiencia ─► CDT ─► consolidación ─► teacher por familia
                   ─► máscara de confianza (residuo < τ) ─► replay confiable ─► Dφ
   test:           CDT borrada · 0 consultas de memoria · auditor de leakage
```

| Módulo | Papel |
|---|---|
| `src/v4_dataset.rs` | Generador de 8 familias geométricas, RNG independientes, sellado y auditoría anti-contaminación |
| `src/field_autonomy_v4.rs` | Harness E31–E44: experiencia → CDT → consolidación → adaptación de Dφ, evaluación |
| `src/v4_controls.rs` | Dφ y controles pareados (MLP, lineal, NN, estático) |
| `src/v4_distill.rs` | Destilación teacher→Dφ; `train_trusted_on` = receta §16 (gated + replay confiable) |
| `src/v4_metrics.rs`, `v4_provenance.rs`, `v4_stability.rs`, `v4_e45.rs`, `v4_language.rs`, `v4_qwen2_hidden.rs` | Métricas, procedencia por predicción, E33S, E45, E43/E44 |
| `src/field_v4.rs` | **Motor de la app**: `train_batch`, `sleep_consolidate`, `run_smoke`, `chat_field`, checkpoints bit-exactos, y el gate científico de CI |
| `src/web/*`, `web/static/*` | App web (Axum) |
| `src/prompt_spider.rs`, `spider_field*.rs` | Prompt Spider |
| `src/native_gemma2*.rs`, `field_gemma_probe.rs` | Cargador Gemma 2 GGUF (pesos congelados) para chat y decoder de campo |

## 2. Qué está demostrado y qué no

Detalle y tablas: [`docs/resultados_v4.md`](docs/resultados_v4.md).
Pre-registro: [`docs/preregistro_v4.md`](docs/preregistro_v4.md).

**Pasa (DEV y CONFIRM, leakage 0, 0 consultas de memoria en TEST):**
- **E35** (experiencia CDT → Dφ): B−A +0.13 / +0.14, IC95 > 0.
- **E36** (ablación de consolidación: el efecto depende del contenido).
- **E37** (borrar memoria tras aprender, proceso nuevo: |pre−post| = 0).
- **E38** (persistencia; el lookup CDT no explica el rendimiento).
- También E31, E32, E39, E40, E41; receta §16: error a un paso ≈ 0.9 %.

**Marginal:**
- **E33** (rollout largo): PASS justo solo con la receta §16 (13/16 DEV,
  9/16 CONFIRM, exactamente en el umbral; región 8 pasos, no 16). El control
  MLP de igual capacidad también pasa → la ganancia no viene de la forma
  "dinámica".

**No pasa / no demostrado:**
- **E43/E44** (lenguaje con periferia LLM congelada / swap de LLM): FAIL
  (2–3/16). E34 parcial (Dφ ≈ MLP), E42 no alcanza el gate estricto, E45
  (líquido vs termodinámico) FAIL.
- El "aprendizaje" es destilación de un teacher cuya clase de hipótesis
  diseñamos nosotros; Dφ no descubre regularidades por sí solo.

## 3. App web (`agentic_web`)

| Pestaña | Qué corre ahora |
|---|---|
| **Chat** | OFF: Gemma/API cruda. ON (*Decoder del campo*): el mensaje se mapea a una familia v4, Dφ predice, se reporta error relativo; el LLM solo verbaliza el estado (no toca Dφ) |
| **Entrenamiento** | Lotes sellados v4 → CDT → consolidación → teacher → Dφ con receta §16; métricas v4 (one-step, acc, leakage, queries) y checkpoints Dφ bit-exactos en `NGM_CHECKPOINT_DIR` |
| **Sueño** | Consolida el buffer de vigilia en CDT/teacher y destila a Dφ; reporta antes/después |
| **Pruebas** | Smoke v4: E35 A vs B con auditoría de memoria cero, error a un paso y rollout E33 corto |
| **Modelos / API** | Proveedores OpenAI-compatibles (`/v1/*`, docker-llm), Gemma local |
| **Prompt Spider** | Sin cambios (router de campo propio) |

Auth: `MASTER_SECRET` → cookie de sesión; todo `/api/*` (salvo login/status)
devuelve 401 sin sesión. `/health` es público.

## 4. Cómo correr

```bash
# Gate científico v4 (≈30 s en release)
cargo test --release --lib scientific -- --nocapture

# Todos los tests (lib + app)
cargo test --release --features web --lib

# App web
MASTER_SECRET=<secreto> cargo run --release --features web --bin agentic_web
# → http://localhost:8080   (PORT, NGM_CHECKPOINT_DIR, LLM_API_* opcionales)

# Harness completo E31–E44
cargo run --release --bin run_field_autonomy_v4 -- --help
```

Docker: `docker build -t ngm .` (Railway despliega `main` con este
Dockerfile; ver [`docs/deploy_railway_agentic.md`](docs/deploy_railway_agentic.md)).

## 5. CI

`.github/workflows/ci.yml`: `fmt`, `clippy -D warnings` (con `web`), tests
smoke de lib + app, y el **gate científico v4**: memoria cero en TEST (0
consultas, 0 LEAKED, CDT borrada), E35 B > A en seeds DEV pequeñas y cota de
error a un paso (< 5 %) con la receta §16.

## 6. Documentación

- **Mejor versión canónica: BQ10** (`orbita atlas-run`) → [`docs/bio_quantum_v10/`](docs/bio_quantum_v10/) · previa: **ATLAS-FW** → [`docs/atlas_fw/README.md`](docs/atlas_fw/README.md)
- **BQ10-GPU** (arnés propio, no comparable) → [`docs/bq10/README.md`](docs/bq10/README.md)
- **Campo sobre LLM** (principal: gemma_campo; alterno: por turno) → [`docs/campo/README.md`](docs/campo/README.md)
- [`docs/resultados_v4.md`](docs/resultados_v4.md) · [`docs/preregistro_v4.md`](docs/preregistro_v4.md) · [`docs/plan_autonomia_campo_v4.md`](docs/plan_autonomia_campo_v4.md) · [`docs/plan_v4_dphi_teacher_distillation.md`](docs/plan_v4_dphi_teacher_distillation.md)
- [`docs/protocolo_v4_agente.md`](docs/protocolo_v4_agente.md) · [`docs/handoff_field_autonomy_v4.md`](docs/handoff_field_autonomy_v4.md) · [`docs/comparativa_v4_vs_token_free.md`](docs/comparativa_v4_vs_token_free.md)
- Prompt Spider: [`docs/experimento_prompt_spider.md`](docs/experimento_prompt_spider.md), [`docs/prompt_spider_metricas.md`](docs/prompt_spider_metricas.md)
- Histórico (línea A CDT-RQM-EPR, líquido, etapa 2, paper): [`docs/historico/`](docs/historico/README.md)
