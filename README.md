# CDT-RQM-EPR · laboratorio neuro-geométrico multi-agente

Laboratorio nativo en Rust (crate `cdt_rqm_epr`) con **dos líneas de
resultados científicos** con estado de evidencia distinto, más una **app web
agentica** (`agentic_web`) que usa el sustrato líquido/CDT/RQM con un LLM
periférico. No se afirma consciencia, cognición general, AGI, ventaja en
julios ni “campo autónomo en producción”.

> **Nota de estado (2026-10-06).** Versiones anteriores de este README decían
> que el cierre v3 de `exp/field-autonomy-next` estaba «no mergeado a main».
> Eso dejó de ser cierto: el **PR #29** (merge `c59b9fe`, 2026-09-26) integró
> la rama completa en `main`. Los documentos de esa rama conservan su texto
> original («sin merge a main») como registro histórico, con una nota de
> estado al inicio. Este README describe `main` tal como está.

Contenido: [1. Qué contiene `main`](#1-qué-contiene-main-hoy) ·
[2. Resultados científicos](#2-resultados-científicos-y-estado-de-la-evidencia) ·
[3. App web](#3-app-web-agentic_web) ·
[4. Puntos de entrada](#4-puntos-de-entrada) ·
[5. Integración continua](#5-integración-continua) ·
[6. Documentación](#6-documentación-detallada)

---

## 1. Qué contiene `main` hoy

| Componente | Código en `main` | Cómo llegó | ¿Lo usa la app web? |
|---|---|---|---|
| **Consolidación de cuenca CDT** (línea A) | `src/consolidation_basin_experiment.rs`, `src/basin_external_baselines.rs`, bin `native_consolidation_basin_experiment` | PR #7 (2026-08-30) y PR #8 (tesis v2, `1a47b7d`) | No (es el experimento canónico + gate científico de CI) |
| **Autonomía del campo, Clean-Room v3.7** (línea B) | `src/field_autonomy_stage2_v2.rs` (el nombre «v2» es histórico; el contenido es idéntico al commit de palancas v3.7 `0b000b1`), examples `run_stage2_v3_long`, `run_stage2_v3_confirm`, `run_stage2_v2_dev`, `smoke_stage2_v2` | **PR #29** (`c59b9fe`, 2026-09-26) | Solo un **smoke** en la pestaña Pruebas (ver abajo) |
| Etapa 2 pre-cleanroom (histórico) | `src/field_autonomy_stage2.rs`, example `smoke_stage2` | `e05cef1` (2026-09-23) | No |
| Experimentos E8–E10 (líquido) y E11–E17 (campo entrenable) | `src/liquid_experiments_8_9_10.rs`, `src/liquid_experiments_11_17.rs`, examples `smoke_e13_e15`, `e13_diag` | `e05cef1` | Smokes E8–E10 y E13/E15 en Pruebas |
| Sustrato líquido + CDT + índice RQM (FUSE) | `src/liquid_cdt_rqm_fuse.rs`, `src/liquid_cdt_memory.rs`, `src/wave_predict_core.rs`, `src/field_substrate.rs` | PR #16 y siguientes | **Sí**: chat ON, entrenamiento, sueño, pruebas |
| Periferia LLM | Gemma 2 2B-it GGUF (pesos congelados; `src/field_gemma_probe.rs`, `src/native_gemma2_runtime.rs`) o API externa OpenAI-compatible (`src/web/llm_provider.rs`) | sonda en PR #16; chat ON/OFF PR #30; Q3_K_L PR #32; APIs PR #34 | Sí |
| App web | `src/bin/agentic_web.rs`, `src/web/*`, `web/static/*` | PR #18 → #36 | — |

### Qué está integrado y qué no está cableado

- **Sí** está en `main`: todo el harness Clean-Room v3.7 (E18–E30, FASE_A,
  locks `HyperparamLock::long()`, seeds DEV `0xA300–0xA30F` y de confirmación
  `0xB300–0xB30F`), sus resultados LONG y de confirmación, el plan v3 y el
  documento de cierre.
- La pestaña **Pruebas** ejecuta `field_autonomy_stage2_v2::run_smoke` con
  **una** seed DEV (`0xA300`), `HyperparamLock::smoke()` y solo FASE_A / E18 /
  E19 / E24. Es un smoke de humo, **no** la suite LONG ni la confirmación
  `0xB300` (la UI lo etiqueta «Clean-Room v2» por el nombre del módulo).
- El **chat «Decoder del campo»** (modo ON) **no** usa la dinámica Dφ de v3.7:
  el estado lo produce el FUSE líquido/CDT/RQM (`FusedLiquidCdt`) y el LLM lo
  verbaliza. La familia de datasets `autonomy_e18_e30` del entrenamiento es un
  curriculum de texto para ese FUSE, no el harness v3.7.
- La suite completa LONG / confirmación v3.7 no corre en CI ni en la app; se
  lanza con los examples (ver §4).

---

## 2. Resultados científicos y estado de la evidencia

Hay dos líneas. No son el mismo resultado ni miden lo mismo; no se mezclan
sus cifras.

| | Línea A · consolidación de cuenca CDT-RQM-EPR | Línea B · autonomía del campo Clean-Room v3.7 |
|---|---|---|
| Pregunta | ¿Consolidar en CDT un patrón verificado amplía causalmente su cuenca de recuperación? | ¿Una dinámica de campo Dφ entrenada desde cero, con RQM/NN/tabla/attractor apagados, aprende reglas y composición que superan baselines estáticos? |
| Estado | **Tesis principal del repositorio** (preprint 0.6, comando canónico, gate científico de CI) | **Resultado experimental confirmado en sus 4 gates**, con hipótesis central del plan aún abierta |
| Manuscrito | [`docs/paper_inferencia_fasorial_consolidacion_cdt.md`](docs/paper_inferencia_fasorial_consolidacion_cdt.md) | [`docs/cierre_exp_field_autonomy_next_v3.md`](docs/cierre_exp_field_autonomy_next_v3.md) (no hay preprint) |

### 2.1 Línea A — CDT-RQM-EPR y tesis de consolidación de cuenca (principal)

> Una consolidación CDT de un patrón verificado deforma el paisaje fasorial
> y amplía de forma causal la cuenca de recuperación.

Evidencia (tabla 7.4 del preprint 0.6, 32 nodos, 24 ensayos por nivel,
semilla `0xBA51_CD72_2026`): éxito **0/24 pre → 24/24 post** en corrupción
10–40 %, `rho_critica` 0,00 → 0,40, `decision=basin_expansion_pass`.
Multisemilla: 8/8 `basin_expansion_pass`
([`docs/arquitectura_siguiente_ciclo.md`](docs/arquitectura_siguiente_ciclo.md) §3.2).

Límites declarados en los propios documentos:

- Es el resultado de **un slot**. El 100 % post del patrón inyectado es el
  techo esperado de escribir un atractor; el fixture satura y no discrimina.
  La métrica discriminante es el **holdout** (patrón nunca consolidado).
- Es evidencia interna de deformación de cuenca, no de cognición emergente.

**Tesis v2 (siguiente resultado, falsable):** almacenar *K* patrones
verificados en el mismo sustrato; consolidar el patrón *k* deja la retención
del conjunto *A* en ΔR ≥ −ε, y se publica *K(N, ρ)* frente a Hopfield y Hebb.
Estado: implementada en código (`1a47b7d`) y convertida en gate
(`scientific_bounded_forgetting_*` exige `bounded_forgetting_pass`;
`scientific_scale_capacity` exige *K_max* ≥ 2 en 128 nodos). El gate científico pasó en
el último CI verde de `main` (`9ed18b2`, 2026-09-23); desde entonces no corre
en CI (ver §5), y una verificación local sobre `27ec710` (2026-10-06) pasó
los 6 tests `scientific` (incluidos `bounded_forgetting_pass` y *K_max* ≥ 2). **Los documentos no publican cifras de ΔR ni de *K_max*
posteriores a esa implementación**; la única medición documentada
(31-ago-2026, antes de `1a47b7d`) fue ΔR = −1,0.
Si ΔR < −ε o la curva no discrimina, no hay resultado. Arquitectura:
[`docs/arquitectura_siguiente_ciclo.md`](docs/arquitectura_siguiente_ciclo.md).

```powershell
cargo test --lib consolidation_basin_experiment -- --nocapture
cargo run --release --bin native_consolidation_basin_experiment
cargo test --release --lib scientific -- --nocapture   # gate científico
```

### 2.2 Línea B — Autonomía del campo: cierre Clean-Room v3.7 (mergeado en PR #29)

Protocolo clean-room `TRAIN → DEV → LOCK → TEST`, evaluación **FIELD_ONLY**
(init aleatorio, CDT vacío, RQM/NN/tabla/attractor OFF), auditor anti-leakage.
Lock `HyperparamLock::long()` fijado en DEV y **sin retune** en confirmación.
Fuentes: [`docs/resultados_etapa_2_v3_long.md`](docs/resultados_etapa_2_v3_long.md)
y [`docs/resultados_etapa_2_v3_confirm.md`](docs/resultados_etapa_2_v3_confirm.md)
(+ CSV, 240 filas cada uno; leakage > 0 = 0, `DATASET_INVALID` = 0 en LONG).

**Gates (POSITIVE + STRONG_POSITIVE sobre 16 seeds):**

| Exp | DEV LONG `0xA300–0xA30F` | Confirmación held-out `0xB300–0xB30F` |
|---|---|---|
| **E18** aprendizaje de reglas | 14/16 | **16/16** (POS 7 · STRONG 9) |
| **E21** interpolación / extrapolación | 10/16 | **12/16** (PARTIAL 4 · POS 6 · STRONG 6) |
| **E22** composición con RQM apagado | 9/16 | **12/16** (NULL 1 · PARTIAL 3 · POS 9 · STRONG 3) |
| **E24** estático vs dinámico pareado | 14/16 | **16/16** (POS 16) |
| E23 rollout sin teacher forcing (sin gate) | 4/16 | 3/16 |

Veredicto del cierre: **Confirmation: YES** (mayoría POS+STRONG en los
cuatro gates).

**Resto de la suite en confirmación:** FASE_A 16/16, E19 procedencia 15/16,
E27 transferencia 5/16; E18C, E20, E25, E26, E28, E29 y E30 en **0/16**
(PARTIAL/NULL).

**Alcance y límites (no sobre-reclamar):**

- E21 y E22 confirman por mayoría (12/16), no de forma unánime; E22 tiene
  composición absoluta débil y residuales NULL en DEV.
- E23 en horizonte largo (h32/h64) sigue **abierto**.
- La hipótesis central del plan v3 —experiencia consolidada en CDT que
  modifica el aprendizaje de Dφ (E25), sus ablaciones (E26), intervención
  causal (E29), persistencia (E30), olvido continuo (E28)— **no** está
  confirmada. Lo confirmado es que Dφ aprende reglas y composición mejor que
  los baselines estáticos bajo clean-room; **no** que CDT actúe como
  experiencia causal.
- Escalado de N, cross-lingual ampliado y cambio de LLM no se ejecutaron.
- El merge a `main` integra el código y la evidencia; **no** amplía los claims
  de §7 del cierre ni convierte la línea B en parte del chat de producción.
- Procedencia: la columna `commit` de los CSV registra `ce282a2` (HEAD al
  correr); el código v3.7 y los resultados se commitearon juntos en `0b000b1`
  (hijo directo de `ce282a2`). El módulo en `main` es idéntico a `0b000b1`.
  No hay re-ejecución independiente posterior.

### 2.3 ¿Cuál es la tesis principal?

La **línea A** sigue siendo la tesis principal del repositorio: es la única
con manuscrito (preprint 0.6), comando canónico reproducible y gate
científico en CI. La **línea B** es un segundo resultado, integrado en `main`
y confirmado en held-out para sus cuatro gates, pero su hipótesis central
(experiencia → CDT → Dφ → salida nueva) sigue abierta y no tiene preprint.
Si en el futuro E25/E26/E29 confirman, la jerarquía debe revisarse.

### 2.4 Otras líneas (exploratorias / históricas)

- **E8–E10 líquido:** [`docs/experimentos_8_9_10_liquido.md`](docs/experimentos_8_9_10_liquido.md), resultados [`docs/resultados_experimentos_8_9_10.md`](docs/resultados_experimentos_8_9_10.md).
- **E11–E17 campo entrenable (LLM congelado como periferia):** [`docs/experimentos_11_17_campo_entrenable.md`](docs/experimentos_11_17_campo_entrenable.md), resultados [`docs/resultados_experimentos_11_17.md`](docs/resultados_experimentos_11_17.md), hallazgos E13/E15 [`docs/hallazgos_e13_e15_endurecimiento.md`](docs/hallazgos_e13_e15_endurecimiento.md). Ojo: ambos archivos de resultados se regeneraron accidentalmente en el PR #30 (ver nota al inicio de cada uno); la corrida sellada con GGUF (E12 PASS×8) está en el historial git.
- **Clean-Room v2** (base de v3; superada): [`docs/resultados_etapa_2_cleanroom_v2.md`](docs/resultados_etapa_2_cleanroom_v2.md). El módulo actual ya contiene v3.7, así que `run_stage2_v2_dev` no reproduce esas cifras.
- **Etapa 2 pre-cleanroom** (seeds `0xE1800..`, no mezclar con v2/v3): [`docs/resultados_etapa_2_autonomia.md`](docs/resultados_etapa_2_autonomia.md).
- Protocolo Etapa 2 completo: [`docs/etapa_2_autonomia_campo_experimentos.md`](docs/etapa_2_autonomia_campo_experimentos.md) — **§29+ (Clean-Room v2/v3) prevalece**.

---

## 3. App web (`agentic_web`)

```bash
cargo run --features web --bin agentic_web   # → http://127.0.0.1:8080
docker compose up --build                    # o en Docker
```

Desplegada en Railway. Despliegue y variables: [`docs/deploy_railway_agentic.md`](docs/deploy_railway_agentic.md).
Es infraestructura / demo de ingeniería: **no** es evidencia de las líneas A o B.

### 3.1 Pestañas

| Pestaña | Qué hace |
|---|---|
| **Chat** | Interruptor **«Decoder del campo»**. **ON** (por defecto): el FUSE líquido/CDT/RQM decide el estado/concepto y el LLM activo solo lo verbaliza; si el LLM falla → Gemma local como respaldo etiquetado o decoder léxico. **OFF**: LLM crudo (Gemma 2 original congelado o la API activa) con historial, sin campo. Historial persistido en el navegador (`localStorage`) + botón **«Nuevo chat»** (borra también el contexto del servidor). Intents sueltos (`entrena`, `sueño`, `estado`). |
| **Entrenamiento** | Job en vivo, **infinito por defecto** (Iniciar/Detener). El LLM activo genera datasets por familia (core / líquido E8–E10 / campo E11–E17 / autonomía E18–E30); encode tokenless → FUSE → `sleep_consolidate`; checkpoint por dataset en `data/checkpoints/`. |
| **Sueño** | Requiere entrenamiento previo. Poda rutas RQM débiles, compacta la geometría fasorial y minimiza energía libre; reporta F / simetría / handshake. Infinito por defecto. No usa LLM. |
| **Modelos / API** | Pegar un `curl` (p. ej. de docker-llm) → auto-configurar → probar → guardar; elegir el LLM activo (Gemma local o API OpenAI-compatible) para chat y datasets. Claves enmascaradas. |
| **Pruebas** | Suite de smokes (E8–E10, E13/E15, Clean-Room v3.7 en 1 seed DEV) + evaluación del FUSE ya entrenado (identidad, shifted, latencia líquido, recall de engramas, F/simetría, histograma de rutas). No usa LLM. |
| **Prompt Spider** *(experimental)* | Recreación de «PROMPT SPIDER // EVERY WORD, ONE FORK»: un crawler determinista recorre el prompt; cada palabra recibe 4 preguntas tipadas con p heurística; `p ≥ umbral` → código, `p < umbral` → LLM activo, aprobación → cola humana (Aprobar/Rechazar). Corridas **corta/completa** según el modelo; si docker-llm no responde → **Gemma local (respaldo)**. Si el LLM falla, queda pendiente. Ver [`docs/experimento_prompt_spider.md`](docs/experimento_prompt_spider.md) y las métricas en [`docs/prompt_spider_metricas.md`](docs/prompt_spider_metricas.md). Interruptor **Decoder del campo**: las escaladas las decide un router del campo entrenado en «Entrenamiento Prompt Spider» en vez del LLM ([`docs/prompt_spider_decoder_campo.md`](docs/prompt_spider_decoder_campo.md)). |

Modelo local: **Gemma 2 2B-it Q3_K_L** (~1,55 GB, sha256 verificado), horneado
en la imagen Docker o auto-descargado al arrancar; sin GGUF arranca en
léxico y hace hot-swap cuando termina la descarga. UI adaptada a móvil.

### 3.2 Acceso con secreto maestro (`MASTER_SECRET`)

La app está protegida con un **único secreto maestro**, igual que `docker-llm`.
Despliegue completo: [`docs/deploy_railway_agentic.md`](docs/deploy_railway_agentic.md).

| Variable | Obligatoria | Default | Descripción |
|---|---|---|---|
| `MASTER_SECRET` | **Sí** | — | Secreto para entrar a la UI y a `/api/*`. Sin él la API responde `503 master_secret_not_configured` (fail-closed). `/health` avisa si falta o si tiene < 32 caracteres. |
| `SESSION_TTL_HOURS` | No | `12` | Duración de la sesión. |
| `AUTH_REVOKED_FILE` | No | `data/revoked_sessions.json` | Sesiones cerradas con «Cerrar sesión» (revocación en servidor). |
| `COOKIE_SECURE` | No | auto | `Secure` en la cookie; auto = sí detrás de HTTPS (`X-Forwarded-Proto`). |

Generar un secreto (guárdalo en tu gestor de contraseñas):

```bash
python3 -c "import secrets; print(secrets.token_urlsafe(48))"
```

En Railway: servicio → **Variables** → `MASTER_SECRET=<secreto>` → redeploy.
Luego `GET /health` debe mostrar `"auth": {"master_configured": true, "warnings": []}`.

Cómo funciona:

- `POST /api/auth/login {"secret": "…"}` compara en tiempo constante y devuelve una
  sesión firmada con HMAC-SHA256 (clave derivada del secreto por HKDF). Cambiar
  `MASTER_SECRET` invalida **todas** las sesiones. Bloqueo por IP tras 5 fallos
  (30 s, duplicando hasta 15 min) + tope global de 30 fallos/min.
- La sesión va en una cookie `HttpOnly; SameSite=Strict; Path=/api` (el navegador
  la manda también en el `EventSource` de la consola en vivo) o en
  `Authorization: Bearer ngs1.…` para scripts. Nunca en la URL. El navegador
  nunca guarda el secreto.
- `POST /api/auth/logout` revoca la sesión en el servidor; `GET /api/auth/status`
  informa del estado. Todo lo demás bajo `/api/*` (chat, entrenamiento, sueño,
  pruebas, procesos, telemetría, SSE, Modelos/API, `/api/status`) exige sesión.
  Las escrituras con cookie exigen mismo origen (anti-CSRF).
- Públicos: la UI estática (muestra el login o el aviso) y `/health` (solo
  `ok` y el estado del secreto, sin datos internos; el estado detallado está en
  `GET /api/status`).

Desde scripts:

```bash
BASE=https://neuro-geometricamulti-agente-production.up.railway.app
TOKEN=$(curl -s -X POST "$BASE/api/auth/login" -H 'Content-Type: application/json' \
  -d "{\"secret\":\"$MASTER_SECRET\"}" | python3 -c 'import sys,json;print(json.load(sys.stdin)["token"])')
curl -s "$BASE/api/status" -H "Authorization: Bearer $TOKEN"
curl -s -X POST "$BASE/api/auth/logout" -H "Authorization: Bearer $TOKEN"
```

---

## 4. Puntos de entrada

| Rol | Comando |
|---|---|
| Línea A — resultado principal | `cargo run --release --bin native_consolidation_basin_experiment` |
| Línea B — smoke v3.7 | `cargo run --example smoke_stage2_v2` |
| Línea B — LONG DEV (16 seeds) | `STAGE2_V3_EXTRA_SEEDS=1 cargo run --release --example run_stage2_v3_long` |
| Línea B — confirmación `0xB300–0xB30F` | `cargo run --release --example run_stage2_v3_confirm` |
| App web | `cargo run --features web --bin agentic_web` |
| Chat circadiano (demo CLI) | `cargo run --release --bin native_gemma2_circadian_chat -- --chat dyamon` |
| Trainer (gated) | `GEMMA_SPIN_MAX_CYCLES=9 cargo run --release --bin native_gemma2_spin_infinite_trainer` |
| Visualizador | `cargo run --release --bin native_cognitive_sleep_visualizer` |

El resto de binarios está en `src/bin/archive/`. **Aviso:** los examples de
la línea B y algunos tests de `liquid_experiments_*` **sobrescriben** sus
archivos de resultados en `docs/`; no commitear esas regeneraciones sin
revisarlas.

El entrenador de desarrollo es **gated**: sin `GEMMA_SPIN_MAX_CYCLES` o
`GEMMA_SPIN_TRAIN_HOURS` no arranca; el infinito exige un checkpoint que ya
pase `symbolic_accuracy` y el gate funcional
([`docs/reproducibilidad.md`](docs/reproducibilidad.md)). Los módulos de
laboratorio (OS cognitivo, generalización 100 %, VMC, PEPS, unificado de
test, red plástica) viven detrás del feature `research` o en
[`docs/archive.md`](docs/archive.md):

```powershell
cargo test --release --lib --features research
```

---

## 5. Integración continua

`.github/workflows/ci.yml` separa **smoke** (`cargo fmt`, `clippy -D warnings`,
`cargo test --release --lib -- --skip scientific`) del **gate científico**
(`cargo test --release --lib scientific`): multisemilla de cuenca, holdout
desmezclado, ΔR como gate y curva de capacidad. Los protocolos cognitivos al
100 % viven detrás de `--features research` y no corren en cada push. La
suite v3.7 de la línea B tampoco corre en CI.

**Estado actual:** desde `73be328` el paso `cargo fmt --check` falla en `main`
(formato preexistente), por lo que el gate científico queda *skipped* en CI.
El último CI verde en `main` es `9ed18b2` (2026-09-23). Verificación local
del gate sobre `27ec710` (2026-10-06): `cargo test --release --lib scientific`
→ 6 passed.

El 100 % post-sueño del patrón inyectado es el techo esperado de escribir un
atractor. La métrica discriminante del slot único es el holdout; la de la
tesis v2 es ΔR y *K_max(N, ρ)* frente a Hopfield/Hebb.
`.cargo/config.toml` fija `target-cpu=native` en local: binarios y tiempos de
pared no son comparables entre máquinas.

---

## 6. Documentación detallada

| Tema | Documento | Estado |
|---|---|---|
| Preprint línea A | [`docs/paper_inferencia_fasorial_consolidacion_cdt.md`](docs/paper_inferencia_fasorial_consolidacion_cdt.md) | vigente (v0.6) |
| Reproducibilidad de cifras | [`docs/reproducibilidad.md`](docs/reproducibilidad.md) | vigente |
| Arquitectura siguiente ciclo (tesis v2) | [`docs/arquitectura_siguiente_ciclo.md`](docs/arquitectura_siguiente_ciclo.md) | snapshot 31-ago-2026 |
| Cierre línea B (v3.7) | [`docs/cierre_exp_field_autonomy_next_v3.md`](docs/cierre_exp_field_autonomy_next_v3.md) | cifras vigentes; frases «sin merge» históricas |
| Plan v3 | [`docs/plan_autonomia_campo_v3.md`](docs/plan_autonomia_campo_v3.md) | especificación |
| Resultados v3.7 LONG / confirmación | [`docs/resultados_etapa_2_v3_long.md`](docs/resultados_etapa_2_v3_long.md) · [`docs/resultados_etapa_2_v3_confirm.md`](docs/resultados_etapa_2_v3_confirm.md) | vigentes |
| Protocolo Etapa 2 (E18–E30) | [`docs/etapa_2_autonomia_campo_experimentos.md`](docs/etapa_2_autonomia_campo_experimentos.md) | §29+ prevalece |
| Despliegue app web | [`docs/deploy_railway_agentic.md`](docs/deploy_railway_agentic.md) | vigente |
| Experimento Prompt Spider (pestaña web) | [`docs/experimento_prompt_spider.md`](docs/experimento_prompt_spider.md) | experimental (en `main`) |
| Prompt Spider: pipeline y métricas de la UI | [`docs/prompt_spider_metricas.md`](docs/prompt_spider_metricas.md) | referencia |
| Prompt Spider: decoder del campo (router líquido+CDT+RQM, entrenamiento, experimento de sustrato A/B/C) | [`docs/prompt_spider_decoder_campo.md`](docs/prompt_spider_decoder_campo.md) | experimental (`exp/spider-field-decoder`) |
| FUSE líquido/CDT/RQM | [`docs/hibrido_liquido_cdt_rqm_fuse.md`](docs/hibrido_liquido_cdt_rqm_fuse.md) · [`docs/arquitectura_liquido_cdt_memoria.md`](docs/arquitectura_liquido_cdt_memoria.md) | diseño |
| Capa lingüística Gemma | [`docs/capa_linguistica_gemma.md`](docs/capa_linguistica_gemma.md) · [`docs/gemma2_runtime_optimization.md`](docs/gemma2_runtime_optimization.md) | diseño |
| Revisión de alcance | [`docs/revision_proyecto.md`](docs/revision_proyecto.md) | snapshot 30-ago-2026 |
| Motores archivados | [`docs/archive.md`](docs/archive.md) | vigente |
| Bitácora histórica (no es el preprint) | [`docs/paper.md`](docs/paper.md) | histórico; no citar como resultado vigente |
