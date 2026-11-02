> **Mejor arquitectura (rama `exp/mejor-arquitectura`): CONSTELACIÓN v4** — único experimento que cumple los 3 criterios preregistrados (primaria CONFIRM −0,107 [−0,181, −0,024], no-daño 0, E33 16/16). Ganancia atribuible al campo rápido W^F, no a la codificación predictiva. Ver [`docs/RESULTADOS_CONSOLIDADOS.md`](docs/RESULTADOS_CONSOLIDADOS.md).

> **`main` = ÓRBITA.** Ver [`docs/ORBITA_MAIN.md`](docs/ORBITA_MAIN.md). v4 está archivado en el tag `v4-final`.

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

- [`docs/resultados_v4.md`](docs/resultados_v4.md) · [`docs/preregistro_v4.md`](docs/preregistro_v4.md) · [`docs/plan_autonomia_campo_v4.md`](docs/plan_autonomia_campo_v4.md) · [`docs/plan_v4_dphi_teacher_distillation.md`](docs/plan_v4_dphi_teacher_distillation.md)
- [`docs/protocolo_v4_agente.md`](docs/protocolo_v4_agente.md) · [`docs/handoff_field_autonomy_v4.md`](docs/handoff_field_autonomy_v4.md) · [`docs/comparativa_v4_vs_token_free.md`](docs/comparativa_v4_vs_token_free.md)
- Prompt Spider: [`docs/experimento_prompt_spider.md`](docs/experimento_prompt_spider.md), [`docs/prompt_spider_metricas.md`](docs/prompt_spider_metricas.md)
- Histórico (línea A CDT-RQM-EPR, líquido, etapa 2, paper): [`docs/historico/`](docs/historico/README.md)
