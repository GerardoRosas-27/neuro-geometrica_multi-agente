# Despliegue en Railway · Chat agentico

**Rama:** `feat/railway-agentic-chat` · **Binario:** `agentic_web` (feature `web`)

## Arquitectura (texto)

```
Usuario (UI web)
    │  texto
    ▼
┌─────────────────────────┐
│ LLM periferia           │  texto → features → concept_id
│ Gemma GGUF (opcional)   │  o GemmaShapedLexicon + LabelDecoder
│ / léxico                │  tokens NUNCA → FieldState
└───────────┬─────────────┘
            │ concept_id
            ▼
┌─────────────────────────┐
│ FusedLiquidCdt          │
│ 1. Líquido (infer)      │  WavePredictCore — hot path
│ 2. CDT termo (sueño)    │  engramas tras sleep_consolidate
│ 3. RQM (fallback)       │  índice relacional / mapas arbitrarios
└───────────┬─────────────┘
            │ concept_out + FuseReport
            ▼
┌─────────────────────────┐
│ Decoder periferia       │  concepto → texto (ES)
└─────────────────────────┘
```

- **Líquido** = toda la inferencia rápida  
- **CDT termo** = memoria durable **después** del sueño  
- **RQM** = índice/fallback (desde fuse)  
- **LLM** = **decoder only** del campo (+ dataset gen en periferia)  

## Variables de entorno

| Variable | Obligatoria | Default | Descripción |
|----------|-------------|---------|-------------|
| `PORT` | Railway la pone | `8080` | Puerto HTTP (`0.0.0.0:$PORT`) |
| `GEMMA2_GGUF` | No | — | Ruta al modelo GGUF (volumen). Sin ella → modo `lexicon` |
| `RUST_LOG` | No | `info` | Nivel de tracing |

**El GGUF es opcional.** El deploy por defecto en Railway arranca en modo léxico sin descargar pesos.

## Cómo desplegar en railway.com

1. Conecta el repo `neuro-geometrica_multi-agente` en [railway.com](https://railway.com).
2. Crea un servicio desde el repo; Railway detecta `Dockerfile` / `railway.toml`.
3. Asegura rama `main` (o la de este PR tras merge) y build con Dockerfile.
4. Healthcheck: `GET /health` → `{ "ok": true, "llm_mode": "lexicon"|"gemma_gguf", ... }`.
5. (Opcional) Monta un volumen con el GGUF y define `GEMMA2_GGUF=/data/model.gguf`.
6. Abre la URL pública: UI en `/`, API bajo `/api/*`.

## Qué hace la UI

- **Chat** (izquierda): mensajes agenticos; intents `entrena`, `sueño`, `estado`.
- **Iniciar / Detener entrenamiento**: job async **infinito por defecto** (dataset → líquido → CDT + checkpoint **por dataset**); checkbox Infinito; Detener cancela.
- **Sueño / consolidar**: `sleep_consolidate` → engramas CDT + reafirma RQM.
- **Paneles** (derecha): Entrenamiento en vivo (barra, eventos, checkpoint, decoder) · Líquido · CDT · RQM.



## Entrenamiento en vivo (LLM dataset + líquido + CDT)

Pipeline por **dataset/lote** (no bloquea el servidor Axum). Corre **hasta Detener** salvo que pidas un tope finito:

1. **Dataset en periferia** — `generate_train_batch` (fuente `gemma` o `lexicon_synth`).
2. Encode texto → features → `concept_id` (firewall; **sin tokens** en FieldState).
3. Inferencia **líquida** (`FusedLiquidCdt` / `WavePredictCore`) + `observe` / `teach_relation`.
4. **`sleep_consolidate`** → engramas CDT.
5. **Checkpoint por dataset** en `data/checkpoints/datasets/train_{ts}_ds_{n}.json` (ejemplos, métricas líquido, sueño, decoder) + índice `data/checkpoints/latest.json`. Resumen de lote compat en `train_{ts}_batch_{i}.json`.
6. **LLM decoder only** — preview en UI (concepto → texto).

### API

| Método | Ruta | Notas |
|--------|------|--------|
| POST | `/api/train/start` | Sin `batches` / `null` / `0` / `"infinite"` → **∞**. Finito: `{ "batches": 4, ... }`. Respuesta: `{ ok, job_id, infinite }` |
| POST | `/api/train/stop` | cancela el job |
| GET | `/api/train/status` | snapshot: `infinite`, `current_batch`, `total_batches` (null si ∞), `datasets_saved`, `last_dataset_path` |
| GET | `/api/train/events?after=N` | eventos nuevos (polling ~500 ms) |
| GET | `/api/train/stream` | SSE `text/event-stream` |

### Checkpoints

- Ruta relativa al CWD: `./data/checkpoints/` (en Docker `/app/data/checkpoints`).
- **Por dataset:** `./data/checkpoints/datasets/train_{ts}_ds_{n}.json` (id, ejemplos, source, liquid scores/preds, sleep, decoder).
- Índice: `./data/checkpoints/latest.json` → último dataset.
- Volumen Railway opcional si quieres persistir entre deploys.

### Rol del LLM

- **Decoder** del modelo de campo (chat: `decoded` viene de `decode_field_concept`).
- Generación de dataset = función periférica separada (no escribe al campo).

## Local

```bash
# Tests (sin red, sin GGUF)
cargo test --features web --lib web::

# Servidor
cargo run --features web --bin agentic_web
# → http://127.0.0.1:8080

# Docker
docker compose up --build
```

## Notas

- No se descargan modelos en el build de Docker (binario razonable).
- `cargo test` por defecto (sin `web`) sigue sin depender de Axum.
- Admin Repositorio puede ayudar con push/merge si los permisos fallan.

## Requisitos de build (Docker)

El `Dockerfile` usa la imagen `rust:bookworm` (stable reciente).
No uses `rust:1.85`: `sysinfo` 0.39 pide rustc ≥ 1.95 y `zip` 8.x pide ≥ 1.88.
