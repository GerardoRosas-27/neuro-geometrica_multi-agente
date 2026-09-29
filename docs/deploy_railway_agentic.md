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
| `MASTER_SECRET` | **Sí** | — | Secreto maestro de acceso (UI + `/api/*`). Sin él la API responde 503 `master_secret_not_configured`. Genera uno con `python3 -c "import secrets; print(secrets.token_urlsafe(48))"`. Ver README · «acceso con secreto maestro» |
| `SESSION_TTL_HOURS` | No | `12` | Duración de la sesión firmada |
| `AUTH_REVOKED_FILE` | No | `data/revoked_sessions.json` | Sesiones revocadas por «Cerrar sesión» |
| `PORT` | Railway la pone | `8080` | Puerto HTTP (`0.0.0.0:$PORT`) |
| `GEMMA2_GGUF` | No | `/app/models/gemma-2-2b-it-Q3_K_L.gguf` (Docker) · `models/gemma-2-2b-it-Q3_K_L.gguf` (local) | Ruta del GGUF |
| `GEMMA2_GGUF_URL` | No | Q3_K_L de `bartowski/gemma-2-2b-it-GGUF` (commit fijado) | URL pública (sin token) para descargar si falta |
| `GEMMA2_GGUF_SHA256` | No | sha256 del Q3_K_L | Verificación; vacío = no verificar |
| `GEMMA2_AUTO_DOWNLOAD` | No | `1` | `0` = no descargar en arranque |
| `CHAT_TIMEOUT_SECS` | No | `75` | Plazo por mensaje; al vencer corta y devuelve texto parcial |
| `RAW_CHAT_MAX_TOKENS` | No | `160` | Tokens máx. modo OFF (Gemma original) |
| `FIELD_DECODER_MAX_TOKENS` | No | `96` | Tokens máx. modo ON (decoder del campo) |
| `LLM_API_BASE` | No | — | API externa OpenAI-compatible fija (p. ej. `https://docker-llm-production.up.railway.app/v1`); aparece como proveedor `env` |
| `LLM_API_KEY` | No | — | Clave Bearer de esa API (`obk1.…`). Nunca se registra ni se devuelve |
| `LLM_API_MODEL` | No | — | Modelo (vacío = el primero de `/v1/models`) |
| `LLM_API_NAME` | No | derivado del host | Nombre visible del proveedor `env` |
| `LLM_API_ACTIVE` | No | `0` | `1` = arrancar con la API `env` como LLM activo |
| `LLM_PROVIDERS_FILE` | No | `data/llm_providers.json` | JSON (0600) con las APIs guardadas desde la UI y el LLM activo |
| `RUST_LOG` | No | `info` | Nivel de tracing |

**Modelo ligero incluido.** El `Dockerfile` descarga en el build Gemma 2 2B-it
**Q3_K_L** (~1.55 GB, sha256 verificado). Si el archivo no está (build con
`--build-arg DOWNLOAD_GGUF=0`, o `GEMMA2_GGUF` apuntando a un volumen vacío), el
binario arranca en léxico, lo descarga en segundo plano con `curl` y hace
hot-swap a Gemma (`GET /api/status` con sesión → `model.state`: `downloading` → `loading` → `ready`).

¿Por qué Q3_K_L? El cargador nativo solo soporta arquitectura `gemma2` y candle
no soporta cuantizaciones IQ*. Q2_K (~1.23 GB) se probó y degenera (bucles,
texto incoherente). Q3_K_L es la opción más ligera usable. RAM: ~1.8 GB estable
(`MALLOC_ARENA_MAX=2`), picos ~2.3 GB → usa un plan Railway con **≥ 3 GB de RAM**.

Chat: ON («Decoder del campo») = campo (líquido/CDT/RQM) decide el estado y
Gemma lo verbaliza (con fallback al decoder léxico si Gemma no está o falla);
OFF = Gemma 2 original congelado. Ambos comparten los mismos pesos en RAM.

## Cómo desplegar en railway.com

1. Conecta el repo `neuro-geometrica_multi-agente` en [railway.com](https://railway.com).
2. Crea un servicio desde el repo; Railway detecta `Dockerfile` / `railway.toml`.
3. Asegura rama `main` (o la de este PR tras merge) y build con Dockerfile.
4. Define `MASTER_SECRET` en **Variables** (sin ella la API queda cerrada).
   Healthcheck: `GET /health` → `{ "ok": true, "auth": { "master_configured": true, "warnings": [] } }`
   (público y sin datos internos; el estado detallado está en `GET /api/status`, con sesión).
5. (Opcional) Para no hornear el GGUF en la imagen: build arg `DOWNLOAD_GGUF=0`, volumen en `/data` y `GEMMA2_GGUF=/data/gemma-2-2b-it-Q3_K_L.gguf` (se descarga una vez al primer arranque).
6. Abre la URL pública: UI en `/` (pide el secreto maestro), API bajo `/api/*` (sesión por cookie o `Authorization: Bearer`).

## Qué hace la UI

## UI · 4 pestañas (Chat / Entrenamiento / Sueño / Pruebas)

- **Chat**: solo historial + input. El LLM **decodifica** lo que el modelo de campo recuerda (concepto / engramas). No comparte conversación como dataset de train.
- **Entrenamiento**: consola en vivo, infinito por defecto, start/stop. Datasets vía `generate_train_batch` en periferia (`source_tag=llm_dataset_decoupled`).
- **Sueño**: `POST /api/sleep` con intensidades de poda/compactación. Minimiza energía libre, compacta fasores, poda rutas RQM débiles, reporta F/simetría/handshake.
- **Pruebas**: `POST /api/tests/run` evalúa el fuse/campo **ya entrenado** (identidad, shifted, latencia líquido, recall de engramas, F/simetría del último sueño, histograma de rutas). No lanza train infinito. `GET /api/tests/last` y `/api/tests/status`.



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



## GGUF local (Docker compose / experimentos)

El `Dockerfile` ya trae el GGUF ligero. Fuera de Docker, el binario lo descarga
solo a `models/gemma-2-2b-it-Q3_K_L.gguf` al arrancar. Manual:

```bash
mkdir -p models
curl -L --retry 5 -C - -o models/gemma-2-2b-it-Q3_K_L.gguf \
  "https://huggingface.co/bartowski/gemma-2-2b-it-GGUF/resolve/855f67caed130e1befc571b52bd181be2e858883/gemma-2-2b-it-Q3_K_L.gguf"
# Experimentos E12 / FrozenGemma2Probe:
export GEMMA2_GGUF=$PWD/models/gemma-2-2b-it-Q3_K_L.gguf
cargo test --release --lib field_gemma_probe -- --nocapture
```
(El Q4_K_M anterior sigue funcionando vía `GEMMA2_GGUF` si prefieres calidad.)
Sin GGUF → periferia léxico; E12 → `SKIPPED_NO_GGUF` (honesto).

## Notas

- No se descargan modelos en el build de Docker (binario razonable).
- `cargo test` por defecto (sin `web`) sigue sin depender de Axum.
- Admin Repositorio puede ayudar con push/merge si los permisos fallan.

## Requisitos de build (Docker)

El `Dockerfile` usa la imagen `rust:bookworm` (stable reciente).
No uses `rust:1.85`: `sysinfo` 0.39 pide rustc ≥ 1.95 y `zip` 8.x pide ≥ 1.88.

## Modelos / API (LLM externo OpenAI-compatible)

La pestaña **Modelos / API** (y el botón del mismo nombre en la cabecera) permite
usar un LLM externo, por ejemplo **docker-llm**, en lugar de Gemma local:

1. En el panel de docker-llm genera una API key y copia el `curl` de un modelo.
2. Pégalo en «Pega aquí el curl» y pulsa **Auto-configurar**: rellena nombre,
   URL base (hasta `/v1`), API key (oculta) y modelo. Acepta continuaciones `\`,
   comillas simples/dobles, `-H/--header`, `-d/--data/--data-raw`, `-N`,
   `"stream":true`, la ruta por modelo `/v1/models/<id>/chat/completions`,
   `/v1/models`, `/v1/completions` y líneas `export BASE=…` / `export API_KEY=…`.
3. **Probar conexión** (`GET /v1/models` + chat mínimo, con latencias) →
   **Guardar** (o **Guardar y usar**).
4. Elige el LLM activo en el selector de la cabecera o en la lista.

El LLM activo se usa en: chat OFF (LLM crudo con historial), chat ON (el campo
decide el estado y el LLM solo lo verbaliza; si la API falla y Gemma está
cargado responde Gemma como respaldo etiquetado, si no el decoder léxico) y la
generación de datasets de entrenamiento (el LLM parafrasea el curriculum; la
etiqueta sigue saliendo del curriculum y el encode al campo lo hace la sonda
local). Sueño y Pruebas no usan LLM. «Nuevo chat» borra también el contexto que
se reenvía a la API.

Endpoints: `GET/POST /api/llm/providers`, `DELETE /api/llm/providers/{id}`,
`POST /api/llm/providers/test`, `GET/POST /api/llm/active` (`{id}`; `gemma_local`
= local), `POST /api/llm/parse-curl`. Las claves salen enmascaradas en las
respuestas GET/POST y no se escriben en logs.

**Persistencia.** Lo guardado desde la UI vive en `data/llm_providers.json`
dentro del contenedor: en Railway **sin volumen** se pierde en cada redeploy.
Para dejar una API fija usa `LLM_API_BASE` / `LLM_API_KEY` / `LLM_API_MODEL`
(+ `LLM_API_ACTIVE=1`), o monta un volumen y apunta `LLM_PROVIDERS_FILE` a él.
