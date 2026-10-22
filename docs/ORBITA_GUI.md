# ÓRBITA en la interfaz gráfica

La GUI (`agentic_web`) tiene una pestaña **ÓRBITA** además de las pestañas v4 existentes (chat con decoder del campo, entrenamiento, sueño, pruebas, Modelos/API, Prompt Spider), que se conservan sin cambios.

## Flujo

`web/static/orbita.js` → `POST /api/orbita/run` (`src/web/orbita_job.rs`) → proceso
`python orbita/llm/demo.py --text … --seed … [--x0 …] [--obs k --observed … | --simulate f,p1,p2] --no-explain --json`
→ JSON (familia leída, parámetros, trayectoria h1–h16, bloque «yo», re-observación).

ÓRBITA no recibe prompts libres: la UI pide **familia (0–7) + p1 + p2** y genera el texto con la plantilla
`Aplica una|un <familia> con parámetros <p1> y <p2>.` (plantilla 0 de `make_texts.py`) (coma decimal, lo que entiende `numnorm`); el texto es editable.
El **estado** son 12 números (6 puntos 2D; vacío = hexágono por defecto de `demo.py`).
Re-observación: ninguna, simulada con la familia/parámetros dados (k = 1–2) o real (1–2 líneas de 12 números).

Validación (cliente y servidor): familia 0–7; p1, p2 en [-1, 1] (rango de entrenamiento; `numnorm.parse2` recorta ahí); familia/p1/p2 juntos; texto ≤ 500 caracteres;
estados de exactamente 12 números finitos (|x| ≤ 100); k ≤ 2; observaciones reales y simuladas excluyentes;
seed en 0xA400–0xA40F (DEV) o 0xB400–0xB40F (CONFIRM). La explicación generativa de Gemma (`llama-server`) no se pide (`--no-explain`).

## Endpoints

| ruta | qué hace |
|---|---|
| `GET /api/orbita/status` | `available`, lista `missing`, configuración efectiva, familias, x0 por defecto |
| `POST /api/orbita/run` | ejecuta la pila; 400 validación, 503 si falta alguna pieza (con `missing`), 504 plazo, 502 si `demo.py` falla |
| `POST /api/orbita/lat` | **stub** → 501 (ver abajo) |

Todos exigen sesión como el resto de `/api/*`.

## Configuración (entorno)

| variable | por defecto | uso |
|---|---|---|
| `ORBITA_PYTHON` | `python3` | intérprete con numpy + torch |
| `ORBITA_ROOT` | directorio actual | raíz del repo (`orbita/llm/demo.py`) |
| `ORBITA_BIN` | `$ORBITA_ROOT/orbita/target/release/orbita` | binario Rust (también lo leen los `.py`) |
| `ORBITA_DATA` | `/workspace/orbita-data` | pesos por seed, encoder D2, calibración del yo (también los `.py`) |
| `GEMMA_GGUF_PATH` (o `GEMMA2_GGUF`) | `/workspace/neuro-geometrica_multi-agente/models/gemma-2-2b-it-Q4_K_M.gguf` | encoder Gemma capa 13 |
| `ORBITA_LLAMACPP` | `/workspace/llamacpp/llama-b11259` | sólo para la explicación generativa (no la usa la GUI) |
| `ORBITA_TIMEOUT_S` | `180` | plazo por petición |

## Degradación

Si falta cualquier pieza, `/api/orbita/status` devuelve `available: false` con la lista y la pestaña muestra
«ÓRBITA no está disponible en este servidor» y desactiva el botón; el resto de la GUI sigue funcionando.
Hoy la imagen Docker/Railway **no** trae nada de ÓRBITA, así que allí se verá ese aviso.

## Qué faltaría para desplegar (Docker/Railway)

1. Compilar `orbita/` en la etapa de build (`cd orbita && cargo build --release`) y copiar el binario (`ORBITA_BIN`).
2. Python 3 con `numpy` y `torch` (CPU) en la imagen final (`ORBITA_PYTHON`) y copiar `orbita/llm/*.py`.
3. Artefactos fuera de git, montados en un volumen (`ORBITA_DATA`), por seed que se quiera servir:
   `dev|confirm/w_<seed>.json`, `llm_d2/stats.npz`, `llm_d2/<fase>/enc_D2_<seed>.pt`, `yo/<fase>/yo_dev_<seed>.csv`.
4. Gemma 2 2B Q4_K_M GGUF (~1,6 GB) en el volumen (`GEMMA_GGUF_PATH`); la GUI ya puede descargar el GGUF para su chat.
5. RAM/CPU: cada petición lanza un proceso que carga Gemma (encoder) y torch; en el box tarda ~6 s. Conviene limitar concurrencia.
No se suben pesos ni secretos al repo.

## LAT en vivo (fuera de alcance)

LAT/LAT_EP sólo existe como evaluación (`orbita lat-eval`, `orbita/llm/latlp.py`). Usarlo en línea requiere un servicio con
estado persistente (adaptador latente por sesión/seed, cola de observaciones, persistencia y reinicio). Se deja como stub 501.
